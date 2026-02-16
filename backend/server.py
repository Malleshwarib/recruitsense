from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import io
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

class DatasetInfo(BaseModel):
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    target_distribution: Dict[str, int]
    sample_data: List[Dict[str, Any]]

class PreprocessingResult(BaseModel):
    success: bool
    message: str
    cleaned_rows: int
    features_created: int

class ModelTrainingResult(BaseModel):
    model_name: str
    accuracy: float
    classification_report: Dict[str, Any]

class PredictionInput(BaseModel):
    skills: str
    experience: float
    education: str
    certifications: str
    job_role: str
    projects_count: int
    salary_expectation: float

class PredictionOutput(BaseModel):
    prediction: str
    probability: float
    model_used: str

class DataExplorationResult(BaseModel):
    summary_stats: Dict[str, Any]
    correlations: Dict[str, float]
    education_distribution: Dict[str, int]

# Global variables to store trained models and preprocessors
trained_models = {}
preprocessors = {}
dataset_cache = None

@api_router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        global dataset_cache
        dataset_cache = df.copy()
        
        target_dist = df['Recruiter Decision'].value_counts().to_dict()
        
        info = DatasetInfo(
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            target_distribution=target_dist,
            sample_data=df.head(5).to_dict('records')
        )
        
        return info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/data-exploration")
async def data_exploration():
    try:
        if dataset_cache is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded")
        
        df = dataset_cache.copy()
        
        numeric_cols = ['Experience (Years)', 'Salary Expectation ($)', 'Projects Count', 'AI Score (0-100)']
        summary = {}
        for col in numeric_cols:
            if col in df.columns:
                summary[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        correlations = {}
        if 'Experience (Years)' in df.columns and 'Salary Expectation ($)' in df.columns:
            correlations['experience_salary'] = float(df['Experience (Years)'].corr(df['Salary Expectation ($)']))
        if 'Projects Count' in df.columns and 'Experience (Years)' in df.columns:
            correlations['projects_experience'] = float(df['Projects Count'].corr(df['Experience (Years)']))
        
        edu_dist = df['Education'].value_counts().to_dict() if 'Education' in df.columns else {}
        
        result = DataExplorationResult(
            summary_stats=summary,
            correlations=correlations,
            education_distribution=edu_dist
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/preprocess")
async def preprocess_data():
    try:
        if dataset_cache is None:
            raise HTTPException(status_code=400, detail="No dataset uploaded")
        
        df = dataset_cache.copy()
        
        df = df.drop(columns=['Resume_ID', 'Name'], errors='ignore')
        
        df['Recruiter Decision'] = df['Recruiter Decision'].map({'Hire': 1, 'Reject': 0})
        
        # Fill NaN values before combining text
        df['Skills'] = df['Skills'].fillna('')
        df['Certifications'] = df['Certifications'].fillna('')
        df['Job Role'] = df['Job Role'].fillna('')
        
        df['combined_text'] = (
            df['Skills'].astype(str) + ' ' + 
            df['Certifications'].astype(str) + ' ' + 
            df['Job Role'].astype(str)
        )
        
        def clean_text(text):
            if pd.isna(text) or text is None:
                return ''
            text = str(text).lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        df['combined_text'] = df['combined_text'].apply(clean_text)
        
        le = LabelEncoder()
        df['Education_Encoded'] = le.fit_transform(df['Education'])
        
        global preprocessors
        preprocessors['label_encoder'] = le
        
        await db.preprocessed_data.delete_many({})
        records = df.to_dict('records')
        await db.preprocessed_data.insert_many(records)
        
        result = PreprocessingResult(
            success=True,
            message="Data preprocessed successfully",
            cleaned_rows=len(df),
            features_created=len(df.columns)
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/train-models")
async def train_models():
    try:
        docs = await db.preprocessed_data.find({}, {"_id": 0}).to_list(10000)
        if not docs:
            raise HTTPException(status_code=400, detail="No preprocessed data found")
        
        df = pd.DataFrame(docs)
        
        tfidf = TfidfVectorizer(max_features=100)
        text_features = tfidf.fit_transform(df['combined_text']).toarray()
        
        numeric_features = df[['Experience (Years)', 'Salary Expectation ($)', 'Projects Count', 'Education_Encoded']].values
        
        scaler = StandardScaler()
        numeric_features_scaled = scaler.fit_transform(numeric_features)
        
        X = np.hstack([text_features, numeric_features_scaled])
        y = df['Recruiter Decision'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        results = []
        global trained_models, preprocessors
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            trained_models[name] = model
            
            results.append(ModelTrainingResult(
                model_name=name,
                accuracy=float(accuracy),
                classification_report=report
            ))
        
        preprocessors['tfidf'] = tfidf
        preprocessors['scaler'] = scaler
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    try:
        if not trained_models:
            raise HTTPException(status_code=400, detail="Models not trained yet")
        
        combined_text = f"{input_data.skills} {input_data.certifications} {input_data.job_role}"
        combined_text = combined_text.lower()
        combined_text = re.sub(r'[^a-z0-9\s]', '', combined_text)
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        
        tfidf = preprocessors.get('tfidf')
        scaler = preprocessors.get('scaler')
        le = preprocessors.get('label_encoder')
        
        if not all([tfidf, scaler, le]):
            raise HTTPException(status_code=400, detail="Preprocessors not initialized")
        
        text_features = tfidf.transform([combined_text]).toarray()
        
        try:
            edu_encoded = le.transform([input_data.education])[0]
        except:
            edu_encoded = 0
        
        numeric_features = np.array([[
            input_data.experience,
            input_data.salary_expectation,
            input_data.projects_count,
            edu_encoded
        ]])
        
        numeric_features_scaled = scaler.transform(numeric_features)
        
        X = np.hstack([text_features, numeric_features_scaled])
        
        best_model = trained_models.get('Random Forest')
        if not best_model:
            best_model = list(trained_models.values())[0]
            model_name = list(trained_models.keys())[0]
        else:
            model_name = 'Random Forest'
        
        prediction = best_model.predict(X)[0]
        probability = best_model.predict_proba(X)[0][int(prediction)]
        
        result = PredictionOutput(
            prediction="Hire" if prediction == 1 else "Reject",
            probability=float(probability),
            model_used=model_name
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/model-status")
async def model_status():
    return {
        "models_trained": len(trained_models),
        "model_names": list(trained_models.keys()),
        "preprocessors_ready": len(preprocessors) > 0,
        "dataset_loaded": dataset_cache is not None
    }

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()