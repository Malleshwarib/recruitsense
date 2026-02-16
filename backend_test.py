import requests
import sys
import io
import pandas as pd
from datetime import datetime

class HiringPredictionAPITester:
    def __init__(self, base_url="https://recruitsense-1.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {}
        
        if files is None:
            headers['Content-Type'] = 'application/json'

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    print(f"   Response: {response.json()}")
                except:
                    print(f"   Response: {response.text}")

            return success, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def create_sample_csv(self):
        """Create a sample CSV file for testing"""
        data = {
            'Resume_ID': ['R001', 'R002', 'R003', 'R004', 'R005'],
            'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
            'Skills': [
                'Python, Machine Learning, Data Analysis',
                'Java, Spring Boot, SQL',
                'JavaScript, React, Node.js',
                'Python, Django, PostgreSQL',
                'Excel, PowerBI'
            ],
            'Experience (Years)': [5, 3, 4, 6, 1],
            'Education': ['Master\'s Degree', 'Bachelor\'s Degree', 'Master\'s Degree', 'PhD', 'High School'],
            'Certifications': ['AWS, TensorFlow', 'Oracle, Spring', 'React, AWS', 'Google Cloud, Kubernetes', 'None'],
            'Job Role': ['Data Scientist', 'Software Engineer', 'Frontend Developer', 'Senior Data Scientist', 'Junior Analyst'],
            'Recruiter Decision': ['Hire', 'Hire', 'Hire', 'Hire', 'Reject'],
            'Salary Expectation ($)': [120000, 85000, 95000, 140000, 40000],
            'Projects Count': [10, 5, 8, 15, 0],
            'AI Score (0-100)': [85, 75, 80, 95, 45]
        }
        
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return csv_content

    def test_model_status(self):
        """Test model status endpoint"""
        success, response = self.run_test(
            "Model Status Check",
            "GET",
            "model-status",
            200
        )
        return success, response

    def test_upload_dataset(self):
        """Test dataset upload"""
        csv_content = self.create_sample_csv()
        files = {'file': ('test_dataset.csv', csv_content, 'text/csv')}
        
        success, response = self.run_test(
            "Dataset Upload",
            "POST",
            "upload-dataset",
            200,
            files=files
        )
        return success, response

    def test_data_exploration(self):
        """Test data exploration"""
        success, response = self.run_test(
            "Data Exploration",
            "GET",
            "data-exploration",
            200
        )
        return success, response

    def test_preprocess_data(self):
        """Test data preprocessing"""
        success, response = self.run_test(
            "Data Preprocessing",
            "POST",
            "preprocess",
            200
        )
        return success, response

    def test_train_models(self):
        """Test model training"""
        success, response = self.run_test(
            "Model Training",
            "POST",
            "train-models",
            200
        )
        return success, response

    def test_prediction_high_qualified(self):
        """Test prediction for high-qualified candidate"""
        prediction_data = {
            "skills": "Python, Machine Learning, Data Analysis",
            "experience": 5.0,
            "education": "Master's Degree",
            "certifications": "AWS, TensorFlow",
            "job_role": "Data Scientist",
            "projects_count": 10,
            "salary_expectation": 120000.0
        }
        
        success, response = self.run_test(
            "Prediction (High-qualified candidate)",
            "POST",
            "predict",
            200,
            data=prediction_data
        )
        return success, response

    def test_prediction_low_qualified(self):
        """Test prediction for low-qualified candidate"""
        prediction_data = {
            "skills": "Excel",
            "experience": 1.0,
            "education": "High School",
            "certifications": "None",
            "job_role": "Junior Analyst",
            "projects_count": 0,
            "salary_expectation": 40000.0
        }
        
        success, response = self.run_test(
            "Prediction (Low-qualified candidate)",
            "POST",
            "predict",
            200,
            data=prediction_data
        )
        return success, response

def main():
    print("🚀 Starting AI Hiring Prediction System API Tests")
    print("=" * 60)
    
    tester = HiringPredictionAPITester()
    
    # Test 1: Check initial model status
    print("\n📊 Phase 1: Initial Status Check")
    status_success, status_response = tester.test_model_status()
    if status_success:
        print(f"   Models trained: {status_response.get('models_trained', 0)}")
        print(f"   Dataset loaded: {status_response.get('dataset_loaded', False)}")
    
    # Test 2: Upload dataset
    print("\n📁 Phase 2: Dataset Upload")
    upload_success, upload_response = tester.test_upload_dataset()
    if upload_success:
        print(f"   Rows: {upload_response.get('rows', 0)}")
        print(f"   Columns: {upload_response.get('columns', 0)}")
        print(f"   Target distribution: {upload_response.get('target_distribution', {})}")
    
    # Test 3: Data exploration
    print("\n🔍 Phase 3: Data Exploration")
    explore_success, explore_response = tester.test_data_exploration()
    if explore_success:
        print(f"   Summary stats available: {len(explore_response.get('summary_stats', {}))}")
        print(f"   Correlations: {explore_response.get('correlations', {})}")
    
    # Test 4: Data preprocessing
    print("\n⚙️ Phase 4: Data Preprocessing")
    preprocess_success, preprocess_response = tester.test_preprocess_data()
    if preprocess_success:
        print(f"   Cleaned rows: {preprocess_response.get('cleaned_rows', 0)}")
        print(f"   Features created: {preprocess_response.get('features_created', 0)}")
    
    # Test 5: Model training
    print("\n🧠 Phase 5: Model Training")
    train_success, train_response = tester.test_train_models()
    if train_success and isinstance(train_response, list):
        print(f"   Models trained: {len(train_response)}")
        for model in train_response:
            print(f"   - {model.get('model_name', 'Unknown')}: {model.get('accuracy', 0)*100:.2f}% accuracy")
    
    # Test 6: Predictions
    print("\n🎯 Phase 6: Predictions")
    pred1_success, pred1_response = tester.test_prediction_high_qualified()
    if pred1_success:
        print(f"   High-qualified prediction: {pred1_response.get('prediction', 'Unknown')} ({pred1_response.get('probability', 0)*100:.1f}% confidence)")
    
    pred2_success, pred2_response = tester.test_prediction_low_qualified()
    if pred2_success:
        print(f"   Low-qualified prediction: {pred2_response.get('prediction', 'Unknown')} ({pred2_response.get('probability', 0)*100:.1f}% confidence)")
    
    # Final status check
    print("\n📊 Phase 7: Final Status Check")
    final_status_success, final_status_response = tester.test_model_status()
    if final_status_success:
        print(f"   Models trained: {final_status_response.get('models_trained', 0)}")
        print(f"   Model names: {final_status_response.get('model_names', [])}")
        print(f"   Preprocessors ready: {final_status_response.get('preprocessors_ready', False)}")
    
    # Print final results
    print("\n" + "=" * 60)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All tests passed! Backend API is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the backend implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())