import { useState, useRef } from 'react';
import { Upload, Brain, BarChart3, Activity, Users, TrendingUp, CheckCircle2, XCircle } from 'lucide-react';
import axios from 'axios';
import { toast } from 'sonner';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Dashboard = () => {
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [explorationData, setExplorationData] = useState(null);
  const [preprocessingDone, setPreprocessingDone] = useState(false);
  const [modelResults, setModelResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [currentStep, setCurrentStep] = useState(1);
  const fileInputRef = useRef(null);

  const [formData, setFormData] = useState({
    skills: '',
    experience: '',
    education: '',
    certifications: '',
    job_role: '',
    projects_count: '',
    salary_expectation: ''
  });

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API}/upload-dataset`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setDatasetInfo(response.data);
      setCurrentStep(2);
      toast.success('Dataset uploaded successfully!');
    } catch (error) {
      toast.error('Failed to upload dataset: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleExploration = async () => {
    setLoading(true);
    try {
      const response = await axios.get(`${API}/data-exploration`);
      setExplorationData(response.data);
      toast.success('Data exploration completed!');
    } catch (error) {
      toast.error('Failed to explore data: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handlePreprocess = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/preprocess`);
      setPreprocessingDone(true);
      setCurrentStep(3);
      toast.success(`Preprocessing complete! ${response.data.cleaned_rows} rows processed`);
    } catch (error) {
      toast.error('Failed to preprocess data: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModels = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API}/train-models`);
      setModelResults(response.data);
      setCurrentStep(4);
      toast.success('All models trained successfully!');
    } catch (error) {
      toast.error('Failed to train models: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const payload = {
        skills: formData.skills,
        experience: parseFloat(formData.experience),
        education: formData.education,
        certifications: formData.certifications,
        job_role: formData.job_role,
        projects_count: parseInt(formData.projects_count),
        salary_expectation: parseFloat(formData.salary_expectation)
      };
      const response = await axios.post(`${API}/predict`, payload);
      setPrediction(response.data);
      toast.success('Prediction generated!');
    } catch (error) {
      toast.error('Failed to make prediction: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const getBestModel = () => {
    if (modelResults.length === 0) return null;
    return modelResults.reduce((best, current) => 
      current.accuracy > best.accuracy ? current : best
    );
  };

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-5xl font-bold mb-3 gradient-text" data-testid="page-title">
            AI Hiring Prediction System
          </h1>
          <p className="text-slate-400 text-lg" data-testid="page-subtitle">
            End-to-End Machine Learning Resume Screening Platform
          </p>
        </div>

        {/* Progress Steps */}
        <div className="glass-effect rounded-2xl p-6 mb-6">
          <div className="flex items-center justify-between">
            {[
              { num: 1, label: 'Upload Dataset', icon: Upload },
              { num: 2, label: 'Explore & Preprocess', icon: Activity },
              { num: 3, label: 'Train Models', icon: Brain },
              { num: 4, label: 'Make Predictions', icon: TrendingUp }
            ].map((step, idx) => {
              const Icon = step.icon;
              return (
                <div key={step.num} className="flex items-center flex-1">
                  <div className="flex flex-col items-center">
                    <div
                      data-testid={`step-${step.num}`}
                      className={`w-12 h-12 rounded-full flex items-center justify-center mb-2 transition-all ${
                        currentStep >= step.num
                          ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white'
                          : 'bg-slate-700 text-slate-400'
                      }`}
                    >
                      <Icon size={20} />
                    </div>
                    <span className="text-xs text-slate-400 text-center">{step.label}</span>
                  </div>
                  {idx < 3 && (
                    <div className="flex-1 h-1 mx-4 bg-slate-700 rounded">
                      <div
                        className={`h-full rounded transition-all ${
                          currentStep > step.num ? 'bg-gradient-to-r from-blue-500 to-purple-600' : ''
                        }`}
                        style={{ width: currentStep > step.num ? '100%' : '0%' }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <Tabs defaultValue="workflow" className="space-y-6">
          <TabsList className="glass-effect" data-testid="tabs-list">
            <TabsTrigger value="workflow" data-testid="tab-workflow">ML Workflow</TabsTrigger>
            <TabsTrigger value="prediction" data-testid="tab-prediction">Prediction Tool</TabsTrigger>
          </TabsList>

          <TabsContent value="workflow" className="space-y-6">
            {/* Step 1: Upload Dataset */}
            <Card className="glass-effect border-slate-700" data-testid="upload-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="text-blue-400" size={24} />
                  Step 1: Upload Dataset
                </CardTitle>
                <CardDescription>Upload your resume screening dataset (CSV format)</CardDescription>
              </CardHeader>
              <CardContent>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  data-testid="file-input"
                />
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
                  data-testid="upload-button"
                >
                  {loading ? 'Uploading...' : 'Choose CSV File'}
                </Button>

                {datasetInfo && (
                  <div className="mt-6 space-y-4" data-testid="dataset-info">
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="stat-card">
                        <div className="text-2xl font-bold text-blue-400">{datasetInfo.rows}</div>
                        <div className="text-sm text-slate-400">Total Rows</div>
                      </div>
                      <div className="stat-card">
                        <div className="text-2xl font-bold text-purple-400">{datasetInfo.columns}</div>
                        <div className="text-sm text-slate-400">Columns</div>
                      </div>
                      <div className="stat-card">
                        <div className="text-2xl font-bold text-green-400">
                          {datasetInfo.target_distribution?.Hire || 0}
                        </div>
                        <div className="text-sm text-slate-400">Hired</div>
                      </div>
                      <div className="stat-card">
                        <div className="text-2xl font-bold text-red-400">
                          {datasetInfo.target_distribution?.Reject || 0}
                        </div>
                        <div className="text-sm text-slate-400">Rejected</div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Step 2: Data Exploration & Preprocessing */}
            {datasetInfo && (
              <Card className="glass-effect border-slate-700" data-testid="explore-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="text-purple-400" size={24} />
                    Step 2: Data Exploration & Preprocessing
                  </CardTitle>
                  <CardDescription>Analyze and clean your data</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-3">
                    <Button
                      onClick={handleExploration}
                      disabled={loading}
                      variant="outline"
                      className="border-purple-500 text-purple-400 hover:bg-purple-500/10"
                      data-testid="explore-button"
                    >
                      Explore Data
                    </Button>
                    <Button
                      onClick={handlePreprocess}
                      disabled={loading || preprocessingDone}
                      className="bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700"
                      data-testid="preprocess-button"
                    >
                      {preprocessingDone ? 'Preprocessed ✓' : 'Preprocess Data'}
                    </Button>
                  </div>

                  {explorationData && (
                    <div className="mt-6 space-y-4" data-testid="exploration-data">
                      <h3 className="text-lg font-semibold text-slate-200">Summary Statistics</h3>
                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(explorationData.summary_stats).map(([key, stats]) => (
                          <div key={key} className="glass-effect rounded-lg p-4">
                            <div className="font-medium text-slate-300 mb-2">{key}</div>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                              <div>
                                <span className="text-slate-500">Mean:</span>
                                <span className="ml-2 text-blue-400">{stats.mean?.toFixed(2)}</span>
                              </div>
                              <div>
                                <span className="text-slate-500">Std:</span>
                                <span className="ml-2 text-purple-400">{stats.std?.toFixed(2)}</span>
                              </div>
                              <div>
                                <span className="text-slate-500">Min:</span>
                                <span className="ml-2 text-green-400">{stats.min?.toFixed(2)}</span>
                              </div>
                              <div>
                                <span className="text-slate-500">Max:</span>
                                <span className="ml-2 text-red-400">{stats.max?.toFixed(2)}</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Step 3: Train Models */}
            {preprocessingDone && (
              <Card className="glass-effect border-slate-700" data-testid="train-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="text-green-400" size={24} />
                    Step 3: Train Machine Learning Models
                  </CardTitle>
                  <CardDescription>
                    Train and compare 4 different ML algorithms
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Button
                    onClick={handleTrainModels}
                    disabled={loading || modelResults.length > 0}
                    className="bg-gradient-to-r from-green-500 to-teal-600 hover:from-green-600 hover:to-teal-700"
                    data-testid="train-models-button"
                  >
                    {loading ? 'Training Models...' : modelResults.length > 0 ? 'Models Trained ✓' : 'Train All Models'}
                  </Button>

                  {modelResults.length > 0 && (
                    <div className="mt-6 space-y-4" data-testid="model-results">
                      <h3 className="text-lg font-semibold text-slate-200">Model Performance Comparison</h3>
                      <div className="grid gap-4">
                        {modelResults.map((result, idx) => (
                          <div
                            key={idx}
                            className="glass-effect rounded-lg p-4 card-hover"
                            data-testid={`model-result-${idx}`}
                          >
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="font-semibold text-slate-200">{result.model_name}</h4>
                              <span className="text-2xl font-bold text-blue-400">
                                {(result.accuracy * 100).toFixed(2)}%
                              </span>
                            </div>
                            <Progress value={result.accuracy * 100} className="h-2" />
                            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                              <div>
                                <span className="text-slate-500">Precision:</span>
                                <span className="ml-2 text-green-400">
                                  {(result.classification_report?.['1']?.precision * 100 || 0).toFixed(1)}%
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-500">Recall:</span>
                                <span className="ml-2 text-purple-400">
                                  {(result.classification_report?.['1']?.recall * 100 || 0).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {getBestModel() && (
                        <div className="mt-6 p-4 bg-gradient-to-r from-green-500/10 to-teal-500/10 border border-green-500/30 rounded-lg">
                          <div className="flex items-center gap-2">
                            <TrendingUp className="text-green-400" size={20} />
                            <span className="font-semibold text-green-400">Best Model:</span>
                            <span className="text-slate-200">{getBestModel().model_name}</span>
                            <span className="ml-auto text-green-400 font-bold">
                              {(getBestModel().accuracy * 100).toFixed(2)}% Accuracy
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="prediction">
            <Card className="glass-effect border-slate-700" data-testid="prediction-card">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="text-yellow-400" size={24} />
                  Candidate Hiring Prediction
                </CardTitle>
                <CardDescription>
                  Enter candidate details to predict hiring decision
                </CardDescription>
              </CardHeader>
              <CardContent>
                {modelResults.length === 0 ? (
                  <div className="text-center py-8 text-slate-400">
                    Please complete the ML workflow first to enable predictions
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="skills">Skills</Label>
                        <Input
                          id="skills"
                          placeholder="Python, Machine Learning, SQL"
                          value={formData.skills}
                          onChange={(e) => setFormData({ ...formData, skills: e.target.value })}
                          className="bg-slate-800 border-slate-600"
                          data-testid="input-skills"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="experience">Experience (Years)</Label>
                        <Input
                          id="experience"
                          type="number"
                          placeholder="5"
                          value={formData.experience}
                          onChange={(e) => setFormData({ ...formData, experience: e.target.value })}
                          className="bg-slate-800 border-slate-600"
                          data-testid="input-experience"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="education">Education</Label>
                        <Select
                          value={formData.education}
                          onValueChange={(value) => setFormData({ ...formData, education: value })}
                        >
                          <SelectTrigger className="bg-slate-800 border-slate-600" data-testid="select-education">
                            <SelectValue placeholder="Select education" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Bachelor's Degree">Bachelor's Degree</SelectItem>
                            <SelectItem value="Master's Degree">Master's Degree</SelectItem>
                            <SelectItem value="PhD">PhD</SelectItem>
                            <SelectItem value="High School">High School</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="certifications">Certifications</Label>
                        <Input
                          id="certifications"
                          placeholder="AWS, Azure, GCP"
                          value={formData.certifications}
                          onChange={(e) => setFormData({ ...formData, certifications: e.target.value })}
                          className="bg-slate-800 border-slate-600"
                          data-testid="input-certifications"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="job_role">Job Role</Label>
                        <Input
                          id="job_role"
                          placeholder="Data Scientist"
                          value={formData.job_role}
                          onChange={(e) => setFormData({ ...formData, job_role: e.target.value })}
                          className="bg-slate-800 border-slate-600"
                          data-testid="input-job-role"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="projects_count">Projects Count</Label>
                        <Input
                          id="projects_count"
                          type="number"
                          placeholder="10"
                          value={formData.projects_count}
                          onChange={(e) => setFormData({ ...formData, projects_count: e.target.value })}
                          className="bg-slate-800 border-slate-600"
                          data-testid="input-projects-count"
                        />
                      </div>

                      <div className="space-y-2">
                        <Label htmlFor="salary_expectation">Salary Expectation ($)</Label>
                        <Input
                          id="salary_expectation"
                          type="number"
                          placeholder="120000"
                          value={formData.salary_expectation}
                          onChange={(e) => setFormData({ ...formData, salary_expectation: e.target.value })}
                          className="bg-slate-800 border-slate-600"
                          data-testid="input-salary"
                        />
                      </div>
                    </div>

                    <Button
                      onClick={handlePredict}
                      disabled={loading}
                      className="w-full bg-gradient-to-r from-yellow-500 to-orange-600 hover:from-yellow-600 hover:to-orange-700"
                      data-testid="predict-button"
                    >
                      {loading ? 'Predicting...' : 'Generate Prediction'}
                    </Button>

                    {prediction && (
                      <div
                        className={`prediction-card ${
                          prediction.prediction === 'Hire' ? 'prediction-hire' : 'prediction-reject'
                        }`}
                        data-testid="prediction-result"
                      >
                        <div className="flex items-center justify-center gap-3 mb-4">
                          {prediction.prediction === 'Hire' ? (
                            <CheckCircle2 className="text-green-400" size={48} />
                          ) : (
                            <XCircle className="text-red-400" size={48} />
                          )}
                          <h3 className="text-3xl font-bold">
                            {prediction.prediction === 'Hire' ? (
                              <span className="text-green-400">HIRE</span>
                            ) : (
                              <span className="text-red-400">REJECT</span>
                            )}
                          </h3>
                        </div>
                        <Separator className="my-4" />
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-slate-400">Confidence Score:</span>
                            <span className="font-bold text-slate-200">
                              {(prediction.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <Progress value={prediction.probability * 100} className="h-2" />
                          <div className="text-xs text-slate-500 mt-2">
                            Model: {prediction.model_used}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Dashboard;