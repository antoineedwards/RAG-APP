import React, { useState, useEffect } from 'react';
import { Upload, Search, Database, FileText, Activity, Brain, AlertCircle, CheckCircle, Loader2, User, Calendar, MapPin, DollarSign } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

const HealthcareApp = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [apiStatus, setApiStatus] = useState('checking');
  const [uploadFile, setUploadFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [processing, setProcessing] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [aiResponse, setAiResponse] = useState('');
  const [searching, setSearching] = useState(false);
  const [collections, setCollections] = useState([]);

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus();
    fetchCollections();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/`);
      if (response.ok) {
        setApiStatus('connected');
      } else {
        setApiStatus('error');
      }
    } catch (error) {
      setApiStatus('error');
    }
  };

  const fetchCollections = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/collections`);
      if (response.ok) {
        const data = await response.json();
        setCollections(data.collections || []);
      }
    } catch (error) {
      console.error('Error fetching collections:', error);
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'text/csv') {
      setUploadFile(file);
      setUploadStatus('');
    } else {
      setUploadStatus('Please select a valid CSV file');
    }
  };

  const processData = async () => {
    if (!uploadFile) {
      setUploadStatus('Please select a file first');
      return;
    }

    setProcessing(true);
    setUploadStatus('');

    try {
      const formData = new FormData();
      formData.append('file', uploadFile);

      const response = await fetch(`${API_BASE_URL}/process-data`, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadStatus(`✅ Successfully processed ${result.records_count} records`);
        fetchCollections(); // Refresh collections list
      } else {
        setUploadStatus(`❌ Error: ${result.detail}`);
      }
    } catch (error) {
      setUploadStatus(`❌ Error: ${error.message}`);
    } finally {
      setProcessing(false);
    }
  };

  const performSearch = async (withAI = false) => {
    if (!searchQuery.trim()) return;

    setSearching(true);
    setSearchResults([]);
    setAiResponse('');

    try {
      const endpoint = withAI ? '/search-with-chat' : '/search';
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: searchQuery,
          limit: 10,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        if (withAI) {
          setSearchResults(result.retrieved_records || []);
          setAiResponse(result.ai_response || '');
        } else {
          setSearchResults(result.records || []);
        }
      } else {
        setAiResponse(`Error: ${result.detail}`);
      }
    } catch (error) {
      setAiResponse(`Error: ${error.message}`);
    } finally {
      setSearching(false);
    }
  };

  const StatusIndicator = ({ status }) => {
    const statusConfig = {
      connected: { icon: CheckCircle, color: 'text-green-500', text: 'API Connected' },
      checking: { icon: Loader2, color: 'text-yellow-500', text: 'Checking...' },
      error: { icon: AlertCircle, color: 'text-red-500', text: 'API Disconnected' }
    };

    const { icon: Icon, color, text } = statusConfig[status];
    
    return (
      <div className={`flex items-center gap-2 ${color}`}>
        <Icon size={16} className={status === 'checking' ? 'animate-spin' : ''} />
        <span className="text-sm font-medium">{text}</span>
      </div>
    );
  };

  const RecordCard = ({ record, index }) => (
    <div className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
            <User size={20} className="text-blue-600" />
          </div>
          <div>
            <h3 className="font-semibold text-lg text-gray-900">
              {record.data?.Name || `Patient ${index + 1}`}
            </h3>
            <p className="text-sm text-gray-500">
              {record.data?.Age} years old • {record.data?.Gender}
            </p>
          </div>
        </div>
        {record.score && (
          <div className="bg-blue-50 px-3 py-1 rounded-full">
            <span className="text-blue-700 text-sm font-medium">
              Score: {record.score.toFixed(3)}
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Activity size={16} className="text-red-500" />
            <span className="text-sm font-medium text-gray-700">Medical Condition</span>
          </div>
          <p className="text-sm text-gray-600 pl-6">{record.data?.Medical_Condition}</p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Calendar size={16} className="text-green-500" />
            <span className="text-sm font-medium text-gray-700">Admission</span>
          </div>
          <p className="text-sm text-gray-600 pl-6">{record.data?.Date_of_Admission}</p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <MapPin size={16} className="text-purple-500" />
            <span className="text-sm font-medium text-gray-700">Hospital</span>
          </div>
          <p className="text-sm text-gray-600 pl-6">{record.data?.Hospital}</p>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <DollarSign size={16} className="text-yellow-500" />
            <span className="text-sm font-medium text-gray-700">Billing</span>
          </div>
          <p className="text-sm text-gray-600 pl-6">
            ${record.data?.Billing_Amount?.toLocaleString()}
          </p>
        </div>
      </div>

      <div className="border-t pt-4 space-y-3">
        <div>
          <span className="text-sm font-medium text-gray-700">Medication:</span>
          <p className="text-sm text-gray-600 mt-1">{record.data?.Medication}</p>
        </div>
        <div>
          <span className="text-sm font-medium text-gray-700">Test Results:</span>
          <p className="text-sm text-gray-600 mt-1">{record.data?.Test_Results}</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-lg flex items-center justify-center">
                <Activity size={24} className="text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Healthcare Data Manager</h1>
                <p className="text-sm text-gray-500">AI-Powered Patient Record Search</p>
              </div>
            </div>
            <StatusIndicator status={apiStatus} />
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg w-fit">
          {[
            { id: 'upload', label: 'Data Upload', icon: Upload },
            { id: 'search', label: 'Search Records', icon: Search },
            { id: 'collections', label: 'Collections', icon: Database },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="space-y-8">
            <div className="bg-white rounded-xl shadow-sm border p-8">
              <div className="text-center">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <FileText size={32} className="text-blue-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Healthcare Data</h2>
                <p className="text-gray-600 mb-8">
                  Upload a CSV file containing patient records to generate embeddings and enable AI search
                </p>

                <div className="max-w-md mx-auto">
                  <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100 transition-colors">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <Upload size={24} className="text-gray-400 mb-2" />
                      <p className="text-sm text-gray-500">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-gray-500">CSV files only</p>
                    </div>
                    <input type="file" className="hidden" accept=".csv" onChange={handleFileUpload} />
                  </label>

                  {uploadFile && (
                    <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                      <p className="text-sm text-blue-700">Selected: {uploadFile.name}</p>
                    </div>
                  )}

                  <button
                    onClick={processData}
                    disabled={!uploadFile || processing}
                    className="w-full mt-6 bg-gradient-to-r from-blue-500 to-indigo-600 text-white py-3 px-6 rounded-lg font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:from-blue-600 hover:to-indigo-700 transition-all flex items-center justify-center gap-2"
                  >
                    {processing ? (
                      <>
                        <Loader2 size={20} className="animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Database size={20} />
                        Process Data
                      </>
                    )}
                  </button>

                  {uploadStatus && (
                    <div className={`mt-4 p-3 rounded-lg ${
                      uploadStatus.includes('✅') 
                        ? 'bg-green-50 text-green-700' 
                        : 'bg-red-50 text-red-700'
                    }`}>
                      <p className="text-sm">{uploadStatus}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Search Tab */}
        {activeTab === 'search' && (
          <div className="space-y-8">
            {/* Search Interface */}
            <div className="bg-white rounded-xl shadow-sm border p-8">
              <div className="text-center mb-8">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Brain size={32} className="text-green-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">AI-Powered Search</h2>
                <p className="text-gray-600">
                  Search patient records using natural language queries
                </p>
              </div>

              <div className="max-w-2xl mx-auto">
                <div className="flex gap-4 mb-6">
                  <div className="flex-1">
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search for patients, conditions, medications..."
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      onKeyPress={(e) => e.key === 'Enter' && performSearch(false)}
                    />
                  </div>
                </div>

                <div className="flex gap-3 justify-center">
                  <button
                    onClick={() => performSearch(false)}
                    disabled={searching || !searchQuery.trim()}
                    className="bg-blue-500 text-white px-6 py-3 rounded-lg font-medium disabled:opacity-50 hover:bg-blue-600 transition-colors flex items-center gap-2"
                  >
                    <Search size={20} />
                    Vector Search
                  </button>
                  <button
                    onClick={() => performSearch(true)}
                    disabled={searching || !searchQuery.trim()}
                    className="bg-gradient-to-r from-green-500 to-emerald-600 text-white px-6 py-3 rounded-lg font-medium disabled:opacity-50 hover:from-green-600 hover:to-emerald-700 transition-all flex items-center gap-2"
                  >
                    <Brain size={20} />
                    AI Search & Analysis
                  </button>
                </div>

                {searching && (
                  <div className="mt-6 text-center">
                    <Loader2 size={24} className="animate-spin text-blue-500 mx-auto" />
                    <p className="text-gray-600 mt-2">Searching records...</p>
                  </div>
                )}
              </div>
            </div>

            {/* AI Response */}
            {aiResponse && (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl border border-green-200 p-8">
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
                    <Brain size={20} className="text-green-600" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-green-900 mb-3">AI Analysis</h3>
                    <div className="prose text-green-800 max-w-none">
                      {aiResponse.split('\n').map((line, index) => (
                        <p key={index} className="mb-2">{line}</p>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Search Results */}
            {searchResults.length > 0 && (
              <div className="space-y-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-xl font-semibold text-gray-900">
                    Search Results ({searchResults.length} found)
                  </h3>
                </div>
                <div className="grid gap-6">
                  {searchResults.map((record, index) => (
                    <RecordCard key={record.id || index} record={record} index={index} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Collections Tab */}
        {activeTab === 'collections' && (
          <div className="bg-white rounded-xl shadow-sm border p-8">
            <div className="text-center mb-8">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Database size={32} className="text-purple-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Collections</h2>
              <p className="text-gray-600">
                Manage your Milvus collections and data storage
              </p>
            </div>

            <div className="max-w-2xl mx-auto">
              {collections.length > 0 ? (
                <div className="space-y-4">
                  {collections.map((collection, index) => (
                    <div key={index} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                      <div className="flex items-center gap-3">
                        <Database size={20} className="text-purple-500" />
                        <span className="font-medium text-gray-900">{collection}</span>
                      </div>
                      <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm">
                        Active
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Database size={48} className="text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">No collections found. Upload data to create collections.</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default HealthcareApp;
