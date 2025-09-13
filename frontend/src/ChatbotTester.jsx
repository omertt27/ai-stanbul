import React, { useState, useEffect } from 'react';
import { testInputs, testsByCategory, analyzeResults } from './test-inputs';
import { fetchStreamingResults } from './api/api';

const ChatbotTester = () => {
  const [testResults, setTestResults] = useState([]);
  const [currentTest, setCurrentTest] = useState(null);
  const [isTestingAll, setIsTestingAll] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [testProgress, setTestProgress] = useState(0);
  const [analysis, setAnalysis] = useState(null);

  // Test a single input
  const testSingleInput = async (testCase) => {
    setCurrentTest(testCase);
    
    try {
      let fullResponse = '';
      
      await fetchStreamingResults(testCase.input, (chunk) => {
        fullResponse += chunk;
      });

      const result = {
        ...testCase,
        response: fullResponse,
        success: true,
        responseLength: fullResponse.length,
        timestamp: new Date().toISOString(),
        containsExpectedTopics: testCase.expectedTopics.some(topic => 
          fullResponse.toLowerCase().includes(topic.toLowerCase())
        )
      };

      setTestResults(prev => [...prev, result]);
      setCurrentTest(null);
      
      return result;
    } catch (error) {
      const result = {
        ...testCase,
        response: null,
        success: false,
        error: error.message,
        timestamp: new Date().toISOString(),
        containsExpectedTopics: false
      };

      setTestResults(prev => [...prev, result]);
      setCurrentTest(null);
      
      return result;
    }
  };

  // Test all inputs or by category
  const testAll = async () => {
    setIsTestingAll(true);
    setTestResults([]);
    setTestProgress(0);
    
    const testsToRun = selectedCategory === 'all' 
      ? testInputs 
      : testsByCategory[selectedCategory] || [];

    for (let i = 0; i < testsToRun.length; i++) {
      setTestProgress(((i + 1) / testsToRun.length) * 100);
      await testSingleInput(testsToRun[i]);
      
      // Add delay between tests to avoid overwhelming the API
      if (i < testsToRun.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 3000));
      }
    }
    
    setIsTestingAll(false);
    setTestProgress(100);
  };

  // Analyze results when they change
  useEffect(() => {
    if (testResults.length > 0) {
      const analysisData = analyzeResults(testResults);
      
      // Add additional metrics
      analysisData.avgResponseLength = testResults
        .filter(r => r.success)
        .reduce((acc, r) => acc + (r.responseLength || 0), 0) / 
        testResults.filter(r => r.success).length;
      
      analysisData.relevantResponses = testResults
        .filter(r => r.success && r.containsExpectedTopics).length;
        
      setAnalysis(analysisData);
    }
  }, [testResults]);

  // Clear results
  const clearResults = () => {
    setTestResults([]);
    setAnalysis(null);
    setTestProgress(0);
  };

  // Export results to JSON
  const exportResults = () => {
    const dataStr = JSON.stringify({ testResults, analysis }, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `chatbot-test-results-${new Date().toISOString().split('T')[0]}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-900 text-white min-h-screen">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-4">🤖 Istanbul Guide Chatbot Tester</h1>
        <p className="text-gray-300 text-lg">
          Test the chatbot with 50 comprehensive questions about Istanbul tourism
        </p>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label htmlFor="category" className="block text-sm font-medium mb-2">
              Test Category
            </label>
            <select
              id="category"
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
              disabled={isTestingAll}
            >
              <option value="all">All Categories (50 tests)</option>
              <option value="restaurants">Restaurants (10 tests)</option>
              <option value="attractions">Attractions (10 tests)</option>
              <option value="districts">Districts (8 tests)</option>
              <option value="transportation">Transportation (6 tests)</option>
              <option value="culture">Culture (6 tests)</option>
              <option value="shopping">Shopping (4 tests)</option>
              <option value="nightlife">Nightlife (3 tests)</option>
              <option value="practical">Practical (3 tests)</option>
            </select>
          </div>

          <div className="flex gap-3">
            <button
              onClick={testAll}
              disabled={isTestingAll}
              className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 rounded-lg font-medium transition-colors"
            >
              {isTestingAll ? 'Testing...' : 'Start Testing'}
            </button>
            
            <button
              onClick={clearResults}
              disabled={isTestingAll}
              className="px-6 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded-lg font-medium transition-colors"
            >
              Clear Results
            </button>

            {testResults.length > 0 && (
              <button
                onClick={exportResults}
                className="px-6 py-2 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors"
              >
                Export Results
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        {isTestingAll && (
          <div className="mt-4">
            <div className="flex justify-between text-sm mb-2">
              <span>Testing Progress</span>
              <span>{Math.round(testProgress)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${testProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Current Test */}
        {currentTest && (
          <div className="mt-4 p-4 bg-yellow-900 border border-yellow-600 rounded-lg">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-400 mr-3"></div>
              <div>
                <p className="font-medium">Testing: {currentTest.category}</p>
                <p className="text-sm text-yellow-200">{currentTest.input}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Analysis Results */}
      {analysis && (
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">📊 Test Analysis</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-blue-900 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-blue-300">{analysis.totalTests}</div>
              <div className="text-sm text-blue-200">Total Tests</div>
            </div>
            
            <div className="bg-green-900 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-green-300">{analysis.successful}</div>
              <div className="text-sm text-green-200">Successful</div>
            </div>
            
            <div className="bg-red-900 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-red-300">{analysis.failed}</div>
              <div className="text-sm text-red-200">Failed</div>
            </div>
            
            <div className="bg-purple-900 rounded-lg p-4 text-center">
              <div className="text-2xl font-bold text-purple-300">{analysis.relevantResponses}</div>
              <div className="text-sm text-purple-200">Relevant Responses</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Success Rate by Category */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Success Rate by Category</h3>
              <div className="space-y-2">
                {Object.entries(analysis.categories).map(([category, stats]) => (
                  <div key={category} className="flex items-center justify-between bg-gray-700 rounded px-3 py-2">
                    <span className="capitalize">{category}</span>
                    <div className="flex items-center">
                      <span className="text-sm text-gray-300 mr-2">
                        {stats.successful}/{stats.total}
                      </span>
                      <span className={`font-medium ${
                        stats.successful === stats.total ? 'text-green-400' : 
                        stats.successful > stats.total / 2 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {Math.round((stats.successful / stats.total) * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Overall Metrics */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Overall Metrics</h3>
              <div className="space-y-3">
                <div className="bg-gray-700 rounded px-3 py-2">
                  <div className="flex justify-between">
                    <span>Success Rate</span>
                    <span className="font-medium text-green-400">
                      {Math.round((analysis.successful / analysis.totalTests) * 100)}%
                    </span>
                  </div>
                </div>
                
                <div className="bg-gray-700 rounded px-3 py-2">
                  <div className="flex justify-between">
                    <span>Relevance Rate</span>
                    <span className="font-medium text-blue-400">
                      {Math.round((analysis.relevantResponses / analysis.successful) * 100)}%
                    </span>
                  </div>
                </div>
                
                <div className="bg-gray-700 rounded px-3 py-2">
                  <div className="flex justify-between">
                    <span>Avg Response Length</span>
                    <span className="font-medium text-purple-400">
                      {Math.round(analysis.avgResponseLength || 0)} chars
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Test Results */}
      {testResults.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-bold mb-4">🧪 Test Results</h2>
          
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {testResults.map((result, index) => (
              <div 
                key={index}
                className={`border rounded-lg p-4 ${
                  result.success 
                    ? result.containsExpectedTopics 
                      ? 'border-green-600 bg-green-900/20' 
                      : 'border-yellow-600 bg-yellow-900/20'
                    : 'border-red-600 bg-red-900/20'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <span className="text-sm font-medium text-gray-300">{result.category}</span>
                    <h3 className="font-medium">{result.input}</h3>
                  </div>
                  <div className="flex items-center space-x-2">
                    {result.success ? (
                      result.containsExpectedTopics ? (
                        <span className="px-2 py-1 bg-green-600 text-white text-xs rounded">✓ Relevant</span>
                      ) : (
                        <span className="px-2 py-1 bg-yellow-600 text-white text-xs rounded">✓ Success</span>
                      )
                    ) : (
                      <span className="px-2 py-1 bg-red-600 text-white text-xs rounded">✗ Failed</span>
                    )}
                  </div>
                </div>
                
                {result.success ? (
                  <div className="text-sm text-gray-300 bg-gray-900 rounded p-3 mt-2">
                    <p className="mb-2"><strong>Response:</strong></p>
                    <p className="whitespace-pre-wrap">{result.response?.substring(0, 300)}
                      {result.response?.length > 300 && '...'}</p>
                    <p className="text-xs text-gray-500 mt-2">
                      Length: {result.responseLength} chars | 
                      Expected topics: {result.expectedTopics.join(', ')}
                    </p>
                  </div>
                ) : (
                  <div className="text-sm text-red-300 bg-red-900/30 rounded p-3 mt-2">
                    <strong>Error:</strong> {result.error}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Test Buttons */}
      <div className="mt-6 bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">🚀 Quick Tests</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {testInputs.slice(0, 8).map((test) => (
            <button
              key={test.id}
              onClick={() => testSingleInput(test)}
              disabled={isTestingAll || currentTest}
              className="text-left p-3 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-600 rounded-lg transition-colors text-sm"
            >
              <div className="font-medium text-white mb-1">{test.category}</div>
              <div className="text-gray-300 text-xs">{test.input.substring(0, 50)}...</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChatbotTester;
