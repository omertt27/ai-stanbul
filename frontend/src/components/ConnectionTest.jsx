import React, { useState, useEffect } from 'react';

const ConnectionTest = () => {
  const [status, setStatus] = useState('Testing...');
  const [details, setDetails] = useState([]);

  const addDetail = (message, type = 'info') => {
    setDetails(prev => [...prev, { message, type, timestamp: Date.now() }]);
  };

  const testConnection = async () => {
    setStatus('🔍 Testing connection...');
    setDetails([]);
    
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
    const cleanApiUrl = apiUrl.replace(/\/ai\/?$/, '');
    addDetail(`Testing API URL: ${cleanApiUrl}`, 'info');

    try {
      // Test 1: Basic connection
      addDetail('Test 1: Basic connection to backend', 'info');
      const basicResponse = await fetch(`${cleanApiUrl}/`);
      if (basicResponse.ok) {
        const data = await basicResponse.json();
        addDetail(`✅ Basic connection successful: ${data.message}`, 'success');
      } else {
        throw new Error(`HTTP ${basicResponse.status}: ${basicResponse.statusText}`);
      }

      // Test 2: Restaurant API
      addDetail('Test 2: Restaurant search API', 'info');
      const restaurantResponse = await fetch(`${cleanApiUrl}/restaurants/search?limit=1`);
      if (restaurantResponse.ok) {
        const data = await restaurantResponse.json();
        addDetail(`✅ Restaurant API working: Found ${data.total_found} restaurants`, 'success');
      } else {
        throw new Error(`Restaurant API failed: HTTP ${restaurantResponse.status}`);
      }

      // Test 3: Main chat API
      addDetail('Test 3: Chat API', 'info');
      const chatResponse = await fetch(`${cleanApiUrl}/ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: 'test' }),
      });
      if (chatResponse.ok) {
        addDetail('✅ Chat API responding', 'success');
      } else {
        addDetail(`⚠️ Chat API issue: HTTP ${chatResponse.status}`, 'warning');
      }

      setStatus('✅ All tests completed successfully!');
    } catch (error) {
      addDetail(`❌ Connection failed: ${error.message}`, 'error');
      setStatus('❌ Connection test failed');
    }
  };

  useEffect(() => {
    testConnection();
  }, []);

  const getDetailColor = (type) => {
    switch (type) {
      case 'success': return 'text-green-600';
      case 'error': return 'text-red-600';
      case 'warning': return 'text-yellow-600';
      default: return 'text-blue-600';
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">🔧 Frontend Connection Test</h2>
      
      <div className="mb-4 p-3 bg-gray-100 rounded">
        <strong>Status:</strong> {status}
      </div>

      <div className="space-y-2">
        {details.map((detail, index) => (
          <div key={index} className={`p-2 text-sm ${getDetailColor(detail.type)}`}>
            {detail.message}
          </div>
        ))}
      </div>

      <button 
        onClick={testConnection}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        🔄 Run Test Again
      </button>

      <div className="mt-4 p-3 bg-yellow-50 border-l-4 border-yellow-400 text-sm">
        <strong>If tests fail:</strong>
        <ul className="mt-2 list-disc list-inside">
          <li>Make sure backend server is running: <code>uvicorn main:app --reload --port 8001</code></li>
          <li>Check that VITE_API_URL in .env is set to: <code>http://localhost:8000</code></li>
          <li>Restart your frontend dev server after making changes</li>
        </ul>
      </div>
    </div>
  );
};

export default ConnectionTest;
