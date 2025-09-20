import React, { useState, useEffect } from 'react';
import { useTheme } from '../contexts/ThemeContext';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const GDPRDataManager = () => {
  const { darkMode } = useTheme();
  const [activeTab, setActiveTab] = useState('overview');
  const [email, setEmail] = useState('');
  const [requestType, setRequestType] = useState('access');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [consentStatus, setConsentStatus] = useState({});

  useEffect(() => {
    // Get or create session ID
    let currentSessionId = sessionStorage.getItem('ai-stanbul-session');
    if (!currentSessionId) {
      currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      sessionStorage.setItem('ai-stanbul-session', currentSessionId);
    }
    setSessionId(currentSessionId);

    // Load consent status
    loadConsentStatus(currentSessionId);
  }, []);

  const loadConsentStatus = async (sessionId) => {
    try {
      const response = await fetch(`${API_BASE}/gdpr/consent-status/${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        setConsentStatus(data.consent_status || {});
      }
    } catch (error) {
      console.error('Failed to load consent status:', error);
    }
  };

  const getUserData = () => {
    // Collect all user data stored locally
    const data = {
      cookieConsent: JSON.parse(localStorage.getItem('ai-istanbul-cookie-consent') || '{}'),
      themePreference: localStorage.getItem('ai-istanbul-theme') || 'light',
      sessionData: sessionStorage.getItem('ai-stanbul-session') || 'none',
      feedbacks: JSON.parse(localStorage.getItem('ai-stanbul-feedbacks') || '[]'),
      chatHistory: 'Session-based, not permanently stored',
      analytics: 'Anonymized usage data via Google Analytics'
    };
    return data;
  };

  const downloadUserData = () => {
    const userData = getUserData();
    const dataStr = JSON.stringify(userData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ai-istanbul-userdata-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const deleteUserData = () => {
    if (window.confirm('Are you sure you want to delete all your data? This action cannot be undone.')) {
      // Clear all local storage
      localStorage.removeItem('ai-istanbul-cookie-consent');
      localStorage.removeItem('ai-istanbul-theme');
      localStorage.removeItem('ai-stanbul-feedbacks');
      sessionStorage.removeItem('ai-stanbul-session');
      
      // Disable analytics
      if (window.gtag) {
        window.gtag('consent', 'update', {
          'analytics_storage': 'denied',
          'ad_storage': 'denied',
          'personalization_storage': 'denied'
        });
      }
      
      alert('Your data has been deleted successfully.');
      window.location.reload();
    }
  };

  const handleDataRequest = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      const endpoint = requestType === 'access' ? '/gdpr/data-request' : '/gdpr/data-deletion';
      
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          email: email
        })
      });

      const result = await response.json();
      
      if (result.status === 'success') {
        setSubmitStatus('success');
        setEmail('');
        
        if (requestType === 'access') {
          // For data access, the data is returned in the response
          console.log('User data:', result.data);
        }
      } else {
        setSubmitStatus('error');
      }
      
      setTimeout(() => setSubmitStatus(''), 5000);
    } catch (error) {
      console.error('GDPR request error:', error);
      setSubmitStatus('error');
      setTimeout(() => setSubmitStatus(''), 5000);
    } finally {
      setIsSubmitting(false);
    }
  };

  const updateConsent = async (consentData) => {
    try {
      const response = await fetch(`${API_BASE}/gdpr/consent`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          consent: consentData
        })
      });

      if (response.ok) {
        await loadConsentStatus(sessionId);
      }
    } catch (error) {
      console.error('Failed to update consent:', error);
    }
  };

  return (
    <div className={`max-w-4xl mx-auto p-6 transition-colors duration-300 ${
      darkMode ? 'text-white' : 'text-gray-800'
    }`}>
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">GDPR Data Management</h1>
        <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Manage your personal data and privacy preferences in compliance with GDPR
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 mb-6">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'data', label: 'My Data' }
        ].map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg transition-colors duration-200 ${
              activeTab === tab.id
                ? 'bg-blue-600 text-white'
                : darkMode
                  ? 'bg-gray-700 hover:bg-gray-600'
                  : 'bg-gray-100 hover:bg-gray-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          <div className={`p-6 rounded-lg border ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          }`}>
            <h2 className="text-xl font-semibold mb-4">🛡️ Your Privacy Status</h2>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <h3 className="font-medium">Cookie Consent</h3>
                <p className={`text-sm ${
                  localStorage.getItem('ai-istanbul-cookie-consent') 
                    ? 'text-green-600' 
                    : 'text-yellow-600'
                }`}>
                  {localStorage.getItem('ai-istanbul-cookie-consent') ? '✅ Configured' : '⚠️ Not set'}
                </p>
              </div>
              <div className="space-y-2">
                <h3 className="font-medium">Data Collection</h3>
                <p className="text-sm text-green-600">✅ Minimal & Transparent</p>
              </div>
            </div>
          </div>

          <div className={`p-6 rounded-lg border ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          }`}>
            <h2 className="text-xl font-semibold mb-4">📋 What We Collect</h2>
            <ul className={`space-y-2 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              <li>• Chat messages (processed temporarily, not stored permanently)</li>
              <li>• Usage analytics (anonymized)</li>
              <li>• Preferences (theme, language)</li>
              <li>• Feedback ratings</li>
              <li>• Technical data (IP address, browser type)</li>
            </ul>
          </div>
        </div>
      )}

      {/* My Data Tab */}
      {activeTab === 'data' && (
        <div className="space-y-6">
          <div className={`p-6 rounded-lg border ${
            darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
          }`}>
            <h2 className="text-xl font-semibold mb-4">📁 Your Data</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-medium mb-2">Local Data</h3>
                <pre className={`text-xs p-3 rounded overflow-auto ${
                  darkMode ? 'bg-gray-900' : 'bg-gray-50'
                }`}>
                  {JSON.stringify(getUserData(), null, 2)}
                </pre>
              </div>
              
              <div className="flex gap-3">
                <button
                  onClick={downloadUserData}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200"
                >
                  📥 Download My Data
                </button>
                <button
                  onClick={deleteUserData}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors duration-200"
                >
                  🗑️ Delete All Data
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GDPRDataManager;
