import React, { useState, useEffect } from 'react';
import { useTheme } from '../contexts/ThemeContext';
import { useTranslation } from 'react-i18next';
import BlogAnalyticsDashboard from '../components/BlogAnalyticsDashboard';

const AdminDashboard = () => {
  const { darkMode } = useTheme();
  const { t } = useTranslation();
  const [authenticated, setAuthenticated] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const API_BASE_URL = process.env.NODE_ENV === 'production' 
        ? 'https://ai-istanbul-backend.render.com'
        : 'http://localhost:8000';

      const response = await fetch(`${API_BASE_URL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('adminAuth', 'true');
        localStorage.setItem('authToken', data.access_token);
        setAuthenticated(true);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Invalid credentials');
      }
    } catch (err) {
      setError('Connection error. Please try again.');
      console.error('Login error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    setAuthenticated(false);
    setUsername('');
    setPassword('');
    localStorage.removeItem('adminAuth');
    localStorage.removeItem('authToken');
  };

  // Check if already authenticated
  useEffect(() => {
    const isAuth = localStorage.getItem('adminAuth') === 'true';
    const token = localStorage.getItem('authToken');
    if (isAuth && token) {
      setAuthenticated(true);
    }
  }, []);

  if (!authenticated) {
    return (
      <div className={`min-h-screen flex items-center justify-center transition-colors duration-200 ${
        darkMode ? 'bg-gray-900' : 'bg-gray-50'
      }`}>
        <div className={`max-w-md w-full mx-4 p-8 rounded-xl shadow-xl transition-colors duration-200 ${
          darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
        }`}>
          <div className="text-center mb-8">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gradient-to-r from-indigo-600 to-purple-600 flex items-center justify-center">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
              </svg>
            </div>
            <h2 className={`text-2xl font-bold transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>
              {t('admin.loginTitle')}
            </h2>
            <p className={`mt-2 transition-colors duration-200 ${
              darkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              {t('admin.loginSubtitle')}
            </p>
          </div>

          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label htmlFor="username" className={`block text-sm font-medium mb-2 transition-colors duration-200 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Username
              </label>
              <input
                type="text"
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className={`w-full px-4 py-3 border rounded-xl focus:outline-none transition-all duration-200 ${
                  darkMode 
                    ? 'bg-gray-700 text-white border-gray-600 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20' 
                    : 'bg-white text-gray-900 border-gray-300 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20'
                }`}
                placeholder="Enter username"
                required
              />
            </div>

            <div>
              <label htmlFor="password" className={`block text-sm font-medium mb-2 transition-colors duration-200 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}>
                Password
              </label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className={`w-full px-4 py-3 border rounded-xl focus:outline-none transition-all duration-200 ${
                  darkMode 
                    ? 'bg-gray-700 text-white border-gray-600 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20' 
                    : 'bg-white text-gray-900 border-gray-300 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20'
                }`}
                placeholder={t('admin.passwordPlaceholder')}
                required
              />
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                <p className="text-red-500 text-sm">{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-semibold rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
            >
              {loading ? 'Signing in...' : t('admin.loginButton')}
            </button>
          </form>

          <div className={`mt-6 p-4 rounded-lg transition-colors duration-200 ${
            darkMode ? 'bg-gray-700/50' : 'bg-gray-50'
          }`}>
            <p className={`text-sm transition-colors duration-200 ${
              darkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              <strong>Secure Login:</strong> Use your admin credentials
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen transition-colors duration-200 ${
      darkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className={`mb-8 p-6 rounded-xl shadow-xl transition-colors duration-200 ${
          darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
        }`}>
          <div className="flex justify-between items-center">
            <div>
              <h1 className={`text-3xl font-bold transition-colors duration-200 ${
                darkMode ? 'text-white' : 'text-gray-900'
              }`}>
                {t('admin.title')}
              </h1>
              <p className={`mt-2 transition-colors duration-200 ${
                darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}>
                {t('admin.welcome')}
              </p>
            </div>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors duration-200"
            >
              {t('admin.logout')}
            </button>
          </div>
        </div>

        {/* Analytics Dashboard */}
        <div className={`rounded-xl shadow-xl transition-colors duration-200 ${
          darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
        }`}>
          <BlogAnalyticsDashboard />
        </div>

        {/* Additional Admin Features */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Quick Actions */}
          <div className={`p-6 rounded-xl shadow-xl transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
          }`}>
            <h3 className={`text-xl font-bold mb-4 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>
              Quick Actions
            </h3>
            <div className="space-y-3">
              <button className="w-full p-3 text-left rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white transition-colors duration-200">
                📝 Create New Post
              </button>
              <button className="w-full p-3 text-left rounded-lg bg-green-600 hover:bg-green-700 text-white transition-colors duration-200">
                📊 Export Analytics
              </button>
              <button className="w-full p-3 text-left rounded-lg bg-purple-600 hover:bg-purple-700 text-white transition-colors duration-200">
                🔧 Manage Content
              </button>
            </div>
          </div>

          {/* System Status */}
          <div className={`p-6 rounded-xl shadow-xl transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
          }`}>
            <h3 className={`text-xl font-bold mb-4 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>
              System Status
            </h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>API Server</span>
                <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full">Online</span>
              </div>
              <div className="flex items-center justify-between">
                <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Database</span>
                <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full">Connected</span>
              </div>
              <div className="flex items-center justify-between">
                <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>Weather API</span>
                <span className="px-2 py-1 bg-green-500 text-white text-xs rounded-full">Active</span>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className={`p-6 rounded-xl shadow-xl transition-colors duration-200 ${
            darkMode ? 'bg-gray-800 border border-gray-700' : 'bg-white border border-gray-200'
          }`}>
            <h3 className={`text-xl font-bold mb-4 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>
              Recent Activity
            </h3>
            <div className="space-y-3">
              <div className={`p-3 rounded-lg transition-colors duration-200 ${
                darkMode ? 'bg-gray-700' : 'bg-gray-50'
              }`}>
                <p className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  New blog post published
                </p>
                <p className={`text-xs transition-colors duration-200 ${
                  darkMode ? 'text-gray-500' : 'text-gray-500'
                }`}>
                  2 hours ago
                </p>
              </div>
              <div className={`p-3 rounded-lg transition-colors duration-200 ${
                darkMode ? 'bg-gray-700' : 'bg-gray-50'
              }`}>
                <p className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Analytics updated
                </p>
                <p className={`text-xs transition-colors duration-200 ${
                  darkMode ? 'text-gray-500' : 'text-gray-500'
                }`}>
                  5 minutes ago
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
