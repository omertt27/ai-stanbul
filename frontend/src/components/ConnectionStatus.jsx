import React, { useState, useEffect } from 'react';

const ConnectionStatus = () => {
  const [status, setStatus] = useState('checking');
  const [lastCheck, setLastCheck] = useState(null);

  const checkConnection = async () => {
    setStatus('checking');
    const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8001';
    
    try {
      const response = await fetch(`${apiUrl}/`, { 
        method: 'GET',
        timeout: 5000 
      });
      
      if (response.ok) {
        setStatus('connected');
        setLastCheck(new Date().toLocaleTimeString());
      } else {
        setStatus('error');
      }
    } catch (error) {
      console.error('Connection check failed:', error);
      setStatus('error');
    }
  };

  useEffect(() => {
    checkConnection();
    // Check connection every 30 seconds
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        return {
          color: 'bg-green-500',
          text: 'Connected',
          textColor: 'text-green-700',
          icon: '✅'
        };
      case 'error':
        return {
          color: 'bg-red-500',
          text: 'Connection Failed',
          textColor: 'text-red-700',
          icon: '❌'
        };
      case 'checking':
        return {
          color: 'bg-yellow-500',
          text: 'Checking...',
          textColor: 'text-yellow-700',
          icon: '🔄'
        };
      default:
        return {
          color: 'bg-gray-500',
          text: 'Unknown',
          textColor: 'text-gray-700',
          icon: '❓'
        };
    }
  };

  const config = getStatusConfig();

  return (
    <div className="fixed top-4 right-4 z-50">
      <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg shadow-lg bg-white border-l-4 ${config.color.replace('bg-', 'border-')}`}>
        <span className="text-sm">{config.icon}</span>
        <div className="text-sm">
          <div className={`font-semibold ${config.textColor}`}>
            Backend: {config.text}
          </div>
          {lastCheck && status === 'connected' && (
            <div className="text-xs text-gray-500">
              Last check: {lastCheck}
            </div>
          )}
          {status === 'error' && (
            <div className="text-xs text-red-600">
              Check backend server on port 8001
            </div>
          )}
        </div>
        <button 
          onClick={checkConnection}
          className="text-xs px-2 py-1 bg-gray-100 rounded hover:bg-gray-200"
          title="Check connection"
        >
          🔄
        </button>
      </div>
    </div>
  );
};

export default ConnectionStatus;
