import React from 'react';

const DebugInfo = () => {
  const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const cleanBaseUrl = baseUrl.replace(/\/ai\/?$/, '');
  const apiUrl = `${cleanBaseUrl}/ai`;

  return (
    <div style={{ 
      position: 'fixed', 
      top: '10px', 
      right: '10px', 
      background: 'rgba(0,0,0,0.8)', 
      color: 'white', 
      padding: '10px', 
      fontSize: '12px',
      borderRadius: '5px',
      zIndex: 1000
    }}>
      <div>Base URL: {baseUrl}</div>
      <div>API URL: {apiUrl}</div>
      <div>Environment: {import.meta.env.MODE}</div>
    </div>
  );
};

export default DebugInfo;
