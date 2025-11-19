import React, { useState, useEffect } from 'react';
import { checkPureLLMHealth } from '../api/api';
import './LLMBackendToggle.css';

const LLMBackendToggle = ({ usePureLLM, onToggle }) => {
  const [healthStatus, setHealthStatus] = useState('checking');
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    checkHealth();
    // Check health every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkHealth = async () => {
    const health = await checkPureLLMHealth();
    setHealthStatus(health.healthy ? 'healthy' : 'unhealthy');
  };

  return (
    <div className="llm-backend-toggle">
      <div className="toggle-header">
        <span className="toggle-label">🦙 Pure LLM Mode</span>
        <label className="switch">
          <input
            type="checkbox"
            checked={usePureLLM}
            onChange={(e) => onToggle(e.target.checked)}
            disabled={healthStatus !== 'healthy'}
          />
          <span className="slider"></span>
        </label>
      </div>
      
      <div className="toggle-status">
        <span className={`status-dot ${healthStatus}`}></span>
        <span className="status-text">
          {healthStatus === 'healthy' ? 'Backend Online' : 
           healthStatus === 'unhealthy' ? 'Backend Offline' : 
           'Checking...'}
        </span>
        <button 
          className="details-btn"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? '▼' : '▶'}
        </button>
      </div>

      {showDetails && (
        <div className="toggle-details">
          <div className="detail-row">
            <span>Model:</span>
            <span>Llama 3.1 8B Instruct</span>
          </div>
          <div className="detail-row">
            <span>Server:</span>
            <span>Port 8002</span>
          </div>
          <div className="detail-row">
            <span>Features:</span>
            <span>10 Use Cases, Caching, RAG</span>
          </div>
          <button 
            className="refresh-btn"
            onClick={checkHealth}
          >
            🔄 Refresh
          </button>
        </div>
      )}

      {usePureLLM && (
        <div className="llm-badge">
          ✨ Using Llama 3.1 8B
        </div>
      )}
    </div>
  );
};

export default LLMBackendToggle;
