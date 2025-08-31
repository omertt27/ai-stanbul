import React, { useState, useEffect } from 'react';
import { feedbackLogger } from '../utils/feedbackLogger';

const FeedbackDashboard = ({ isVisible, onClose }) => {
  const [stats, setStats] = useState({});
  const [feedbacks, setFeedbacks] = useState([]);

  useEffect(() => {
    if (isVisible) {
      refreshData();
    }
  }, [isVisible]);

  const refreshData = () => {
    setStats(feedbackLogger.getStatistics());
    setFeedbacks(feedbackLogger.getAllFeedbacks().reverse()); // Most recent first
  };

  const handleExport = () => {
    feedbackLogger.exportFeedbacks();
  };

  const handleClear = () => {
    if (window.confirm('Are you sure you want to clear all feedback data?')) {
      feedbackLogger.clearFeedbacks();
      refreshData();
    }
  };

  if (!isVisible) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      background: 'rgba(0, 0, 0, 0.8)',
      zIndex: 10000,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center'
    }}>
      <div style={{
        background: '#1e1e1e',
        borderRadius: '12px',
        padding: '2rem',
        maxWidth: '90vw',
        maxHeight: '90vh',
        overflow: 'auto',
        color: '#f3f4f6',
        minWidth: '600px'
      }}>
        {/* Header */}
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '1.5rem',
          borderBottom: '1px solid #374151',
          paddingBottom: '1rem'
        }}>
          <h2 style={{ margin: 0, color: '#818cf8' }}>📊 Feedback Dashboard</h2>
          <button 
            onClick={onClose}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#9ca3af',
              fontSize: '1.5rem',
              cursor: 'pointer',
              padding: '0.5rem'
            }}
          >
            ✕
          </button>
        </div>

        {/* Statistics */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: '1rem',
          marginBottom: '2rem'
        }}>
          <div style={{
            background: 'rgba(129, 140, 248, 0.1)',
            padding: '1rem',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#818cf8' }}>
              {stats.total || 0}
            </div>
            <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>Total Feedbacks</div>
          </div>

          <div style={{
            background: 'rgba(16, 185, 129, 0.1)',
            padding: '1rem',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#10b981' }}>
              {stats.good || 0} ({stats.goodPercentage}%)
            </div>
            <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>👍 Good Answers</div>
          </div>

          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            padding: '1rem',
            borderRadius: '8px',
            textAlign: 'center'
          }}>
            <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#ef4444' }}>
              {stats.bad || 0} ({stats.badPercentage}%)
            </div>
            <div style={{ fontSize: '0.875rem', color: '#9ca3af' }}>👎 Bad Answers</div>
          </div>
        </div>

        {/* Actions */}
        <div style={{
          display: 'flex',
          gap: '1rem',
          marginBottom: '2rem'
        }}>
          <button
            onClick={refreshData}
            style={{
              background: '#374151',
              color: '#f3f4f6',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            🔄 Refresh
          </button>
          
          <button
            onClick={handleExport}
            style={{
              background: '#059669',
              color: 'white',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            💾 Export JSON
          </button>
          
          <button
            onClick={handleClear}
            style={{
              background: '#dc2626',
              color: 'white',
              border: 'none',
              padding: '0.5rem 1rem',
              borderRadius: '6px',
              cursor: 'pointer'
            }}
          >
            🗑️ Clear All
          </button>
        </div>

        {/* Feedback List */}
        <div>
          <h3 style={{ color: '#e5e7eb', marginBottom: '1rem' }}>Recent Feedback</h3>
          <div style={{
            maxHeight: '300px',
            overflow: 'auto',
            background: '#111827',
            borderRadius: '8px',
            padding: '1rem'
          }}>
            {feedbacks.length === 0 ? (
              <p style={{ color: '#9ca3af', textAlign: 'center' }}>No feedback recorded yet</p>
            ) : (
              feedbacks.map((feedback, idx) => (
                <div key={feedback.id} style={{
                  padding: '1rem',
                  borderBottom: idx < feedbacks.length - 1 ? '1px solid #374151' : 'none',
                  marginBottom: idx < feedbacks.length - 1 ? '1rem' : 0
                }}>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginBottom: '0.5rem' 
                  }}>
                    <span style={{
                      background: feedback.feedbackType === 'good' ? '#065f46' : '#7f1d1d',
                      color: 'white',
                      padding: '0.25rem 0.5rem',
                      borderRadius: '12px',
                      fontSize: '0.75rem'
                    }}>
                      {feedback.feedbackType === 'good' ? '👍 Good' : '👎 Bad'}
                    </span>
                    <span style={{ fontSize: '0.75rem', color: '#9ca3af' }}>
                      {new Date(feedback.timestamp).toLocaleString()}
                    </span>
                  </div>
                  
                  {feedback.userQuery && (
                    <div style={{ marginBottom: '0.5rem' }}>
                      <strong style={{ color: '#818cf8', fontSize: '0.875rem' }}>Query:</strong>
                      <span style={{ color: '#d1d5db', fontSize: '0.875rem', marginLeft: '0.5rem' }}>
                        {feedback.userQuery}
                      </span>
                    </div>
                  )}
                  
                  <div>
                    <strong style={{ color: '#10b981', fontSize: '0.875rem' }}>Answer:</strong>
                    <div style={{ 
                      color: '#e5e7eb', 
                      fontSize: '0.875rem', 
                      marginTop: '0.25rem',
                      maxHeight: '60px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis'
                    }}>
                      {feedback.messageText}...
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeedbackDashboard;
