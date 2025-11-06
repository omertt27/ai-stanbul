/**
 * Route Controls Component
 * ========================
 * Controls for route planning (save, share, transport mode)
 * 
 * Features:
 * - Save route to database
 * - Share route via URL
 * - Export to GPX/KML
 * - Transport mode selector
 * - Route options
 */

import React, { useState } from 'react';
import './RouteControls.css';

const RouteControls = ({
  transportMode = 'walk',
  onTransportModeChange,
  onSaveRoute,
  onShareRoute,
  onExportRoute,
  onOptimizeRoute,
  isSaving = false,
  className = ''
}) => {
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [copied, setCopied] = useState(false);

  const transportModes = [
    { id: 'walk', icon: '🚶', label: 'Walk', color: '#3b82f6' },
    { id: 'drive', icon: '🚗', label: 'Drive', color: '#f59e0b' },
    { id: 'bike', icon: '🚴', label: 'Bike', color: '#10b981' },
    { id: 'transit', icon: '🚇', label: 'Transit', color: '#8b5cf6' }
  ];

  const handleShare = async () => {
    if (onShareRoute) {
      try {
        const url = await onShareRoute();
        setShareUrl(url);
        setShowShareModal(true);
      } catch (error) {
        console.error('Error sharing route:', error);
      }
    }
  };

  const handleCopyLink = () => {
    navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`route-controls ${className}`}>
      {/* Transport mode selector */}
      <div className="transport-modes">
        <label className="control-label">Travel Mode</label>
        <div className="mode-buttons">
          {transportModes.map((mode) => (
            <button
              key={mode.id}
              className={`mode-btn ${transportMode === mode.id ? 'active' : ''}`}
              onClick={() => onTransportModeChange && onTransportModeChange(mode.id)}
              style={{
                '--mode-color': mode.color
              }}
              title={mode.label}
            >
              <span className="mode-icon">{mode.icon}</span>
              <span className="mode-label">{mode.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Action buttons */}
      <div className="action-buttons">
        {/* Optimize route */}
        {onOptimizeRoute && (
          <button
            className="action-btn optimize-btn"
            onClick={onOptimizeRoute}
            title="Optimize route order"
          >
            <span className="btn-icon">🎯</span>
            <span className="btn-label">Optimize</span>
          </button>
        )}

        {/* Save route */}
        {onSaveRoute && (
          <button
            className="action-btn save-btn"
            onClick={onSaveRoute}
            disabled={isSaving}
            title="Save route"
          >
            <span className="btn-icon">{isSaving ? '⏳' : '💾'}</span>
            <span className="btn-label">{isSaving ? 'Saving...' : 'Save'}</span>
          </button>
        )}

        {/* Share route */}
        {onShareRoute && (
          <button
            className="action-btn share-btn"
            onClick={handleShare}
            title="Share route"
          >
            <span className="btn-icon">🔗</span>
            <span className="btn-label">Share</span>
          </button>
        )}

        {/* Export dropdown */}
        {onExportRoute && (
          <div className="export-dropdown">
            <button className="action-btn export-btn" title="Export route">
              <span className="btn-icon">📥</span>
              <span className="btn-label">Export</span>
            </button>
            <div className="dropdown-menu">
              <button onClick={() => onExportRoute('gpx')}>
                Export as GPX
              </button>
              <button onClick={() => onExportRoute('kml')}>
                Export as KML
              </button>
              <button onClick={() => onExportRoute('json')}>
                Export as JSON
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Share modal */}
      {showShareModal && (
        <div className="share-modal-overlay" onClick={() => setShowShareModal(false)}>
          <div className="share-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Share Route</h3>
              <button
                className="close-btn"
                onClick={() => setShowShareModal(false)}
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <p>Share this link with others to view your route:</p>
              <div className="share-link-container">
                <input
                  type="text"
                  value={shareUrl}
                  readOnly
                  className="share-link-input"
                />
                <button
                  className="copy-btn"
                  onClick={handleCopyLink}
                >
                  {copied ? '✓ Copied!' : '📋 Copy'}
                </button>
              </div>
              <div className="social-share">
                <button
                  className="social-btn whatsapp"
                  onClick={() => window.open(`https://wa.me/?text=${encodeURIComponent(shareUrl)}`)}
                >
                  <span>📱</span> WhatsApp
                </button>
                <button
                  className="social-btn twitter"
                  onClick={() => window.open(`https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=Check out my Istanbul route!`)}
                >
                  <span>🐦</span> Twitter
                </button>
                <button
                  className="social-btn facebook"
                  onClick={() => window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}`)}
                >
                  <span>📘</span> Facebook
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default RouteControls;
