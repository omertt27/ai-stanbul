import React, { useState, useRef, useEffect } from 'react';
import { useNotifications } from '../contexts/NotificationContext';
import './NotificationPanel.css';

const NotificationPanel = () => {
  const {
    notifications,
    unreadCount,
    isConnected,
    markAsRead,
    markAllAsRead,
    requestPermission,
    preferences,
    updatePreferences
  } = useNotifications();

  const [isOpen, setIsOpen] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const panelRef = useRef(null);

  // Close panel when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (panelRef.current && !panelRef.current.contains(event.target)) {
        setIsOpen(false);
        setShowSettings(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  const handleNotificationClick = async (notification) => {
    if (!notification.read) {
      await markAsRead(notification.id);
    }

    // Navigate to action URL if available
    if (notification.action_url) {
      window.location.href = notification.action_url;
    }
  };

  const handlePermissionRequest = async () => {
    const permission = await requestPermission();
    if (permission === 'granted') {
      alert('✅ Notifications enabled! You\'ll now receive real-time updates.');
    } else {
      alert('❌ Notifications blocked. You can enable them in your browser settings.');
    }
  };

  const handlePreferenceChange = (type, enabled) => {
    updatePreferences({
      ...preferences,
      [type]: enabled
    });
  };

  const getNotificationIcon = (type) => {
    const icons = {
      'route_update': '🗺️',
      'attraction_recommendation': '🏛️',
      'weather_alert': '🌦️',
      'traffic_update': '🚦',
      'personalized_tip': '💡',
      'chat_response': '💬',
      'system_message': '🔔',
      'location_based': '📍'
    };
    return icons[type] || '📢';
  };

  const formatTime = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  const getPriorityClass = (priority) => {
    switch (priority) {
      case 'urgent': return 'priority-urgent';
      case 'high': return 'priority-high';
      case 'normal': return 'priority-normal';
      case 'low': return 'priority-low';
      default: return 'priority-normal';
    }
  };

  return (
    <div className="notification-panel" ref={panelRef}>
      {/* Notification Bell Button */}
      <button 
        className={`notification-bell ${unreadCount > 0 ? 'has-unread' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        title={`${unreadCount} unread notifications`}
      >
        🔔
        {unreadCount > 0 && (
          <span className="notification-badge">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      {/* Notification Panel */}
      {isOpen && (
        <div className="notification-dropdown">
          <div className="notification-header">
            <h3>Notifications</h3>
            <div className="notification-actions">
              <button 
                className="connection-status"
                title={isConnected ? 'Connected' : 'Disconnected'}
              >
                <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></span>
              </button>
              <button 
                className="settings-btn"
                onClick={() => setShowSettings(!showSettings)}
                title="Settings"
              >
                ⚙️
              </button>
              {unreadCount > 0 && (
                <button 
                  className="mark-all-read"
                  onClick={markAllAsRead}
                  title="Mark all as read"
                >
                  ✓
                </button>
              )}
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="notification-settings">
              <h4>Notification Settings</h4>
              
              {/* Permission Request */}
              {Notification.permission !== 'granted' && (
                <div className="permission-request">
                  <p>Enable browser notifications for real-time updates:</p>
                  <button onClick={handlePermissionRequest} className="enable-btn">
                    Enable Notifications
                  </button>
                </div>
              )}

              {/* Preferences */}
              <div className="preferences">
                <h5>Notification Types</h5>
                {Object.entries(preferences).map(([type, enabled]) => (
                  <label key={type} className="preference-item">
                    <input
                      type="checkbox"
                      checked={enabled}
                      onChange={(e) => handlePreferenceChange(type, e.target.checked)}
                    />
                    <span className="preference-label">
                      {type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </span>
                  </label>
                ))}
              </div>
            </div>
          )}

          {/* Notifications List */}
          <div className="notifications-list">
            {notifications.length === 0 ? (
              <div className="no-notifications">
                <p>🔕 No notifications yet</p>
                <small>You'll see updates here when they arrive</small>
              </div>
            ) : (
              notifications.map((notification) => (
                <div
                  key={notification.id}
                  className={`notification-item ${!notification.read ? 'unread' : ''} ${getPriorityClass(notification.priority)}`}
                  onClick={() => handleNotificationClick(notification)}
                >
                  <div className="notification-icon">
                    {notification.icon || getNotificationIcon(notification.type)}
                  </div>
                  <div className="notification-content">
                    <div className="notification-title">
                      {notification.title}
                    </div>
                    <div className="notification-message">
                      {notification.message}
                    </div>
                    <div className="notification-meta">
                      <span className="notification-time">
                        {formatTime(notification.created_at)}
                      </span>
                      <span className="notification-type">
                        {notification.type.replace(/_/g, ' ')}
                      </span>
                    </div>
                  </div>
                  {!notification.read && (
                    <div className="unread-indicator"></div>
                  )}
                </div>
              ))
            )}
          </div>

          {/* Footer */}
          <div className="notification-footer">
            <small>
              {isConnected ? '🟢 Real-time updates active' : '🔴 Reconnecting...'}
            </small>
          </div>
        </div>
      )}
    </div>
  );
};

export default NotificationPanel;
