import React, { createContext, useContext, useState, useEffect, useRef } from 'react';

// 🔧 TEMPORARY: Disable notifications until backend endpoints are ready
const NOTIFICATIONS_ENABLED = false;

const NotificationContext = createContext();

export const useNotifications = () => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
};

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [userId, setUserId] = useState(null);
  const [pushSupported, setPushSupported] = useState(false);
  const [deviceToken, setDeviceToken] = useState(null);
  const [preferences, setPreferences] = useState({
    route_updates: true,
    attraction_recommendations: true,
    weather_alerts: true,
    traffic_updates: true,
    personalized_tips: true,
    chat_responses: true
  });
  
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  // Initialize notifications system
  useEffect(() => {
    if (!NOTIFICATIONS_ENABLED) {
      console.log('ℹ️ Notifications disabled - backend endpoints not ready yet');
      return;
    }
    
    initializeNotifications();
    return () => {
      cleanup();
    };
  }, []);

  const initializeNotifications = async () => {
    // Generate or get user ID
    let storedUserId = localStorage.getItem('user_id');
    if (!storedUserId) {
      storedUserId = generateUserId();
      localStorage.setItem('user_id', storedUserId);
    }
    setUserId(storedUserId);

    // Check push notification support
    if ('serviceWorker' in navigator && 'PushManager' in window) {
      setPushSupported(true);
      await initializePushNotifications();
    }

    // Load stored notifications
    loadStoredNotifications();

    // Connect WebSocket
    connectWebSocket(storedUserId);

    // Fetch existing notifications from server
    await fetchNotifications(storedUserId);
  };

  const generateUserId = () => {
    return 'user_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
  };

  const connectWebSocket = (userIdToConnect) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      const wsUrl = `${API_BASE.replace('http', 'ws')}/ws/notifications/${userIdToConnect}`;
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        console.log('🔔 WebSocket connected for notifications');
        setIsConnected(true);
        clearTimeout(reconnectTimeoutRef.current);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const notification = JSON.parse(event.data);
          handleIncomingNotification(notification);
        } catch (error) {
          console.error('Failed to parse notification:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('🔔 WebSocket disconnected');
        setIsConnected(false);
        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          connectWebSocket(userIdToConnect);
        }, 3000);
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
    }
  };

  const handleIncomingNotification = (notification) => {
    // Add to notifications list
    setNotifications(prev => [notification, ...prev.slice(0, 49)]); // Keep max 50
    
    // Update unread count
    if (!notification.read) {
      setUnreadCount(prev => prev + 1);
    }

    // Show browser notification if permission granted
    if (Notification.permission === 'granted') {
      showBrowserNotification(notification);
    }

    // Store locally
    storeNotificationLocally(notification);

    // Trigger haptic feedback on mobile
    if ('vibrate' in navigator) {
      navigator.vibrate(100);
    }
  };

  const showBrowserNotification = (notification) => {
    if (document.hidden && Notification.permission === 'granted') {
      const notif = new Notification(notification.title, {
        body: notification.message,
        icon: '/favicon.svg',
        tag: notification.id,
        data: notification.data,
        requireInteraction: notification.priority === 'urgent'
      });

      notif.onclick = () => {
        window.focus();
        markAsRead(notification.id);
        notif.close();
        
        // Navigate to action URL if available
        if (notification.action_url) {
          window.location.href = notification.action_url;
        }
      };

      // Auto close after 5 seconds for non-urgent notifications
      if (notification.priority !== 'urgent') {
        setTimeout(() => notif.close(), 5000);
      }
    }
  };

  const initializePushNotifications = async () => {
    try {
      // Register service worker
      const registration = await navigator.serviceWorker.register('/sw.js');
      
      // Request permission
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') {
        console.log('Push notifications permission denied');
        return;
      }

      // Get push subscription (mock FCM for now)
      const mockToken = `mock_token_${Date.now()}`;
      setDeviceToken(mockToken);
      
      // Register device token with server
      if (userId) {
        await registerDeviceToken(userId, mockToken);
      }

    } catch (error) {
      console.error('Push notification setup failed:', error);
    }
  };

  const registerDeviceToken = async (userIdToRegister, token) => {
    try {
      await fetch(`${API_BASE}/api/v1/notifications/device-token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userIdToRegister,
          device_token: token,
          platform: 'web'
        })
      });
      console.log('📱 Device token registered');
    } catch (error) {
      console.error('Failed to register device token:', error);
    }
  };

  const fetchNotifications = async (userIdToFetch) => {
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/notifications?user_id=${userIdToFetch}&limit=20`
      );
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications || []);
        setUnreadCount(data.unread_count || 0);
      }
    } catch (error) {
      console.error('Failed to fetch notifications:', error);
    }
  };

  const markAsRead = async (notificationId) => {
    try {
      await fetch(`${API_BASE}/api/v1/notifications/${notificationId}/read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId })
      });

      // Update local state
      setNotifications(prev => 
        prev.map(n => n.id === notificationId ? { ...n, read: true } : n)
      );
      setUnreadCount(prev => Math.max(0, prev - 1));
      
    } catch (error) {
      console.error('Failed to mark notification as read:', error);
    }
  };

  const markAllAsRead = async () => {
    try {
      await fetch(`${API_BASE}/api/v1/notifications/read-all`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId })
      });

      // Update local state
      setNotifications(prev => prev.map(n => ({ ...n, read: true })));
      setUnreadCount(0);
      
    } catch (error) {
      console.error('Failed to mark all notifications as read:', error);
    }
  };

  const updatePreferences = async (newPreferences) => {
    try {
      await fetch(`${API_BASE}/api/v1/notifications/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          preferences: newPreferences
        })
      });

      setPreferences(newPreferences);
      localStorage.setItem('notification_preferences', JSON.stringify(newPreferences));
      
    } catch (error) {
      console.error('Failed to update preferences:', error);
    }
  };

  const requestPermission = async () => {
    if (!('Notification' in window)) {
      return 'not-supported';
    }

    if (Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      return permission;
    }

    return Notification.permission;
  };

  const storeNotificationLocally = (notification) => {
    try {
      const stored = JSON.parse(localStorage.getItem('notifications') || '[]');
      const updated = [notification, ...stored.slice(0, 19)]; // Keep max 20
      localStorage.setItem('notifications', JSON.stringify(updated));
    } catch (error) {
      console.error('Failed to store notification locally:', error);
    }
  };

  const loadStoredNotifications = () => {
    try {
      const stored = JSON.parse(localStorage.getItem('notifications') || '[]');
      if (stored.length > 0) {
        setNotifications(stored);
        setUnreadCount(stored.filter(n => !n.read).length);
      }

      const storedPreferences = JSON.parse(localStorage.getItem('notification_preferences') || '{}');
      if (Object.keys(storedPreferences).length > 0) {
        setPreferences({ ...preferences, ...storedPreferences });
      }
    } catch (error) {
      console.error('Failed to load stored notifications:', error);
    }
  };

  const cleanup = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
  };

  const value = {
    notifications,
    unreadCount,
    isConnected,
    userId,
    pushSupported,
    deviceToken,
    preferences,
    markAsRead,
    markAllAsRead,
    updatePreferences,
    requestPermission,
    refreshNotifications: () => fetchNotifications(userId)
  };

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  );
};

export default NotificationProvider;
