/**
 * Push Notifications Manager
 * ==========================
 * Manages browser push notifications for chat responses
 * Notifies users when LLM responds while app is backgrounded
 * 
 * Features:
 * - Request notification permission
 * - Show notifications for new messages
 * - Handle notification clicks
 * - Respect user preferences
 */

const NOTIFICATION_PERMISSION_KEY = 'kam_notification_permission';
const NOTIFICATION_ENABLED_KEY = 'kam_notifications_enabled';

class PushNotificationManager {
  constructor() {
    this.supported = 'Notification' in window;
    this.permission = this.supported ? Notification.permission : 'denied';
    this._enabled = null; // Lazy load
    this._initialized = false;
  }

  /**
   * Lazy initialization
   */
  _ensureInitialized() {
    if (this._initialized) return;
    this._enabled = this.getEnabledPreference();
    this._initialized = true;
  }

  /**
   * Check if notifications are supported
   */
  isSupported() {
    return this.supported;
  }

  /**
   * Get current permission status
   */
  getPermission() {
    return this.permission;
  }

  /**
   * Check if notifications are enabled by user
   */
  isEnabled() {
    this._ensureInitialized();
    return this._enabled && this.permission === 'granted';
  }

  /**
   * Get user's notification preference from storage
   */
  getEnabledPreference() {
    try {
      const stored = localStorage.getItem(NOTIFICATION_ENABLED_KEY);
      return stored === null ? true : stored === 'true'; // Default to true
    } catch (e) {
      return true;
    }
  }

  /**
   * Set user's notification preference
   */
  setEnabledPreference(enabled) {
    this._ensureInitialized();
    this._enabled = enabled;
    try {
      localStorage.setItem(NOTIFICATION_ENABLED_KEY, enabled.toString());
    } catch (e) {
      console.warn('Failed to save notification preference:', e);
    }
  }

  /**
   * Request notification permission from user
   */
  async requestPermission() {
    if (!this.supported) {
      console.warn('Notifications not supported in this browser');
      return 'denied';
    }

    if (this.permission === 'granted') {
      return 'granted';
    }

    try {
      this.permission = await Notification.requestPermission();
      
      if (this.permission === 'granted') {
        console.log('✅ Notification permission granted');
        this.setEnabledPreference(true);
      } else {
        console.log('❌ Notification permission denied');
      }

      return this.permission;
    } catch (error) {
      console.error('Error requesting notification permission:', error);
      return 'denied';
    }
  }

  /**
   * Show a notification
   */
  async showNotification(title, options = {}) {
    if (!this.isEnabled()) {
      console.log('Notifications disabled or not permitted');
      return null;
    }

    // Check if page is focused
    if (document.hasFocus()) {
      console.log('Page is focused, skipping notification');
      return null;
    }

    const defaultOptions = {
      icon: '/favicon.ico',
      badge: '/favicon.ico',
      vibrate: [200, 100, 200],
      requireInteraction: false,
      ...options
    };

    try {
      const notification = new Notification(title, defaultOptions);

      // Auto-close after 10 seconds
      setTimeout(() => {
        notification.close();
      }, 10000);

      // Handle notification click
      notification.onclick = (event) => {
        event.preventDefault();
        window.focus();
        notification.close();
        
        // Navigate to chat if provided
        if (options.url) {
          window.location.href = options.url;
        }
      };

      return notification;
    } catch (error) {
      console.error('Error showing notification:', error);
      return null;
    }
  }

  /**
   * Show notification for new chat message
   */
  async notifyNewMessage(message, sender = 'KAM') {
    const title = `New message from ${sender}`;
    const options = {
      body: message.substring(0, 100) + (message.length > 100 ? '...' : ''),
      tag: 'chat-message',
      renotify: true,
      url: '/#/chat'
    };

    return this.showNotification(title, options);
  }

  /**
   * Show notification for system message
   */
  async notifySystem(title, message) {
    const options = {
      body: message,
      tag: 'system',
      icon: '/favicon.ico'
    };

    return this.showNotification(title, options);
  }

  /**
   * Enable notifications
   */
  async enable() {
    if (!this.supported) {
      throw new Error('Notifications not supported');
    }

    if (this.permission !== 'granted') {
      const permission = await this.requestPermission();
      if (permission !== 'granted') {
        throw new Error('Notification permission denied');
      }
    }

    this.setEnabledPreference(true);
    return true;
  }

  /**
   * Disable notifications
   */
  disable() {
    this.setEnabledPreference(false);
  }

  /**
   * Get notification status
   */
  getStatus() {
    return {
      supported: this.supported,
      permission: this.permission,
      enabled: this.enabled,
      canNotify: this.isEnabled()
    };
  }
}

// Export singleton instance
export const pushNotifications = new PushNotificationManager();

// Export convenience functions
export const requestNotificationPermission = () => 
  pushNotifications.requestPermission();

export const notifyNewMessage = (message, sender) => 
  pushNotifications.notifyNewMessage(message, sender);

export const notifySystem = (title, message) => 
  pushNotifications.notifySystem(title, message);

export const enableNotifications = () => 
  pushNotifications.enable();

export const disableNotifications = () => 
  pushNotifications.disable();

export const getNotificationStatus = () => 
  pushNotifications.getStatus();

export default pushNotifications;
