/**
 * Offline Support Manager
 * =======================
 * Manages offline functionality for chat messages
 * Uses IndexedDB for robust offline storage
 * 
 * Features:
 * - Message caching in IndexedDB
 * - Offline indicator
 * - Sync queue for pending messages
 * - Automatic retry when back online
 */

const DB_NAME = 'ai_istanbul_offline';
const DB_VERSION = 1;
const MESSAGES_STORE = 'messages';
const PENDING_STORE = 'pending_messages';

class OfflineManager {
  constructor() {
    this.db = null;
    this.isOnline = navigator.onLine;
    this.syncQueue = [];
    this.listeners = [];
    
    this.init();
    this.setupEventListeners();
  }

  /**
   * Initialize IndexedDB
   */
  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error('IndexedDB failed to open');
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        console.log('✅ IndexedDB initialized for offline support');
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;

        // Create messages store
        if (!db.objectStoreNames.contains(MESSAGES_STORE)) {
          const messagesStore = db.createObjectStore(MESSAGES_STORE, { 
            keyPath: 'id', 
            autoIncrement: true 
          });
          messagesStore.createIndex('sessionId', 'sessionId', { unique: false });
          messagesStore.createIndex('timestamp', 'timestamp', { unique: false });
        }

        // Create pending messages store
        if (!db.objectStoreNames.contains(PENDING_STORE)) {
          db.createObjectStore(PENDING_STORE, { 
            keyPath: 'id', 
            autoIncrement: true 
          });
        }
      };
    });
  }

  /**
   * Setup online/offline event listeners
   */
  setupEventListeners() {
    window.addEventListener('online', () => {
      this.isOnline = true;
      console.log('🌐 Back online');
      this.notifyListeners('online');
      this.processSyncQueue();
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      console.log('📴 Gone offline');
      this.notifyListeners('offline');
    });
  }

  /**
   * Add status change listener
   */
  addListener(callback) {
    this.listeners.push(callback);
  }

  /**
   * Remove status change listener
   */
  removeListener(callback) {
    this.listeners = this.listeners.filter(l => l !== callback);
  }

  /**
   * Notify all listeners of status change
   */
  notifyListeners(status) {
    this.listeners.forEach(callback => callback(status, this.isOnline));
  }

  /**
   * Save message to IndexedDB
   */
  async saveMessage(message, sessionId) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([MESSAGES_STORE], 'readwrite');
      const store = transaction.objectStore(MESSAGES_STORE);
      
      const messageWithSession = {
        ...message,
        sessionId,
        savedAt: new Date().toISOString()
      };

      const request = store.add(messageWithSession);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Get all messages for a session
   */
  async getMessages(sessionId) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([MESSAGES_STORE], 'readonly');
      const store = transaction.objectStore(MESSAGES_STORE);
      const index = store.index('sessionId');
      const request = index.getAll(sessionId);

      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Add message to sync queue (for pending sends when offline)
   */
  async addToSyncQueue(message) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([PENDING_STORE], 'readwrite');
      const store = transaction.objectStore(PENDING_STORE);
      
      const pendingMessage = {
        ...message,
        addedAt: new Date().toISOString(),
        retryCount: 0
      };

      const request = store.add(pendingMessage);

      request.onsuccess = () => {
        this.syncQueue.push(pendingMessage);
        resolve(request.result);
      };
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Process sync queue when back online
   */
  async processSyncQueue() {
    if (!this.isOnline || this.syncQueue.length === 0) return;

    console.log(`📤 Processing ${this.syncQueue.length} pending messages`);

    const pending = [...this.syncQueue];
    this.syncQueue = [];

    for (const message of pending) {
      try {
        // Attempt to send the message
        // This would call your API to send the message
        console.log('Sending pending message:', message);
        
        // Remove from pending store after successful send
        await this.removeFromPendingStore(message.id);
      } catch (error) {
        console.error('Failed to sync message:', error);
        // Re-add to queue if failed
        this.syncQueue.push(message);
      }
    }
  }

  /**
   * Remove message from pending store
   */
  async removeFromPendingStore(id) {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([PENDING_STORE], 'readwrite');
      const store = transaction.objectStore(PENDING_STORE);
      const request = store.delete(id);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Clear all cached messages
   */
  async clearCache() {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db.transaction([MESSAGES_STORE, PENDING_STORE], 'readwrite');
      
      const messagesStore = transaction.objectStore(MESSAGES_STORE);
      const pendingStore = transaction.objectStore(PENDING_STORE);
      
      messagesStore.clear();
      pendingStore.clear();

      transaction.oncomplete = () => {
        console.log('🗑️ Offline cache cleared');
        resolve();
      };
      transaction.onerror = () => reject(transaction.error);
    });
  }

  /**
   * Get offline status
   */
  getStatus() {
    return {
      isOnline: this.isOnline,
      pendingMessages: this.syncQueue.length,
      dbReady: this.db !== null
    };
  }
}

// Export singleton instance
export const offlineManager = new OfflineManager();

// Export convenience functions
export const saveMessageOffline = (message, sessionId) => 
  offlineManager.saveMessage(message, sessionId);

export const getOfflineMessages = (sessionId) => 
  offlineManager.getMessages(sessionId);

export const addToSyncQueue = (message) => 
  offlineManager.addToSyncQueue(message);

export const getOfflineStatus = () => 
  offlineManager.getStatus();

export const addOfflineListener = (callback) => 
  offlineManager.addListener(callback);

export const removeOfflineListener = (callback) => 
  offlineManager.removeListener(callback);

export default offlineManager;
