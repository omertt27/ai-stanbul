/**
 * Enhanced Service Worker
 * Integrates map tile caching, periodic sync, and improved offline handling
 * 
 * @version 2.0.0
 * @features Map tiles, Periodic sync, Background sync, Push notifications
 */

const CACHE_VERSION = 'ai-istanbul-v2.0.0';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const MAP_TILES_CACHE = 'map-tiles-v1';

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/favicon.svg',
  '/apple-touch-icon.svg'
];

// Map tile URL pattern
const MAP_TILE_PATTERN = /tile\.openstreetmap\.org\/\d+\/\d+\/\d+\.png/;

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('🔧 Service Worker: Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(cache => {
        console.log('📦 Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('✅ Service Worker: Installed');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('❌ Installation failed:', error);
      })
  );
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
  console.log('🔄 Service Worker: Activating...');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(name => {
              return name.startsWith('ai-istanbul-') && 
                     name !== STATIC_CACHE && 
                     name !== DYNAMIC_CACHE &&
                     name !== MAP_TILES_CACHE;
            })
            .map(name => {
              console.log('🗑️ Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => {
        console.log('✅ Service Worker: Activated');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Handle map tiles separately
  if (MAP_TILE_PATTERN.test(request.url)) {
    event.respondWith(handleMapTileRequest(request));
    return;
  }

  // Handle API requests (network-first)
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleAPIRequest(request));
    return;
  }

  // Handle static assets (cache-first)
  event.respondWith(handleStaticRequest(request));
});

/**
 * Handle map tile requests (cache-first with network fallback)
 */
async function handleMapTileRequest(request) {
  try {
    // Try cache first
    const cache = await caches.open(MAP_TILES_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }

    // Fetch from network
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('❌ Map tile request failed:', error);
    // Return a transparent placeholder tile
    return new Response(
      new Blob([''], { type: 'image/png' }),
      { status: 200, statusText: 'OK' }
    );
  }
}

/**
 * Handle API requests (network-first with cache fallback)
 */
async function handleAPIRequest(request) {
  try {
    const networkResponse = await fetch(request);
    
    // Cache successful GET requests
    if (request.method === 'GET' && networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    // Try cache fallback for GET requests
    if (request.method === 'GET') {
      const cache = await caches.open(DYNAMIC_CACHE);
      const cachedResponse = await cache.match(request);
      
      if (cachedResponse) {
        console.log('📦 Serving cached API response');
        return cachedResponse;
      }
    }

    // Return offline response
    return new Response(
      JSON.stringify({
        error: 'offline',
        message: 'You are currently offline. This request has been queued and will be sent when you reconnect.',
        offline: true
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

/**
 * Handle static asset requests (cache-first with network fallback)
 */
async function handleStaticRequest(request) {
  try {
    // Try cache first
    const cache = await caches.open(STATIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }

    // Fetch from network
    const networkResponse = await fetch(request);
    
    // Cache successful responses
    if (networkResponse.ok) {
      const dynamicCache = await caches.open(DYNAMIC_CACHE);
      dynamicCache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      const cache = await caches.open(STATIC_CACHE);
      return cache.match('/offline.html');
    }

    // Return error response
    return new Response('Offline', { status: 503, statusText: 'Service Unavailable' });
  }
}

// Background Sync - sync queued chat messages
self.addEventListener('sync', (event) => {
  console.log('🔄 Background sync triggered:', event.tag);
  
  if (event.tag === 'chat-sync') {
    event.waitUntil(syncChatMessages());
  } else if (event.tag === 'data-sync') {
    event.waitUntil(syncOfflineData());
  }
});

/**
 * Sync queued chat messages
 */
async function syncChatMessages() {
  try {
    console.log('💬 Syncing chat messages...');
    
    // Get pending messages from IndexedDB
    const db = await openDatabase();
    const messages = await getQueuedMessages(db);
    
    if (messages.length === 0) {
      console.log('✅ No messages to sync');
      return;
    }

    // Send each message
    for (const message of messages) {
      try {
        const response = await fetch('/api/ai/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(message)
        });

        if (response.ok) {
          await removeQueuedMessage(db, message.id);
          console.log('✅ Message synced:', message.id);
        }
      } catch (error) {
        console.error('❌ Failed to sync message:', message.id, error);
      }
    }
    
    console.log('✅ Chat sync complete');
  } catch (error) {
    console.error('❌ Chat sync failed:', error);
    throw error;
  }
}

/**
 * Sync offline data (restaurants, attractions, etc.)
 */
async function syncOfflineData() {
  try {
    console.log('🔄 Syncing offline data...');
    
    const endpoints = [
      { url: '/api/restaurants', store: 'restaurants' },
      { url: '/api/attractions', store: 'attractions' },
      { url: '/api/pois', store: 'pois' }
    ];

    for (const { url, store } of endpoints) {
      try {
        const response = await fetch(url);
        if (response.ok) {
          const data = await response.json();
          await saveToIndexedDB(store, data);
          console.log(`✅ Synced ${store}`);
        }
      } catch (error) {
        console.error(`❌ Failed to sync ${store}:`, error);
      }
    }
    
    console.log('✅ Offline data sync complete');
  } catch (error) {
    console.error('❌ Data sync failed:', error);
  }
}

// Periodic Background Sync - update cache periodically
self.addEventListener('periodicsync', (event) => {
  console.log('⏰ Periodic sync triggered:', event.tag);
  
  if (event.tag === 'update-cache') {
    event.waitUntil(updateCache());
  }
});

/**
 * Update cache periodically (daily)
 */
async function updateCache() {
  try {
    console.log('🔄 Updating cache...');
    
    // Sync offline data
    await syncOfflineData();
    
    // Update static assets if needed
    const cache = await caches.open(STATIC_CACHE);
    await cache.addAll(STATIC_ASSETS);
    
    console.log('✅ Cache updated');
  } catch (error) {
    console.error('❌ Cache update failed:', error);
  }
}

// Push Notifications
self.addEventListener('push', (event) => {
  console.log('📬 Push notification received');
  
  const data = event.data ? event.data.json() : {};
  const title = data.title || 'Istanbul AI';
  const options = {
    body: data.body || 'You have a new notification',
    icon: '/icon-192.png',
    badge: '/badge-72.png',
    data: data.url || '/',
    tag: data.tag || 'default',
    requireInteraction: false
  };

  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

// Notification click
self.addEventListener('notificationclick', (event) => {
  console.log('🔔 Notification clicked');
  
  event.notification.close();
  
  event.waitUntil(
    clients.openWindow(event.notification.data || '/')
  );
});

// Helper: Open IndexedDB
function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('istanbul-ai', 1);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Helper: Get queued messages
function getQueuedMessages(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['queuedMessages'], 'readonly');
    const store = transaction.objectStore('queuedMessages');
    const request = store.getAll();
    request.onsuccess = () => resolve(request.result || []);
    request.onerror = () => reject(request.error);
  });
}

// Helper: Remove queued message
function removeQueuedMessage(db, messageId) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['queuedMessages'], 'readwrite');
    const store = transaction.objectStore('queuedMessages');
    const request = store.delete(messageId);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}

// Helper: Save to IndexedDB
function saveToIndexedDB(storeName, data) {
  return new Promise((resolve, reject) => {
    openDatabase()
      .then(db => {
        const transaction = db.transaction([storeName], 'readwrite');
        const store = transaction.objectStore(storeName);
        
        const items = Array.isArray(data) ? data : [data];
        items.forEach(item => store.put(item));
        
        transaction.oncomplete = () => resolve();
        transaction.onerror = () => reject(transaction.error);
      })
      .catch(reject);
  });
}

console.log('✅ Enhanced Service Worker loaded (v2.0.0)');
