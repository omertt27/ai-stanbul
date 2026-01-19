/**
 * Enhanced Service Worker
 * Integrates map tile caching, periodic sync, and improved offline handling
 * 
 * @version 2.5.1
 * @features Map tiles, Periodic sync, Background sync, Push notifications, Cache busting for JS/CSS
 * @updated 2025-01-19 - Fix HEAD request caching errors, improve request method filtering
 */

const CACHE_VERSION = 'ai-istanbul-v2.5.1';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const MAP_TILES_CACHE = 'map-tiles-v2';

// Assets to cache on install (only HTML pages and icons, NOT JS/CSS)
const STATIC_ASSETS = [
  '/offline.html',
  '/manifest.json',
  '/favicon.svg',
  '/apple-touch-icon.svg'
];

// Map tile URL pattern
const MAP_TILE_PATTERN = /tile\.openstreetmap\.org\/\d+\/\d+\/\d+\.png/;

// Build assets pattern (hashed Vite files like index-CfNLh5xt.js)
const BUILD_ASSET_PATTERN = /\.(js|css|json)$/;
const HASHED_ASSET_PATTERN = /[-.][\da-f]{8,}\.(js|css)$/i;

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
        // Skip waiting to activate immediately
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('❌ Installation failed:', error);
      })
  );
});

// Message event - handle commands from clients
self.addEventListener('message', (event) => {
  console.log('📨 Service Worker received message:', event.data);
  
  if (event.data && event.data.type === 'SKIP_WAITING') {
    console.log('⏩ Skipping waiting...');
    self.skipWaiting();
  }
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
  console.log('🔄 Service Worker: Activating...');
  
  event.waitUntil(
    Promise.all([
      // 1. Delete old version caches
      caches.keys().then(cacheNames => {
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
      }),
      
      // 2. Clean potentially corrupted cached responses from dynamic cache
      caches.open(DYNAMIC_CACHE).then(cache => {
        return cache.keys().then(requests => {
          return Promise.all(
            requests.map(request => {
              // Remove cached JS/CSS to force fresh fetch
              if (BUILD_ASSET_PATTERN.test(request.url) || HASHED_ASSET_PATTERN.test(request.url)) {
                console.log('🧹 Clearing cached build asset:', request.url);
                return cache.delete(request);
              }
            })
          );
        });
      })
    ])
    .then(() => {
      console.log('✅ Service Worker: Activated and cleaned');
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  
  // Only handle http and https requests
  if (!request.url.startsWith('http://') && !request.url.startsWith('https://')) {
    return;
  }
  
  const url = new URL(request.url);

  // Bypass service worker for external analytics and fonts
  const externalDomains = [
    'google-analytics.com',
    'googletagmanager.com', 
    'analytics.google.com',
    'fonts.googleapis.com',
    'fonts.gstatic.com'
  ];
  
  if (externalDomains.some(domain => url.hostname.includes(domain))) {
    // Let browser handle these directly - don't intercept
    return;
  }

  // Handle map tiles separately
  if (MAP_TILE_PATTERN.test(request.url)) {
    event.respondWith(handleMapTileRequest(request));
    return;
  }

  // CRITICAL FIX: Let API requests pass through directly to network
  // Don't intercept them - the app has its own error handling
  if (url.pathname.startsWith('/api/')) {
    // Just pass through, don't use respondWith
    return;
  }

  // CRITICAL FIX: Handle build assets (JS/CSS) with network-first strategy
  // This prevents serving stale cached JavaScript that causes "Unexpected token '<'" errors
  if (BUILD_ASSET_PATTERN.test(url.pathname) || HASHED_ASSET_PATTERN.test(url.pathname)) {
    event.respondWith(handleBuildAssetRequest(request));
    return;
  }

  // Handle HTML/navigation requests (network-first)
  if (request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html')) {
    event.respondWith(handleNavigationRequest(request));
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
      try {
        await cache.put(request, networkResponse.clone());
      } catch (cacheError) {
        console.warn('⚠️ Failed to cache map tile:', cacheError);
      }
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
 * IMPORTANT: Let actual network errors pass through, don't intercept them
 */
async function handleAPIRequest(request) {
  try {
    // Try network request with explicit timeout
    const networkResponse = await fetch(request, {
      cache: 'no-cache' // Don't use browser cache, always hit network
    });
    
    // Cache successful GET requests
    if (request.method === 'GET' && networkResponse.ok) {
      try {
        const cache = await caches.open(DYNAMIC_CACHE);
        await cache.put(request, networkResponse.clone());
      } catch (cacheError) {
        console.warn('⚠️ Failed to cache API response:', cacheError);
      }
    }
    
    return networkResponse;
  } catch (error) {
    // Only use cache for GET requests when truly offline
    if (request.method === 'GET' && !navigator.onLine) {
      const cache = await caches.open(DYNAMIC_CACHE);
      const cachedResponse = await cache.match(request);
      
      if (cachedResponse) {
        console.log('📦 Serving cached API response (offline)');
        return cachedResponse;
      }
    }

    // For all other cases, let the error propagate to the app
    // The app's error handling will deal with it properly
    console.log('⚠️ API request failed, passing error to app:', error.message);
    throw error;
  }
}

/**
 * Handle build asset requests (JS/CSS) - NETWORK FIRST to prevent stale cache
 * This is CRITICAL to fix "Unexpected token '<'" errors
 */
async function handleBuildAssetRequest(request) {
  // Only handle GET requests
  if (request.method !== 'GET') {
    return fetch(request);
  }
  
  try {
    // ALWAYS try network first for JS/CSS to get latest version
    const networkResponse = await fetch(request, {
      cache: 'no-cache' // Force revalidation
    });
    
    if (networkResponse.ok) {
      // Only cache if it's actually a valid JS/CSS file (not HTML error page)
      const contentType = networkResponse.headers.get('content-type') || '';
      const isValidAsset = contentType.includes('javascript') || 
                          contentType.includes('css') || 
                          contentType.includes('json');
      
      if (isValidAsset) {
        try {
          const cache = await caches.open(DYNAMIC_CACHE);
          await cache.put(request, networkResponse.clone());
        } catch (cacheError) {
          console.warn('⚠️ Failed to cache build asset:', cacheError);
        }
      }
    }
    
    return networkResponse;
  } catch (error) {
    // Only use cache as fallback if network fails
    console.log('⚠️ Network failed for build asset, trying cache...');
    const cache = await caches.open(DYNAMIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      // Verify cached response is valid
      const contentType = cachedResponse.headers.get('content-type') || '';
      if (contentType.includes('javascript') || contentType.includes('css')) {
        console.log('📦 Serving cached build asset (offline)');
        return cachedResponse;
      }
    }

    // If no valid cache, return error
    return new Response('Build asset not available offline', { 
      status: 503, 
      statusText: 'Service Unavailable' 
    });
  }
}

/**
 * Handle navigation requests (HTML pages) - NETWORK FIRST
 */
async function handleNavigationRequest(request) {
  try {
    // Don't cache HEAD requests (not supported by Cache API)
    if (request.method !== 'GET') {
      return fetch(request);
    }
    
    // ALWAYS try network first for HTML to get latest version
    const networkResponse = await fetch(request, {
      cache: 'no-cache' // Force revalidation
    });
    
    if (networkResponse.ok) {
      // Cache the fresh HTML (only for GET requests)
      try {
        const cache = await caches.open(STATIC_CACHE);
        await cache.put(request, networkResponse.clone());
      } catch (cacheError) {
        console.warn('⚠️ Failed to cache navigation:', cacheError);
      }
    }
    
    return networkResponse;
  } catch (error) {
    // Try cache fallback (only for GET requests)
    if (request.method === 'GET') {
      const cache = await caches.open(STATIC_CACHE);
      const cachedResponse = await cache.match(request);
      
      if (cachedResponse) {
        console.log('📦 Serving cached navigation (offline)');
        return cachedResponse;
      }
    }

    // Return offline page as last resort
    return cache.match('/offline.html');
  }
}

/**
 * Handle static asset requests (cache-first with network fallback)
 */
async function handleStaticRequest(request) {
  try {
    // Skip caching for HEAD requests
    if (request.method !== 'GET') {
      return fetch(request);
    }

    // Allow external resources (Google Analytics, Google Fonts, etc.) to bypass service worker
    const url = new URL(request.url);
    const externalDomains = [
      'google-analytics.com',
      'googletagmanager.com',
      'analytics.google.com',
      'fonts.googleapis.com',
      'fonts.gstatic.com'
    ];
    
    if (externalDomains.some(domain => url.hostname.includes(domain))) {
      // Let browser handle these requests directly without service worker interference
      return fetch(request, {
        mode: 'cors',
        credentials: 'omit'
      });
    }

    // Try cache first for same-origin requests
    const cache = await caches.open(STATIC_CACHE);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }

    // Fetch from network
    const networkResponse = await fetch(request);
    
    // Cache successful GET responses only (only for same-origin)
    if (networkResponse.ok && request.method === 'GET' && url.origin === self.location.origin) {
      try {
        const dynamicCache = await caches.open(DYNAMIC_CACHE);
        await dynamicCache.put(request, networkResponse.clone());
      } catch (cacheError) {
        console.warn('⚠️ Failed to cache static asset:', cacheError);
      }
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

console.log('✅ Enhanced Service Worker loaded (v2.5.0) - API requests bypass service worker');
