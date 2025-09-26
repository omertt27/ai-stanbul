// AI Istanbul - Service Worker for PWA functionality
const CACHE_NAME = 'ai-istanbul-v1.0.0';
const OFFLINE_URL = '/offline.html';

// Resources to cache for offline functionality
const CACHE_URLS = [
  '/',
  '/offline.html',
  '/favicon.svg',
  '/apple-touch-icon.svg',
  '/manifest.json',
  // Add main app resources
  '/src/main.jsx',
  '/src/App.css'
];

// Install event - cache resources
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Install');
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Caching app shell');
      return cache.addAll(CACHE_URLS);
    }).catch((error) => {
      console.log('[ServiceWorker] Cache failed:', error);
    })
  );
  // Force the waiting service worker to become the active service worker
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activate');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  // Ensure the service worker takes control immediately
  return self.clients.claim();
});

// Fetch event - serve cached content when offline
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    // Handle navigation requests
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.open(CACHE_NAME).then((cache) => {
          return cache.match(OFFLINE_URL);
        });
      })
    );
  } else {
    // Handle other requests
    event.respondWith(
      caches.match(event.request).then((response) => {
        return response || fetch(event.request);
      }).catch(() => {
        // Return a generic offline response for failed requests
        if (event.request.destination === 'image') {
          return new Response('', {
            status: 200,
            statusText: 'OK',
            headers: { 'Content-Type': 'image/svg+xml' }
          });
        }
      })
    );
  }
});

// Handle background sync for better offline experience
self.addEventListener('sync', (event) => {
  if (event.tag === 'chat-sync') {
    console.log('[ServiceWorker] Background sync: chat-sync');
    // Handle chat message synchronization when online
    event.waitUntil(syncChatMessages());
  }
});

// Handle push notifications (for future implementation)
self.addEventListener('push', (event) => {
  console.log('[ServiceWorker] Push received');
  const options = {
    body: event.data ? event.data.text() : 'New travel update available!',
    icon: '/favicon.svg',
    badge: '/favicon.svg',
    vibrate: [200, 100, 200],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'Explore',
        icon: '/favicon.svg'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/favicon.svg'
      }
    ]
  };
  
  event.waitUntil(
    self.registration.showNotification('AI Istanbul', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  console.log('[ServiceWorker] Notification click received');
  
  event.notification.close();
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Sync chat messages function (placeholder for future implementation)
async function syncChatMessages() {
  try {
    // This would sync any pending chat messages when back online
    console.log('[ServiceWorker] Syncing chat messages...');
    // Implementation would depend on your backend API
  } catch (error) {
    console.log('[ServiceWorker] Sync failed:', error);
  }
}
