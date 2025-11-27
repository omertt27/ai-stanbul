// AI Istanbul - Service Worker for PWA functionality
const CACHE_NAME = 'ai-istanbul-v1.3.0'; // Updated version to force cache refresh and fix MIME types
const OFFLINE_URL = '/offline.html';

// Resources to cache for offline functionality
const CACHE_URLS = [
  '/',
  '/offline.html',
  '/favicon.svg',
  '/apple-touch-icon.svg',
  '/manifest.json'
  // Removed main.jsx from cache - let Vite handle it directly
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
  // Skip non-GET requests (HEAD, POST, etc.)
  if (event.request.method !== 'GET') {
    return;
  }

  const url = new URL(event.request.url);
  
  // Skip caching for JavaScript modules and assets to avoid MIME type issues
  if (url.pathname.endsWith('.jsx') || 
      url.pathname.endsWith('.js') || 
      url.pathname.endsWith('.css') ||
      url.pathname.startsWith('/src/') ||
      url.pathname.startsWith('/assets/')) {
    event.respondWith(fetch(event.request));
    return;
  }

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
        // Return a proper error response for other failed requests
        return new Response('Network Error', {
          status: 503,
          statusText: 'Service Unavailable',
          headers: { 'Content-Type': 'text/plain' }
        });
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
  } else if (event.tag === 'notification-sync') {
    console.log('[ServiceWorker] Background sync: notification-sync');
    // Handle notification synchronization when online
    event.waitUntil(syncNotifications());
  }
});

// Handle messages from the main thread
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'GET_USER_ID') {
    // Send back user ID from localStorage equivalent in service worker context
    event.ports[0].postMessage({ userId: null }); // Simplified for now
  }
});

// Handle push notifications
self.addEventListener('push', (event) => {
  console.log('[ServiceWorker] Push notification received');
  
  let notificationData = {
    title: 'Istanbul AI',
    body: 'You have a new notification',
    icon: '/favicon.svg',
    badge: '/favicon.svg',
    tag: 'default',
    data: {}
  };

  if (event.data) {
    try {
      const payload = event.data.json();
      notificationData = {
        title: payload.title || notificationData.title,
        body: payload.message || payload.body || notificationData.body,
        icon: payload.icon || notificationData.icon,
        badge: payload.badge || notificationData.badge,
        tag: payload.id || payload.tag || notificationData.tag,
        data: payload.data || payload,
        actions: getNotificationActions(payload.type),
        requireInteraction: payload.priority === 'urgent',
        silent: payload.priority === 'low'
      };
    } catch (error) {
      console.error('[ServiceWorker] Failed to parse push payload:', error);
      // Use text if JSON parsing fails
      notificationData.body = event.data.text();
    }
  }

  event.waitUntil(
    self.registration.showNotification(notificationData.title, {
      body: notificationData.body,
      icon: notificationData.icon,
      badge: notificationData.badge,
      tag: notificationData.tag,
      data: notificationData.data,
      actions: notificationData.actions,
      requireInteraction: notificationData.requireInteraction,
      silent: notificationData.silent,
      vibrate: notificationData.silent ? [] : [100, 50, 100]
    })
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  console.log('[ServiceWorker] Notification clicked:', event.action);
  
  event.notification.close();

  const notification = event.notification;
  const action = event.action;
  const data = notification.data || {};

  let urlToOpen = '/';

  // Handle action buttons
  if (action) {
    switch (action) {
      case 'view':
      case 'view_route':
      case 'view_attraction':
      case 'view_weather':
        urlToOpen = data.action_url || '/';
        break;
      case 'dismiss':
        return; // Just close, don't open anything
      case 'explore':
        urlToOpen = '/';
        break;
      default:
        urlToOpen = data.action_url || '/';
    }
  } else {
    // Regular click
    urlToOpen = data.action_url || '/';
  }

  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true })
      .then((clientList) => {
        // Check if there's already a window/tab open
        for (const client of clientList) {
          if ('focus' in client) {
            return client.focus().then(() => {
              // Send message to client about notification click
              client.postMessage({
                type: 'NOTIFICATION_CLICKED',
                notificationId: data.notification_id || data.id,
                data: data,
                action: action
              });
            });
          }
        }

        // If no existing window, open a new one
        if (self.clients.openWindow) {
          return self.clients.openWindow(urlToOpen);
        }
      })
  );
});

// Get notification actions based on type
function getNotificationActions(type) {
  const commonActions = [
    {
      action: 'view',
      title: 'View'
    },
    {
      action: 'dismiss',
      title: 'Dismiss'
    }
  ];

  switch (type) {
    case 'route_update':
      return [
        {
          action: 'view_route',
          title: 'View Route'
        },
        {
          action: 'dismiss',
          title: 'Dismiss'
        }
      ];
    case 'attraction_recommendation':
      return [
        {
          action: 'view_attraction',
          title: 'Learn More'
        },
        {
          action: 'dismiss',
          title: 'Later'
        }
      ];
    case 'weather_alert':
      return [
        {
          action: 'view_weather',
          title: 'Check Weather'
        },
        {
          action: 'dismiss',
          title: 'OK'
        }
      ];
    default:
      return commonActions;
  }
}

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

// Sync notifications when back online
async function syncNotifications() {
  try {
    console.log('[ServiceWorker] Syncing notifications...');
    // This would fetch missed notifications when back online
    // For now, just log the action
  } catch (error) {
    console.log('[ServiceWorker] Notification sync failed:', error);
  }
}
