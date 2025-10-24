import React from 'react'
import { createRoot } from 'react-dom/client'
import { Analytics } from '@vercel/analytics/react'
import { HelmetProvider } from 'react-helmet-async'
import './index.css'
import './styles/arabic.css' // Arabic language support
// import './styles/anti-copy.css' // Anti-copy protection styles - DISABLED
import './i18n' // Initialize i18n
import AppRouter from './AppRouter.jsx'
import { ThemeProvider } from './contexts/ThemeContext.jsx'
import { BlogProvider } from './contexts/BlogContext.jsx'
import { LocationProvider } from './contexts/LocationContext.jsx'
import { NotificationProvider } from './contexts/NotificationContext.jsx'
// import './utils/websiteProtection.js' // Initialize website protection - DISABLED
import offlineEnhancementManager from './services/offlineEnhancementManager.js'

console.log('Starting React app...')

// Initialize offline enhancements
async function initializeOfflineFeatures() {
  try {
    const result = await offlineEnhancementManager.initialize({
      autoSyncOnReconnect: true,
      enablePeriodicSync: true,
      enableOfflineIntents: true,
      cacheMapTilesOnInstall: false // User must opt-in via settings
    });
    
    console.log('✅ Offline enhancements initialized:', result);
  } catch (error) {
    console.error('⚠️ Failed to initialize offline enhancements:', error);
    // Non-critical failure, app continues to work
  }
}

// Initialize offline features (non-blocking)
initializeOfflineFeatures();

// Ensure page starts at top
window.scrollTo(0, 0);
document.documentElement.scrollTop = 0;
document.body.scrollTop = 0;

try {
  const container = document.getElementById('root')
  if (!container) {
    throw new Error('Root element not found')
  }
  
  const root = createRoot(container)
  root.render(
    <React.StrictMode>
      <HelmetProvider>
        <ThemeProvider>
          <BlogProvider>
            <LocationProvider>
              <NotificationProvider>
                <AppRouter />
                <Analytics />
              </NotificationProvider>
            </LocationProvider>
          </BlogProvider>
        </ThemeProvider>
      </HelmetProvider>
    </React.StrictMode>,
  )
  
  console.log('React app mounted successfully')
  
  // Additional scroll reset after React renders
  setTimeout(() => {
    window.scrollTo(0, 0);
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
  }, 100);
} catch (error) {
  console.error('Failed to mount React app:', error)
  document.getElementById('root').innerHTML = `
    <div style="padding: 20px; color: red; font-family: Arial;">
      <h2>Error Loading App</h2>
      <p>Error: ${error.message}</p>
      <p>Check the browser console for more details.</p>
    </div>
  `
}
