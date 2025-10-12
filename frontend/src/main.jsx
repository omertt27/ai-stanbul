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
// import './utils/websiteProtection.js' // Initialize website protection - DISABLED

console.log('Starting React app...')

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
              <AppRouter />
              <Analytics />
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
