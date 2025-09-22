import React from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import './styles/arabic.css' // Arabic language support
import './styles/anti-copy.css' // Anti-copy protection styles
import './i18n' // Initialize i18n
import AppRouter from './AppRouter.jsx'
import { ThemeProvider } from './contexts/ThemeContext.jsx'
import './utils/websiteProtection.js' // Initialize website protection

console.log('Starting React app...')

try {
  const container = document.getElementById('root')
  if (!container) {
    throw new Error('Root element not found')
  }
  
  const root = createRoot(container)
  root.render(
    <React.StrictMode>
      <ThemeProvider>
        <AppRouter />
      </ThemeProvider>
    </React.StrictMode>,
  )
  
  console.log('React app mounted successfully')
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
