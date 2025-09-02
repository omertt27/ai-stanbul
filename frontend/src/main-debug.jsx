import React from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'

// Simple test component instead of AppRouter
function TestApp() {
  return (
    <div style={{ padding: '20px' }}>
      <h1>🔧 Debugging React App</h1>
      <p>If you see this, basic React is working.</p>
      <div style={{ 
        backgroundColor: '#f0f0f0', 
        padding: '15px', 
        margin: '20px 0',
        borderRadius: '8px'
      }}>
        <h2>KAM Definition Test</h2>
        <p>
          <strong>Kam</strong>, in Turkish, Altaic, and Mongolian folk culture, is a shaman, a religious leader, wisdom person. Also referred to as "Gam" or Ham.
        </p>
        <p style={{ fontStyle: 'italic' }}>
          A religious leader believed to communicate with supernatural powers within communities.
        </p>
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <TestApp />
  </React.StrictMode>,
)
