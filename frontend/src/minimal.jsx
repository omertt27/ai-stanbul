import React from 'react'
import { createRoot } from 'react-dom/client'

function MinimalApp() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial' }}>
      <h1 style={{ color: 'green' }}>✅ React is Working!</h1>
      <p>If you see this, React is rendering correctly.</p>
      <button onClick={() => alert('Button clicked!')}>
        Test Button
      </button>
    </div>
  )
}

const container = document.getElementById('root')
const root = createRoot(container)
root.render(<MinimalApp />)
