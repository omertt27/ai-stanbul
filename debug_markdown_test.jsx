import React from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const testText = `Istanbul is a treasure trove of historical and cultural landmarks, and I'd be delighted to share with you its top attractions.

**The Hagia Sophia: A Marvel of Byzantine Architecture**

The Hagia Sophia is one of Istanbul's most iconic landmarks, a former Byzantine church, Ottoman mosque, and now a museum. This magnificent structure has stood the test of time.

**The Topkapi Palace: A Glimpse into Ottoman Splendor**

The Topkapi Palace was the primary residence of the Ottoman sultans for nearly 400 years. This sprawling complex is a testament to the grandeur of the Ottoman Empire.

**The Blue Mosque: A Masterpiece of Ottoman Architecture**

The Blue Mosque, also known as the Sultan Ahmed Mosque, is one of the most beautiful mosques in the world. Its six minarets and stunning blue tiles make it a breathtaking sight.`;

function App() {
  console.log('Test text with newlines:', JSON.stringify(testText));
  
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>ReactMarkdown Test</h1>
      
      <h3>Raw Text with \\n\\n:</h3>
      <pre style={{ background: '#f0f0f0', padding: '10px' }}>
        {JSON.stringify(testText)}
      </pre>
      
      <h3>ReactMarkdown Output:</h3>
      <div style={{ border: '1px solid #ccc', padding: '15px' }}>
        <ReactMarkdown
          components={{
            p: ({ node, ...props }) => (
              <p {...props} style={{ marginBottom: '0.75em', backgroundColor: '#f9f9f9', border: '1px solid #ddd', padding: '5px' }} />
            ),
          }}
        >
          {testText}
        </ReactMarkdown>
      </div>
      
      <h3>Manual P Tags:</h3>
      <div style={{ border: '1px solid #ccc', padding: '15px' }}>
        <p style={{ marginBottom: '0.75em' }}>Istanbul is a treasure trove of historical and cultural landmarks, and I'd be delighted to share with you its top attractions.</p>
        <p style={{ marginBottom: '0.75em' }}><strong>The Hagia Sophia: A Marvel of Byzantine Architecture</strong></p>
        <p style={{ marginBottom: '0.75em' }}>The Hagia Sophia is one of Istanbul's most iconic landmarks, a former Byzantine church, Ottoman mosque, and now a museum.</p>
        <p style={{ marginBottom: '0.75em' }}><strong>The Topkapi Palace: A Glimpse into Ottoman Splendor</strong></p>
        <p style={{ marginBottom: '0.75em' }}>The Topkapi Palace was the primary residence of the Ottoman sultans for nearly 400 years.</p>
      </div>
    </div>
  );
}

export default App;
