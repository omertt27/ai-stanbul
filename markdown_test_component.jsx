import React from 'react';
import ReactMarkdown from 'react-markdown';

const testMessage = `Istanbul is a treasure trove of historical and cultural landmarks, and I'd be delighted to share with you its top attractions.

**The Hagia Sophia: A Marvel of Byzantine Architecture**

The Hagia Sophia is one of Istanbul's most iconic landmarks, a former Byzantine church, Ottoman mosque, and now a museum.

**The Topkapi Palace: A Glimpse into Ottoman Splendor**

The Topkapi Palace was the primary residence of the Ottoman sultans for nearly 400 years.

Here are the top landmarks:

1. Hagia Sophia - A former Byzantine church, Ottoman mosque, and now a museum
📍 Location: Sultanahmet Square, Fatih

2. Topkapi Palace - The primary residence of the Ottoman sultans for over 400 years
📍 Location: Sultanahmet, Fatih`;

function MarkdownTest() {
  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif', maxWidth: '600px' }}>
      <h1>React Markdown Test - AI Istanbul Chat Fix</h1>
      
      <h3>Original Raw Text:</h3>
      <pre style={{ background: '#f0f0f0', padding: '10px', fontSize: '12px', overflow: 'auto' }}>
        {testMessage}
      </pre>
      
      <h3>ReactMarkdown Rendering (With paragraph spacing):</h3>
      <div style={{ border: '2px solid #007acc', padding: '15px', background: '#f9f9f9' }}>
        <ReactMarkdown
          components={{
            // Custom paragraph with better spacing
            p: ({ node, ...props }) => (
              <p {...props} style={{ marginBottom: '16px', marginTop: '0' }} />
            ),
            // Bold text
            strong: ({ node, ...props }) => (
              <strong {...props} style={{ fontWeight: 'bold', color: '#333' }} />
            )
          }}
        >
          {testMessage}
        </ReactMarkdown>
      </div>
      
      <div style={{ marginTop: '20px', padding: '10px', background: '#d4edda', border: '1px solid #c3e6cb', borderRadius: '5px' }}>
        <strong>✅ Expected Results:</strong>
        <ul>
          <li>Bold headers should appear in <strong>bold text</strong></li>
          <li>Each paragraph should be visually separated</li>
          <li>List items should be properly formatted</li>
          <li>No more "wall of text" appearance</li>
        </ul>
      </div>
    </div>
  );
}

export default MarkdownTest;
