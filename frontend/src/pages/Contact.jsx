import React from 'react';

function Contact({ darkMode }) {
  return (
    <div className={`static-page ${darkMode ? 'dark' : ''}`}>
      <h1>Contact Us</h1>
      
      <p>
        Have questions, suggestions, or feedback about AI Istanbul Guide? We'd love to hear from you! 
        Here are the best ways to get in touch.
      </p>

      <h2>📧 Get in Touch</h2>
      
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', 
        gap: '2rem',
        marginTop: '2rem' 
      }}>
        <div style={{ 
          padding: '1.5rem', 
          borderRadius: '0.75rem', 
          background: darkMode ? '#1f2937' : '#f9fafb',
          border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb'
        }}>
          <h3>🐛 Bug Reports & Issues</h3>
          <p>Found a bug or technical issue? Report it on our GitHub repository where our development team can track and fix it quickly.</p>
          <a 
            href="https://github.com/aiistanbul/issues" 
            target="_blank" 
            rel="noopener noreferrer"
            className="donate-button" 
            style={{ display: 'inline-block', marginTop: '1rem' }}
          >
            Report on GitHub
          </a>
        </div>

        <div style={{ 
          padding: '1.5rem', 
          borderRadius: '0.75rem', 
          background: darkMode ? '#1f2937' : '#f9fafb',
          border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb'
        }}>
          <h3>💡 Feature Requests</h3>
          <p>Have an idea for a new feature or improvement? Share your suggestions and help us make AI Istanbul Guide even better!</p>
          <a 
            href="https://github.com/aiistanbul/discussions" 
            target="_blank" 
            rel="noopener noreferrer"
            className="donate-button" 
            style={{ display: 'inline-block', marginTop: '1rem' }}
          >
            Share Ideas
          </a>
        </div>

        <div style={{ 
          padding: '1.5rem', 
          borderRadius: '0.75rem', 
          background: darkMode ? '#1f2937' : '#f9fafb',
          border: darkMode ? '1px solid #374151' : '1px solid #e5e7eb'
        }}>
          <h3>📨 General Inquiries</h3>
          <p>For general questions, partnerships, or other inquiries, reach out to us via email.</p>
          <a 
            href="mailto:hello@aiistanbul.guide"
            className="donate-button" 
            style={{ display: 'inline-block', marginTop: '1rem' }}
          >
            Send Email
          </a>
        </div>
      </div>

      <h2>🤝 How to Contribute</h2>
      
      <h3>For Developers</h3>
      <ul>
        <li><strong>GitHub Repository:</strong> Fork our repo and submit pull requests</li>
        <li><strong>Code Reviews:</strong> Help review code changes and improvements</li>
        <li><strong>Documentation:</strong> Improve our README, API docs, and guides</li>
        <li><strong>Testing:</strong> Help test new features and report issues</li>
      </ul>

      <h3>For Non-Developers</h3>
      <ul>
        <li><strong>User Testing:</strong> Try new features and provide feedback</li>
        <li><strong>Content:</strong> Help improve our database of Istanbul information</li>
        <li><strong>Translation:</strong> Assist with multilingual content</li>
        <li><strong>Feedback:</strong> Share your experience and suggestions</li>
      </ul>

      <h2>📱 Social & Community</h2>
      <ul>
        <li><strong>GitHub:</strong> <a href="https://github.com/aiistanbul" target="_blank" rel="noopener noreferrer">@aiistanbul</a></li>
        <li><strong>Twitter:</strong> <a href="https://twitter.com/aiistanbul" target="_blank" rel="noopener noreferrer">@aiistanbul</a></li>
        <li><strong>Discord:</strong> Join our community server for real-time chat</li>
        <li><strong>Reddit:</strong> r/aiistanbul for discussions and tips</li>
      </ul>

      <h2>🏢 Business Inquiries</h2>
      <p>
        For business partnerships, API access, or commercial licensing inquiries:
      </p>
      <ul>
        <li><strong>Email:</strong> business@aiistanbul.guide</li>
        <li><strong>Response Time:</strong> 2-3 business days</li>
      </ul>

      <h2>📍 About Response Times</h2>
      <div style={{ 
        padding: '1rem', 
        borderRadius: '0.5rem', 
        background: darkMode ? '#065f46' : '#d1fae5',
        color: darkMode ? '#a7f3d0' : '#047857',
        margin: '1.5rem 0'
      }}>
        <p><strong>Expected Response Times:</strong></p>
        <ul style={{ color: 'inherit', marginTop: '0.5rem' }}>
          <li>GitHub Issues: 1-2 days</li>
          <li>General Email: 3-5 days</li>
          <li>Business Inquiries: 2-3 business days</li>
          <li>Critical Bugs: Same day (if possible)</li>
        </ul>
      </div>

      <h2>🌍 Community Guidelines</h2>
      <p>We're committed to maintaining a welcoming, inclusive community. Please:</p>
      <ul>
        <li>Be respectful and constructive in all communications</li>
        <li>Provide clear, detailed information when reporting issues</li>
        <li>Search existing issues before creating new ones</li>
        <li>Follow our code of conduct in all interactions</li>
      </ul>

      <div style={{ 
        marginTop: '3rem', 
        padding: '1.5rem', 
        borderRadius: '0.75rem', 
        background: darkMode ? '#1e1b4b' : '#ede9fe',
        border: darkMode ? '1px solid #4c1d95' : '1px solid #c4b5fd'
      }}>
        <h2>🚀 Join Our Mission</h2>
        <p>
          AI Istanbul Guide is more than just a chatbot - it's a community-driven project 
          to make Istanbul more accessible and enjoyable for everyone. Whether you're a 
          developer, designer, content creator, or just someone who loves Istanbul, 
          there's a place for you in our community.
        </p>
        <p style={{ marginTop: '1rem', fontWeight: '600' }}>
          Together, let's help people discover the magic of Istanbul! 🇹🇷
        </p>
      </div>
    </div>
  );
}

export default Contact;
