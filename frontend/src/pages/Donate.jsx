import React from 'react';

function Donate({ darkMode }) {
  return (
    <div className={`static-page ${darkMode ? 'dark' : ''}`}>
      <h1>Support AI Istanbul Guide</h1>
      
      <p>
        AI Istanbul Guide is a free, open-source project dedicated to helping visitors and locals 
        discover the best of Istanbul. Your support helps us maintain and improve this service.
      </p>

      <h2>Why Donate?</h2>
      <ul>
        <li>Keep the service free for everyone</li>
        <li>Cover API costs for Google Maps and other services</li>
        <li>Support ongoing development and new features</li>
        <li>Maintain and update our database of Istanbul attractions</li>
        <li>Improve user experience and add new languages</li>
      </ul>

      <h2>How Your Donation Helps</h2>
      <p>
        Every contribution, no matter the size, makes a difference. Your donations go directly towards:
      </p>
      <ul>
        <li><strong>Server costs:</strong> Keeping our API running 24/7</li>
        <li><strong>Data accuracy:</strong> Regular updates to restaurant, museum, and attraction information</li>
        <li><strong>Feature development:</strong> Adding new capabilities like event recommendations and transport info</li>
        <li><strong>Multilingual support:</strong> Expanding to serve visitors from around the world</li>
      </ul>

      <h2>Ways to Support</h2>
      <div style={{ marginTop: '2rem' }}>
        <a 
          href="https://ko-fi.com/aiistanbul" 
          target="_blank" 
          rel="noopener noreferrer"
          className="donate-button"
        >
          ☕ Buy us a coffee
        </a>
      </div>
      
      <div style={{ marginTop: '1rem' }}>
        <a 
          href="https://github.com/sponsors/aiistanbul" 
          target="_blank" 
          rel="noopener noreferrer"
          className="donate-button"
        >
          💖 Sponsor on GitHub
        </a>
      </div>

      <h2>Other Ways to Help</h2>
      <p>
        Can't donate right now? Here are other ways you can support the project:
      </p>
      <ul>
        <li>Share AI Istanbul Guide with friends and fellow travelers</li>
        <li>Rate and review places you visit to help improve our recommendations</li>
        <li>Report bugs or suggest new features on our GitHub page</li>
        <li>Contribute code if you're a developer</li>
        <li>Follow us on social media for updates</li>
      </ul>

      <h2>Transparency</h2>
      <p>
        We believe in full transparency. All donations are used exclusively for project-related expenses. 
        We regularly publish updates on how funds are used and what improvements they enable.
      </p>

      <p style={{ marginTop: '2rem', fontStyle: 'italic' }}>
        Thank you for considering a donation. Together, we can make exploring Istanbul easier 
        and more enjoyable for everyone! 🇹🇷
      </p>
    </div>
  );
}

export default Donate;
