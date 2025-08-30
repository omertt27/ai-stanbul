import React from 'react';

const Donate = () => (
  <div className="page-content">
    <h1>Support / Donate</h1>
    <p>
      If you find AIstanbul helpful, consider supporting the project! Your donation helps cover server costs and enables further development.
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
    <ul>
      <li><b>Server costs:</b> Keeping our API running 24/7</li>
      <li><b>Data accuracy:</b> Regular updates to restaurant, museum, and attraction information</li>
      <li><b>Feature development:</b> Adding new capabilities like event recommendations and transport info</li>
      <li><b>Multilingual support:</b> Expanding to serve visitors from around the world</li>
    </ul>
    <h2>Ways to Support</h2>
    <p><b>Crypto:</b> <code>0x1234...abcd</code> (Ethereum)</p>
    <p><b>Patreon:</b> <a href="https://patreon.com/ai-stanbul" target="_blank" rel="noopener noreferrer">patreon.com/ai-stanbul</a></p>
    <p>Thank you for your support!</p>
  </div>
);

export default Donate;
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
