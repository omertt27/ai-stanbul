import React from 'react';
import { Link } from 'react-router-dom';
import '../App.css';

const Donate = () => (
  <div className="static-page">
    {/* AI Istanbul Logo - Top Left */}
    <Link to="/" style={{textDecoration: 'none'}} className="fixed z-50">
      <div className="logo-istanbul logo-move-top-left">
        <span className="logo-text">
          A/<span style={{fontWeight: 400}}>STANBUL</span>
        </span>
      </div>
    </Link>
    
    <h1>Support AIstanbul</h1>
    <p>
      If you find AIstanbul helpful, consider supporting the project! Your support helps cover server costs and enables further development.
    </p>
    
    <h2>☕ Buy Me a Coffee</h2>
    <p>
      The easiest way to show your appreciation is through "Buy Me a Coffee" – a simple platform that lets you support creators with a small contribution.
    </p>
    
    <div style={{ margin: '2rem 0', textAlign: 'center' }}>
      <a 
        href="https://www.buymeacoffee.com/aistanbul" 
        target="_blank" 
        rel="noopener noreferrer"
        className="buy-me-coffee-btn"
        style={{
          display: 'inline-block',
          background: 'linear-gradient(135deg, #FFDD00, #FF813F)',
          color: '#000',
          padding: '1rem 2rem',
          borderRadius: '0.75rem',
          textDecoration: 'none',
          fontWeight: '700',
          fontSize: '1.2rem',
          transition: 'all 0.3s ease',
          boxShadow: '0 6px 20px rgba(255, 221, 0, 0.4)',
          border: 'none',
          cursor: 'pointer',
          transform: 'translateY(0px)'
        }}
        onMouseOver={(e) => {
          e.target.style.transform = 'translateY(-3px)';
          e.target.style.boxShadow = '0 8px 25px rgba(255, 221, 0, 0.6)';
        }}
        onMouseOut={(e) => {
          e.target.style.transform = 'translateY(0px)';
          e.target.style.boxShadow = '0 6px 20px rgba(255, 221, 0, 0.4)';
        }}
      >
        ☕ Buy me a coffee
      </a>
    </div>

    <h2>💝 Multiple Ways to Support</h2>
    <div style={{ 
      display: 'grid', 
      gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
      gap: '1rem', 
      margin: '2rem 0' 
    }}>
      <div style={{
        background: 'rgba(129, 140, 248, 0.1)',
        padding: '1.5rem',
        borderRadius: '0.75rem',
        border: '1px solid rgba(129, 140, 248, 0.2)'
      }}>
        <h3 style={{ color: '#818cf8', margin: '0 0 1rem 0', fontSize: '1.2rem' }}>☕ Single Coffee</h3>
        <p style={{ margin: '0 0 1rem 0', fontSize: '0.95rem' }}>$3 - Show your appreciation</p>
        <a href="https://www.buymeacoffee.com/aistanbul" target="_blank" rel="noopener noreferrer"
           style={{ color: '#818cf8', textDecoration: 'underline', fontSize: '0.9rem' }}>
          Buy me a coffee →
        </a>
      </div>
      
      <div style={{
        background: 'rgba(255, 221, 0, 0.1)',
        padding: '1.5rem',
        borderRadius: '0.75rem',
        border: '1px solid rgba(255, 221, 0, 0.2)'
      }}>
        <h3 style={{ color: '#FFDD00', margin: '0 0 1rem 0', fontSize: '1.2rem' }}>🍰 Monthly Support</h3>
        <p style={{ margin: '0 0 1rem 0', fontSize: '0.95rem' }}>$10/month - Ongoing development</p>
        <a href="https://www.buymeacoffee.com/aistanbul/membership" target="_blank" rel="noopener noreferrer"
           style={{ color: '#FFDD00', textDecoration: 'underline', fontSize: '0.9rem' }}>
          Become a supporter →
        </a>
      </div>
    </div>

    <h2>Why Your Support Matters</h2>
    <ul>
      <li>Keep the service completely free for all users</li>
      <li>Cover API costs for Google Maps and other premium services</li>
      <li>Support ongoing development and new features</li>
      <li>Maintain and update our database of Istanbul attractions</li>
      <li>Improve user experience and add new languages</li>
    </ul>
    
    <h2>How Your Donation Helps</h2>
    <ul>
      <li><strong>Server Infrastructure:</strong> Keeping our AI chatbot running 24/7</li>
      <li><strong>Data Accuracy:</strong> Regular updates to restaurant, museum, and attraction information</li>
      <li><strong>Feature Development:</strong> Adding new capabilities like event recommendations and transport info</li>
      <li><strong>Multilingual Support:</strong> Expanding to serve visitors from around the world</li>
    </ul>

    <h2>Other Ways to Help</h2>
    <p>
      Can't donate right now? Here are other ways you can support the project:
    </p>
    <ul>
      <li>Share AIstanbul with friends and fellow travelers</li>
      <li>Provide feedback to help improve our recommendations</li>
      <li>Report bugs or suggest new features</li>
      <li>Follow us on social media for updates</li>
      <li>Leave a review or rating if you found the service helpful</li>
    </ul>

    <h2>Transparency Promise</h2>
    <p>
      We believe in complete transparency. All donations are used exclusively for project-related expenses including server costs, API fees, and development resources. We're committed to keeping AIstanbul free and accessible to everyone.
    </p>
    
    <div style={{ 
      marginTop: '3rem', 
      padding: '1.5rem', 
      background: 'rgba(129, 140, 248, 0.1)', 
      borderRadius: '0.5rem',
      border: '1px solid rgba(129, 140, 248, 0.2)'
    }}>
      <p style={{ marginBottom: '0.5rem', fontWeight: '600' }}>
        🙏 Thank you for considering supporting AIstanbul!
      </p>
      <p style={{ margin: 0, fontStyle: 'italic', opacity: 0.9 }}>
        Together, we can make exploring Istanbul easier and more enjoyable for everyone! 🇹🇷
      </p>
    </div>
  </div>
);

export default Donate;

