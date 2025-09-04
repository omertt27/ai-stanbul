import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App';
import Chatbot from './Chatbot';
import SimpleChatbot from './SimpleChatbot';
import TestComponent from './TestComponent';
import About from './pages/About';
import Sources from './pages/Sources';
import Donate from './pages/Donate';
import FAQ from './pages/FAQ';
import Contact from './pages/Contact';
import BlogList from './pages/BlogList';
import BlogPost from './pages/BlogPost';
import NewBlogPost from './pages/NewBlogPost';
import EnhancedDemo from './pages/EnhancedDemo';
import NavBar from './components/NavBar';
import Footer from './components/Footer';

const AppRouter = () => {
  const [isLightMode, setIsLightMode] = useState(false);

  // Toggle light mode on both body and chat containers
  useEffect(() => {
    const body = document.body;
    const chatContainers = document.querySelectorAll('.chat-container');
    
    if (isLightMode) {
      body.classList.add('light');
      chatContainers.forEach(el => el.classList.add('light-chat'));
    } else {
      body.classList.remove('light');
      chatContainers.forEach(el => el.classList.remove('light-chat'));
    }
  }, [isLightMode]);

  const toggleTheme = () => {
    setIsLightMode(!isLightMode);
  };

  const buttonStyle = {
    position: 'fixed',
    top: '2.5rem',  // Same as NavBar
    right: '1.5rem',
    zIndex: 1000,
    background: isLightMode ? 'rgba(255, 255, 255, 0.9)' : 'transparent',
    border: isLightMode ? '1px solid #e2e8f0' : 'none',
    borderRadius: '50%',
    padding: 0,
    cursor: 'pointer',
    transition: 'all 0.3s ease',
    color: isLightMode ? '#475569' : '#e5e7eb',
    boxShadow: isLightMode ? '0 2px 8px rgba(99, 102, 241, 0.15)' : 'none',
    width: '32px',
    height: '32px',
    minWidth: '32px',
    minHeight: '32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  };

  return (
    <Router>
      <button
        onClick={toggleTheme}
        style={buttonStyle}
        title={`Switch to ${isLightMode ? 'dark' : 'light'} mode`}
      >
        {isLightMode ? (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="5"></circle>
            <line x1="12" y1="1" x2="12" y2="3"></line>
            <line x1="12" y1="21" x2="12" y2="23"></line>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
            <line x1="1" y1="12" x2="3" y2="12"></line>
            <line x1="21" y1="12" x2="23" y2="12"></line>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
          </svg>
        )}
      </button>

      <div className="chatbot-outline"></div>
      <NavBar />
      <Footer />
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/test" element={<TestComponent />} />
        <Route path="/simple" element={<SimpleChatbot />} />
        <Route path="/chatbot" element={<Chatbot />} />
        <Route path="/demo" element={<EnhancedDemo />} />
        <Route path="/about" element={<About />} />
        <Route path="/sources" element={<Sources />} />
        <Route path="/donate" element={<Donate />} />
        <Route path="/faq" element={<FAQ />} />
        <Route path="/contact" element={<Contact />} />
        <Route path="/blog" element={<BlogList />} />
        <Route path="/blog/new" element={<NewBlogPost />} />
        <Route path="/blog/:id" element={<BlogPost />} />
      </Routes>
    </Router>
  );
};

export default AppRouter;
