import { useState, useEffect } from 'react';
import Chatbot from './Chatbot';
import About from './pages/About';
import Source from './pages/Source';
import Donate from './pages/Donate';
import Privacy from './pages/Privacy';
import FAQ from './pages/FAQ';
import Tips from './pages/Tips';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('chat');
  const [darkMode, setDarkMode] = useState(false);

  // Apply dark mode to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'about':
        return <About darkMode={darkMode} />;
      case 'source':
        return <Source darkMode={darkMode} />;
      case 'donate':
        return <Donate darkMode={darkMode} />;
      case 'privacy':
        return <Privacy darkMode={darkMode} />;
      case 'faq':
        return <FAQ darkMode={darkMode} />;
      case 'tips':
        return <Tips darkMode={darkMode} />;
      default:
        return <Chatbot onDarkModeToggle={toggleDarkMode} />;
    }
  };

  return (
    <div className="w-full h-screen">
      {/* Navigation Bar */}
      <nav className={`nav-container ${darkMode ? 'dark' : ''}`}>
        <div className="nav-content">
          <a 
            href="#" 
            onClick={() => setCurrentPage('chat')}
            className={`nav-logo ${darkMode ? 'dark' : ''}`}
          >
            AI Istanbul Guide
          </a>
          
          <ul className="nav-links">
            <li>
              <a 
                href="#" 
                onClick={() => setCurrentPage('chat')}
                className={`nav-link ${currentPage === 'chat' ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
              >
                Chat
              </a>
            </li>
            <li>
              <a 
                href="#" 
                onClick={() => setCurrentPage('tips')}
                className={`nav-link ${currentPage === 'tips' ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
              >
                Tips
              </a>
            </li>
            <li>
              <a 
                href="#" 
                onClick={() => setCurrentPage('faq')}
                className={`nav-link ${currentPage === 'faq' ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
              >
                FAQ
              </a>
            </li>
            <li>
              <a 
                href="#" 
                onClick={() => setCurrentPage('about')}
                className={`nav-link ${currentPage === 'about' ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
              >
                About
              </a>
            </li>
            <li>
              <a 
                href="#" 
                onClick={() => setCurrentPage('source')}
                className={`nav-link ${currentPage === 'source' ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
              >
                Source
              </a>
            </li>
            <li>
              <a 
                href="#" 
                onClick={() => setCurrentPage('donate')}
                className={`nav-link ${currentPage === 'donate' ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
              >
                Donate
              </a>
            </li>
            <li>
              <a 
                href="#" 
                onClick={() => setCurrentPage('privacy')}
                className={`nav-link ${currentPage === 'privacy' ? 'active' : ''} ${darkMode ? 'dark' : ''}`}
              >
                Privacy
              </a>
            </li>
          </ul>

          {/* Dark mode toggle */}
          <button 
            onClick={toggleDarkMode}
            className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors duration-200"
          >
            {darkMode ? (
              <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 0 1 8.646 3.646 9.003 9.003 0 1 0 12 21a9.003 9.003 0 0 0 8.354-5.646z" />
              </svg>
            )}
          </button>
        </div>
      </nav>

      {/* Page Content */}
      <div className="page-content">
        {renderPage()}
      </div>
    </div>
  );
}

export default App;
