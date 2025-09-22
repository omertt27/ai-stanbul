import React from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';

const Donate = () => {
  const { darkMode } = useTheme();

  const benefitItem = (icon, title, description) => (
    <div className="flex items-start space-x-4">
      {icon && (
        <div className="flex-shrink-0">
          <div className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600">
            <span className="text-white text-lg">{icon}</span>
          </div>
        </div>
      )}
      <div>
        <h3 className={`font-semibold text-lg mb-2 transition-colors duration-300 ${
          darkMode ? 'text-white' : 'text-gray-800'
        }`}>
          {title}
        </h3>
        <p className={`transition-colors duration-300 ${
          darkMode ? 'text-gray-300' : 'text-gray-600'
        }`}>
          {description}
        </p>
      </div>
    </div>
  );

  return (
    <div className={`min-h-screen w-full transition-colors duration-300 ${
      darkMode ? 'bg-gray-900' : 'bg-gradient-to-br from-yellow-50 via-orange-50 to-red-50'
    }`}>
      {/* Header with Centered Logo */}
      <header className={`w-full px-4 py-6 border-b transition-colors duration-200 ${
        darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white/80 backdrop-blur-sm border-gray-200'
      }`}>
        <div className="max-w-6xl mx-auto flex justify-center">
          <Link to="/" style={{textDecoration: 'none'}}>
            <div 
              style={{
                cursor: 'pointer',
                pointerEvents: 'auto',
                transition: 'transform 0.2s ease, opacity 0.2s ease',
                display: 'flex',
                alignItems: 'center',
              }}
            >
              <span style={{
                fontSize: '2.6rem',
                fontWeight: 700,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                background: 'linear-gradient(90deg, #e5e7eb 0%, #8b5cf6 50%, #6366f1 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                textShadow: '0 2px 10px rgba(139, 92, 246, 0.3)',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
              }}>
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
        </div>
      </header>

      {/* Scrollable Content */}
      <div className="h-screen overflow-y-auto pt-8 pb-20">
        {/* Hero Section */}
        <div className="pb-12">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className="mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-yellow-500 to-orange-600 mb-6">
              <svg className="w-10 h-10 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
              </svg>
            </div>
            <h1 className={`text-5xl font-bold mb-6 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Support <span className="bg-gradient-to-r from-yellow-600 to-orange-600 bg-clip-text text-transparent font-black">A/</span><span className="bg-gradient-to-r from-yellow-600 to-orange-600 bg-clip-text text-transparent font-normal">STANBUL</span>
            </h1>
            <p className={`text-xl leading-relaxed transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Help us keep Istanbul's best AI travel guide free and amazing for everyone
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 pb-20">
        {/* Buy Me a Coffee Section */}
        <div className={`rounded-2xl p-8 mb-12 text-center transition-colors duration-300 ${
          darkMode 
            ? 'bg-gradient-to-r from-yellow-900/30 to-orange-900/30 border border-yellow-800/50' 
            : 'bg-gradient-to-r from-yellow-100 to-orange-100 border border-yellow-200'
        }`}>
          <h2 className={`text-3xl font-bold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-yellow-300' : 'text-yellow-800'
          }`}>
            Buy Me a Coffee
          </h2>
          <p className={`text-lg mb-6 transition-colors duration-300 ${
            darkMode ? 'text-yellow-100' : 'text-yellow-700'
          }`}>
            The easiest way to show your appreciation and keep AIstanbul running
          </p>
          <a 
            href="https://www.buymeacoffee.com/aistanbul" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-block px-8 py-4 bg-yellow-500 hover:bg-yellow-600 text-white font-bold text-lg rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-lg"
          >
            ☕ Buy Me a Coffee
          </a>
        </div>

        {/* Benefits Section */}
        <div className={`rounded-2xl p-8 mb-12 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-white shadow-lg border border-gray-100'
        }`}>
          <h2 className={`text-3xl font-bold text-center mb-8 transition-colors duration-300 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            How Your Support Helps
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-6">
              {benefitItem('', 'Keep It Free Forever', 'Your donations ensure AIstanbul remains completely free for all travelers exploring Istanbul.')}
              {benefitItem('', 'Server & API Costs', 'Cover hosting, Google Maps API, and OpenAI costs for lightning-fast, accurate responses.')}
              {benefitItem('', 'Continuous Updates', 'Fund regular database updates with the latest restaurant, museum, and attraction information.')}
            </div>
            <div className="space-y-6">
              {benefitItem('', 'New Features', 'Enable development of exciting features like event recommendations and real-time transport info.')}
              {benefitItem('', 'Multilingual Support', 'Help us expand to serve visitors from around the world in their native languages.')}
              {benefitItem('', 'Better Experience', 'Improve mobile experience, add offline features, and enhance user interface.')}
            </div>
          </div>
        </div>

        {/* Alternative Support Methods */}
        <div className={`rounded-2xl p-8 mb-12 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-white shadow-lg border border-gray-100'
        }`}>
          <h2 className={`text-3xl font-bold text-center mb-8 transition-colors duration-300 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Other Ways to Help
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className={`p-6 rounded-lg border-2 border-dashed transition-colors duration-300 ${
              darkMode 
                ? 'border-gray-600 hover:border-blue-500' 
                : 'border-gray-300 hover:border-blue-400'
            }`}>
              <h3 className={`font-semibold text-lg mb-2 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Share with Friends
              </h3>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                <span>Share AIstanbul with fellow travelers</span>
              </p>
            </div>
            
            <div className={`p-6 rounded-lg border-2 border-dashed transition-colors duration-300 ${
              darkMode 
                ? 'border-gray-600 hover:border-green-500' 
                : 'border-gray-300 hover:border-green-400'
            }`}>
              <h3 className={`font-semibold text-lg mb-2 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Feedback & Reviews
              </h3>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Help us improve with your valuable feedback
              </p>
            </div>
            
            <div className={`p-6 rounded-lg border-2 border-dashed transition-colors duration-300 ${
              darkMode 
                ? 'border-gray-600 hover:border-purple-500' 
                : 'border-gray-300 hover:border-purple-400'
            }`}>
              <h3 className={`font-semibold text-lg mb-2 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Contribute Code
              </h3>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Help improve AIstanbul on GitHub
              </p>
            </div>
          </div>
        </div>

        {/* Transparency Section */}

      </div>
      </div>
    </div>
  );
};

export default Donate;
