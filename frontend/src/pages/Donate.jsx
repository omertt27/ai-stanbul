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
            <div className="header-logo chat-title logo-istanbul">
              <span className="logo-text">
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
        </div>
      </header>

      {/* Scrollable Content */}
      <div className="pt-8 pb-20">
        {/* Hero Section */}
        <div className="pb-12">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className="mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-yellow-500 to-orange-600 mb-6">
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
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
            <a 
              href="https://twitter.com/intent/tweet?text=Check%20out%20AIstanbul%20-%20the%20best%20AI%20travel%20guide%20for%20Istanbul!%20🇹🇷✨&url=https://aistanbul.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className={`block p-6 rounded-lg border-2 border-dashed transition-all duration-300 hover:scale-105 ${
                darkMode 
                  ? 'border-gray-600 hover:border-blue-500 hover:bg-blue-900/20' 
                  : 'border-gray-300 hover:border-blue-400 hover:bg-blue-50'
              }`}
            >
              <div className="flex items-center mb-2">
                <svg className="w-5 h-5 mr-2 text-blue-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                </svg>
                <h3 className={`font-semibold text-lg transition-colors duration-300 ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  Share on Social Media
                </h3>
              </div>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Share AIstanbul with fellow travelers on Twitter, Instagram, or Facebook
              </p>
            </a>
            
            <a 
              href="mailto:feedback@aistanbul.com?subject=AIstanbul%20Feedback&body=Hi!%20I%20have%20some%20feedback%20about%20AIstanbul..." 
              className={`block p-6 rounded-lg border-2 border-dashed transition-all duration-300 hover:scale-105 ${
                darkMode 
                  ? 'border-gray-600 hover:border-green-500 hover:bg-green-900/20' 
                  : 'border-gray-300 hover:border-green-400 hover:bg-green-50'
              }`}
            >
              <div className="flex items-center mb-2">
                <svg className="w-5 h-5 mr-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                </svg>
                <h3 className={`font-semibold text-lg transition-colors duration-300 ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  Send Feedback
                </h3>
              </div>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Help us improve with your valuable feedback and suggestions
              </p>
            </a>
            
            <a 
              href="https://github.com/omertt27/ai-stanbul" 
              target="_blank" 
              rel="noopener noreferrer"
              className={`block p-6 rounded-lg border-2 border-dashed transition-all duration-300 hover:scale-105 ${
                darkMode 
                  ? 'border-gray-600 hover:border-purple-500 hover:bg-purple-900/20' 
                  : 'border-gray-300 hover:border-purple-400 hover:bg-purple-50'
              }`}
            >
              <div className="flex items-center mb-2">
                <svg className="w-5 h-5 mr-2 text-purple-500" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0C5.374 0 0 5.373 0 12 0 17.302 3.438 21.8 8.207 23.387c.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.300 24 12c0-6.627-5.373-12-12-12z"/>
                </svg>
                <h3 className={`font-semibold text-lg transition-colors duration-300 ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}>
                  Contribute on GitHub
                </h3>
              </div>
              <p className={`text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                Help improve AIstanbul by contributing code or reporting issues
              </p>
            </a>
          </div>
        </div>

        {/* Transparency Section */}
        <div className={`rounded-2xl p-8 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-white shadow-lg border border-gray-100'
        }`}>
          <h3 className={`text-xl font-bold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Our Transparency Promise
          </h3>
          <p className={`transition-colors duration-300 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            All donations go directly to project expenses: server costs, API fees, and development resources. 
            We're committed to keeping AIstanbul free and accessible to everyone exploring beautiful Istanbul!
          </p>
        </div>
      </div>
      </div>
    </div>
  );
};

export default Donate;
