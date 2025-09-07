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
      <div className="h-screen overflow-y-auto pt-8 pb-20">
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
