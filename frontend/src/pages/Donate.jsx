import React from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';

const Donate = () => {
  const { darkMode } = useTheme();

  const benefitItem = (icon, title, description) => (
    <div className="flex items-start space-x-4">
      <div>
        <h3 className="font-semibold text-lg mb-2 transition-colors duration-300 text-white">
          {title}
        </h3>
        <p className="transition-colors duration-300 text-gray-300">
          {description}
        </p>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen w-full transition-colors duration-300 bg-gray-900" style={{ marginTop: '0px', paddingLeft: '1.5rem', paddingRight: '1.5rem', paddingBottom: '1rem' }}>
      <div className="max-w-6xl mx-auto">

      {/* Scrollable Content */}
      <div className="pt-4 pb-24">
        {/* Hero Section */}
        <div className="pb-26">
        <div className="max-w-4xl mx-auto px-12 text-center">
          <div className="mb-8">
            <h1 className="text-5xl font-bold mb-6 pt-28 transition-colors duration-300 text-white">
              Support <span className="bg-gradient-to-r from-yellow-600 to-orange-600 bg-clip-text text-transparent font-black">A/</span><span className="bg-gradient-to-r from-yellow-600 to-orange-600 bg-clip-text text-transparent font-normal">STANBUL</span>
            </h1>
            <p className="text-xl leading-relaxed transition-colors duration-300 text-gray-300">
              Help us keep Istanbul's best AI travel guide free and amazing for everyone
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 pb-20">
        {/* Buy Me a Coffee Section */}
        <div className="rounded-2xl p-8 mb-12 text-center transition-colors duration-300 bg-gradient-to-r from-yellow-900/30 to-orange-900/30 border border-yellow-800/50">
          <h2 className="text-3xl font-bold mb-4 transition-colors duration-300 text-yellow-300">
            Buy Me a Coffee
          </h2>
          <p className="text-lg mb-6 transition-colors duration-300 text-yellow-100">
            The easiest way to show your appreciation and keep AIstanbul running
          </p>
          <a 
            href="https://www.buymeacoffee.com/aistanbul" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-block px-8 py-4 bg-yellow-500 hover:bg-yellow-600 text-white font-bold text-lg rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-lg"
          >
            Buy Me a Coffee
          </a>
        </div>

        {/* Benefits Section */}
        <div className="rounded-2xl p-8 mb-12 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <h2 className="text-3xl font-bold text-center mb-8 transition-colors duration-300 text-white">
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
        <div className="rounded-2xl p-8 mb-12 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <h2 className="text-3xl font-bold text-center mb-8 transition-colors duration-300 text-white">
            Other Ways to Help
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <a 
              href="https://twitter.com/intent/tweet?text=Check%20out%20AIstanbul%20-%20the%20best%20AI%20travel%20guide%20for%20Istanbul!%20🇹🇷✨&url=https://aistanbul.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block p-6 rounded-lg border-2 border-dashed transition-all duration-300 hover:scale-105 border-gray-600 hover:border-blue-500 hover:bg-blue-900/20"
            >
              <div className="flex items-center mb-2">
                <h3 className="font-semibold text-lg transition-colors duration-300 text-white">
                  Share on Social Media
                </h3>
              </div>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                Share AIstanbul with fellow travelers on Twitter, Instagram, or Facebook
              </p>
            </a>
            
            <a 
              href="mailto:feedback@aistanbul.com?subject=AIstanbul%20Feedback&body=Hi!%20I%20have%20some%20feedback%20about%20AIstanbul..." 
              className="block p-6 rounded-lg border-2 border-dashed transition-all duration-300 hover:scale-105 border-gray-600 hover:border-green-500 hover:bg-green-900/20"
            >
              <div className="flex items-center mb-2">
                <h3 className="font-semibold text-lg transition-colors duration-300 text-white">
                  Send Feedback
                </h3>
              </div>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                Help us improve with your valuable feedback and suggestions
              </p>
            </a>
            
            <a 
              href="https://github.com/omertt27/ai-stanbul" 
              target="_blank" 
              rel="noopener noreferrer"
              className="block p-6 rounded-lg border-2 border-dashed transition-all duration-300 hover:scale-105 border-gray-600 hover:border-purple-500 hover:bg-purple-900/20"
            >
              <div className="flex items-center mb-2">
                <h3 className="font-semibold text-lg transition-colors duration-300 text-white">
                  Contribute on GitHub
                </h3>
              </div>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                Help improve AIstanbul by contributing code or reporting issues
              </p>
            </a>
          </div>
        </div>


      </div>
      </div>
      </div>
    </div>
  );
};

export default Donate;
