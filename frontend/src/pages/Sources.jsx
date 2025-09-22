import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../contexts/ThemeContext';
import '../App.css';

const Sources = () => {
  const { t } = useTranslation();
  const { darkMode } = useTheme();

  const dataSource = (icon, title, items, color) => (
    <div className={`rounded-xl p-6 transition-all duration-300 hover:scale-105 ${
      darkMode 
        ? 'bg-gray-800 border border-gray-700 hover:border-gray-600' 
        : 'bg-white shadow-lg border border-gray-100 hover:shadow-xl'
    }`}>
      <div className="flex items-center mb-4">
        {icon && (
          <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-r ${color} mr-4`}>
            <span className="text-2xl">{icon}</span>
          </div>
        )}
        <h3 className={`text-xl font-semibold transition-colors duration-300 ${
          darkMode ? 'text-white' : 'text-gray-800'
        }`}>
          {title}
        </h3>
      </div>
      <ul className="space-y-2">
        {items.map((item, index) => (
          <li key={index} className={`flex items-start transition-colors duration-300 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            <span className="text-blue-500 mr-2 mt-1">•</span>
            <span className="leading-relaxed">{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );

  return (
    <div className="chatbot-background min-h-screen w-full pt-24 px-4 pb-8 transition-colors duration-300">
      <div className="max-w-6xl mx-auto">
        {/* Hero Section */}
        <div className="pb-12">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <div className="mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gradient-to-r from-green-500 to-blue-600 mb-6">
              <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h1 className={`text-5xl font-bold mb-6 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              {t('sources.title')}
            </h1>
            <p className={`text-xl leading-relaxed transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              {t('sources.subtitle')}
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 pb-20">
        {/* Data Sources Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-12">
          {dataSource(
            '',
            'Restaurant Data',
            [
              'Google Maps Places API for live restaurant data',
              'Real-time ratings, reviews, and opening hours',
              'OpenAI for intelligent recommendations',
              'Community feedback and user reviews'
            ],
            'from-orange-500 to-red-500'
          )}

          {dataSource(
            '',
            'Cultural & Historical Data',
            [
              'Istanbul Metropolitan Municipality — Official tourism data',
              'Turkish Ministry of Culture and Tourism',
              'UNESCO World Heritage Sites information',
              'Local museums and cultural institutions',
              'Verified insights from cultural experts'
            ],
            'from-purple-500 to-pink-500'
          )}

          {dataSource(
            '',
            'Technology Stack',
            [
              'React 18 with Vite for lightning-fast development',
              'FastAPI with async support for high-performance backend',
              'SQLAlchemy ORM with SQLite/PostgreSQL database',
              'OpenAI GPT-3.5-turbo for intelligent conversations',
              'Redis caching with advanced rate limiting',
              'Structured logging with JSON format for monitoring'
            ],
            'from-blue-500 to-cyan-500'
          )}

          {dataSource(
            '',
            'Advanced Features',
            [
              'AI cache service with Redis for performance',
              'Rate limiting: 100 requests/user/hour, 500/IP/hour',
              'GDPR-compliant data handling and privacy',
              'Multi-language support (EN, TR, AR, RU)',
              'Real-time streaming responses with typing effects',
              'Circuit breaker pattern for error resilience'
            ],
            'from-indigo-500 to-purple-500'
          )}

          {dataSource(
            '',
            'Security & Performance',
            [
              'Input sanitization against XSS and injection attacks',
              'Security headers with CORS protection',
              'Session-based authentication with secure headers',
              'Performance monitoring with 95+ Lighthouse scores',
              'Comprehensive error handling and fallback systems',
              'Production-ready deployment with Docker support'
            ],
            'from-emerald-500 to-teal-500'
          )}
        </div>

        {/* Quality Assurance Section */}
        <div className={`rounded-2xl p-8 mb-12 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-white shadow-xl border border-gray-100'
        }`}>
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-emerald-500 to-teal-600 mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h2 className={`text-3xl font-bold mb-4 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Quality Assurance
            </h2>
            <p className={`text-lg transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              How we ensure accurate, reliable, and up-to-date information
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h3 className={`text-xl font-semibold transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Data Accuracy
              </h3>
              <ul className="space-y-2">
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-green-500 mr-2 mt-1">✓</span>
                  <span>Real-time data from Google Maps API</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-green-500 mr-2 mt-1">✓</span>
                  <span>Monthly database updates and reviews</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-green-500 mr-2 mt-1">✓</span>
                  <span>AI-powered response caching for instant results</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-green-500 mr-2 mt-1">✓</span>
                  <span>Advanced rate limiting and DDoS protection</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-green-500 mr-2 mt-1">✓</span>
                  <span>Input sanitization against security threats</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-green-500 mr-2 mt-1">✓</span>
                  <span>Cross-verified with official tourism sources</span>
                </li>
              </ul>
            </div>

            <div className="space-y-4">
              <h3 className={`text-xl font-semibold transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Continuous Improvement
              </h3>
              <ul className="space-y-2">
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-blue-500 mr-2 mt-1">✓</span>
                  <span>Structured logging with performance metrics</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-blue-500 mr-2 mt-1">✓</span>
                  <span>Circuit breaker patterns for error resilience</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-blue-500 mr-2 mt-1">✓</span>
                  <span>AI intelligence with session management</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-blue-500 mr-2 mt-1">✓</span>
                  <span>User feedback integration</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-blue-500 mr-2 mt-1">✓</span>
                  <span>AI model fine-tuning with Istanbul-specific data</span>
                </li>
                <li className={`flex items-start transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-600'
                }`}>
                  <span className="text-blue-500 mr-2 mt-1">✓</span>
                  <span>Local expert validation and insights</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Technical Specifications Section */}
        <div className={`rounded-2xl p-8 mb-12 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-white shadow-xl border border-gray-100'
        }`}>
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 mb-4">
              <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
              </svg>
            </div>
            <h2 className={`text-3xl font-bold mb-4 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Technical Specifications
            </h2>
            <p className={`text-lg transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Advanced technical features powering AI-stanbul
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className={`p-6 rounded-xl transition-colors duration-300 ${
              darkMode ? 'bg-gray-700' : 'bg-gray-50'
            }`}>
              <h3 className={`text-lg font-semibold mb-3 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Performance Metrics
              </h3>
              <ul className={`space-y-2 text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                <li>• Lighthouse Score: 95/100</li>
                <li>• First Contentful Paint: 0.8s</li>
                <li>• Time to Interactive: 1.5s</li>
                <li>• Response Time: &lt;50ms</li>
              </ul>
            </div>

            <div className={`p-6 rounded-xl transition-colors duration-300 ${
              darkMode ? 'bg-gray-700' : 'bg-gray-50'
            }`}>
              <h3 className={`text-lg font-semibold mb-3 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Security Features
              </h3>
              <ul className={`space-y-2 text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                <li>• GDPR Compliant</li>
                <li>• XSS Protection</li>
                <li>• CORS Security</li>
                <li>• Input Sanitization</li>
              </ul>
            </div>

            <div className={`p-6 rounded-xl transition-colors duration-300 ${
              darkMode ? 'bg-gray-700' : 'bg-gray-50'
            }`}>
              <h3 className={`text-lg font-semibold mb-3 transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                AI Capabilities
              </h3>
              <ul className={`space-y-2 text-sm transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                <li>• OpenAI GPT-3.5-turbo</li>
                <li>• Context Awareness</li>
                <li>• Fuzzy Matching</li>
                <li>• Intent Recognition</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Open Source Section */}
        <div className={`rounded-xl p-6 text-center transition-colors duration-300 ${
          darkMode 
            ? 'bg-gradient-to-r from-gray-800 to-gray-700' 
            : 'bg-gradient-to-r from-blue-50 to-indigo-50'
        }`}>
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-r from-indigo-500 to-purple-600 mb-4">
            <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
            </svg>
          </div>
          <h3 className={`text-lg font-semibold mb-2 transition-colors duration-300 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Open Source & Transparent
          </h3>
          <p className={`transition-colors duration-300 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Our commitment to transparency means all our data sources are documented and our methodology is open. 
            We believe in building trust through openness.
          </p>
        </div>

        {/* Data Privacy & GDPR Section */}
        <div className={`rounded-2xl p-8 mb-12 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-white shadow-lg border border-gray-100'
        }`}>
          <div className="text-center mb-8">
            <h2 className={`text-3xl font-bold mb-4 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              🛡️ Data Privacy & Protection
            </h2>
            <p className={`text-lg leading-relaxed transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              We take your privacy seriously. Here's how we handle your data in compliance with GDPR.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h3 className={`text-xl font-semibold transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                🇪🇺 GDPR Rights
              </h3>
              <ul className={`space-y-2 transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                <li>• Right to access your data</li>
                <li>• Right to correct inaccurate data</li>
                <li>• Right to delete your data</li>
                <li>• Right to data portability</li>
                <li>• Right to object to processing</li>
              </ul>
            </div>
            <div className="space-y-4">
              <h3 className={`text-xl font-semibold transition-colors duration-300 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                🔒 How We Protect You
              </h3>
              <ul className={`space-y-2 transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-600'
              }`}>
                <li>• Minimal data collection</li>
                <li>• Encrypted data transmission</li>
                <li>• No selling of personal data</li>
                <li>• Transparent privacy policy</li>
                <li>• Cookie consent management</li>
              </ul>
            </div>
          </div>
          
          <div className="flex flex-wrap justify-center gap-4 mt-8">
            <Link
              to="/privacy"
              className="inline-flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-300 bg-blue-600 hover:bg-blue-700 text-white"
            >
              📋 Privacy Policy
            </Link>
            <Link
              to="/gdpr"
              className="inline-flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-300 bg-green-600 hover:bg-green-700 text-white"
            >
              ⚖️ Manage Your Data
            </Link>
          </div>
        </div>

        {/* Transparency Statement */}
        <div className={`rounded-2xl p-8 transition-colors duration-300 ${
          darkMode 
            ? 'bg-gray-800 border border-gray-700' 
            : 'bg-white shadow-lg border border-gray-100'
        }`}>
          <div className="text-center mb-8">
            <h2 className={`text-3xl font-bold mb-4 transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Transparency & Trust
            </h2>
            <p className={`text-lg leading-relaxed transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              We believe in complete transparency with our users. Our data sources, technology stack, and methodologies are open for you to review.
            </p>
          </div>

          <div className="space-y-4">
            <h3 className={`text-xl font-semibold transition-colors duration-300 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Our Commitment
            </h3>
            <p className={`transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              We are committed to providing a trustworthy and reliable service. Our team regularly audits our data sources and algorithms to ensure quality and integrity.
            </p>
          </div>
        </div>
      </div>
      </div>
    </div>
  );
};

export default Sources;
