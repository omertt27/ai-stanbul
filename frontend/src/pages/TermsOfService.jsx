import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../contexts/ThemeContext';
import MainPageMobileNavbar from '../components/MainPageMobileNavbar';

const TermsOfService = () => {
  const { t } = useTranslation();
  const { darkMode } = useTheme();

  return (
    <div className={`min-h-screen transition-colors duration-300 mobile-scroll-optimized ${
      darkMode ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'
    }`} style={{ paddingTop: '6rem', paddingBottom: '4rem', paddingLeft: '2rem', paddingRight: '2rem' }}>
      <MainPageMobileNavbar />
      <div className="max-w-4xl mx-auto mobile-touch-optimized">
      <div className="text-center mb-8">
        <h1 className={`text-3xl font-bold mb-2 transition-colors duration-300 ${
          darkMode ? 'text-red-400' : 'text-red-600'
        }`}>
          🔒 Terms of Service - AI Istanbul
        </h1>
        <p className={`transition-colors duration-300 ${
          darkMode ? 'text-gray-400' : 'text-gray-600'
        }`}>Last Updated: September 22, 2025</p>
      </div>

      <div className="space-y-8">
        <section className={`border-l-4 pl-6 transition-colors duration-300 ${
          darkMode ? 'border-red-500' : 'border-red-600'
        }`}>
          <h2 className={`text-2xl font-bold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-gray-200' : 'text-gray-800'
          }`}>
            🔒 COPYRIGHT AND INTELLECTUAL PROPERTY
          </h2>
          
          <div className="mb-6">
            <h3 className={`text-xl font-semibold mb-3 transition-colors duration-300 ${
              darkMode ? 'text-gray-300' : 'text-gray-700'
            }`}>Ownership</h3>
            <p className={`mb-3 transition-colors duration-300 ${
              darkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              AI Istanbul and all its contents, including but not limited to:
            </p>
            <ul className={`list-disc list-inside space-y-1 ml-4 transition-colors duration-300 ${
              darkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              <li>Source code and algorithms</li>
              <li>User interface design and graphics</li>
              <li>AI model implementations</li>
              <li>Database structures and content</li>
              <li>API endpoints and responses</li>
              <li>Documentation and text content</li>
            </ul>
            <p className={`mt-3 transition-colors duration-300 ${
              darkMode ? 'text-gray-400' : 'text-gray-600'
            }`}>
              Are the exclusive property of AI Istanbul and are protected by international copyright laws, 
              software patents, trade secret protection, and trademark rights.
            </p>
          </div>
        </section>

        <section className={`p-6 rounded-lg border transition-colors duration-300 ${
          darkMode ? 'bg-red-900 border-red-700' : 'bg-red-50 border-red-200'
        }`}>
          <h2 className={`text-2xl font-bold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-red-300' : 'text-red-700'
          }`}>
            ❌ PROHIBITED ACTIVITIES
          </h2>
          <p className={`font-semibold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-red-400' : 'text-red-600'
          }`}>Users are STRICTLY PROHIBITED from:</p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Code Copying:</strong> Copying, reproducing, or reverse-engineering any part of the source code</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Design Theft:</strong> Replicating the user interface, design patterns, or visual elements</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>API Scraping:</strong> Unauthorized access to or scraping of API endpoints</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Database Extraction:</strong> Attempting to extract or copy database contents</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Algorithm Replication:</strong> Reverse-engineering or copying AI algorithms and models</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Trademark Infringement:</strong> Using "AI Istanbul" name, logo, or branding</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Commercial Use:</strong> Using any part of the system for commercial purposes without license</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-red-500' : 'text-red-600'
                }`}>❌</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Redistribution:</strong> Sharing, selling, or distributing any component</span>
              </div>
            </div>
          </div>
        </section>

        <section className={`p-6 rounded-lg border transition-colors duration-300 ${
          darkMode ? 'bg-yellow-900 border-yellow-700' : 'bg-yellow-50 border-yellow-200'
        }`}>
          <h2 className={`text-2xl font-bold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-yellow-300' : 'text-yellow-700'
          }`}>
            ⚖️ ENFORCEMENT
          </h2>
          <p className={`font-semibold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-yellow-400' : 'text-yellow-600'
          }`}>Violations will result in:</p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-yellow-400' : 'text-yellow-600'
                }`}>⚖️</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Legal Action:</strong> Immediate cease and desist orders</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-yellow-400' : 'text-yellow-600'
                }`}>💰</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Financial Penalties:</strong> Damages up to $100,000 per violation</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-yellow-400' : 'text-yellow-600'
                }`}>🚫</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Service Termination:</strong> Permanent ban from all services</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-yellow-400' : 'text-yellow-600'
                }`}>📞</span>
                <span className={`transition-colors duration-300 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}><strong>Criminal Charges:</strong> Filing with appropriate authorities</span>
              </div>
            </div>
          </div>
        </section>

        <section className={`p-6 rounded-lg border transition-colors duration-300 ${
          darkMode ? 'bg-blue-900 border-blue-700' : 'bg-blue-50 border-blue-200'
        }`}>
          <h2 className={`text-2xl font-bold mb-4 transition-colors duration-300 ${
            darkMode ? 'text-blue-300' : 'text-blue-700'
          }`}>
            📧 LICENSING & CONTACT
          </h2>
          <p className={`mb-4 transition-colors duration-300 ${
            darkMode ? 'text-blue-400' : 'text-blue-600'
          }`}>For legitimate business use:</p>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className={`transition-colors duration-300 ${
                darkMode ? 'text-blue-400' : 'text-blue-600'
              }`}>📧</span>
              <span className={`transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}><strong>Contact:</strong> omertahtaci@aistanbul.net</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className={`transition-colors duration-300 ${
                darkMode ? 'text-blue-400' : 'text-blue-600'
              }`}>💼</span>
              <span className={`transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}><strong>Business License:</strong> Available for qualified organizations</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className={`transition-colors duration-300 ${
                darkMode ? 'text-blue-400' : 'text-blue-600'
              }`}>🤝</span>
              <span className={`transition-colors duration-300 ${
                darkMode ? 'text-gray-300' : 'text-gray-700'
              }`}><strong>Partnership:</strong> Collaboration opportunities available</span>
            </div>
          </div>
        </section>

        <div className={`p-6 rounded-lg text-center transition-colors duration-300 ${
          darkMode ? 'bg-gray-800 text-white border border-gray-700' : 'bg-gray-100 text-gray-800 border border-gray-300'
        }`}>
          <h3 className={`text-xl font-bold mb-2 transition-colors duration-300 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>⚠️ LEGAL WARNING</h3>
          <p className={`mb-2 transition-colors duration-300 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            This website and its contents are monitored. Unauthorized access, copying, or reproduction is illegal and will be prosecuted to the full extent of the law.
          </p>
          <p className={`text-sm transition-colors duration-300 ${
            darkMode ? 'text-gray-400' : 'text-gray-500'
          }`}>
            © 2025 AI Istanbul™. All rights reserved. Unauthorized use is prohibited.
          </p>
        </div>
      </div>
      </div>
    </div>
  );
};

export default TermsOfService;
