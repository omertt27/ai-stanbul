import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../contexts/ThemeContext';
import '../App.css';

const About = () => {
  const { darkMode } = useTheme();
  const { t } = useTranslation();

  return (
    <div className="min-h-screen w-full transition-colors duration-300 bg-gray-900" style={{ marginTop: '0px', paddingLeft: '1.5rem', paddingRight: '1.5rem', paddingBottom: '1rem' }}>
      <div className="max-w-6xl mx-auto">

      {/* Scrollable Content */}
      <div className="pt-4 pb-24">
        {/* Hero Section */}
        <div className="pb-16">
        <div className="max-w-4xl mx-auto px-12 text-center">
          <div className="mb-8">
            <h1 className="text-5xl font-bold mb-6 pt-28 transition-colors duration-300 text-white">
              {t('about.title')} <span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent font-black">A/</span><span className="bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent font-normal">STANBUL</span>
            </h1>
            <p className="text-xl leading-relaxed transition-colors duration-300 text-gray-300">
              {t('about.subtitle')}
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-8 pb-24">
        {/* Mission Section */}
        <div className="rounded-2xl p-12 mb-16 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-4 transition-colors duration-300 text-white">
              {t('about.mission.title')}
            </h2>
            <p className="text-lg leading-relaxed transition-colors duration-300 text-gray-300">
              {t('about.mission.description')}
            </p>
          </div>
          
          {/* Feature Cards */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-10">
            {[
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                ),
                title: t('about.features.personalizedAI.title'),
                description: t('about.features.personalizedAI.description')
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                ),
                title: t('about.features.localExpertise.title'),
                description: t('about.features.localExpertise.description')
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                ),
                title: t('about.features.oneTapActions.title'),
                description: t('about.features.oneTapActions.description')
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064" />
                  </svg>
                ),
                title: t('about.features.culturalBridge.title'),
                description: t('about.features.culturalBridge.description')
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                ),
                title: t('about.features.realTimeData.title'),
                description: t('about.features.realTimeData.description')
              },
              {
                icon: (
                  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                  </svg>
                ),
                title: t('about.features.madeForIstanbul.title'),
                description: t('about.features.madeForIstanbul.description')
              }
            ].map((feature, index) => (
              <div key={index} className="p-8 rounded-xl transition-all duration-300 bg-gray-700 hover:bg-gray-600">
                <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2 transition-colors duration-300 text-white">
                  {feature.title}
                </h3>
                <p className="text-sm leading-relaxed transition-colors duration-300 text-gray-300">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* What We Offer Section */}
        <div className="rounded-2xl p-12 mb-16 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <h2 className="text-3xl font-bold mb-8 text-center transition-colors duration-300 text-white">
            {t('about.whatWeOffer.title')}
          </h2>
          
          <div className="grid lg:grid-cols-2 gap-10">
            {[
              {
                title: t('about.whatWeOffer.restaurants.title'),
                description: t('about.whatWeOffer.restaurants.description')
              },
              {
                title: t('about.whatWeOffer.museums.title'),
                description: t('about.whatWeOffer.museums.description')
              },
              {
                title: t('about.whatWeOffer.neighborhoods.title'),
                description: t('about.whatWeOffer.neighborhoods.description')
              },
              {
                title: t('about.whatWeOffer.transportation.title'),
                description: t('about.whatWeOffer.transportation.description')
              }
            ].map((item, index) => (
              <div key={index} className="p-8 rounded-xl border transition-all duration-300 bg-gray-700 border-gray-600 hover:border-blue-500">
                <h3 className="text-xl font-semibold mb-3 transition-colors duration-300 text-white">
                  {item.title}
                </h3>
                <p className="leading-relaxed transition-colors duration-300 text-gray-300">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Technical Innovation Section */}
        <div className="rounded-2xl p-12 mb-16 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-4 transition-colors duration-300 text-white">
              ⚡ Technical Innovation
            </h2>
            <p className="text-lg leading-relaxed transition-colors duration-300 text-gray-300">
              Cutting-edge technology powering intelligent travel assistance
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="p-6 rounded-xl transition-all duration-300 bg-gray-700 hover:bg-gray-600">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r from-blue-500 to-cyan-600 text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">High Performance</h3>
              <p className="text-sm text-gray-300">95+ Lighthouse score with sub-second response times and real-time streaming</p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-300 bg-gray-700 hover:bg-gray-600">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r from-purple-500 to-pink-600 text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">AI Intelligence</h3>
              <p className="text-sm text-gray-300">OpenAI GPT-3.5 with context awareness, session memory, and intent recognition</p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-300 bg-gray-700 hover:bg-gray-600">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">Enterprise Security</h3>
              <p className="text-sm text-gray-300">GDPR compliance, XSS protection, rate limiting, and comprehensive input sanitization</p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-300 bg-gray-700 hover:bg-gray-600">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r from-yellow-500 to-orange-600 text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">Smart Caching</h3>
              <p className="text-sm text-gray-300">Redis-powered AI cache with intelligent rate limiting and session management</p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-300 bg-gray-700 hover:bg-gray-600">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r from-indigo-500 to-purple-600 text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9-9v18" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">Multilingual</h3>
              <p className="text-sm text-gray-300">Support for English, Turkish, Arabic, and Russian with cultural adaptation</p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-300 bg-gray-700 hover:bg-gray-600">
              <div className="w-12 h-12 rounded-lg flex items-center justify-center mb-4 bg-gradient-to-r from-red-500 to-pink-600 text-white">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.031 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold mb-2 text-white">Error Resilience</h3>
              <p className="text-sm text-gray-300">Circuit breaker patterns, fallback systems, and zero-downtime architecture</p>
            </div>
          </div>
        </div>

        {/* GDPR & Privacy Section */}
        <div className="rounded-2xl p-12 mb-16 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold mb-4 transition-colors duration-300 text-white">
              🛡️ {t('about.privacy.title')}
            </h2>
            <p className="text-lg leading-relaxed transition-colors duration-300 text-gray-300">
              {t('about.privacy.description')}
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-white">🇪🇺 {t('about.privacy.gdpr.title')}</h3>
              <ul className="space-y-2 text-gray-300">
                <li>• {t('about.privacy.gdpr.transparent')}</li>
                <li>• {t('about.privacy.gdpr.consent')}</li>
                <li>• {t('about.privacy.gdpr.deletion')}</li>
                <li>• {t('about.privacy.gdpr.portability')}</li>
              </ul>
            </div>
            <div className="space-y-4">
              <h3 className="text-xl font-semibold text-white">🔒 {t('about.privacy.security.title')}</h3>
              <ul className="space-y-2 text-gray-300">
                <li>• {t('about.privacy.security.encryption')}</li>
                <li>• {t('about.privacy.security.minimal')}</li>
                <li>• {t('about.privacy.security.noSelling')}</li>
                <li>• {t('about.privacy.security.audits')}</li>
              </ul>
            </div>
          </div>
          
          <div className="flex flex-wrap justify-center gap-4 mt-8">
            <Link
              to="/privacy"
              className="inline-flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-300 bg-blue-600 hover:bg-blue-700 text-white"
            >
              📋 {t('about.privacy.privacyPolicy')}
            </Link>
            <Link
              to="/gdpr"
              className="inline-flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-300 bg-green-600 hover:bg-green-700 text-white"
            >
              ⚖️ {t('about.privacy.manageData')}
            </Link>
          </div>
        </div>

        {/* Open Source Section */}
        <div className="rounded-2xl p-12 text-center transition-colors duration-300 bg-gradient-to-r from-gray-800 to-gray-700 border border-gray-600">
          <div className="mb-6">
            <svg className="w-16 h-16 mx-auto mb-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            <h2 className="text-3xl font-bold mb-4 text-white">
              {t('about.openSource.title')}
            </h2>
            <p className="text-lg mb-6 text-gray-200">
              {t('about.openSource.description')}
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <a
                href="https://github.com/yourusername/ai-stanbul"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-300 bg-blue-600 hover:bg-blue-700 text-white"
              >
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                {t('about.openSource.viewGithub')}
              </a>
              <Link
                to="/"
                className="inline-flex items-center px-6 py-3 rounded-lg font-semibold transition-all duration-300 bg-purple-600 hover:bg-purple-700 text-white"
              >
                {t('about.openSource.tryNow')}
              </Link>
            </div>
          </div>
        </div>
      </div>
      </div>
      </div>
    </div>
  );
};

export default About;
