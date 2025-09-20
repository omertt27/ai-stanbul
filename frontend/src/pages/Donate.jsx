import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../contexts/ThemeContext';

const Donate = () => {
  const { t } = useTranslation();
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
              {t('donate.title')}
            </h1>
            <p className="text-xl leading-relaxed transition-colors duration-300 text-gray-300">
              {t('donate.subtitle')}
            </p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto px-6 pb-20">
        {/* Buy Me a Coffee Section */}
        <div className="rounded-2xl p-8 mb-12 text-center transition-colors duration-300 bg-gradient-to-r from-yellow-900/30 to-orange-900/30 border border-yellow-800/50">
          <h2 className="text-3xl font-bold mb-4 transition-colors duration-300 text-yellow-300">
            {t('donate.buyMeCoffee')}
          </h2>
          <p className="text-lg mb-6 transition-colors duration-300 text-yellow-100">
            {t('donate.coffeeDescription')}
          </p>
          <a 
            href="https://www.buymeacoffee.com/aistanbul" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-block px-8 py-4 bg-yellow-500 hover:bg-yellow-600 text-white font-bold text-lg rounded-lg transition-all duration-300 hover:scale-105 hover:shadow-lg"
          >
            {t('donate.buyMeCoffeeButton')}
          </a>
        </div>

        {/* Benefits Section */}
        <div className="rounded-2xl p-8 mb-12 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <h2 className="text-3xl font-bold text-center mb-8 transition-colors duration-300 text-white">
            {t('donate.howSupportHelps')}
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            <div className="space-y-6">
              {benefitItem('', t('donate.keepItFree'), t('donate.keepItFreeDesc'))}
              {benefitItem('', t('donate.serverCosts'), t('donate.serverCostsDesc'))}
              {benefitItem('', t('donate.continuousUpdates'), t('donate.continuousUpdatesDesc'))}
            </div>
            <div className="space-y-6">
              {benefitItem('', t('donate.newFeatures'), t('donate.newFeaturesDesc'))}
              {benefitItem('', t('donate.multilingualSupport'), t('donate.multilingualSupportDesc'))}
              {benefitItem('', t('donate.betterExperience'), t('donate.betterExperienceDesc'))}
            </div>
          </div>
        </div>

        {/* Alternative Support Methods */}
        <div className="rounded-2xl p-8 mb-12 transition-colors duration-300 bg-gray-800 border border-gray-700">
          <h2 className="text-3xl font-bold text-center mb-8 transition-colors duration-300 text-white">
            {t('donate.otherWaysToHelp')}
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
                  {t('donate.shareOnSocialMedia')}
                </h3>
              </div>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                {t('donate.shareOnSocialMediaDesc')}
              </p>
            </a>
            
            <a 
              href="mailto:feedback@aistanbul.com?subject=AIstanbul%20Feedback&body=Hi!%20I%20have%20some%20feedback%20about%20AIstanbul..." 
              className="block p-6 rounded-lg border-2 border-dashed transition-all duration-300 hover:scale-105 border-gray-600 hover:border-green-500 hover:bg-green-900/20"
            >
              <div className="flex items-center mb-2">
                <h3 className="font-semibold text-lg transition-colors duration-300 text-white">
                  {t('donate.sendFeedback')}
                </h3>
              </div>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                {t('donate.sendFeedbackDesc')}
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
                  {t('donate.contributeOnGithub')}
                </h3>
              </div>
              <p className="text-sm transition-colors duration-300 text-gray-300">
                {t('donate.contributeOnGithubDesc')}
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
