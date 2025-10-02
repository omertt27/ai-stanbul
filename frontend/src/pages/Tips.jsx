import React from 'react';
import { useTranslation } from 'react-i18next';
import MainPageMobileNavbar from '../components/MainPageMobileNavbar';
import '../App.css';

function Tips() {
  const { t } = useTranslation();

  return (
    <div className="min-h-screen w-full transition-colors duration-300 bg-gray-900" style={{ marginTop: '0px', paddingLeft: '1.5rem', paddingRight: '1.5rem', paddingBottom: '3rem' }}>
      <MainPageMobileNavbar />
      <div className="max-w-6xl mx-auto">
        <div className="pt-4 pb-24">
          {/* Hero Section */}
          <div className="pb-16">
            <div className="max-w-4xl mx-auto px-8 text-center">
              <div className="mb-8">
                <h1 className="text-5xl font-bold mb-6 pt-28 transition-colors duration-300 text-white">
                  {t('tips.title')}
                </h1>
                <p className="text-xl leading-relaxed transition-colors duration-300 text-gray-300">
                  {t('tips.subtitle')}
                </p>
              </div>
            </div>
          </div>

          {/* Tips Content */}
          <div className="max-w-4xl mx-auto space-y-8">
            {/* Getting Around */}
            <div className="bg-gray-800 rounded-lg p-8 shadow-lg">
              <h2 className="text-2xl font-bold mb-6 text-blue-400">{t('tips.gettingAround')}</h2>
              <ul className="space-y-3 text-gray-300">
                <li>• {t('tips.publicTransport')}</li>
                <li>• {t('tips.traffic')}</li>
                <li>• {t('tips.ferries')}</li>
              </ul>
            </div>

            {/* Cultural Tips */}
            <div className="bg-gray-800 rounded-lg p-8 shadow-lg">
              <h2 className="text-2xl font-bold mb-6 text-purple-400">{t('tips.cultural')}</h2>
              <ul className="space-y-3 text-gray-300">
                <li>• {t('tips.mosques')}</li>
                <li>• {t('tips.dress')}</li>
                <li>• {t('tips.tea')}</li>
              </ul>
            </div>

            {/* Money & Shopping */}
            <div className="bg-gray-800 rounded-lg p-8 shadow-lg">
              <h2 className="text-2xl font-bold mb-6 text-green-400">{t('tips.money')}</h2>
              <ul className="space-y-3 text-gray-300">
                <li>• {t('tips.bargain')}</li>
                <li>• {t('tips.cash')}</li>
                <li>• {t('tips.tipping')}</li>
              </ul>
            </div>

            {/* Safety & Practical */}
            <div className="bg-gray-800 rounded-lg p-8 shadow-lg">
              <h2 className="text-2xl font-bold mb-6 text-orange-400">{t('tips.safety')}</h2>
              <ul className="space-y-3 text-gray-300">
                <li>• {t('tips.water')}</li>
                <li>• {t('tips.language')}</li>
                <li>• {t('tips.emergency')}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Tips;