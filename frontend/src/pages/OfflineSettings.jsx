import React from 'react';
import { Helmet } from 'react-helmet-async';
import { useTranslation } from 'react-i18next';
import OfflineEnhancementsUI from '../components/OfflineEnhancementsUI';

export default function OfflineSettings() {
  const { t } = useTranslation();

  return (
    <>
      <Helmet>
        <title>{t('offline.settings')} - KAM Istanbul AI</title>
        <meta name="description" content="Download Istanbul travel data with KAM for offline use without internet connection" />
      </Helmet>

      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 py-12 px-4">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
              📴 {t('offline.title', 'Offline Settings')}
            </h1>
            <p className="text-gray-300 text-lg">
              {t('offline.subtitle', 'Download Istanbul data for offline access')}
            </p>
          </div>

          <OfflineEnhancementsUI />

          <div className="mt-8 p-6 bg-gray-800 rounded-lg border border-gray-700">
            <h2 className="text-2xl font-bold text-white mb-4">💡 {t('offline.tips', 'Tips')}</h2>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-start gap-3">
                <span className="text-green-400 mt-1">✓</span>
                <span>Download map tiles before traveling to areas with poor connectivity</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-green-400 mt-1">✓</span>
                <span>Sync restaurant and attraction data while on WiFi to save mobile data</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-green-400 mt-1">✓</span>
                <span>The app automatically syncs when you go back online</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-green-400 mt-1">✓</span>
                <span>Clear cache if storage space is low on your device</span>
              </li>
            </ul>
          </div>

          <div className="mt-6 p-4 bg-blue-900/30 rounded-lg border border-blue-700">
            <p className="text-blue-200 text-sm">
              ℹ️ <strong>Note:</strong> Offline features work best when you download data before going offline. 
              The initial download may take 2-5 minutes depending on your connection speed.
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
