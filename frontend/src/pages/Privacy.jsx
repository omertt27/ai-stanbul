import React from 'react';
import { Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../contexts/ThemeContext';

function Privacy() {
  const { t } = useTranslation();
  const { darkMode } = useTheme();
  
  return (
    <div className={`min-h-screen transition-colors duration-300 mobile-scroll-optimized ${darkMode ? 'bg-gray-900 text-white' : 'bg-white text-gray-900'}`} style={{ paddingTop: '6rem', paddingBottom: '4rem', paddingLeft: '2rem', paddingRight: '2rem' }}>
      <div className="max-w-4xl mx-auto mobile-touch-optimized">
        <h1>{t('privacy.title')}</h1>
      <p><em>{t('privacy.lastUpdated')}: September 2025</em></p>

      <div className="gdpr-notice" style={{
        background: darkMode ? '#1e3a8a' : '#dbeafe',
        color: darkMode ? '#e5e7eb' : '#1e3a8a',
        padding: '1rem',
        borderRadius: '8px',
        marginBottom: '2rem'
      }}>
        <h3>{t('privacy.gdprNoticeTitle')}</h3>
        <p>
          {t('privacy.gdprNoticeText')}
          <br />
          <Link to="/gdpr" style={{color: darkMode ? '#93c5fd' : '#1d4ed8', textDecoration: 'underline'}}>
            {t('privacy.gdprLinkText')}
          </Link>
        </p>
      </div>

      <h2>{t('privacy.dataControllerTitle')}</h2>
      <p>
        <strong>{t('privacy.dataController')}:</strong> {t('privacy.dataControllerName')}<br />
        <strong>{t('privacy.contactEmail')}:</strong> {t('privacy.contactEmailValue')}<br />
        <strong>{t('privacy.legalBasis')}:</strong> {t('privacy.legalBasisValue')}
      </p>

      <h2>{t('privacy.informationWeCollectTitle')}</h2>
      
      <h3>{t('privacy.personalDataCategoriesTitle')}</h3>
      <ul>
        <li><strong>{t('privacy.chatMessages')}:</strong> {t('privacy.chatMessagesDesc')}</li>
        <li><strong>{t('privacy.technicalData')}:</strong> {t('privacy.technicalDataDesc')}</li>
        <li><strong>{t('privacy.usageAnalytics')}:</strong> {t('privacy.usageAnalyticsDesc')}</li>
        <li><strong>{t('privacy.preferences')}:</strong> {t('privacy.preferencesDesc')}</li>
        <li><strong>{t('privacy.feedbackData')}:</strong> {t('privacy.feedbackDataDesc')}</li>
        <li><strong>{t('privacy.cookieData')}:</strong> {t('privacy.cookieDataDesc')}</li>
      </ul>

      <h3>{t('privacy.legalBasisProcessingTitle')}</h3>
      <ul>
        <li><strong>{t('privacy.consentGdpr')}:</strong> {t('privacy.consentGdprDesc')}</li>
        <li><strong>{t('privacy.legitimateInterest')}:</strong> {t('privacy.legitimateInterestDesc')}</li>
        <li><strong>{t('privacy.contractualNecessity')}:</strong> {t('privacy.contractualNecessityDesc')}</li>
      </ul>

      <h2>{t('privacy.howWeUseTitle')}</h2>
      <ul>
        <li>{t('privacy.usePersonalized')}</li>
        <li>{t('privacy.useImprove')}</li>
        <li>{t('privacy.useAnalyze')}</li>
        <li>{t('privacy.useSecurity')}</li>
        <li>{t('privacy.useComply')}</li>
      </ul>

      <h2>{t('privacy.dataSharingTitle')}</h2>
      <p>{t('privacy.dataSharingIntro')}</p>
      <ul>
        <li><strong>{t('privacy.googleMaps')}:</strong> {t('privacy.googleMapsDesc')}</li>
        <li><strong>{t('privacy.openaiApi')}:</strong> {t('privacy.openaiApiDesc')}</li>
        <li><strong>{t('privacy.googleAnalytics')}:</strong> {t('privacy.googleAnalyticsDesc')}</li>
        <li><strong>{t('privacy.renderCom')}:</strong> {t('privacy.renderComDesc')}</li>
      </ul>
      
      <p>
        <strong>{t('privacy.safeguards')}:</strong> {t('privacy.safeguardsDesc')}
      </p>

      <h2>{t('privacy.dataRetentionTitle')}</h2>
      <ul>
        <li><strong>{t('privacy.chatMessagesRetention')}:</strong> {t('privacy.chatMessagesRetentionDesc')}</li>
        <li><strong>{t('privacy.analyticsDataRetention')}:</strong> {t('privacy.analyticsDataRetentionDesc')}</li>
        <li><strong>{t('privacy.userPreferencesRetention')}:</strong> {t('privacy.userPreferencesRetentionDesc')}</li>
        <li><strong>{t('privacy.errorLogsRetention')}:</strong> {t('privacy.errorLogsRetentionDesc')}</li>
      </ul>

      <h2>{t('privacy.cookiesTrackingTitle')}</h2>
      <p>{t('privacy.cookiesIntro')}</p>
      <ul>
        <li><strong>{t('privacy.necessaryCookies')}:</strong> {t('privacy.necessaryCookiesDesc')}</li>
        <li><strong>{t('privacy.analyticsCookies')}:</strong> {t('privacy.analyticsCookiesDesc')}</li>
        <li><strong>{t('privacy.preferenceCookies')}:</strong> {t('privacy.preferenceCookiesDesc')}</li>
      </ul>
      <p>
        {t('privacy.cookieManagement')}
      </p>

      <h2>{t('privacy.dataSecurityTitle')}</h2>
      <ul>
        <li>{t('privacy.endToEndEncryption')}</li>
        <li>{t('privacy.securityAudits')}</li>
        <li>{t('privacy.accessControls')}</li>
        <li>{t('privacy.dataMinimization')}</li>
        <li>{t('privacy.incidentResponse')}</li>
      </ul>

      <h2>{t('privacy.childrensPrivacyTitle')}</h2>
      <p>
        {t('privacy.childrensPrivacyText')}
      </p>

      <h2>{t('privacy.dataBreachTitle')}</h2>
      <p>
        {t('privacy.dataBreachText')}
      </p>

      <h2>{t('privacy.changesToPolicyTitle')}</h2>
      <p>
        {t('privacy.changesToPolicyText')}
      </p>

      <h2>{t('privacy.contactDataProtectionTitle')}</h2>
      <p>{t('privacy.contactDataProtectionIntro')}</p>
      <ul>
        <li><strong>{t('privacy.email')}:</strong> {t('privacy.emailValue')}</li>
        <li><strong>{t('privacy.dataProtectionOfficer')}:</strong> {t('privacy.dataProtectionOfficerValue')}</li>
        <li><strong>{t('privacy.gdprPortal')}:</strong> <Link to="/gdpr">{t('privacy.gdprPortalValue')}</Link></li>
      </ul>

      <h3>{t('privacy.supervisoryAuthorityTitle')}</h3>
      <p>
        {t('privacy.supervisoryAuthorityIntro')}
        <br />
        <strong>{t('privacy.turkeyAuthority')}:</strong> {t('privacy.turkeyAuthorityValue')}
        <br />
        <strong>{t('privacy.euAuthority')}:</strong> {t('privacy.euAuthorityValue')}
      </p>

      <div style={{
        background: darkMode ? '#065f46' : '#d1fae5',
        color: darkMode ? '#a7f3d0' : '#065f46',
        padding: '1rem',
        borderRadius: '8px',
        marginTop: '2rem'
      }}>
        <p style={{ marginBottom: 0, fontStyle: 'italic' }}>
          <strong>{t('privacy.commitmentTitle')}:</strong> {t('privacy.commitmentText')}
        </p>
      </div>
      </div>
    </div>
  );
}

export default Privacy;
