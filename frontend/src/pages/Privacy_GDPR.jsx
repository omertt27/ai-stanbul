import React from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';

function Privacy() {
  const { darkMode } = useTheme();
  
  return (
    <div className={`static-page ${darkMode ? 'dark' : ''}`}>
      <h1>Privacy Policy & GDPR Compliance</h1>
      <p><em>Last updated: September 2025</em></p>

      <div className="gdpr-notice" style={{
        background: darkMode ? '#1e3a8a' : '#dbeafe',
        color: darkMode ? '#e5e7eb' : '#1e3a8a',
        padding: '1rem',
        borderRadius: '8px',
        marginBottom: '2rem'
      }}>
        <h3>🇪🇺 GDPR Compliance Notice</h3>
        <p>
          AI Istanbul is fully compliant with the General Data Protection Regulation (GDPR). 
          As a user, you have specific rights regarding your personal data. 
          <br />
          <Link to="/gdpr" style={{color: darkMode ? '#93c5fd' : '#1d4ed8', textDecoration: 'underline'}}>
            Manage your data and privacy settings here →
          </Link>
        </p>
      </div>

      <h2>Data Controller Information</h2>
      <p>
        <strong>Data Controller:</strong> AI Istanbul Guide<br />
        <strong>Contact:</strong> privacy@aiistanbul.guide<br />
        <strong>Legal Basis:</strong> Legitimate interest for analytics, consent for personalization
      </p>

      <h2>Information We Collect</h2>
      
      <h3>Personal Data Categories</h3>
      <ul>
        <li><strong>Chat Messages:</strong> Your questions and requests (processed temporarily, not permanently stored)</li>
        <li><strong>Technical Data:</strong> IP address, browser type, device information, session IDs</li>
        <li><strong>Usage Analytics:</strong> Page views, interaction patterns (anonymized via Google Analytics)</li>
        <li><strong>Preferences:</strong> Theme settings, language preferences, dietary restrictions</li>
        <li><strong>Feedback Data:</strong> Optional ratings and comments you provide</li>
        <li><strong>Cookie Data:</strong> Consent preferences, session data</li>
      </ul>

      <h3>Legal Basis for Processing</h3>
      <ul>
        <li><strong>Consent (Art. 6(1)(a) GDPR):</strong> Cookie preferences, personalization features</li>
        <li><strong>Legitimate Interest (Art. 6(1)(f) GDPR):</strong> Website analytics, security, service improvement</li>
        <li><strong>Contractual Necessity (Art. 6(1)(b) GDPR):</strong> Providing travel recommendations</li>
      </ul>

      <h2>How We Use Your Information</h2>
      <ul>
        <li>Provide personalized Istanbul travel recommendations</li>
        <li>Improve AI response accuracy and relevance</li>
        <li>Analyze usage patterns to enhance user experience</li>
        <li>Ensure website security and prevent abuse</li>
        <li>Comply with legal obligations</li>
      </ul>

      <h2>Data Sharing & International Transfers</h2>
      <p>We share limited data only with essential service providers:</p>
      <ul>
        <li><strong>Google Maps API:</strong> Location queries for restaurant recommendations (US)</li>
        <li><strong>OpenAI API:</strong> AI-powered chat responses (US)</li>
        <li><strong>Google Analytics:</strong> Anonymized usage statistics (US)</li>
        <li><strong>Render.com:</strong> Web hosting services (US)</li>
      </ul>
      
      <p>
        <strong>Safeguards:</strong> All international transfers are protected by Standard Contractual Clauses (SCCs) 
        or adequacy decisions where available.
      </p>

      <h2>🛡️ Your GDPR Rights</h2>
      <div style={{background: darkMode ? '#374151' : '#f9fafb', padding: '1rem', borderRadius: '8px', margin: '1rem 0'}}>
        <p>Under GDPR, you have the following rights:</p>
        <ul>
          <li><strong>Right to Access (Art. 15):</strong> Request information about your personal data</li>
          <li><strong>Right to Rectification (Art. 16):</strong> Correct inaccurate personal data</li>
          <li><strong>Right to Erasure (Art. 17):</strong> Request deletion of your data ("right to be forgotten")</li>
          <li><strong>Right to Data Portability (Art. 20):</strong> Receive your data in machine-readable format</li>
          <li><strong>Right to Object (Art. 21):</strong> Object to processing based on legitimate interests</li>
          <li><strong>Right to Withdraw Consent (Art. 7(3)):</strong> Withdraw consent at any time</li>
          <li><strong>Right to Lodge a Complaint:</strong> Contact your local Data Protection Authority</li>
        </ul>
        <p>
          <Link to="/gdpr" style={{color: darkMode ? '#60a5fa' : '#2563eb', fontWeight: 'bold'}}>
            Exercise your rights here →
          </Link>
        </p>
      </div>

      <h2>Data Retention</h2>
      <ul>
        <li><strong>Chat Messages:</strong> Processed in real-time, not permanently stored</li>
        <li><strong>Analytics Data:</strong> Anonymized, retained for 26 months (Google Analytics default)</li>
        <li><strong>User Preferences:</strong> Until you clear browser data or request deletion</li>
        <li><strong>Error Logs:</strong> 30 days for technical troubleshooting</li>
      </ul>

      <h2>Cookies & Tracking</h2>
      <p>We use the following types of cookies:</p>
      <ul>
        <li><strong>Necessary Cookies:</strong> Essential for website functionality (cannot be disabled)</li>
        <li><strong>Analytics Cookies:</strong> Google Analytics for usage statistics (requires consent)</li>
        <li><strong>Preference Cookies:</strong> Remember your theme and language settings (requires consent)</li>
      </ul>
      <p>
        You can manage cookie preferences through our consent banner or by visiting your browser settings.
      </p>

      <h2>Data Security</h2>
      <ul>
        <li>End-to-end encryption for data transmission (HTTPS)</li>
        <li>Regular security audits and updates</li>
        <li>Access controls and authentication</li>
        <li>Data minimization principles</li>
        <li>Incident response procedures</li>
      </ul>

      <h2>Children's Privacy</h2>
      <p>
        Our service is not directed to children under 16. We do not knowingly collect 
        personal information from children under 16. If you are a parent and believe 
        your child has provided us with personal information, please contact us immediately.
      </p>

      <h2>Data Breach Notification</h2>
      <p>
        In case of a data breach affecting your personal data, we will notify you and 
        relevant authorities within 72 hours as required by GDPR Article 33 and 34.
      </p>

      <h2>Changes to This Policy</h2>
      <p>
        We may update this privacy policy from time to time. Material changes will be 
        communicated through our website with at least 30 days notice. Continued use 
        after changes constitutes acceptance.
      </p>

      <h2>Contact & Data Protection</h2>
      <p>For privacy-related questions or to exercise your GDPR rights:</p>
      <ul>
        <li><strong>Email:</strong> privacy@aiistanbul.guide</li>
        <li><strong>Data Protection Officer:</strong> dpo@aiistanbul.guide</li>
        <li><strong>GDPR Portal:</strong> <Link to="/gdpr">aiistanbul.guide/gdpr</Link></li>
      </ul>

      <h3>Supervisory Authority</h3>
      <p>
        If you're not satisfied with our response, you can lodge a complaint with:
        <br />
        <strong>Turkey:</strong> Personal Data Protection Authority (KVKK)
        <br />
        <strong>EU:</strong> Your local Data Protection Authority
      </p>

      <div style={{
        background: darkMode ? '#065f46' : '#d1fae5',
        color: darkMode ? '#a7f3d0' : '#065f46',
        padding: '1rem',
        borderRadius: '8px',
        marginTop: '2rem'
      }}>
        <p style={{ marginBottom: 0, fontStyle: 'italic' }}>
          <strong>🌍 Our Commitment:</strong> AI Istanbul is committed to privacy by design and transparency. 
          We collect only what's necessary to provide you with the best Istanbul travel experience while 
          respecting your fundamental right to privacy.
        </p>
      </div>
    </div>
  );
}

export default Privacy;
