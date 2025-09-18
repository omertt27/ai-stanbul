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
      
      <h3>Information You Provide</h3>
      <ul>
        <li><strong>Chat Messages:</strong> Your questions and requests sent to our AI assistant</li>
        <li><strong>Location Preferences:</strong> Areas of Istanbul you're interested in exploring</li>
        <li><strong>Usage Feedback:</strong> Optional ratings and feedback you provide</li>
      </ul>

      <h3>Information Automatically Collected</h3>
      <ul>
        <li><strong>Technical Data:</strong> IP address, browser type, device information</li>
        <li><strong>Usage Analytics:</strong> How you interact with our service (anonymized)</li>
        <li><strong>Error Logs:</strong> Technical errors to help us improve the service</li>
      </ul>

      <h2>How We Use Your Information</h2>
      <p>We use your information to:</p>
      <ul>
        <li>Provide personalized Istanbul recommendations</li>
        <li>Improve our AI responses and accuracy</li>
        <li>Analyze usage patterns to enhance user experience</li>
        <li>Fix technical issues and bugs</li>
        <li>Comply with legal obligations</li>
      </ul>

      <h2>Information Sharing</h2>
      <p>We share information only in these limited circumstances:</p>
      <ul>
        <li><strong>Google Maps API:</strong> Location queries to provide restaurant recommendations</li>
        <li><strong>Anonymous Analytics:</strong> Aggregated, non-identifiable usage statistics</li>
        <li><strong>Legal Requirements:</strong> When required by law or to protect our rights</li>
      </ul>
      
      <p>
        <strong>We never sell your personal data or share it with advertisers.</strong>
      </p>

      <h2>Data Storage and Security</h2>
      <ul>
        <li>Your data is stored securely using industry-standard encryption</li>
        <li>Chat messages are processed temporarily and not permanently stored</li>
        <li>We implement regular security audits and updates</li>
        <li>Access to data is limited to essential personnel only</li>
      </ul>

      <h2>Your Rights</h2>
      <p>You have the right to:</p>
      <ul>
        <li><strong>Access:</strong> Request what personal data we have about you</li>
        <li><strong>Rectification:</strong> Correct any inaccurate information</li>
        <li><strong>Erasure:</strong> Request deletion of your personal data</li>
        <li><strong>Portability:</strong> Receive your data in a machine-readable format</li>
        <li><strong>Objection:</strong> Object to processing of your personal data</li>
      </ul>

      <h2>Cookies and Tracking</h2>
      <p>
        We use minimal cookies and local storage to:
      </p>
      <ul>
        <li>Remember your dark/light mode preference</li>
        <li>Maintain your session for better user experience</li>
        <li>Analyze website performance (anonymized)</li>
      </ul>
      <p>
        You can disable cookies in your browser settings, though this may affect functionality.
      </p>

      <h2>Third-Party Services</h2>
      <p>We integrate with:</p>
      <ul>
        <li><strong>Google Maps API:</strong> For restaurant locations and reviews</li>
        <li><strong>OpenAI API:</strong> For AI-powered responses</li>
      </ul>
      <p>
        These services have their own privacy policies, which we encourage you to review.
      </p>

      <h2>Children's Privacy</h2>
      <p>
        Our service is not directed to children under 13. We do not knowingly collect 
        personal information from children under 13. If you are a parent and believe 
        your child has provided us with personal information, please contact us.
      </p>

      <h2>International Transfers</h2>
      <p>
        Your information may be processed in countries other than your own. We ensure 
        adequate protection through appropriate safeguards and compliance with applicable laws.
      </p>

      <h2>Changes to This Policy</h2>
      <p>
        We may update this privacy policy from time to time. We will notify you of any 
        material changes by posting the new policy on this page with an updated date.
      </p>

      <h2>Contact Us</h2>
      <p>
        If you have questions about this privacy policy or our data practices, please contact us:
      </p>
      <ul>
        <li>Email: privacy@aiistanbul.guide</li>
        <li>GitHub: github.com/aiistanbul/issues</li>
      </ul>

      <p style={{ marginTop: '2rem', fontStyle: 'italic' }}>
        This privacy policy is designed to be transparent and user-friendly. We believe 
        privacy is a fundamental right and strive to protect yours while providing the 
        best possible service.
      </p>
    </div>
  );
}

export default Privacy;
