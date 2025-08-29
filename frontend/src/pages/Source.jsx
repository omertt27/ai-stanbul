import React from 'react';

function Source({ darkMode }) {
  return (
    <div className={`static-page ${darkMode ? 'dark' : ''}`}>
      <h1>Data Sources & Technology</h1>
      
      <p>
        Transparency is important to us. Here's detailed information about where our data comes from 
        and the technologies that power AI Istanbul Guide.
      </p>

      <h2>Restaurant Data</h2>
      <p>
        All restaurant recommendations are sourced in real-time from <strong>Google Maps Places API</strong>. 
        This ensures you get:
      </p>
      <ul>
        <li>Current ratings and reviews</li>
        <li>Up-to-date opening hours</li>
        <li>Real-time availability information</li>
        <li>Accurate location and contact details</li>
        <li>Price level indicators</li>
        <li>Direct links to Google Maps for navigation</li>
      </ul>

      <h2>Cultural & Historical Data</h2>
      <p>
        Our museum, monument, and cultural site information is carefully curated from multiple authoritative sources:
      </p>
      <ul>
        <li><strong>Istanbul Metropolitan Municipality</strong> - Official tourism data</li>
        <li><strong>Turkish Ministry of Culture and Tourism</strong> - Cultural heritage information</li>
        <li><strong>Local Museums</strong> - Direct partnerships for accurate details</li>
        <li><strong>UNESCO World Heritage Sites</strong> - Historical significance data</li>
        <li><strong>Local Cultural Experts</strong> - Verified insights and recommendations</li>
      </ul>

      <h2>Technology Stack</h2>
      <h3>Frontend</h3>
      <ul>
        <li><strong>React 18</strong> - Modern user interface framework</li>
        <li><strong>Tailwind CSS</strong> - Responsive design and styling</li>
        <li><strong>Vite</strong> - Fast development and build tool</li>
      </ul>

      <h3>Backend</h3>
      <ul>
        <li><strong>FastAPI</strong> - High-performance Python web framework</li>
        <li><strong>SQLite</strong> - Local database for curated content</li>
        <li><strong>OpenAI GPT</strong> - AI-powered conversation handling</li>
        <li><strong>Google Places API</strong> - Real-time restaurant data</li>
      </ul>

      <h3>Infrastructure</h3>
      <ul>
        <li><strong>Vercel</strong> - Frontend hosting and deployment</li>
        <li><strong>Railway/Render</strong> - Backend API hosting</li>
        <li><strong>GitHub</strong> - Version control and source code management</li>
      </ul>

      <h2>Data Quality Assurance</h2>
      <p>We maintain data quality through:</p>
      <ul>
        <li><strong>Regular Updates</strong> - Cultural data reviewed and updated monthly</li>
        <li><strong>Local Verification</strong> - Information verified by Istanbul residents</li>
        <li><strong>User Feedback</strong> - Continuous improvement based on user reports</li>
        <li><strong>Cross-Referencing</strong> - Multiple source verification for accuracy</li>
      </ul>

      <h2>API Rate Limits & Fair Use</h2>
      <p>
        To ensure service availability for all users, we implement reasonable rate limits:
      </p>
      <ul>
        <li>Maximum 10 queries per minute per user</li>
        <li>Restaurant search limited to 5 results to optimize response time</li>
        <li>Cached responses for frequently asked questions</li>
      </ul>

      <h2>Open Source</h2>
      <p>
        We believe in open source development. Our code is available on GitHub for:
      </p>
      <ul>
        <li>Community contributions and improvements</li>
        <li>Transparency in our algorithms and data handling</li>
        <li>Educational purposes for developers</li>
        <li>Local adaptations for other cities</li>
      </ul>
      
      <p>
        <a 
          href="https://github.com/your-username/ai-istanbul" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-blue-600 dark:text-blue-400 hover:underline"
        >
          View Source Code on GitHub →
        </a>
      </p>

      <h2>Attribution</h2>
      <p>Special thanks to:</p>
      <ul>
        <li>Google Maps Platform for restaurant data</li>
        <li>OpenAI for AI conversation capabilities</li>
        <li>Istanbul tourism authorities for cultural information</li>
        <li>Local contributors and beta testers</li>
        <li>Open source community for development tools</li>
      </ul>
    </div>
  );
}

export default Source;
