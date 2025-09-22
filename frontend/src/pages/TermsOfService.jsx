import React from 'react';

const TermsOfService = () => {
  return (
    <div className="min-h-screen bg-gray-50" style={{ paddingTop: '6rem', paddingBottom: '2rem' }}>
      <div className="max-w-4xl mx-auto p-6 bg-white" style={{ marginTop: '2rem', borderRadius: '0.5rem', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)' }}>
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-red-600 mb-2">
          🔒 Terms of Service - AI Istanbul
        </h1>
        <p className="text-gray-600">Last Updated: September 22, 2025</p>
      </div>

      <div className="space-y-8">
        <section className="border-l-4 border-red-500 pl-6">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            🔒 COPYRIGHT AND INTELLECTUAL PROPERTY
          </h2>
          
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-gray-700 mb-3">Ownership</h3>
            <p className="text-gray-600 mb-3">
              AI Istanbul and all its contents, including but not limited to:
            </p>
            <ul className="list-disc list-inside text-gray-600 space-y-1 ml-4">
              <li>Source code and algorithms</li>
              <li>User interface design and graphics</li>
              <li>AI model implementations</li>
              <li>Database structures and content</li>
              <li>API endpoints and responses</li>
              <li>Documentation and text content</li>
            </ul>
            <p className="text-gray-600 mt-3">
              Are the exclusive property of AI Istanbul and are protected by international copyright laws, 
              software patents, trade secret protection, and trademark rights.
            </p>
          </div>
        </section>

        <section className="bg-red-50 p-6 rounded-lg border border-red-200">
          <h2 className="text-2xl font-bold text-red-700 mb-4">
            ❌ PROHIBITED ACTIVITIES
          </h2>
          <p className="text-red-600 font-semibold mb-4">Users are STRICTLY PROHIBITED from:</p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>Code Copying:</strong> Copying, reproducing, or reverse-engineering any part of the source code</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>Design Theft:</strong> Replicating the user interface, design patterns, or visual elements</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>API Scraping:</strong> Unauthorized access to or scraping of API endpoints</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>Database Extraction:</strong> Attempting to extract or copy database contents</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>Algorithm Replication:</strong> Reverse-engineering or copying AI algorithms and models</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>Trademark Infringement:</strong> Using "AI Istanbul" name, logo, or branding</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>Commercial Use:</strong> Using any part of the system for commercial purposes without license</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="text-red-500">❌</span>
                <span className="text-gray-700"><strong>Redistribution:</strong> Sharing, selling, or distributing any component</span>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-yellow-50 p-6 rounded-lg border border-yellow-200">
          <h2 className="text-2xl font-bold text-yellow-700 mb-4">
            ⚖️ ENFORCEMENT
          </h2>
          <p className="text-yellow-600 font-semibold mb-4">Violations will result in:</p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="text-yellow-600">⚖️</span>
                <span className="text-gray-700"><strong>Legal Action:</strong> Immediate cease and desist orders</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-yellow-600">💰</span>
                <span className="text-gray-700"><strong>Financial Penalties:</strong> Damages up to $100,000 per violation</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <span className="text-yellow-600">🚫</span>
                <span className="text-gray-700"><strong>Service Termination:</strong> Permanent ban from all services</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-yellow-600">📞</span>
                <span className="text-gray-700"><strong>Criminal Charges:</strong> Filing with appropriate authorities</span>
              </div>
            </div>
          </div>
        </section>

        <section className="bg-blue-50 p-6 rounded-lg border border-blue-200">
          <h2 className="text-2xl font-bold text-blue-700 mb-4">
            📧 LICENSING & CONTACT
          </h2>
          <p className="text-blue-600 mb-4">For legitimate business use:</p>
          <div className="space-y-2">
            <div className="flex items-center space-x-2">
              <span className="text-blue-500">📧</span>
              <span className="text-gray-700"><strong>Contact:</strong> legal@ai-istanbul.com</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-blue-500">💼</span>
              <span className="text-gray-700"><strong>Business License:</strong> Available for qualified organizations</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-blue-500">🤝</span>
              <span className="text-gray-700"><strong>Partnership:</strong> Collaboration opportunities available</span>
            </div>
          </div>
        </section>

        <div className="bg-gray-800 text-white p-6 rounded-lg text-center">
          <h3 className="text-xl font-bold mb-2">⚠️ LEGAL WARNING</h3>
          <p className="mb-2">
            This website and its contents are monitored. Unauthorized access, copying, or reproduction is illegal and will be prosecuted to the full extent of the law.
          </p>
          <p className="text-sm text-gray-300">
            © 2025 AI Istanbul™. All rights reserved. Unauthorized use is prohibited.
          </p>
        </div>
      </div>
      </div>
    </div>
  );
};

export default TermsOfService;
