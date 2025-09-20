import React, { useState, useEffect } from 'react';
import { EnhancedChatInterface, SmartResponse, ResponseTypeIndicator } from '../components/EnhancedChat';
import { 
  RestaurantSkeleton, 
  MuseumSkeleton, 
  BlogPostSkeleton,
  TypingIndicator,
  SearchResultsSkeleton 
} from '../components/LoadingSkeletons';
import { TypingSimulator, WordByWordTyping, StreamingText } from '../components/TypingAnimation';

/**
 * UX Enhancements Demo Page
 * Showcases all the new UX features
 */
const UXEnhancementsDemo = () => {
  const [activeTab, setActiveTab] = useState('typing');
  const [demoMessages, setDemoMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Welcome to Istanbul AI Assistant! I can help you discover amazing restaurants, museums, and attractions in Istanbul.',
      typed: false
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);

  const tabs = [
    { id: 'typing', label: 'Typing Animation', icon: '⌨️' },
    { id: 'skeletons', label: 'Loading Skeletons', icon: '💀' },
    { id: 'chat', label: 'Enhanced Chat', icon: '💬' },
    { id: 'responses', label: 'Smart Responses', icon: '🧠' }
  ];

  const handleSendMessage = (message) => {
    // Add user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: message,
      typed: true
    };
    
    setDemoMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Simulate API response
    setTimeout(() => {
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `Here's information about "${message}" in Istanbul. I found several great options including restaurants in Beyoğlu, museums in Sultanahmet, and cultural sites near Galata Tower. Would you like me to provide more specific details about any of these areas?`,
        typed: false,
        source: Math.random() > 0.5 ? 'ai' : 'cache'
      };
      
      setDemoMessages(prev => [...prev, botMessage]);
      setIsLoading(false);
    }, 2000);
  };

  const TypingDemos = () => (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Character-by-Character Typing</h3>
        <div className="bg-gray-50 p-4 rounded border">
          <TypingSimulator 
            text="Istanbul is a magnificent city that bridges Europe and Asia, offering incredible history, delicious cuisine, and stunning architecture."
            speed={30}
            variation={20}
            className="text-gray-800"
          />
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Word-by-Word Typing</h3>
        <div className="bg-gray-50 p-4 rounded border">
          <WordByWordTyping 
            text="🍽️ Best Restaurants in Beyoğlu: Mikla offers modern Turkish cuisine with panoramic city views, while Karaköy Lokantası serves traditional Ottoman dishes in an elegant setting."
            speed={100}
            variation={50}
            className="text-gray-800"
          />
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Streaming Text (ChatGPT Style)</h3>
        <div className="bg-gray-50 p-4 rounded border">
          <StreamingText 
            text="🏛️ Must-Visit Museums: Hagia Sophia showcases Byzantine and Ottoman architecture, Topkapi Palace displays imperial treasures, and the Basilica Cistern offers an underground marvel with ancient columns."
            speed={25}
            className="text-gray-800"
          />
        </div>
      </div>
    </div>
  );

  const SkeletonDemos = () => (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Restaurant Loading Skeletons</h3>
        <RestaurantSkeleton count={2} />
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Museum Loading Skeletons</h3>
        <MuseumSkeleton count={1} />
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Blog Post Loading Skeletons</h3>
        <BlogPostSkeleton count={3} variant="card" />
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Search Results Loading</h3>
        <SearchResultsSkeleton count={3} />
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Typing Indicator</h3>
        <TypingIndicator />
      </div>
    </div>
  );

  const ChatDemo = () => (
    <div className="bg-white rounded-lg shadow-md" style={{ height: '600px' }}>
      <EnhancedChatInterface
        messages={demoMessages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        enableTypingAnimation={true}
        placeholder="Try asking about restaurants, museums, or attractions..."
      />
    </div>
  );

  const ResponseDemos = () => (
    <div className="space-y-8">
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">AI Response with Animation</h3>
        <ResponseTypeIndicator source="ai" />
        <div className="bg-gray-50 p-4 rounded border">
          <SmartResponse 
            response="Based on your interest in Turkish cuisine, I recommend visiting Çiya Sofrası in Kadıköy for authentic regional dishes, or Pandeli for a historic dining experience near the Grand Bazaar."
            source="ai"
          />
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Cached Response (Instant)</h3>
        <ResponseTypeIndicator source="cache" />
        <div className="bg-gray-50 p-4 rounded border">
          <SmartResponse 
            response="🚇 Istanbul Metro Guide: The M2 line connects Vezneciler (near Grand Bazaar) to Şişhane (near Galata Tower). Single ride costs 15 TL, or use an Istanbulkart for discounted travel."
            source="cache"
          />
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-semibold mb-4">Fallback Knowledge Base</h3>
        <ResponseTypeIndicator source="fallback" />
        <div className="bg-gray-50 p-4 rounded border">
          <SmartResponse 
            response="🏛️ Topkapi Palace is open 9:00-18:00 (closed Tuesdays). Entry fee: 200 TL. Highlights include imperial chambers, holy relics, and stunning Bosphorus views. Allow 2-3 hours for your visit."
            source="fallback"
          />
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'typing':
        return <TypingDemos />;
      case 'skeletons':
        return <SkeletonDemos />;
      case 'chat':
        return <ChatDemo />;
      case 'responses':
        return <ResponseDemos />;
      default:
        return <TypingDemos />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            🎨 UX Enhancements Demo
          </h1>
          <p className="text-gray-600">
            Explore the enhanced user experience features for Istanbul AI Assistant
          </p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white border-b">
        <div className="max-w-6xl mx-auto px-4">
          <nav className="flex space-x-8">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-2 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 py-8">
        {renderContent()}
      </div>

      {/* Footer */}
      <div className="bg-white border-t mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6">
          <div className="flex justify-between items-center">
            <div>
              <h3 className="font-semibold text-gray-900">Features Implemented</h3>
              <p className="text-sm text-gray-600">
                ✅ Typing Animation • ✅ Loading Skeletons • ✅ Enhanced Chat • ✅ Smart Responses
              </p>
            </div>
            <div className="text-sm text-gray-500">
              Istanbul AI Assistant v2.0
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UXEnhancementsDemo;
