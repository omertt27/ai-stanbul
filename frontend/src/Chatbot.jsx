import { useState, useEffect } from 'react';
import { fetchStreamingResults, fetchRestaurantRecommendations, extractLocationFromQuery } from './api/api';

console.log('🔄 Chatbot component loaded with restaurant functionality');

// Helper function to render text with clickable links
const renderMessageContent = (content, darkMode) => {
  // Convert Markdown-style links [text](url) to clickable HTML links
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  
  console.log('Rendering content:', content.substring(0, 100) + '...');
  
  const parts = [];
  let lastIndex = 0;
  let match;
  
  while ((match = linkRegex.exec(content)) !== null) {
    const linkText = match[1];
    const linkUrl = match[2];
    
    console.log('Found link:', linkText, '->', linkUrl);
    
    // Add text before the link
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {content.substring(lastIndex, match.index)}
        </span>
      );
    }
    
    // Add the clickable link
    parts.push(
      <a
        key={`link-${match.index}`}
        href={linkUrl}
        target="_blank"
        rel="noopener noreferrer"
        className={`underline transition-colors duration-200 hover:opacity-80 cursor-pointer ${
          darkMode 
            ? 'text-blue-400 hover:text-blue-300' 
            : 'text-blue-600 hover:text-blue-700'
        }`}
        onClick={(e) => {
          console.log('Link clicked:', linkUrl);
        }}
      >
        {linkText}
      </a>
    );
    
    lastIndex = linkRegex.lastIndex;
  }
  
  // Add any remaining text after the last link
  if (lastIndex < content.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {content.substring(lastIndex)}
      </span>
    );
  }
  
  console.log('Generated parts:', parts.length);
  return parts.length > 0 ? parts : content;
};

// Helper function to format restaurant recommendations
const formatRestaurantRecommendations = (restaurants, locationInfo = null) => {
  console.log('formatRestaurantRecommendations called with:', { restaurants, count: restaurants?.length });
  
  if (!restaurants || restaurants.length === 0) {
    console.log('No restaurants found, returning error message');
    return "I'm sorry, I couldn't find any restaurant recommendations at the moment. Please try again or be more specific about your preferences.";
  }

  let formattedResponse = "🍽️ **Here are 4 great restaurant recommendations for you:**\n\n";
  
  restaurants.slice(0, 4).forEach((restaurant, index) => {
    const name = restaurant.name || 'Unknown Restaurant';
    const rating = restaurant.rating ? `⭐ ${restaurant.rating}` : '';
    const address = restaurant.address || restaurant.vicinity || '';
    const description = restaurant.description || 'A popular dining spot in Istanbul.';
    
    formattedResponse += `**${index + 1}. ${name}**\n`;
    if (rating) formattedResponse += `${rating}\n`;
    if (address) formattedResponse += `📍 ${address}\n`;
    formattedResponse += `${description}\n\n`;
  });

  formattedResponse += "Would you like more details about any of these restaurants or recommendations for a specific type of cuisine?";
  
  return formattedResponse;
};

// Simplified helper function - only catch very explicit restaurant+location requests
const isExplicitRestaurantRequest = (userInput) => {
  console.log('🔍 Checking for explicit restaurant request:', userInput);
  const input = userInput.toLowerCase();

  // Only intercept very specific restaurant requests with location
  const explicitRestaurantRequests = [
    'restaurants in',        // "restaurants in Beyoglu"
    'where to eat in',       // "where to eat in Sultanahmet"
    'restaurant recommendations for', // "restaurant recommendations for Taksim"
    'good restaurants in',   // "good restaurants in Galata"
    'best restaurants in',   // "best restaurants in Kadikoy"
    'restaurants near',      // "restaurants near Taksim Square"
    'where to eat near',     // "where to eat near Galata Tower"
    'dining in',             // "dining in Beyoglu"
    'food in',               // "food in Sultanahmet"
    'eat kebab in',          // "i want eat kebab in fatih"
    'want kebab in',         // "i want kebab in beyoglu"
    'eat turkish food in',   // "eat turkish food in taksim"
    'eat in',                // "i want to eat in fatih"
    'want to eat in',        // "i want to eat in sultanahmet"
    'find restaurants in',   // "find restaurants in kadikoy"
    'show me restaurants in',// "show me restaurants in galata"
    'give me restaurants in',// "give me restaurants in taksim"
    'best place to eat in',  // "best place to eat in besiktas"
    'good place to eat in',  // "good place to eat in uskudar"
    'recommend restaurants in', // "recommend restaurants in balat"
    'suggest restaurants in',   // "suggest restaurants in eminonu"
    'kebab in',              // "kebab in fatih"
    'seafood in',            // "seafood in kadikoy"
    'pizza in',              // "pizza in sisli"
    'cafe in',               // "cafe in galata"
    'breakfast in',          // "breakfast in ortakoy"
    'brunch in',             // "brunch in bebek"
    'dinner in',             // "dinner in taksim"
    'lunch in',              // "lunch in sultanahmet"
    'eat something in',      // "eat something in karakoy"
    'hungry in',             // "hungry in balat"
    'where can i eat in',    // "where can i eat in eminonu"
    'where should i eat in', // "where should i eat in maltepe"
    'food places in',        // "food places in kadikoy"
    'local food in',         // "local food in fatih"
    'authentic food in',     // "authentic food in balat"
    'traditional food in',   // "traditional food in uskudar"
    'vegetarian in',         // "vegetarian in sisli"
    'vegan in',              // "vegan in besiktas"
    'halal in',              // "halal in uskudar"
    'rooftop in',            // "rooftop in galata"
    'restaurants around',    // "restaurants around sultanahmet"
    'eat around',            // "eat around taksim"
    'food around',           // "food around galata"
    'dining around',         // "dining around kadikoy"
    'places to eat in',      // "places to eat in beyoglu"
    'best food in',          // "best food in istanbul"
    'good food in',          // "good food in istanbul"
    'find food in',          // "find food in fatih"
    'find a restaurant in',  // "find a restaurant in kadikoy"
    'find me a restaurant in', // "find me a restaurant in sultanahmet"
    'suggest a restaurant in', // "suggest a restaurant in galata"
    'recommend a restaurant in', // "recommend a restaurant in taksim"
    'show restaurants in',   // "show restaurants in besiktas"
    'show me food in',       // "show me food in kadikoy"
    'show me places to eat in', // "show me places to eat in fatih"
    'give me food in',       // "give me food in uskudar"
    'give me a restaurant in', // "give me a restaurant in balat"
    'give me places to eat in', // "give me places to eat in eminonu"
    'any restaurants in',    // "any restaurants in maltepe"
    'any good restaurants in', // "any good restaurants in taksim"
    'any food in',           // "any food in galata"
    'any place to eat in',   // "any place to eat in kadikoy"
    'any suggestions for food in', // "any suggestions for food in fatih"
    'any suggestions for restaurants in', // "any suggestions for restaurants in sultanahmet"
  ];
  
  // Only allow Istanbul or known districts
  const istanbulDistricts = [
    'istanbul', 'beyoglu', 'beyoğlu', 'galata', 'taksim', 'sultanahmet', 'fatih',
    'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş', 'uskudar', 'üsküdar', 'ortakoy',
    'ortaköy', 'sisli', 'şişli', 'karakoy', 'karaköy', 'bebek', 'arnavutkoy',
    'arnavutköy', 'balat', 'fener', 'eminonu', 'eminönü', 'bakirkoy', 'bakırköy', 'maltepe'
  ];

  const isExplicit = explicitRestaurantRequests.some(keyword => input.includes(keyword));
  if (!isExplicit) return false;
  // Extract location and check if it's Istanbul or a known district
  const { district, location } = extractLocationFromQuery(userInput);
  if (!district && !location) return false;
  // Use either district or location for matching
  const normalized = (district || location || '').trim().toLowerCase();
  // Only allow if normalized exactly matches a known Istanbul district (no partial matches)
  const isIstanbul = istanbulDistricts.includes(normalized);
  if (!isIstanbul) {
    console.log('❌ Location is not Istanbul or a known district:', normalized);
    return false;
  }
  return true;
};

function Chatbot({ onDarkModeToggle }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(true)

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  const handleSend = async (customInput = null) => {
    const userInput = customInput || input.trim();
    if (!userInput) return;

    const userMessage = { role: 'user', content: userInput };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    // Check if user is asking for restaurant recommendations
    if (isExplicitRestaurantRequest(userInput)) {
      try {
        console.log('Detected restaurant advice request, fetching recommendations...');
        console.log('User input:', userInput);
        const restaurantData = await fetchRestaurantRecommendations(userInput);
        console.log('Restaurant API response:', restaurantData);
        const formattedResponse = formatRestaurantRecommendations(restaurantData.restaurants);
        console.log('Formatted response:', formattedResponse);
        
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: formattedResponse }
        ]);
        setLoading(false);
        return;
      } catch (error) {
        console.error('Restaurant recommendation error:', error);
        // Fall back to regular AI response if restaurant API fails
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: `Sorry, I had trouble getting restaurant recommendations: ${error.message}. Let me try a different approach.` }
        ]);
        setLoading(false);
        return;
      }
    }

    // Regular streaming response for non-restaurant queries
    let streamedContent = '';
    let hasError = false;
    try {
      await fetchStreamingResults(userInput, (chunk) => {
        streamedContent += chunk;
        // If assistant message already exists, update it; else, add it
        setMessages((prev) => {
          // If last message is assistant and was streaming, update it
          if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && prev[prev.length - 1].streaming) {
            return [
              ...prev.slice(0, -1),
              { role: 'assistant', content: streamedContent, streaming: true }
            ];
          } else {
            return [
              ...prev,
              { role: 'assistant', content: streamedContent, streaming: true }
            ];
          }
        });
      });
    } catch (error) {
      hasError = true;
      console.error('Chat API Error:', error);
      const errorMessage = error.message.includes('fetch')
        ? 'Sorry, I encountered an error connecting to the server. Please make sure the backend is running on http://localhost:8001 and try again.'
        : `Sorry, there was an error: ${error.message}. Please try again.`;
      
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: errorMessage }
      ]);
    } finally {
      setLoading(false);
      // Remove streaming flag on last assistant message
      setMessages((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].role === 'assistant' && prev[prev.length - 1].streaming) {
          return [
            ...prev.slice(0, -1),
            { role: 'assistant', content: prev[prev.length - 1].content }
          ];
        }
        return prev;
      });
    }
  }

  const handleSampleClick = (question) => {
    // Automatically send the message
    handleSend(question);
  }

  return (
    <div className={`flex flex-col h-screen w-full pt-16 transition-colors duration-200 ${
      darkMode ? 'bg-gray-900' : 'bg-white'
    }`}>
      
      {/* Header - Simplified since nav is handled by parent */}
      <div className={`flex items-center justify-center px-4 py-3 border-b transition-colors duration-200 ${
        darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-white'
      }`}>
        <div className="flex items-center space-x-3">
          <div className={`w-8 h-8 rounded-sm flex items-center justify-center transition-colors duration-200 ${
            darkMode ? 'bg-white' : 'bg-black'
          }`}>
            <svg className={`w-5 h-5 transition-colors duration-200 ${
              darkMode ? 'text-black' : 'text-white'
            }`} fill="currentColor" viewBox="0 0 24 24">
              <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
            </svg>
          </div>
          <h1 className={`text-lg font-semibold transition-colors duration-200 ${
            darkMode ? 'text-white' : 'text-gray-900'
          }`}>Your AI Istanbul Assistant</h1>
        </div>
      </div>

      {/* Chat Messages Container - Full screen like ChatGPT */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center px-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-6 transition-colors duration-200 ${
              darkMode ? 'bg-white' : 'bg-black'
            }`}>
              <svg className={`w-8 h-8 transition-colors duration-200 ${
                darkMode ? 'text-black' : 'text-white'
              }`} fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
              </svg>
            </div>
            <h2 className={`text-3xl font-bold mb-4 transition-colors duration-200 ${
              darkMode ? 'text-white' : 'text-gray-900'
            }`}>How can I help you today?</h2>
            <p className={`text-center max-w-2xl text-lg leading-relaxed mb-8 transition-colors duration-200 ${
              darkMode ? 'text-gray-300' : 'text-gray-500'
            }`}>
              I'm your AI assistant for exploring Istanbul. Ask me about restaurants, attractions, 
              neighborhoods, culture, history, or anything else about this amazing city!
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl w-full px-4">
              <div 
                onClick={() => handleSampleClick('Show me the best attractions and landmarks in Istanbul')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>�️ Top Attractions</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Show me the best attractions and landmarks in Istanbul</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Give me restaurant advice - recommend 4 good restaurants')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🍽️ Restaurant Advice</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Give me restaurant advice - recommend 4 good restaurants</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('Tell me about Istanbul neighborhoods and districts to visit')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>�️ Neighborhoods</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>Tell me about Istanbul neighborhoods and districts to visit</div>
              </div>
              
              <div 
                onClick={() => handleSampleClick('What are the best cultural experiences and activities in Istanbul?')}
                className={`p-4 rounded-xl border transition-all duration-200 cursor-pointer hover:shadow-md ${
                  darkMode 
                    ? 'bg-gray-800 border-gray-700 hover:bg-gray-750' 
                    : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                }`}
              >
                <div className={`font-semibold mb-2 transition-colors duration-200 ${
                  darkMode ? 'text-white' : 'text-gray-900'
                }`}>🎭 Culture & Activities</div>
                <div className={`text-sm transition-colors duration-200 ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  What are the best cultural experiences and activities in Istanbul?
                </div>
              </div>
            </div>
          </div>
        )}
            
        <div className="max-w-full mx-auto px-4">
          {messages.map((msg, index) => (
            <div key={index} className="group py-4">
              <div className="flex items-start space-x-3">
                {msg.role === 'user' ? (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500' 
                        : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500'
                    }`}>
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>You</div>
                      <div className={`text-sm whitespace-pre-wrap transition-colors duration-200 ${
                        darkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        {renderMessageContent(msg.content, darkMode)}
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                      darkMode 
                        ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                        : 'bg-gradient-to-br from_blue-500 via-indigo-500 to-purple-500'
                    }`}>
                      <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                        darkMode ? 'text-gray-300' : 'text-gray-600'
                      }`}>AI Assistant</div>
                      <div className={`text-sm whitespace-pre-wrap leading-relaxed transition-colors duration-200 ${
                        darkMode ? 'text-white' : 'text-gray-800'
                      }`}>
                        {renderMessageContent(msg.content, darkMode)}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="group py-4">
              <div className="flex items-start space-x-3">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${
                  darkMode 
                    ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600' 
                    : 'bg-gradient-to-br from_blue-500 via-indigo-500 to-purple-500'
                }`}>
                  <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                  </svg>
                </div>
                <div className="flex-1">
                  <div className={`text-xs font-semibold mb-1 transition-colors duration-200 ${
                    darkMode ? 'text-gray-300' : 'text-gray-600'
                  }`}>AI Assistant</div>
                  <div className="flex items-center space-x-1">
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`}></div>
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`} style={{animationDelay: '0.1s'}}></div>
                    <div className={`w-1.5 h-1.5 rounded-full animate-bounce transition-colors duration-200 ${
                      darkMode ? 'bg-indigo-400' : 'bg-indigo-500'
                    }`} style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Area */}
      <div className={`border-t p-4 transition-colors duration-200 ${
        darkMode 
          ? 'border-gray-700 bg-gray-900' 
          : 'border-gray-200 bg-white'
      }`}>
        <div className="w-full max-w-4xl mx-auto">
          <div className="relative">
            <div className={`flex items-center space-x-3 rounded-xl px-4 py-3 transition-colors duration-200 border ${
              darkMode 
                ? 'bg-gray-800 border-gray-600' 
                : 'bg-white border-gray-300'
            }`}>
              <div className="flex-1 min-h-[20px] max-h-[100px] overflow-y-auto">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSend();
                    }
                  }}
                  placeholder="Ask about Istanbul..."
                  className={`w-full bg-transparent border-0 outline-none focus:outline-none focus:ring-0 text-base resize-none transition-colors duration-200 ${
                    darkMode 
                      ? 'placeholder-gray-400 text-white' 
                      : 'placeholder-gray-500 text-gray-900'
                  }`}
                  disabled={loading}
                  autoComplete="off"
                />
              </div>
              <button 
                onClick={handleSend} 
                disabled={loading || !input.trim()}
                className={`p-2 rounded-lg transition-all duration-200 ${
                  darkMode 
                    ? 'bg-gradient-to-br from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-700 hover:via-indigo-700 hover:to-blue-700 disabled:from-gray-600 disabled:to-gray-600' 
                    : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 hover:from-blue-600 hover:via-indigo-600 hover:to-purple-600 disabled:from-gray-400 disabled:to-gray-400'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {loading ? (
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          <div className={`text-xs text-center mt-2 transition-colors duration-200 ${
            darkMode ? 'text-gray-500' : 'text-gray-500'
          }`}>
            Your AI-powered Istanbul guide
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chatbot
