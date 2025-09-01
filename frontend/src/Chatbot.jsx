import { useState, useEffect } from 'react';
import { fetchStreamingResults, fetchRestaurantRecommendations, extractLocationFromQuery } from './api/api';

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
  if (!restaurants || restaurants.length === 0) {
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

// Helper function to detect if user is asking for restaurant recommendations
const isRestaurantAdviceRequest = (userInput) => {
  const input = userInput.toLowerCase();
  const restaurantKeywords = [
    'restaurant', 'restaurants', 'food', 'eat', 'dining', 'meal',
    'recommend', 'suggestion', 'advice', 'where to eat',
    'good food', 'best restaurant', 'restaurant recommendation',
    'places to eat', 'food recommendation', 'cuisine', 'turkish cuisine',
    'restaurant advice', 'give me restaurant', 'find restaurant',
    'show me restaurant', 'good restaurants', 'best places to eat',
    'restaurants in', 'food in', 'dining in', 'eat in', 'places to eat in'
  ];
  
  return restaurantKeywords.some(keyword => input.includes(keyword));
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
    if (isRestaurantAdviceRequest(userInput)) {
      try {
        console.log('Detected restaurant advice request, fetching recommendations...');
        const restaurantData = await fetchRestaurantRecommendations(userInput);
        const formattedResponse = formatRestaurantRecommendations(restaurantData.restaurants);
        
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: formattedResponse }
        ]);
        setLoading(false);
        return;
      } catch (error) {
        console.error('Restaurant recommendation error:', error);
        // Fall back to regular AI response if restaurant API fails
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
                }`}>What are the best cultural experiences and activities in Istanbul?</div>
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
                        : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500'
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
                    : 'bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500'
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
