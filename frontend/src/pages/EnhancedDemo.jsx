import React, { useState, useEffect } from 'react';
import '../App.css';

const EnhancedDemo = () => {
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId] = useState(`demo_${Date.now()}`);
  const [context, setContext] = useState(null);

  // Demo scenarios to showcase enhanced features
  const demoScenarios = [
    {
      title: "Context Awareness Test",
      steps: [
        "recommend restaurants in Kadikoy",
        "what about vegetarian options?",
        "how much should I tip?"
      ]
    },
    {
      title: "Typo Correction Test", 
      steps: [
        "restorant recomendations in galata",
        "musium near sultanahmet",
        "best atractions to visit"
      ]
    },
    {
      title: "Knowledge Base Test",
      steps: [
        "tell me about Ottoman history",
        "what's the etiquette for visiting mosques?",
        "Byzantine Constantinople"
      ]
    },
    {
      title: "Follow-up Questions Test",
      steps: [
        "places to visit in Beyoglu", 
        "what about similar areas?",
        "more recommendations?"
      ]
    }
  ];

  const fetchContext = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const cleanApiUrl = apiUrl.replace(/\/ai\/?$/, '');
      const response = await fetch(`${cleanApiUrl}/ai/context/${sessionId}`);
      const data = await response.json();
      setContext(data);
    } catch (error) {
      console.error('Error fetching context:', error);
    }
  };

  const sendMessage = async (message) => {
    const userMessage = { type: 'user', content: message, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const cleanApiUrl = apiUrl.replace(/\/ai\/?$/, '');
      const response = await fetch(`${cleanApiUrl}/ai`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query: message, 
          session_id: sessionId 
        })
      });

      const data = await response.json();
      const botMessage = { 
        type: 'bot', 
        content: data.message, 
        timestamp: new Date() 
      };
      
      setMessages(prev => [...prev, botMessage]);
      await fetchContext(); // Update context after each message
    } catch (error) {
      const errorMessage = { 
        type: 'error', 
        content: 'Failed to get response. Please try again.', 
        timestamp: new Date() 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (currentMessage.trim()) {
      sendMessage(currentMessage.trim());
      setCurrentMessage('');
    }
  };

  const runScenario = async (steps) => {
    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      await sendMessage(step);
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  };

  const clearChat = () => {
    setMessages([]);
    setContext(null);
  };

  useEffect(() => {
    fetchContext();
  }, []);

  return (
    <div className="chatbot-background min-h-screen p-4">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            🎯 Enhanced AI Istanbul Chatbot Demo
          </h1>
          <p className="text-lg text-gray-600">
            Showcasing improved context awareness, query understanding, knowledge scope, and follow-up handling
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Demo Scenarios */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              ◉ Test Scenarios
            </h2>
            <div className="space-y-4">
              {demoScenarios.map((scenario, index) => (
                <div key={index} className="border rounded-lg p-4 hover:bg-gray-50">
                  <h3 className="font-medium text-gray-700 mb-2">
                    {scenario.title}
                  </h3>
                  <div className="text-sm text-gray-600 mb-3">
                    {scenario.steps.map((step, stepIndex) => (
                      <div key={stepIndex} className="mb-1">
                        {stepIndex + 1}. "{step}"
                      </div>
                    ))}
                  </div>
                  <button
                    onClick={() => runScenario(scenario.steps)}
                    className="w-full bg-blue-500 text-white px-3 py-2 rounded-md hover:bg-blue-600 transition-colors text-sm"
                    disabled={isTyping}
                  >
                    Run Test
                  </button>
                </div>
              ))}
            </div>
            
            <button
              onClick={clearChat}
              className="w-full mt-4 bg-red-500 text-white px-3 py-2 rounded-md hover:bg-red-600 transition-colors"
            >
              Clear Chat
            </button>
          </div>

          {/* Chat Interface */}
          <div className="bg-white rounded-xl shadow-lg flex flex-col h-[600px]">
            <div className="p-4 border-b bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-t-xl">
              <h2 className="text-lg font-semibold">💬 Chat Interface</h2>
              <p className="text-sm opacity-90">Session: {sessionId.split('_')[1]}</p>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] p-3 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-blue-500 text-white rounded-br-none'
                        : message.type === 'error'
                        ? 'bg-red-100 text-red-800 rounded-bl-none'
                        : 'bg-gray-100 text-gray-800 rounded-bl-none'
                    }`}
                  >
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 p-3 rounded-lg rounded-bl-none">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <form onSubmit={handleSubmit} className="p-4 border-t">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={currentMessage}
                  onChange={(e) => setCurrentMessage(e.target.value)}
                  placeholder="Ask about Istanbul..."
                  className="flex-1 p-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  disabled={isTyping}
                />
                <button
                  type="submit"
                  disabled={isTyping || !currentMessage.trim()}
                  className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-300 transition-colors"
                >
                  Send
                </button>
              </div>
            </form>
          </div>

          {/* Context Display */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              ◆ Context & Memory
            </h2>
            
            {context && context.active ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-medium text-gray-700 mb-2">Previous Queries</h3>
                  <div className="bg-gray-50 p-3 rounded-lg text-sm">
                    {context.previous_queries.length > 0 ? (
                      context.previous_queries.map((query, index) => (
                        <div key={index} className="mb-1">• {query}</div>
                      ))
                    ) : (
                      <div className="text-gray-500">No previous queries</div>
                    )}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium text-gray-700 mb-2">Mentioned Places</h3>
                  <div className="bg-gray-50 p-3 rounded-lg text-sm">
                    {context.mentioned_places.length > 0 ? (
                      context.mentioned_places.map((place, index) => (
                        <span key={index} className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2 mb-1">
                          {place}
                        </span>
                      ))
                    ) : (
                      <div className="text-gray-500">No places mentioned yet</div>
                    )}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium text-gray-700 mb-2">User Preferences</h3>
                  <div className="bg-gray-50 p-3 rounded-lg text-sm">
                    {Object.keys(context.user_preferences).length > 0 ? (
                      Object.entries(context.user_preferences).map(([key, value]) => (
                        <div key={key} className="mb-1">
                          <span className="font-medium">{key}:</span> {value}
                        </div>
                      ))
                    ) : (
                      <div className="text-gray-500">No preferences detected</div>
                    )}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium text-gray-700 mb-2">Last Recommendation</h3>
                  <div className="bg-gray-50 p-3 rounded-lg text-sm">
                    {context.last_recommendation_type || 'None'}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium text-gray-700 mb-2">Conversation Topics</h3>
                  <div className="bg-gray-50 p-3 rounded-lg text-sm">
                    {context.conversation_topics.length > 0 ? (
                      context.conversation_topics.map((topic, index) => (
                        <span key={index} className="inline-block bg-green-100 text-green-800 px-2 py-1 rounded mr-2 mb-1">
                          {topic}
                        </span>
                      ))
                    ) : (
                      <div className="text-gray-500">No topics yet</div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500">
                <p>No active conversation context</p>
                <p className="text-sm mt-2">Start chatting to see context build up!</p>
              </div>
            )}
          </div>
        </div>

        {/* Enhancement Status */}
        <div className="mt-6 bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            ⚡ Enhancement Status
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">◆</div>
              <h3 className="font-medium text-green-800">Context Awareness</h3>
              <p className="text-sm text-green-600 mt-1">⭐⭐⭐⭐⭐</p>
              <p className="text-xs text-green-700 mt-2">Maintains conversation history and references previous interactions</p>
            </div>
            
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">🔍</div>
              <h3 className="font-medium text-blue-800">Query Understanding</h3>
              <p className="text-sm text-blue-600 mt-1">⭐⭐⭐⭐⭐</p>
              <p className="text-xs text-blue-700 mt-2">Corrects typos and understands context</p>
            </div>
            
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">📚</div>
              <h3 className="font-medium text-purple-800">Knowledge Scope</h3>
              <p className="text-sm text-purple-600 mt-1">⭐⭐⭐⭐⭐</p>
              <p className="text-xs text-purple-700 mt-2">Expanded knowledge of history, culture, and etiquette</p>
            </div>
            
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4 text-center">
              <div className="text-2xl mb-2">💬</div>
              <h3 className="font-medium text-orange-800">Follow-up Questions</h3>
              <p className="text-sm text-orange-600 mt-1">⭐⭐⭐⭐⭐</p>
              <p className="text-xs text-orange-700 mt-2">Handles conversational flow and related questions</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedDemo;
