import { useState } from 'react';

function Chatbot() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSend = async () => {
    console.log('handleSend called with input:', input); // Debug log
    if (!input.trim()) return;

    const userInput = input.trim(); // Store the input value before clearing it
    console.log('userInput after trim:', userInput); // Debug log
    const userMessage = { role: 'user', content: userInput };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput(''); // Clear input after storing the value
    setLoading(true);

    try {
      console.log('Sending request to:', import.meta.env.VITE_API_URL);
      console.log('With data:', { user_input: userInput });
      console.log('JSON body:', JSON.stringify({ user_input: userInput }));
      
      const response = await fetch(import.meta.env.VITE_API_URL + `?t=${Date.now()}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Cache-Control': 'no-cache'
        },
        body: JSON.stringify({ user_input: userInput }),
      });
      console.log('Raw response:', response);
      let data;
      try {
        data = await response.json();
        console.log('Parsed data:', data);
      } catch (jsonErr) {
        console.error('Failed to parse JSON:', jsonErr);
        setMessages([
          ...newMessages,
          { role: 'assistant', content: 'Sorry, I could not understand the server response.' }
        ]);
        return;
      }
      if (data && typeof data.message === 'string') {
        const botMessage = { role: 'assistant', content: data.message };
        setMessages([...newMessages, botMessage]);
      } else {
        setMessages([
          ...newMessages,
          { role: 'assistant', content: 'Sorry, I did not get a valid answer from the AI.' }
        ]);
        console.error('Unexpected data format:', data);
      }
    } catch (error) {
      console.error('Network or fetch error:', error);
      setMessages([
        ...newMessages,
        { role: 'assistant', content: 'Sorry, there was a network error. Please try again.' }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header with mixed styling */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-4">
            <span className="text-blue-600">Chat</span>
            <span className="text-purple-600">AI</span>
          </h1>
          <p className="text-gray-600">Powered by advanced AI models</p>
        </div>

        {/* Chat Messages */}
        <div className="space-y-6 mb-6">
          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'user' ? (
                // User message - Gemini style
                <div className="bg-blue-500 text-white rounded-2xl px-6 py-4 max-w-xs shadow-lg">
                  <p className="text-sm font-medium">{msg.content}</p>
                </div>
              ) : (
                // AI message - Mixed GPT/Gemini style
                <div className="bg-white rounded-2xl p-6 max-w-2xl shadow-lg border border-gray-200">
                  <div className="flex items-start space-x-4">
                    <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center flex-shrink-0">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm text-gray-500 mb-2 font-medium">AI Assistant</div>
                      <div className="text-gray-800 leading-relaxed whitespace-pre-wrap">
                        {msg.content}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {loading && (
            <div className="flex justify-start">
              <div className="bg-white rounded-2xl p-6 max-w-2xl shadow-lg border border-gray-200">
                <div className="flex items-start space-x-4">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center animate-pulse">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <div className="text-sm text-gray-500 mb-2 font-medium">AI Assistant</div>
                    <div className="animate-pulse space-y-2">
                      <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                      <div className="h-4 bg-gray-300 rounded w-1/2"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Area - GPT style */}
        <div className="sticky bottom-0 bg-white rounded-2xl shadow-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Message AI..."
              className="flex-1 border-none outline-none text-lg bg-transparent resize-none"
              disabled={loading}
            />
            <button 
              onClick={handleSend} 
              disabled={loading || !input.trim()}
              className="p-3 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 text-white hover:from-blue-600 hover:to-purple-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-all duration-200"
            >
              {loading ? (
                <svg className="animate-spin w-5 h-5" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              ) : (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chatbot
