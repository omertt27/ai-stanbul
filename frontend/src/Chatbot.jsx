import { useState, useEffect } from 'react';

function Chatbot() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [darkMode, setDarkMode] = useState(false)

  // Toggle dark mode
  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
  }

  // Apply dark mode class to document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

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
    <div className={`flex flex-col h-screen w-full max-w-2xl ml-auto transition-colors duration-200 ${darkMode ? 'bg-gray-900' : 'bg-white'}`}>
      {/* Header - ChatGPT style with dark mode */}
      <div className={`flex items-center justify-between px-4 py-3 border-b border-l transition-colors duration-200 ${darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-white'}`}>
        <div className="flex items-center space-x-3">
          <div className={`w-8 h-8 rounded-sm flex items-center justify-center transition-colors duration-200 ${darkMode ? 'bg-white' : 'bg-black'}`}>
            <svg className={`w-5 h-5 transition-colors duration-200 ${darkMode ? 'text-black' : 'text-white'}`} fill="currentColor" viewBox="0 0 24 24">
              <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
            </svg>
          </div>
          <h1 className={`text-lg font-semibold transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>AISTANBUL</h1>
        </div>
        <div className="flex items-center space-x-2">
          {/* Dark mode toggle */}
          <button 
            onClick={toggleDarkMode}
            className={`p-2 rounded-md transition-colors duration-200 ${darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-100'}`}
          >
            {darkMode ? (
              <svg className="w-5 h-5 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            )}
          </button>
          <button className={`p-2 rounded-md transition-colors duration-200 ${darkMode ? 'hover:bg-gray-800' : 'hover:bg-gray-100'}`}>
            <svg className={`w-5 h-5 transition-colors duration-200 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 5v.01M12 12v.01M12 19v.01M12 6a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2zm0 7a1 1 0 110-2 1 1 0 010 2z" />
            </svg>
          </button>
        </div>
      </div>

      {/* Chat Messages Container - Made larger */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center px-4">
            <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-4 transition-colors duration-200 ${darkMode ? 'bg-white' : 'bg-black'}`}>
              <svg className={`w-8 h-8 transition-colors duration-200 ${darkMode ? 'text-black' : 'text-white'}`} fill="currentColor" viewBox="0 0 24 24">
                <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
              </svg>
            </div>
            <h2 className={`text-2xl font-semibold mb-3 transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>How can I help you today?</h2>
            <p className={`text-center max-w-md text-base leading-relaxed transition-colors duration-200 ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>I'm your Istanbul travel assistant. Ask me about restaurants, attractions, culture, or anything about Istanbul!</p>
            
            {/* Sample questions */}
            <div className="mt-6 grid grid-cols-1 gap-3 max-w-lg w-full px-4">
              <div className={`p-3 rounded-lg border transition-all duration-200 cursor-pointer hover:shadow-md ${darkMode ? 'bg-gray-800 border-gray-600 hover:bg-gray-700' : 'bg-gray-50 border-gray-200 hover:bg-gray-100'}`}>
                <div className={`font-medium mb-1 text-sm transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>🍽️ Restaurant Recommendations</div>
                <div className={`text-xs transition-colors duration-200 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Find the best Turkish restaurants in Istanbul</div>
              </div>
              <div className={`p-3 rounded-lg border transition-all duration-200 cursor-pointer hover:shadow-md ${darkMode ? 'bg-gray-800 border-gray-600 hover:bg-gray-700' : 'bg-gray-50 border-gray-200 hover:bg-gray-100'}`}>
                <div className={`font-medium mb-1 text-sm transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>🏛️ Tourist Attractions</div>
                <div className={`text-xs transition-colors duration-200 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Discover Istanbul's historical sites and museums</div>
              </div>
              <div className={`p-3 rounded-lg border transition-all duration-200 cursor-pointer hover:shadow-md ${darkMode ? 'bg-gray-800 border-gray-600 hover:bg-gray-700' : 'bg-gray-50 border-gray-200 hover:bg-gray-100'}`}>
                <div className={`font-medium mb-1 text-sm transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>🚗 Transportation Tips</div>
                <div className={`text-xs transition-colors duration-200 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Navigate Istanbul like a local</div>
              </div>
              <div className={`p-3 rounded-lg border transition-all duration-200 cursor-pointer hover:shadow-md ${darkMode ? 'bg-gray-800 border-gray-600 hover:bg-gray-700' : 'bg-gray-50 border-gray-200 hover:bg-gray-100'}`}>
                <div className={`font-medium mb-1 text-sm transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>🎭 Culture & Events</div>
                <div className={`text-xs transition-colors duration-200 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Experience Istanbul's rich cultural scene</div>
              </div>
            </div>
          </div>
        )}
        
        <div className="max-w-full mx-auto px-4">
          {messages.map((msg, index) => (
            <div key={index} className="group py-6">
              <div className="flex items-start space-x-4">
                {msg.role === 'user' ? (
                  <>
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-sm font-semibold mb-2 transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>You</div>
                      <div className={`text-sm whitespace-pre-wrap transition-colors duration-200 ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        {msg.content}
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${darkMode ? 'bg-white' : 'bg-black'}`}>
                      <svg className={`w-4 h-4 transition-colors duration-200 ${darkMode ? 'text-black' : 'text-white'}`} fill="currentColor" viewBox="0 0 24 24">
                        <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                      </svg>
                    </div>
                    <div className="flex-1">
                      <div className={`text-sm font-semibold mb-2 transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>KAM</div>
                      <div className={`text-sm whitespace-pre-wrap leading-relaxed transition-colors duration-200 ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        {msg.content}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          ))}
          
          {loading && (
            <div className="group py-6">
              <div className="flex items-start space-x-4">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 transition-colors duration-200 ${darkMode ? 'bg-white' : 'bg-black'}`}>
                  <svg className={`w-4 h-4 transition-colors duration-200 ${darkMode ? 'text-black' : 'text-white'}`} fill="currentColor" viewBox="0 0 24 24">
                    <path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91A6.046 6.046 0 0 0 17.094 2H6.906a6.046 6.046 0 0 0-4.672 2.91 5.985 5.985 0 0 0-.516 4.911L3.75 18.094A2.003 2.003 0 0 0 5.734 20h12.532a2.003 2.003 0 0 0 1.984-1.906l2.032-8.273Z"/>
                  </svg>
                </div>
                <div className="flex-1">
                  <div className={`text-sm font-semibold mb-2 transition-colors duration-200 ${darkMode ? 'text-white' : 'text-gray-900'}`}>KAM</div>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full animate-bounce transition-colors duration-200 ${darkMode ? 'bg-gray-500' : 'bg-gray-400'}`}></div>
                    <div className={`w-2 h-2 rounded-full animate-bounce transition-colors duration-200 ${darkMode ? 'bg-gray-500' : 'bg-gray-400'}`} style={{animationDelay: '0.1s'}}></div>
                    <div className={`w-2 h-2 rounded-full animate-bounce transition-colors duration-200 ${darkMode ? 'bg-gray-500' : 'bg-gray-400'}`} style={{animationDelay: '0.2s'}}></div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Input Area - Fixed at bottom, narrower */}
      <div className={`border-t border-l p-4 transition-colors duration-200 ${darkMode ? 'border-gray-700 bg-gray-900' : 'border-gray-200 bg-white'}`}>
        <div className="w-full">
          <div className="relative">
            <div className={`flex items-center space-x-3 rounded-2xl px-4 py-2 transition-colors duration-200 ${darkMode ? 'bg-gray-800' : 'bg-gray-100'}`}>
              <div className="flex-1 min-h-[20px] max-h-[120px] overflow-y-auto">
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
                  placeholder="Message KAM"
                  className={`w-full bg-transparent border-0 outline-none focus:outline-none focus:ring-0 text-base resize-none transition-colors duration-200 ${darkMode ? 'placeholder-gray-400 text-white' : 'placeholder-gray-500 text-gray-900'}`}
                  disabled={loading}
                  autoComplete="off"
                />
              </div>
              <button 
                onClick={handleSend} 
                disabled={loading || !input.trim()}
                className={`p-2 rounded-full transition-all duration-200 ${darkMode ? 'bg-gray-700 hover:bg-gray-600 disabled:bg-gray-700' : 'bg-gray-200 hover:bg-gray-300 disabled:bg-gray-200'} disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {loading ? (
                  <div className={`w-4 h-4 border-2 rounded-full animate-spin transition-colors duration-200 ${darkMode ? 'border-gray-400 border-t-transparent' : 'border-gray-600 border-t-transparent'}`}></div>
                ) : (
                  <svg className={`w-4 h-4 transition-colors duration-200 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          <div className={`text-xs text-center mt-2 transition-colors duration-200 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            KAM can make mistakes. Check important info.
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chatbot
