
import { useState } from 'react';

function Chatbot() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(import.meta.env.VITE_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_input: input }),
      });
      const data = await response.json();
      let botContent = '';
      if (data.results) {
        botContent = Array.isArray(data.results) ? data.results.join(', ') : String(data.results);
      } else if (data.message) {
        botContent = data.message;
      } else if (data.error) {
        botContent = data.error;
      } else {
        botContent = 'No response';
      }
      const botMessage = { role: 'assistant', content: botContent };
      setMessages([...newMessages, botMessage]);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  }  

  return (
    <div className="chatbot">
      <h2>AI-Stanbul Chatbot</h2>
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask me anything about Istanbul..."
      />
      <button onClick={handleSend} disabled={loading}>
        {loading ? 'Thinking...' : 'Send'}
      </button>
    </div>
  )
}

export default Chatbot
