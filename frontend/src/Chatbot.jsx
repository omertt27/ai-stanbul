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
    <div className="chatbot">
      <h2>AI Chatbot</h2>
      <div className="chat-window">
        {messages.map((msg, index) => (
          <div key={index} className="mb-2">
            <span className="font-semibold">{msg.role === 'user' ? 'You' : 'AI'}: </span>
            <span>{msg.content}</span>
          </div>
        ))}
      </div>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask me anything!"
      />
      <button onClick={handleSend} disabled={loading}>
        {loading ? 'Thinking...' : 'Send'}
      </button>
    </div>
  )
}

export default Chatbot
