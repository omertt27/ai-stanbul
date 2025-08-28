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
        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
        placeholder="Ask me anything!"
      />
      <button onClick={handleSend} disabled={loading}>
        {loading ? 'Thinking...' : 'Send'}
      </button>
    </div>
  )
}

export default Chatbot
