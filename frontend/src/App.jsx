import React, { useState, useEffect, useRef } from 'react';
import SearchBar from './components/SearchBar';
import Chat from './components/Chat';
import ResultCard from './components/ResultCard';
import { fetchResults } from './api/api';
import './App.css';

const App = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [messages, setMessages] = useState([]);
  const [expanded, setExpanded] = useState(false);
  const chatScrollRef = useRef(null);

  useEffect(() => {
    const audio = new Audio('/welcome_baskan.mp3');
    audio.volume = 0.5;
    audio.play().catch(() => {/* ignore audio errors */});
  }, []);

  useEffect(() => {
    if (expanded && chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, expanded]);

  const handleSearch = async (e) => {
    e.preventDefault();
    setMessages([...messages, { user: 'You', text: query }]);
    setExpanded(true);
    setTimeout(() => {
      document.getElementById('chat-animated-container')?.classList.add('expand-animate');
    }, 10);
    try {
      const data = await fetchResults(query);
      if (data && data.message) {
        setMessages(msgs => [...msgs, { user: 'AI', text: data.message }]);
      }
      setResults(data.results || []);
    } catch (err) {
      setResults([]);
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', minHeight: '100vh', background: 'none', display: 'flex', flexDirection: 'column' }}>
      {!expanded ? (
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100vw', height: '100vh'}}>
          <div className="chat-title" style={{marginBottom: '2.5rem', letterSpacing: '0.1em'}}>
            <span style={{fontWeight: 900, fontSize: '2.5rem', letterSpacing: '0.15em', background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'}}>
              A/<span style={{fontWeight: 400}}>STANBUL</span>
            </span>
          </div>
          <div style={{width: '100%', maxWidth: 400}}>
            <SearchBar value={query} onChange={e => setQuery(e.target.value)} onSubmit={handleSearch} />
          </div>
        </div>
      ) : (
        <>
          <div className="chat-title" style={{paddingTop: '2.5rem', letterSpacing: '0.1em'}}>
            <span style={{fontWeight: 900, fontSize: '2.5rem', letterSpacing: '0.15em', background: 'linear-gradient(90deg, #818cf8 0%, #6366f1 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'}}>
              A/<span style={{fontWeight: 400}}>STANBUL</span>
            </span>
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', transition: 'all 0.4s', height: '100%' }}>
            <div style={{ width: '100%', maxWidth: 900, flex: 1, display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div ref={chatScrollRef} style={{flex: 1, overflowY: 'auto', marginBottom: 0, paddingBottom: 16, minHeight: 0}}>
                <Chat messages={messages} />
                <div style={{ marginTop: '1.5rem' }}>
                  {results.map((res, idx) => (
                    <ResultCard key={idx} title={res.title} description={res.description} />
                  ))}
                </div>
              </div>
              <div style={{width: '100%', position: 'sticky', bottom: 0, background: 'transparent', zIndex: 10}}>
                <SearchBar value={query} onChange={e => setQuery(e.target.value)} onSubmit={handleSearch} />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default App;
