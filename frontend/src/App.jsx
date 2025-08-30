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
      {/* Navigation bar for main page */}
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '2rem', marginTop: '2.5rem', marginBottom: '2.5rem' }}>
        <a href="/about" style={{ color: '#818cf8', textDecoration: 'none', fontWeight: 600, fontSize: '1.1rem', borderBottom: '2px solid #818cf8', padding: '0.2rem 0.5rem', borderRadius: '0.5rem', background: 'rgba(35,38,58,0.7)', marginRight: '0.5rem' }}>About</a>
        <a href="/sources" style={{ color: '#818cf8', textDecoration: 'none', fontWeight: 600, fontSize: '1.1rem', borderBottom: '2px solid #818cf8', padding: '0.2rem 0.5rem', borderRadius: '0.5rem', background: 'rgba(35,38,58,0.7)', marginRight: '0.5rem' }}>Sources</a>
        <a href="/donate" style={{ color: '#818cf8', textDecoration: 'none', fontWeight: 600, fontSize: '1.1rem', borderBottom: '2px solid #818cf8', padding: '0.2rem 0.5rem', borderRadius: '0.5rem', background: 'rgba(35,38,58,0.7)' }}>Donate</a>
      </div>
      {!expanded ? (
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '100vw', height: '100vh'}}>
          <div className={`chat-title logo-istanbul${expanded ? ' logo-move-top-left' : ''}`} id="logo-istanbul">
            <span className="logo-text">
              A/<span style={{fontWeight: 400}}>STANBUL</span>
            </span>
          </div>
          <div style={{width: '100%', maxWidth: 520, minWidth: 320, margin: '0 auto', boxShadow: '0 4px 32px 0 #0002', borderRadius: '1.5rem', background: 'rgba(35,38,58,0.98)', padding: '1.75rem 1.5rem 1.25rem 1.5rem', transition: 'max-width 0.3s cubic-bezier(.4,2,.6,1)'}}>
            <SearchBar value={query} onChange={e => setQuery(e.target.value)} onSubmit={handleSearch} />
          </div>
        </div>
      ) : (
        <>
          <div className={`chat-title logo-istanbul logo-move-top-left`} id="logo-istanbul">
            <span className="logo-text">
              A/<span style={{fontWeight: 400}}>STANBUL</span>
            </span>
          </div>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', transition: 'all 0.4s', height: '100vh' }}>
            <div style={{ width: '100%', maxWidth: 850, flex: 1, display: 'flex', flexDirection: 'column', height: '100vh' }}>
              {/* Unified chat area and search bar */}
              <div style={{display: 'flex', flexDirection: 'column', height: '100%', background: 'none', borderRadius: '1.5rem', boxShadow: '0 4px 24px 0 rgba(20, 20, 40, 0.18)'}}>
                <div ref={chatScrollRef} style={{flex: 1, overflowY: 'auto', marginBottom: 0, paddingBottom: 0, minHeight: 0, height: '100%'}}>
                  <Chat messages={messages} />
                  {/* Remove or reduce margin below chat */}
                  <div style={{ marginTop: '0.5rem' }}>
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
          </div>
        </>
      )}
    </div>
  );
};

export default App;
