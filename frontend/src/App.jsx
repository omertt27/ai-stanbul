import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
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
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', width: '100vw', height: '100vh', paddingTop: '25vh'}}>
          <Link to="/" style={{textDecoration: 'none'}}>
            <div className={`chat-title logo-istanbul${expanded ? ' logo-move-top-left' : ''}`} id="logo-istanbul">
              <span className="logo-text">
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
          <div style={{width: '100%', maxWidth: 950, minWidth: 320, margin: '0 auto', padding: '1rem'}}>
            <SearchBar value={query} onChange={e => setQuery(e.target.value)} onSubmit={handleSearch} />
          </div>
        </div>
      ) : (
        <>
          <Link to="/" style={{textDecoration: 'none'}}>
            <div className={`chat-title logo-istanbul logo-move-top-left`} id="logo-istanbul">
              <span className="logo-text">
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', transition: 'all 0.4s', height: '100vh', paddingTop: '8rem' }}>
            <div style={{ width: '100%', maxWidth: 950, flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 4rem)' }}>
              {/* Unified chat area and search bar */}
              <div style={{display: 'flex', flexDirection: 'column', height: '100%', background: 'none', borderRadius: '1.5rem', boxShadow: '0 4px 24px 0 rgba(20, 20, 40, 0.18)'}}>
                <div ref={chatScrollRef} style={{flex: 1, overflowY: 'scroll', overflowX: 'hidden', marginBottom: 0, paddingBottom: '2rem', minHeight: 0, paddingTop: '0.5rem'}}>
                  <Chat messages={messages} />
                  {/* Remove or reduce margin below chat */}
                  <div style={{ marginTop: '0.5rem' }}>
                    {results.map((res, idx) => (
                      <ResultCard key={idx} title={res.title} description={res.description} />
                    ))}
                  </div>
                </div>
                <div style={{width: '100%', position: 'sticky', bottom: '1rem', background: 'transparent', zIndex: 10}}>
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
