import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import SearchBar from './components/SearchBar';
import Chat from './components/Chat';
import ResultCard from './components/ResultCard';
// import DebugInfo from './components/DebugInfo';
import { fetchResults, fetchStreamingResults } from './api/api';
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
    if (!query.trim()) return;
    setMessages([...messages, { user: 'You', text: query }]);
    setExpanded(true);
    const searchQuery = query;
    setQuery('');
    setTimeout(() => {
      document.getElementById('chat-animated-container')?.classList.add('expand-animate');
    }, 10);
    // Streaming KAM response
    let aiMessage = '';
    setMessages(msgs => [...msgs, { user: 'KAM', text: '' }]);
    try {
      await fetchStreamingResults(searchQuery, (chunk) => {
        aiMessage += chunk;
        setMessages(msgs => {
          const updated = [...msgs];
          // Find last KAM message and update its text
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].user === 'KAM') {
              updated[i] = { ...updated[i], text: aiMessage };
              break;
            }
          }
          return updated;
        });
      });
    } catch (err) {
      console.error('API Error:', err);
      setMessages(msgs => [...msgs, {
        user: 'KAM',
        text: `Sorry, I encountered an error connecting to the server: ${err.message}. Please make sure the backend is running and try again.`
      }]);
      setResults([]);
    }
  };

  const handleLogoClick = () => {
    // Reset chat state to go back to homepage
    setExpanded(false);
    setMessages([]);
    setResults([]);
    setQuery('');
  };

  return (
    <div style={{ width: '100vw', height: '100vh', minHeight: '100vh', background: 'none', display: 'flex', flexDirection: 'column' }}>
      {/* <DebugInfo /> */}

      {!expanded ? (
        <div style={{flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-start', width: '100vw', height: '100vh', paddingTop: '12vh'}}>
          <Link to="/" style={{textDecoration: 'none'}} onClick={handleLogoClick}>
            <div className={`chat-title logo-istanbul${expanded ? ' logo-move-top-left' : ''}`} id="logo-istanbul">
              <span className="logo-text">
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
          <div style={{width: '100%', maxWidth: 950, minWidth: 320, margin: '1rem auto 0', padding: '1rem'}}>
            <SearchBar
              value={query}
              onChange={e => setQuery(e.target.value)}
              onSubmit={handleSearch}
              placeholder="Welcome to Istanbul!"
            />
          </div>
        </div>
      ) : (
        <>
          <Link to="/" style={{textDecoration: 'none'}} onClick={handleLogoClick}>
            <div className={`chat-title logo-istanbul logo-move-top-left`} id="logo-istanbul">
              <span className="logo-text">
                A/<span style={{fontWeight: 400}}>STANBUL</span>
              </span>
            </div>
          </Link>
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', width: '100%', transition: 'all 0.4s', height: '100vh', paddingTop: '6rem', paddingBottom: '2rem' }}>
            <div style={{ width: '100%', maxWidth: 950, flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 12rem)', minHeight: '400px' }}>
              {/* Unified chat area and search bar */}
              <div className="chat-container" style={{display: 'flex', flexDirection: 'column', height: '100%', background: 'none', borderRadius: '1.5rem', boxShadow: '0 4px 24px 0 rgba(20, 20, 40, 0.18)', position: 'relative'}}>
                <div ref={chatScrollRef} className="chat-scroll-area" style={{flex: 1, overflowY: 'scroll', overflowX: 'hidden', marginBottom: '1rem', paddingBottom: '0.5rem', minHeight: 0, paddingTop: '0.5rem'}}>
                  <Chat messages={messages} />
                  {/* Remove or reduce margin below chat */}
                  <div style={{ marginTop: '0.5rem' }}>
                    {results.map((res, idx) => (
                      <ResultCard key={idx} title={res.title} description={res.description} />
                    ))}
                  </div>
                </div>
                <div style={{position: 'relative', bottom: 'auto', left: '0', right: '0', background: 'transparent', padding: '0.5rem 0.5rem 1rem 0.5rem', zIndex: 15}}>
                  <SearchBar
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    onSubmit={handleSearch}
                    placeholder=""
                  />
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
