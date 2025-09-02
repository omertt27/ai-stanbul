import React, { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import SearchBar from './components/SearchBar';
import Chat from './components/Chat';
import ResultCard from './components/ResultCard';
import ConnectionStatus from './components/ConnectionStatus';
// import DebugInfo from './components/DebugInfo';
import { fetchResults, fetchStreamingResults, fetchRestaurantRecommendations, extractLocationFromQuery } from './api/api';
import './App.css';

// Simplified helper function - only catch very explicit restaurant+location requests
const isExplicitRestaurantRequest = (userInput) => {
  console.log('🔍 App.jsx: Checking for explicit restaurant request:', userInput);
  const input = userInput.toLowerCase();
  
  // Only intercept very specific restaurant requests with location
  // These are the ONLY phrases that trigger the restaurant API
  const explicitRestaurantRequests = [
    'restaurants in',        // "restaurants in Beyoglu"
    'where to eat in',       // "where to eat in Sultanahmet" 
    'restaurant recommendations for', // "restaurant recommendations for Taksim"
    'good restaurants in',   // "good restaurants in Galata"
    'best restaurants in',   // "best restaurants in Kadikoy"
    'restaurants near',      // "restaurants near Taksim Square"
    'where to eat near',     // "where to eat near Galata Tower"
    'dining in',            // "dining in Beyoglu"
    'food in'               // "food in Sultanahmet"
  ];
  
  const isExplicit = explicitRestaurantRequests.some(keyword => input.includes(keyword));
  console.log('🔍 App.jsx: Explicit restaurant detection result:', isExplicit);
  return isExplicit;
};

// Helper function to format restaurant recommendations
const formatRestaurantRecommendations = (restaurants) => {
  console.log('App.jsx: formatRestaurantRecommendations called with:', { restaurants, count: restaurants?.length });
  
  if (!restaurants || restaurants.length === 0) {
    console.log('App.jsx: No restaurants found, returning error message');
    return "Sorry, I couldn't find any restaurants matching your request in Istanbul.";
  }

  let formattedResponse = "🍽️ Here are 4 great restaurant recommendations for you:\n\n";
  
  restaurants.slice(0, 4).forEach((restaurant, index) => {
    const name = restaurant.name || 'Unknown Restaurant';
    const rating = restaurant.rating ? `⭐ ${restaurant.rating}` : '';
    const address = restaurant.address || restaurant.vicinity || '';
    const description = restaurant.description || 'A popular dining spot in Istanbul.';
    
    formattedResponse += `${index + 1}. ${name}\n`;
    if (rating) formattedResponse += `${rating}\n`;
    if (address) formattedResponse += `📍 ${address}\n`;
    formattedResponse += `${description}\n\n`;
  });

  formattedResponse += "Would you like more details about any of these restaurants?";
  
  return formattedResponse;
};

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

    // Check if user is making an explicit restaurant request (only very specific cases)
    if (isExplicitRestaurantRequest(searchQuery)) {
      console.log('🍽️ App.jsx: Detected restaurant request, fetching recommendations...');
      setMessages(msgs => [...msgs, { user: 'KAM', text: '' }]);
      
      try {
        const restaurantData = await fetchRestaurantRecommendations(searchQuery);
        console.log('🍽️ App.jsx: Restaurant API response:', restaurantData);
        const formattedResponse = formatRestaurantRecommendations(restaurantData.restaurants);
        console.log('🍽️ App.jsx: Formatted response:', formattedResponse);
        
        setMessages(msgs => {
          const updated = [...msgs];
          // Find last KAM message and update its text
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].user === 'KAM') {
              updated[i] = { ...updated[i], text: formattedResponse };
              break;
            }
          }
          return updated;
        });
        return; // Exit early, don't do regular AI chat
      } catch (error) {
        console.error('🍽️ App.jsx: Restaurant recommendation error:', error);
        setMessages(msgs => {
          const updated = [...msgs];
          for (let i = updated.length - 1; i >= 0; i--) {
            if (updated[i].user === 'KAM') {
              updated[i] = { ...updated[i], text: `Sorry, I had trouble getting restaurant recommendations: ${error.message}. Let me try a different approach.` };
              break;
            }
          }
          return updated;
        });
        return;
      }
    }

    // Regular streaming KAM response for non-restaurant queries
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
      {/* Connection Status Indicator */}
      <ConnectionStatus />
      
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
