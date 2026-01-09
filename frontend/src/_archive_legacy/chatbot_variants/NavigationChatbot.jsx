import { useState, useEffect, useRef } from 'react';
import LeafletNavigationMap from './components/LeafletNavigationMap';
import { chatNavigation, getTurnByTurnDirections, formatDistance, formatDuration } from './api/navigationApi';
import GPSLocationService from './services/gpsLocationService';
import './App.css';

/**
 * NavigationChatbot - Enhanced chatbot with integrated navigation
 * Features:
 * - AI conversational navigation
 * - Live map with route rendering
 * - Turn-by-turn directions
 * - POI recommendations
 * - GPS location tracking
 */
function NavigationChatbot() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [routeData, setRouteData] = useState(null);
  const [pois, setPois] = useState([]);
  const [userLocation, setUserLocation] = useState(null);
  const [showMap, setShowMap] = useState(false);
  const [turnByTurn, setTurnByTurn] = useState([]);
  const [darkMode, setDarkMode] = useState(false);
  const messagesEndRef = useRef(null);
  const gpsService = useRef(new GPSLocationService());

  // Initialize GPS tracking
  useEffect(() => {
    const initGPS = async () => {
      try {
        const location = await gpsService.current.requestLocationPermission();
        setUserLocation({
          latitude: location.lat,
          longitude: location.lng
        });
        
        // Start continuous tracking
        gpsService.current.startLocationTracking();
        gpsService.current.onLocationUpdate((newLocation) => {
          setUserLocation({
            latitude: newLocation.lat,
            longitude: newLocation.lng
          });
        });
      } catch (error) {
        console.warn('GPS not available:', error.message);
      }
    };

    initGPS();

    return () => {
      gpsService.current.stopLocationTracking();
    };
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const userInput = input.trim();
    if (!userInput) return;

    const userMessage = { role: 'user', content: userInput };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      // Call navigation API with user context
      const userContext = userLocation ? {
        current_location: {
          latitude: userLocation.latitude,
          longitude: userLocation.longitude
        }
      } : {};

      const response = await chatNavigation(userInput, userContext);
      
      // Add assistant message
      const assistantMessage = {
        role: 'assistant',
        content: response.response,
        metadata: response.metadata
      };
      
      setMessages([...newMessages, assistantMessage]);

      // Process navigation metadata
      if (response.metadata) {
        // Update route data
        if (response.metadata.route_data) {
          const route = response.metadata.route_data;
          setRouteData(route);
          setShowMap(true);
          
          // Extract turn-by-turn directions
          const directions = getTurnByTurnDirections(route);
          setTurnByTurn(directions);
        }

        // Update POI recommendations
        if (response.metadata.poi_recommendations) {
          setPois(response.metadata.poi_recommendations);
          setShowMap(true);
        }

        // Handle other metadata
        if (response.metadata.nearby_pois) {
          setPois(response.metadata.nearby_pois);
          setShowMap(true);
        }
      }
      
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message}`
      };
      setMessages([...newMessages, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearRoute = () => {
    setRouteData(null);
    setTurnByTurn([]);
    setPois([]);
  };

  return (
    <div className={`navigation-chatbot ${darkMode ? 'dark' : ''}`}>
      <div className="container" style={{ maxWidth: '1400px', margin: '0 auto', padding: '20px' }}>
        {/* Header */}
        <div className="header" style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '20px',
          padding: '15px',
          background: darkMode ? '#1f2937' : 'white',
          borderRadius: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
        }}>
          <h1 style={{ margin: 0, fontSize: '24px', fontWeight: 'bold' }}>
            🗺️ AI Istanbul Navigation
          </h1>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <button
              onClick={() => setShowMap(!showMap)}
              style={{
                padding: '8px 16px',
                borderRadius: '8px',
                border: 'none',
                background: showMap ? '#2196F3' : '#e5e7eb',
                color: showMap ? 'white' : '#374151',
                cursor: 'pointer',
                fontWeight: '500'
              }}
            >
              {showMap ? '🗺️ Map ON' : '🗺️ Map OFF'}
            </button>
            <button
              onClick={() => setDarkMode(!darkMode)}
              style={{
                padding: '8px 16px',
                borderRadius: '8px',
                border: 'none',
                background: '#374151',
                color: 'white',
                cursor: 'pointer'
              }}
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: showMap ? '1fr 1fr' : '1fr', gap: '20px' }}>
          {/* Chat Panel */}
          <div style={{
            background: darkMode ? '#1f2937' : 'white',
            borderRadius: '12px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            display: 'flex',
            flexDirection: 'column',
            height: '700px'
          }}>
            {/* Messages */}
            <div style={{ 
              flex: 1, 
              overflowY: 'auto', 
              padding: '20px',
              display: 'flex',
              flexDirection: 'column',
              gap: '15px'
            }}>
              {messages.length === 0 && (
                <div style={{ 
                  textAlign: 'center', 
                  padding: '40px 20px',
                  color: darkMode ? '#9ca3af' : '#6b7280'
                }}>
                  <h2 style={{ fontSize: '20px', marginBottom: '10px' }}>
                    Welcome to AI Istanbul Navigation!
                  </h2>
                  <p>Ask me for directions, route suggestions, or nearby places.</p>
                  <div style={{ marginTop: '20px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <button
                      onClick={() => setInput('How do I get to Sultanahmet from Taksim?')}
                      style={{
                        padding: '12px',
                        borderRadius: '8px',
                        border: '1px solid #e5e7eb',
                        background: darkMode ? '#374151' : '#f9fafb',
                        cursor: 'pointer'
                      }}
                    >
                      "How do I get to Sultanahmet from Taksim?"
                    </button>
                    <button
                      onClick={() => setInput('Show me restaurants near Galata Tower')}
                      style={{
                        padding: '12px',
                        borderRadius: '8px',
                        border: '1px solid #e5e7eb',
                        background: darkMode ? '#374151' : '#f9fafb',
                        cursor: 'pointer'
                      }}
                    >
                      "Show me restaurants near Galata Tower"
                    </button>
                    <button
                      onClick={() => setInput('Plan a walking tour of Beyoğlu')}
                      style={{
                        padding: '12px',
                        borderRadius: '8px',
                        border: '1px solid #e5e7eb',
                        background: darkMode ? '#374151' : '#f9fafb',
                        cursor: 'pointer'
                      }}
                    >
                      "Plan a walking tour of Beyoğlu"
                    </button>
                  </div>
                </div>
              )}

              {messages.map((msg, index) => (
                <div
                  key={index}
                  style={{
                    display: 'flex',
                    justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start'
                  }}
                >
                  <div
                    style={{
                      maxWidth: '80%',
                      padding: '12px 16px',
                      borderRadius: '12px',
                      background: msg.role === 'user' 
                        ? '#2196F3' 
                        : darkMode ? '#374151' : '#f3f4f6',
                      color: msg.role === 'user' ? 'white' : darkMode ? '#e5e7eb' : '#1f2937'
                    }}
                  >
                    <div style={{ whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>
                      {msg.content}
                    </div>
                    
                    {/* Show route summary in message */}
                    {msg.metadata?.route_data && (
                      <div style={{ 
                        marginTop: '10px', 
                        paddingTop: '10px', 
                        borderTop: '1px solid rgba(255,255,255,0.2)'
                      }}>
                        <div style={{ fontSize: '13px', opacity: 0.9 }}>
                          📍 Distance: {formatDistance(msg.metadata.route_data.distance)}
                          <br />
                          ⏱️ Duration: {formatDuration(msg.metadata.route_data.duration)}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {loading && (
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <div className="loading-dot" />
                  <div className="loading-dot" style={{ animationDelay: '0.2s' }} />
                  <div className="loading-dot" style={{ animationDelay: '0.4s' }} />
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div style={{ 
              padding: '15px', 
              borderTop: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`
            }}>
              <div style={{ display: 'flex', gap: '10px' }}>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask for directions or places..."
                  disabled={loading}
                  style={{
                    flex: 1,
                    padding: '12px 16px',
                    borderRadius: '8px',
                    border: `1px solid ${darkMode ? '#374151' : '#e5e7eb'}`,
                    background: darkMode ? '#374151' : 'white',
                    color: darkMode ? 'white' : 'black',
                    outline: 'none'
                  }}
                />
                <button
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                  style={{
                    padding: '12px 24px',
                    borderRadius: '8px',
                    border: 'none',
                    background: loading || !input.trim() ? '#9ca3af' : '#2196F3',
                    color: 'white',
                    cursor: loading || !input.trim() ? 'not-allowed' : 'pointer',
                    fontWeight: '500'
                  }}
                >
                  {loading ? '...' : 'Send'}
                </button>
              </div>
            </div>
          </div>

          {/* Map & Navigation Panel */}
          {showMap && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
              {/* Map */}
              <div style={{
                background: darkMode ? '#1f2937' : 'white',
                borderRadius: '12px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',              overflow: 'hidden'
            }}>
                <LeafletNavigationMap
                  routeData={routeData}
                  pois={pois}
                  userLocation={userLocation}
                  height="400px"
                />
              </div>

              {/* Turn-by-Turn Directions */}
              {turnByTurn.length > 0 && (
                <div style={{
                  background: darkMode ? '#1f2937' : 'white',
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  padding: '20px',
                  maxHeight: '270px',
                  overflowY: 'auto'
                }}>
                  <div style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    marginBottom: '15px'
                  }}>
                    <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 'bold' }}>
                      Turn-by-Turn Directions
                    </h3>
                    <button
                      onClick={clearRoute}
                      style={{
                        padding: '6px 12px',
                        borderRadius: '6px',
                        border: 'none',
                        background: '#ef4444',
                        color: 'white',
                        cursor: 'pointer',
                        fontSize: '12px'
                      }}
                    >
                      Clear Route
                    </button>
                  </div>
                  {turnByTurn.map((step, index) => (
                    <div
                      key={index}
                      style={{
                        padding: '12px',
                        marginBottom: '8px',
                        background: darkMode ? '#374151' : '#f9fafb',
                        borderRadius: '8px',
                        borderLeft: '3px solid #2196F3'
                      }}
                    >
                      <div style={{ fontWeight: '500', marginBottom: '4px' }}>
                        {step.step}. {step.instruction}
                      </div>
                      <div style={{ fontSize: '13px', opacity: 0.7 }}>
                        {formatDistance(step.distance)} · {formatDuration(step.duration)}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* POI List */}
              {pois.length > 0 && turnByTurn.length === 0 && (
                <div style={{
                  background: darkMode ? '#1f2937' : 'white',
                  borderRadius: '12px',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                  padding: '20px',
                  maxHeight: '270px',
                  overflowY: 'auto'
                }}>
                  <h3 style={{ margin: '0 0 15px 0', fontSize: '16px', fontWeight: 'bold' }}>
                    Nearby Places
                  </h3>
                  {pois.map((poi, index) => (
                    <div
                      key={index}
                      style={{
                        padding: '12px',
                        marginBottom: '8px',
                        background: darkMode ? '#374151' : '#f9fafb',
                        borderRadius: '8px'
                      }}
                    >
                      <div style={{ fontWeight: '500', marginBottom: '4px' }}>
                        {poi.name}
                      </div>
                      {poi.category && (
                        <div style={{ fontSize: '13px', opacity: 0.7 }}>
                          {poi.category}
                        </div>
                      )}
                      {poi.distance && (
                        <div style={{ fontSize: '12px', marginTop: '4px', color: '#2196F3' }}>
                          {formatDistance(poi.distance)} away
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style>{`
        .loading-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: ${darkMode ? '#9ca3af' : '#6b7280'};
          animation: bounce 1.4s infinite ease-in-out;
        }
        @keyframes bounce {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1); }
        }
      `}</style>
    </div>
  );
}

export default NavigationChatbot;
