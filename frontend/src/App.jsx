import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation as useRouterLocation, useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useLocation } from './contexts/LocationContext';

import SearchBar from './components/SearchBar';
import ResultCard from './components/ResultCard';
import InteractiveMainPage from './components/InteractiveMainPage';
import WeatherThemeProvider from './components/WeatherThemeProvider';
import CookieConsent from './components/CookieConsent';
import NavBar from './components/NavBar';
import LanguageSwitcher from './components/LanguageSwitcher';
import MainPageMobileNavbar from './components/MainPageMobileNavbar';
import LocationPermissionModal from './components/LocationPermissionModal';
import RoutePlanningForm from './components/RoutePlanningForm';
import POICard from './components/POICard';
import DistrictInfo from './components/DistrictInfo';
import ItineraryTimeline from './components/ItineraryTimeline';
import MLInsights from './components/MLInsights';
import MapVisualization from './components/MapVisualization';

import { useMobileUtils, InstallPWAButton, MobileSwipe } from './hooks/useMobileUtils.jsx';
import { fetchResults, getSessionId } from './api/api';
import GoogleAnalytics, { trackChatEvent, trackEvent } from './utils/analytics';
import './App.css';
import './components/InteractiveMainPage.css';

const App = () => {
  const routerLocation = useRouterLocation();
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
  
  // Location context
  const {
    currentLocation,
    hasLocation,
    locationLoading,
    locationError,
    gpsPermission,
    neighborhood,
    locationSource,
    isTracking,
    requestGPSLocation,
    setManualLocation,
    startGPSTracking,
    stopGPSTracking,
    clearLocation,
    formatLocationForAI
  } = useLocation();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [userId] = useState(() => {
    // Get or create user ID
    let id = localStorage.getItem('user_id');
    if (!id) {
      id = 'user_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('user_id', id);
    }
    return id;
  });
  const [messages, setMessages] = useState(() => {
    // Load saved messages from localStorage
    try {
      const saved = localStorage.getItem('chat-messages');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [expanded, setExpanded] = useState(() => {
    // If accessed via /chat route, auto-expand
    if (window.location.pathname === '/chat') {
      return true;
    }
    // If there are saved messages, start in expanded mode
    try {
      const saved = localStorage.getItem('chat-messages');
      return saved ? JSON.parse(saved).length > 0 : false;
    } catch {
      return false;
    }
  });
  const [sessionId] = useState(() => getSessionId()); // Get persistent session ID
  const chatScrollRef = useRef(null);
  
  // Location-based features state
  const [showLocationModal, setShowLocationModal] = useState(false);
  const [showRoutePlanning, setShowRoutePlanning] = useState(false);

  // Mobile utilities hook
  const { 
    isMobile, 
    isTouch, 
    orientation, 
    hapticFeedback, 
    handleTouchStart, 
    handleTouchEnd 
  } = useMobileUtils();

  // Auto-expand when navigating to /chat route, collapse when leaving
  useEffect(() => {
    // Ensure page stays at top when component mounts
    window.scrollTo(0, 0);
    if (routerLocation.pathname === '/chat') {
      setExpanded(true);
    } else if (routerLocation.pathname === '/') {
      // When returning to main page, collapse the chat to hide messages
      setExpanded(false);
    }
  }, [routerLocation.pathname]);

  // Add/remove main-page class on body
  useEffect(() => {
    if (routerLocation.pathname === '/') {
      document.body.classList.add('main-page');
    } else {
      document.body.classList.remove('main-page');
    }
    
    // Cleanup on unmount
    return () => {
      document.body.classList.remove('main-page');
    };
  }, [routerLocation.pathname]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chat-messages', JSON.stringify(messages));
    }
  }, [messages]);

  // Notify AppRouter when chat state changes
  useEffect(() => {
    // Dispatch custom event when expanded state changes
    window.dispatchEvent(new CustomEvent('chatStateChanged', { 
      detail: { expanded, hasMessages: messages.length > 0 } 
    }));
  }, [expanded, messages.length]);

  useEffect(() => {
    // Audio file temporarily disabled - uncomment when audio file is available
    // const audio = new Audio('/welcome_baskan.mp3');
    // audio.volume = 0.5;
    // audio.play().catch(() => {/* ignore audio errors */});
  }, []);

  useEffect(() => {
    if (expanded && chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, expanded]);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    // Track the search event
    trackChatEvent('search_initiated', query);
    
    // Navigate to chat page with the query
    navigate('/chat', { state: { initialQuery: query } });
    
    // Clear the query after navigation
    setQuery('');
  };

  const handleQuickStart = (quickQuery) => {
    // Navigate to chat page with the quick query
    navigate('/chat', { state: { initialQuery: quickQuery } });
  };

  // Location-based feature handlers
  const handleLocationSet = async (location) => {
    try {
      if (location.source === 'gps') {
        // GPS location from modal
        await requestGPSLocation();
      } else {
        // Manual location from modal
        await setManualLocation(location);
      }
      console.log('Location set:', location);
    } catch (error) {
      console.error('Error setting location:', error);
    }
  };

  const handleRouteRequest = async (routeData) => {
    try {
      // Simulate route planning - route planning functionality moved to chat page
      console.log('Route request:', routeData);
    } catch (error) {
      console.error('Route planning error:', error);
    }
  };

  // Logo click handler removed (logo no longer displayed on main page)
  /*
  const handleLogoClick = () => {
    // Track navigation back to home
    trackEvent('logo_click', 'navigation', 'home');
    
    // Navigate to home page
    navigate('/');
    
    // Reset chat state to go back to homepage
    setExpanded(false);
    setMessages([]);
    setResults([]);
    setQuery('');
    
    // Clear session storage to start fresh conversation
    localStorage.removeItem('chat_session_id');
    localStorage.removeItem('chat-messages');
  };
  */

  const handleLanguageChange = (languageCode) => {
    i18n.changeLanguage(languageCode);
    
    // Handle RTL for Arabic
    if (languageCode === 'ar') {
      document.documentElement.setAttribute('dir', 'rtl');
      document.documentElement.setAttribute('lang', 'ar');
    } else {
      document.documentElement.setAttribute('dir', 'ltr');
      document.documentElement.setAttribute('lang', languageCode);
    }
  };

  return (
    <WeatherThemeProvider>
      <div style={{ width: '100vw', minHeight: '100vh', background: 'none', display: 'flex', flexDirection: 'column' }}>
        <GoogleAnalytics />
        {/* <DebugInfo /> */}

        {/* Mobile Top Navbar for main page only */}
        {location.pathname === '/' && <MainPageMobileNavbar />}

        <div className="main-page-background main-to-chat-transition main" style={{
          flex: 1, 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'flex-start',
          width: '100vw', 
          minHeight: '100vh', 
          paddingTop: isMobile || window.innerWidth <= 768 ? '80px' : '6rem', // Account for mobile navbar
          paddingBottom: '2rem',
          paddingLeft: isMobile || window.innerWidth <= 768 ? '1rem' : '2rem',
          paddingRight: isMobile || window.innerWidth <= 768 ? '1rem' : '2rem'
        }}>
          


          {/* Logo removed from main page as per user request */}
          
          {/* Simple search bar - ChatGPT style */}
          <div style={{
            width: '100%', 
            maxWidth: isMobile || window.innerWidth <= 768 ? '100%' : 1200, 
            margin: isMobile || window.innerWidth <= 768 ? '6rem auto 2rem auto' : '0 auto 2rem auto', 
            padding: isMobile || window.innerWidth <= 768 ? '0 0.5rem' : '1rem',
            zIndex: 10,
          }}>
            {/* GPS Location Status */}
            {hasLocation && (
              <div style={{
                textAlign: 'center',
                marginBottom: '0.5rem',
                padding: '0.5rem',
                background: 'rgba(16, 185, 129, 0.1)',
                borderRadius: '8px',
                fontSize: '0.875rem',
                color: '#059669'
              }}>
                📍 GPS Active - Location-aware responses enabled
                <button 
                  onClick={clearLocation}
                  style={{
                    marginLeft: '1rem',
                    padding: '0.25rem 0.75rem',
                    background: 'transparent',
                    border: '1px solid #059669',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    color: '#059669'
                  }}
                >
                  Disable
                </button>
              </div>
            )}
            {!hasLocation && (
              <div style={{
                textAlign: 'center',
                marginBottom: '0.5rem',
                padding: '0.5rem',
                background: 'rgba(99, 102, 241, 0.1)',
                borderRadius: '8px',
                fontSize: '0.875rem',
                color: '#4f46e5'
              }}>
                💡 Enable GPS for personalized, location-aware recommendations
                <button 
                  onClick={requestGPSLocation}
                  style={{
                    marginLeft: '1rem',
                    padding: '0.25rem 0.75rem',
                    background: '#6366f1',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '0.75rem',
                    color: 'white'
                  }}
                >
                  📍 Enable GPS
                </button>
              </div>
            )}
            
            <SearchBar
              value={query}
              onChange={e => setQuery(e.target.value)}
              onSubmit={handleSearch}
              placeholder={t("chat.searchPlaceholder")}
              isLoading={searchLoading}
              expanded={expanded}
            />
          </div>
          
          {/* Chat Messages - Only show if expanded (not when returning from /chat page) */}
          {messages.length > 0 && expanded && routerLocation.pathname !== '/' && (
            <div className="mt-6 max-w-4xl mx-auto px-4">{messages.map((msg) => (
                <div key={msg.id} className={`mb-4 ${msg.sender === "user" ? "text-right" : "text-left"}`}>
                  <div className={`inline-block p-3 rounded-lg max-w-[80%] ${
                    msg.sender === "user" 
                      ? "bg-blue-500 text-white ml-auto" 
                      : "bg-gray-100 text-gray-900"
                  }`}>
                    <div>{msg.content}</div>
                    
                    {/* Route Planning Action Button */}
                    {msg.sender === "assistant" && msg.metadata?.route_intent && (
                      <button
                        onClick={() => {
                          // Navigate to route planner with extracted intent
                          navigate('/route-planner', {
                            state: {
                              query: msg.metadata.original_query,
                              intent: msg.metadata.route_intent
                            }
                          });
                        }}
                        className="mt-3 bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-all"
                      >
                        🗺️ Open Route Planner
                      </button>
                    )}
                    
                    {/* Map Visualization */}
                    {msg.sender === "assistant" && msg.map_data && (msg.map_data.markers || msg.map_data.routes) && (
                      <div className="mt-3">
                        <MapVisualization 
                          mapData={msg.map_data} 
                          height="400px" 
                          className="rounded-lg shadow-md"
                        />
                        <div className="text-xs text-gray-600 mt-2 text-center">
                          🗺️ {msg.map_data.markers?.length || 0} locations • {msg.map_data.routes?.length || 0} routes
                        </div>
                      </div>
                    )}
                    
                    {/* Metadata Components */}
                    {msg.sender === "assistant" && msg.metadata && (
                      <div className="mt-3 space-y-3">
                        {/* ML Insights */}
                        {msg.metadata.ml_predictions && (
                          <MLInsights predictions={msg.metadata.ml_predictions} darkMode={false} />
                        )}
                        
                        {/* POI Cards */}
                        {msg.metadata.pois?.map((poi, idx) => (
                          <POICard key={idx} poi={poi} darkMode={false} />
                        ))}
                        
                        {/* District Info */}
                        {msg.metadata.district_info && (
                          <DistrictInfo district={msg.metadata.district_info} darkMode={false} />
                        )}
                        
                        {/* Itinerary */}
                        {msg.metadata.total_itinerary && (
                          <ItineraryTimeline itinerary={msg.metadata.total_itinerary} darkMode={false} />
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
          
          {/* Interactive Main Page Content - Always show when on main route */}
          {routerLocation.pathname === '/' && (
            <div>
              <InteractiveMainPage onQuickStart={handleQuickStart} />
            </div>
          )}
          
          {/* Districts and interactive content now visible on all devices */}
        </div>

        {/* Location Permission Modal */}
        <LocationPermissionModal
          isOpen={showLocationModal}
          onClose={() => setShowLocationModal(false)}
          onLocationSet={handleLocationSet}
        />

        {/* Route Planning Form */}
        <RoutePlanningForm
          isOpen={showRoutePlanning}
          onClose={() => setShowRoutePlanning(false)}
          onRouteRequest={handleRouteRequest}
          userLocation={currentLocation}
        />

        {/* Cookie Consent Banner */}
        <CookieConsent />
        
        {/* PWA Install Button */}
        <InstallPWAButton />


      </div>
    </WeatherThemeProvider>
  );
};

export default App;
