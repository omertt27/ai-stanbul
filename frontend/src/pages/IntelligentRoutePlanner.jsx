/**
 * Intelligent Route Planner - Week 3 Complete Integration
 * LLM-powered conversational route planning with interactive map
 */

import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import IntelligentRouteMap from '../components/IntelligentRouteMap';
import RouteSidebar from '../components/RouteSidebar';
import RouteControls from '../components/RouteControls';
import { generateRoute, analyzeTSP, fetchAttractions } from '../api/routeApi';
import { getSessionId } from '../api/api';
import './IntelligentRoutePlanner.css';

const IntelligentRoutePlanner = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { t, i18n } = useTranslation();
  const chatInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // State management
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentRoute, setCurrentRoute] = useState(null);
  const [selectedWaypoint, setSelectedWaypoint] = useState(null);
  const [sessionId] = useState(() => getSessionId());
  const [userId] = useState(() => localStorage.getItem('user_id') || `user_${Date.now()}`);
  const [showSidebar, setShowSidebar] = useState(true);
  const [showChat, setShowChat] = useState(true);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);
  const [routeHistory, setRouteHistory] = useState([]);
  const [optimizationMode, setOptimizationMode] = useState('auto');

  // Initialize from route state if passed from App
  useEffect(() => {
    if (location.state?.route) {
      setCurrentRoute(location.state.route);
      addMessage('assistant', `I've loaded your route: ${location.state.route.name}`);
    }
    if (location.state?.query) {
      setInputMessage(location.state.query);
    }
  }, [location.state]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth <= 768;
      setIsMobile(mobile);
      // Auto-hide sidebar on mobile if route is active
      if (mobile && currentRoute) {
        setShowSidebar(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [currentRoute]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Add message to chat
  const addMessage = (sender, content, metadata = null) => {
    const message = {
      id: Date.now() + Math.random(),
      sender,
      content,
      metadata,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, message]);
    return message;
  };

  // Extract route intent from message using pattern matching
  const extractRouteIntent = (message) => {
    const text = message.toLowerCase();
    
    // Route request patterns
    const routePatterns = [
      /plan.*route|create.*route|make.*route/i,
      /show.*places|visit.*places|explore/i,
      /tour|itinerary|day trip/i,
      /from.*to|starting.*from/i,
      /(\d+)\s*(hour|hr)/i,
      /walking|driving|transit/i
    ];

    const isRouteRequest = routePatterns.some(pattern => pattern.test(text));
    if (!isRouteRequest) return null;

    // Extract parameters
    const intent = {
      type: 'route_planning',
      preferences: {
        start_location: null,
        max_distance_km: 5.0,
        available_time_hours: 4.0,
        preferred_categories: [],
        route_style: 'balanced',
        transport_mode: 'walking',
        include_food: true,
        max_attractions: 8
      }
    };

    // Extract time
    const timeMatch = text.match(/(\d+)\s*(hour|hr|h)/i);
    if (timeMatch) {
      intent.preferences.available_time_hours = parseFloat(timeMatch[1]);
    }

    // Extract distance
    const distanceMatch = text.match(/(\d+)\s*(km|kilometer|mile)/i);
    if (distanceMatch) {
      const dist = parseFloat(distanceMatch[1]);
      intent.preferences.max_distance_km = text.includes('mile') ? dist * 1.6 : dist;
    }

    // Extract style
    if (text.match(/quick|fast|efficient/i)) intent.preferences.route_style = 'efficient';
    if (text.match(/scenic|beautiful|view/i)) intent.preferences.route_style = 'scenic';
    if (text.match(/cultural|history|museum/i)) intent.preferences.route_style = 'cultural';

    // Extract transport mode
    if (text.match(/drive|driving|car/i)) intent.preferences.transport_mode = 'driving';
    if (text.match(/walk|walking/i)) intent.preferences.transport_mode = 'walking';
    if (text.match(/public|transit|metro|bus/i)) intent.preferences.transport_mode = 'public_transport';

    // Extract location
    const locationMatch = text.match(/(?:from|starting|begin|near)\s+([a-z\s]+?)(?:\s|$|,|to)/i);
    if (locationMatch) {
      intent.preferences.start_location = locationMatch[1].trim();
    }

    // Extract categories from keywords
    const categoryMap = {
      'Historical Sites': ['history', 'historical', 'ancient', 'heritage'],
      'Museums': ['museum', 'gallery', 'art'],
      'Religious Sites': ['mosque', 'church', 'hagia sophia', 'religious'],
      'Markets & Shopping': ['market', 'bazaar', 'shopping', 'grand bazaar'],
      'Parks & Gardens': ['park', 'garden', 'green'],
      'Viewpoints': ['view', 'panorama', 'tower'],
      'Food & Restaurants': ['food', 'restaurant', 'eat', 'dining'],
      'Waterfront': ['bosphorus', 'sea', 'waterfront', 'ferry']
    };

    for (const [category, keywords] of Object.entries(categoryMap)) {
      if (keywords.some(kw => text.includes(kw))) {
        intent.preferences.preferred_categories.push(category);
      }
    }

    return intent;
  };

  // Get default starting coordinates for known locations
  const getStartCoordinates = (locationName) => {
    const locations = {
      'sultanahmet': { lat: 41.0086, lng: 28.9802 },
      'taksim': { lat: 41.0369, lng: 28.9850 },
      'galata': { lat: 41.0256, lng: 28.9744 },
      'kadikoy': { lat: 40.9833, lng: 29.0331 },
      'besiktas': { lat: 41.0422, lng: 29.0008 },
      'eminonu': { lat: 41.0175, lng: 28.9720 },
      'ortakoy': { lat: 41.0547, lng: 29.0267 },
      'balat': { lat: 41.0297, lng: 28.9489 }
    };

    if (!locationName) {
      return { lat: 41.0082, lng: 28.9784 }; // Default: Sultanahmet
    }

    const normalized = locationName.toLowerCase().trim();
    for (const [key, coords] of Object.entries(locations)) {
      if (normalized.includes(key) || key.includes(normalized)) {
        return coords;
      }
    }

    return { lat: 41.0082, lng: 28.9784 }; // Default
  };

  // Generate route from intent
  const generateRouteFromIntent = async (intent) => {
    setIsProcessing(true);
    
    try {
      // Build route request
      const startCoords = getStartCoordinates(intent.preferences.start_location);
      
      const routeRequest = {
        start_lat: startCoords.lat,
        start_lng: startCoords.lng,
        max_distance_km: intent.preferences.max_distance_km,
        available_time_hours: intent.preferences.available_time_hours,
        preferred_categories: intent.preferences.preferred_categories,
        route_style: intent.preferences.route_style,
        transport_mode: intent.preferences.transport_mode,
        include_food: intent.preferences.include_food,
        max_attractions: intent.preferences.max_attractions,
        optimization_method: optimizationMode
      };

      console.log('🗺️ Generating route:', routeRequest);
      addMessage('assistant', 'Analyzing your request and planning the optimal route...', {
        loading: true
      });

      const route = await generateRoute(routeRequest);
      
      setCurrentRoute(route);
      setRouteHistory(prev => [...prev, route]);
      
      // Create detailed response message
      const response = `✅ I've created your personalized route: **${route.name}**

📍 **${route.points.length} stops** along **${route.total_distance_km.toFixed(1)} km**
⏱️ Estimated duration: **${route.estimated_duration_hours.toFixed(1)} hours**
🚶 Transport mode: **${route.transport_mode}**

Your route includes:
${route.points.slice(0, 5).map((p, i) => `${i + 1}. ${p.name} (${p.category})`).join('\n')}
${route.points.length > 5 ? `... and ${route.points.length - 5} more amazing places!` : ''}

💡 This route is optimized using ${route.metadata?.optimization_method || 'intelligent'} pathfinding to minimize travel time while maximizing your experience!

You can now:
- Click on markers to see details
- Drag waypoints to reorder your route
- Use controls to save or share your route`;

      addMessage('assistant', response, {
        route: route,
        actionable: true
      });

      // Provide suggestions
      addMessage('assistant', '💬 You can ask me to:\n• "Add the Galata Tower"\n• "Make it a 3-hour route"\n• "Remove museums"\n• "Show restaurants along the way"', {
        suggestions: true
      });

    } catch (error) {
      console.error('Route generation failed:', error);
      addMessage('assistant', `❌ Sorry, I couldn't create your route: ${error.message}. Please try again with different preferences.`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle route modification requests
  const modifyCurrentRoute = async (message) => {
    if (!currentRoute) {
      addMessage('assistant', '⚠️ No active route to modify. Please create a route first!');
      return;
    }

    const text = message.toLowerCase();
    
    // Remove attraction
    if (text.match(/remove|delete|skip/i)) {
      const attractionMatch = text.match(/remove|delete|skip\s+(.+?)(?:\s|$)/i);
      if (attractionMatch) {
        const query = attractionMatch[1];
        const index = currentRoute.points.findIndex(p => 
          p.name.toLowerCase().includes(query)
        );
        
        if (index !== -1) {
          const removed = currentRoute.points[index];
          const newPoints = currentRoute.points.filter((_, i) => i !== index);
          
          setCurrentRoute({
            ...currentRoute,
            points: newPoints,
            total_distance_km: currentRoute.total_distance_km * 0.9 // Approximate
          });
          
          addMessage('assistant', `✅ Removed "${removed.name}" from your route.`);
          return;
        }
      }
    }

    // Add attraction
    if (text.match(/add|include/i)) {
      addMessage('assistant', '🔍 Searching for attractions to add... (This feature is coming soon!)');
      return;
    }

    // Change duration
    const timeMatch = text.match(/(\d+)\s*hour/i);
    if (timeMatch && text.match(/make|change|adjust/i)) {
      const newHours = parseFloat(timeMatch[1]);
      addMessage('assistant', `📝 I'll regenerate the route for ${newHours} hours...`);
      
      // Regenerate with new time
      const intent = extractRouteIntent(message);
      if (intent) {
        intent.preferences.available_time_hours = newHours;
        await generateRouteFromIntent(intent);
      }
      return;
    }

    // General modification
    addMessage('assistant', '🤔 I understand you want to modify the route. Could you be more specific? For example:\n• "Remove Hagia Sophia"\n• "Add more restaurants"\n• "Make it a 3-hour tour"');
  };

  // Send message handler
  const handleSendMessage = async (e) => {
    e?.preventDefault();
    
    if (!inputMessage.trim() || isProcessing) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    addMessage('user', userMessage);

    // Check for route intent
    const intent = extractRouteIntent(userMessage);
    
    if (intent) {
      // Route planning request
      await generateRouteFromIntent(intent);
    } else if (currentRoute) {
      // Check if it's a modification request
      const modificationKeywords = ['add', 'remove', 'change', 'modify', 'adjust', 'make'];
      if (modificationKeywords.some(kw => userMessage.toLowerCase().includes(kw))) {
        await modifyCurrentRoute(userMessage);
      } else {
        // General question about current route
        handleGeneralQuery(userMessage);
      }
    } else {
      // No route context, provide guidance
      addMessage('assistant', `I'm your AI route planning assistant! 🗺️

To get started, try asking:
• "Plan a 4-hour walking tour in Sultanahmet"
• "Create a route with museums and historical sites"
• "Show me the best places to visit in Galata"
• "Make a food tour in Kadıköy"

I'll create an optimized route with turn-by-turn directions, interactive map, and detailed information about each stop!`);
    }
  };

  // Handle general queries about current route
  const handleGeneralQuery = (query) => {
    const text = query.toLowerCase();
    
    if (text.match(/how long|duration|time/i)) {
      addMessage('assistant', `⏱️ Your current route takes approximately **${currentRoute.estimated_duration_hours.toFixed(1)} hours** to complete, covering **${currentRoute.total_distance_km.toFixed(1)} km**.`);
    } else if (text.match(/how many|count|number/i)) {
      addMessage('assistant', `📍 Your route includes **${currentRoute.points.length} stops**: ${currentRoute.points.slice(0, 3).map(p => p.name).join(', ')}${currentRoute.points.length > 3 ? ', and more!' : ''}`);
    } else if (text.match(/restaurant|food|eat/i)) {
      const foodPlaces = currentRoute.points.filter(p => 
        p.category?.toLowerCase().includes('food') || 
        p.category?.toLowerCase().includes('restaurant')
      );
      if (foodPlaces.length > 0) {
        addMessage('assistant', `🍽️ Food stops on your route:\n${foodPlaces.map((p, i) => `${i + 1}. ${p.name}`).join('\n')}`);
      } else {
        addMessage('assistant', '🍽️ Your route doesn\'t include restaurants yet. Would you like me to add some?');
      }
    } else {
      addMessage('assistant', `I have your route loaded! You can ask me:
• Details about specific attractions
• How to modify the route
• Transportation options
• Best times to visit

Or say "create a new route" to start fresh!`);
    }
  };

  // Handle quick suggestions
  const handleSuggestion = (suggestion) => {
    setInputMessage(suggestion);
    setTimeout(() => chatInputRef.current?.focus(), 0);
  };

  // Waypoint reordered callback
  const handleWaypointReorder = (newPoints) => {
    setCurrentRoute({
      ...currentRoute,
      points: newPoints
    });
    addMessage('assistant', '✅ Route order updated! The map will reflect your changes.');
  };

  // Waypoint removed callback
  const handleWaypointRemove = (pointId) => {
    const newPoints = currentRoute.points.filter(p => p.id !== pointId);
    setCurrentRoute({
      ...currentRoute,
      points: newPoints
    });
    addMessage('assistant', `✅ Removed waypoint. ${newPoints.length} stops remaining.`);
  };

  // Save route handler
  const handleSaveRoute = async () => {
    if (!currentRoute) return;
    
    // Save to localStorage for now (backend endpoint can be added later)
    const savedRoutes = JSON.parse(localStorage.getItem('saved_routes') || '[]');
    const routeToSave = {
      ...currentRoute,
      savedAt: new Date().toISOString(),
      id: `route_${Date.now()}`
    };
    
    savedRoutes.push(routeToSave);
    localStorage.setItem('saved_routes', JSON.stringify(savedRoutes));
    
    addMessage('assistant', `✅ Route saved successfully! You can access it anytime from your saved routes.`);
  };

  // Share route handler
  const handleShareRoute = async () => {
    if (!currentRoute) return;
    
    const shareText = `Check out my Istanbul route: ${currentRoute.name}\n${currentRoute.points.length} stops | ${currentRoute.total_distance_km.toFixed(1)} km`;
    
    if (navigator.share) {
      try {
        await navigator.share({
          title: currentRoute.name,
          text: shareText,
          url: window.location.href
        });
        addMessage('assistant', '✅ Route shared!');
      } catch (err) {
        console.log('Share cancelled');
      }
    } else {
      // Fallback: copy to clipboard
      navigator.clipboard.writeText(`${shareText}\n${window.location.href}`);
      addMessage('assistant', '✅ Route link copied to clipboard!');
    }
  };

  // Export route handler
  const handleExportRoute = (format) => {
    if (!currentRoute) return;
    
    let exportData;
    let filename;
    let mimeType;
    
    if (format === 'json') {
      exportData = JSON.stringify(currentRoute, null, 2);
      filename = `${currentRoute.name.replace(/\s+/g, '_')}.json`;
      mimeType = 'application/json';
    } else if (format === 'gpx') {
      // Basic GPX export
      exportData = generateGPX(currentRoute);
      filename = `${currentRoute.name.replace(/\s+/g, '_')}.gpx`;
      mimeType = 'application/gpx+xml';
    }
    
    const blob = new Blob([exportData], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    
    addMessage('assistant', `✅ Route exported as ${format.toUpperCase()}!`);
  };

  // Generate GPX format
  const generateGPX = (route) => {
    const waypoints = route.points.map((p, i) => `
    <wpt lat="${p.lat}" lon="${p.lng}">
      <name>${p.name}</name>
      <desc>${p.description || ''}</desc>
    </wpt>`).join('');
    
    return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="AI Istanbul Route Planner">
  <metadata>
    <name>${route.name}</name>
    <desc>${route.description || ''}</desc>
  </metadata>
  ${waypoints}
</gpx>`;
  };

  // Quick start templates
  const quickStartTemplates = [
    {
      icon: '🏛️',
      title: 'Historical Tour',
      query: 'Plan a 4-hour historical tour starting from Sultanahmet with museums and ancient sites'
    },
    {
      icon: '🍽️',
      title: 'Food Journey',
      query: 'Create a 3-hour food tour in Kadıköy with restaurants and local markets'
    },
    {
      icon: '🌆',
      title: 'Scenic Views',
      query: 'Show me a 5-hour route with the best viewpoints and waterfront locations'
    },
    {
      icon: '🕌',
      title: 'Cultural Sites',
      query: 'Make a cultural tour with mosques, bazaars and traditional districts'
    }
  ];

  return (
    <div className="intelligent-route-planner">
      {/* Header */}
      <div className="planner-header">
        <div className="header-content">
          <button onClick={() => navigate('/')} className="back-button">
            ← Back
          </button>
          <div className="header-title">
            <h1>🗺️ AI Route Planner</h1>
            <p>Plan your perfect Istanbul journey with AI assistance</p>
          </div>
          <div className="header-actions">
            <button 
              onClick={() => setShowChat(!showChat)}
              className="toggle-button"
              title={showChat ? 'Hide chat' : 'Show chat'}
            >
              💬 {showChat ? 'Hide' : 'Show'} Chat
            </button>
            {currentRoute && (
              <button 
                onClick={() => setShowSidebar(!showSidebar)}
                className="toggle-button"
                title={showSidebar ? 'Hide sidebar' : 'Show sidebar'}
              >
                📋 {showSidebar ? 'Hide' : 'Show'} Itinerary
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="planner-content">
        {/* Chat Panel */}
        {showChat && (
          <div className={`chat-panel ${!currentRoute ? 'full-width' : ''}`}>
            <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <div className="welcome-icon">🗺️</div>
                  <h2>Welcome to AI Route Planner!</h2>
                  <p>I'll help you create the perfect route through Istanbul with AI-powered optimization.</p>
                  
                  <div className="quick-start-templates">
                    <h3>Quick Start:</h3>
                    <div className="template-grid">
                      {quickStartTemplates.map((template, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleSuggestion(template.query)}
                          className="template-card"
                        >
                          <span className="template-icon">{template.icon}</span>
                          <span className="template-title">{template.title}</span>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="example-queries">
                    <p><strong>Or try asking:</strong></p>
                    <ul>
                      <li>"Plan a walking tour with historical sites"</li>
                      <li>"Create a 3-hour route near Galata"</li>
                      <li>"Show me the best food spots in Kadıköy"</li>
                      <li>"Make a scenic route along the Bosphorus"</li>
                    </ul>
                  </div>
                </div>
              ) : (
                <>
                  {messages.map(msg => (
                    <div key={msg.id} className={`message ${msg.sender}`}>
                      <div className="message-content">
                        {msg.content}
                      </div>
                      {msg.metadata?.loading && (
                        <div className="loading-indicator">
                          <span className="dot"></span>
                          <span className="dot"></span>
                          <span className="dot"></span>
                        </div>
                      )}
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Chat Input */}
            <form onSubmit={handleSendMessage} className="chat-input-form">
              <input
                ref={chatInputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Ask me to plan a route, add stops, or modify your journey..."
                disabled={isProcessing}
                className="chat-input"
              />
              <button 
                type="submit" 
                disabled={!inputMessage.trim() || isProcessing}
                className="send-button"
              >
                {isProcessing ? '⏳' : '🚀'} Send
              </button>
            </form>
          </div>
        )}

        {/* Map and Route Display */}
        {currentRoute && (
          <>
            <div className="map-container">
              <IntelligentRouteMap
                route={currentRoute}
                onWaypointClick={(point) => setSelectedWaypoint(point)}
                onWaypointDragEnd={(pointId, newLat, newLng) => {
                  // Update waypoint position
                  const newPoints = currentRoute.points.map(p =>
                    p.id === pointId ? { ...p, lat: newLat, lng: newLng } : p
                  );
                  setCurrentRoute({ ...currentRoute, points: newPoints });
                }}
                selectedWaypoint={selectedWaypoint}
              />
              
              {/* Map Overlay Controls */}
              <div className="map-overlay-controls">
                <RouteControls
                  onSave={handleSaveRoute}
                  onShare={handleShareRoute}
                  onExport={handleExportRoute}
                  onTransportChange={(mode) => {
                    setCurrentRoute({ ...currentRoute, transport_mode: mode });
                  }}
                  currentTransportMode={currentRoute.transport_mode}
                />
              </div>
            </div>

            {/* Route Sidebar */}
            {showSidebar && (
              <div className="sidebar-container">
                <RouteSidebar
                  route={currentRoute}
                  selectedWaypoint={selectedWaypoint}
                  onWaypointClick={(point) => setSelectedWaypoint(point)}
                  onWaypointReorder={handleWaypointReorder}
                  onWaypointRemove={handleWaypointRemove}
                  onClose={() => setShowSidebar(false)}
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default IntelligentRoutePlanner;
