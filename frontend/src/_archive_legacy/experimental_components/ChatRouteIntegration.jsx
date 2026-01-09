/**
 * Chat Route Integration
 * Phase 3: Connect route maker with existing chat system
 */

import React, { useState, useEffect } from 'react';
import RouteMap from './RouteMap';
import { generateRoute, analyzeTSP } from '../api/routeApi';
import { extractLocationFromQuery } from '../api/api';

const ChatRouteIntegration = ({ 
  message, 
  onRouteGenerated = null,
  className = ""
}) => {
  const [isGeneratingRoute, setIsGeneratingRoute] = useState(false);
  const [generatedRoute, setGeneratedRoute] = useState(null);
  const [routeError, setRouteError] = useState(null);
  const [extractedIntent, setExtractedIntent] = useState(null);

  // Route-related keywords to detect route requests
  const routeKeywords = [
    'route', 'plan', 'itinerary', 'directions', 'path', 'journey',
    'walking tour', 'visit', 'explore', 'day trip', 'attractions',
    'places to see', 'best route', 'optimize', 'efficient', 'tour'
  ];

  // Location extraction patterns
  const locationPatterns = [
    /from\s+([^,\s]+(?:\s+[^,\s]+)*)/i,
    /starting\s+at\s+([^,\s]+(?:\s+[^,\s]+)*)/i,
    /begin\s+at\s+([^,\s]+(?:\s+[^,\s]+)*)/i,
    /near\s+([^,\s]+(?:\s+[^,\s]+)*)/i,
    /around\s+([^,\s]+(?:\s+[^,\s]+)*)/i
  ];

  // Detect if message is asking for route planning
  const detectRouteIntent = (messageText) => {
    const text = messageText.toLowerCase();
    
    // Check for route keywords
    const hasRouteKeyword = routeKeywords.some(keyword => 
      text.includes(keyword.toLowerCase())
    );
    
    if (!hasRouteKeyword) return null;

    // Extract location information
    const locationInfo = extractLocationFromQuery(messageText);
    
    // Extract preferences from message
    const preferences = {
      start_location: null,
      max_distance_km: 5.0,
      available_time_hours: 4.0,
      preferred_categories: [],
      route_style: 'balanced',
      transport_mode: 'walking',
      include_food: true,
      max_attractions: 6
    };

    // Try to extract starting location
    for (const pattern of locationPatterns) {
      const match = messageText.match(pattern);
      if (match) {
        preferences.start_location = match[1].trim();
        break;
      }
    }

    // Extract time constraints
    const timeMatch = text.match(/(\d+)\s*(hour|hr|hours|hrs)/i);
    if (timeMatch) {
      preferences.available_time_hours = parseInt(timeMatch[1]);
    }

    // Extract distance constraints
    const distanceMatch = text.match(/(\d+)\s*(km|kilometer|kilometers|mile|miles)/i);
    if (distanceMatch) {
      const distance = parseInt(distanceMatch[1]);
      preferences.max_distance_km = text.includes('mile') ? distance * 1.6 : distance;
    }

    // Extract style preferences
    if (text.includes('quick') || text.includes('efficient') || text.includes('fast')) {
      preferences.route_style = 'efficient';
    } else if (text.includes('scenic') || text.includes('beautiful') || text.includes('view')) {
      preferences.route_style = 'scenic';
    } else if (text.includes('cultural') || text.includes('history') || text.includes('museum')) {
      preferences.route_style = 'cultural';
    }

    // Extract transport mode
    if (text.includes('drive') || text.includes('car') || text.includes('taxi')) {
      preferences.transport_mode = 'driving';
    } else if (text.includes('metro') || text.includes('bus') || text.includes('public transport')) {
      preferences.transport_mode = 'public_transport';
    }

    // Extract categories
    const categoryKeywords = {
      'Historical Sites': ['history', 'historical', 'ancient', 'heritage'],
      'Museums': ['museum', 'gallery', 'art', 'exhibition'],
      'Religious Sites': ['mosque', 'church', 'religious', 'spiritual', 'hagia sophia'],
      'Markets & Shopping': ['market', 'bazaar', 'shopping', 'grand bazaar'],
      'Parks & Gardens': ['park', 'garden', 'green', 'nature'],
      'Viewpoints': ['view', 'panorama', 'lookout', 'observation'],
      'Food & Restaurants': ['food', 'restaurant', 'eat', 'dining', 'cuisine'],
      'Waterfront': ['bosphorus', 'sea', 'waterfront', 'ferry', 'bridge']
    };

    for (const [category, keywords] of Object.entries(categoryKeywords)) {
      if (keywords.some(keyword => text.includes(keyword))) {
        preferences.preferred_categories.push(category);
      }
    }

    return {
      isRouteRequest: true,
      confidence: 0.8,
      preferences,
      locationInfo
    };
  };

  // Generate route based on detected intent
  const handleRouteGeneration = async (intent) => {
    setIsGeneratingRoute(true);
    setRouteError(null);

    try {
      // Use default Istanbul coordinates if no specific location detected
      let startCoords = { lat: 41.0082, lng: 28.9784 }; // Sultanahmet default
      
      // Try to get coordinates for detected location
      if (intent.preferences.start_location) {
        // For demo, we'll use predefined locations
        const knownLocations = {
          'sultanahmet': { lat: 41.0086, lng: 28.9802 },
          'taksim': { lat: 41.0369, lng: 28.9850 },
          'galata': { lat: 41.0256, lng: 28.9744 },
          'kadikoy': { lat: 40.9833, lng: 29.0331 },
          'besiktas': { lat: 41.0422, lng: 29.0008 },
          'eminonu': { lat: 41.0175, lng: 28.9720 }
        };
        
        const locationKey = intent.preferences.start_location.toLowerCase();
        for (const [key, coords] of Object.entries(knownLocations)) {
          if (locationKey.includes(key) || key.includes(locationKey)) {
            startCoords = coords;
            break;
          }
        }
      }

      // Build route request
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
        optimization_method: 'auto'
      };

      console.log('🗺️ Generating route from chat:', routeRequest);
      
      const route = await generateRoute(routeRequest);
      setGeneratedRoute(route);
      
      if (onRouteGenerated) {
        onRouteGenerated(route, intent);
      }

    } catch (error) {
      console.error('Route generation failed:', error);
      setRouteError(error.message);
    } finally {
      setIsGeneratingRoute(false);
    }
  };

  // Analyze message when it changes
  useEffect(() => {
    if (message && message.content) {
      const intent = detectRouteIntent(message.content);
      setExtractedIntent(intent);
      
      // Auto-generate route if high confidence
      if (intent && intent.confidence > 0.7 && !generatedRoute && !isGeneratingRoute) {
        handleRouteGeneration(intent);
      }
    }
  }, [message]);

  // Don't render if not a route-related message
  if (!extractedIntent || !extractedIntent.isRouteRequest) {
    return null;
  }

  return (
    <div className={`chat-route-integration ${className}`}>
      {/* Route Intent Detection */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <div className="flex items-center space-x-2 mb-2">
          <span className="text-blue-600">🗺️</span>
          <span className="text-blue-800 font-medium">Route Planning Detected</span>
        </div>
        
        <div className="text-sm text-blue-700 space-y-1">
          {extractedIntent.preferences.start_location && (
            <div>📍 Starting from: {extractedIntent.preferences.start_location}</div>
          )}
          <div>⏱️ Duration: {extractedIntent.preferences.available_time_hours} hours</div>
          <div>📏 Max distance: {extractedIntent.preferences.max_distance_km} km</div>
          <div>🎯 Style: {extractedIntent.preferences.route_style}</div>
          {extractedIntent.preferences.preferred_categories.length > 0 && (
            <div>🏛️ Interests: {extractedIntent.preferences.preferred_categories.join(', ')}</div>
          )}
        </div>
        
        {!generatedRoute && !isGeneratingRoute && (
          <button
            onClick={() => handleRouteGeneration(extractedIntent)}
            className="mt-3 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 text-sm"
          >
            Generate Route
          </button>
        )}
      </div>

      {/* Loading State */}
      {isGeneratingRoute && (
        <div className="bg-white border rounded-lg p-4 mb-4">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            <span className="text-gray-700">Generating your personalized route...</span>
          </div>
        </div>
      )}

      {/* Error State */}
      {routeError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
          <div className="text-red-800">
            <strong>Route Generation Failed:</strong> {routeError}
          </div>
          <button
            onClick={() => {
              setRouteError(null);
              handleRouteGeneration(extractedIntent);
            }}
            className="mt-2 text-red-600 hover:text-red-700 text-sm"
          >
            Try Again
          </button>
        </div>
      )}

      {/* Generated Route */}
      {generatedRoute && (
        <div className="bg-white border rounded-lg overflow-hidden mb-4">
          {/* Route Header */}
          <div className="bg-green-50 border-b border-green-200 p-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="font-semibold text-green-900 flex items-center">
                  <span className="mr-2">✅</span>
                  {generatedRoute.name}
                </h4>
                <p className="text-green-700 text-sm mt-1">{generatedRoute.description}</p>
              </div>
              <div className="text-right text-sm text-green-700">
                <div><strong>{generatedRoute.total_distance_km.toFixed(1)}</strong> km</div>
                <div><strong>{generatedRoute.estimated_duration_hours.toFixed(1)}</strong> hrs</div>
              </div>
            </div>
          </div>

          {/* Route Map */}
          <div className="p-4">
            <RouteMap 
              route={generatedRoute}
              style={{ height: '300px' }}
              showControls={true}
              onAttractionClick={(id, point) => {
                console.log('Clicked attraction in chat:', id, point);
              }}
            />
          </div>

          {/* Route Summary */}
          <div className="border-t border-gray-200 p-4">
            <h5 className="font-medium mb-2">Route Highlights:</h5>
            <div className="space-y-1 text-sm">
              {generatedRoute.points.slice(0, 5).map((point, index) => (
                <div key={index} className="flex items-center space-x-2">
                  <span className="w-5 h-5 bg-blue-100 text-blue-800 text-xs font-medium rounded-full flex items-center justify-center">
                    {index + 1}
                  </span>
                  <span className="text-gray-700">{point.name}</span>
                  <span className="text-gray-500 text-xs">({point.category})</span>
                </div>
              ))}
              {generatedRoute.points.length > 5 && (
                <div className="text-gray-500 text-xs ml-7">
                  ... and {generatedRoute.points.length - 5} more stops
                </div>
              )}
            </div>

            {/* Route Actions */}
            <div className="flex space-x-2 mt-4">
              <button className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">
                View Full Route
              </button>
              <button className="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700">
                Save Route
              </button>
              <button className="bg-gray-200 text-gray-700 px-3 py-1 rounded text-sm hover:bg-gray-300">
                Modify
              </button>
            </div>
          </div>

          {/* Optimization Info */}
          {generatedRoute.metadata?.tsp_optimized && (
            <div className="bg-purple-50 border-t border-purple-200 p-3">
              <div className="text-purple-800 text-sm">
                <strong>🧮 Route Optimized:</strong> Using {generatedRoute.metadata.optimization_method} 
                algorithm for {generatedRoute.metadata.num_attractions} attractions.
                This route minimizes walking distance while maximizing your experience!
              </div>
            </div>
          )}
        </div>
      )}

      {/* Suggested Follow-ups */}
      {generatedRoute && (
        <div className="bg-gray-50 border rounded-lg p-4">
          <div className="text-sm text-gray-700 mb-2">
            <strong>💬 You can also ask me:</strong>
          </div>
          <div className="flex flex-wrap gap-2">
            {[
              "Tell me more about the Blue Mosque",
              "What's the best time to visit Hagia Sophia?",
              "How do I get from Sultanahmet to Galata Tower?",
              "Recommend restaurants near my route",
              "Create a shorter 2-hour version"
            ].map((suggestion, index) => (
              <button
                key={index}
                className="bg-white border border-gray-300 text-gray-700 px-3 py-1 rounded-full text-xs hover:bg-gray-100"
                onClick={() => {
                  // This would trigger a new chat message
                  if (window.dispatchEvent) {
                    window.dispatchEvent(new CustomEvent('chatSuggestionClick', {
                      detail: { message: suggestion }
                    }));
                  }
                }}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatRouteIntegration;
