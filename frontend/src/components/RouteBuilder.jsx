/**
 * Route Builder Component
 * Phase 3: Interactive Route Planning with Map Integration
 */

import React, { useState, useEffect } from 'react';
import RouteMap from './RouteMap';
import { generateRoute, fetchAttractions, analyzeTSP } from '../api/routeApi';

const RouteBuilder = ({ 
  initialLocation = { lat: 41.0082, lng: 28.9784 },
  onRouteCreated = null,
  className = ""
}) => {
  const [currentStep, setCurrentStep] = useState('preferences'); // preferences, attractions, route, map
  const [preferences, setPreferences] = useState({
    start_lat: initialLocation.lat,
    start_lng: initialLocation.lng,
    end_lat: null,
    end_lng: null,
    max_distance_km: 5.0,
    available_time_hours: 4.0,
    preferred_categories: [],
    route_style: 'balanced',
    transport_mode: 'walking',
    include_food: true,
    max_attractions: 6,
    optimization_method: 'auto'
  });
  
  const [availableAttractions, setAvailableAttractions] = useState([]);
  const [selectedAttractions, setSelectedAttractions] = useState([]);
  const [generatedRoute, setGeneratedRoute] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [optimizationAnalysis, setOptimizationAnalysis] = useState(null);

  const routeStyles = [
    { value: 'balanced', label: 'Balanced', icon: '⚖️', description: 'Mix of efficiency and attractions' },
    { value: 'efficient', label: 'Efficient', icon: '⚡', description: 'Shortest routes, less walking' },
    { value: 'scenic', label: 'Scenic', icon: '🌅', description: 'Beautiful views and landscapes' },
    { value: 'cultural', label: 'Cultural', icon: '🏛️', description: 'Museums, history, architecture' }
  ];

  const transportModes = [
    { value: 'walking', label: 'Walking', icon: '🚶', description: 'Best for exploring neighborhoods' },
    { value: 'driving', label: 'Driving', icon: '🚗', description: 'Cover more distance quickly' },
    { value: 'public_transport', label: 'Public Transport', icon: '🚌', description: 'Metro, bus, ferry' }
  ];

  const categories = [
    'Historical Sites', 'Museums', 'Religious Sites', 'Markets & Shopping',
    'Parks & Gardens', 'Viewpoints', 'Neighborhoods', 'Food & Restaurants',
    'Entertainment', 'Cultural Centers', 'Architecture', 'Waterfront'
  ];

  // Fetch attractions based on preferences
  const fetchNearbyAttractions = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const attractions = await fetchAttractions({
        lat: preferences.start_lat,
        lng: preferences.start_lng,
        radius_km: preferences.max_distance_km,
        categories: preferences.preferred_categories,
        limit: 20
      });
      
      setAvailableAttractions(attractions);
    } catch (err) {
      setError(`Failed to fetch attractions: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Generate route with current selections
  const handleGenerateRoute = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const routeRequest = {
        ...preferences,
        selected_attraction_ids: selectedAttractions.map(a => a.id)
      };
      
      const route = await generateRoute(routeRequest);
      setGeneratedRoute(route);
      
      // If we have multiple attractions, run TSP analysis
      if (selectedAttractions.length > 2) {
        try {
          const analysis = await analyzeTSP({
            attraction_ids: selectedAttractions.map(a => a.id),
            start_lat: preferences.start_lat,
            start_lng: preferences.start_lng,
            methods: ['tsp', 'heuristic', 'nearest']
          });
          setOptimizationAnalysis(analysis);
        } catch (tspError) {
          console.warn('TSP analysis failed:', tspError);
        }
      }
      
      setCurrentStep('map');
      
      if (onRouteCreated) {
        onRouteCreated(route);
      }
      
    } catch (err) {
      setError(`Failed to generate route: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle attraction selection
  const toggleAttractionSelection = (attraction) => {
    setSelectedAttractions(prev => {
      const isSelected = prev.find(a => a.id === attraction.id);
      if (isSelected) {
        return prev.filter(a => a.id !== attraction.id);
      } else if (prev.length < preferences.max_attractions) {
        return [...prev, attraction];
      }
      return prev;
    });
  };

  // Preferences Step Component
  const PreferencesStep = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Route Style */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">Route Style</label>
          <div className="space-y-2">
            {routeStyles.map(style => (
              <button
                key={style.value}
                onClick={() => setPreferences(prev => ({ ...prev, route_style: style.value }))}
                className={`w-full p-3 text-left border rounded-lg hover:bg-gray-50 ${
                  preferences.route_style === style.value 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{style.icon}</span>
                  <div>
                    <div className="font-medium">{style.label}</div>
                    <div className="text-sm text-gray-500">{style.description}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Transport Mode */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">Transportation</label>
          <div className="space-y-2">
            {transportModes.map(mode => (
              <button
                key={mode.value}
                onClick={() => setPreferences(prev => ({ ...prev, transport_mode: mode.value }))}
                className={`w-full p-3 text-left border rounded-lg hover:bg-gray-50 ${
                  preferences.transport_mode === mode.value 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{mode.icon}</span>
                  <div>
                    <div className="font-medium">{mode.label}</div>
                    <div className="text-sm text-gray-500">{mode.description}</div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Distance and Time */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Maximum Distance: {preferences.max_distance_km} km
          </label>
          <input
            type="range"
            min="1"
            max="15"
            step="0.5"
            value={preferences.max_distance_km}
            onChange={(e) => setPreferences(prev => ({ 
              ...prev, 
              max_distance_km: parseFloat(e.target.value) 
            }))}
            className="w-full"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Available Time: {preferences.available_time_hours} hours
          </label>
          <input
            type="range"
            min="1"
            max="12"
            step="0.5"
            value={preferences.available_time_hours}
            onChange={(e) => setPreferences(prev => ({ 
              ...prev, 
              available_time_hours: parseFloat(e.target.value) 
            }))}
            className="w-full"
          />
        </div>
      </div>

      {/* Categories */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Preferred Attractions (optional)
        </label>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {categories.map(category => (
            <button
              key={category}
              onClick={() => {
                setPreferences(prev => ({
                  ...prev,
                  preferred_categories: prev.preferred_categories.includes(category)
                    ? prev.preferred_categories.filter(c => c !== category)
                    : [...prev.preferred_categories, category]
                }));
              }}
              className={`p-2 text-sm border rounded hover:bg-gray-50 ${
                preferences.preferred_categories.includes(category)
                  ? 'border-blue-500 bg-blue-50 text-blue-700'
                  : 'border-gray-200'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Options */}
      <div className="flex items-center space-x-6">
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={preferences.include_food}
            onChange={(e) => setPreferences(prev => ({ 
              ...prev, 
              include_food: e.target.checked 
            }))}
            className="mr-2"
          />
          Include food stops
        </label>
        
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Max Attractions: {preferences.max_attractions}
          </label>
          <input
            type="range"
            min="3"
            max="15"
            value={preferences.max_attractions}
            onChange={(e) => setPreferences(prev => ({ 
              ...prev, 
              max_attractions: parseInt(e.target.value) 
            }))}
            className="w-24"
          />
        </div>
      </div>

      <button
        onClick={() => {
          fetchNearbyAttractions();
          setCurrentStep('attractions');
        }}
        disabled={isLoading}
        className="w-full bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50"
      >
        {isLoading ? 'Finding Attractions...' : 'Find Attractions'}
      </button>
    </div>
  );

  // Attractions Step Component
  const AttractionsStep = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">
          Select Attractions ({selectedAttractions.length}/{preferences.max_attractions})
        </h3>
        <button
          onClick={() => setCurrentStep('preferences')}
          className="text-blue-600 hover:text-blue-700"
        >
          ← Back to Preferences
        </button>
      </div>

      {availableAttractions.length === 0 ? (
        <div className="text-center py-8 text-gray-500">
          No attractions found. Try adjusting your preferences.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-96 overflow-y-auto">
          {availableAttractions.map(attraction => {
            const isSelected = selectedAttractions.find(a => a.id === attraction.id);
            return (
              <button
                key={attraction.id}
                onClick={() => toggleAttractionSelection(attraction)}
                disabled={!isSelected && selectedAttractions.length >= preferences.max_attractions}
                className={`p-4 text-left border rounded-lg hover:bg-gray-50 disabled:opacity-50 ${
                  isSelected 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-200'
                }`}
              >
                <div className="flex items-start space-x-3">
                  <div className={`w-6 h-6 border-2 rounded ${
                    isSelected 
                      ? 'bg-blue-600 border-blue-600' 
                      : 'border-gray-300'
                  } flex items-center justify-center`}>
                    {isSelected && <span className="text-white text-sm">✓</span>}
                  </div>
                  <div className="flex-1">
                    <h4 className="font-medium">{attraction.name}</h4>
                    <p className="text-sm text-gray-600">{attraction.category}</p>
                    <p className="text-sm text-gray-500">{attraction.district}</p>
                    <div className="text-sm text-blue-600 mt-1">
                      Score: {attraction.popularity_score}/10
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}

      <div className="flex space-x-4">
        <button
          onClick={() => setCurrentStep('preferences')}
          className="flex-1 bg-gray-200 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-300"
        >
          Back
        </button>
        <button
          onClick={handleGenerateRoute}
          disabled={selectedAttractions.length < 2 || isLoading}
          className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {isLoading ? 'Generating Route...' : `Generate Route (${selectedAttractions.length} stops)`}
        </button>
      </div>
    </div>
  );

  // Map Step Component
  const MapStep = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium">Your Route</h3>
        <button
          onClick={() => setCurrentStep('attractions')}
          className="text-blue-600 hover:text-blue-700"
        >
          ← Edit Route
        </button>
      </div>

      {generatedRoute && (
        <>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="font-medium">Distance:</span><br/>
                {generatedRoute.total_distance_km.toFixed(1)} km
              </div>
              <div>
                <span className="font-medium">Duration:</span><br/>
                {generatedRoute.estimated_duration_hours.toFixed(1)} hours
              </div>
              <div>
                <span className="font-medium">Stops:</span><br/>
                {generatedRoute.points.length}
              </div>
              <div>
                <span className="font-medium">Score:</span><br/>
                {generatedRoute.overall_score.toFixed(1)}/10
              </div>
            </div>
            
            {optimizationAnalysis && (
              <div className="mt-4 p-3 bg-blue-50 rounded border-l-4 border-blue-400">
                <div className="text-sm">
                  <strong>Route Optimization:</strong> Using {optimizationAnalysis.recommendation} method.
                  {optimizationAnalysis.summary.optimization_benefit !== '0%' && (
                    <span> Improved by {optimizationAnalysis.summary.optimization_benefit} over basic routing.</span>
                  )}
                </div>
              </div>
            )}
          </div>

          <RouteMap 
            route={generatedRoute}
            style={{ height: '500px' }}
            showControls={true}
            onAttractionClick={(id, point) => {
              console.log('Clicked attraction:', id, point);
            }}
          />
        </>
      )}

      <button
        onClick={() => {
          setCurrentStep('preferences');
          setSelectedAttractions([]);
          setGeneratedRoute(null);
          setOptimizationAnalysis(null);
        }}
        className="w-full bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700"
      >
        Create New Route
      </button>
    </div>
  );

  return (
    <div className={`route-builder ${className}`}>
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="text-red-800">{error}</div>
          <button 
            onClick={() => setError(null)}
            className="text-red-600 text-sm mt-2 hover:text-red-700"
          >
            Dismiss
          </button>
        </div>
      )}

      {currentStep === 'preferences' && <PreferencesStep />}
      {currentStep === 'attractions' && <AttractionsStep />}
      {currentStep === 'map' && <MapStep />}
    </div>
  );
};

export default RouteBuilder;
