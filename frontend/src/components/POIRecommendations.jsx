/**
 * POIRecommendations Component - Displays and manages POI recommendations
 */

import React, { useEffect, useState } from 'react';
import { useLocation } from '../contexts/LocationContext';

const POIRecommendations = ({ autoLoad = true, showFilters = true }) => {
  const {
    recommendations,
    recommendationsLoading,
    recommendationsError,
    currentLocation,
    preferences,
    getRecommendations,
    updatePreferences,
    planRoute
  } = useLocation();

  const [selectedPOIs, setSelectedPOIs] = useState([]);
  const [filters, setFilters] = useState({
    categories: [],
    rating: 0,
    radius: 2.0,
    openNow: false,
    priceRange: ''
  });

  // Available categories
  const categories = [
    { value: 'restaurant', label: '🍽️ Restaurants', color: 'bg-red-100 text-red-800' },
    { value: 'museum', label: '🏛️ Museums', color: 'bg-blue-100 text-blue-800' },
    { value: 'landmark', label: '🏗️ Landmarks', color: 'bg-purple-100 text-purple-800' },
    { value: 'shopping', label: '🛍️ Shopping', color: 'bg-pink-100 text-pink-800' },
    { value: 'entertainment', label: '🎭 Entertainment', color: 'bg-green-100 text-green-800' },
    { value: 'religious', label: '🕌 Religious Sites', color: 'bg-yellow-100 text-yellow-800' },
    { value: 'park', label: '🌳 Parks', color: 'bg-green-100 text-green-800' },
    { value: 'viewpoint', label: '🔭 Viewpoints', color: 'bg-indigo-100 text-indigo-800' }
  ];

  // Load recommendations when component mounts or location changes
  useEffect(() => {
    if (autoLoad && currentLocation) {
      handleLoadRecommendations();
    }
  }, [currentLocation, autoLoad]);

  const handleLoadRecommendations = async () => {
    try {
      await getRecommendations({
        categories: filters.categories,
        radius: filters.radius,
        filters: {
          rating: filters.rating > 0 ? filters.rating : undefined,
          open_now: filters.openNow,
          price_range: filters.priceRange || undefined
        }
      });
    } catch (error) {
      console.error('Failed to load recommendations:', error);
    }
  };

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    
    // Update global preferences
    updatePreferences({
      categories: newFilters.categories,
      radius: newFilters.radius,
      filters: {
        rating: newFilters.rating,
        open_now: newFilters.openNow,
        price_range: newFilters.priceRange
      }
    });
  };

  const handleCategoryToggle = (category) => {
    const newCategories = filters.categories.includes(category)
      ? filters.categories.filter(c => c !== category)
      : [...filters.categories, category];
    
    handleFilterChange('categories', newCategories);
  };

  const handlePOISelect = (poi) => {
    const isSelected = selectedPOIs.some(p => p.id === poi.id);
    
    if (isSelected) {
      setSelectedPOIs(selectedPOIs.filter(p => p.id !== poi.id));
    } else {
      setSelectedPOIs([...selectedPOIs, poi]);
    }
  };

  const handlePlanRoute = async () => {
    if (selectedPOIs.length === 0) return;
    
    try {
      const poiIds = selectedPOIs.map(poi => poi.id);
      await planRoute(poiIds, {
        transport: preferences.transportMode,
        algorithm: 'tsp_nearest'
      });
    } catch (error) {
      console.error('Failed to plan route:', error);
    }
  };

  const getCategoryInfo = (categoryValue) => {
    return categories.find(cat => cat.value === categoryValue) || 
           { label: categoryValue, color: 'bg-gray-100 text-gray-800' };
  };

  const formatDistance = (distance) => {
    if (distance < 1) {
      return `${Math.round(distance * 1000)}m`;
    }
    return `${distance.toFixed(1)}km`;
  };

  const formatWalkingTime = (timeMinutes) => {
    if (timeMinutes < 60) {
      return `${Math.round(timeMinutes)}min walk`;
    }
    return `${Math.round(timeMinutes / 60)}h ${Math.round(timeMinutes % 60)}min walk`;
  };

  return (
    <div className="poi-recommendations bg-white rounded-lg shadow-md p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">
          POI Recommendations
          {recommendations.length > 0 && (
            <span className="ml-2 text-sm text-gray-500">({recommendations.length})</span>
          )}
        </h3>
        
        <button
          onClick={handleLoadRecommendations}
          disabled={recommendationsLoading || !currentLocation}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {recommendationsLoading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      {/* Filters Section */}
      {showFilters && (
        <div className="mb-6 p-4 bg-gray-50 rounded-md space-y-4">
          <h4 className="font-medium text-gray-700">Filters</h4>
          
          {/* Categories */}
          <div>
            <label className="block text-sm font-medium text-gray-600 mb-2">Categories</label>
            <div className="flex flex-wrap gap-2">
              {categories.map(category => (
                <button
                  key={category.value}
                  onClick={() => handleCategoryToggle(category.value)}
                  className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                    filters.categories.includes(category.value)
                      ? `${category.color} ring-2 ring-offset-1 ring-blue-500`
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  {category.label}
                </button>
              ))}
            </div>
          </div>

          {/* Other Filters */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">
                Radius: {filters.radius}km
              </label>
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.5"
                value={filters.radius}
                onChange={(e) => handleFilterChange('radius', parseFloat(e.target.value))}
                className="w-full"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">
                Min Rating: {filters.rating || 'Any'}
              </label>
              <select
                value={filters.rating}
                onChange={(e) => handleFilterChange('rating', parseFloat(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="0">Any Rating</option>
                <option value="3">3.0+ Stars</option>
                <option value="3.5">3.5+ Stars</option>
                <option value="4">4.0+ Stars</option>
                <option value="4.5">4.5+ Stars</option>
              </select>
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="openNow"
                checked={filters.openNow}
                onChange={(e) => handleFilterChange('openNow', e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="openNow" className="text-sm font-medium text-gray-600">
                Open Now Only
              </label>
            </div>
          </div>

          <button
            onClick={handleLoadRecommendations}
            disabled={recommendationsLoading}
            className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            Apply Filters
          </button>
        </div>
      )}

      {/* Loading State */}
      {recommendationsLoading && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading recommendations...</p>
        </div>
      )}

      {/* Error State */}
      {recommendationsError && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
          <p className="text-red-800">
            <strong>Error:</strong> {recommendationsError}
          </p>
        </div>
      )}

      {/* No Location State */}
      {!currentLocation && !recommendationsLoading && (
        <div className="text-center py-8 text-gray-500">
          <p>Enable location tracking to get personalized recommendations</p>
        </div>
      )}

      {/* Recommendations List */}
      {recommendations.length > 0 && (
        <div className="space-y-4">
          {recommendations.map((recommendation) => {
            const poi = recommendation.poi;
            const categoryInfo = getCategoryInfo(poi.category);
            const isSelected = selectedPOIs.some(p => p.id === poi.id);
            
            return (
              <div
                key={poi.id}
                className={`border rounded-lg p-4 cursor-pointer transition-all ${
                  isSelected 
                    ? 'border-blue-500 bg-blue-50 shadow-md' 
                    : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
                }`}
                onClick={() => handlePOISelect(poi)}
              >
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800 flex items-center">
                      {poi.name}
                      {isSelected && <span className="ml-2 text-blue-600">✓</span>}
                    </h4>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium ${categoryInfo.color}`}>
                        {categoryInfo.label}
                      </span>
                      {poi.rating && (
                        <span className="text-yellow-500">
                          {'★'.repeat(Math.floor(poi.rating))} {poi.rating.toFixed(1)}
                        </span>
                      )}
                      {poi.is_open !== undefined && (
                        <span className={`text-xs px-2 py-1 rounded ${
                          poi.is_open ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                        }`}>
                          {poi.is_open ? 'Open' : 'Closed'}
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="text-right text-sm text-gray-600">
                    <div>{formatDistance(recommendation.distance_km)}</div>
                    <div>{formatWalkingTime(recommendation.walking_time_minutes)}</div>
                  </div>
                </div>
                
                {poi.description && (
                  <p className="text-sm text-gray-600 mb-2">{poi.description}</p>
                )}
                
                {poi.estimated_visit_duration && (
                  <p className="text-xs text-gray-500">
                    Estimated visit: {poi.estimated_visit_duration} minutes
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Route Planning Section */}
      {selectedPOIs.length > 0 && (
        <div className="mt-6 p-4 bg-blue-50 rounded-md">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-gray-800">
              Selected POIs ({selectedPOIs.length})
            </h4>
            <button
              onClick={() => setSelectedPOIs([])}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear All
            </button>
          </div>
          
          <div className="flex flex-wrap gap-2 mb-3">
            {selectedPOIs.map(poi => (
              <span
                key={poi.id}
                className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm"
              >
                {poi.name}
              </span>
            ))}
          </div>
          
          <button
            onClick={handlePlanRoute}
            className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Plan Route to Selected POIs
          </button>
        </div>
      )}

      {/* Empty State */}
      {!recommendationsLoading && !recommendationsError && recommendations.length === 0 && currentLocation && (
        <div className="text-center py-8 text-gray-500">
          <p>No recommendations found for your current location and filters.</p>
          <p className="text-sm mt-1">Try adjusting your filters or radius.</p>
        </div>
      )}
    </div>
  );
};

export default POIRecommendations;
