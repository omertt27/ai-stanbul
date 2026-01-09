/**
 * Location-Aware AI Chat Integration
 * 
 * This component enhances the AI chat system with location intelligence:
 * - Gets user's current location (with permission)
 * - Provides location-aware recommendations
 * - Integrates with POI and routing system
 * - Offers contextual advice based on user's position
 */

import React, { useState, useEffect, useRef } from 'react';
import { useLocation as useLocationContext } from '../contexts/LocationContext';
import locationApi from '../services/locationApi';

const LocationAwareChatEnhancer = ({ onLocationUpdate, onRecommendationsUpdate }) => {
  const {
    currentLocation,
    sessionId,
    sessionActive,
    recommendations,
    startLocationSession,
    updateLocation,
    getRecommendations,
    isTracking,
    startTracking,
    stopTracking
  } = useLocationContext();

  const [locationPermission, setLocationPermission] = useState('unknown');
  const [locationStatus, setLocationStatus] = useState('inactive');

  // Initialize location services
  useEffect(() => {
    if ('geolocation' in navigator) {
      // Check current permission status
      if (navigator.permissions) {
        navigator.permissions.query({ name: 'geolocation' }).then((result) => {
          setLocationPermission(result.state);
          if (result.state === 'granted') {
            initializeLocationServices();
          }
        });
      }
    }
  }, []);

  // Initialize location services when permission is granted
  const initializeLocationServices = async () => {
    try {
      setLocationStatus('initializing');
      
      // Get current position
      const position = await getCurrentPosition();
      
      // Start location session
      if (!sessionActive) {
        const preferences = {
          interests: ['historical', 'restaurants', 'museums'],
          budget: 'medium',
          mobility: 'walking',
          language: 'en'
        };
        
        await startLocationSession(position.coords, preferences);
      }
      
      setLocationStatus('active');
      
      // Notify parent components
      if (onLocationUpdate) {
        onLocationUpdate({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy
        });
      }
      
    } catch (error) {
      console.error('Failed to initialize location services:', error);
      setLocationStatus('error');
    }
  };

  // Get current position with promise wrapper
  const getCurrentPosition = () => {
    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        resolve,
        reject,
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000 // 5 minutes
        }
      );
    });
  };

  // Request location permission
  const requestLocationPermission = async () => {
    try {
      const position = await getCurrentPosition();
      setLocationPermission('granted');
      await initializeLocationServices();
      return true;
    } catch (error) {
      setLocationPermission('denied');
      console.error('Location permission denied:', error);
      return false;
    }
  };

  // Get location-aware recommendations for chat
  const getLocationAwareRecommendations = async (query, intent = 'general') => {
    if (!currentLocation || !sessionId) {
      return null;
    }

    try {
      // Map chat intent to location filters
      const filters = mapChatIntentToLocationFilters(intent, query);
      
      const recommendations = await getRecommendations(filters);
      
      if (onRecommendationsUpdate) {
        onRecommendationsUpdate(recommendations);
      }
      
      return recommendations;
    } catch (error) {
      console.error('Failed to get location recommendations:', error);
      return null;
    }
  };

  // Map chat intents to location system filters
  const mapChatIntentToLocationFilters = (intent, query) => {
    const queryLower = query.toLowerCase();
    
    let filters = {
      categories: [],
      cuisine_types: [],
      price_ranges: [],
      open_now: null,
      min_rating: null
    };

    // Restaurant queries
    if (intent === 'restaurant_search' || 
        queryLower.includes('restaurant') || 
        queryLower.includes('food') || 
        queryLower.includes('eat')) {
      filters.categories = ['restaurant'];
      
      // Cuisine detection
      if (queryLower.includes('turkish')) filters.cuisine_types.push('turkish');
      if (queryLower.includes('italian')) filters.cuisine_types.push('italian');
      if (queryLower.includes('seafood')) filters.cuisine_types.push('seafood');
      
      // Price detection
      if (queryLower.includes('cheap') || queryLower.includes('budget')) {
        filters.price_ranges = ['budget'];
      } else if (queryLower.includes('expensive') || queryLower.includes('luxury')) {
        filters.price_ranges = ['luxury'];
      }
      
      // Time-based filtering
      if (queryLower.includes('open') || queryLower.includes('now')) {
        filters.open_now = true;
      }
    }
    
    // Museum queries
    else if (intent === 'museum_inquiry' || 
             queryLower.includes('museum') || 
             queryLower.includes('art') || 
             queryLower.includes('gallery')) {
      filters.categories = ['museum'];
      filters.open_now = true;
    }
    
    // Attraction queries
    else if (intent === 'place_recommendation' || 
             queryLower.includes('attraction') || 
             queryLower.includes('visit') || 
             queryLower.includes('see')) {
      filters.categories = ['landmark', 'viewpoint', 'park'];
      filters.min_rating = 4.0;
    }
    
    // Shopping queries
    else if (queryLower.includes('shop') || 
             queryLower.includes('buy') || 
             queryLower.includes('market')) {
      filters.categories = ['shopping'];
    }

    return filters;
  };

  // Format location data for chat context
  const getLocationContextForChat = () => {
    if (!currentLocation) return null;

    return {
      hasLocation: true,
      latitude: currentLocation.latitude,
      longitude: currentLocation.longitude,
      accuracy: currentLocation.accuracy,
      sessionId: sessionId,
      district: currentLocation.district || 'Unknown',
      nearbyPOIs: recommendations.slice(0, 5), // Top 5 nearby POIs
      locationStatus: locationStatus
    };
  };

  // Start real-time location tracking for better recommendations
  const enableRealtimeTracking = async () => {
    if (locationPermission !== 'granted') {
      const granted = await requestLocationPermission();
      if (!granted) return false;
    }

    try {
      await startTracking();
      return true;
    } catch (error) {
      console.error('Failed to start tracking:', error);
      return false;
    }
  };

  return {
    // Status
    locationPermission,
    locationStatus,
    hasLocation: !!currentLocation,
    isTracking,
    
    // Actions
    requestLocationPermission,
    enableRealtimeTracking,
    getLocationAwareRecommendations,
    getLocationContextForChat,
    
    // Data
    currentLocation,
    recommendations,
    sessionId,
    sessionActive
  };
};

export default LocationAwareChatEnhancer;
