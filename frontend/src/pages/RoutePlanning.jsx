/**
 * Route Planning Page
 * Phase 3: Complete route planning interface with map integration
 */

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import RouteBuilder from '../components/RouteBuilder';
import RouteMap from '../components/RouteMap';
import { getRouteInfo, getDistrictStatus, getCacheStats } from '../api/routeApi';

const RoutePlanning = () => {
  const navigate = useNavigate();
  const [serviceInfo, setServiceInfo] = useState(null);
  const [districtStatus, setDistrictStatus] = useState(null);
  const [cacheStats, setCacheStats] = useState(null);
  const [createdRoute, setCreatedRoute] = useState(null);
  const [showQuickStart, setShowQuickStart] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  // Load service information on mount
  useEffect(() => {
    const loadServiceInfo = async () => {
      try {
        const [info, status, cache] = await Promise.allSettled([
          getRouteInfo(),
          getDistrictStatus(),
          getCacheStats()
        ]);
        
        if (info.status === 'fulfilled') setServiceInfo(info.value);
        if (status.status === 'fulfilled') setDistrictStatus(status.value);
        if (cache.status === 'fulfilled') setCacheStats(cache.value);
        
      } catch (error) {
        console.warn('Failed to load service info:', error);
      } finally {
        setIsLoading(false);
      }
    };
    
    loadServiceInfo();
  }, []);

  // Quick start locations
  const quickStartLocations = [
    {
      name: "Sultanahmet (Historic)",
      description: "Blue Mosque, Hagia Sophia, Topkapi Palace",
      lat: 41.0086,
      lng: 28.9802,
      icon: "🏛️"
    },
    {
      name: "Galata & Beyoğlu",
      description: "Galata Tower, Istiklal Street, Taksim",
      lat: 41.0290,
      lng: 28.9742,
      icon: "🗼"
    },
    {
      name: "Kadıköy (Asian Side)",
      description: "Moda, Fenerbahçe, Local markets",
      lat: 40.9833,
      lng: 29.0331,
      icon: "🛥️"
    },
    {
      name: "Beşiktaş",
      description: "Dolmabahçe Palace, Ortaköy, Bosphorus",
      lat: 41.0422,
      lng: 29.0008,
      icon: "🏰"
    }
  ];

  const handleQuickStart = (location) => {
    setShowQuickStart(false);
    // The RouteBuilder will use this location as starting point
  };

  const handleRouteCreated = (route) => {
    setCreatedRoute(route);
    setShowQuickStart(false);
  };

  const ServiceStatus = () => (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <span className="mr-2">⚙️</span>
        Route Maker Status
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {serviceInfo && (
          <div className="bg-green-50 p-4 rounded border-l-4 border-green-400">
            <div className="text-sm font-medium text-green-800">Service</div>
            <div className="text-lg font-semibold text-green-900">
              {serviceInfo.llm_free ? '✅ LLM-Free' : '⚠️ LLM-Dependent'}
            </div>
            <div className="text-sm text-green-700">
              Max {serviceInfo.max_attractions} attractions
            </div>
          </div>
        )}
        
        {districtStatus && (
          <div className="bg-blue-50 p-4 rounded border-l-4 border-blue-400">
            <div className="text-sm font-medium text-blue-800">Coverage</div>
            <div className="text-lg font-semibold text-blue-900">
              {districtStatus.available_districts?.length || 0} Districts
            </div>
            <div className="text-sm text-blue-700">
              Primary: {districtStatus.primary_district}
            </div>
          </div>
        )}
        
        {cacheStats && (
          <div className="bg-purple-50 p-4 rounded border-l-4 border-purple-400">
            <div className="text-sm font-medium text-purple-800">Performance</div>
            <div className="text-lg font-semibold text-purple-900">
              {cacheStats.hit_rate ? `${(cacheStats.hit_rate * 100).toFixed(1)}%` : 'N/A'} Hit Rate
            </div>
            <div className="text-sm text-purple-700">
              {cacheStats.total_requests || 0} requests
            </div>
          </div>
        )}
      </div>
      
      {districtStatus?.available_districts && (
        <div className="mt-4">
          <div className="text-sm font-medium text-gray-700 mb-2">Available Districts:</div>
          <div className="flex flex-wrap gap-2">
            {districtStatus.available_districts.map(district => (
              <span 
                key={district}
                className={`px-3 py-1 rounded-full text-sm ${
                  district === districtStatus.primary_district
                    ? 'bg-blue-100 text-blue-800 font-medium'
                    : 'bg-gray-100 text-gray-700'
                }`}
              >
                {district}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const QuickStartSection = () => (
    <div className="bg-white rounded-lg shadow-md p-6 mb-6">
      <h3 className="text-lg font-semibold mb-4 flex items-center">
        <span className="mr-2">🚀</span>
        Quick Start Locations
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {quickStartLocations.map((location, index) => (
          <button
            key={index}
            onClick={() => handleQuickStart(location)}
            className="p-4 text-left border-2 border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors"
          >
            <div className="flex items-start space-x-3">
              <span className="text-2xl">{location.icon}</span>
              <div>
                <h4 className="font-medium text-gray-900">{location.name}</h4>
                <p className="text-sm text-gray-600 mt-1">{location.description}</p>
              </div>
            </div>
          </button>
        ))}
      </div>
      
      <div className="mt-4 text-center">
        <button
          onClick={() => setShowQuickStart(false)}
          className="text-blue-600 hover:text-blue-700 text-sm"
        >
          Or create a custom route →
        </button>
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <div className="text-gray-600">Loading route planner...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                <span className="mr-3">🗺️</span>
                Istanbul Route Planner
              </h1>
              <p className="text-gray-600 mt-1">
                Create optimized walking routes with AI-free, map-based planning
              </p>
            </div>
            <button
              onClick={() => navigate('/')}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg transition-colors"
            >
              ← Back to Home
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <ServiceStatus />
        
        {showQuickStart && <QuickStartSection />}
        
        {/* Main Route Builder */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold mb-6 flex items-center">
            <span className="mr-2">🎯</span>
            {createdRoute ? 'Your Route' : 'Plan Your Route'}
          </h2>
          
          <RouteBuilder
            initialLocation={
              quickStartLocations.find(loc => !showQuickStart) || 
              { lat: 41.0082, lng: 28.9784 }
            }
            onRouteCreated={handleRouteCreated}
            className="w-full"
          />
        </div>

        {/* Route Summary (when route is created) */}
        {createdRoute && (
          <div className="mt-6 bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <span className="mr-2">📋</span>
              Route Summary
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-3">{createdRoute.name}</h4>
                <p className="text-gray-600 mb-4">{createdRoute.description}</p>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span>Total Distance:</span>
                    <span className="font-medium">{createdRoute.total_distance_km.toFixed(1)} km</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Estimated Duration:</span>
                    <span className="font-medium">{createdRoute.estimated_duration_hours.toFixed(1)} hours</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Number of Stops:</span>
                    <span className="font-medium">{createdRoute.points.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Overall Score:</span>
                    <span className="font-medium">{createdRoute.overall_score.toFixed(1)}/10</span>
                  </div>
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-3">Route Stops</h4>
                <div className="space-y-2 text-sm max-h-32 overflow-y-auto">
                  {createdRoute.points.map((point, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <span className="w-6 h-6 bg-blue-100 text-blue-800 text-xs font-medium rounded-full flex items-center justify-center">
                        {index + 1}
                      </span>
                      <div className="flex-1">
                        <div className="font-medium">{point.name}</div>
                        <div className="text-gray-500">{point.category}</div>
                      </div>
                      {point.arrival_time && (
                        <span className="text-gray-500 text-xs">{point.arrival_time}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="mt-6 flex space-x-3">
              <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
                Save Route
              </button>
              <button className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700">
                Export Route
              </button>
              <button className="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg hover:bg-gray-300">
                Share Route
              </button>
            </div>
          </div>
        )}

        {/* Tips Section */}
        <div className="mt-6 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3 flex items-center">
            <span className="mr-2">💡</span>
            Route Planning Tips
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
            <div>
              <div className="font-medium mb-1">⏰ Best Times to Visit</div>
              <div>Start early (9-10 AM) to avoid crowds at popular attractions</div>
            </div>
            <div>
              <div className="font-medium mb-1">🚶 Walking Routes</div>
              <div>Most Istanbul attractions are walkable within districts</div>
            </div>
            <div>
              <div className="font-medium mb-1">🍽️ Food Stops</div>
              <div>Include meal breaks every 3-4 hours for optimal experience</div>
            </div>
            <div>
              <div className="font-medium mb-1">🗺️ District Switching</div>
              <div>Use ferry or metro to move between European and Asian sides</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoutePlanning;
