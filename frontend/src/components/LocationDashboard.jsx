/**
 * LocationDashboard - Main dashboard combining all location and routing features
 */

import React, { useState, useEffect } from 'react';
import { useLocation } from '../contexts/LocationContext';
import LocationTracker from './LocationTracker';
import POIRecommendations from './POIRecommendations';
import RouteDisplay from './RouteDisplay';
import InteractiveMap from './InteractiveMap';

const LocationDashboard = () => {
  const {
    currentLocation,
    sessionActive,
    hasLocation,
    hasRecommendations,
    hasRoute,
    preferences,
    updatePreferences,
    cleanupSession
  } = useLocation();

  const [activeTab, setActiveTab] = useState('overview');
  const [dashboardView, setDashboardView] = useState('split'); // 'split', 'map', 'panels'
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Auto-cleanup session on unmount
  useEffect(() => {
    return () => {
      // Cleanup session when component unmounts
      cleanupSession();
    };
  }, []);

  const tabs = [
    { id: 'overview', label: '🏠 Overview', count: null },
    { id: 'recommendations', label: '🎯 Recommendations', count: hasRecommendations ? 'new' : null },
    { id: 'route', label: '🗺️ Route', count: hasRoute ? 'active' : null },
    { id: 'map', label: '📍 Map', count: null }
  ];

  const viewModes = [
    { id: 'split', label: '⊞ Split View', description: 'Map + Panels' },
    { id: 'map', label: '🗺️ Map Only', description: 'Full Map' },
    { id: 'panels', label: '📋 Panels Only', description: 'No Map' }
  ];

  const handleLocationUpdate = (location) => {
    console.log('Location updated in dashboard:', location);
  };

  const handlePreferenceUpdate = (newPrefs) => {
    updatePreferences(newPrefs);
  };

  const renderOverviewTab = () => (
    <div className="space-y-6">
      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className={`p-4 rounded-lg border-2 ${hasLocation ? 'border-green-200 bg-green-50' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-gray-800">Location Status</h4>
              <p className="text-sm text-gray-600">
                {hasLocation ? 'Location Available' : 'No Location'}
              </p>
            </div>
            <div className={`text-2xl ${hasLocation ? 'text-green-600' : 'text-gray-400'}`}>
              📍
            </div>
          </div>
        </div>

        <div className={`p-4 rounded-lg border-2 ${sessionActive ? 'border-blue-200 bg-blue-50' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-gray-800">Session Status</h4>
              <p className="text-sm text-gray-600">
                {sessionActive ? 'Session Active' : 'No Session'}
              </p>
            </div>
            <div className={`text-2xl ${sessionActive ? 'text-blue-600' : 'text-gray-400'}`}>
              🔄
            </div>
          </div>
        </div>

        <div className={`p-4 rounded-lg border-2 ${hasRoute ? 'border-purple-200 bg-purple-50' : 'border-gray-200 bg-gray-50'}`}>
          <div className="flex items-center justify-between">
            <div>
              <h4 className="font-medium text-gray-800">Route Status</h4>
              <p className="text-sm text-gray-600">
                {hasRoute ? 'Route Planned' : 'No Route'}
              </p>
            </div>
            <div className={`text-2xl ${hasRoute ? 'text-purple-600' : 'text-gray-400'}`}>
              🛣️
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <button
            onClick={() => setActiveTab('recommendations')}
            className="p-3 text-center border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="text-2xl mb-1">🎯</div>
            <div className="text-sm font-medium text-gray-700">Find POIs</div>
          </button>
          
          <button
            onClick={() => setActiveTab('route')}
            className="p-3 text-center border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="text-2xl mb-1">🗺️</div>
            <div className="text-sm font-medium text-gray-700">Plan Route</div>
          </button>
          
          <button
            onClick={() => setActiveTab('map')}
            className="p-3 text-center border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="text-2xl mb-1">📍</div>
            <div className="text-sm font-medium text-gray-700">View Map</div>
          </button>
          
          <button
            onClick={() => setIsSettingsOpen(true)}
            className="p-3 text-center border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="text-2xl mb-1">⚙️</div>
            <div className="text-sm font-medium text-gray-700">Settings</div>
          </button>
        </div>
      </div>

      {/* Current Location Info */}
      {currentLocation && (
        <div className="bg-white rounded-lg shadow-md p-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Current Location</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">Coordinates</p>
              <p className="font-mono text-sm">
                {currentLocation.lat.toFixed(6)}, {currentLocation.lng.toFixed(6)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Accuracy</p>
              <p className="text-sm">
                {currentLocation.accuracy ? `±${Math.round(currentLocation.accuracy)}m` : 'Unknown'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return renderOverviewTab();
      case 'recommendations':
        return <POIRecommendations autoLoad={true} showFilters={true} />;
      case 'route':
        return <RouteDisplay showOptimization={true} showExport={true} />;
      case 'map':
        return (
          <div className="bg-white rounded-lg shadow-md p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Interactive Map</h3>
            <InteractiveMap 
              height="600px" 
              showUserLocation={true} 
              showPOIs={true} 
              showRoute={true}
              zoom={14}
            />
          </div>
        );
      default:
        return renderOverviewTab();
    }
  };

  return (
    <div className="location-dashboard min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-bold text-gray-900">
                🏛️ Istanbul AI - Location Dashboard
              </h1>
              {sessionActive && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  <span className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></span>
                  Live Session
                </span>
              )}
            </div>
            
            <div className="flex items-center space-x-4">
              {/* View Mode Selector */}
              <div className="hidden md:flex items-center space-x-2">
                {viewModes.map(mode => (
                  <button
                    key={mode.id}
                    onClick={() => setDashboardView(mode.id)}
                    className={`px-3 py-1 text-sm rounded-md transition-colors ${
                      dashboardView === mode.id
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                    title={mode.description}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>

              <button
                onClick={() => setIsSettingsOpen(true)}
                className="p-2 text-gray-400 hover:text-gray-600 rounded-md"
              >
                ⚙️
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {dashboardView === 'split' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Panel - Controls */}
            <div className="lg:col-span-1 space-y-6">
              <LocationTracker onLocationUpdate={handleLocationUpdate} autoStart={true} />
              {activeTab !== 'overview' && (
                <div className="bg-white rounded-lg shadow-md p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-semibold text-gray-800">Quick Switch</h3>
                    <button
                      onClick={() => setActiveTab('overview')}
                      className="text-sm text-blue-600 hover:text-blue-800"
                    >
                      Overview
                    </button>
                  </div>
                  {renderContent()}
                </div>
              )}
            </div>

            {/* Right Panel - Map */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-md p-4">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">Live Map</h3>
                  <button
                    onClick={() => setActiveTab('map')}
                    className="text-sm text-blue-600 hover:text-blue-800"
                  >
                    Full Screen
                  </button>
                </div>
                <InteractiveMap 
                  height="500px" 
                  showUserLocation={true} 
                  showPOIs={true} 
                  showRoute={true}
                  zoom={13}
                />
              </div>
            </div>
          </div>
        )}

        {dashboardView === 'map' && (
          <div className="space-y-6">
            <LocationTracker onLocationUpdate={handleLocationUpdate} autoStart={true} />
            <div className="bg-white rounded-lg shadow-md p-4">
              <InteractiveMap 
                height="70vh" 
                showUserLocation={true} 
                showPOIs={true} 
                showRoute={true}
                zoom={13}
              />
            </div>
          </div>
        )}

        {dashboardView === 'panels' && (
          <div className="space-y-6">
            {/* Navigation Tabs */}
            <div className="bg-white rounded-lg shadow-md">
              <div className="border-b border-gray-200">
                <nav className="flex space-x-8 px-6">
                  {tabs.map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                        activeTab === tab.id
                          ? 'border-blue-500 text-blue-600'
                          : 'border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700'
                      }`}
                    >
                      <div className="flex items-center space-x-2">
                        <span>{tab.label}</span>
                        {tab.count && (
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            tab.count === 'new' ? 'bg-green-100 text-green-800' :
                            tab.count === 'active' ? 'bg-blue-100 text-blue-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {tab.count}
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </nav>
              </div>

              <div className="p-6">
                {activeTab === 'overview' && <LocationTracker onLocationUpdate={handleLocationUpdate} autoStart={true} />}
                {renderContent()}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Settings Modal */}
      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-md w-full max-h-90vh overflow-y-auto">
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-800">Settings</h3>
                <button
                  onClick={() => setIsSettingsOpen(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
            </div>
            
            <div className="p-4 space-y-4">
              {/* Transport Mode */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Default Transport Mode
                </label>
                <select
                  value={preferences.transportMode || 'walking'}
                  onChange={(e) => handlePreferenceUpdate({ transportMode: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="walking">🚶 Walking</option>
                  <option value="public">🚌 Public Transport</option>
                  <option value="driving">🚗 Driving</option>
                  <option value="mixed">🔄 Mixed</option>
                </select>
              </div>

              {/* Default Radius */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Default Search Radius: {preferences.radius || 2.0}km
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="10"
                  step="0.5"
                  value={preferences.radius || 2.0}
                  onChange={(e) => handlePreferenceUpdate({ radius: parseFloat(e.target.value) })}
                  className="w-full"
                />
              </div>

              {/* Language */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Language
                </label>
                <select
                  value={preferences.language || 'en'}
                  onChange={(e) => handlePreferenceUpdate({ language: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="en">🇺🇸 English</option>
                  <option value="tr">🇹🇷 Türkçe</option>
                  <option value="de">🇩🇪 Deutsch</option>
                  <option value="fr">🇫🇷 Français</option>
                </select>
              </div>

              {/* Auto-tracking */}
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="autoTracking"
                  checked={preferences.autoTracking || false}
                  onChange={(e) => handlePreferenceUpdate({ autoTracking: e.target.checked })}
                  className="mr-2"
                />
                <label htmlFor="autoTracking" className="text-sm font-medium text-gray-700">
                  Enable automatic location tracking
                </label>
              </div>

              <div className="pt-4 border-t border-gray-200">
                <button
                  onClick={() => setIsSettingsOpen(false)}
                  className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Save Settings
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LocationDashboard;
