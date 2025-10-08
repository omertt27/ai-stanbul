/**
 * LocationPage - Main page integrating the location system with the existing Istanbul AI app
 */

import React, { Suspense } from 'react';
import { LocationProvider } from '../contexts/LocationContext';
import LocationDashboard from '../components/LocationDashboard';

// Loading component
const LoadingSpinner = () => (
  <div className="min-h-screen bg-gray-100 flex items-center justify-center">
    <div className="text-center">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
      <p className="text-gray-600">Loading Istanbul Location System...</p>
    </div>
  </div>
);

// Error boundary component
class LocationErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Location system error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
          <div className="max-w-md w-full bg-white rounded-lg shadow-md p-6 text-center">
            <div className="text-4xl mb-4">⚠️</div>
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Location System Error
            </h2>
            <p className="text-gray-600 mb-4">
              Something went wrong with the location system. Please try refreshing the page.
            </p>
            <div className="space-y-2">
              <button
                onClick={() => window.location.reload()}
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Refresh Page
              </button>
              <button
                onClick={() => this.setState({ hasError: false, error: null })}
                className="w-full px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300"
              >
                Try Again
              </button>
            </div>
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="mt-4 text-left">
                <summary className="cursor-pointer text-sm text-gray-500">
                  Error Details (Development)
                </summary>
                <pre className="mt-2 text-xs text-red-600 bg-red-50 p-2 rounded overflow-auto">
                  {this.state.error.toString()}
                </pre>
              </details>
            )}
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

const LocationPage = () => {
  return (
    <LocationErrorBoundary>
      <LocationProvider>
        <Suspense fallback={<LoadingSpinner />}>
          <div className="location-page">
            {/* Header with navigation back to main app */}
            <div className="bg-blue-600 text-white">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-12">
                  <div className="flex items-center space-x-4">
                    <a
                      href="/"
                      className="text-sm text-blue-100 hover:text-white transition-colors"
                    >
                      ← Back to Istanbul AI
                    </a>
                    <span className="text-blue-200">|</span>
                    <span className="text-sm font-medium">Live Location & Routing</span>
                  </div>
                  
                  <div className="text-sm text-blue-100">
                    Real-time POI recommendations & route planning
                  </div>
                </div>
              </div>
            </div>

            {/* Main dashboard */}
            <LocationDashboard />

            {/* Footer */}
            <footer className="bg-gray-50 border-t border-gray-200">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-3">Features</h3>
                    <ul className="space-y-2 text-sm text-gray-600">
                      <li>• Real-time location tracking</li>
                      <li>• Smart POI recommendations</li>
                      <li>• Route optimization (TSP algorithms)</li>
                      <li>• Multi-transport mode support</li>
                      <li>• Offline capability</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-3">Privacy & Security</h3>
                    <ul className="space-y-2 text-sm text-gray-600">
                      <li>• Privacy-safe location hashing</li>
                      <li>• No location data stored permanently</li>
                      <li>• Session-based tracking only</li>
                      <li>• GDPR compliant</li>
                      <li>• Local processing preferred</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-3">Technical</h3>
                    <ul className="space-y-2 text-sm text-gray-600">
                      <li>• FastAPI backend</li>
                      <li>• React frontend</li>
                      <li>• Leaflet maps</li>
                      <li>• Algorithm-based routing</li>
                      <li>• No LLM dependencies</li>
                    </ul>
                  </div>
                </div>
                
                <div className="mt-8 pt-8 border-t border-gray-200 text-center text-sm text-gray-500">
                  <p>Istanbul AI Location System - Powered by algorithmic routing and real-time recommendations</p>
                </div>
              </div>
            </footer>
          </div>
        </Suspense>
      </LocationProvider>
    </LocationErrorBoundary>
  );
};

export default LocationPage;
