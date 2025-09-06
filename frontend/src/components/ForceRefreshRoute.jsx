import React, { useEffect, useState, useMemo } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * ForceRefreshRoute - A route wrapper that forces complete component remount
 * This is the most aggressive approach to ensure no stale state persists
 */
const ForceRefreshRoute = ({ component: Component, componentProps = {}, routeName }) => {
  const location = useLocation();
  const [refreshKey, setRefreshKey] = useState(0);
  const [isReady, setIsReady] = useState(false);

  // Create a stable component key to avoid unnecessary re-renders
  const componentKey = useMemo(() => {
    return `${routeName}-${refreshKey}-${location.pathname}-${location.search}`;
  }, [routeName, refreshKey, location.pathname, location.search]);

  useEffect(() => {
    console.log(`🔄 ForceRefreshRoute [${routeName}]: Navigation to ${location.pathname}`);
    
    // Reset ready state
    setIsReady(false);
    
    // Force scroll to top immediately
    window.scrollTo(0, 0);
    
    // Clear any potential cache or stored state
    try {
      if (window.sessionStorage) {
        // Clear session-specific cache but keep user preferences
        const keysToRemove = [];
        for (let i = 0; i < window.sessionStorage.length; i++) {
          const key = window.sessionStorage.key(i);
          if (key && (key.includes('cache') || key.includes('temp') || key.includes('state'))) {
            keysToRemove.push(key);
          }
        }
        keysToRemove.forEach(key => window.sessionStorage.removeItem(key));
      }
    } catch (error) {
      console.warn('ForceRefreshRoute: Could not clear session storage:', error);
    }
    
    // Force refresh by updating key
    setRefreshKey(prev => prev + 1);
    
    // Small delay to ensure clean state transition
    const timer = setTimeout(() => {
      setIsReady(true);
      console.log(`✅ ForceRefreshRoute [${routeName}]: Ready to render component`);
    }, 30); // Reduced delay for better UX

    return () => clearTimeout(timer);
  }, [location.pathname, location.search, routeName]);

  // Show loading state during transition
  if (!isReady) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-300">Loading {routeName}...</p>
        </div>
      </div>
    );
  }

  // Render component with unique key for forced remount
  return <Component key={componentKey} {...componentProps} />;
};

export default ForceRefreshRoute;
