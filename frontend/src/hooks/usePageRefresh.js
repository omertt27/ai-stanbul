import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * Custom hook to handle page refresh issues when navigating between routes
 * This ensures components properly reset state and reload data when navigated to
 */
export const usePageRefresh = (callback, dependencies = []) => {
  const location = useLocation();
  const isFirstRender = useRef(true);
  const previousPath = useRef(location.pathname);

  useEffect(() => {
    // Skip on first render
    if (isFirstRender.current) {
      isFirstRender.current = false;
      previousPath.current = location.pathname;
      
      // Call callback on initial mount
      if (callback) {
        callback();
      }
      return;
    }

    // If the path changed, it means we navigated to this page
    if (previousPath.current !== location.pathname) {
      console.log(`Page refresh triggered: ${previousPath.current} → ${location.pathname}`);
      previousPath.current = location.pathname;
      
      // Small delay to ensure DOM is ready and state is clean
      setTimeout(() => {
        if (callback) {
          callback();
        }
      }, 50);
    }
  }, [location.pathname, ...dependencies]);

  // Also trigger on dependencies change
  useEffect(() => {
    if (!isFirstRender.current && callback) {
      callback();
    }
  }, dependencies);
};

/**
 * Hook specifically for components that need to reset state on navigation
 */
export const useStateReset = (resetFunction) => {
  const location = useLocation();
  
  useEffect(() => {
    // Reset state when location changes
    if (resetFunction) {
      resetFunction();
    }
  }, [location.pathname]);
};

/**
 * Hook for components that load data and need to refresh on navigation
 */
export const useDataRefresh = (loadDataFunction, dependencies = []) => {
  return usePageRefresh(loadDataFunction, dependencies);
};
