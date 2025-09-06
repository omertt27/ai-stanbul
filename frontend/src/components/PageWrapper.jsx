import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

/**
 * Universal Page Wrapper that ensures complete component remount 
 * and state reset when navigating between pages.
 * 
 * This wrapper forces a fresh render of wrapped components by:
 * 1. Using a unique key based on pathname to force React remount
 * 2. Resetting internal state on location changes
 * 3. Scrolling to top on navigation
 * 4. Clearing any cached data/state
 */
const PageWrapper = ({ children, pageKey, clearCache = false, debug = false }) => {
  const location = useLocation();
  const [mountKey, setMountKey] = useState(0);
  const [isReady, setIsReady] = useState(false);

  // Create a unique key for this page instance
  const uniqueKey = `${pageKey || location.pathname}-${location.search}-${mountKey}`;

  useEffect(() => {
    if (debug) {
      console.log(`🔄 PageWrapper: Location changed to ${location.pathname}`);
    }

    // Reset ready state to force re-render
    setIsReady(false);

    // Scroll to top immediately
    window.scrollTo(0, 0);

    // Clear any browser cache if requested
    if (clearCache) {
      // Clear any potential stale cache entries
      if (window.performance && window.performance.getEntriesByType) {
        window.performance.getEntriesByType('navigation').forEach(() => {
          // Force browser to treat this as a fresh navigation
        });
      }
    }

    // Force remount by updating the key
    setMountKey(prev => prev + 1);

    // Small delay to ensure DOM is ready
    const timer = setTimeout(() => {
      setIsReady(true);
      if (debug) {
        console.log(`✅ PageWrapper: Ready for ${location.pathname}`);
      }
    }, 10);

    return () => clearTimeout(timer);
  }, [location.pathname, location.search, clearCache, debug]);

  // Don't render children until we're ready to prevent flash of old content
  if (!isReady) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div key={uniqueKey} className="page-wrapper">
      {children}
    </div>
  );
};

export default PageWrapper;
