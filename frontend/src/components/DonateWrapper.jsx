import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Donate from '../pages/Donate';

/**
 * DonateWrapper component that ensures Donate page is completely remounted
 * and properly initialized every time user navigates to /donate
 */
const DonateWrapper = () => {
  const location = useLocation();
  const [donateKey, setDonateKey] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    console.log('🔄 DonateWrapper: Navigation detected to', location.pathname);
    
    // Only handle donate routes
    if (location.pathname === '/donate') {
      setIsLoading(true);
      
      // Force complete remount by changing key
      setDonateKey(prev => prev + 1);
      
      // Scroll to top
      window.scrollTo(0, 0);
      
      // Small delay to ensure clean state
      const timer = setTimeout(() => {
        setIsLoading(false);
        console.log('✅ DonateWrapper: Ready to render Donate');
      }, 30);
      
      return () => clearTimeout(timer);
    }
  }, [location.pathname]);

  // Show loading during transition to prevent stale content
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-500 mx-auto mb-4"></div>
          <p className="text-gray-300">Loading donate page...</p>
        </div>
      </div>
    );
  }

  // Render Donate with unique key to force complete remount
  return <Donate key={`donate-${donateKey}-${location.pathname}`} />;
};

export default DonateWrapper;
