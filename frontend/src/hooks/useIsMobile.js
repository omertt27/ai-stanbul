/**
 * useIsMobile Hook
 * =================
 * Reactive mobile detection hook using matchMedia API
 * 
 * Features:
 * - Uses matchMedia for proper mobile detection
 * - Reacts to screen resize and orientation changes
 * - SSR-safe (doesn't crash on server-side rendering)
 * - Cleaner than inline window.innerWidth checks
 * 
 * Usage:
 *   const isMobile = useIsMobile();
 *   {isMobile ? <MobileComponent /> : <DesktopComponent />}
 */

import { useState, useEffect } from 'react';

const useIsMobile = (breakpoint = 768) => {
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    // Check if window is available (not SSR)
    if (typeof window === 'undefined') {
      return;
    }

    // Create media query
    const mediaQuery = window.matchMedia(`(max-width: ${breakpoint}px)`);
    
    // Set initial value
    setIsMobile(mediaQuery.matches);
    
    // Handler for media query changes
    const handleChange = (e) => {
      setIsMobile(e.matches);
    };
    
    // Add listener (modern API)
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
      
      return () => {
        mediaQuery.removeEventListener('change', handleChange);
      };
    } else {
      // Fallback for older browsers
      mediaQuery.addListener(handleChange);
      
      return () => {
        mediaQuery.removeListener(handleChange);
      };
    }
  }, [breakpoint]);

  return isMobile;
};

export default useIsMobile;
