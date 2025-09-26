import React, { useEffect } from 'react';

/**
 * Mobile Optimization Component
 * Handles mobile-specific optimizations and PWA features
 */
const MobileOptimizer = () => {
  useEffect(() => {
    // Set viewport meta tag for proper mobile rendering
    const setViewport = () => {
      let viewport = document.querySelector("meta[name=viewport]");
      if (!viewport) {
        viewport = document.createElement('meta');
        viewport.name = 'viewport';
        document.getElementsByTagName('head')[0].appendChild(viewport);
      }
      
      // Optimized viewport for mobile chat interface
      viewport.content = 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=0, viewport-fit=cover';
    };

    // Prevent zoom on input focus (iOS Safari)
    const preventZoomOnFocus = () => {
      const inputs = document.querySelectorAll('input[type="text"], textarea');
      inputs.forEach(input => {
        if (!input.style.fontSize || parseFloat(input.style.fontSize) < 16) {
          input.style.fontSize = '16px';
        }
      });
    };

    // Handle iOS Safari bottom bar behavior
    const handleSafariBottomBar = () => {
      if (/iPhone|iPad|iPod/.test(navigator.userAgent) && /Safari/.test(navigator.userAgent) && !/Chrome/.test(navigator.userAgent)) {
        // Add class to handle iOS Safari's dynamic viewport
        document.body.classList.add('ios-safari');
        
        // Listen for viewport height changes
        const setViewportHeight = () => {
          const vh = window.innerHeight * 0.01;
          document.documentElement.style.setProperty('--vh', `${vh}px`);
        };
        
        setViewportHeight();
        window.addEventListener('resize', setViewportHeight);
        window.addEventListener('orientationchange', setViewportHeight);
        
        return () => {
          window.removeEventListener('resize', setViewportHeight);
          window.removeEventListener('orientationchange', setViewportHeight);
        };
      }
    };

    // Improve touch scrolling on iOS
    const enableMomentumScrolling = () => {
      const scrollElements = document.querySelectorAll('.chatbot-messages, .chatbot-scrollable');
      scrollElements.forEach(element => {
        element.style.webkitOverflowScrolling = 'touch';
        element.style.overscrollBehavior = 'contain';
      });
    };

    // Add touch gesture support
    const addTouchGestures = () => {
      let touchStartY = 0;
      let touchEndY = 0;
      
      const handleTouchStart = (e) => {
        touchStartY = e.changedTouches[0].screenY;
      };
      
      const handleTouchEnd = (e) => {
        touchEndY = e.changedTouches[0].screenY;
        handleSwipe();
      };
      
      const handleSwipe = () => {
        const swipeThreshold = 50;
        const swipeDistance = touchStartY - touchEndY;
        
        if (Math.abs(swipeDistance) > swipeThreshold) {
          if (swipeDistance > 0) {
            // Swiped up - could trigger "scroll to bottom" or hide keyboard
            const activeElement = document.activeElement;
            if (activeElement && activeElement.tagName === 'INPUT') {
              activeElement.blur();
            }
          }
          // Swiped down - could trigger "scroll to top" or refresh
        }
      };
      
      document.addEventListener('touchstart', handleTouchStart, { passive: true });
      document.addEventListener('touchend', handleTouchEnd, { passive: true });
      
      return () => {
        document.removeEventListener('touchstart', handleTouchStart);
        document.removeEventListener('touchend', handleTouchEnd);
      };
    };

    // Optimize for mobile performance
    const optimizeForMobile = () => {
      // Reduce motion for better performance on mobile
      if (window.matchMedia('(max-width: 768px)').matches) {
        document.body.classList.add('mobile-optimized');
        
        // Disable hover effects on touch devices
        if ('ontouchstart' in window) {
          document.body.classList.add('touch-device');
        }
      }
    };

    // Initialize all optimizations
    setViewport();
    preventZoomOnFocus();
    enableMomentumScrolling();
    optimizeForMobile();
    
    const cleanupSafari = handleSafariBottomBar();
    const cleanupGestures = addTouchGestures();

    // Cleanup function
    return () => {
      if (cleanupSafari) cleanupSafari();
      if (cleanupGestures) cleanupGestures();
    };
  }, []);

  return null; // This component doesn't render anything
};

export default MobileOptimizer;
