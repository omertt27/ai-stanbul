import { useState, useEffect } from 'react';

/**
 * Custom React Hook for Mobile Utilities
 * Provides mobile-specific functionality and device detection
 */
export const useMobileUtils = () => {
  const [isMobile, setIsMobile] = useState(false);
  const [isTouch, setIsTouch] = useState(false);
  const [orientation, setOrientation] = useState('portrait');
  const [screenSize, setScreenSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 0,
    height: typeof window !== 'undefined' ? window.innerHeight : 0
  });

  // Detect mobile device
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth <= 768 || 
                   /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
      setIsMobile(mobile);
    };

    const checkTouch = () => {
      setIsTouch('ontouchstart' in window || navigator.maxTouchPoints > 0);
    };

    const updateScreenSize = () => {
      setScreenSize({
        width: window.innerWidth,
        height: window.innerHeight
      });
      
      // Update orientation
      setOrientation(window.innerWidth > window.innerHeight ? 'landscape' : 'portrait');
    };

    checkMobile();
    checkTouch();
    updateScreenSize();

    window.addEventListener('resize', () => {
      checkMobile();
      updateScreenSize();
    });

    window.addEventListener('orientationchange', updateScreenSize);

    return () => {
      window.removeEventListener('resize', checkMobile);
      window.removeEventListener('resize', updateScreenSize);
      window.removeEventListener('orientationchange', updateScreenSize);
    };
  }, []);

  // Add haptic feedback (if available)
  const hapticFeedback = (type = 'light') => {
    if ('vibrate' in navigator) {
      switch (type) {
        case 'light':
          navigator.vibrate(10);
          break;
        case 'medium':
          navigator.vibrate(20);
          break;
        case 'heavy':
          navigator.vibrate([30, 10, 30]);
          break;
        case 'success':
          navigator.vibrate([100, 30, 100]);
          break;
        case 'error':
          navigator.vibrate([200, 100, 200]);
          break;
        default:
          navigator.vibrate(10);
      }
    }
  };

  // Prevent zoom on iOS
  const preventZoom = () => {
    if (isMobile) {
      const viewport = document.querySelector('meta[name="viewport"]');
      if (viewport) {
        viewport.setAttribute('content', 
          'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'
        );
      }
    }
  };

  // Restore zoom capability
  const restoreZoom = () => {
    const viewport = document.querySelector('meta[name="viewport"]');
    if (viewport) {
      viewport.setAttribute('content', 
        'width=device-width, initial-scale=1.0'
      );
    }
  };

  // Add to home screen prompt
  const [deferredPrompt, setDeferredPrompt] = useState(null);
  const [showInstallButton, setShowInstallButton] = useState(false);

  useEffect(() => {
    const handleBeforeInstallPrompt = (e) => {
      // Prevent the mini-infobar from appearing on mobile
      e.preventDefault();
      setDeferredPrompt(e);
      setShowInstallButton(true);
    };

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt);

    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt);
    };
  }, []);

  const installPWA = async () => {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      console.log(`User ${outcome} the install prompt`);
      setDeferredPrompt(null);
      setShowInstallButton(false);
    }
  };

  // Swipe gesture detection
  const [swipeStart, setSwipeStart] = useState(null);
  
  const handleTouchStart = (e) => {
    if (isTouch) {
      setSwipeStart({
        x: e.touches[0].clientX,
        y: e.touches[0].clientY,
        time: Date.now()
      });
    }
  };

  const handleTouchEnd = (e, onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown) => {
    if (!swipeStart || !isTouch) return;

    const touchEnd = {
      x: e.changedTouches[0].clientX,
      y: e.changedTouches[0].clientY,
      time: Date.now()
    };

    const deltaX = touchEnd.x - swipeStart.x;
    const deltaY = touchEnd.y - swipeStart.y;
    const deltaTime = touchEnd.time - swipeStart.time;
    
    const minSwipeDistance = 50;
    const maxSwipeTime = 300;

    if (deltaTime < maxSwipeTime) {
      if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > minSwipeDistance) {
        // Horizontal swipe
        if (deltaX > 0 && onSwipeRight) {
          onSwipeRight();
          hapticFeedback('light');
        } else if (deltaX < 0 && onSwipeLeft) {
          onSwipeLeft();
          hapticFeedback('light');
        }
      } else if (Math.abs(deltaY) > minSwipeDistance) {
        // Vertical swipe
        if (deltaY > 0 && onSwipeDown) {
          onSwipeDown();
          hapticFeedback('light');
        } else if (deltaY < 0 && onSwipeUp) {
          onSwipeUp();
          hapticFeedback('light');
        }
      }
    }

    setSwipeStart(null);
  };

  // Breakpoint detection
  const breakpoints = {
    xs: screenSize.width < 480,
    sm: screenSize.width >= 480 && screenSize.width < 640,
    md: screenSize.width >= 640 && screenSize.width < 768,
    lg: screenSize.width >= 768 && screenSize.width < 1024,
    xl: screenSize.width >= 1024
  };

  return {
    // Device detection
    isMobile,
    isTouch,
    orientation,
    screenSize,
    breakpoints,
    
    // PWA features
    showInstallButton,
    installPWA,
    
    // Touch interactions
    hapticFeedback,
    handleTouchStart,
    handleTouchEnd,
    
    // Utilities
    preventZoom,
    restoreZoom
  };
};

/**
 * Mobile Swipe Component
 * Wrapper component for easy swipe gesture handling
 */
export const MobileSwipe = ({ 
  children, 
  onSwipeLeft, 
  onSwipeRight, 
  onSwipeUp, 
  onSwipeDown,
  className = '' 
}) => {
  const { handleTouchStart, handleTouchEnd, isTouch } = useMobileUtils();

  if (!isTouch) {
    return <div className={className}>{children}</div>;
  }

  return (
    <div 
      className={`swipeable ${className}`}
      onTouchStart={handleTouchStart}
      onTouchEnd={(e) => handleTouchEnd(e, onSwipeLeft, onSwipeRight, onSwipeUp, onSwipeDown)}
    >
      {children}
    </div>
  );
};

/**
 * Install PWA Button Component
 */
export const InstallPWAButton = ({ className = '' }) => {
  const { showInstallButton, installPWA } = useMobileUtils();

  if (!showInstallButton) return null;

  return (
    <button 
      onClick={installPWA}
      className={`install-pwa-btn haptic-feedback ${className}`}
      style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        background: 'linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%)',
        color: 'white',
        border: 'none',
        borderRadius: '25px',
        padding: '12px 20px',
        fontSize: '14px',
        fontWeight: '600',
        zIndex: 1000,
        boxShadow: '0 4px 15px rgba(99, 102, 241, 0.3)',
        cursor: 'pointer',
        transition: 'all 0.3s ease'
      }}
    >
      📱 Install App
    </button>
  );
};

export default useMobileUtils;
