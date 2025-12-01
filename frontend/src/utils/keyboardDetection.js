/**
 * Keyboard Detection Utility
 * ===========================
 * Detects virtual keyboard visibility on mobile devices
 * 
 * Features:
 * - Visual viewport API support
 * - Fallback to window resize detection
 * - iOS and Android compatibility
 * - Keyboard height estimation
 * 
 * Usage:
 *   const { isKeyboardVisible, keyboardHeight } = useKeyboardDetection();
 */

import { useState, useEffect } from 'react';

/**
 * Detects if virtual keyboard is visible
 * @returns {Object} { isKeyboardVisible, keyboardHeight }
 */
export const useKeyboardDetection = () => {
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  const [keyboardHeight, setKeyboardHeight] = useState(0);
  const [initialHeight, setInitialHeight] = useState(0);

  useEffect(() => {
    // Store initial viewport height
    const initHeight = window.visualViewport?.height || window.innerHeight;
    setInitialHeight(initHeight);

    // Modern approach: Visual Viewport API
    if ('visualViewport' in window && window.visualViewport) {
      const handleViewportResize = () => {
        const viewport = window.visualViewport;
        const currentHeight = viewport.height;
        const screenHeight = window.screen.height;
        
        // Keyboard is visible if viewport shrunk significantly
        // Use 150px threshold to avoid false positives from browser UI
        const heightDiff = initHeight - currentHeight;
        const isVisible = heightDiff > 150;
        
        setIsKeyboardVisible(isVisible);
        setKeyboardHeight(isVisible ? heightDiff : 0);
        
        // Add class to body for CSS-based adjustments
        if (isVisible) {
          document.body.classList.add('keyboard-visible');
          document.body.style.setProperty('--keyboard-height', `${heightDiff}px`);
        } else {
          document.body.classList.remove('keyboard-visible');
          document.body.style.removeProperty('--keyboard-height');
        }
      };

      window.visualViewport.addEventListener('resize', handleViewportResize);
      window.visualViewport.addEventListener('scroll', handleViewportResize);

      return () => {
        window.visualViewport.removeEventListener('resize', handleViewportResize);
        window.visualViewport.removeEventListener('scroll', handleViewportResize);
        document.body.classList.remove('keyboard-visible');
        document.body.style.removeProperty('--keyboard-height');
      };
    }
    
    // Fallback: Window resize detection (less reliable)
    else {
      const handleResize = () => {
        const currentHeight = window.innerHeight;
        const heightDiff = initHeight - currentHeight;
        const isVisible = heightDiff > 150;
        
        setIsKeyboardVisible(isVisible);
        setKeyboardHeight(isVisible ? heightDiff : 0);
        
        if (isVisible) {
          document.body.classList.add('keyboard-visible');
          document.body.style.setProperty('--keyboard-height', `${heightDiff}px`);
        } else {
          document.body.classList.remove('keyboard-visible');
          document.body.style.removeProperty('--keyboard-height');
        }
      };

      window.addEventListener('resize', handleResize);
      
      return () => {
        window.removeEventListener('resize', handleResize);
        document.body.classList.remove('keyboard-visible');
        document.body.style.removeProperty('--keyboard-height');
      };
    }
  }, [initialHeight]);

  return { isKeyboardVisible, keyboardHeight };
};

/**
 * Gets current keyboard state without using hooks
 * Useful for imperative checks
 */
export const getKeyboardState = () => {
  if ('visualViewport' in window && window.visualViewport) {
    const viewport = window.visualViewport;
    const screenHeight = window.screen.height;
    const currentHeight = viewport.height;
    const heightDiff = screenHeight - currentHeight;
    
    return {
      isVisible: heightDiff > 150,
      height: heightDiff,
    };
  }
  
  return {
    isVisible: false,
    height: 0,
  };
};

/**
 * Safely scrolls element into view when keyboard appears
 * Prevents input from being hidden behind keyboard
 */
export const scrollIntoViewSafe = (element, options = {}) => {
  if (!element) return;
  
  // Wait for keyboard animation to complete
  setTimeout(() => {
    // Use scrollIntoViewIfNeeded if available (Safari/Chrome)
    if (element.scrollIntoViewIfNeeded) {
      element.scrollIntoViewIfNeeded(true);
    }
    // Fallback to standard scrollIntoView
    else {
      element.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
        inline: 'nearest',
        ...options,
      });
    }
  }, 300); // Wait for keyboard animation
};

/**
 * Locks body scroll when keyboard is visible
 * Prevents page bounce on iOS
 */
export const useKeyboardScrollLock = () => {
  const { isKeyboardVisible } = useKeyboardDetection();

  useEffect(() => {
    if (isKeyboardVisible) {
      // Store current scroll position
      const scrollY = window.scrollY;
      
      // Lock scroll
      document.body.style.position = 'fixed';
      document.body.style.top = `-${scrollY}px`;
      document.body.style.width = '100%';
    } else {
      // Restore scroll
      const scrollY = document.body.style.top;
      document.body.style.position = '';
      document.body.style.top = '';
      document.body.style.width = '';
      
      if (scrollY) {
        window.scrollTo(0, parseInt(scrollY || '0') * -1);
      }
    }

    return () => {
      // Cleanup on unmount
      document.body.style.position = '';
      document.body.style.top = '';
      document.body.style.width = '';
    };
  }, [isKeyboardVisible]);
};

export default useKeyboardDetection;
