/**
 * Safe Storage Wrapper
 * ====================
 * Prevents TDZ errors by wrapping all localStorage/sessionStorage access
 * Import this FIRST in main.jsx before anything else
 */

// Check if we're in a browser environment
const isBrowser = typeof window !== 'undefined';

// Create safe wrappers
const safeLocalStorage = {
  getItem: (key) => {
    try {
      return isBrowser && window.localStorage ? localStorage.getItem(key) : null;
    } catch (e) {
      return null;
    }
  },
  setItem: (key, value) => {
    try {
      if (isBrowser && window.localStorage) {
        localStorage.setItem(key, value);
      }
    } catch (e) {
      // Silently fail
    }
  },
  removeItem: (key) => {
    try {
      if (isBrowser && window.localStorage) {
        localStorage.removeItem(key);
      }
    } catch (e) {
      // Silently fail
    }
  },
  clear: () => {
    try {
      if (isBrowser && window.localStorage) {
        localStorage.clear();
      }
    } catch (e) {
      // Silently fail
    }
  },
  get length() {
    try {
      return isBrowser && window.localStorage ? localStorage.length : 0;
    } catch (e) {
      return 0;
    }
  },
  key: (index) => {
    try {
      return isBrowser && window.localStorage ? localStorage.key(index) : null;
    } catch (e) {
      return null;
    }
  }
};

const safeSessionStorage = {
  getItem: (key) => {
    try {
      return isBrowser && window.sessionStorage ? sessionStorage.getItem(key) : null;
    } catch (e) {
      return null;
    }
  },
  setItem: (key, value) => {
    try {
      if (isBrowser && window.sessionStorage) {
        sessionStorage.setItem(key, value);
      }
    } catch (e) {
      // Silently fail
    }
  },
  removeItem: (key) => {
    try {
      if (isBrowser && window.sessionStorage) {
        sessionStorage.removeItem(key);
      }
    } catch (e) {
      // Silently fail
    }
  },
  clear: () => {
    try {
      if (isBrowser && window.sessionStorage) {
        sessionStorage.clear();
      }
    } catch (e) {
      // Silently fail
    }
  },
  get length() {
    try {
      return isBrowser && window.sessionStorage ? sessionStorage.length : 0;
    } catch (e) {
      return 0;
    }
  },
  key: (index) => {
    try {
      return isBrowser && window.sessionStorage ? sessionStorage.key(index) : null;
    } catch (e) {
      return null;
    }
  }
};

export { safeLocalStorage, safeSessionStorage };
export default safeLocalStorage;
