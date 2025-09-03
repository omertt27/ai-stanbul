import React, { createContext, useContext, useState, useEffect } from 'react';

const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(() => {
    try {
      const saved = localStorage.getItem('dark-mode');
      return saved ? JSON.parse(saved) : true; // Default to dark mode
    } catch (error) {
      console.error('Failed to load dark mode from localStorage:', error);
      return true;
    }
  });

  // Persist dark mode preference
  useEffect(() => {
    try {
      localStorage.setItem('dark-mode', JSON.stringify(darkMode));
    } catch (error) {
      console.error('Failed to save dark mode to localStorage:', error);
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
};
