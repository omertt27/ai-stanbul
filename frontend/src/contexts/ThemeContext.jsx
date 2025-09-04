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
  const [theme, setTheme] = useState(() => {
    try {
      const saved = localStorage.getItem('theme');
      return saved || 'dark'; // Default to dark theme
    } catch (error) {
      console.error('Failed to load theme from localStorage:', error);
      return 'dark';
    }
  });

  // Persist theme preference
  useEffect(() => {
    try {
      localStorage.setItem('theme', theme);
    } catch (error) {
      console.error('Failed to save theme to localStorage:', error);
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  // For backward compatibility
  const darkMode = theme === 'dark';
  const toggleDarkMode = toggleTheme;

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, darkMode, toggleDarkMode }}>
      {children}
    </ThemeContext.Provider>
  );
};
