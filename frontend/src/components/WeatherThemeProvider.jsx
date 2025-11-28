import React, { useState, useEffect } from 'react';

const WeatherThemeProvider = ({ children }) => {
  const [weatherData, setWeatherData] = useState(null);
  const [theme, setTheme] = useState('default');

  useEffect(() => {
    // Weather feature temporarily disabled - using default theme
    // This can be re-enabled when weather API endpoint is implemented
    setTheme('default');
    
    // Optional: Add weather API call here in the future
    /*
    const fetchWeather = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
        const response = await fetch(`${apiUrl}/api/weather?location=Istanbul`);
        if (response.ok) {
          const data = await response.json();
          // Process weather data and set theme
        }
      } catch (error) {
        console.log('Weather fetch failed, using default theme');
      }
    };
    */
  }, []);

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute('data-weather-theme', theme);
  }, [theme]);

  return (
    <div className={`weather-theme-provider theme-${theme}`}>
      {children}
      {/* Weather indicator - Removed as requested */}
      {/* <div className="weather-indicator">
        <span className="weather-icon">{getWeatherIcon(theme)}</span>
        <span className="weather-text">{getWeatherText(theme)}</span>
      </div> */}
    </div>
  );
};

const getWeatherIcon = (theme) => {
  switch (theme) {
    case 'sunny': return '●';
    case 'rainy': return '◇';
    case 'cloudy': return '◈';
    case 'snowy': return '❋';
    default: return '◐';
  }
};

const getWeatherText = (theme) => {
  switch (theme) {
    case 'sunny': return 'Beautiful weather in Istanbul!';
    case 'rainy': return 'Great day for indoor cultural experiences!';
    case 'cloudy': return 'Ideal weather for city walks!';
    case 'snowy': return 'Magical winter atmosphere!';
    default: return 'Beautiful day in Istanbul!';
  }
};

export default WeatherThemeProvider;
