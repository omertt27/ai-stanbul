import React, { useState, useEffect } from 'react';

const WeatherThemeProvider = ({ children }) => {
  const [weatherData, setWeatherData] = useState(null);
  const [theme, setTheme] = useState('default');

  useEffect(() => {
    // Fetch weather data from our backend
    const fetchWeather = async () => {
      try {
        const response = await fetch('http://localhost:8000/blog/recommendations/weather?location=Istanbul&limit=1');
        const data = await response.json();
        
        if (data.success && data.recommendations && data.recommendations.length > 0) {
          const weatherContext = data.recommendations[0].weather_context;
          setWeatherData(weatherContext);
          
          // Determine theme based on weather
          if (weatherContext.includes('rain') || weatherContext.includes('storm')) {
            setTheme('rainy');
          } else if (weatherContext.includes('cloud')) {
            setTheme('cloudy');
          } else if (weatherContext.includes('snow')) {
            setTheme('snowy');
          } else {
            setTheme('sunny');
          }
        }
      } catch (error) {
        console.log('Weather fetch failed, using default theme');
        setTheme('default');
      }
    };

    fetchWeather();
    // Refresh weather every 15 minutes
    const interval = setInterval(fetchWeather, 15 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  // Apply theme to document root
  useEffect(() => {
    document.documentElement.setAttribute('data-weather-theme', theme);
  }, [theme]);

  return (
    <div className={`weather-theme-provider theme-${theme}`}>
      {children}
      {/* Weather indicator */}
      <div className="weather-indicator">
        <span className="weather-icon">{getWeatherIcon(theme)}</span>
        <span className="weather-text">{getWeatherText(theme)}</span>
      </div>
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
