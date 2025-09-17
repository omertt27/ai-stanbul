import React, { useState, useEffect } from 'react';
import { Card, CardContent, Typography, Chip, Box, Alert, CircularProgress } from '@mui/material';
import { WbSunny, CloudQueue, AcUnit, Opacity } from '@mui/icons-material';

const WeatherAwareBlogRecommendations = ({ userLocation = 'Istanbul' }) => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [weatherContext, setWeatherContext] = useState(null);

  useEffect(() => {
    fetchWeatherRecommendations();
  }, [userLocation]);

  const fetchWeatherRecommendations = async () => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8000/blog/recommendations/weather?location=${userLocation}&limit=5`);
      const data = await response.json();
      
      if (data.success) {
        setRecommendations(data.recommendations);
        // Extract weather context from first recommendation
        if (data.recommendations.length > 0) {
          setWeatherContext(data.recommendations[0].weather_context);
        }
        setError(null);
      } else {
        setError('Failed to fetch weather recommendations');
      }
    } catch (err) {
      setError('Error connecting to server');
      console.error('Weather recommendations error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getWeatherIcon = (weatherContext) => {
    if (!weatherContext) return <WbSunny />;
    
    const context = weatherContext.toLowerCase();
    if (context.includes('rain') || context.includes('storm')) {
      return <Opacity color="primary" />;
    } else if (context.includes('cloud')) {
      return <CloudQueue color="action" />;
    } else if (context.includes('cold') || context.includes('snow')) {
      return <AcUnit color="info" />;
    } else {
      return <WbSunny color="warning" />;
    }
  };

  const getRelevanceColor = (score) => {
    if (score >= 0.9) return 'success';
    if (score >= 0.7) return 'primary';
    if (score >= 0.5) return 'warning';
    return 'default';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
        <Typography variant="body2" sx={{ ml: 2 }}>
          Getting weather-aware recommendations...
        </Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ mb: 4 }}>
      <Box display="flex" alignItems="center" sx={{ mb: 2 }}>
        {getWeatherIcon(weatherContext)}
        <Typography variant="h6" sx={{ ml: 1 }}>
          Perfect for Today's Weather
        </Typography>
      </Box>
      
      {weatherContext && (
        <Alert severity="info" sx={{ mb: 2 }}>
          {weatherContext}
        </Alert>
      )}

      <Box sx={{ display: 'grid', gap: 2 }}>
        {recommendations.map((rec, index) => (
          <Card 
            key={rec.post_id} 
            sx={{ 
              cursor: 'pointer',
              transition: 'transform 0.2s, box-shadow 0.2s',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: 3
              }
            }}
            onClick={() => {
              // Navigate to blog post
              window.location.href = `/blog/${rec.post_id}`;
            }}
          >
            <CardContent>
              <Box display="flex" justifyContent="between" alignItems="flex-start" sx={{ mb: 1 }}>
                <Typography variant="h6" component="h3" sx={{ flexGrow: 1 }}>
                  {rec.title}
                </Typography>
                <Chip 
                  label={`${Math.round(rec.relevance_score * 100)}% match`}
                  color={getRelevanceColor(rec.relevance_score)}
                  size="small"
                />
              </Box>
              
              <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                {rec.reason}
              </Typography>
              
              <Box display="flex" alignItems="center">
                {getWeatherIcon(rec.weather_context)}
                <Typography variant="caption" sx={{ ml: 1 }}>
                  Weather-optimized recommendation
                </Typography>
              </Box>
            </CardContent>
          </Card>
        ))}
      </Box>
      
      {recommendations.length === 0 && (
        <Alert severity="info">
          No weather-specific recommendations available at the moment.
        </Alert>
      )}
    </Box>
  );
};

export default WeatherAwareBlogRecommendations;
