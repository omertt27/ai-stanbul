import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  IconButton,
  Divider,
  Avatar,
  Alert,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails
} from '@mui/material';
import {
  ThumbUp,
  Share,
  Bookmark,
  BookmarkBorder,
  AccessTime,
  LocationOn,
  WbSunny,
  ExpandMore,
  Map,
  Restaurant,
  DirectionsBus
} from '@mui/icons-material';

const EnhancedBlogPost = ({ postId }) => {
  const [post, setPost] = useState(null);
  const [weatherData, setWeatherData] = useState(null);
  const [nearbyPlaces, setNearbyPlaces] = useState([]);
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [isLiked, setIsLiked] = useState(false);
  const [loading, setLoading] = useState(true);
  const [readingProgress, setReadingProgress] = useState(0);

  useEffect(() => {
    fetchBlogPost();
    trackEngagement('view');
    
    // Track reading progress
    const handleScroll = () => {
      const totalHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setReadingProgress(Math.min(progress, 100));
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, [postId]);

  const fetchBlogPost = async () => {
    try {
      // Fetch blog post
      const postResponse = await fetch(`/blog/${postId}`);
      const postData = await postResponse.json();
      
      if (postData.success) {
        setPost(postData.post);
        
        // Fetch contextual data
        await fetchContextualData(postData.post);
      }
    } catch (error) {
      console.error('Error fetching blog post:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchContextualData = async (post) => {
    try {
      // Fetch weather data for mentioned locations
      if (post.locations && post.locations.length > 0) {
        const weatherResponse = await fetch(`/weather/current?city=${post.locations[0]}`);
        const weatherData = await weatherResponse.json();
        if (weatherData.success) {
          setWeatherData(weatherData.data);
        }
      }

      // Fetch nearby restaurants if it's a food-related post
      if (post.category === 'food' || post.tags.includes('restaurants')) {
        const placesResponse = await fetch(`/places/restaurants?query=${post.title}&limit=3`);
        const placesData = await placesResponse.json();
        if (placesData.success) {
          setNearbyPlaces(placesData.restaurants);
        }
      }
    } catch (error) {
      console.error('Error fetching contextual data:', error);
    }
  };

  const trackEngagement = async (eventType, metadata = {}) => {
    try {
      const formData = new FormData();
      formData.append('post_id', postId);
      formData.append('user_id', 'anonymous_user'); // Replace with actual user ID
      formData.append('event_type', eventType);
      formData.append('metadata', JSON.stringify(metadata));

      await fetch('/blog/analytics/track', {
        method: 'POST',
        body: formData
      });
    } catch (error) {
      console.error('Error tracking engagement:', error);
    }
  };

  const handleLike = async () => {
    setIsLiked(!isLiked);
    await trackEngagement('like', { action: isLiked ? 'unlike' : 'like' });
  };

  const handleBookmark = async () => {
    setIsBookmarked(!isBookmarked);
    await trackEngagement('bookmark', { action: isBookmarked ? 'unbookmark' : 'bookmark' });
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: post.title,
          text: post.excerpt || post.content.substring(0, 150) + '...',
          url: window.location.href
        });
        await trackEngagement('share', { method: 'native' });
      } catch (error) {
        console.error('Error sharing:', error);
      }
    } else {
      // Fallback to clipboard
      navigator.clipboard.writeText(window.location.href);
      await trackEngagement('share', { method: 'clipboard' });
      alert('Link copied to clipboard!');
    }
  };

  const calculateReadingTime = (content) => {
    const wordsPerMinute = 200;
    const wordCount = content.split(' ').length;
    return Math.ceil(wordCount / wordsPerMinute);
  };

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2 }}>Loading enhanced blog post...</Typography>
      </Box>
    );
  }

  if (!post) {
    return (
      <Alert severity="error" sx={{ m: 3 }}>
        Blog post not found
      </Alert>
    );
  }

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      {/* Reading Progress Bar */}
      <LinearProgress 
        variant="determinate" 
        value={readingProgress} 
        sx={{ 
          position: 'fixed', 
          top: 0, 
          left: 0, 
          right: 0, 
          zIndex: 1000,
          height: 3
        }} 
      />

      {/* Main Content */}
      <Card>
        <CardContent>
          {/* Header */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="h3" component="h1" sx={{ mb: 2 }}>
              {post.title}
            </Typography>
            
            <Box display="flex" alignItems="center" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
              <Avatar sx={{ width: 32, height: 32 }}>
                {post.author.charAt(0)}
              </Avatar>
              <Typography variant="body2" color="text.secondary">
                {post.author}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                •
              </Typography>
              <Box display="flex" alignItems="center">
                <AccessTime fontSize="small" sx={{ mr: 0.5 }} />
                <Typography variant="body2" color="text.secondary">
                  {calculateReadingTime(post.content)} min read
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                •
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {new Date(post.created_at).toLocaleDateString()}
              </Typography>
            </Box>

            {/* Tags */}
            <Box display="flex" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
              <Chip label={post.category} color="primary" size="small" />
              {post.tags.map((tag, index) => (
                <Chip key={index} label={tag} variant="outlined" size="small" />
              ))}
            </Box>

            {/* Action Buttons */}
            <Box display="flex" gap={1}>
              <IconButton 
                onClick={handleLike}
                color={isLiked ? "primary" : "default"}
              >
                <ThumbUp />
              </IconButton>
              
              <IconButton 
                onClick={handleBookmark}
                color={isBookmarked ? "primary" : "default"}
              >
                {isBookmarked ? <Bookmark /> : <BookmarkBorder />}
              </IconButton>
              
              <IconButton onClick={handleShare}>
                <Share />
              </IconButton>
            </Box>
          </Box>

          <Divider sx={{ mb: 3 }} />

          {/* Weather Context */}
          {weatherData && (
            <Alert 
              severity="info" 
              icon={<WbSunny />}
              sx={{ mb: 3 }}
            >
              <Typography variant="body2">
                <strong>Current weather:</strong> {weatherData.description}, {weatherData.temperature}°C
                {weatherData.activity_recommendations && (
                  <>
                    <br />
                    <strong>Perfect for:</strong> {weatherData.activity_recommendations[0]}
                  </>
                )}
              </Typography>
            </Alert>
          )}

          {/* Main Content */}
          <Typography variant="body1" sx={{ lineHeight: 1.8, mb: 4 }}>
            {post.content}
          </Typography>

          {/* Interactive Sections */}
          <Box sx={{ mt: 4 }}>
            {/* Nearby Places */}
            {nearbyPlaces.length > 0 && (
              <Accordion>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box display="flex" alignItems="center">
                    <Restaurant sx={{ mr: 1 }} />
                    <Typography variant="h6">
                      Related Restaurants
                    </Typography>
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Box sx={{ display: 'grid', gap: 2 }}>
                    {nearbyPlaces.map((place, index) => (
                      <Card key={index} variant="outlined">
                        <CardContent>
                          <Typography variant="subtitle1">
                            {place.name}
                          </Typography>
                          <Box display="flex" alignItems="center" sx={{ mt: 1 }}>
                            <LocationOn fontSize="small" color="action" />
                            <Typography variant="body2" color="text.secondary" sx={{ ml: 0.5 }}>
                              {place.vicinity || place.address}
                            </Typography>
                          </Box>
                          {place.rating && (
                            <Typography variant="body2" sx={{ mt: 1 }}>
                              ⭐ {place.rating} ({place.user_ratings_total} reviews)
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </Box>
                </AccordionDetails>
              </Accordion>
            )}

            {/* Map Integration Placeholder */}
            <Accordion sx={{ mt: 2 }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center">
                  <Map sx={{ mr: 1 }} />
                  <Typography variant="h6">
                    Interactive Map
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Alert severity="info">
                  Interactive map showing mentioned locations will be displayed here.
                  This could integrate with Google Maps to show all places mentioned in the article.
                </Alert>
              </AccordionDetails>
            </Accordion>

            {/* Transportation Info */}
            <Accordion sx={{ mt: 2 }}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box display="flex" alignItems="center">
                  <DirectionsBus sx={{ mr: 1 }} />
                  <Typography variant="h6">
                    How to Get There
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Alert severity="info">
                  Transportation options and routes to mentioned locations will be shown here.
                  This could integrate with Istanbul transport APIs for real-time information.
                </Alert>
              </AccordionDetails>
            </Accordion>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default EnhancedBlogPost;
