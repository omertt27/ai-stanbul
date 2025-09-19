import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  Alert,
  Divider,
  Avatar
} from '@mui/material';
import {
  TrendingUp,
  Visibility,
  ThumbUp,
  Share,
  AccessTime,
  People,
  Create,
  Analytics,
  AutoAwesome
} from '@mui/icons-material';

const BlogAnalyticsDashboard = () => {
  const [analytics, setAnalytics] = useState(null);
  const [realtimeMetrics, setRealtimeMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchAnalytics();
    fetchRealtimeMetrics();
    
    // Set up real-time updates every 30 seconds
    const interval = setInterval(fetchRealtimeMetrics, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchAnalytics = async () => {
    try {
      const response = await fetch('http://localhost:8000/blog/analytics/performance');
      const data = await response.json();
      
      if (data.success) {
        setAnalytics(data.analytics);
      } else {
        setError('Failed to fetch analytics');
      }
    } catch (err) {
      setError('Error fetching analytics');
      console.error('Analytics error:', err);
    }
  };

  const fetchRealtimeMetrics = async () => {
    try {
      const response = await fetch('http://localhost:8000/blog/analytics/realtime');
      const data = await response.json();
      
      if (data.success) {
        setRealtimeMetrics(data.metrics);
        setError(null);
      }
    } catch (err) {
      console.error('Real-time metrics error:', err);
    } finally {
      setLoading(false);
    }
  };

  const MetricCard = ({ icon, title, value, subtitle, color = 'primary' }) => (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" sx={{ mb: 1 }}>
          {icon}
          <Typography variant="h6" sx={{ ml: 1 }}>
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" color={color} sx={{ mb: 1 }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box sx={{ p: 3 }}>
        <Typography variant="h4" sx={{ mb: 3 }}>
          Blog Analytics Dashboard
        </Typography>
        <LinearProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 3 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Creator Header */}
      <Box display="flex" alignItems="center" sx={{ mb: 3 }}>
        <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
          <Analytics />
        </Avatar>
        <Box>
          <Typography variant="h4" sx={{ mb: 0 }}>
            Istanbul Travel Analytics
          </Typography>
          <Typography variant="body2" color="text.secondary">
            AI Istanbul Blog Performance Dashboard - Track your travel content impact 🏛️
          </Typography>
        </Box>
      </Box>

      {/* Real-time Metrics */}
      {realtimeMetrics && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            � Live Travel Blog Metrics
          </Typography>
          
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                icon={<People color="success" />}
                title="Reading Your Content"
                value={realtimeMetrics.current_active_readers}
                subtitle="Active readers now"
                color="success.main"
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                icon={<Visibility color="primary" />}
                title="Your Posts Read Today"
                value={realtimeMetrics.posts_read_today}
                subtitle="Total views since midnight"
                color="primary.main"
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                icon={<Create color="warning" />}
                title="Published Guides"
                value={realtimeMetrics.total_posts_published}
                subtitle="Travel guides & blog posts"
                color="warning.main"
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                icon={<AccessTime color="secondary" />}
                title="Avg. Visit Duration"
                value={realtimeMetrics.average_session_duration}
                subtitle="Time spent exploring"
                color="secondary.main"
              />
            </Grid>
          </Grid>

          {/* Trending Now */}
          <Card sx={{ mb: 4 }}>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                🔥 What's Trending from Your Content
              </Typography>
              <Box display="flex" flexWrap="wrap" gap={1}>
                {realtimeMetrics.trending_now.map((topic, index) => (
                  <Chip
                    key={index}
                    label={topic}
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            </CardContent>
          </Card>

          {/* Travel Insights */}
          {realtimeMetrics.travel_insights && (
            <Card sx={{ mb: 4 }}>
              <CardContent>
                <Typography variant="h6" sx={{ mb: 2 }}>
                  🏛️ Istanbul Travel Insights
                </Typography>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Popular Districts
                    </Typography>
                    <Box display="flex" flexDirection="column" gap={1}>
                      {realtimeMetrics.travel_insights.popular_districts.map((district, index) => (
                        <Chip key={index} label={district} size="small" color="success" />
                      ))}
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Seasonal Trend
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {realtimeMetrics.travel_insights.seasonal_trend}
                    </Typography>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      User Interests
                    </Typography>
                    <Box display="flex" flexDirection="column" gap={1}>
                      {realtimeMetrics.travel_insights.user_interests.map((interest, index) => (
                        <Chip key={index} label={interest} size="small" variant="outlined" />
                      ))}
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {/* Performance Analytics */}
      {analytics && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            📈 Travel Content Performance
          </Typography>
          
          <Grid container spacing={3}>
            {/* Top Performing Posts */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    🏆 Top Performing Posts
                  </Typography>
                  <List>
                    {analytics.top_performing_posts.map((post, index) => (
                      <ListItem key={post.post_id} divider>
                        <ListItemText
                          primary={post.title}
                          secondary={
                            <Box>
                              <Typography component="span" variant="body2">
                                {post.views} views • {Math.round(post.engagement_rate * 100)}% engagement
                              </Typography>
                              <br />
                              <Typography component="span" variant="caption" color="text.secondary">
                                Avg. reading time: {post.avg_time_spent}
                              </Typography>
                            </Box>
                          }
                        />
                        <Chip 
                          label={`#${index + 1}`} 
                          color="primary" 
                          size="small" 
                        />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* Trending Categories */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    📊 Trending Categories
                  </Typography>
                  <List>
                    {analytics.trending_categories.map((category, index) => (
                      <ListItem key={category.category} divider>
                        <ListItemText
                          primary={category.category.charAt(0).toUpperCase() + category.category.slice(1)}
                          secondary={`Growth: ${category.growth}`}
                        />
                        <TrendingUp color="success" />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* User Behavior Insights */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    👥 Visitor Behavior
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Peak Reading Hours
                    </Typography>
                    <Box display="flex" gap={1}>
                      {analytics.user_behavior.peak_reading_hours.map((hour, index) => (
                        <Chip key={index} label={hour} size="small" color="primary" />
                      ))}
                    </Box>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Preferred Content Length:</strong> {analytics.user_behavior.preferred_content_length}
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Most Shared Content:</strong> {analytics.user_behavior.most_shared_content_type}
                  </Typography>
                  {analytics.user_behavior.average_pages_per_visit && (
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Pages per Visit:</strong> {analytics.user_behavior.average_pages_per_visit}
                    </Typography>
                  )}
                  {analytics.user_behavior.mobile_vs_desktop && (
                    <Typography variant="body2">
                      <strong>Device Usage:</strong> {analytics.user_behavior.mobile_vs_desktop}
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Content Gaps */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" sx={{ mb: 2 }}>
                    💡 Content Opportunities
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 2 }}>
                    High-demand topics based on visitor searches and engagement:
                  </Typography>
                  <List>
                    {analytics.content_gaps.map((gap, index) => (
                      <ListItem key={index}>
                        <ListItemText 
                          primary={gap}
                          secondary="High search volume, low competition"
                        />
                        <Chip label="Opportunity" color="warning" size="small" />
                      </ListItem>
                    ))}
                  </List>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </>
      )}
    </Box>
  );
};

export default BlogAnalyticsDashboard;
