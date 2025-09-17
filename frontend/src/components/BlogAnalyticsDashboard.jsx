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
          <Create />
        </Avatar>
        <Box>
          <Typography variant="h4" sx={{ mb: 0 }}>
            Your Creator Dashboard
          </Typography>
          <Typography variant="body2" color="text.secondary">
            AI Istanbul Blog Analytics - Welcome back, Creator! 👋
          </Typography>
        </Box>
      </Box>

      {/* Real-time Metrics */}
      {realtimeMetrics && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            🔴 Your Content Performance Right Now
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
                icon={<TrendingUp color="warning" />}
                title="New Followers"
                value={realtimeMetrics.new_subscribers_today}
                subtitle="Following your content today"
                color="warning.main"
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <MetricCard
                icon={<ThumbUp color="secondary" />}
                title="Engagement"
                value={`${realtimeMetrics.live_engagement.likes_per_hour}/hr`}
                subtitle="Likes on your content"
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
        </>
      )}

      {/* Performance Analytics */}
      {analytics && (
        <>
          <Typography variant="h5" sx={{ mb: 2 }}>
            📈 Your Content Performance Insights
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
                    👥 User Behavior
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Peak Reading Hours
                    </Typography>
                    <Box display="flex" gap={1}>
                      {analytics.user_behavior.peak_reading_hours.map((hour, index) => (
                        <Chip key={index} label={hour} size="small" />
                      ))}
                    </Box>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Preferred Content Length:</strong> {analytics.user_behavior.preferred_content_length}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Most Shared Content:</strong> {analytics.user_behavior.most_shared_content_type}
                  </Typography>
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
                    Suggested topics based on user searches and engagement:
                  </Typography>
                  <List>
                    {analytics.content_gaps.map((gap, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={gap} />
                        <Chip label="High demand" color="warning" size="small" />
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
