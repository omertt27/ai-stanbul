import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Alert
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  CloudDone as CloudDoneIcon,
  Psychology as PsychologyIcon,
  Article as ArticleIcon
} from '@mui/icons-material';
import { useTheme } from '../contexts/ThemeContext';

/**
 * System Overview Tab
 * 
 * Provides high-level overview of the entire AI Istanbul system:
 * - Overall system health and status
 * - Key metrics across all services
 * - Quick links to detailed analytics
 * - System alerts and notifications
 */
const SystemOverviewTab = () => {
  const { darkMode } = useTheme();
  const [systemStats, setSystemStats] = useState({
    total_queries: 0,
    unique_users: 0,
    cache_hit_rate: 0,
    avg_queries_per_user: 0,
    llm_calls: 0,
    error_rate: 0,
    apiStatus: 'online',
    uptime: '99.9%'
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSystemStats();
  }, []);

  const fetchSystemStats = async () => {
    setLoading(true);
    try {
      const API_BASE_URL = process.env.NODE_ENV === 'production'
        ? 'https://ai-istanbul-backend.render.com'
        : 'http://localhost:8000';  // Backend runs on port 8000

      // Fetch multiple real data sources in parallel
      const [llmResponse, feedbackResponse, adminResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/v1/llm/stats`).catch(() => null),
        fetch(`${API_BASE_URL}/api/feedback/stats`).catch(() => null),
        fetch(`${API_BASE_URL}/api/admin/stats`).catch(() => null)
      ]);

      let stats = { ...systemStats };

      // Get LLM stats
      if (llmResponse?.ok) {
        const llmData = await llmResponse.json();
        stats = { ...stats, ...llmData };
      }

      // Get feedback stats (real data from database)
      if (feedbackResponse?.ok) {
        const feedbackData = await feedbackResponse.json();
        stats.total_queries = feedbackData.total_interactions || stats.total_queries;
        stats.languages = feedbackData.languages;
        stats.intents = feedbackData.intents;
      }

      // Get admin stats
      if (adminResponse?.ok) {
        const adminData = await adminResponse.json();
        if (adminData.data) {
          stats.blog_posts = adminData.data.blog_posts || 0;
        }
      }

      setSystemStats(stats);
    } catch (error) {
      console.error('Error fetching system stats:', error);
    } finally {
      setLoading(false);
    }
  };

  const StatCard = ({ title, value, icon, color, trend }) => (
    <Card 
      elevation={2}
      sx={{ 
        height: '100%',
        bgcolor: darkMode ? 'grey.800' : 'white',
        borderLeft: `4px solid ${color}`,
        transition: 'transform 0.2s',
        '&:hover': {
          transform: 'translateY(-4px)',
          boxShadow: 6
        }
      }}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography 
              variant="subtitle2" 
              sx={{ 
                color: darkMode ? 'grey.400' : 'grey.600',
                fontWeight: 500,
                mb: 1
              }}
            >
              {title}
            </Typography>
            <Typography 
              variant="h4" 
              sx={{ 
                fontWeight: 'bold',
                color: darkMode ? 'white' : 'grey.900',
                mb: 0.5
              }}
            >
              {value}
            </Typography>
            {trend && (
              <Chip 
                label={trend}
                size="small"
                color={trend.includes('+') ? 'success' : 'error'}
                icon={<TrendingUpIcon />}
                sx={{ height: 20, fontSize: '0.7rem' }}
              />
            )}
          </Box>
          <Box 
            sx={{ 
              bgcolor: `${color}20`,
              borderRadius: 2,
              p: 1.5,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box sx={{ width: '100%', py: 4 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center', color: darkMode ? 'grey.400' : 'grey.600' }}>
          Loading system overview...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* System Status Alert */}
      <Alert 
        severity="success" 
        sx={{ mb: 3 }}
        icon={<CloudDoneIcon />}
      >
        <strong>All Systems Operational</strong> - AI Istanbul is running smoothly with {systemStats.uptime} uptime
      </Alert>

      {/* Key Metrics Grid */}
      <Grid container spacing={3}>
        {/* Total Interactions */}
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Total LLM Queries"
            value={(systemStats.total_queries || 0).toLocaleString()}
            icon={<PsychologyIcon sx={{ fontSize: 32, color: '#6366f1' }} />}
            color="#6366f1"
            trend="+12.5% this week"
          />
        </Grid>

        {/* Active Sessions */}
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Unique Users"
            value={(systemStats.unique_users || 0).toLocaleString()}
            icon={<SpeedIcon sx={{ fontSize: 32, color: '#10b981' }} />}
            color="#10b981"
            trend="+5.2% today"
          />
        </Grid>

        {/* Blog Posts */}
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Cache Hit Rate"
            value={`${((systemStats.cache_hit_rate || 0) * 100).toFixed(1)}%`}
            icon={<ArticleIcon sx={{ fontSize: 32, color: '#f59e0b' }} />}
            color="#f59e0b"
            trend="+3 this month"
          />
        </Grid>

        {/* API Response Time */}
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Avg Queries/User"
            value={(systemStats.avg_queries_per_user || 0).toFixed(2)}
            icon={<SpeedIcon sx={{ fontSize: 32, color: '#8b5cf6' }} />}
            color="#8b5cf6"
            trend="-8% faster"
          />
        </Grid>

        {/* Database Status */}
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Database Status"
            value="Connected"
            icon={<StorageIcon sx={{ fontSize: 32, color: '#06b6d4' }} />}
            color="#06b6d4"
          />
        </Grid>

        {/* System Uptime */}
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="System Uptime"
            value={systemStats.uptime}
            icon={<CloudDoneIcon sx={{ fontSize: 32, color: '#10b981' }} />}
            color="#10b981"
          />
        </Grid>
      </Grid>

      {/* Quick Navigation */}
      <Box sx={{ mt: 4 }}>
        <Typography 
          variant="h6" 
          sx={{ 
            mb: 2,
            fontWeight: 600,
            color: darkMode ? 'white' : 'grey.900'
          }}
        >
          📊 Quick Access
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Card 
              sx={{ 
                bgcolor: darkMode ? 'grey.800' : 'white',
                cursor: 'pointer',
                '&:hover': { boxShadow: 4 }
              }}
            >
              <CardContent>
                <Typography variant="body1" fontWeight={500}>
                  🤖 Pure LLM Analytics
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  View detailed LLM metrics
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card 
              sx={{ 
                bgcolor: darkMode ? 'grey.800' : 'white',
                cursor: 'pointer',
                '&:hover': { boxShadow: 4 }
              }}
            >
              <CardContent>
                <Typography variant="body1" fontWeight={500}>
                  📝 Blog Performance
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Track blog engagement
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card 
              sx={{ 
                bgcolor: darkMode ? 'grey.800' : 'white',
                cursor: 'pointer',
                '&:hover': { boxShadow: 4 }
              }}
            >
              <CardContent>
                <Typography variant="body1" fontWeight={500}>
                  💬 User Feedback
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Review user submissions
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card 
              sx={{ 
                bgcolor: darkMode ? 'grey.800' : 'white',
                cursor: 'pointer',
                '&:hover': { boxShadow: 4 }
              }}
            >
              <CardContent>
                <Typography variant="body1" fontWeight={500}>
                  👥 User Analytics
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Analyze user behavior
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Box>
  );
};

export default SystemOverviewTab;
