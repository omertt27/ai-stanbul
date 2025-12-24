import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Avatar
} from '@mui/material';
import {
  People as PeopleIcon,
  TrendingUp as TrendingUpIcon,
  Schedule as ScheduleIcon,
  Language as LanguageIcon,
  LocationOn as LocationOnIcon
} from '@mui/icons-material';
import { useTheme } from '../contexts/ThemeContext';

/**
 * User Analytics Tab
 * 
 * Provides insights into user behavior and demographics:
 * - User engagement metrics
 * - Geographic distribution
 * - Language preferences
 * - Session duration and frequency
 * - User retention analysis
 */
const UserAnalyticsTab = () => {
  const { darkMode } = useTheme();
  const [userStats, setUserStats] = useState({
    totalUsers: 0,
    activeUsers: 0,
    newUsers: 0,
    avgSessionDuration: '0m',
    topLanguages: [],
    topLocations: [],
    recentSessions: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUserStats();
  }, []);

  const fetchUserStats = async () => {
    setLoading(true);
    try {
      const API_BASE_URL = process.env.NODE_ENV === 'production'
        ? 'https://ai-istanbul-backend.render.com'
        : 'http://localhost:8000';  // Backend runs on port 8000

      // Fetch user analytics data
      const response = await fetch(`${API_BASE_URL}/api/v1/llm/stats/users`);
      if (response.ok) {
        const data = await response.json();
        setUserStats(data);
      } else {
        // Use mock data if endpoint doesn't exist yet
        setUserStats({
          totalUsers: 1247,
          activeUsers: 89,
          newUsers: 23,
          avgSessionDuration: '8m 32s',
          topLanguages: [
            { language: 'Turkish', percentage: 45, count: 561 },
            { language: 'English', percentage: 38, count: 474 },
            { language: 'German', percentage: 10, count: 125 },
            { language: 'Arabic', percentage: 7, count: 87 }
          ],
          topLocations: [
            { city: 'Istanbul', country: 'Turkey', count: 432 },
            { city: 'Ankara', country: 'Turkey', count: 178 },
            { city: 'London', country: 'UK', count: 95 },
            { city: 'Berlin', country: 'Germany', count: 67 },
            { city: 'Dubai', country: 'UAE', count: 54 }
          ],
          recentSessions: [
            { id: 1, user: 'Anonymous', language: 'Turkish', duration: '12m 45s', queries: 8, time: '5 mins ago' },
            { id: 2, user: 'Anonymous', language: 'English', duration: '6m 20s', queries: 4, time: '12 mins ago' },
            { id: 3, user: 'Anonymous', language: 'Turkish', duration: '15m 10s', queries: 11, time: '18 mins ago' },
            { id: 4, user: 'Anonymous', language: 'German', duration: '8m 30s', queries: 6, time: '25 mins ago' },
            { id: 5, user: 'Anonymous', language: 'English', duration: '10m 05s', queries: 7, time: '32 mins ago' }
          ]
        });
      }
    } catch (error) {
      console.error('Error fetching user stats:', error);
      // Use mock data on error
      setUserStats({
        totalUsers: 1247,
        activeUsers: 89,
        newUsers: 23,
        avgSessionDuration: '8m 32s',
        topLanguages: [
          { language: 'Turkish', percentage: 45, count: 561 },
          { language: 'English', percentage: 38, count: 474 },
          { language: 'German', percentage: 10, count: 125 },
          { language: 'Arabic', percentage: 7, count: 87 }
        ],
        topLocations: [
          { city: 'Istanbul', country: 'Turkey', count: 432 },
          { city: 'Ankara', country: 'Turkey', count: 178 },
          { city: 'London', country: 'UK', count: 95 },
          { city: 'Berlin', country: 'Germany', count: 67 },
          { city: 'Dubai', country: 'UAE', count: 54 }
        ],
        recentSessions: [
          { id: 1, user: 'Anonymous', language: 'Turkish', duration: '12m 45s', queries: 8, time: '5 mins ago' },
          { id: 2, user: 'Anonymous', language: 'English', duration: '6m 20s', queries: 4, time: '12 mins ago' },
          { id: 3, user: 'Anonymous', language: 'Turkish', duration: '15m 10s', queries: 11, time: '18 mins ago' },
          { id: 4, user: 'Anonymous', language: 'German', duration: '8m 30s', queries: 6, time: '25 mins ago' },
          { id: 5, user: 'Anonymous', language: 'English', duration: '10m 05s', queries: 7, time: '32 mins ago' }
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  const MetricCard = ({ title, value, icon, color, subtitle }) => (
    <Card 
      elevation={2}
      sx={{ 
        height: '100%',
        bgcolor: darkMode ? 'grey.800' : 'white',
        borderLeft: `4px solid ${color}`
      }}
    >
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box>
            <Typography variant="subtitle2" color="textSecondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" fontWeight="bold" sx={{ color: darkMode ? 'white' : 'grey.900' }}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="caption" color="textSecondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          <Avatar sx={{ bgcolor: `${color}20`, width: 56, height: 56 }}>
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box sx={{ width: '100%', py: 4 }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center', color: darkMode ? 'grey.400' : 'grey.600' }}>
          Loading user analytics...
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      {/* User Metrics */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total Users"
            value={userStats.totalUsers.toLocaleString()}
            icon={<PeopleIcon sx={{ fontSize: 28, color: '#6366f1' }} />}
            color="#6366f1"
            subtitle="All-time users"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Users"
            value={userStats.activeUsers}
            icon={<TrendingUpIcon sx={{ fontSize: 28, color: '#10b981' }} />}
            color="#10b981"
            subtitle="Currently active"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="New Users"
            value={userStats.newUsers}
            icon={<PeopleIcon sx={{ fontSize: 28, color: '#f59e0b' }} />}
            color="#f59e0b"
            subtitle="Last 24 hours"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Session"
            value={userStats.avgSessionDuration}
            icon={<ScheduleIcon sx={{ fontSize: 28, color: '#8b5cf6' }} />}
            color="#8b5cf6"
            subtitle="Duration"
          />
        </Grid>
      </Grid>

      {/* Language Distribution */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ bgcolor: darkMode ? 'grey.800' : 'white' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} mb={2} sx={{ color: darkMode ? 'white' : 'grey.900' }}>
                <LanguageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Top Languages
              </Typography>
              {userStats.topLanguages.map((lang, index) => (
                <Box key={index} mb={2}>
                  <Box display="flex" justifyContent="space-between" mb={0.5}>
                    <Typography variant="body2" fontWeight={500}>
                      {lang.language}
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      {lang.count} users ({lang.percentage}%)
                    </Typography>
                  </Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={lang.percentage} 
                    sx={{ 
                      height: 8, 
                      borderRadius: 1,
                      bgcolor: darkMode ? 'grey.700' : 'grey.200'
                    }}
                  />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>

        {/* Top Locations */}
        <Grid item xs={12} md={6}>
          <Card elevation={2} sx={{ bgcolor: darkMode ? 'grey.800' : 'white' }}>
            <CardContent>
              <Typography variant="h6" fontWeight={600} mb={2} sx={{ color: darkMode ? 'white' : 'grey.900' }}>
                <LocationOnIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Top Locations
              </Typography>
              {userStats.topLocations.map((location, index) => (
                <Box 
                  key={index} 
                  display="flex" 
                  justifyContent="space-between" 
                  alignItems="center"
                  py={1.5}
                  borderBottom={index < userStats.topLocations.length - 1 ? 1 : 0}
                  borderColor={darkMode ? 'grey.700' : 'grey.200'}
                >
                  <Box>
                    <Typography variant="body1" fontWeight={500}>
                      {location.city}, {location.country}
                    </Typography>
                  </Box>
                  <Chip 
                    label={location.count}
                    size="small"
                    color="primary"
                  />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Sessions */}
      <Card elevation={2} sx={{ bgcolor: darkMode ? 'grey.800' : 'white' }}>
        <CardContent>
          <Typography variant="h6" fontWeight={600} mb={2} sx={{ color: darkMode ? 'white' : 'grey.900' }}>
            🕐 Recent Sessions
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>User</TableCell>
                  <TableCell>Language</TableCell>
                  <TableCell align="right">Duration</TableCell>
                  <TableCell align="right">Queries</TableCell>
                  <TableCell align="right">Time</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {userStats.recentSessions.map((session) => (
                  <TableRow key={session.id} hover>
                    <TableCell>
                      <Box display="flex" alignItems="center">
                        <Avatar sx={{ width: 32, height: 32, mr: 1, fontSize: '0.9rem' }}>
                          {session.user.charAt(0)}
                        </Avatar>
                        {session.user}
                      </Box>
                    </TableCell>
                    <TableCell>
                      <Chip label={session.language} size="small" />
                    </TableCell>
                    <TableCell align="right">{session.duration}</TableCell>
                    <TableCell align="right">{session.queries}</TableCell>
                    <TableCell align="right">
                      <Typography variant="caption" color="textSecondary">
                        {session.time}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
};

export default UserAnalyticsTab;
