import React, { useState } from 'react';
import { 
  Box, 
  Tabs, 
  Tab, 
  Paper,
  Typography,
  Container
} from '@mui/material';
import { 
  Dashboard as DashboardIcon,
  Psychology as PsychologyIcon,
  Article as ArticleIcon,
  Feedback as FeedbackIcon,
  People as PeopleIcon
} from '@mui/icons-material';
import { useTheme } from '../contexts/ThemeContext';

// Import existing dashboard components
import LLMAnalyticsDashboard from './LLMAnalyticsDashboard';
import BlogAnalyticsDashboard from './BlogAnalyticsDashboard';
import FeedbackDashboard from './FeedbackDashboard';
import SystemOverviewTab from './SystemOverviewTab';
import UserAnalyticsTab from './UserAnalyticsTab';

/**
 * Unified Analytics Dashboard
 * 
 * Single entry point for all admin analytics with tabbed interface.
 * Combines LLM, Blog, Feedback, and User analytics in one place.
 * 
 * Created: November 15, 2025
 * Part of: Priority 4 Implementation
 */
const UnifiedAnalyticsDashboard = () => {
  const { darkMode } = useTheme();
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  // Tab configuration
  const tabs = [
    { 
      label: 'System Overview', 
      icon: <DashboardIcon />,
      component: <SystemOverviewTab />
    },
    { 
      label: 'LLM Analytics', 
      icon: <PsychologyIcon />,
      component: <LLMAnalyticsDashboard />
    },
    { 
      label: 'Blog Analytics', 
      icon: <ArticleIcon />,
      component: <BlogAnalyticsDashboard />
    },
    { 
      label: 'Feedback', 
      icon: <FeedbackIcon />,
      component: <FeedbackDashboard isVisible={activeTab === 3} onClose={() => {}} />
    },
    { 
      label: 'User Analytics', 
      icon: <PeopleIcon />,
      component: <UserAnalyticsTab />
    }
  ];

  return (
    <Box sx={{ 
      minHeight: '100vh',
      bgcolor: darkMode ? 'grey.900' : 'grey.50',
      py: 3
    }}>
      <Container maxWidth="xl">
        {/* Header */}
        <Box sx={{ mb: 3 }}>
          <Typography 
            variant="h4" 
            component="h1" 
            sx={{ 
              fontWeight: 'bold',
              color: darkMode ? 'white' : 'grey.900',
              mb: 1
            }}
          >
            🎯 AI Istanbul Analytics Dashboard
          </Typography>
          <Typography 
            variant="body1" 
            sx={{ 
              color: darkMode ? 'grey.400' : 'grey.600'
            }}
          >
            Comprehensive analytics and monitoring for Pure LLM system, blog, and user engagement
          </Typography>
        </Box>

        {/* Tabbed Interface */}
        <Paper 
          elevation={3}
          sx={{ 
            bgcolor: darkMode ? 'grey.800' : 'white',
            borderRadius: 2,
            overflow: 'hidden'
          }}
        >
          {/* Tab Navigation */}
          <Box sx={{ 
            borderBottom: 1, 
            borderColor: 'divider',
            bgcolor: darkMode ? 'grey.900' : 'grey.50'
          }}>
            <Tabs 
              value={activeTab} 
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
              sx={{
                '& .MuiTab-root': {
                  minHeight: 64,
                  textTransform: 'none',
                  fontSize: '0.95rem',
                  fontWeight: 500,
                  color: darkMode ? 'grey.400' : 'grey.600',
                  '&.Mui-selected': {
                    color: darkMode ? 'primary.light' : 'primary.main',
                    fontWeight: 600
                  }
                },
                '& .MuiTabs-indicator': {
                  height: 3,
                  borderRadius: '3px 3px 0 0'
                }
              }}
            >
              {tabs.map((tab, index) => (
                <Tab 
                  key={index}
                  icon={tab.icon}
                  label={tab.label}
                  iconPosition="start"
                />
              ))}
            </Tabs>
          </Box>

          {/* Tab Content */}
          <Box sx={{ p: 3 }}>
            {tabs.map((tab, index) => (
              <Box
                key={index}
                role="tabpanel"
                hidden={activeTab !== index}
                sx={{
                  animation: activeTab === index ? 'fadeIn 0.3s ease-in' : 'none',
                  '@keyframes fadeIn': {
                    from: { opacity: 0, transform: 'translateY(10px)' },
                    to: { opacity: 1, transform: 'translateY(0)' }
                  }
                }}
              >
                {activeTab === index && tab.component}
              </Box>
            ))}
          </Box>
        </Paper>

        {/* Footer Info */}
        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Typography 
            variant="caption" 
            sx={{ 
              color: darkMode ? 'grey.600' : 'grey.500'
            }}
          >
            Last updated: {new Date().toLocaleString()} | 
            System Status: <span style={{ color: '#10b981', fontWeight: 'bold' }}>● Online</span>
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default UnifiedAnalyticsDashboard;
