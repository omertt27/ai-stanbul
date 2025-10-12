import React from 'react';
import { useNotifications } from '../contexts/NotificationContext';

const NotificationTestPanel = () => {
  const { userId } = useNotifications();
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

  const sendTestNotification = async (type) => {
    if (!userId) {
      alert('User ID not available yet');
      return;
    }

    const testNotifications = {
      route_update: {
        title: '🗺️ Route Updated',
        message: 'Your route has been optimized! New estimated time: 3.5 hours',
        type: 'route_update',
        priority: 'high',
        data: {
          route_name: 'Historic Sultanahmet Tour',
          optimization_message: 'Found shorter path avoiding traffic',
          action_url: '/route/123'
        }
      },
      attraction_recommendation: {
        title: '🏛️ Hidden Gem Nearby',
        message: 'Check out Basilica Cistern - only 200m away! Perfect for your history interests.',
        type: 'attraction_recommendation',
        priority: 'normal',
        data: {
          attraction_name: 'Basilica Cistern',
          distance: 200,
          action_url: '/attraction/basilica-cistern'
        }
      },
      weather_alert: {
        title: '🌦️ Weather Alert',
        message: 'Light rain expected in 30 minutes. Consider indoor attractions!',
        type: 'weather_alert',
        priority: 'high',
        data: {
          condition: 'light rain',
          temperature: 18,
          advice: 'Perfect time to visit museums'
        }
      },
      traffic_update: {
        title: '🚦 Traffic Update',
        message: 'Heavy traffic on your route. Alternative path suggested.',
        type: 'traffic_update',
        priority: 'normal',
        data: {
          delay_minutes: 15,
          alternative_available: true
        }
      },
      personalized_tip: {
        title: '💡 Personal Tip',
        message: 'Based on your interests, visit the Archaeology Museum after Hagia Sophia!',
        type: 'personalized_tip',
        priority: 'low',
        data: {
          tip_category: 'cultural_recommendation',
          confidence: 0.85
        }
      }
    };

    try {
      const response = await fetch(`${API_BASE}/api/v1/notifications/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          ...testNotifications[type]
        })
      });

      if (response.ok) {
        console.log(`✅ Test ${type} notification sent`);
      } else {
        console.error('❌ Failed to send notification:', await response.text());
      }
    } catch (error) {
      console.error('❌ Error sending notification:', error);
    }
  };

  // Only show in development
  if (import.meta.env.PROD) {
    return null;
  }

  return (
    <div style={{
      position: 'fixed',
      bottom: '20px',
      left: '20px',
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '15px',
      borderRadius: '10px',
      zIndex: 1000,
      fontSize: '0.8rem',
      maxWidth: '300px'
    }}>
      <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem' }}>🧪 Notification Tests</h4>
      <p style={{ margin: '0 0 10px 0', fontSize: '0.7rem', opacity: 0.7 }}>
        User ID: {userId ? userId.slice(-8) : 'Loading...'}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
        <button 
          onClick={() => sendTestNotification('route_update')}
          style={{
            background: '#ff6b35',
            border: 'none',
            color: 'white',
            padding: '5px 8px',
            borderRadius: '5px',
            fontSize: '0.7rem',
            cursor: 'pointer'
          }}
        >
          🗺️ Route
        </button>
        <button 
          onClick={() => sendTestNotification('attraction_recommendation')}
          style={{
            background: '#8b5cf6',
            border: 'none',
            color: 'white',
            padding: '5px 8px',
            borderRadius: '5px',
            fontSize: '0.7rem',
            cursor: 'pointer'
          }}
        >
          🏛️ Attraction
        </button>
        <button 
          onClick={() => sendTestNotification('weather_alert')}
          style={{
            background: '#06b6d4',
            border: 'none',
            color: 'white',
            padding: '5px 8px',
            borderRadius: '5px',
            fontSize: '0.7rem',
            cursor: 'pointer'
          }}
        >
          🌦️ Weather
        </button>
        <button 
          onClick={() => sendTestNotification('traffic_update')}
          style={{
            background: '#f59e0b',
            border: 'none',
            color: 'white',
            padding: '5px 8px',
            borderRadius: '5px',
            fontSize: '0.7rem',
            cursor: 'pointer'
          }}
        >
          🚦 Traffic
        </button>
        <button 
          onClick={() => sendTestNotification('personalized_tip')}
          style={{
            background: '#10b981',
            border: 'none',
            color: 'white',
            padding: '5px 8px',
            borderRadius: '5px',
            fontSize: '0.7rem',
            cursor: 'pointer'
          }}
        >
          💡 Tip
        </button>
      </div>
    </div>
  );
};

export default NotificationTestPanel;
