import { useNotifications } from '../contexts/NotificationContext';

/**
 * Hook for integrating notifications with chat responses
 */
export const useChatNotifications = () => {
  const { userId } = useNotifications();
  const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

  const sendChatResponseNotification = async (responseData) => {
    if (!userId || !responseData) return;

    try {
      // Extract response preview for notification
      const preview = responseData.content 
        ? responseData.content.substring(0, 100) + (responseData.content.length > 100 ? '...' : '')
        : 'New response available';

      await fetch(`${API_BASE}/api/v1/notifications/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          title: '💬 Istanbul AI Response',
          message: preview,
          type: 'chat_response',
          priority: 'normal',
          data: {
            preview: preview,
            response_id: responseData.id || Date.now().toString(),
            timestamp: new Date().toISOString(),
            query: responseData.query || '',
            action_url: '/chat'
          }
        })
      });

      console.log('📤 Chat response notification sent');
    } catch (error) {
      console.error('Failed to send chat response notification:', error);
    }
  };

  const sendRouteGeneratedNotification = async (routeData) => {
    if (!userId || !routeData) return;

    try {
      const attractionCount = routeData.attractions?.length || routeData.points?.length || 0;
      const duration = routeData.estimated_duration_hours || routeData.duration || 3;

      await fetch(`${API_BASE}/api/v1/notifications/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          title: '🗺️ Your Route is Ready!',
          message: `Custom route with ${attractionCount} stops planned for ${duration} hours`,
          type: 'route_update',
          priority: 'high',
          data: {
            route_name: routeData.name || 'Custom Route',
            total_attractions: attractionCount,
            duration_hours: duration,
            optimization_message: `Optimized route with ${attractionCount} attractions`,
            map_url: `/route/${routeData.id || 'preview'}`,
            ...routeData
          }
        })
      });

      console.log('📤 Route generated notification sent');
    } catch (error) {
      console.error('Failed to send route notification:', error);
    }
  };

  const sendLocationBasedRecommendation = async (recommendation) => {
    if (!userId || !recommendation) return;

    try {
      await fetch(`${API_BASE}/api/v1/notifications/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          title: '🏛️ Nearby Recommendation',
          message: `${recommendation.name} is nearby! ${recommendation.description || 'Perfect for your interests.'}`,
          type: 'attraction_recommendation',
          priority: 'normal',
          data: {
            attraction_name: recommendation.name,
            distance: recommendation.distance_meters || 0,
            description: recommendation.description || '',
            action_url: `/attraction/${recommendation.id || recommendation.slug || ''}`,
            ...recommendation
          }
        })
      });

      console.log('📤 Location-based recommendation sent');
    } catch (error) {
      console.error('Failed to send location recommendation:', error);
    }
  };

  const sendRealTimeAdvice = async (advice) => {
    if (!userId || !advice) return;

    try {
      await fetch(`${API_BASE}/api/v1/notifications/send`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          title: '💡 Real-time Advice',
          message: advice.message || advice.content || 'New advice available',
          type: 'real_time_advice',
          priority: advice.priority || 'normal',
          data: {
            advice_type: advice.type || 'general',
            confidence: advice.confidence || 0.8,
            context: advice.context || '',
            ...advice
          }
        })
      });

      console.log('📤 Real-time advice notification sent');
    } catch (error) {
      console.error('Failed to send advice notification:', error);
    }
  };

  return {
    sendChatResponseNotification,
    sendRouteGeneratedNotification,
    sendLocationBasedRecommendation,
    sendRealTimeAdvice,
    userId
  };
};

export default useChatNotifications;
