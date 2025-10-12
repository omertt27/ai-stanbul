#!/usr/bin/env python3
"""
Notification Integration Helpers
Integration layer between various Istanbul AI systems and the notification service
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from services.push_notification_service import notification_service, NotificationType, NotificationPriority
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NotificationIntegration:
    """Helper class for integrating notifications with various AI systems"""
    
    def __init__(self):
        self.enabled = NOTIFICATIONS_AVAILABLE
        
    async def send_route_ready_notification(self, user_id: str, route_data: Dict[str, Any]):
        """Send notification when a route is ready"""
        if not self.enabled:
            return False
        
        try:
            route_name = route_data.get('name', 'Your Route')
            total_attractions = len(route_data.get('points', []))
            duration = route_data.get('estimated_duration_hours', 0)
            
            message = f"🗺️ {route_name} is ready! {total_attractions} stops in {duration:.1f} hours."
            
            return await notification_service.send_route_update(
                user_id=user_id,
                route_data={
                    'route_name': route_name,
                    'total_attractions': total_attractions,
                    'duration_hours': duration,
                    'optimization_message': f"Optimized route with {total_attractions} attractions",
                    'map_url': f"/route/{route_data.get('id', 'preview')}",
                    **route_data
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send route ready notification: {e}")
            return False
    
    async def send_attraction_nearby_notification(self, user_id: str, attraction_data: Dict[str, Any]):
        """Send notification when user is near an interesting attraction"""
        if not self.enabled:
            return False
        
        try:
            attraction_name = attraction_data.get('name', 'an interesting place')
            distance = attraction_data.get('distance_meters', 0)
            
            if distance < 100:
                proximity = "right next to you"
            elif distance < 500:
                proximity = f"{distance}m away"
            else:
                proximity = f"{distance/1000:.1f}km away"
            
            return await notification_service.send_attraction_recommendation(
                user_id=user_id,
                attraction_data={
                    'attraction_name': attraction_name,
                    'distance': distance,
                    'proximity_text': proximity,
                    'visit_tip': f"Perfect time to visit {attraction_name}!",
                    **attraction_data
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send attraction nearby notification: {e}")
            return False
    
    async def send_weather_change_notification(self, user_id: str, weather_data: Dict[str, Any]):
        """Send notification for significant weather changes"""
        if not self.enabled:
            return False
        
        try:
            condition = weather_data.get('condition', 'weather')
            temperature = weather_data.get('temperature')
            advice = weather_data.get('advice', '')
            
            if temperature:
                message = f"Weather update: {condition}, {temperature}°C. {advice}"
            else:
                message = f"Weather update: {condition}. {advice}"
            
            return await notification_service.send_weather_alert(
                user_id=user_id,
                weather_data={
                    'message': message,
                    'condition': condition,
                    'temperature': temperature,
                    'advice': advice,
                    'timestamp': datetime.now().isoformat(),
                    **weather_data
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send weather change notification: {e}")
            return False
    
    async def send_traffic_alert_notification(self, user_id: str, traffic_data: Dict[str, Any]):
        """Send notification for traffic updates on user's route"""
        if not self.enabled:
            return False
        
        try:
            route_segment = traffic_data.get('segment', 'your route')
            delay_minutes = traffic_data.get('delay_minutes', 0)
            alternative = traffic_data.get('alternative_suggestion', '')
            
            if delay_minutes > 0:
                message = f"Traffic alert: {delay_minutes} min delay on {route_segment}. {alternative}"
            else:
                message = f"Traffic cleared on {route_segment}!"
            
            return await notification_service.send_traffic_update(
                user_id=user_id,
                traffic_data={
                    'message': message,
                    'segment': route_segment,
                    'delay_minutes': delay_minutes,
                    'alternative': alternative,
                    'timestamp': datetime.now().isoformat(),
                    **traffic_data
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send traffic alert notification: {e}")
            return False
    
    async def send_personalized_discovery_notification(self, user_id: str, discovery_data: Dict[str, Any]):
        """Send notification for personalized discoveries based on user preferences"""
        if not self.enabled:
            return False
        
        try:
            tip_type = discovery_data.get('type', 'tip')
            content = discovery_data.get('content', 'We found something you might like!')
            location = discovery_data.get('location', '')
            
            message = f"💡 Personal discovery: {content}"
            if location:
                message += f" at {location}"
            
            return await notification_service.send_personalized_tip(
                user_id=user_id,
                tip_data={
                    'message': message,
                    'tip_type': tip_type,
                    'content': content,
                    'location': location,
                    'discovery_reason': discovery_data.get('reason', 'Based on your preferences'),
                    'timestamp': datetime.now().isoformat(),
                    **discovery_data
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to send personalized discovery notification: {e}")
            return False
    
    async def send_location_context_notification(self, user_id: str, context_data: Dict[str, Any]):
        """Send notification with location-specific context and tips"""
        if not self.enabled:
            return False
        
        try:
            location_name = context_data.get('location', 'your current area')
            tips = context_data.get('tips', [])
            history = context_data.get('historical_context', '')
            
            # Create a rich context message
            message_parts = [f"📍 Welcome to {location_name}!"]
            
            if history:
                message_parts.append(f"🏛️ {history}")
            
            if tips:
                tip_text = tips[0] if len(tips) == 1 else f"{tips[0]} (+ {len(tips)-1} more tips)"
                message_parts.append(f"💡 {tip_text}")
            
            message = " ".join(message_parts)
            
            return await notification_service.send_notification(
                notification_service.Notification(
                    user_id=user_id,
                    type=NotificationType.LOCATION_BASED,
                    title=f"📍 {location_name}",
                    message=message,
                    data={
                        'location': location_name,
                        'tips': tips,
                        'historical_context': history,
                        'timestamp': datetime.now().isoformat(),
                        **context_data
                    },
                    priority=NotificationPriority.NORMAL
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to send location context notification: {e}")
            return False
    
    async def send_smart_suggestion_notification(self, user_id: str, suggestion_data: Dict[str, Any]):
        """Send smart AI-generated suggestions based on user behavior"""
        if not self.enabled:
            return False
        
        try:
            suggestion_type = suggestion_data.get('type', 'suggestion')
            title = suggestion_data.get('title', '🤖 Smart Suggestion')
            message = suggestion_data.get('message', 'Here\'s something you might find interesting!')
            confidence = suggestion_data.get('confidence', 0.0)
            
            # Adjust priority based on confidence
            if confidence > 0.8:
                priority = NotificationPriority.HIGH
            elif confidence > 0.6:
                priority = NotificationPriority.NORMAL
            else:
                priority = NotificationPriority.LOW
            
            return await notification_service.send_notification(
                notification_service.Notification(
                    user_id=user_id,
                    type=NotificationType.PERSONALIZED_TIP,
                    title=title,
                    message=message,
                    data={
                        'suggestion_type': suggestion_type,
                        'confidence': confidence,
                        'ai_reasoning': suggestion_data.get('reasoning', ''),
                        'action_suggestions': suggestion_data.get('actions', []),
                        'timestamp': datetime.now().isoformat(),
                        **suggestion_data
                    },
                    priority=priority
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to send smart suggestion notification: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if notification system is available"""
        return self.enabled
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification integration statistics"""
        if not self.enabled:
            return {'available': False, 'message': 'Notification system not available'}
        
        try:
            return {
                'available': True,
                'service_stats': notification_service.get_service_stats() if hasattr(notification_service, 'get_service_stats') else {},
                'integration_active': True
            }
        except Exception as e:
            return {'available': True, 'error': str(e), 'integration_active': False}


# Global notification integration instance
notification_integration = NotificationIntegration()

# Async helper functions for background tasks
async def notify_route_ready(user_id: str, route_data: Dict[str, Any]):
    """Background task to notify when route is ready"""
    await notification_integration.send_route_ready_notification(user_id, route_data)

async def notify_attraction_nearby(user_id: str, attraction_data: Dict[str, Any]):
    """Background task to notify about nearby attractions"""
    await notification_integration.send_attraction_nearby_notification(user_id, attraction_data)

async def notify_weather_change(user_id: str, weather_data: Dict[str, Any]):
    """Background task to notify about weather changes"""
    await notification_integration.send_weather_change_notification(user_id, weather_data)

async def notify_traffic_alert(user_id: str, traffic_data: Dict[str, Any]):
    """Background task to notify about traffic updates"""
    await notification_integration.send_traffic_alert_notification(user_id, traffic_data)

async def notify_personalized_discovery(user_id: str, discovery_data: Dict[str, Any]):
    """Background task to notify about personalized discoveries"""
    await notification_integration.send_personalized_discovery_notification(user_id, discovery_data)

async def notify_location_context(user_id: str, context_data: Dict[str, Any]):
    """Background task to send location context notifications"""
    await notification_integration.send_location_context_notification(user_id, context_data)

async def notify_smart_suggestion(user_id: str, suggestion_data: Dict[str, Any]):
    """Background task to send smart AI suggestions"""
    await notification_integration.send_smart_suggestion_notification(user_id, suggestion_data)

# Export all notification functions
__all__ = [
    'NotificationIntegration',
    'notification_integration',
    'notify_route_ready',
    'notify_attraction_nearby', 
    'notify_weather_change',
    'notify_traffic_alert',
    'notify_personalized_discovery',
    'notify_location_context',
    'notify_smart_suggestion'
]
