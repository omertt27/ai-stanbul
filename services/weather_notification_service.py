#!/usr/bin/env python3
"""
Weather-Aware Notification System
Sends intelligent weather-based notifications to users
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

try:
    from services.weather_cache_service import weather_cache, WeatherAlert
    from services.push_notification_service import notification_service, NotificationType, NotificationPriority
    NOTIFICATIONS_AVAILABLE = True
except ImportError as e:
    NOTIFICATIONS_AVAILABLE = False
    print(f"Weather notifications not available: {e}")

logger = logging.getLogger(__name__)


class WeatherNotificationManager:
    """Manages weather-based notifications and alerts"""
    
    def __init__(self):
        self.last_alert_check = None
        self.sent_alerts = {}  # Track sent alerts to avoid spam
        self.user_preferences = {}  # User weather notification preferences
        
    async def check_and_send_weather_alerts(self, user_ids: List[str] = None):
        """Check weather conditions and send alerts if needed"""
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        try:
            weather_summary = weather_cache.get_weather_summary()
            if 'error' in weather_summary:
                logger.warning("No weather data available for alerts")
                return
            
            alerts = weather_summary.get('alerts', [])
            if not alerts:
                return
            
            # Send alerts to all users or specified users
            for alert in alerts:
                await self._send_weather_alert(alert, user_ids)
                
        except Exception as e:
            logger.error(f"Failed to check weather alerts: {e}")
    
    async def _send_weather_alert(self, alert: WeatherAlert, user_ids: List[str] = None):
        """Send a weather alert notification"""
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        alert_key = f"{alert.alert_type}_{alert.severity}_{alert.start_time.hour}"
        
        # Avoid sending duplicate alerts within 2 hours
        if alert_key in self.sent_alerts:
            last_sent = self.sent_alerts[alert_key]
            if datetime.now() - last_sent < timedelta(hours=2):
                return
        
        # Determine notification priority
        priority_map = {
            'low': NotificationPriority.LOW,
            'medium': NotificationPriority.NORMAL,
            'high': NotificationPriority.HIGH,
            'extreme': NotificationPriority.URGENT
        }
        priority = priority_map.get(alert.severity, NotificationPriority.NORMAL)
        
        # Get emoji for alert type
        emoji_map = {
            'rain': '🌧️',
            'wind': '💨', 
            'temperature': '🌡️',
            'storm': '⛈️',
            'fog': '🌫️'
        }
        emoji = emoji_map.get(alert.alert_type, '⚠️')
        
        # If no specific users, send to all connected users (this would need user management)
        target_users = user_ids or ['default_user']  # In production, get from user management
        
        for user_id in target_users:
            try:
                await notification_service.send_weather_alert(
                    user_id=user_id,
                    weather_data={
                        'alert_type': alert.alert_type,
                        'severity': alert.severity,
                        'title': alert.title,
                        'message': alert.description,
                        'affected_areas': alert.affected_areas,
                        'start_time': alert.start_time.isoformat(),
                        'end_time': alert.end_time.isoformat(),
                        'emoji': emoji,
                        'priority': alert.severity
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to send weather alert to user {user_id}: {e}")
        
        # Mark alert as sent
        self.sent_alerts[alert_key] = datetime.now()
        logger.info(f"📤 Sent weather alert: {alert.title}")
    
    async def send_daily_weather_summary(self, user_ids: List[str]):
        """Send daily weather summary to users"""
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        try:
            weather_summary = weather_cache.get_weather_summary()
            if 'error' in weather_summary:
                return
            
            current = weather_summary['current']
            conditions = weather_summary['conditions']
            recommendations = weather_summary['recommendations']
            
            # Create summary message
            temp = current['current_temp']
            condition = current['condition']
            comfort = conditions['comfort_level']
            
            summary_text = f"Today's weather: {condition}, {temp}°C"
            if recommendations:
                summary_text += f". {recommendations[0]}"
            
            # Determine weather emoji
            weather_emoji = self._get_weather_emoji(current['condition'], current['icon'])
            
            for user_id in user_ids:
                try:
                    await notification_service.send_personalized_tip(
                        user_id=user_id,
                        tip_data={
                            'type': 'daily_weather',
                            'title': f"{weather_emoji} Daily Weather Update",
                            'message': summary_text,
                            'weather_data': current,
                            'recommendations': recommendations[:2],
                            'comfort_level': comfort
                        }
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to send daily weather summary to user {user_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to send daily weather summaries: {e}")
    
    async def send_location_weather_update(self, user_id: str, location_data: Dict[str, Any]):
        """Send weather update when user changes location"""
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        try:
            weather_summary = weather_cache.get_weather_summary()
            if 'error' in weather_summary:
                return
            
            current = weather_summary['current']
            recommendations = weather_summary['recommendations']
            
            # Check if weather significantly affects outdoor activities
            outdoor_suitability = weather_summary['conditions']['outdoor_suitability']
            
            if outdoor_suitability in ['poor', 'fair']:
                # Send notification about weather conditions
                weather_emoji = self._get_weather_emoji(current['condition'], current['icon'])
                
                message = f"Weather update for your location: {current['condition']}, {current['current_temp']}°C"
                if recommendations:
                    message += f". {recommendations[0]}"
                
                await notification_service.send_personalized_tip(
                    user_id=user_id,
                    tip_data={
                        'type': 'location_weather',
                        'title': f"{weather_emoji} Weather Notice",
                        'message': message,
                        'location': location_data,
                        'weather_data': current,
                        'recommendations': recommendations[:2]
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to send location weather update: {e}")
    
    async def send_route_weather_advice(self, user_id: str, route_data: Dict[str, Any]):
        """Send weather-specific advice for planned routes"""
        if not NOTIFICATIONS_AVAILABLE:
            return
        
        try:
            weather_summary = weather_cache.get_weather_summary()
            if 'error' in weather_summary:
                return
            
            current = weather_summary['current']
            conditions = weather_summary['conditions']
            
            # Analyze route suitability based on weather
            route_advice = self._analyze_route_weather_suitability(route_data, weather_summary)
            
            if route_advice['needs_notification']:
                weather_emoji = self._get_weather_emoji(current['condition'], current['icon'])
                
                await notification_service.send_route_update(
                    user_id=user_id,
                    route_data={
                        'weather_advice': route_advice,
                        'current_weather': current,
                        'route_name': route_data.get('name', 'Your Route'),
                        'weather_impact': route_advice['impact_level'],
                        'recommendations': route_advice['recommendations'],
                        'title': f"{weather_emoji} Weather Update for Your Route",
                        'message': route_advice['message']
                    }
                )
                
        except Exception as e:
            logger.error(f"Failed to send route weather advice: {e}")
    
    def _get_weather_emoji(self, condition: str, icon: str) -> str:
        """Get appropriate emoji for weather condition"""
        condition_lower = condition.lower()
        
        if 'clear' in condition_lower:
            return '☀️' if 'd' in icon else '🌙'
        elif 'cloud' in condition_lower:
            return '☁️' if 'few' not in condition_lower else '🌤️'
        elif 'rain' in condition_lower:
            return '🌧️'
        elif 'drizzle' in condition_lower:
            return '🌦️'
        elif 'thunderstorm' in condition_lower or 'storm' in condition_lower:
            return '⛈️'
        elif 'snow' in condition_lower:
            return '❄️'
        elif 'mist' in condition_lower or 'fog' in condition_lower:
            return '🌫️'
        else:
            return '🌤️'
    
    def _analyze_route_weather_suitability(self, route_data: Dict[str, Any], weather_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how weather affects a planned route"""
        current = weather_summary['current']
        conditions = weather_summary['conditions']
        
        recommendations = []
        impact_level = 'low'
        needs_notification = False
        
        # Check temperature impact
        temp = current['current_temp']
        if temp > 30:
            recommendations.append("Very hot weather - plan breaks in shaded areas and carry water")
            impact_level = 'medium'
            needs_notification = True
        elif temp < 5:
            recommendations.append("Cold weather - dress warmly and consider shorter outdoor segments")
            impact_level = 'medium'
            needs_notification = True
        
        # Check precipitation impact
        precipitation = conditions['precipitation_status']
        if precipitation in ['moderate_rain', 'heavy_rain']:
            recommendations.append("Rain expected - bring umbrella and prioritize covered attractions")
            impact_level = 'high'
            needs_notification = True
        elif precipitation == 'light_rain':
            recommendations.append("Light rain possible - consider bringing a light rain jacket")
            impact_level = 'medium'
            needs_notification = True
        
        # Check wind impact
        wind_status = conditions['wind_status']
        if wind_status in ['strong', 'very_strong']:
            recommendations.append("Strong winds - be cautious near waterfront areas")
            if 'bosphorus' in route_data.get('name', '').lower():
                recommendations.append("Consider postponing Bosphorus activities due to strong winds")
                impact_level = 'high'
                needs_notification = True
        
        # Check outdoor suitability
        outdoor_suit = conditions['outdoor_suitability']
        if outdoor_suit == 'poor':
            recommendations.append("Weather not ideal for outdoor activities - consider indoor alternatives")
            impact_level = 'high'
            needs_notification = True
        
        # Create message
        if impact_level == 'high':
            message = f"Weather conditions may significantly impact your route ({current['condition']}, {temp}°C)"
        elif impact_level == 'medium':
            message = f"Weather advisory for your route: {current['condition']}, {temp}°C"
        else:
            message = f"Good weather for your route: {current['condition']}, {temp}°C"
        
        return {
            'needs_notification': needs_notification,
            'impact_level': impact_level,
            'recommendations': recommendations,
            'message': message,
            'weather_suitable': outdoor_suit in ['excellent', 'good']
        }


# Global weather notification manager
weather_notification_manager = WeatherNotificationManager()

# Convenience functions for integration
async def send_weather_alerts(user_ids: List[str] = None):
    """Send weather alerts to users"""
    await weather_notification_manager.check_and_send_weather_alerts(user_ids)

async def send_daily_weather_summary(user_ids: List[str]):
    """Send daily weather summary"""
    await weather_notification_manager.send_daily_weather_summary(user_ids)

async def notify_location_weather(user_id: str, location_data: Dict[str, Any]):
    """Notify user about weather at their location"""
    await weather_notification_manager.send_location_weather_update(user_id, location_data)

async def notify_route_weather(user_id: str, route_data: Dict[str, Any]):
    """Send weather advice for a route"""
    await weather_notification_manager.send_route_weather_advice(user_id, route_data)

# Export for use in other modules
__all__ = [
    'WeatherNotificationManager',
    'weather_notification_manager',
    'send_weather_alerts',
    'send_daily_weather_summary',
    'notify_location_weather',
    'notify_route_weather'
]
