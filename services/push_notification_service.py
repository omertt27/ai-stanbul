#!/usr/bin/env python3
"""
Push Notification System for Istanbul Daily Talk AI
Supports WebSocket real-time notifications, FCM push notifications, and in-app notifications
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, asdict
import uuid

try:
    import firebase_admin
    from firebase_admin import credentials, messaging
    FCM_AVAILABLE = True
except ImportError:
    FCM_AVAILABLE = False

try:
    from fastapi import WebSocket
    from fastapi.websockets import WebSocketDisconnect
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications"""
    ROUTE_UPDATE = "route_update"
    ATTRACTION_RECOMMENDATION = "attraction_recommendation" 
    WEATHER_ALERT = "weather_alert"
    TRAFFIC_UPDATE = "traffic_update"
    PERSONALIZED_TIP = "personalized_tip"
    SYSTEM_MESSAGE = "system_message"
    CHAT_RESPONSE = "chat_response"
    REAL_TIME_ADVICE = "real_time_advice"
    LOCATION_BASED = "location_based"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Notification data structure"""
    id: str
    user_id: str
    type: NotificationType
    title: str
    message: str
    data: Dict[str, Any] = None
    priority: NotificationPriority = NotificationPriority.NORMAL
    created_at: datetime = None
    expires_at: datetime = None
    read: bool = False
    action_url: Optional[str] = None
    icon: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.expires_at is None:
            # Default expiry: 24 hours for normal, 1 hour for urgent
            hours = 1 if self.priority == NotificationPriority.URGENT else 24
            self.expires_at = self.created_at + timedelta(hours=hours)
        if self.data is None:
            self.data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'type': self.type.value,
            'title': self.title,
            'message': self.message,
            'data': self.data,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat(),
            'read': self.read,
            'action_url': self.action_url,
            'icon': self.icon
        }


class WebSocketManager:
    """Manages WebSocket connections for real-time notifications"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_connections: Dict[str, Set[str]] = {}  # user_id -> set of connection_ids
        self.connection_users: Dict[str, str] = {}  # connection_id -> user_id
        
    async def connect(self, websocket: WebSocket, user_id: str) -> str:
        """Connect a new WebSocket"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available")
            return None
            
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        
        self.active_connections[connection_id] = websocket
        self.connection_users[connection_id] = user_id
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id} for user {user_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket"""
        if connection_id in self.active_connections:
            user_id = self.connection_users.get(connection_id)
            
            del self.active_connections[connection_id]
            del self.connection_users[connection_id]
            
            if user_id and user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_user(self, user_id: str, notification: Notification) -> bool:
        """Send notification to all connections for a user"""
        if user_id not in self.user_connections:
            return False
        
        message = json.dumps(notification.to_dict())
        connections_to_remove = []
        sent_count = 0
        
        for connection_id in self.user_connections[user_id].copy():
            websocket = self.active_connections.get(connection_id)
            if websocket:
                try:
                    await websocket.send_text(message)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send to connection {connection_id}: {e}")
                    connections_to_remove.append(connection_id)
        
        # Clean up failed connections
        for connection_id in connections_to_remove:
            self.disconnect(connection_id)
        
        return sent_count > 0
    
    async def broadcast(self, notification: Notification):
        """Broadcast notification to all connected users"""
        message = json.dumps(notification.to_dict())
        
        for connection_id, websocket in list(self.active_connections.items()):
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to broadcast to {connection_id}: {e}")
                self.disconnect(connection_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            'total_connections': len(self.active_connections),
            'unique_users': len(self.user_connections),
            'average_connections_per_user': len(self.active_connections) / max(len(self.user_connections), 1),
            'websocket_available': WEBSOCKET_AVAILABLE
        }


class FCMNotificationService:
    """Firebase Cloud Messaging notification service"""
    
    def __init__(self, service_account_path: Optional[str] = None):
        self.initialized = False
        self.fcm_available = FCM_AVAILABLE
        
        if FCM_AVAILABLE and service_account_path:
            try:
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
                self.initialized = True
                logger.info("✅ FCM service initialized")
            except Exception as e:
                logger.warning(f"FCM initialization failed: {e}")
        else:
            logger.info("📱 FCM service not available (mock mode)")
    
    async def send_notification(self, device_token: str, notification: Notification) -> bool:
        """Send push notification via FCM"""
        if not self.initialized:
            logger.info(f"📱 Mock FCM notification to {device_token[:10]}...")
            logger.info(f"   Title: {notification.title}")
            logger.info(f"   Body: {notification.message}")
            return True
        
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=notification.title,
                    body=notification.message,
                    image=notification.icon
                ),
                data={
                    'notification_id': notification.id,
                    'type': notification.type.value,
                    'priority': notification.priority.value,
                    'action_url': notification.action_url or '',
                    **{k: str(v) for k, v in notification.data.items()}
                },
                token=device_token,
                android=messaging.AndroidConfig(
                    priority='high' if notification.priority in [NotificationPriority.HIGH, NotificationPriority.URGENT] else 'normal',
                    notification=messaging.AndroidNotification(
                        icon='ic_notification',
                        color='#FF6B35',
                        channel_id='istanbul_ai_notifications'
                    )
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            alert=messaging.ApsAlert(
                                title=notification.title,
                                body=notification.message
                            ),
                            badge=1,
                            sound='default'
                        )
                    )
                )
            )
            
            response = messaging.send(message)
            logger.info(f"✅ FCM notification sent: {response}")
            return True
            
        except Exception as e:
            logger.error(f"❌ FCM notification failed: {e}")
            return False
    
    async def send_to_topic(self, topic: str, notification: Notification) -> bool:
        """Send notification to a topic (group of users)"""
        if not self.initialized:
            logger.info(f"📱 Mock FCM topic notification to {topic}")
            return True
        
        try:
            message = messaging.Message(
                notification=messaging.Notification(
                    title=notification.title,
                    body=notification.message
                ),
                data={
                    'notification_id': notification.id,
                    'type': notification.type.value,
                    **{k: str(v) for k, v in notification.data.items()}
                },
                topic=topic
            )
            
            response = messaging.send(message)
            logger.info(f"✅ FCM topic notification sent to {topic}: {response}")
            return True
            
        except Exception as e:
            logger.error(f"❌ FCM topic notification failed: {e}")
            return False


class NotificationStore:
    """In-memory store for notifications with persistence support"""
    
    def __init__(self, max_per_user: int = 100):
        self.notifications: Dict[str, List[Notification]] = {}  # user_id -> notifications
        self.max_per_user = max_per_user
    
    def store_notification(self, notification: Notification):
        """Store a notification"""
        user_id = notification.user_id
        
        if user_id not in self.notifications:
            self.notifications[user_id] = []
        
        self.notifications[user_id].append(notification)
        
        # Keep only the most recent notifications
        if len(self.notifications[user_id]) > self.max_per_user:
            self.notifications[user_id] = self.notifications[user_id][-self.max_per_user:]
        
        # Clean up expired notifications
        self._cleanup_expired_notifications(user_id)
    
    def get_notifications(self, user_id: str, unread_only: bool = False, limit: int = 50) -> List[Notification]:
        """Get notifications for a user"""
        if user_id not in self.notifications:
            return []
        
        notifications = self.notifications[user_id]
        
        if unread_only:
            notifications = [n for n in notifications if not n.read]
        
        # Sort by creation time (newest first)
        notifications.sort(key=lambda x: x.created_at, reverse=True)
        
        return notifications[:limit]
    
    def mark_as_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read"""
        if user_id not in self.notifications:
            return False
        
        for notification in self.notifications[user_id]:
            if notification.id == notification_id:
                notification.read = True
                return True
        
        return False
    
    def mark_all_as_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user"""
        if user_id not in self.notifications:
            return 0
        
        count = 0
        for notification in self.notifications[user_id]:
            if not notification.read:
                notification.read = True
                count += 1
        
        return count
    
    def _cleanup_expired_notifications(self, user_id: str):
        """Remove expired notifications for a user"""
        if user_id not in self.notifications:
            return
        
        current_time = datetime.now()
        self.notifications[user_id] = [
            n for n in self.notifications[user_id]
            if n.expires_at > current_time
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification store statistics"""
        total_notifications = sum(len(notifications) for notifications in self.notifications.values())
        total_unread = sum(
            len([n for n in notifications if not n.read])
            for notifications in self.notifications.values()
        )
        
        return {
            'total_users': len(self.notifications),
            'total_notifications': total_notifications,
            'total_unread': total_unread,
            'average_per_user': total_notifications / max(len(self.notifications), 1)
        }


class NotificationService:
    """Main notification service orchestrating all notification channels"""
    
    def __init__(self, fcm_service_account_path: Optional[str] = None):
        self.websocket_manager = WebSocketManager()
        self.fcm_service = FCMNotificationService(fcm_service_account_path)
        self.notification_store = NotificationStore()
        self.user_device_tokens: Dict[str, str] = {}  # user_id -> FCM token
        self.user_preferences: Dict[str, Dict[str, bool]] = {}  # user_id -> notification preferences
        
        logger.info("🔔 Notification Service initialized")
    
    async def send_notification(self, notification: Notification, channels: List[str] = None) -> Dict[str, bool]:
        """Send notification through specified channels"""
        if channels is None:
            channels = ['websocket', 'fcm', 'store']
        
        results = {}
        
        # Store notification
        if 'store' in channels:
            self.notification_store.store_notification(notification)
            results['store'] = True
        
        # Send via WebSocket
        if 'websocket' in channels:
            results['websocket'] = await self.websocket_manager.send_to_user(
                notification.user_id, notification
            )
        
        # Send via FCM
        if 'fcm' in channels and notification.user_id in self.user_device_tokens:
            device_token = self.user_device_tokens[notification.user_id]
            results['fcm'] = await self.fcm_service.send_notification(device_token, notification)
        
        logger.info(f"📤 Notification sent: {notification.title} -> {results}")
        return results
    
    async def send_route_update(self, user_id: str, route_data: Dict[str, Any]):
        """Send route update notification"""
        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=NotificationType.ROUTE_UPDATE,
            title="🗺️ Route Updated",
            message=f"Your route has been optimized! {route_data.get('optimization_message', '')}",
            data=route_data,
            priority=NotificationPriority.HIGH,
            icon="🗺️"
        )
        
        return await self.send_notification(notification)
    
    async def send_attraction_recommendation(self, user_id: str, attraction_data: Dict[str, Any]):
        """Send attraction recommendation notification"""
        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=NotificationType.ATTRACTION_RECOMMENDATION,
            title="🏛️ New Recommendation",
            message=f"Check out {attraction_data.get('name', 'this amazing place')} nearby!",
            data=attraction_data,
            priority=NotificationPriority.NORMAL,
            icon="🏛️"
        )
        
        return await self.send_notification(notification)
    
    async def send_weather_alert(self, user_id: str, weather_data: Dict[str, Any]):
        """Send weather alert notification"""
        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=NotificationType.WEATHER_ALERT,
            title="🌦️ Weather Alert",
            message=weather_data.get('message', 'Weather conditions have changed'),
            data=weather_data,
            priority=NotificationPriority.HIGH,
            icon="🌦️"
        )
        
        return await self.send_notification(notification)
    
    async def send_traffic_update(self, user_id: str, traffic_data: Dict[str, Any]):
        """Send traffic update notification"""
        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=NotificationType.TRAFFIC_UPDATE,
            title="🚦 Traffic Update",
            message=traffic_data.get('message', 'Traffic conditions have changed on your route'),
            data=traffic_data,
            priority=NotificationPriority.NORMAL,
            icon="🚦"
        )
        
        return await self.send_notification(notification)
    
    async def send_personalized_tip(self, user_id: str, tip_data: Dict[str, Any]):
        """Send personalized tip notification"""
        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=NotificationType.PERSONALIZED_TIP,
            title="💡 Personal Tip",
            message=tip_data.get('message', 'Here\'s a personalized tip for you!'),
            data=tip_data,
            priority=NotificationPriority.LOW,
            icon="💡"
        )
        
        return await self.send_notification(notification)
    
    async def send_chat_response(self, user_id: str, response_data: Dict[str, Any]):
        """Send chat response notification (for background responses)"""
        notification = Notification(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=NotificationType.CHAT_RESPONSE,
            title="💬 New Response",
            message=response_data.get('preview', 'You have a new message'),
            data=response_data,
            priority=NotificationPriority.NORMAL,
            icon="💬"
        )
        
        return await self.send_notification(notification)
    
    def register_device_token(self, user_id: str, device_token: str):
        """Register FCM device token for a user"""
        self.user_device_tokens[user_id] = device_token
        logger.info(f"📱 Device token registered for user {user_id}")
    
    def update_notification_preferences(self, user_id: str, preferences: Dict[str, bool]):
        """Update notification preferences for a user"""
        self.user_preferences[user_id] = preferences
        logger.info(f"⚙️ Notification preferences updated for user {user_id}")
    
    def get_user_notifications(self, user_id: str, unread_only: bool = False, limit: int = 50) -> List[Dict[str, Any]]:
        """Get notifications for a user"""
        notifications = self.notification_store.get_notifications(user_id, unread_only, limit)
        return [n.to_dict() for n in notifications]
    
    def mark_notification_read(self, user_id: str, notification_id: str) -> bool:
        """Mark a notification as read"""
        return self.notification_store.mark_as_read(user_id, notification_id)
    
    def mark_all_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user"""
        return self.notification_store.mark_all_as_read(user_id)
    
    async def connect_websocket(self, websocket: WebSocket, user_id: str) -> str:
        """Connect a WebSocket for real-time notifications"""
        return await self.websocket_manager.connect(websocket, user_id)
    
    def disconnect_websocket(self, connection_id: str):
        """Disconnect a WebSocket"""
        self.websocket_manager.disconnect(connection_id)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        return {
            'websocket': self.websocket_manager.get_connection_stats(),
            'fcm': {
                'available': self.fcm_service.fcm_available,
                'initialized': self.fcm_service.initialized,
                'registered_devices': len(self.user_device_tokens)
            },
            'notifications': self.notification_store.get_stats(),
            'preferences': {
                'users_with_preferences': len(self.user_preferences)
            }
        }


# Global notification service instance
notification_service = NotificationService()

# Export for use in other modules
__all__ = [
    'NotificationService',
    'NotificationType', 
    'NotificationPriority',
    'Notification',
    'notification_service'
]
