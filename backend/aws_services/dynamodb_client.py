"""
AWS DynamoDB Client
Handles sessions, cache, and analytics storage in DynamoDB
"""

import os
import json
import time
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DynamoDBClient:
    """Client for interacting with AWS DynamoDB"""
    
    def __init__(self):
        """Initialize DynamoDB client"""
        self.aws_region = os.getenv('AWS_REGION', 'eu-central-1')
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        # Table names
        self.session_table = os.getenv('AWS_DYNAMODB_SESSION_TABLE', 'ai_istanbul_sessions')
        self.cache_table = os.getenv('AWS_DYNAMODB_CACHE_TABLE', 'ai_istanbul_cache')
        self.analytics_table = os.getenv('AWS_DYNAMODB_ANALYTICS_TABLE', 'ai_istanbul_analytics')
        
        # Initialize client
        self.dynamodb = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize boto3 DynamoDB client"""
        try:
            if self.aws_access_key and self.aws_secret_key:
                self.dynamodb = boto3.resource(
                    'dynamodb',
                    region_name=self.aws_region,
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key
                )
                logger.info(f"✅ DynamoDB client initialized (region: {self.aws_region})")
            else:
                logger.warning("⚠️ AWS credentials not configured for DynamoDB")
        except Exception as e:
            logger.error(f"❌ Failed to initialize DynamoDB client: {e}")
    
    def is_available(self) -> bool:
        """Check if DynamoDB is available"""
        return self.dynamodb is not None
    
    # ===== Session Management =====
    
    async def save_session(self, session_id: str, user_id: str, data: Dict[str, Any]) -> bool:
        """
        Save user session to DynamoDB
        
        Args:
            session_id: Unique session identifier
            user_id: User identifier
            data: Session data to store
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            table = self.dynamodb.Table(self.session_table)
            
            item = {
                'session_id': session_id,
                'timestamp': int(time.time()),
                'user_id': user_id,
                'data': json.dumps(data),
                'ttl': int(time.time()) + 86400  # 24 hours TTL
            }
            
            table.put_item(Item=item)
            logger.debug(f"Session saved: {session_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session from DynamoDB
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session data if found, None otherwise
        """
        if not self.is_available():
            return None
        
        try:
            table = self.dynamodb.Table(self.session_table)
            
            response = table.get_item(Key={'session_id': session_id})
            
            if 'Item' in response:
                item = response['Item']
                return {
                    'session_id': item['session_id'],
                    'user_id': item.get('user_id'),
                    'data': json.loads(item.get('data', '{}')),
                    'timestamp': item.get('timestamp')
                }
            
            return None
            
        except ClientError as e:
            logger.error(f"Error retrieving session: {e}")
            return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session from DynamoDB"""
        if not self.is_available():
            return False
        
        try:
            table = self.dynamodb.Table(self.session_table)
            table.delete_item(Key={'session_id': session_id})
            logger.debug(f"Session deleted: {session_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    # ===== Cache Management =====
    
    async def set_cache(self, cache_key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """
        Store value in DynamoDB cache
        
        Args:
            cache_key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl_seconds: Time to live in seconds (default: 1 hour)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            table = self.dynamodb.Table(self.cache_table)
            
            item = {
                'cache_key': cache_key,
                'value': json.dumps(value),
                'created_at': int(time.time()),
                'ttl': int(time.time()) + ttl_seconds
            }
            
            table.put_item(Item=item)
            logger.debug(f"Cache set: {cache_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def get_cache(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve value from DynamoDB cache
        
        Args:
            cache_key: Cache key
        
        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.is_available():
            return None
        
        try:
            table = self.dynamodb.Table(self.cache_table)
            
            response = table.get_item(Key={'cache_key': cache_key})
            
            if 'Item' in response:
                item = response['Item']
                
                # Check if expired
                if item.get('ttl', 0) < int(time.time()):
                    await self.delete_cache(cache_key)
                    return None
                
                return json.loads(item.get('value', 'null'))
            
            return None
            
        except ClientError as e:
            logger.error(f"Error retrieving cache: {e}")
            return None
    
    async def delete_cache(self, cache_key: str) -> bool:
        """Delete cache entry"""
        if not self.is_available():
            return False
        
        try:
            table = self.dynamodb.Table(self.cache_table)
            table.delete_item(Key={'cache_key': cache_key})
            return True
            
        except ClientError as e:
            logger.error(f"Error deleting cache: {e}")
            return False
    
    # ===== Analytics =====
    
    async def log_analytics_event(
        self,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """
        Log analytics event to DynamoDB
        
        Args:
            user_id: User identifier
            event_type: Type of event (e.g., 'query', 'click', 'error')
            event_data: Event data
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            table = self.dynamodb.Table(self.analytics_table)
            
            item = {
                'user_id': user_id,
                'event_timestamp': int(time.time() * 1000),  # Milliseconds
                'event_type': event_type,
                'event_data': json.dumps(event_data),
                'date': datetime.utcnow().strftime('%Y-%m-%d')
            }
            
            table.put_item(Item=item)
            logger.debug(f"Analytics event logged: {event_type} for {user_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error logging analytics: {e}")
            return False
    
    async def get_user_analytics(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve analytics events for a user
        
        Args:
            user_id: User identifier
            days: Number of days to retrieve (default: 7)
        
        Returns:
            List of analytics events
        """
        if not self.is_available():
            return []
        
        try:
            table = self.dynamodb.Table(self.analytics_table)
            
            # Calculate timestamp for filtering
            start_timestamp = int((time.time() - (days * 86400)) * 1000)
            
            response = table.query(
                KeyConditionExpression='user_id = :uid AND event_timestamp > :ts',
                ExpressionAttributeValues={
                    ':uid': user_id,
                    ':ts': start_timestamp
                }
            )
            
            events = []
            for item in response.get('Items', []):
                events.append({
                    'user_id': item['user_id'],
                    'timestamp': item['event_timestamp'],
                    'type': item.get('event_type'),
                    'data': json.loads(item.get('event_data', '{}')),
                    'date': item.get('date')
                })
            
            return events
            
        except ClientError as e:
            logger.error(f"Error retrieving analytics: {e}")
            return []


# Singleton instance
_dynamodb_client: Optional[DynamoDBClient] = None


def get_dynamodb_client() -> DynamoDBClient:
    """Get or create DynamoDB client singleton"""
    global _dynamodb_client
    if _dynamodb_client is None:
        _dynamodb_client = DynamoDBClient()
    return _dynamodb_client
