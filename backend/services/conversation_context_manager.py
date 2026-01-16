"""
Conversation Context Manager
Tracks conversation history and maintains context across multiple turns
"""

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field
from collections import deque

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - using in-memory context storage")


@dataclass
class Turn:
    """Represents a single conversation turn"""
    query: str
    preprocessed_query: str
    intent: str
    entities: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'query': self.query,
            'preprocessed_query': self.preprocessed_query,
            'intent': self.intent,
            'entities': self.entities,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Turn':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConversationContext:
    """Represents the conversation context for a session"""
    session_id: str
    conversation_history: List[Turn] = field(default_factory=list)
    persistent_entities: Dict[str, Any] = field(default_factory=dict)
    intent_history: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    last_location: Optional[str] = None
    last_restaurant: Optional[str] = None
    last_museum: Optional[str] = None
    last_attraction: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_turn(self, turn: Turn, max_history: int = 5):
        """Add a turn to conversation history with sliding window"""
        self.conversation_history.append(turn)
        
        # Keep only last N turns (sliding window)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
        
        self.updated_at = datetime.now()
    
    def add_intent(self, intent: str, max_history: int = 10):
        """Add intent to history"""
        self.intent_history.append(intent)
        
        # Keep only last N intents
        if len(self.intent_history) > max_history:
            self.intent_history = self.intent_history[-max_history:]
    
    def update_entities(self, entities: Dict[str, Any]):
        """Update persistent entities from turn"""
        # Update locations
        if 'locations' in entities and entities['locations']:
            locations = entities['locations']
            if isinstance(locations, list) and locations:
                self.last_location = locations[-1]  # Most recent location
                if 'locations' not in self.persistent_entities:
                    self.persistent_entities['locations'] = []
                # Add new locations that aren't already tracked
                for loc in locations:
                    if loc not in self.persistent_entities['locations']:
                        self.persistent_entities['locations'].append(loc)
        
        # Update restaurants
        if 'restaurant' in entities:
            self.last_restaurant = entities['restaurant']
        
        # Update museums
        if 'museum' in entities:
            self.last_museum = entities['museum']
        
        # Update attractions
        if 'attraction' in entities:
            self.last_attraction = entities['attraction']
        
        # Update cuisines
        if 'cuisines' in entities and entities['cuisines']:
            cuisines = entities['cuisines']
            cuisines_list = cuisines if isinstance(cuisines, list) else [cuisines]
            if 'cuisines' not in self.persistent_entities:
                self.persistent_entities['cuisines'] = []
            # Add new cuisines that aren't already tracked
            for cuisine in cuisines_list:
                if cuisine not in self.persistent_entities['cuisines']:
                    self.persistent_entities['cuisines'].append(cuisine)
        
        # Update price preferences
        if 'price_range' in entities:
            self.user_preferences['price_range'] = entities['price_range']
        
        # Update party size
        if 'party_size' in entities:
            self.user_preferences['party_size'] = entities['party_size']
    
    def get_last_intent(self) -> Optional[str]:
        """Get the most recent intent"""
        return self.intent_history[-1] if self.intent_history else None
    
    def get_last_turn(self) -> Optional[Turn]:
        """Get the most recent turn"""
        return self.conversation_history[-1] if self.conversation_history else None
    
    def get_intent_sequence(self, n: int = 3) -> List[str]:
        """Get last N intents as sequence"""
        return self.intent_history[-n:] if len(self.intent_history) >= n else self.intent_history
    
    def last_mentioned_entity(self) -> Optional[str]:
        """Get the most recently mentioned entity"""
        return self.last_restaurant or self.last_museum or self.last_attraction or self.last_location
    
    def has_recent_intent(self, intent: str, within_turns: int = 3) -> bool:
        """Check if intent appeared in recent turns"""
        recent_intents = self.intent_history[-within_turns:]
        return intent in recent_intents
    
    def get_conversation_depth(self) -> int:
        """Get number of turns in conversation"""
        return len(self.conversation_history)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'conversation_history': [turn.to_dict() for turn in self.conversation_history],
            'persistent_entities': self.persistent_entities,
            'intent_history': self.intent_history,
            'user_preferences': self.user_preferences,
            'last_location': self.last_location,
            'last_restaurant': self.last_restaurant,
            'last_museum': self.last_museum,
            'last_attraction': self.last_attraction,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationContext':
        """Create from dictionary"""
        data['conversation_history'] = [Turn.from_dict(turn) for turn in data['conversation_history']]
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ConversationContext':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ConversationContextManager:
    """
    Manages conversation context across multiple turns
    Uses Redis for persistent storage with in-memory fallback
    """
    
    def __init__(self, redis_url: str = None, redis_host: str = None, redis_port: int = 6379, redis_db: int = 1):
        """Initialize context manager with Redis or in-memory storage"""
        self.redis_client = None
        self.in_memory_storage = {}  # Fallback storage
        
        if REDIS_AVAILABLE:
            try:
                # Prefer REDIS_URL from environment, then redis_url parameter, then host/port
                redis_connection_url = redis_url or os.getenv('REDIS_URL')
                
                if redis_connection_url:
                    # Use redis.from_url for full URL support (including auth)
                    self.redis_client = redis.from_url(
                        redis_connection_url,
                        decode_responses=True,
                        socket_connect_timeout=30,
                        socket_timeout=30,
                        db=redis_db  # Override DB if needed
                    )
                else:
                    # Fallback to host/port for backward compatibility
                    host = redis_host or 'localhost'
                    self.redis_client = redis.Redis(
                        host=host,
                        port=redis_port,
                        db=redis_db,
                        decode_responses=True,
                        socket_connect_timeout=30,
                        socket_timeout=30
                    )
                
                # Test connection
                self.redis_client.ping()
                logger.info("✅ ConversationContextManager initialized with Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using in-memory storage")
                self.redis_client = None
        else:
            logger.info("✅ ConversationContextManager initialized with in-memory storage")
        
        self.ttl = timedelta(hours=24)  # Context expires after 24 hours
        self.max_history = 5  # Maximum turns to keep
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Retrieve conversation context for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            ConversationContext or None if not found
        """
        try:
            if self.redis_client:
                # Get from Redis
                key = f"context:{session_id}"
                data = self.redis_client.get(key)
                
                if data:
                    context = ConversationContext.from_json(data)
                    logger.debug(f"📖 Retrieved context for session {session_id} (depth: {context.get_conversation_depth()})")
                    return context
            else:
                # Get from in-memory storage
                if session_id in self.in_memory_storage:
                    context = self.in_memory_storage[session_id]
                    logger.debug(f"📖 Retrieved context for session {session_id} from memory")
                    return context
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving context for session {session_id}: {e}")
            return None
    
    def update_context(self, session_id: str, turn: Turn) -> bool:
        """
        Update conversation context with a new turn
        
        Args:
            session_id: Session identifier
            turn: New conversation turn
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get or create context
            context = self.get_context(session_id)
            if context is None:
                context = ConversationContext(session_id=session_id)
                logger.info(f"🆕 Created new context for session {session_id}")
            
            # Add turn to history
            context.add_turn(turn, self.max_history)
            
            # Update persistent entities
            context.update_entities(turn.entities)
            
            # Update intent history
            context.add_intent(turn.intent)
            
            # Store context
            if self.redis_client:
                # Store in Redis with TTL
                key = f"context:{session_id}"
                self.redis_client.setex(
                    key,
                    self.ttl,
                    context.to_json()
                )
                logger.debug(f"💾 Updated context for session {session_id} in Redis")
            else:
                # Store in memory
                self.in_memory_storage[session_id] = context
                logger.debug(f"💾 Updated context for session {session_id} in memory")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating context for session {session_id}: {e}")
            return False
    
    def extract_context_features(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Extract features from conversation context
        
        Args:
            context: Conversation context
            
        Returns:
            Dictionary of context features
        """
        last_turn = context.get_last_turn()
        last_intent = context.get_last_intent()
        
        features = {
            'has_previous_intent': last_intent is not None,
            'previous_intent': last_intent,
            'intent_sequence': '-'.join(context.get_intent_sequence(3)),
            'has_persistent_location': context.last_location is not None,
            'persistent_location': context.last_location,
            'persistent_entities_count': len(context.persistent_entities),
            'conversation_depth': context.get_conversation_depth(),
            'time_since_last_turn': (datetime.now() - last_turn.timestamp).total_seconds() if last_turn else 999999,
            'last_confidence': last_turn.confidence if last_turn else 0.0,
            'has_user_preferences': len(context.user_preferences) > 0,
            'user_preferences': context.user_preferences
        }
        
        return features
    
    def resolve_references(self, query: str, context: ConversationContext) -> str:
        """
        Resolve pronouns and implicit references in query
        
        Args:
            query: Original query with possible references
            context: Conversation context
            
        Returns:
            Query with references resolved
        """
        resolved = query
        query_lower = query.lower()
        
        # Pronoun references
        reference_patterns = {
            r'\b(it|that)\b': context.last_mentioned_entity(),
            r'\b(there|that place)\b': context.last_location,
            r'\boraya\b': context.last_location,  # Turkish: "there"
            r'\borda\b': context.last_location,   # Turkish: "there"
        }
        
        for pattern, replacement in reference_patterns.items():
            if replacement and re.search(pattern, query_lower):
                # Replace the reference with the actual entity
                resolved = re.sub(pattern, replacement, resolved, flags=re.IGNORECASE)
                logger.info(f"🔗 Resolved reference: '{query}' → '{resolved}'")
        
        # Implicit location context (if query needs location but doesn't have one)
        needs_location_patterns = [
            r'\brestoranlar\b', r'\bmüzeler\b', r'\bnasıl gid', r'\byakında\b',
            r'\restaurants?\b', r'\bmuseums?\b', r'\bhow to get\b', r'\bnearby\b'
        ]
        
        has_location_patterns = [
            r'\b(beyoğlu|sultanahmet|kadıköy|taksim|beşiktaş|üsküdar|fatih)\b'
        ]
        
        needs_location = any(re.search(p, query_lower) for p in needs_location_patterns)
        has_location = any(re.search(p, query_lower) for p in has_location_patterns)
        
        if needs_location and not has_location and context.last_location:
            resolved = f"{resolved} {context.last_location}'de"
            logger.info(f"🗺️ Added location context: '{query}' → '{resolved}'")
        
        return resolved
    
    def get_relevant_entities(self, context: ConversationContext, intent: str) -> Dict[str, Any]:
        """
        Get entities relevant to the current intent from context
        
        Args:
            context: Conversation context
            intent: Current intent
            
        Returns:
            Dictionary of relevant entities
        """
        relevant = {}
        
        # For restaurant queries
        if intent in ['restaurant_query', 'restaurant_recommendation']:
            if context.last_location:
                relevant['location'] = context.last_location
            if 'cuisines' in context.persistent_entities:
                relevant['cuisines'] = context.persistent_entities['cuisines']
            if 'price_range' in context.user_preferences:
                relevant['price_range'] = context.user_preferences['price_range']
        
        # For attraction queries
        elif intent in ['attraction_query', 'museum_query']:
            if context.last_location:
                relevant['location'] = context.last_location
        
        # For transport queries
        elif intent == 'transport_query':
            if context.last_location:
                relevant['destination'] = context.last_location
        
        return relevant
    
    def clear_context(self, session_id: str) -> bool:
        """
        Clear context for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.redis_client:
                key = f"context:{session_id}"
                self.redis_client.delete(key)
            else:
                if session_id in self.in_memory_storage:
                    del self.in_memory_storage[session_id]
            
            logger.info(f"🗑️ Cleared context for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing context for session {session_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics"""
        if self.redis_client:
            try:
                # Count contexts in Redis
                keys = self.redis_client.keys("context:*")
                total_contexts = len(keys)
            except:
                total_contexts = 0
        else:
            total_contexts = len(self.in_memory_storage)
        
        return {
            'storage_type': 'redis' if self.redis_client else 'memory',
            'total_contexts': total_contexts,
            'max_history': self.max_history,
            'ttl_hours': self.ttl.total_seconds() / 3600
        }


# Singleton instance
_context_manager_instance = None

def get_context_manager() -> ConversationContextManager:
    """Get or create context manager singleton"""
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = ConversationContextManager()
    return _context_manager_instance
