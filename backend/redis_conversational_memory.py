#!/usr/bin/env python3
"""
Redis-Based Conversational Memory System
=======================================

Redis-backed conversation memory for persistence and scalability.
Replaces in-memory dictionaries with Redis storage.
"""

import json
import redis
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    timestamp: str
    user_query: str
    normalized_query: str
    intent: str
    entities: Dict[str, Any]
    response: str
    confidence: float

@dataclass
class UserPreferences:
    """User preferences learned from conversation"""
    preferred_districts: List[str]
    preferred_cuisines: List[str]
    budget_level: Optional[str]
    dietary_restrictions: List[str]
    favorite_vibes: List[str]
    visit_frequency: int
    last_active: str

class RedisConversationalMemory:
    """Redis-backed conversational memory system"""
    
    def __init__(self, redis_client, max_turns: int = 10, session_timeout_hours: int = 24):
        self.redis_client = redis_client
        self.max_turns = max_turns
        self.session_timeout_seconds = session_timeout_hours * 3600
        
        # Redis key prefixes
        self.conversation_prefix = "conversation:"
        self.preferences_prefix = "preferences:"
        self.session_list_key = "active_sessions"
        
    def _get_conversation_key(self, session_id: str) -> str:
        """Get Redis key for conversation"""
        return f"{self.conversation_prefix}{session_id}"
    
    def _get_preferences_key(self, session_id: str) -> str:
        """Get Redis key for user preferences"""
        return f"{self.preferences_prefix}{session_id}"
    
    def add_turn(self, session_id: str, turn: ConversationTurn):
        """Add a conversation turn with Redis persistence"""
        try:
            conv_key = self._get_conversation_key(session_id)
            
            # Get existing conversation
            existing_data = self.redis_client.get(conv_key)
            if existing_data:
                conversation = json.loads(existing_data)
            else:
                conversation = []
            
            # Add new turn
            conversation.append(asdict(turn))
            
            # Keep only last max_turns
            if len(conversation) > self.max_turns:
                conversation = conversation[-self.max_turns:]
            
            # Store back in Redis with TTL
            self.redis_client.setex(
                conv_key, 
                self.session_timeout_seconds, 
                json.dumps(conversation, ensure_ascii=False)
            )
            
            # Add to active sessions set
            self.redis_client.sadd(self.session_list_key, session_id)
            self.redis_client.expire(self.session_list_key, self.session_timeout_seconds)
            
            return True
            
        except Exception as e:
            print(f"❌ Redis error adding turn: {e}")
            return False
    
    def get_conversation(self, session_id: str) -> List[ConversationTurn]:
        """Get conversation history from Redis"""
        try:
            conv_key = self._get_conversation_key(session_id)
            data = self.redis_client.get(conv_key)
            
            if not data:
                return []
            
            conversation_data = json.loads(data)
            return [ConversationTurn(**turn) for turn in conversation_data]
            
        except Exception as e:
            print(f"❌ Redis error getting conversation: {e}")
            return []
    
    def update_preferences(self, session_id: str, preferences: UserPreferences):
        """Update user preferences in Redis"""
        try:
            pref_key = self._get_preferences_key(session_id)
            
            self.redis_client.setex(
                pref_key,
                self.session_timeout_seconds,
                json.dumps(asdict(preferences), ensure_ascii=False)
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Redis error updating preferences: {e}")
            return False
    
    def get_preferences(self, session_id: str) -> Optional[UserPreferences]:
        """Get user preferences from Redis"""
        try:
            pref_key = self._get_preferences_key(session_id)
            data = self.redis_client.get(pref_key)
            
            if not data:
                return None
            
            pref_data = json.loads(data)
            return UserPreferences(**pref_data)
            
        except Exception as e:
            print(f"❌ Redis error getting preferences: {e}")
            return None
    
    def get_context(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """Get conversation context for query processing"""
        conversation = self.get_conversation(session_id)
        preferences = self.get_preferences(session_id)
        
        # Build context from recent conversation
        context = {
            "session_id": session_id,
            "turn_count": len(conversation),
            "recent_intents": [],
            "recent_entities": {},
            "user_preferences": asdict(preferences) if preferences else {},
            "last_response": None,
            "conversation_flow": []
        }
        
        # Extract context from recent turns
        for turn in conversation[-3:]:  # Last 3 turns
            context["recent_intents"].append(turn.intent)
            
            # Merge entities
            for entity_type, values in turn.entities.items():
                if entity_type not in context["recent_entities"]:
                    context["recent_entities"][entity_type] = []
                if isinstance(values, list):
                    context["recent_entities"][entity_type].extend(values)
                else:
                    context["recent_entities"][entity_type].append(values)
            
            context["conversation_flow"].append({
                "query": turn.user_query,
                "intent": turn.intent,
                "response_summary": turn.response[:100] + "..." if len(turn.response) > 100 else turn.response
            })
        
        if conversation:
            context["last_response"] = conversation[-1].response
        
        return context
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions (called periodically)"""
        try:
            # Get all active sessions
            active_sessions = self.redis_client.smembers(self.session_list_key)
            
            cleaned = 0
            for session_id in active_sessions:
                # Check if conversation key exists
                conv_key = self._get_conversation_key(session_id)
                if not self.redis_client.exists(conv_key):
                    # Remove from active sessions
                    self.redis_client.srem(self.session_list_key, session_id)
                    cleaned += 1
            
            return cleaned
            
        except Exception as e:
            print(f"❌ Redis cleanup error: {e}")
            return 0
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get Redis session statistics"""
        try:
            active_count = self.redis_client.scard(self.session_list_key)
            
            # Get memory usage
            info = self.redis_client.info('memory')
            memory_usage = info.get('used_memory_human', 'unknown')
            
            return {
                "active_sessions": active_count,
                "redis_memory": memory_usage,
                "redis_connected": True
            }
            
        except Exception as e:
            return {
                "active_sessions": 0,
                "redis_memory": "unknown",
                "redis_connected": False,
                "error": str(e)
            }

# Global Redis memory instance
redis_memory = None

def initialize_redis_memory(redis_client):
    """Initialize Redis-based memory system"""
    global redis_memory
    if redis_client:
        redis_memory = RedisConversationalMemory(redis_client)
        print("✅ Redis conversational memory initialized")
        return redis_memory
    else:
        print("⚠️ Redis not available, using fallback memory")
        return None

def process_with_redis_context(session_id: str, query: str, response: str, intent: str, entities: dict, confidence: float):
    """Process conversation turn with Redis context"""
    if not redis_memory:
        return {"error": "Redis memory not initialized"}
    
    # Create turn
    turn = ConversationTurn(
        timestamp=datetime.now().isoformat(),
        user_query=query,
        normalized_query=query.lower().strip(),
        intent=intent,
        entities=entities,
        response=response,
        confidence=confidence
    )
    
    # Add to Redis
    success = redis_memory.add_turn(session_id, turn)
    
    # Get updated context
    context = redis_memory.get_context(session_id, query)
    
    return {
        "success": success,
        "context": context,
        "session_stats": redis_memory.get_session_stats()
    }
