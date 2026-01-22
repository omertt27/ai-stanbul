"""
Conversation Memory Service
===========================

Redis-backed conversation memory for maintaining context across chat turns.
Enables the chatbot to remember previous messages and provide coherent multi-turn conversations.

Features:
- Session-based conversation storage (Redis)
- Automatic sliding window (keeps last N turns)
- Language preference persistence
- Entity tracking across turns
- TTL-based automatic cleanup

Author: AI Istanbul Team
Date: January 2026
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - using in-memory conversation storage")


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    language: Optional[str] = None
    intent: Optional[str] = None
    entities: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        return cls(**data)


class ConversationMemory:
    """
    Redis-backed conversation memory service.
    
    Stores and retrieves conversation history for multi-turn chat.
    Falls back to in-memory storage if Redis is unavailable.
    """
    
    def __init__(
        self,
        redis_client: Optional[Any] = None,
        max_turns: int = 10,
        ttl_seconds: int = 3600  # 1 hour default
    ):
        """
        Initialize conversation memory.
        
        Args:
            redis_client: Redis client instance (optional)
            max_turns: Maximum turns to keep in history (sliding window)
            ttl_seconds: Time-to-live for conversation data
        """
        self.redis = redis_client
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        
        # In-memory fallback storage
        self._memory_store: Dict[str, Dict[str, Any]] = {}
        
        if self.redis:
            logger.info(f"✅ Conversation Memory initialized with Redis (max_turns={max_turns}, ttl={ttl_seconds}s)")
        else:
            logger.warning("⚠️ Conversation Memory using in-memory storage (no Redis)")
    
    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"conv_memory:{session_id}"
    
    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        language: Optional[str] = None,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to memory.
        
        Args:
            session_id: Unique session identifier
            role: 'user' or 'assistant'
            content: Message content
            language: Detected language
            intent: Detected intent
            entities: Extracted entities
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            language=language,
            intent=intent,
            entities=entities
        )
        
        # Get existing conversation
        conversation = await self.get_conversation(session_id)
        
        # Add new turn
        conversation['turns'].append(turn.to_dict())
        
        # Apply sliding window
        if len(conversation['turns']) > self.max_turns:
            conversation['turns'] = conversation['turns'][-self.max_turns:]
        
        # Update language preference (use most recent)
        if language:
            conversation['language'] = language
        
        # Update last interaction time
        conversation['updated_at'] = datetime.now().isoformat()
        
        # Save back to storage
        await self._save_conversation(session_id, conversation)
        
        logger.debug(f"💬 Added {role} turn to session {session_id} (total: {len(conversation['turns'])} turns)")
    
    async def get_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Conversation data with turns, language, etc.
        """
        key = self._get_key(session_id)
        
        if self.redis:
            try:
                data = self.redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"Redis read error: {e}")
        else:
            if session_id in self._memory_store:
                return self._memory_store[session_id]
        
        # Return empty conversation structure
        return {
            'session_id': session_id,
            'turns': [],
            'language': 'en',  # Default language
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
    
    async def get_history_for_llm(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM context.
        
        Returns list of {"role": "user/assistant", "content": "..."} dicts
        suitable for passing to chat models.
        
        Args:
            session_id: Unique session identifier
            max_turns: Override max turns (optional)
            
        Returns:
            List of message dicts for LLM
        """
        conversation = await self.get_conversation(session_id)
        turns = conversation.get('turns', [])
        
        # Apply optional limit
        limit = max_turns or self.max_turns
        recent_turns = turns[-limit:] if len(turns) > limit else turns
        
        # Format for LLM
        return [
            {"role": turn['role'], "content": turn['content']}
            for turn in recent_turns
        ]
    
    async def get_session_language(self, session_id: str) -> str:
        """
        Get the language preference for a session.
        
        Returns the most recently detected language, or 'en' as default.
        """
        conversation = await self.get_conversation(session_id)
        return conversation.get('language', 'en')
    
    async def get_last_entities(self, session_id: str) -> Dict[str, Any]:
        """
        Get the most recent entities from conversation.
        
        Useful for resolving references like "there", "that restaurant", etc.
        """
        conversation = await self.get_conversation(session_id)
        turns = conversation.get('turns', [])
        
        # Aggregate entities from recent turns (most recent first)
        entities = {}
        for turn in reversed(turns[-5:]):  # Last 5 turns
            if turn.get('entities'):
                for key, value in turn['entities'].items():
                    if key not in entities:  # Don't overwrite more recent
                        entities[key] = value
        
        return entities
    
    async def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        key = self._get_key(session_id)
        
        if self.redis:
            try:
                self.redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        else:
            self._memory_store.pop(session_id, None)
        
        logger.info(f"🗑️ Cleared conversation for session {session_id}")
    
    async def _save_conversation(self, session_id: str, data: Dict[str, Any]) -> None:
        """Save conversation to storage."""
        key = self._get_key(session_id)
        
        if self.redis:
            try:
                self.redis.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(data, ensure_ascii=False)
                )
            except Exception as e:
                logger.warning(f"Redis write error: {e}, falling back to memory")
                self._memory_store[session_id] = data
        else:
            self._memory_store[session_id] = data


# Singleton instance
_conversation_memory: Optional[ConversationMemory] = None


def get_conversation_memory(redis_client: Optional[Any] = None) -> ConversationMemory:
    """
    Get or create the conversation memory singleton.
    
    Args:
        redis_client: Optional Redis client to use
        
    Returns:
        ConversationMemory instance
    """
    global _conversation_memory
    
    if _conversation_memory is None:
        # Try to get Redis from centralized client
        if redis_client is None:
            try:
                from core.redis_client import get_redis_client
                redis_client = get_redis_client()
            except Exception as e:
                logger.warning(f"Could not get centralized Redis client: {e}")
        
        _conversation_memory = ConversationMemory(
            redis_client=redis_client,
            max_turns=10,  # Keep last 10 turns
            ttl_seconds=3600  # 1 hour TTL
        )
    
    return _conversation_memory
