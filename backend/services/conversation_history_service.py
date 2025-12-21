"""
Enhanced Conversation History Service

Persists conversation history to database for:
- Context continuity across sessions
- User preference learning
- Analytics and improvement

Author: AI Istanbul Team
Date: December 2024
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """A single message in a conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        data = data.copy()
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class Conversation:
    """A complete conversation session"""
    id: str
    user_id: Optional[str]
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    language: str = "en"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation"""
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.updated_at = datetime.now()
    
    def get_context_window(self, max_turns: int = 5) -> List[Dict[str, str]]:
        """Get recent messages for LLM context"""
        recent = self.messages[-max_turns * 2:]  # user + assistant pairs
        return [{"role": m.role, "content": m.content} for m in recent]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "language": self.language,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        data = data.copy()
        data['messages'] = [ConversationMessage.from_dict(m) for m in data.get('messages', [])]
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


class ConversationHistoryService:
    """
    Service for managing conversation history.
    
    Supports multiple storage backends:
    - In-memory (default, for development)
    - Redis (for distributed systems)
    - PostgreSQL (for persistence)
    """
    
    def __init__(
        self,
        storage_backend: str = "memory",
        redis_url: Optional[str] = None,
        db_session = None,
        max_conversations_per_user: int = 50,
        conversation_ttl_hours: int = 72
    ):
        """
        Initialize the conversation history service.
        
        Args:
            storage_backend: "memory", "redis", or "postgres"
            redis_url: Redis connection URL (for redis backend)
            db_session: SQLAlchemy session (for postgres backend)
            max_conversations_per_user: Maximum conversations to keep per user
            conversation_ttl_hours: Hours before conversation expires
        """
        self.storage_backend = storage_backend
        self.max_conversations = max_conversations_per_user
        self.ttl_hours = conversation_ttl_hours
        
        # In-memory storage
        self._conversations: Dict[str, Conversation] = {}
        self._user_index: Dict[str, List[str]] = {}  # user_id -> [conversation_ids]
        self._session_index: Dict[str, str] = {}  # session_id -> conversation_id
        
        # Redis client
        self._redis = None
        if storage_backend == "redis" and redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url)
                logger.info("✅ Redis connection established for conversation history")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}, falling back to memory")
                self.storage_backend = "memory"
        
        # Database session
        self._db = db_session
        
        logger.info(f"✅ Conversation History Service initialized (backend: {self.storage_backend})")
    
    def _generate_id(self, session_id: str, user_id: Optional[str] = None) -> str:
        """Generate a unique conversation ID"""
        seed = f"{session_id}:{user_id or 'anonymous'}:{datetime.now().timestamp()}"
        return hashlib.md5(seed.encode()).hexdigest()[:16]
    
    def get_or_create_conversation(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        language: str = "en"
    ) -> Conversation:
        """
        Get existing conversation or create a new one.
        
        Args:
            session_id: Session identifier
            user_id: Optional user identifier
            language: Conversation language
            
        Returns:
            Conversation object
        """
        # Check session index first
        if session_id in self._session_index:
            conv_id = self._session_index[session_id]
            conv = self._get_conversation(conv_id)
            if conv:
                return conv
        
        # Create new conversation
        conv_id = self._generate_id(session_id, user_id)
        conv = Conversation(
            id=conv_id,
            user_id=user_id,
            session_id=session_id,
            language=language
        )
        
        self._save_conversation(conv)
        self._session_index[session_id] = conv_id
        
        if user_id:
            if user_id not in self._user_index:
                self._user_index[user_id] = []
            self._user_index[user_id].append(conv_id)
            
            # Cleanup old conversations if over limit
            self._cleanup_user_conversations(user_id)
        
        return conv
    
    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Add a user-assistant exchange to the conversation.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            assistant_response: Assistant's response
            user_id: Optional user identifier
            metadata: Optional metadata for the exchange
            
        Returns:
            Updated conversation
        """
        conv = self.get_or_create_conversation(session_id, user_id)
        
        # Add user message
        conv.add_message("user", user_message, {
            **(metadata or {}),
            "type": "user_query"
        })
        
        # Add assistant response
        conv.add_message("assistant", assistant_response, {
            **(metadata or {}),
            "type": "assistant_response"
        })
        
        self._save_conversation(conv)
        
        return conv
    
    def get_conversation_context(
        self,
        session_id: str,
        max_turns: int = 5
    ) -> List[Dict[str, str]]:
        """
        Get conversation context for LLM.
        
        Returns recent messages formatted for LLM consumption.
        """
        conv = self._get_conversation_by_session(session_id)
        if not conv:
            return []
        
        return conv.get_context_window(max_turns)
    
    def get_user_conversations(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Conversation]:
        """Get recent conversations for a user."""
        if user_id not in self._user_index:
            return []
        
        conv_ids = self._user_index[user_id][-limit:]
        conversations = []
        
        for conv_id in reversed(conv_ids):
            conv = self._get_conversation(conv_id)
            if conv:
                conversations.append(conv)
        
        return conversations
    
    def search_conversations(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search user's conversations.
        
        Returns matching messages with context.
        """
        results = []
        query_lower = query.lower()
        
        for conv in self.get_user_conversations(user_id, limit=50):
            for i, msg in enumerate(conv.messages):
                if query_lower in msg.content.lower():
                    # Get surrounding context
                    context_start = max(0, i - 1)
                    context_end = min(len(conv.messages), i + 2)
                    context = conv.messages[context_start:context_end]
                    
                    results.append({
                        "conversation_id": conv.id,
                        "message": msg.to_dict(),
                        "context": [m.to_dict() for m in context],
                        "timestamp": conv.updated_at.isoformat()
                    })
                    
                    if len(results) >= limit:
                        return results
        
        return results
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Includes statistics and key topics.
        """
        conv = self._get_conversation_by_session(session_id)
        if not conv:
            return {"error": "Conversation not found"}
        
        user_messages = [m for m in conv.messages if m.role == "user"]
        assistant_messages = [m for m in conv.messages if m.role == "assistant"]
        
        return {
            "id": conv.id,
            "session_id": conv.session_id,
            "language": conv.language,
            "total_messages": len(conv.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "duration_minutes": (conv.updated_at - conv.created_at).total_seconds() / 60
        }
    
    def delete_conversation(self, session_id: str) -> bool:
        """Delete a conversation by session ID."""
        conv = self._get_conversation_by_session(session_id)
        if not conv:
            return False
        
        # Remove from indices
        if session_id in self._session_index:
            del self._session_index[session_id]
        
        if conv.user_id and conv.user_id in self._user_index:
            self._user_index[conv.user_id] = [
                cid for cid in self._user_index[conv.user_id] 
                if cid != conv.id
            ]
        
        # Remove from storage
        if self.storage_backend == "memory":
            if conv.id in self._conversations:
                del self._conversations[conv.id]
        elif self.storage_backend == "redis" and self._redis:
            self._redis.delete(f"conv:{conv.id}")
        
        return True
    
    def _get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get conversation by ID from storage."""
        if self.storage_backend == "memory":
            return self._conversations.get(conv_id)
        
        elif self.storage_backend == "redis" and self._redis:
            data = self._redis.get(f"conv:{conv_id}")
            if data:
                return Conversation.from_dict(json.loads(data))
        
        return None
    
    def _get_conversation_by_session(self, session_id: str) -> Optional[Conversation]:
        """Get conversation by session ID."""
        conv_id = self._session_index.get(session_id)
        if conv_id:
            return self._get_conversation(conv_id)
        return None
    
    def _save_conversation(self, conv: Conversation):
        """Save conversation to storage."""
        if self.storage_backend == "memory":
            self._conversations[conv.id] = conv
        
        elif self.storage_backend == "redis" and self._redis:
            data = json.dumps(conv.to_dict())
            self._redis.setex(
                f"conv:{conv.id}",
                timedelta(hours=self.ttl_hours),
                data
            )
    
    def _cleanup_user_conversations(self, user_id: str):
        """Remove old conversations if over limit."""
        if user_id not in self._user_index:
            return
        
        conv_ids = self._user_index[user_id]
        
        while len(conv_ids) > self.max_conversations:
            old_id = conv_ids.pop(0)
            
            # Remove from storage
            if self.storage_backend == "memory":
                if old_id in self._conversations:
                    del self._conversations[old_id]
            elif self.storage_backend == "redis" and self._redis:
                self._redis.delete(f"conv:{old_id}")
    
    def export_conversations(
        self,
        user_id: str,
        format: str = "json"
    ) -> str:
        """
        Export user's conversations.
        
        Args:
            user_id: User identifier
            format: Export format ("json" or "text")
            
        Returns:
            Exported data as string
        """
        conversations = self.get_user_conversations(user_id, limit=100)
        
        if format == "json":
            return json.dumps([c.to_dict() for c in conversations], indent=2)
        
        elif format == "text":
            lines = []
            for conv in conversations:
                lines.append(f"=== Conversation {conv.id} ===")
                lines.append(f"Date: {conv.created_at.strftime('%Y-%m-%d %H:%M')}")
                lines.append("")
                
                for msg in conv.messages:
                    role = "You" if msg.role == "user" else "Assistant"
                    lines.append(f"{role}: {msg.content}")
                    lines.append("")
                
                lines.append("")
            
            return "\n".join(lines)
        
        return ""


# Singleton instance
_history_service: Optional[ConversationHistoryService] = None


def get_conversation_history_service(
    storage_backend: str = "memory",
    redis_url: Optional[str] = None
) -> ConversationHistoryService:
    """Get or create the conversation history service singleton."""
    global _history_service
    
    if _history_service is None:
        import os
        redis_url = redis_url or os.getenv("REDIS_URL")
        backend = "redis" if redis_url else storage_backend
        
        _history_service = ConversationHistoryService(
            storage_backend=backend,
            redis_url=redis_url
        )
    
    return _history_service
