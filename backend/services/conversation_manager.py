"""
Conversational Context Manager for Pure LLM Handler

PRIORITY 3.2: Maintain conversation history for follow-up questions.

This module enables:
- Conversation history storage and retrieval
- Follow-up question handling
- Reference resolution (pronouns, "there", "it", etc.)
- Context continuity across multiple turns
- Session management with TTL

Features:
- Redis-backed persistent storage
- Automatic context summarization
- Reference resolution using conversation history
- Configurable history length
- Session expiration management

Architecture:
- ConversationManager: Main orchestration class
- ContextResolver: Resolves references in queries
- SessionManager: Handles session lifecycle
- ContextSummarizer: Creates compact context summaries

Author: AI Istanbul Team
Date: November 2025
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import redis

logger = logging.getLogger(__name__)


class ConversationTurn:
    """
    Represents a single turn in a conversation.
    """
    
    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conversation turn.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            timestamp: When the turn occurred
            metadata: Additional data (signals, context used, etc.)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary."""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        """Create turn from dictionary."""
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class SessionManager:
    """
    Manages conversation session lifecycle.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, session_ttl: int = 3600):
        """
        Initialize session manager.
        
        Args:
            redis_client: Redis client for storage
            session_ttl: Session time-to-live in seconds (default: 1 hour)
        """
        self.redis = redis_client
        self.session_ttl = session_ttl
    
    def create_session(self, session_id: str) -> bool:
        """
        Create a new conversation session.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            True if created, False if already exists
        """
        if not self.redis:
            return True
        
        try:
            key = f"session:{session_id}:created"
            
            # Check if session already exists
            if self.redis.exists(key):
                # Extend TTL
                self.redis.expire(key, self.session_ttl)
                return False
            
            # Create new session
            self.redis.setex(
                key,
                self.session_ttl,
                datetime.now().isoformat()
            )
            
            logger.info(f"📝 Created new session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        if not self.redis:
            return True
        
        try:
            return self.redis.exists(f"session:{session_id}:created")
        except Exception as e:
            logger.error(f"Failed to check session: {e}")
            return False
    
    def extend_session(self, session_id: str):
        """Extend session TTL."""
        if not self.redis:
            return
        
        try:
            # Extend all session keys
            pattern = f"session:{session_id}:*"
            keys = self.redis.keys(pattern)
            
            for key in keys:
                self.redis.expire(key, self.session_ttl)
            
            logger.debug(f"⏱️ Extended session TTL: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to extend session: {e}")
    
    def delete_session(self, session_id: str):
        """Delete session and all associated data."""
        if not self.redis:
            return
        
        try:
            pattern = f"session:{session_id}:*"
            keys = self.redis.keys(pattern)
            
            if keys:
                self.redis.delete(*keys)
                logger.info(f"🗑️ Deleted session: {session_id}")
                
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")


class ContextResolver:
    """
    Resolves references in queries using conversation context.
    
    Handles:
    - Pronouns (it, them, they, etc.)
    - Location references (there, here)
    - Implicit references (what about, how about)
    - Follow-up questions
    """
    
    def __init__(self):
        """Initialize context resolver."""
        # Common reference patterns
        self.pronouns = {
            'it', 'its', 'them', 'they', 'their', 'theirs',
            'that', 'those', 'this', 'these'
        }
        
        self.location_refs = {
            'there', 'here', 'that place', 'this place',
            'that location', 'this location'
        }
        
        self.implicit_refs = {
            'what about', 'how about', 'tell me about',
            'more about', 'details on', 'info on'
        }
    
    def needs_resolution(self, query: str) -> bool:
        """
        Check if query needs reference resolution.
        
        Args:
            query: User query
            
        Returns:
            True if query contains references
        """
        query_lower = query.lower()
        
        # Check for pronouns
        query_words = set(query_lower.split())
        if query_words & self.pronouns:
            return True
        
        # Check for location references
        if any(ref in query_lower for ref in self.location_refs):
            return True
        
        # Check for implicit references
        if any(ref in query_lower for ref in self.implicit_refs):
            return True
        
        # Check if query is very short (likely follow-up)
        if len(query.split()) <= 2:
            return True
        
        return False
    
    def resolve_references(
        self,
        query: str,
        history: List[ConversationTurn]
    ) -> Tuple[str, bool]:
        """
        Resolve references in query using conversation history.
        
        Args:
            query: User query with potential references
            history: Conversation history (recent turns)
            
        Returns:
            Tuple of (resolved_query, was_resolved)
        """
        if not history or not self.needs_resolution(query):
            return query, False
        
        # Get last few turns for context
        recent_turns = history[-3:] if len(history) >= 3 else history
        
        # Extract entities from recent turns
        entities = self._extract_entities(recent_turns)
        
        if not entities:
            return query, False
        
        # Try to resolve references
        resolved_query = self._resolve(query, entities)
        
        if resolved_query != query:
            logger.info(f"🔗 Resolved reference: '{query}' → '{resolved_query}'")
            return resolved_query, True
        
        return query, False
    
    def _extract_entities(self, turns: List[ConversationTurn]) -> Dict[str, str]:
        """
        Extract entities (places, things) from conversation turns.
        
        Args:
            turns: Recent conversation turns
            
        Returns:
            Dict of entity_type -> entity_value
        """
        entities = {}
        
        for turn in reversed(turns):
            content = turn.content.lower()
            
            # Extract place names (simple heuristic)
            # Look for common Istanbul landmarks
            landmarks = [
                'hagia sophia', 'blue mosque', 'topkapi palace',
                'grand bazaar', 'basilica cistern', 'galata tower',
                'dolmabahce palace', 'bosphorus', 'taksim',
                'sultanahmet', 'beyoglu', 'kadikoy'
            ]
            
            for landmark in landmarks:
                if landmark in content:
                    entities['last_place'] = landmark
                    break
            
            # Extract from metadata if available
            if turn.metadata.get('detected_signals'):
                signals = turn.metadata['detected_signals']
                
                # If user asked about restaurants, remember that
                if signals.get('likely_restaurant'):
                    entities['last_topic'] = 'restaurant'
                
                # If user asked about attractions
                if signals.get('likely_attraction'):
                    entities['last_topic'] = 'attraction'
        
        return entities
    
    def _resolve(self, query: str, entities: Dict[str, str]) -> str:
        """
        Resolve references using extracted entities.
        
        Args:
            query: Original query
            entities: Extracted entities
            
        Returns:
            Resolved query
        """
        query_lower = query.lower()
        resolved = query
        
        # Resolve location references
        last_place = entities.get('last_place')
        if last_place:
            # "How do I get there?" → "How do I get to Hagia Sophia?"
            if 'there' in query_lower:
                resolved = resolved.replace('there', last_place)
                resolved = resolved.replace('There', last_place.title())
            
            # "How far is it?" → "How far is Hagia Sophia?"
            if 'how far is it' in query_lower:
                resolved = f"How far is {last_place}?"
            
            # Handle other " it " or " it?" patterns
            elif ' it?' in query_lower or ' it ' in query_lower:
                # Replace "it" with place name
                resolved = resolved.replace(' it?', f' {last_place}?')
                resolved = resolved.replace(' it ', f' {last_place} ')
                resolved = resolved.replace(' it.', f' {last_place}.')
            
            # "Take me there" → "Take me to Hagia Sophia"
            if 'take me there' in query_lower:
                resolved = f"Take me to {last_place}"
        
        # Resolve implicit references
        last_topic = entities.get('last_topic')
        if last_topic:
            # "What about cheap ones?" → "What about cheap restaurants?"
            if 'what about' in query_lower or 'how about' in query_lower:
                if 'cheap' in query_lower or 'affordable' in query_lower:
                    resolved = f"{query} {last_topic}s"
        
        return resolved


class ContextSummarizer:
    """
    Creates compact context summaries from conversation history.
    """
    
    def __init__(self):
        """Initialize context summarizer."""
        pass
    
    def summarize_history(
        self,
        history: List[ConversationTurn],
        max_length: int = 200
    ) -> str:
        """
        Create compact summary of conversation history.
        
        Args:
            history: Conversation turns
            max_length: Maximum summary length
            
        Returns:
            Compact context summary
        """
        if not history:
            return ""
        
        # Extract key information from recent turns
        topics = []
        places = []
        intents = []
        
        for turn in history[-5:]:  # Last 5 turns
            if turn.role == 'user':
                content = turn.content.lower()
                
                # Extract topics
                if 'restaurant' in content or 'food' in content or 'eat' in content:
                    topics.append('restaurants')
                elif 'museum' in content or 'attraction' in content:
                    topics.append('attractions')
                elif 'hotel' in content or 'accommodation' in content:
                    topics.append('accommodation')
                
                # Extract places
                if 'sultanahmet' in content:
                    places.append('Sultanahmet')
                elif 'taksim' in content:
                    places.append('Taksim')
                
                # Extract intents from metadata
                if turn.metadata.get('detected_signals'):
                    signals = turn.metadata['detected_signals']
                    if signals.get('needs_map'):
                        intents.append('directions')
                    if signals.get('needs_weather'):
                        intents.append('weather')
        
        # Build summary
        summary_parts = []
        
        if topics:
            unique_topics = list(set(topics))
            summary_parts.append(f"discussing {', '.join(unique_topics)}")
        
        if places:
            unique_places = list(set(places))
            summary_parts.append(f"in {', '.join(unique_places)}")
        
        if intents:
            unique_intents = list(set(intents))
            summary_parts.append(f"requesting {', '.join(unique_intents)}")
        
        if summary_parts:
            summary = f"Context: User was {' '.join(summary_parts)}."
        else:
            summary = "Context: New conversation."
        
        # Trim if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary


class ConversationManager:
    """
    Main conversation management class.
    
    Orchestrates:
    - History storage and retrieval
    - Session management
    - Reference resolution
    - Context summarization
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        max_history_turns: int = 10,
        session_ttl: int = 3600
    ):
        """
        Initialize conversation manager.
        
        Args:
            redis_client: Redis client for storage
            max_history_turns: Maximum turns to keep in history
            session_ttl: Session time-to-live in seconds (default: 1 hour)
        """
        self.redis = redis_client
        self.max_history_turns = max_history_turns
        self.session_ttl = session_ttl
        
        self.session_manager = SessionManager(redis_client, session_ttl)
        self.context_resolver = ContextResolver()
        self.context_summarizer = ContextSummarizer()
        
        # In-memory cache for active sessions
        self.history_cache = {}  # session_id -> deque of turns
        
        logger.info("💬 ConversationManager initialized")
        logger.info(f"   Max history turns: {max_history_turns}")
        logger.info(f"   Session TTL: {session_ttl}s")
    
    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add conversation turn to history.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Additional data
        """
        # Create session if doesn't exist
        self.session_manager.create_session(session_id)
        
        # Create turn
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata
        )
        
        # Add to cache
        if session_id not in self.history_cache:
            self.history_cache[session_id] = deque(maxlen=self.max_history_turns)
        
        self.history_cache[session_id].append(turn)
        
        # Store in Redis
        if self.redis:
            try:
                key = f"session:{session_id}:history"
                
                # Get existing history
                history_json = self.redis.get(key)
                if history_json:
                    history_data = json.loads(history_json.decode('utf-8'))
                else:
                    history_data = []
                
                # Add new turn
                history_data.append(turn.to_dict())
                
                # Keep only max_history_turns
                if len(history_data) > self.max_history_turns:
                    history_data = history_data[-self.max_history_turns:]
                
                # Save back to Redis
                self.redis.setex(
                    key,
                    self.session_ttl,
                    json.dumps(history_data)
                )
                
                logger.debug(f"💾 Saved turn for session {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to save turn to Redis: {e}")
        
        # Extend session TTL
        self.session_manager.extend_session(session_id)
    
    def get_conversation_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Retrieve conversation history for session.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum turns to retrieve (default: all)
            
        Returns:
            List of conversation turns (oldest to newest)
        """
        # Check cache first
        if session_id in self.history_cache:
            history = list(self.history_cache[session_id])
            if max_turns:
                history = history[-max_turns:]
            return history
        
        # Load from Redis
        if self.redis:
            try:
                key = f"session:{session_id}:history"
                history_json = self.redis.get(key)
                
                if history_json:
                    history_data = json.loads(history_json.decode('utf-8'))
                    turns = [ConversationTurn.from_dict(turn_data) for turn_data in history_data]
                    
                    # Update cache
                    self.history_cache[session_id] = deque(turns, maxlen=self.max_history_turns)
                    
                    if max_turns:
                        turns = turns[-max_turns:]
                    
                    logger.debug(f"📚 Retrieved {len(turns)} turns for session {session_id}")
                    return turns
                
            except Exception as e:
                logger.error(f"Failed to load history from Redis: {e}")
        
        return []
    
    def resolve_query(
        self,
        session_id: str,
        query: str
    ) -> Tuple[str, bool]:
        """
        Resolve references in query using conversation history.
        
        Args:
            session_id: Session identifier
            query: User query
            
        Returns:
            Tuple of (resolved_query, was_resolved)
        """
        history = self.get_conversation_history(session_id, max_turns=5)
        return self.context_resolver.resolve_references(query, history)
    
    def get_context_summary(
        self,
        session_id: str,
        max_length: int = 200
    ) -> str:
        """
        Get compact context summary for session.
        
        Args:
            session_id: Session identifier
            max_length: Maximum summary length
            
        Returns:
            Context summary string
        """
        history = self.get_conversation_history(session_id, max_turns=5)
        return self.context_summarizer.summarize_history(history, max_length)
    
    def clear_session(self, session_id: str):
        """
        Clear session history.
        
        Args:
            session_id: Session identifier
        """
        # Remove from cache
        if session_id in self.history_cache:
            del self.history_cache[session_id]
        
        # Remove from Redis
        self.session_manager.delete_session(session_id)
        
        logger.info(f"🗑️ Cleared session: {session_id}")
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of session IDs
        """
        active_sessions = []
        
        # From cache
        active_sessions.extend(self.history_cache.keys())
        
        # From Redis
        if self.redis:
            try:
                pattern = "session:*:created"
                keys = self.redis.keys(pattern)
                
                for key in keys:
                    session_id = key.decode('utf-8').split(':')[1]
                    if session_id not in active_sessions:
                        active_sessions.append(session_id)
                        
            except Exception as e:
                logger.error(f"Failed to get active sessions: {e}")
        
        return active_sessions
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get conversation manager statistics.
        
        Returns:
            Statistics dict
        """
        active_sessions = self.get_active_sessions()
        
        total_turns = 0
        for session_id in active_sessions:
            history = self.get_conversation_history(session_id)
            total_turns += len(history)
        
        return {
            'active_sessions': len(active_sessions),
            'cached_sessions': len(self.history_cache),
            'total_turns': total_turns,
            'avg_turns_per_session': total_turns / len(active_sessions) if active_sessions else 0,
            'max_history_turns': self.max_history_turns,
            'session_ttl': self.session_ttl
        }
