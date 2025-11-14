"""
Simple Conversational Context System - LLM-Based Approach

PRIORITY 3.2: Conversational context using LLM's natural language understanding.

Instead of complex keyword matching, we leverage the LLM's ability to understand
context by simply including conversation history in the prompt. This is:
- More accurate (LLM understands nuance)
- Simpler (no complex pattern matching)
- More maintainable (no brittle rules)
- More flexible (handles any language)

Author: AI Istanbul Team
Date: November 14, 2025
"""

import logging
import json
import redis
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class SimpleConversationManager:
    """
    Simple conversation manager that stores history and formats it for LLM context.
    
    Philosophy: Let the LLM do what it does best - understand natural language context.
    We just need to store the conversation and present it properly.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        max_history_turns: int = 5,
        session_ttl: int = 3600
    ):
        """
        Initialize conversation manager.
        
        Args:
            redis_client: Redis client for persistent storage
            max_history_turns: Maximum conversation turns to keep
            session_ttl: Session time-to-live in seconds (1 hour default)
        """
        self.redis = redis_client
        self.max_history_turns = max_history_turns
        self.session_ttl = session_ttl
        
        # In-memory cache for active conversations
        self.conversation_cache = {}
        
        logger.info(f"💬 Simple Conversation Manager initialized (max {max_history_turns} turns)")
    
    def add_turn(
        self,
        session_id: str,
        role: str,  # 'user' or 'assistant'
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a conversation turn.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (signals, location, etc.)
        """
        turn = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Add to cache
        if session_id not in self.conversation_cache:
            self.conversation_cache[session_id] = deque(maxlen=self.max_history_turns)
        
        self.conversation_cache[session_id].append(turn)
        
        # Store in Redis
        if self.redis:
            try:
                key = f"conversation:{session_id}"
                
                # Get existing history
                existing = self.redis.get(key)
                if existing:
                    history = json.loads(existing.decode('utf-8'))
                else:
                    history = []
                
                # Add new turn
                history.append(turn)
                
                # Keep only last N turns
                history = history[-self.max_history_turns:]
                
                # Save back to Redis
                self.redis.setex(
                    key,
                    self.session_ttl,
                    json.dumps(history)
                )
                
                logger.debug(f"💬 Added {role} turn to session {session_id}")
                
            except Exception as e:
                logger.error(f"Failed to store conversation turn: {e}")
    
    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_turns: Max turns to return (None = use default)
            
        Returns:
            List of conversation turns
        """
        max_turns = max_turns or self.max_history_turns
        
        # Check cache first
        if session_id in self.conversation_cache:
            history = list(self.conversation_cache[session_id])
            return history[-max_turns:] if len(history) > max_turns else history
        
        # Load from Redis
        if self.redis:
            try:
                key = f"conversation:{session_id}"
                data = self.redis.get(key)
                
                if data:
                    history = json.loads(data.decode('utf-8'))
                    
                    # Update cache
                    self.conversation_cache[session_id] = deque(
                        history,
                        maxlen=self.max_history_turns
                    )
                    
                    return history[-max_turns:] if len(history) > max_turns else history
                    
            except Exception as e:
                logger.error(f"Failed to retrieve conversation history: {e}")
        
        return []
    
    def format_context_for_llm(
        self,
        session_id: str,
        max_turns: int = 3,
        include_metadata: bool = False
    ) -> str:
        """
        Format conversation history for LLM prompt.
        
        This is the key method - it presents the conversation in a way
        that lets the LLM naturally understand follow-up questions.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum turns to include
            include_metadata: Include metadata in context
            
        Returns:
            Formatted context string ready for LLM prompt
        """
        history = self.get_history(session_id, max_turns)
        
        if not history:
            return ""
        
        # Build context string
        context_lines = ["Previous conversation:"]
        
        for turn in history:
            role = turn['role'].capitalize()
            content = turn['content']
            
            if include_metadata and turn.get('metadata'):
                metadata = turn['metadata']
                if metadata:
                    context_lines.append(f"{role}: {content} [Context: {metadata}]")
                else:
                    context_lines.append(f"{role}: {content}")
            else:
                context_lines.append(f"{role}: {content}")
        
        context_lines.append("")  # Empty line
        context_lines.append(
            "The user's new message below may reference the above conversation. "
            "Please understand it in that context."
        )
        
        return "\n".join(context_lines)
    
    def has_context(self, session_id: str) -> bool:
        """
        Check if session has conversation history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session has history
        """
        history = self.get_history(session_id, max_turns=1)
        return len(history) > 0
    
    def clear_session(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        # Clear cache
        if session_id in self.conversation_cache:
            del self.conversation_cache[session_id]
        
        # Clear Redis
        if self.redis:
            try:
                key = f"conversation:{session_id}"
                self.redis.delete(key)
                logger.info(f"🗑️ Cleared conversation: {session_id}")
            except Exception as e:
                logger.error(f"Failed to clear conversation: {e}")
    
    def get_last_topic(self, session_id: str) -> Optional[str]:
        """
        Get the last topic/entity discussed in conversation.
        
        This is useful for analytics and understanding conversation flow.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Last topic or None
        """
        history = self.get_history(session_id, max_turns=3)
        
        if not history:
            return None
        
        # Look for location/place names in recent messages
        # Common Istanbul locations
        common_places = [
            'hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar',
            'sultanahmet', 'taksim', 'galata tower', 'bosphorus',
            'dolmabahce palace', 'basilica cistern', 'spice bazaar',
            'ortakoy', 'kadikoy', 'besiktas', 'eminonu', 'karakoy'
        ]
        
        # Search from most recent to oldest
        for turn in reversed(history):
            content_lower = turn['content'].lower()
            
            for place in common_places:
                if place in content_lower:
                    return place.title()
        
        return None
    
    def get_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Statistics dict
        """
        history = self.get_history(session_id)
        
        if not history:
            return {
                'turn_count': 0,
                'has_context': False
            }
        
        user_turns = [t for t in history if t['role'] == 'user']
        assistant_turns = [t for t in history if t['role'] == 'assistant']
        
        return {
            'turn_count': len(history),
            'user_turns': len(user_turns),
            'assistant_turns': len(assistant_turns),
            'has_context': True,
            'last_topic': self.get_last_topic(session_id),
            'first_turn_time': history[0]['timestamp'] if history else None,
            'last_turn_time': history[-1]['timestamp'] if history else None
        }


# Example usage:
"""
# Initialize
conversation_manager = SimpleConversationManager(
    redis_client=redis_client,
    max_history_turns=5,
    session_ttl=3600
)

# User asks first question
session_id = "user123_session_abc"
query1 = "Tell me about Hagia Sophia"

# Process query... get response...
response1 = "Hagia Sophia is a historic..."

# Store conversation
conversation_manager.add_turn(session_id, 'user', query1)
conversation_manager.add_turn(session_id, 'assistant', response1)

# User asks follow-up (with reference)
query2 = "How do I get there?"  # "there" = Hagia Sophia

# Get conversation context for LLM
context = conversation_manager.format_context_for_llm(session_id, max_turns=3)

# Build prompt with context
full_prompt = f'''
{context}

User: {query2}

Please respond to the user's question, understanding "there" refers to Hagia Sophia from the previous conversation.
'''

# LLM will naturally understand that "there" = "Hagia Sophia"
# No complex keyword matching needed!
"""
