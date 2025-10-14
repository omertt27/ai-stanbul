"""
Conversation Context classes for Istanbul AI System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from .user_profile import UserProfile


@dataclass
class ConversationContext:
    """Multi-turn conversation context with temporal awareness"""
    session_id: str
    user_profile: UserProfile
    conversation_history: List[Dict] = field(default_factory=list)
    current_topic: Optional[str] = None
    pending_questions: List[str] = field(default_factory=list)
    context_memory: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal context
    session_start: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    
    def add_interaction(self, user_input: str, system_response: str, intent: str):
        """Add interaction to conversation history"""
        interaction = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'system_response': system_response,
            'detected_intent': intent,
            'context_at_time': self.context_memory.copy()
        }
        self.conversation_history.append(interaction)
        self.last_interaction = datetime.now()

    def get_recent_topics(self, limit: int = 5) -> List[str]:
        """Get recent conversation topics"""
        recent_interactions = self.conversation_history[-limit:]
        topics = []
        for interaction in recent_interactions:
            if interaction.get('detected_intent'):
                topics.append(interaction['detected_intent'])
        return topics

    def get_recent_interactions(self, limit: int = 5) -> List[Dict]:
        """Get recent conversation interactions"""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def set_context(self, key: str, value: Any):
        """Set context memory"""
        self.context_memory[key] = value

    def get_context(self, key: str, default=None):
        """Get context memory"""
        return self.context_memory.get(key, default)

    def clear_context(self):
        """Clear context memory"""
        self.context_memory.clear()

    def is_session_active(self, timeout_minutes: int = 30) -> bool:
        """Check if session is still active"""
        time_diff = datetime.now() - self.last_interaction
        return time_diff.total_seconds() < (timeout_minutes * 60)
