"""
Istanbul AI Models
Data classes and enums for the Istanbul Daily Talk AI system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class ConversationTone(Enum):
    """Conversation tone adaptation"""
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    LOCAL_EXPERT = "local_expert"
    TOURIST_GUIDE = "tourist_guide"


class UserType(Enum):
    """User classification"""
    FIRST_TIME_VISITOR = "first_time_visitor"
    REPEAT_VISITOR = "repeat_visitor"
    LOCAL_RESIDENT = "local_resident"
    BUSINESS_TRAVELER = "business_traveler"
    CULTURAL_EXPLORER = "cultural_explorer"


@dataclass
class UserProfile:
    """Advanced user profiling system with deep personalization"""
    user_id: str
    user_type: UserType = UserType.FIRST_TIME_VISITOR
    preferred_tone: ConversationTone = ConversationTone.FRIENDLY
    
    # Basic preferences
    favorite_neighborhoods: List[str] = field(default_factory=list)
    dietary_restrictions: List[str] = field(default_factory=list)
    cuisine_preferences: List[str] = field(default_factory=list)
    budget_range: str = "moderate"  # budget, mid, luxury
    
    # ADVANCED PERSONALIZATION FEATURES
    # Interests and hobbies
    interests: List[str] = field(default_factory=list)  # ['history', 'art', 'food', 'nightlife', 'shopping', 'architecture']
    
    # Travel style and preferences
    travel_style: Optional[str] = None  # 'solo', 'couple', 'family', 'group', 'business'
    pace_preference: str = "moderate"  # 'slow', 'moderate', 'fast'
    adventure_level: str = "moderate"  # 'conservative', 'moderate', 'adventurous'
    cultural_immersion_level: str = "moderate"  # 'tourist', 'moderate', 'local_experience'
    
    # Accessibility and special needs
    accessibility_needs: Optional[str] = None  # 'wheelchair', 'hearing', 'visual', 'mobility', None
    mobility_restrictions: List[str] = field(default_factory=list)  # ['no_stairs', 'short_walks', 'rest_breaks']
    
    # Group dynamics
    group_type: Optional[str] = None  # 'family', 'friends', 'couple', 'business', 'solo'
    group_size: int = 1
    has_children: bool = False
    children_ages: List[int] = field(default_factory=list)
    
    # Time and scheduling preferences
    preferred_visit_times: List[str] = field(default_factory=list)  # ['morning', 'afternoon', 'evening', 'night']
    time_availability: str = "flexible"  # 'limited', 'moderate', 'flexible'
    duration_preference: str = "moderate"  # 'quick', 'moderate', 'extended'
    
    # Behavioral patterns
    visit_frequency: Dict[str, int] = field(default_factory=dict)  # location -> count
    preferred_times: List[str] = field(default_factory=list)  # breakfast, lunch, dinner
    interaction_history: List[Dict] = field(default_factory=list)
    
    # ML-based adaptation metrics
    recommendation_feedback: Dict[str, float] = field(default_factory=dict)  # recommendation_id -> rating
    learning_patterns: Dict[str, Any] = field(default_factory=dict)  # ML-derived patterns
    adaptation_weights: Dict[str, float] = field(default_factory=dict)  # feature importance weights
    
    # Temporal context
    current_location: Optional[str] = None
    gps_location: Optional[Dict[str, float]] = None  # {'lat': 41.0082, 'lng': 28.9784}
    location_accuracy: Optional[float] = None  # GPS accuracy in meters
    location_timestamp: Optional[datetime] = None  # When GPS was last updated
    last_interaction: Optional[datetime] = None
    session_context: Dict[str, Any] = field(default_factory=dict)
    
    # Learning metrics
    satisfaction_score: float = 0.8
    recommendation_success_rate: float = 0.7
    profile_completeness: float = 0.3  # How much of the profile is filled out


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
            'intent': intent,
            'context': dict(self.context_memory)
        }
        self.conversation_history.append(interaction)
        self.last_interaction = datetime.now()
        
        # Keep only recent history (last 10 interactions)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
