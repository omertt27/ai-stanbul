"""
User Profile and User Type classes for Istanbul AI System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from ..utils.constants import UserType, ConversationTone


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

    def update_location(self, location: str, gps_coords: Optional[Dict[str, float]] = None, accuracy: Optional[float] = None):
        """Update user's current location"""
        self.current_location = location
        if gps_coords:
            self.gps_location = gps_coords
            self.location_accuracy = accuracy
            self.location_timestamp = datetime.now()

    def add_interest(self, interest: str):
        """Add a new interest to the profile"""
        if interest not in self.interests:
            self.interests.append(interest)

    def update_satisfaction(self, rating: float):
        """Update satisfaction score with running average"""
        self.satisfaction_score = (self.satisfaction_score + rating) / 2

    def get_preference_summary(self) -> Dict:
        """Get a summary of user preferences for ML models"""
        return {
            'user_type': self.user_type.value,
            'budget_range': self.budget_range,
            'interests': self.interests,
            'travel_style': self.travel_style,
            'pace_preference': self.pace_preference,
            'cultural_immersion_level': self.cultural_immersion_level,
            'favorite_neighborhoods': self.favorite_neighborhoods,
            'cuisine_preferences': self.cuisine_preferences,
            'dietary_restrictions': self.dietary_restrictions
        }
