"""
User Management
User profile and session management for the Istanbul AI system.
"""

import hashlib
from datetime import datetime
from typing import Dict, Optional
from ..core.models import UserProfile, ConversationContext, UserType, ConversationTone


class UserManager:
    """Manages user profiles and conversation sessions"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.active_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.conversation_contexts: Dict[str, ConversationContext] = {}  # session_id -> context
    
    def get_or_create_user_profile(self, user_id: str) -> UserProfile:
        """Get existing user profile or create new one"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                user_type=UserType.FIRST_TIME_VISITOR,
                preferred_tone=ConversationTone.FRIENDLY
            )
            
            # Update profile completeness
            self._recalculate_profile_completeness(self.user_profiles[user_id])
        
        # Update last interaction
        self.user_profiles[user_id].last_interaction = datetime.now()
        return self.user_profiles[user_id]
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID"""
        timestamp = str(datetime.now().timestamp())
        session_data = f"{user_id}_{timestamp}"
        return hashlib.md5(session_data.encode()).hexdigest()[:12]
    
    def _get_active_session_id(self, user_id: str) -> Optional[str]:
        """Get active session ID for user"""
        if user_id in self.active_sessions:
            session_id = self.active_sessions[user_id]
            if session_id in self.conversation_contexts:
                # Check if session is still active (within last 2 hours)
                context = self.conversation_contexts[session_id]
                time_diff = datetime.now() - context.last_interaction
                if time_diff.total_seconds() < 7200:  # 2 hours
                    return session_id
                else:
                    # Clean up expired session
                    del self.conversation_contexts[session_id]
                    del self.active_sessions[user_id]
        return None
    
    def start_conversation(self, user_id: str) -> str:
        """Start new conversation session"""
        user_profile = self.get_or_create_user_profile(user_id)
        
        # Check for existing active session
        session_id = self._get_active_session_id(user_id)
        if session_id:
            return session_id
        
        # Create new session
        session_id = self._generate_session_id(user_id)
        context = ConversationContext(
            session_id=session_id,
            user_profile=user_profile
        )
        
        self.conversation_contexts[session_id] = context
        self.active_sessions[user_id] = session_id
        
        return session_id
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context for session"""
        return self.conversation_contexts.get(session_id)
    
    def update_user_interests(self, user_id: str, interests: list, travel_style: str = None, 
                            accessibility_needs: str = None) -> bool:
        """Update user interests and preferences"""
        if user_id not in self.user_profiles:
            return False
        
        profile = self.user_profiles[user_id]
        
        if interests:
            profile.interests = interests
        if travel_style:
            profile.travel_style = travel_style
        if accessibility_needs:
            profile.accessibility_needs = accessibility_needs
        
        # Recalculate profile completeness
        self._recalculate_profile_completeness(profile)
        return True
    
    def collect_recommendation_feedback(self, user_id: str, recommendation_id: str, 
                                      rating: float, feedback_text: str = None) -> bool:
        """Collect feedback on recommendations"""
        if user_id not in self.user_profiles:
            return False
        
        profile = self.user_profiles[user_id]
        profile.recommendation_feedback[recommendation_id] = rating
        
        # Update satisfaction metrics
        avg_rating = sum(profile.recommendation_feedback.values()) / len(profile.recommendation_feedback)
        profile.satisfaction_score = avg_rating / 5.0  # Normalize to 0-1
        
        return True
    
    def _recalculate_profile_completeness(self, user_profile: UserProfile):
        """Calculate how complete the user profile is"""
        completeness_factors = {
            'interests': 0.2 if user_profile.interests else 0,
            'travel_style': 0.15 if user_profile.travel_style else 0,
            'dietary_restrictions': 0.1 if user_profile.dietary_restrictions else 0,
            'cuisine_preferences': 0.15 if user_profile.cuisine_preferences else 0,
            'favorite_neighborhoods': 0.1 if user_profile.favorite_neighborhoods else 0,
            'budget_range': 0.1 if user_profile.budget_range != "moderate" else 0.05,  # Default is moderate
            'accessibility_needs': 0.05 if user_profile.accessibility_needs else 0,
            'group_info': 0.1 if user_profile.group_type or user_profile.group_size > 1 else 0,
            'interaction_history': 0.05 if len(user_profile.interaction_history) > 0 else 0
        }
        
        user_profile.profile_completeness = sum(completeness_factors.values())
    
    def clear_user_data(self, user_id: str) -> bool:
        """Clear all user data"""
        success = True
        
        # Remove user profile
        if user_id in self.user_profiles:
            del self.user_profiles[user_id]
        else:
            success = False
        
        # Remove active session
        if user_id in self.active_sessions:
            session_id = self.active_sessions[user_id]
            if session_id in self.conversation_contexts:
                del self.conversation_contexts[session_id]
            del self.active_sessions[user_id]
        
        return success
    
    def show_user_data(self, user_id: str) -> Dict:
        """Show user data summary"""
        if user_id not in self.user_profiles:
            return {}
        
        profile = self.user_profiles[user_id]
        
        return {
            'user_id': profile.user_id,
            'user_type': profile.user_type.value,
            'profile_completeness': f"{profile.profile_completeness:.1%}",
            'interests': profile.interests,
            'travel_style': profile.travel_style,
            'dietary_restrictions': profile.dietary_restrictions,
            'favorite_neighborhoods': profile.favorite_neighborhoods,
            'satisfaction_score': f"{profile.satisfaction_score:.1%}",
            'total_interactions': len(profile.interaction_history),
            'last_interaction': profile.last_interaction.strftime('%Y-%m-%d %H:%M') if profile.last_interaction else None
        }
