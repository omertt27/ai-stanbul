"""
User Profiling and Preference Memory System for AI Istanbul
Lightweight personalization that learns user preferences and improves responses
Targets $2k/month savings through better personalization and reduced query repetition
"""

import json
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class UserPreference:
    """Individual user preference with confidence and context"""
    preference_type: str  # 'interest', 'diet', 'budget', 'transport_mode', etc.
    value: str           # 'historical_sites', 'vegetarian', 'budget', 'metro'
    confidence: float    # 0.0 to 1.0
    source: str         # 'explicit', 'inferred', 'behavioral'
    created_at: datetime = field(default_factory=datetime.now)
    last_confirmed: datetime = field(default_factory=datetime.now)
    times_confirmed: int = 1

@dataclass
class UserProfile:
    """Complete user profile with preferences and behavioral patterns"""
    user_id: str
    preferences: Dict[str, List[UserPreference]] = field(default_factory=dict)
    
    # Behavioral patterns
    query_history: List[Dict] = field(default_factory=list)
    visited_locations: Set[str] = field(default_factory=set)
    preferred_times: List[str] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Profile metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    total_interactions: int = 0
    satisfaction_scores: List[float] = field(default_factory=list)
    
    def get_preference_strength(self, preference_type: str, value: str) -> float:
        """Get strength of a specific preference"""
        if preference_type not in self.preferences:
            return 0.0
        
        for pref in self.preferences[preference_type]:
            if pref.value.lower() == value.lower():
                return pref.confidence
        
        return 0.0
    
    def get_top_preferences(self, preference_type: str, limit: int = 3) -> List[UserPreference]:
        """Get top preferences of a specific type"""
        if preference_type not in self.preferences:
            return []
        
        prefs = self.preferences[preference_type]
        return sorted(prefs, key=lambda p: p.confidence, reverse=True)[:limit]
    
    def get_avg_satisfaction(self) -> float:
        """Get average user satisfaction"""
        if not self.satisfaction_scores:
            return 0.7  # Default neutral
        return sum(self.satisfaction_scores) / len(self.satisfaction_scores)

class UserProfilingSystem:
    """
    Advanced user profiling system that learns from interactions
    Provides personalized recommendations and context-aware responses
    """
    
    def __init__(self, db_path: str = "user_profiles.db"):
        self.db_path = Path(db_path)
        self.profiles: Dict[str, UserProfile] = {}
        
        # Preference extraction patterns
        self.preference_patterns = {
            'interests': {
                'historical': ['history', 'historical', 'ancient', 'byzantine', 'ottoman', 'museum', 'palace'],
                'religious': ['mosque', 'church', 'religious', 'spiritual', 'prayer', 'faith'],
                'art': ['art', 'gallery', 'painting', 'sculpture', 'artist', 'exhibition'],
                'architecture': ['architecture', 'building', 'design', 'structure', 'dome', 'minaret'],
                'food': ['food', 'restaurant', 'cuisine', 'eat', 'dining', 'breakfast', 'lunch', 'dinner'],
                'shopping': ['shopping', 'market', 'bazaar', 'souvenir', 'buy', 'store'],
                'nightlife': ['nightlife', 'bar', 'club', 'night', 'evening', 'drink'],
                'nature': ['park', 'garden', 'outdoor', 'nature', 'walk', 'scenic'],
                'culture': ['culture', 'traditional', 'local', 'authentic', 'customs']
            },
            'budget': {
                'budget': ['budget', 'cheap', 'affordable', 'inexpensive', 'low cost'],
                'mid_range': ['moderate', 'reasonable', 'mid-range', 'average'],
                'luxury': ['luxury', 'expensive', 'high-end', 'premium', 'exclusive']
            },
            'transport_mode': {
                'walking': ['walk', 'walking', 'on foot', 'pedestrian'],
                'metro': ['metro', 'subway', 'underground', 'M1', 'M2'],
                'bus': ['bus', 'public transport', 'city bus'],
                'taxi': ['taxi', 'cab', 'uber', 'ride'],
                'ferry': ['ferry', 'boat', 'bosphorus cruise', 'sea']
            },
            'dining_preferences': {
                'vegetarian': ['vegetarian', 'veggie', 'no meat', 'plant-based'],
                'halal': ['halal', 'islamic', 'muslim food'],
                'seafood': ['seafood', 'fish', 'shrimp', 'calamari'],
                'traditional': ['traditional', 'authentic', 'local', 'turkish'],
                'international': ['international', 'foreign', 'western', 'asian']
            },
            'travel_style': {
                'solo': ['solo', 'alone', 'by myself', 'single'],
                'couple': ['couple', 'romantic', 'two of us', 'partner'],
                'family': ['family', 'kids', 'children', 'parents'],
                'group': ['group', 'friends', 'together', 'us']
            }
        }
        
        # Initialize database
        self._init_database()
        self._load_profiles()
    
    def _init_database(self):
        """Initialize SQLite database for user profiles"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User profiles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_data TEXT,
                    created_at TIMESTAMP,
                    last_updated TIMESTAMP,
                    total_interactions INTEGER,
                    avg_satisfaction REAL
                )
            ''')
            
            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    preference_type TEXT,
                    preference_value TEXT,
                    confidence REAL,
                    source TEXT,
                    created_at TIMESTAMP,
                    last_confirmed TIMESTAMP,
                    times_confirmed INTEGER,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            # Query history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    query TEXT,
                    response_type TEXT,
                    satisfaction REAL,
                    timestamp TIMESTAMP,
                    extracted_preferences TEXT,
                    FOREIGN KEY (user_id) REFERENCES user_profiles (user_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ User profiling database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error initializing database: {e}")
    
    def _load_profiles(self):
        """Load existing user profiles from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT user_id, profile_data FROM user_profiles')
            rows = cursor.fetchall()
            
            for user_id, profile_data in rows:
                try:
                    profile_dict = json.loads(profile_data)
                    profile = self._dict_to_profile(profile_dict)
                    self.profiles[user_id] = profile
                except Exception as e:
                    logger.warning(f"Could not load profile for {user_id}: {e}")
            
            conn.close()
            logger.info(f"✅ Loaded {len(self.profiles)} user profiles")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load profiles: {e}")
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get existing user profile"""
        return self.profiles.get(user_id)
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one"""
        if user_id not in self.profiles:
            self.profiles[user_id] = UserProfile(user_id=user_id)
            logger.info(f"📝 Created new profile for user {user_id}")
        
        return self.profiles[user_id]
    
    def analyze_query_for_preferences(self, query: str, user_id: str, 
                                    response_metadata: Dict = None) -> Dict[str, List[UserPreference]]:
        """Analyze query to extract user preferences"""
        extracted_preferences = {}
        query_lower = query.lower()
        
        for pref_type, patterns in self.preference_patterns.items():
            for pref_value, keywords in patterns.items():
                # Check if any keyword matches
                matches = sum(1 for keyword in keywords if keyword in query_lower)
                
                if matches > 0:
                    # Calculate confidence based on number of matches and keyword specificity
                    confidence = min(0.9, matches * 0.3 + 0.2)
                    
                    preference = UserPreference(
                        preference_type=pref_type,
                        value=pref_value,
                        confidence=confidence,
                        source='inferred'
                    )
                    
                    if pref_type not in extracted_preferences:
                        extracted_preferences[pref_type] = []
                    extracted_preferences[pref_type].append(preference)
        
        # Extract location preferences
        location_keywords = [
            'sultanahmet', 'beyoglu', 'galata', 'karakoy', 'besiktas', 
            'kadikoy', 'uskudar', 'taksim', 'eminonu'
        ]
        
        for location in location_keywords:
            if location in query_lower:
                preference = UserPreference(
                    preference_type='preferred_areas',
                    value=location,
                    confidence=0.7,
                    source='inferred'
                )
                
                if 'preferred_areas' not in extracted_preferences:
                    extracted_preferences['preferred_areas'] = []
                extracted_preferences['preferred_areas'].append(preference)
        
        return extracted_preferences
    
    def update_user_profile(self, user_id: str, query: str, response: str,
                           satisfaction: float = None, response_metadata: Dict = None):
        """Update user profile based on interaction"""
        try:
            profile = self.get_or_create_profile(user_id)
            
            # Extract preferences from query
            extracted_prefs = self.analyze_query_for_preferences(query, user_id, response_metadata)
            
            # Update preferences
            for pref_type, new_prefs in extracted_prefs.items():
                if pref_type not in profile.preferences:
                    profile.preferences[pref_type] = []
                
                existing_prefs = {p.value: p for p in profile.preferences[pref_type]}
                
                for new_pref in new_prefs:
                    if new_pref.value in existing_prefs:
                        # Strengthen existing preference
                        existing_pref = existing_prefs[new_pref.value]
                        existing_pref.confidence = min(1.0, existing_pref.confidence + 0.1)
                        existing_pref.times_confirmed += 1
                        existing_pref.last_confirmed = datetime.now()
                    else:
                        # Add new preference
                        profile.preferences[pref_type].append(new_pref)
            
            # Update interaction data
            profile.query_history.append({
                'query': query,
                'response_type': response_metadata.get('source', 'unknown') if response_metadata else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'satisfaction': satisfaction
            })
            
            profile.total_interactions += 1
            profile.last_updated = datetime.now()
            
            if satisfaction is not None:
                profile.satisfaction_scores.append(satisfaction)
                # Keep only recent satisfaction scores
                if len(profile.satisfaction_scores) > 50:
                    profile.satisfaction_scores = profile.satisfaction_scores[-50:]
            
            # Extract visited locations from response
            if response_metadata and 'locations' in response_metadata:
                for location in response_metadata['locations']:
                    profile.visited_locations.add(location)
            
            # Save to database
            self._save_profile(profile)
            
            logger.info(f"📊 Updated profile for user {user_id} (total interactions: {profile.total_interactions})")
            
        except Exception as e:
            logger.error(f"❌ Error updating user profile: {e}")
    
    def get_personalized_context(self, user_id: str, query: str) -> Dict[str, Any]:
        """Get personalized context for query processing"""
        if user_id not in self.profiles:
            return {'is_new_user': True}
        
        profile = self.profiles[user_id]
        context = {
            'is_new_user': False,
            'user_preferences': {},
            'behavioral_hints': {},
            'personalization_strength': min(1.0, profile.total_interactions / 10.0)  # 0-1 based on interaction count
        }
        
        # Extract top preferences for each type
        for pref_type, prefs in profile.preferences.items():
            top_prefs = profile.get_top_preferences(pref_type, limit=3)
            if top_prefs:
                context['user_preferences'][pref_type] = [
                    {'value': p.value, 'confidence': p.confidence} for p in top_prefs
                ]
        
        # Behavioral hints
        context['behavioral_hints'] = {
            'avg_satisfaction': profile.get_avg_satisfaction(),
            'total_interactions': profile.total_interactions,
            'recent_areas': list(profile.visited_locations)[-5:] if profile.visited_locations else [],
            'is_frequent_user': profile.total_interactions > 5
        }
        
        # Query-specific personalization
        query_lower = query.lower()
        
        # Suggest related preferences
        if 'restaurant' in query_lower and 'dining_preferences' in context['user_preferences']:
            context['suggested_dietary'] = context['user_preferences']['dining_preferences']
        
        if 'get to' in query_lower and 'transport_mode' in context['user_preferences']:
            context['preferred_transport'] = context['user_preferences']['transport_mode']
        
        return context
    
    def get_preference_based_suggestions(self, user_id: str, query_type: str) -> List[str]:
        """Get suggestions based on user preferences"""
        if user_id not in self.profiles:
            return []
        
        profile = self.profiles[user_id]
        suggestions = []
        
        # Interest-based suggestions
        interests = profile.get_top_preferences('interests', limit=2)
        for interest in interests:
            if query_type == 'exploration':
                suggestions.append(f"Based on your interest in {interest.value}, you might enjoy...")
            elif query_type == 'food' and interest.value == 'food':
                suggestions.append("Since you're a food enthusiast, I recommend...")
        
        # Budget-based suggestions
        budget_prefs = profile.get_top_preferences('budget', limit=1)
        if budget_prefs:
            budget = budget_prefs[0].value
            if query_type in ['food', 'shopping', 'exploration']:
                suggestions.append(f"For {budget} options, consider...")
        
        # Area-based suggestions
        area_prefs = profile.get_top_preferences('preferred_areas', limit=2)
        for area in area_prefs:
            if query_type in ['food', 'exploration', 'shopping']:
                suggestions.append(f"Since you've shown interest in {area.value.title()}, check out...")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _save_profile(self, profile: UserProfile):
        """Save profile to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            profile_data = self._profile_to_dict(profile)
            profile_json = json.dumps(profile_data)
            
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (user_id, profile_data, created_at, last_updated, total_interactions, avg_satisfaction)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                profile.user_id,
                profile_json,
                profile.created_at.isoformat(),
                profile.last_updated.isoformat(),
                profile.total_interactions,
                profile.get_avg_satisfaction()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error saving profile: {e}")
    
    def _profile_to_dict(self, profile: UserProfile) -> Dict:
        """Convert profile to dictionary for storage"""
        return {
            'user_id': profile.user_id,
            'preferences': {
                pref_type: [
                    {
                        'preference_type': p.preference_type,
                        'value': p.value,
                        'confidence': p.confidence,
                        'source': p.source,
                        'created_at': p.created_at.isoformat(),
                        'last_confirmed': p.last_confirmed.isoformat(),
                        'times_confirmed': p.times_confirmed
                    } for p in prefs
                ] for pref_type, prefs in profile.preferences.items()
            },
            'query_history': profile.query_history[-20:],  # Keep only recent history
            'visited_locations': list(profile.visited_locations),
            'preferred_times': profile.preferred_times,
            'interaction_patterns': profile.interaction_patterns,
            'created_at': profile.created_at.isoformat(),
            'last_updated': profile.last_updated.isoformat(),
            'total_interactions': profile.total_interactions,
            'satisfaction_scores': profile.satisfaction_scores[-20:]  # Keep recent scores
        }
    
    def _dict_to_profile(self, data: Dict) -> UserProfile:
        """Convert dictionary to profile object"""
        profile = UserProfile(user_id=data['user_id'])
        
        # Restore preferences
        for pref_type, prefs_data in data.get('preferences', {}).items():
            profile.preferences[pref_type] = []
            for pref_data in prefs_data:
                pref = UserPreference(
                    preference_type=pref_data['preference_type'],
                    value=pref_data['value'],
                    confidence=pref_data['confidence'],
                    source=pref_data['source'],
                    created_at=datetime.fromisoformat(pref_data['created_at']),
                    last_confirmed=datetime.fromisoformat(pref_data['last_confirmed']),
                    times_confirmed=pref_data['times_confirmed']
                )
                profile.preferences[pref_type].append(pref)
        
        # Restore other data
        profile.query_history = data.get('query_history', [])
        profile.visited_locations = set(data.get('visited_locations', []))
        profile.preferred_times = data.get('preferred_times', [])
        profile.interaction_patterns = data.get('interaction_patterns', {})
        profile.created_at = datetime.fromisoformat(data['created_at'])
        profile.last_updated = datetime.fromisoformat(data['last_updated'])
        profile.total_interactions = data.get('total_interactions', 0)
        profile.satisfaction_scores = data.get('satisfaction_scores', [])
        
        return profile
    
    def get_user_statistics(self) -> Dict:
        """Get system-wide user statistics"""
        total_users = len(self.profiles)
        active_users = len([p for p in self.profiles.values() if p.total_interactions >= 3])
        avg_interactions = sum(p.total_interactions for p in self.profiles.values()) / max(1, total_users)
        avg_satisfaction = sum(p.get_avg_satisfaction() for p in self.profiles.values()) / max(1, total_users)
        
        # Preference distribution
        pref_distribution = defaultdict(Counter)
        for profile in self.profiles.values():
            for pref_type, prefs in profile.preferences.items():
                for pref in prefs:
                    pref_distribution[pref_type][pref.value] += 1
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'avg_interactions_per_user': round(avg_interactions, 1),
            'avg_user_satisfaction': round(avg_satisfaction, 2),
            'preference_distribution': dict(pref_distribution),
            'top_interests': dict(pref_distribution['interests'].most_common(5)),
            'budget_distribution': dict(pref_distribution['budget']),
            'transport_preferences': dict(pref_distribution['transport_mode'])
        }
    
    def export_user_insights(self, filepath: str = None) -> str:
        """Export user insights for analysis"""
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"user_insights_{timestamp}.json"
        
        insights = {
            'export_timestamp': datetime.now().isoformat(),
            'statistics': self.get_user_statistics(),
            'user_segments': self._analyze_user_segments(),
            'preference_trends': self._analyze_preference_trends(),
            'personalization_opportunities': self._identify_personalization_opportunities()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(insights, f, indent=2)
            
            logger.info(f"📊 User insights exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error exporting insights: {e}")
            return ""
    
    def _analyze_user_segments(self) -> Dict:
        """Analyze user segments based on behavior and preferences"""
        segments = {
            'culture_enthusiasts': 0,
            'food_lovers': 0,
            'budget_travelers': 0,
            'luxury_seekers': 0,
            'frequent_users': 0,
            'new_users': 0
        }
        
        for profile in self.profiles.values():
            # Culture enthusiasts
            if any(pref.value in ['historical', 'art', 'culture', 'religious'] 
                   for prefs in profile.preferences.get('interests', []) 
                   for pref in [prefs] if pref.confidence > 0.5):
                segments['culture_enthusiasts'] += 1
            
            # Food lovers
            if any(pref.value == 'food' 
                   for prefs in profile.preferences.get('interests', []) 
                   for pref in [prefs] if pref.confidence > 0.5):
                segments['food_lovers'] += 1
            
            # Budget/luxury segmentation
            budget_prefs = profile.get_top_preferences('budget', 1)
            if budget_prefs:
                if budget_prefs[0].value == 'budget':
                    segments['budget_travelers'] += 1
                elif budget_prefs[0].value == 'luxury':
                    segments['luxury_seekers'] += 1
            
            # Activity segmentation
            if profile.total_interactions >= 10:
                segments['frequent_users'] += 1
            elif profile.total_interactions <= 2:
                segments['new_users'] += 1
        
        return segments
    
    def _analyze_preference_trends(self) -> Dict:
        """Analyze trends in user preferences over time"""
        # This could be enhanced with time-series analysis
        recent_profiles = [p for p in self.profiles.values() 
                          if (datetime.now() - p.last_updated).days <= 30]
        
        trends = {
            'growing_interests': [],
            'declining_interests': [],
            'stable_preferences': [],
            'new_preferences': []
        }
        
        # Simple trend analysis - could be enhanced with more sophisticated methods
        interest_counts = Counter()
        for profile in recent_profiles:
            for prefs in profile.preferences.get('interests', []):
                interest_counts[prefs.value] += 1
        
        trends['popular_interests'] = dict(interest_counts.most_common(5))
        
        return trends
    
    def _identify_personalization_opportunities(self) -> List[str]:
        """Identify opportunities for better personalization"""
        opportunities = []
        
        stats = self.get_user_statistics()
        
        if stats['active_users'] > 10:
            opportunities.append("High user engagement - implement advanced recommendation algorithms")
        
        if stats['avg_user_satisfaction'] < 0.7:
            opportunities.append("Low satisfaction scores - improve personalization accuracy")
        
        if len(stats['top_interests']) > 3:
            opportunities.append("Diverse interests detected - create specialized content paths")
        
        budget_dist = stats.get('budget_distribution', {})
        if budget_dist.get('budget', 0) > budget_dist.get('luxury', 0):
            opportunities.append("Budget-conscious users dominant - emphasize cost-effective options")
        
        return opportunities

# Integration with existing query routing system
class PersonalizedQueryEnhancer:
    """
    Enhances queries with personalization data
    Integrates with existing query routing and caching systems
    """
    
    def __init__(self, profiling_system: UserProfilingSystem):
        self.profiling_system = profiling_system
    
    def enhance_query_context(self, query: str, user_id: str, 
                            base_context: Dict = None) -> Dict:
        """Enhance query context with personalization data"""
        enhanced_context = dict(base_context or {})
        
        # Get personalization context
        personal_context = self.profiling_system.get_personalized_context(user_id, query)
        enhanced_context.update(personal_context)
        
        # Add preference-based suggestions
        query_type = self._detect_query_type(query)
        suggestions = self.profiling_system.get_preference_based_suggestions(user_id, query_type)
        if suggestions:
            enhanced_context['personalized_suggestions'] = suggestions
        
        return enhanced_context
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type for personalization"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining']):
            return 'food'
        elif any(word in query_lower for word in ['see', 'visit', 'attraction', 'explore']):
            return 'exploration'
        elif any(word in query_lower for word in ['get to', 'transport', 'metro', 'bus']):
            return 'transportation'
        elif any(word in query_lower for word in ['buy', 'shop', 'market', 'souvenir']):
            return 'shopping'
        else:
            return 'general'
    
    def personalize_response(self, response: str, user_id: str, 
                           query: str, context: Dict = None) -> str:
        """Add personalization to response"""
        if user_id not in self.profiling_system.profiles:
            return response
        
        profile = self.profiling_system.profiles[user_id]
        personalized_response = response
        
        # Add personalized suggestions based on preferences
        suggestions = self.profiling_system.get_preference_based_suggestions(
            user_id, self._detect_query_type(query)
        )
        
        if suggestions and len(personalized_response) < 1000:  # Don't over-personalize long responses
            personalized_response += "\n\n💡 **Personalized for you:**\n"
            for suggestion in suggestions[:2]:  # Limit to 2 suggestions
                personalized_response += f"• {suggestion}\n"
        
        # Add experience-based context
        if profile.total_interactions > 5:
            personalized_response += f"\n\n👋 *Welcome back! This is our {profile.total_interactions}th conversation.*"
        
        return personalized_response
