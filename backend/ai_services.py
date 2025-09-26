# AI Intelligence Services for Enhanced Istanbul Travel Guide
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc, update
from thefuzz import fuzz, process
import re

from models import (
    UserSession, UserPreference, ConversationContext, 
    UserInteraction, IntelligentRecommendation, EnhancedChatHistory
)

class SessionManager:
    """Manages user sessions and conversation continuity"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_session(self, session_id: Optional[str] = None, user_ip: Optional[str] = None, user_agent: Optional[str] = None) -> UserSession:
        """Get existing session or create new one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        session = self.db.query(UserSession).filter(UserSession.session_id == session_id).first()
        
        if not session:
            session = UserSession(
                session_id=session_id,
                user_ip=user_ip,
                user_agent=user_agent,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                is_active=True
            )
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            
            # Create default preferences
            self._create_default_preferences(session_id)
        else:
            # Update last activity using update statement
            self.db.execute(
                update(UserSession)
                .where(UserSession.session_id == session_id)
                .values(last_activity=datetime.utcnow())
            )
            self.db.commit()
        
        return session
    
    def _create_default_preferences(self, session_id: str):
        """Create default user preferences"""
        preferences = UserPreference(
            session_id=session_id,
            preferred_cuisines=[],
            avoided_cuisines=[],
            budget_level="any",
            interests=[],
            travel_style="solo",
            preferred_time_of_day=["any"],
            preferred_districts=[],
            transportation_preference="mixed",
            language="en",
            confidence_score=0.0,
            total_interactions=0
        )
        self.db.add(preferences)
        self.db.commit()
    
    def cleanup_old_sessions(self, days_old: int = 7):
        """Clean up sessions older than specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        self.db.execute(
            update(UserSession)
            .where(UserSession.last_activity < cutoff_date)
            .values(is_active=False)
        )
        self.db.commit()

class PreferenceManager:
    """Manages and learns user preferences"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_preferences(self, session_id: str) -> Optional[UserPreference]:
        """Get user preferences for session"""
        return self.db.query(UserPreference).filter(
            UserPreference.session_id == session_id
        ).first()
    
    def update_preferences_from_query(self, session_id: str, user_input: str, detected_intent: str):
        """Learn preferences from user queries"""
        preferences = self.get_preferences(session_id)
        if not preferences:
            return
        
        updated = False
        user_input_lower = user_input.lower()
        
        # Get current values to work with
        current_cuisines = preferences.preferred_cuisines or []
        current_budget = preferences.budget_level
        current_interests = preferences.interests or []
        current_districts = preferences.preferred_districts or []
        current_travel_style = preferences.travel_style
        
        # Learn cuisine preferences
        cuisine_keywords = {
            'turkish': ['turkish', 'ottoman', 'kebab', 'döner', 'traditional'],
            'italian': ['italian', 'pizza', 'pasta', 'mediterranean'],
            'seafood': ['fish', 'seafood', 'balık', 'deniz'],
            'asian': ['asian', 'sushi', 'japanese', 'chinese'],
            'european': ['european', 'french', 'german'],
            'american': ['burger', 'american', 'fast food'],
            'vegetarian': ['vegetarian', 'vegan', 'salad'],
            'desserts': ['dessert', 'sweet', 'bakery', 'pastane']
        }
        
        for cuisine, keywords in cuisine_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                if cuisine not in current_cuisines:
                    current_cuisines.append(cuisine)
                    updated = True
        
        # Learn budget preferences from keywords
        new_budget = current_budget
        if any(word in user_input_lower for word in ['cheap', 'budget', 'affordable', 'inexpensive']):
            new_budget = 'budget'
            updated = True
        elif any(word in user_input_lower for word in ['expensive', 'luxury', 'high-end', 'premium']):
            new_budget = 'luxury'
            updated = True
        elif any(word in user_input_lower for word in ['moderate', 'mid-range', 'reasonable']):
            new_budget = 'mid-range'
            updated = True
        
        # Learn interests from intent and keywords
        interest_mapping = {
            'restaurant_search': 'dining',
            'museum_query': 'museums',
            'attraction_query': 'attractions',
            'transportation_query': 'transportation',
            'nightlife_query': 'nightlife',
            'shopping_query': 'shopping',
            'culture_query': 'culture'
        }
        
        if detected_intent in interest_mapping:
            interest = interest_mapping[detected_intent]
            if interest not in current_interests:
                current_interests.append(interest)
                updated = True
        
        # Learn district preferences
        districts = ['sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar', 
                    'fatih', 'taksim', 'karakoy', 'ortakoy', 'bebek']
        for district in districts:
            if district in user_input_lower:
                if district not in current_districts:
                    current_districts.append(district)
                    updated = True
        
        # Learn travel style
        new_travel_style = current_travel_style
        if any(word in user_input_lower for word in ['family', 'kids', 'children']):
            new_travel_style = 'family'
            updated = True
        elif any(word in user_input_lower for word in ['couple', 'romantic', 'date']):
            new_travel_style = 'couple'
            updated = True
        elif any(word in user_input_lower for word in ['business', 'work', 'meeting']):
            new_travel_style = 'business'
            updated = True
        elif any(word in user_input_lower for word in ['group', 'friends', 'together']):
            new_travel_style = 'group'
            updated = True
        
        if updated:
            # Update using SQL update statement to avoid SQLAlchemy column assignment issues
            # Safely get current values as Python types, not SQLAlchemy columns
            confidence_val = getattr(preferences, 'confidence_score', None)
            current_confidence = float(confidence_val) if confidence_val is not None else 0.0
            new_confidence = min(1.0, current_confidence + 0.1)
            
            interactions_val = getattr(preferences, 'total_interactions', None)
            current_interactions = int(interactions_val) if interactions_val is not None else 0
            
            self.db.execute(
                update(UserPreference)
                .where(UserPreference.session_id == session_id)
                .values(
                    preferred_cuisines=current_cuisines,
                    budget_level=new_budget,
                    interests=current_interests,
                    preferred_districts=current_districts,
                    travel_style=new_travel_style,
                    confidence_score=new_confidence,
                    total_interactions=current_interactions + 1,
                    last_updated=datetime.utcnow()
                )
            )
            self.db.commit()
    
    def get_personalized_filter(self, session_id: str) -> Dict[str, Any]:
        """Get filter criteria based on user preferences"""
        preferences = self.get_preferences(session_id)
        if not preferences:
            return {}
        
        return {
            'preferred_cuisines': preferences.preferred_cuisines or [],
            'budget_level': preferences.budget_level,
            'interests': preferences.interests or [],
            'preferred_districts': preferences.preferred_districts or [],
            'travel_style': preferences.travel_style,
            'confidence_score': float(getattr(preferences, 'confidence_score', None) or 0.0)
        }

class ConversationContextManager:
    """Manages conversation context and follow-up handling"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_context(self, session_id: str) -> ConversationContext:
        """Get current conversation context"""
        context = self.db.query(ConversationContext).filter(
            ConversationContext.session_id == session_id
        ).order_by(desc(ConversationContext.updated_at)).first()
        
        if not context:
            context = ConversationContext(
                session_id=session_id,
                current_intent="initial",
                context_data={},
                current_location="",
                previous_locations=[],
                current_topic="",
                topic_history=[],
                expecting_followup=False,
                conversation_stage="initial"
            )
            self.db.add(context)
            self.db.commit()
            self.db.refresh(context)
        
        return context
    
    def update_context(self, session_id: str, intent: str, location: Optional[str] = None, 
                      topic: Optional[str] = None, stage: Optional[str] = None, context_data: Optional[Dict] = None):
        """Update conversation context"""
        context = self.get_or_create_context(session_id)
        
        # Prepare update values with explicit typing
        update_values: Dict[str, Any] = {
            'updated_at': datetime.utcnow()
        }
        
        # Update intent
        if intent != getattr(context, 'current_intent', None):
            update_values['current_intent'] = intent
        
        # Update location context
        if location and location != getattr(context, 'current_location', None):
            # Get current values
            current_location = getattr(context, 'current_location', None)
            previous_locations = getattr(context, 'previous_locations', None) or []
            
            if current_location and current_location not in previous_locations:
                previous_locations.append(current_location)
                update_values['previous_locations'] = previous_locations
            
            update_values['current_location'] = location
        
        # Update topic
        if topic and topic != getattr(context, 'current_topic', None):
            current_topic = getattr(context, 'current_topic', None)
            topic_history = getattr(context, 'topic_history', None) or []
            
            if current_topic and current_topic not in topic_history:
                topic_history.append(current_topic)
                update_values['topic_history'] = topic_history
            
            update_values['current_topic'] = topic
        
        # Update stage
        if stage:
            update_values['conversation_stage'] = stage
        
        # Update context data
        if context_data:
            current_context_data = getattr(context, 'context_data', None) or {}
            current_context_data.update(context_data)
            update_values['context_data'] = current_context_data
        
        # Execute update
        self.db.execute(
            update(ConversationContext)
            .where(ConversationContext.session_id == session_id)
            .values(**update_values)
        )
        self.db.commit()
    
    def is_followup_question(self, user_input: str, context: ConversationContext) -> bool:
        """Detect if this is a follow-up question"""
        followup_indicators = [
            r'\bthere\b', r'\bwhat about\b', r'\bany others?\b', r'\bmore\b',
            r'\balso\b', r'\badditionally\b', r'\bbesides\b', r'\belse\b',
            r'\bnear there\b', r'\bin that area\b', r'\baround there\b'
        ]
        
        user_input_lower = user_input.lower()
        return any(re.search(pattern, user_input_lower) for pattern in followup_indicators)
    
    def get_conversation_history_context(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Get recent conversation history for context"""
        history = self.db.query(EnhancedChatHistory).filter(
            EnhancedChatHistory.session_id == session_id
        ).order_by(desc(EnhancedChatHistory.timestamp)).limit(limit).all()
        
        return [
            {
                'user_message': h.user_message,
                'bot_response': h.bot_response,
                'intent': h.detected_intent,
                'entities': h.extracted_entities,
                'timestamp': h.timestamp
            }
            for h in reversed(history)  # Return in chronological order
        ]

class IntelligentIntentRecognizer:
    """Enhanced intent recognition with NLP and context awareness"""
    
    def __init__(self, db: Session):
        self.db = db
        
        # Enhanced intent patterns with confidence scoring
        self.intent_patterns = {
            'restaurant_search': {
                'keywords': ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal'],
                'patterns': [
                    r'restaurants?\s+in\s+\w+', r'food\s+in\s+\w+', r'where\s+to\s+eat',
                    r'dining\s+options', r'good\s+restaurants?', r'best\s+food'
                ],
                'confidence_boost': 0.2
            },
            'transportation_query': {
                'keywords': ['transport', 'metro', 'bus', 'taxi', 'how to get', 'directions'],
                'patterns': [
                    r'how.*get.*from.*to', r'go.*from.*to', r'transport.*to',
                    r'metro.*to', r'bus.*to', r'directions'
                ],
                'confidence_boost': 0.3
            },
            'museum_query': {
                'keywords': ['museum', 'exhibition', 'art', 'gallery', 'cultural', 'history'],
                'patterns': [
                    r'museums?\s+in\s+\w+', r'art\s+gallery', r'exhibition',
                    r'cultural\s+sites', r'history\s+museum'
                ],
                'confidence_boost': 0.2
            },
            'attraction_query': {
                'keywords': ['attraction', 'tourist', 'sightseeing', 'visit', 'places', 'landmarks'],
                'patterns': [
                    r'places\s+to\s+visit', r'tourist\s+attractions?', r'things\s+to\s+do',
                    r'sightseeing', r'landmarks?', r'what\s+to\s+see'
                ],
                'confidence_boost': 0.2
            },
            'accommodation_query': {
                'keywords': ['hotel', 'accommodation', 'stay', 'hostel', 'booking'],
                'patterns': [
                    r'where\s+to\s+stay', r'hotels?\s+in', r'accommodation',
                    r'booking.*hotel', r'best\s+hotels?'
                ],
                'confidence_boost': 0.2
            },
            'shopping_query': {
                'keywords': ['shopping', 'shop', 'buy', 'market', 'bazaar', 'mall'],
                'patterns': [
                    r'shopping\s+in', r'where\s+to\s+shop', r'markets?',
                    r'bazaars?', r'buy.*souvenirs?'
                ],
                'confidence_boost': 0.2
            },
            'nightlife_query': {
                'keywords': ['nightlife', 'bars', 'clubs', 'drinks', 'party', 'night out'],
                'patterns': [
                    r'night.*life', r'bars?\s+in', r'clubs?\s+in',
                    r'night\s+out', r'drinks?', r'party'
                ],
                'confidence_boost': 0.2
            }
        }
    def recognize_intent(self, user_input: str, context: Optional[ConversationContext] = None,
                        preferences: Optional[UserPreference] = None) -> Tuple[str, float]:
        """Recognize intent with confidence scoring"""
        user_input_lower = user_input.lower()
        intent_scores = {}
        
        # Score based on keywords and patterns
        for intent, config in self.intent_patterns.items():
            score = 0.0
            
            # Keyword matching with fuzzy logic
            for keyword in config['keywords']:
                best_match = process.extractOne(keyword, user_input_lower.split())
                if best_match and best_match[1] > 80:  # 80% similarity threshold
                    score += 0.1 * (best_match[1] / 100)
            
            # Pattern matching
            for pattern in config['patterns']:
                if re.search(pattern, user_input_lower):
                    score += config['confidence_boost']
            
            # Context-based boosting
            if context and getattr(context, 'current_intent', None) == intent:
                score += 0.1  # Boost for conversation continuity
            
            # Preference-based boosting
            if preferences and intent.replace('_query', '').replace('_search', '') in preferences.interests:
                score += 0.1
            
            intent_scores[intent] = score
        
        # Handle multi-intent queries
        top_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        
        if top_intents and top_intents[0][1] > 0.2:
            return top_intents[0][0], top_intents[0][1]
        else:
            return 'general_query', 0.1
    
    def extract_entities(self, user_input: str) -> Dict[str, Any]:
        """Extract entities from user input"""
        entities = {
            'locations': [],
            'time_references': [],
            'cuisine_types': [],
            'budget_indicators': [],
            'group_size_indicators': []
        }
        
        user_input_lower = user_input.lower()
        
        # Extract locations (Istanbul districts and neighborhoods)
        locations = [
            'sultanahmet', 'beyoglu', 'galata', 'kadikoy', 'besiktas', 'uskudar',
            'fatih', 'taksim', 'karakoy', 'ortakoy', 'bebek', 'sisli', 'bakirkoy'
        ]
        
        for location in locations:
            if location in user_input_lower:
                entities['locations'].append(location)
        
        # Extract time references
        time_patterns = [
            (r'morning|breakfast', 'morning'),
            (r'lunch|afternoon', 'afternoon'), 
            (r'dinner|evening', 'evening'),
            (r'night|late', 'night'),
            (r'weekend', 'weekend'),
            (r'today', 'today'),
            (r'tomorrow', 'tomorrow')
        ]
        
        for pattern, time_ref in time_patterns:
            if re.search(pattern, user_input_lower):
                entities['time_references'].append(time_ref)
        
        # Extract cuisine types
        cuisines = ['turkish', 'italian', 'seafood', 'asian', 'european', 'vegetarian']
        for cuisine in cuisines:
            if cuisine in user_input_lower:
                entities['cuisine_types'].append(cuisine)
        
        # Extract budget indicators
        budget_patterns = [
            (r'cheap|budget|affordable', 'budget'),
            (r'expensive|luxury|high-end', 'luxury'),
            (r'moderate|mid-range', 'mid-range')
        ]
        
        for pattern, budget in budget_patterns:
            if re.search(pattern, user_input_lower):
                entities['budget_indicators'].append(budget)
        
        # Extract group size indicators
        group_patterns = [
            (r'family|kids|children', 'family'),
            (r'couple|romantic|date', 'couple'),
            (r'group|friends', 'group'),
            (r'solo|alone|myself', 'solo')
        ]
        
        for pattern, group in group_patterns:
            if re.search(pattern, user_input_lower):
                entities['group_size_indicators'].append(group)
        
        return entities

class PersonalizedRecommendationEngine:
    """Generates personalized recommendations based on user context and preferences"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def generate_personalized_recommendations(self, session_id: str, intent: str, 
                                           base_results: List[Dict], 
                                           context: Optional[Dict] = None) -> List[Dict]:
        """Generate personalized recommendations with scoring"""
        preferences = self.db.query(UserPreference).filter(
            UserPreference.session_id == session_id
        ).first()
        
        if not preferences or not base_results:
            return base_results
        
        scored_results = []
        current_time = datetime.now().hour
        
        for item in base_results:
            score = self._calculate_personalization_score(item, preferences, current_time, context or {})
            
            scored_item = item.copy()
            scored_item['personalization_score'] = score
            scored_item['recommendation_reason'] = self._generate_recommendation_reason(
                item, preferences, current_time
            )
            
            scored_results.append(scored_item)
        
        # Sort by personalization score
        return sorted(scored_results, key=lambda x: x.get('personalization_score', 0), reverse=True)
    
    def _calculate_personalization_score(self, item: Dict, preferences: UserPreference, 
                                       current_time: int, context: Optional[Dict] = None) -> float:
        """Calculate personalization score for an item"""
        score = 0.5  # Base score
        
        # Cuisine preference matching
        if 'cuisine' in item or 'category' in item:
            item_cuisine = (item.get('cuisine', '') + ' ' + item.get('category', '')).lower()
            for pref_cuisine in preferences.preferred_cuisines:
                if pref_cuisine in item_cuisine:
                    score += 0.2
        
        # District preference matching
        if 'district' in item or 'location' in item:
            item_location = (item.get('district', '') + ' ' + item.get('location', '')).lower()
            for pref_district in preferences.preferred_districts:
                if pref_district in item_location:
                    score += 0.15
        
        # Time-based scoring
        if current_time < 11:  # Morning
            if any(word in item.get('name', '').lower() for word in ['cafe', 'breakfast', 'kahve']):
                score += 0.1
        elif 11 <= current_time < 15:  # Lunch
            if any(word in item.get('name', '').lower() for word in ['lunch', 'quick', 'fast']):
                score += 0.1
        elif current_time >= 18:  # Evening/Dinner
            if any(word in item.get('name', '').lower() for word in ['restaurant', 'dinner', 'fine']):
                score += 0.1
        
        # Budget preference matching
        budget_level = getattr(preferences, 'budget_level', None)
        if budget_level and 'price_level' in item:
            price_level = item.get('price_level', 2)
            if budget_level == 'budget' and price_level <= 2:
                score += 0.1
            elif budget_level == 'luxury' and price_level >= 3:
                score += 0.1
            elif budget_level == 'mid-range' and price_level == 2:
                score += 0.1
        
        # Travel style matching
        travel_style = getattr(preferences, 'travel_style', None)
        if travel_style == 'family':
            if any(word in item.get('name', '').lower() for word in ['family', 'kid', 'child']):
                score += 0.1
        elif travel_style == 'couple':
            if any(word in item.get('name', '').lower() for word in ['romantic', 'intimate', 'cozy']):
                score += 0.1
        
        return min(1.0, score)  # Cap at 1.0
    
    def _generate_recommendation_reason(self, item: Dict, preferences: UserPreference, 
                                      current_time: int) -> str:
        """Generate human-readable recommendation reason"""
        reasons = []
        
        # Check cuisine match
        item_cuisine = (item.get('cuisine', '') + ' ' + item.get('category', '')).lower()
        preferred_cuisines = getattr(preferences, 'preferred_cuisines', None) or []
        matching_cuisines = [c for c in preferred_cuisines if c in item_cuisine]
        if matching_cuisines:
            reasons.append(f"matches your interest in {', '.join(str(c) for c in matching_cuisines)} cuisine")
        
        # Check location match
        item_location = (item.get('district', '') + ' ' + item.get('location', '')).lower()
        preferred_districts = getattr(preferences, 'preferred_districts', None) or []
        matching_districts = [d for d in preferred_districts if d in item_location]
        if matching_districts:
            reasons.append(f"located in your preferred area of {', '.join(str(d) for d in matching_districts)}")
        
        # Time-based reasons
        if current_time < 11 and any(word in item.get('name', '').lower() for word in ['cafe', 'breakfast']):
            reasons.append("perfect for morning coffee or breakfast")
        elif current_time >= 18 and 'restaurant' in item.get('name', '').lower():
            reasons.append("great for dinner")
        
        # Travel style reasons
        travel_style = getattr(preferences, 'travel_style', None)
        if travel_style == 'family':
            reasons.append("suitable for families")
        elif travel_style == 'couple':
            reasons.append("perfect for couples")
        
        if not reasons:
            return "popular choice among visitors"
        
        return "Recommended because it " + " and ".join(reasons)

class AIAnalyticsTracker:
    """Track AI performance and user satisfaction"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def track_interaction(self, session_id: str, user_message: str, bot_response: str,
                         intent: str, confidence: float, processing_time: int,
                         recommendations: Optional[List[Dict]] = None):
        """Track user interaction for analytics"""
        interaction = UserInteraction(
            session_id=session_id,
            user_message=user_message,
            processed_intent=intent,
            bot_response=bot_response,
            confidence_score=confidence,
            response_time_ms=processing_time,
            recommendations_given=recommendations or []
        )
        
        self.db.add(interaction)
        self.db.commit()
    
    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a specific session"""
        interactions = self.db.query(UserInteraction).filter(
            UserInteraction.session_id == session_id
        ).all()
        
        if not interactions:
            return {}
        
        total_interactions = len(interactions)
        avg_confidence = sum(i.confidence_score or 0 for i in interactions) / total_interactions
        avg_response_time = sum(i.response_time_ms or 0 for i in interactions) / total_interactions
        
        intent_distribution = {}
        for interaction in interactions:
            intent = interaction.processed_intent
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
        
        return {
            'total_interactions': total_interactions,
            'average_confidence': avg_confidence,
            'average_response_time_ms': avg_response_time,
            'intent_distribution': intent_distribution,
            'satisfaction_indicators': self._calculate_satisfaction_indicators(interactions)
        }
    
    def _calculate_satisfaction_indicators(self, interactions: List[UserInteraction]) -> Dict[str, Any]:
        """Calculate satisfaction indicators from interactions"""
        # Simple heuristics for satisfaction
        high_confidence_interactions = sum(1 for i in interactions if (getattr(i, 'confidence_score', None) or 0) > 0.7)
        fast_responses = sum(1 for i in interactions if (getattr(i, 'response_time_ms', None) or 0) < 1000)
        
        return {
            'high_confidence_rate': high_confidence_interactions / len(interactions) if interactions else 0,
            'fast_response_rate': fast_responses / len(interactions) if interactions else 0,
            'conversation_length': len(interactions),
            'engagement_score': min(1.0, len(interactions) / 10)  # Longer conversations indicate engagement
        }
