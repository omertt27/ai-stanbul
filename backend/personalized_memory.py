#!/usr/bin/env python3
"""
AI Istanbul - Personalized Memory System
========================================

This module implements contextual and personalized memory for AI Istanbul,
enabling the system to:
1. Remember user preferences across sessions
2. Track visited places and experiences
3. Provide tailored responses based on conversation history
4. Build user profiles for improved recommendations

Features:
- Preference learning and storage
- Contextual memory extraction
- Conversation continuity
- Personalized recommendations
- Memory-based response enhancement
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from models import UserMemory, UserPreference, ConversationContext, ChatSession, ChatHistory
import logging

logger = logging.getLogger(__name__)

class PersonalizedMemorySystem:
    """Manages personalized memory and context for AI Istanbul users"""
    
    def __init__(self):
        self.memory_extractors = {
            'preferences': self._extract_preferences,
            'places': self._extract_places,
            'experiences': self._extract_experiences,
            'context': self._extract_context,
            'travel_style': self._extract_travel_style
        }
    
    def process_conversation(self, session_id: str, user_message: str, ai_response: str, db: Session) -> Dict[str, Any]:
        """Process a conversation turn and extract/update memory"""
        try:
            # Extract memories from the conversation
            memories = self._extract_memories(user_message, ai_response)
            
            # Update database with new memories
            updated_memories = self._update_memories(session_id, memories, db)
            
            # Update conversation context
            context_updates = self._update_conversation_context(session_id, user_message, ai_response, db)
            
            return {
                "memories_extracted": len(memories),
                "memories_updated": updated_memories,
                "context_updated": context_updates,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing conversation memory: {e}")
            return {"success": False, "error": str(e)}
    
    def get_personalized_context(self, session_id: str, current_query: str, db: Session) -> Dict[str, Any]:
        """Get personalized context for enhancing AI responses"""
        try:
            # Get user memories
            memories = self._get_user_memories(session_id, db)
            
            # Get user preferences
            preferences = self._get_user_preferences(session_id, db)
            
            # Get conversation context
            context = self._get_conversation_context(session_id, db)
            
            # Generate personalization hints
            personalization = self._generate_personalization_hints(memories, preferences, context, current_query)
            
            return {
                "memories": memories,
                "preferences": preferences,
                "context": context,
                "personalization_hints": personalization,
                "has_history": len(memories) > 0 or len(preferences) > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting personalized context: {e}")
            return {"has_history": False, "personalization_hints": {}}
    
    def _extract_memories(self, user_message: str, ai_response: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract different types of memories from conversation"""
        memories = {
            'preferences': [],
            'places': [],
            'experiences': [],
            'context': [],
            'travel_style': []
        }
        
        # Extract from user message
        for memory_type, extractor in self.memory_extractors.items():
            try:
                extracted = extractor(user_message, ai_response)
                if extracted:
                    memories[memory_type].extend(extracted)
            except Exception as e:
                logger.warning(f"Error extracting {memory_type}: {e}")
        
        return memories
    
    def _extract_preferences(self, user_message: str, ai_response: str) -> List[Dict[str, Any]]:
        """Extract user preferences from conversation"""
        preferences = []
        user_lower = user_message.lower()
        
        # Preference indicators
        preference_patterns = {
            'like': ['i like', 'i love', 'i enjoyed', 'i prefer', 'favorite', 'loved'],
            'dislike': ['i dislike', 'i hate', 'not a fan', 'dont like', "don't like", 'avoid'],
            'love': ['amazing', 'incredible', 'wonderful', 'beautiful', 'perfect'],
            'prefer': ['prefer', 'better', 'rather', 'instead of']
        }
        
        # District preferences
        districts = ['sultanahmet', 'beyoglu', 'galata', 'karakoy', 'besiktas', 'kadikoy', 'taksim', 'ortakoy', 'balat', 'fener']
        for district in districts:
            if district in user_lower:
                for pref_type, patterns in preference_patterns.items():
                    if any(pattern in user_lower for pattern in patterns):
                        preferences.append({
                            'category': 'districts',
                            'preference_name': district.title(),
                            'preference_value': pref_type,
                            'strength': 0.8,
                            'context_info': {'extracted_from': user_message[:100]}
                        })
        
        # Food preferences
        food_keywords = ['vegetarian', 'vegan', 'halal', 'seafood', 'meat', 'turkish cuisine', 'kebab', 'breakfast', 'dessert', 'street food']
        for food in food_keywords:
            if food in user_lower:
                for pref_type, patterns in preference_patterns.items():
                    if any(pattern in user_lower for pattern in patterns):
                        preferences.append({
                            'category': 'food',
                            'preference_name': food.title(),
                            'preference_value': pref_type,
                            'strength': 0.7,
                            'context_info': {'extracted_from': user_message[:100]}
                        })
        
        # Activity preferences
        activities = ['museums', 'shopping', 'nightlife', 'walking', 'photography', 'history', 'culture', 'art', 'music', 'architecture']
        for activity in activities:
            if activity in user_lower:
                for pref_type, patterns in preference_patterns.items():
                    if any(pattern in user_lower for pattern in patterns):
                        preferences.append({
                            'category': 'activities',
                            'preference_name': activity.title(),
                            'preference_value': pref_type,
                            'strength': 0.6,
                            'context_info': {'extracted_from': user_message[:100]}
                        })
        
        return preferences
    
    def _extract_places(self, user_message: str, ai_response: str) -> List[Dict[str, Any]]:
        """Extract mentioned places and user's relationship to them"""
        places = []
        user_lower = user_message.lower()
        
        # Istanbul landmarks and attractions
        landmarks = [
            'hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar', 'spice bazaar',
            'galata tower', 'bosphorus', 'golden horn', 'basilica cistern', 'dolmabahce palace',
            'taksim square', 'istiklal street', 'karakoy', 'ortakoy', 'balat', 'fener'
        ]
        
        # Visit indicators
        visit_patterns = {
            'visited': ['visited', 'went to', 'been to', 'saw', 'at the'],
            'planning': ['want to visit', 'planning to go', 'going to', 'will visit'],
            'interested': ['interested in', 'want to see', 'heard about', 'tell me about'],
            'loved': ['loved', 'amazing', 'beautiful', 'incredible'],
            'disappointed': ['disappointed', 'not worth it', 'overrated', 'crowded']
        }
        
        for landmark in landmarks:
            if landmark in user_lower:
                for status, patterns in visit_patterns.items():
                    if any(pattern in user_lower for pattern in patterns):
                        places.append({
                            'memory_type': 'visited',
                            'memory_key': landmark.replace(' ', '_'),
                            'memory_value': status,
                            'memory_context': {
                                'place_name': landmark.title(),
                                'status': status,
                                'mentioned_at': datetime.utcnow().isoformat(),
                                'context': user_message[:150]
                            },
                            'confidence_score': 0.8
                        })
        
        return places
    
    def _extract_experiences(self, user_message: str, ai_response: str) -> List[Dict[str, Any]]:
        """Extract user experiences and emotions"""
        experiences = []
        user_lower = user_message.lower()
        
        # Experience indicators
        experience_patterns = {
            'positive': ['enjoyed', 'loved', 'amazing', 'wonderful', 'great time', 'beautiful', 'impressed'],
            'negative': ['disappointed', 'crowded', 'overrated', 'not worth', 'too expensive', 'rushed'],
            'neutral': ['okay', 'fine', 'average', 'normal'],
            'challenging': ['difficult', 'confusing', 'lost', 'hard to find', 'complicated']
        }
        
        for exp_type, patterns in experience_patterns.items():
            if any(pattern in user_lower for pattern in patterns):
                experiences.append({
                    'memory_type': 'experience',
                    'memory_key': f'general_{exp_type}_experience',
                    'memory_value': exp_type,
                    'memory_context': {
                        'experience_type': exp_type,
                        'description': user_message[:200],
                        'timestamp': datetime.utcnow().isoformat()
                    },
                    'confidence_score': 0.6
                })
        
        return experiences
    
    def _extract_context(self, user_message: str, ai_response: str) -> List[Dict[str, Any]]:
        """Extract conversation context clues"""
        context = []
        user_lower = user_message.lower()
        
        # Travel stage indicators
        stage_patterns = {
            'planning': ['planning', 'will visit', 'going to visit', 'itinerary'],
            'visiting': ['here now', 'currently in', 'just arrived', 'visiting now'],
            'exploring': ['looking for', 'want to find', 'where can i', 'recommendations'],
            'departing': ['last day', 'leaving tomorrow', 'going home']
        }
        
        for stage, patterns in stage_patterns.items():
            if any(pattern in user_lower for pattern in patterns):
                context.append({
                    'memory_type': 'context',
                    'memory_key': 'travel_stage',
                    'memory_value': stage,
                    'memory_context': {'detected_at': datetime.utcnow().isoformat()},
                    'confidence_score': 0.7
                })
        
        # Duration indicators
        duration_patterns = ['1 day', '2 days', '3 days', '1 week', '2 weeks', 'weekend', 'short visit', 'quick trip']
        for duration in duration_patterns:
            if duration in user_lower:
                context.append({
                    'memory_type': 'context',
                    'memory_key': 'visit_duration',
                    'memory_value': duration,
                    'memory_context': {'detected_at': datetime.utcnow().isoformat()},
                    'confidence_score': 0.8
                })
        
        return context
    
    def _extract_travel_style(self, user_message: str, ai_response: str) -> List[Dict[str, Any]]:
        """Extract travel style and companions"""
        travel_style = []
        user_lower = user_message.lower()
        
        # Travel style indicators
        style_patterns = {
            'solo': ['solo', 'alone', 'by myself', 'traveling alone'],
            'family': ['family', 'kids', 'children', 'parents', 'with family'],
            'couple': ['romantic', 'date', 'anniversary', 'honeymoon', 'with partner'],
            'friends': ['friends', 'group', 'with friends', 'together'],
            'business': ['business', 'work', 'conference', 'meeting', 'corporate']
        }
        
        for style, patterns in style_patterns.items():
            if any(pattern in user_lower for pattern in patterns):
                travel_style.append({
                    'memory_type': 'travel_style',
                    'memory_key': 'companion_type',
                    'memory_value': style,
                    'memory_context': {'detected_at': datetime.utcnow().isoformat()},
                    'confidence_score': 0.8
                })
        
        return travel_style
    
    def _ensure_user_session_exists(self, session_id: str, db: Session):
        """Ensure user session exists in database, create if it doesn't"""
        from models import UserSession
        
        existing_session = db.query(UserSession).filter(
            UserSession.session_id == session_id
        ).first()
        
        if not existing_session:
            # Create new user session
            new_session = UserSession(
                session_id=session_id,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                is_active=True
            )
            db.add(new_session)
            db.flush()  # Ensure the session is created before we continue
    
    def _update_memories(self, session_id: str, memories: Dict[str, List[Dict]], db: Session) -> int:
        """Update user memories in database"""
        updated_count = 0
        
        try:
            # Ensure user session exists before storing memories
            self._ensure_user_session_exists(session_id, db)
            
            for memory_type, memory_list in memories.items():
                for memory in memory_list:
                    if memory_type == 'preferences':
                        # Skip preferences - they are handled by PreferenceManager in ai_services.py
                        # This avoids schema conflicts between old and new preference storage
                        logger.info(f"Skipping preference storage: {memory}")
                        continue
                    else:
                        # Handle general memories
                        self._upsert_memory(session_id, memory, db)
                        updated_count += 1
            
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating memories: {e}")
            
        return updated_count
    
    def _upsert_preference(self, session_id: str, pref_data: Dict, db: Session):
        """Insert or update a user preference"""
        existing = db.query(UserPreference).filter(
            UserPreference.session_id == session_id,
            UserPreference.category == pref_data['category'],
            UserPreference.preference_name == pref_data['preference_name']
        ).first()
        
        if existing:
            # Update existing preference
            existing.preference_value = pref_data['preference_value']
            existing.strength = max(existing.strength, pref_data.get('strength', 0.7))
            existing.updated_at = datetime.utcnow()
        else:
            # Create new preference
            new_pref = UserPreference(
                session_id=session_id,
                category=pref_data['category'],
                preference_name=pref_data['preference_name'],
                preference_value=pref_data['preference_value'],
                strength=pref_data.get('strength', 0.7),
                context_info=pref_data.get('context_info', {}),
                inferred_from=pref_data.get('context_info', {}).get('extracted_from', '')[:200]
            )
            db.add(new_pref)
    
    def _upsert_memory(self, session_id: str, memory_data: Dict, db: Session):
        """Insert or update a user memory"""
        existing = db.query(UserMemory).filter(
            UserMemory.session_id == session_id,
            UserMemory.memory_type == memory_data['memory_type'],
            UserMemory.memory_key == memory_data['memory_key']
        ).first()
        
        if existing:
            # Update existing memory
            existing.memory_value = memory_data['memory_value']
            existing.memory_context = memory_data.get('memory_context', {})
            existing.updated_at = datetime.utcnow()
            existing.last_referenced = datetime.utcnow()
            existing.reference_count += 1
        else:
            # Create new memory
            new_memory = UserMemory(
                session_id=session_id,
                memory_type=memory_data['memory_type'],
                memory_key=memory_data['memory_key'],
                memory_value=memory_data['memory_value'],
                memory_context=memory_data.get('memory_context', {}),
                confidence_score=memory_data.get('confidence_score', 0.7)
            )
            db.add(new_memory)
    
    def _update_conversation_context(self, session_id: str, user_message: str, ai_response: str, db: Session) -> bool:
        """Update conversation context"""
        try:
            # Ensure user session exists
            self._ensure_user_session_exists(session_id, db)
            
            # Get or create conversation context
            context = db.query(ConversationContext).filter(
                ConversationContext.session_id == session_id
            ).first()
            
            if not context:
                context = ConversationContext(session_id=session_id)
                db.add(context)
            
            # Update context based on conversation
            self._analyze_conversation_context(context, user_message, ai_response)
            context.updated_at = datetime.utcnow()
            
            db.commit()
            return True
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating conversation context: {e}")
            return False
    
    def _analyze_conversation_context(self, context: ConversationContext, user_message: str, ai_response: str):
        """Analyze and update conversation context"""
        user_lower = user_message.lower()
        
        # Detect current topic
        topic_keywords = {
            'transportation': ['metro', 'bus', 'taxi', 'airport', 'transport', 'get to', 'directions'],
            'restaurants': ['restaurant', 'food', 'eat', 'dining', 'breakfast', 'dinner'],
            'museums': ['museum', 'palace', 'mosque', 'cultural', 'historical', 'art'],
            'shopping': ['shopping', 'bazaar', 'market', 'buy', 'souvenir', 'clothes'],
            'accommodation': ['hotel', 'stay', 'accommodation', 'where to sleep', 'booking'],
            'general_info': ['information', 'tell me about', 'what is', 'explain', 'help']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                context.current_topic = topic
                break
        
        # Update topics discussed
        if context.topics_discussed is None:
            context.topics_discussed = []
        
        if context.current_topic and context.current_topic not in context.topics_discussed:
            context.topics_discussed.append(context.current_topic)
        
        # Extract places mentioned
        places = ['sultanahmet', 'galata tower', 'blue mosque', 'hagia sophia', 'grand bazaar', 
                 'taksim', 'beyoglu', 'kadikoy', 'bosphorus', 'topkapi palace']
        
        if context.places_mentioned is None:
            context.places_mentioned = []
            
        for place in places:
            if place in user_lower and place not in context.places_mentioned:
                context.places_mentioned.append(place)
                context.last_location_discussed = place
    
    def _get_user_memories(self, session_id: str, db: Session) -> Dict[str, List[Dict]]:
        """Get all user memories for a session"""
        memories = db.query(UserMemory).filter(
            UserMemory.session_id == session_id
        ).order_by(UserMemory.last_referenced.desc()).limit(20).all()
        
        organized_memories = {}
        for memory in memories:
            if memory.memory_type not in organized_memories:
                organized_memories[memory.memory_type] = []
            
            organized_memories[memory.memory_type].append({
                'key': memory.memory_key,
                'value': memory.memory_value,
                'context': memory.memory_context,
                'confidence': memory.confidence_score,
                'last_referenced': memory.last_referenced.isoformat() if memory.last_referenced else None
            })
        
        return organized_memories
    
    def _get_user_preferences(self, session_id: str, db: Session) -> Dict[str, List[Dict]]:
        """Get all user preferences for a session"""
        preferences = db.query(UserPreference).filter(
            UserPreference.session_id == session_id
        ).first()  # Get the single preference record
        
        organized_prefs = {}
        
        if preferences:
            # Convert the current JSON-based schema to organized format
            if preferences.preferred_cuisines:
                organized_prefs['cuisines'] = [
                    {'name': cuisine, 'value': 'preferred', 'strength': preferences.confidence_score or 0.8}
                    for cuisine in preferences.preferred_cuisines
                ]
            
            if preferences.preferred_districts:
                organized_prefs['districts'] = [
                    {'name': district, 'value': 'preferred', 'strength': preferences.confidence_score or 0.8}
                    for district in preferences.preferred_districts
                ]
            
            if preferences.interests:
                organized_prefs['interests'] = [
                    {'name': interest, 'value': 'interested', 'strength': preferences.confidence_score or 0.8}
                    for interest in preferences.interests
                ]
            
            if preferences.budget_level:
                organized_prefs['budget'] = [
                    {'name': preferences.budget_level, 'value': 'preferred', 'strength': preferences.confidence_score or 0.8}
                ]
            
            if preferences.travel_style:
                organized_prefs['travel_style'] = [
                    {'name': preferences.travel_style, 'value': 'preferred', 'strength': preferences.confidence_score or 0.8}
                ]
        
        return organized_prefs
    
    def _get_conversation_context(self, session_id: str, db: Session) -> Optional[Dict]:
        """Get conversation context for a session"""
        context = db.query(ConversationContext).filter(
            ConversationContext.session_id == session_id
        ).first()
        
        if not context:
            return None
        
        return {
            'current_topic': context.current_topic,
            'topics_discussed': context.topics_discussed or [],
            'places_mentioned': context.places_mentioned or [],
            'travel_stage': context.travel_stage,
            'visit_duration': context.visit_duration,
            'travel_style': context.travel_style,
            'last_location_discussed': context.last_location_discussed,
            'current_need': context.current_need,
            'conversation_mood': context.conversation_mood
        }
    
    def _generate_personalization_hints(self, memories: Dict, preferences: Dict, context: Dict, current_query: str) -> Dict[str, str]:
        """Generate hints for personalizing AI responses"""
        hints = {}
        
        # Generate preference-based hints
        if preferences:
            preference_hints = []
            
            # District preferences
            if 'districts' in preferences:
                liked_districts = [p['name'] for p in preferences['districts'] if p['value'] in ['like', 'love']]
                if liked_districts:
                    preference_hints.append(f"User likes these districts: {', '.join(liked_districts)}")
            
            # Food preferences  
            if 'food' in preferences:
                food_prefs = [p['name'] for p in preferences['food'] if p['value'] in ['like', 'love', 'prefer']]
                if food_prefs:
                    preference_hints.append(f"User prefers: {', '.join(food_prefs)}")
            
            # Activity preferences
            if 'activities' in preferences:
                activity_prefs = [p['name'] for p in preferences['activities'] if p['value'] in ['like', 'love']]
                if activity_prefs:
                    preference_hints.append(f"User enjoys: {', '.join(activity_prefs)}")
            
            if preference_hints:
                hints['preferences'] = " | ".join(preference_hints)
        
        # Generate memory-based hints
        if memories:
            memory_hints = []
            
            # Visited places
            if 'visited' in memories:
                visited_places = [m['context'].get('place_name', '') for m in memories['visited'] 
                                if m['value'] in ['visited', 'loved']]
                if visited_places:
                    memory_hints.append(f"Previously discussed/visited: {', '.join(visited_places[:3])}")
            
            if memory_hints:
                hints['memories'] = " | ".join(memory_hints)
        
        # Generate context-based hints
        if context:
            context_hints = []
            
            if context.get('travel_stage'):
                context_hints.append(f"Travel stage: {context['travel_stage']}")
            
            if context.get('travel_style'):
                context_hints.append(f"Travel style: {context['travel_style']}")
            
            if context.get('last_location_discussed'):
                context_hints.append(f"Last location: {context['last_location_discussed']}")
            
            if context_hints:
                hints['context'] = " | ".join(context_hints)
        
        return hints

# Global instance
memory_system = PersonalizedMemorySystem()

def get_personalized_context(session_id: str, current_query: str, db: Session) -> Dict[str, Any]:
    """Get personalized context for AI responses"""
    return memory_system.get_personalized_context(session_id, current_query, db)

def process_conversation_memory(session_id: str, user_message: str, ai_response: str, db: Session) -> Dict[str, Any]:
    """Process conversation and extract/store memories"""
    return memory_system.process_conversation(session_id, user_message, ai_response, db)

def generate_personalized_prompt_enhancement(personalization: Dict[str, Any], base_prompt: str) -> str:
    """Enhance base prompt with personalization context"""
    if not personalization.get('has_history'):
        return base_prompt
    
    personalization_section = "\n\n🧠 PERSONALIZATION CONTEXT:\n"
    
    hints = personalization.get('personalization_hints', {})
    if hints:
        personalization_section += "Use this personal context to enhance your response:\n"
        
        if 'preferences' in hints:
            personalization_section += f"• PREFERENCES: {hints['preferences']}\n"
        
        if 'memories' in hints:
            personalization_section += f"• PREVIOUS DISCUSSIONS: {hints['memories']}\n"
        
        if 'context' in hints:
            personalization_section += f"• CURRENT CONTEXT: {hints['context']}\n"
        
        personalization_section += "\nPersonalization Guidelines:\n"
        personalization_section += "• Reference previous experiences naturally: 'Since you enjoyed Galata Tower, you might also like...'\n"
        personalization_section += "• Tailor recommendations to their preferences and travel style\n"
        personalization_section += "• Build on previous conversations while providing new value\n"
        personalization_section += "• Show continuity: 'Last time we discussed... now let me add...'\n"
    
    return base_prompt + personalization_section
