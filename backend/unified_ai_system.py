#!/usr/bin/env python3
"""
AI Istanbul - Unified AI System
===============================

This module unifies all AI prompt systems and implements persistent context storage
to resolve context continuity failures and prompt system conflicts.

Key Fixes:
1. Unified prompt system (resolves conflicts between multiple prompt systems)
2. Persistent context storage with 48-hour memory retention
3. Seamless conversation continuity across sessions
4. Database-backed memory management
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

# Import all the conflicting prompt systems to unify them
try:
    from enhanced_gpt_prompts import EnhancedGPTPromptsSystem, PromptCategory
    ENHANCED_PROMPTS_AVAILABLE = True
except ImportError:
    ENHANCED_PROMPTS_AVAILABLE = False
    # Create dummy classes
    class PromptCategory:
        GENERAL = "general"
        RESTAURANT_SPECIFIC = "restaurant_specific"
        MUSEUM_ADVICE = "museum_advice"
        TRANSPORTATION = "transportation"
        SAFETY_PRACTICAL = "safety_practical"
        DAILY_TALK = "daily_talk"

try:
    from enhanced_chatbot import EnhancedContextManager
    ENHANCED_CHATBOT_AVAILABLE = True
except ImportError:
    ENHANCED_CHATBOT_AVAILABLE = False

try:
    from personalized_memory import PersonalizedMemorySystem, get_personalized_context, process_conversation_memory
    PERSONALIZED_MEMORY_AVAILABLE = True
except ImportError:
    PERSONALIZED_MEMORY_AVAILABLE = False

# Import database models
from models import (
    UserSession, ConversationContext, UserMemory, UserPreference, 
    ChatSession, ChatHistory, EnhancedChatHistory
)

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single conversation turn with context"""
    session_id: str
    user_message: str
    ai_response: str
    intent: str
    entities: Dict[str, Any]
    timestamp: datetime
    context_data: Dict[str, Any]

class UnifiedContextManager:
    """Unified context manager that handles persistent conversation memory"""
    
    def __init__(self, db: Session):
        self.db = db
        self.memory_retention_hours = 48  # 48-hour context retention
        
        # Initialize subsystems if available
        self.personalized_memory = PersonalizedMemorySystem() if PERSONALIZED_MEMORY_AVAILABLE else None
        self.enhanced_context = EnhancedContextManager() if ENHANCED_CHATBOT_AVAILABLE else None
    
    def get_or_create_persistent_session(self, session_id: Optional[str] = None, 
                                       user_ip: Optional[str] = None,
                                       user_agent: Optional[str] = None) -> str:
        """Get or create a persistent session with database storage"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Check if session exists and is active
        session = self.db.query(UserSession).filter(
            UserSession.session_id == session_id,
            UserSession.is_active == True
        ).first()
        
        if not session:
            # Create new persistent session
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
            logger.info(f"Created new persistent session: {session_id}")
        else:
            # Update last activity
            session.last_activity = datetime.utcnow()
            self.db.commit()
            logger.debug(f"Updated session activity: {session_id}")
        
        return session_id
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for context continuity"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.memory_retention_hours)
        
        # Get recent chat history from enhanced chat history table
        history = self.db.query(EnhancedChatHistory).filter(
            and_(
                EnhancedChatHistory.session_id == session_id,
                EnhancedChatHistory.timestamp >= cutoff_time
            )
        ).order_by(desc(EnhancedChatHistory.timestamp)).limit(limit).all()
        
        conversation_turns = []
        for entry in reversed(history):  # Chronological order
            conversation_turns.append({
                'user_message': entry.user_message,
                'ai_response': entry.bot_response,
                'intent': entry.detected_intent or 'general',
                'entities': entry.extracted_entities or {},
                'timestamp': entry.timestamp,
                'context_data': entry.conversation_context_snapshot or {}
            })
        
        return conversation_turns
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session context including conversation history and memories"""
        # Get conversation history
        conversation_history = self.get_conversation_history(session_id)
        
        # Get conversation context from database
        context = self.db.query(ConversationContext).filter(
            ConversationContext.session_id == session_id
        ).order_by(desc(ConversationContext.updated_at)).first()
        
        context_data = {}
        if context:
            context_data = {
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
        
        # Get personalized context if available
        personalized_context = {}
        if self.personalized_memory and PERSONALIZED_MEMORY_AVAILABLE:
            try:
                personalized_context = get_personalized_context(session_id, "", self.db)
            except Exception as e:
                logger.warning(f"Failed to get personalized context: {e}")
        
        return {
            'session_id': session_id,
            'conversation_history': conversation_history,
            'context_data': context_data,
            'personalized_context': personalized_context,
            'has_conversation_history': len(conversation_history) > 0,
            'conversation_turns': len(conversation_history)
        }
    
    def store_conversation_turn(self, session_id: str, user_message: str, 
                              ai_response: str, intent: str = "general",
                              entities: Optional[Dict[str, Any]] = None,
                              context_data: Optional[Dict[str, Any]] = None,
                              user_ip: Optional[str] = None) -> bool:
        """Store conversation turn with persistent memory"""
        try:
            # Store in enhanced chat history
            chat_entry = EnhancedChatHistory(
                session_id=session_id,
                user_message=user_message,
                bot_response=ai_response,
                detected_intent=intent,
                extracted_entities=entities or {},
                conversation_context_snapshot=context_data or {},
                timestamp=datetime.utcnow(),
                processing_time_ms=0,  # Not tracking response time here
                user_ip=user_ip
            )
            self.db.add(chat_entry)
            
            # Update conversation context
            self._update_conversation_context(session_id, user_message, ai_response, intent, entities)
            
            # Process memory if personalized memory is available
            if self.personalized_memory and PERSONALIZED_MEMORY_AVAILABLE:
                try:
                    memory_result = process_conversation_memory(session_id, user_message, ai_response, self.db)
                    logger.debug(f"Memory processing result: {memory_result}")
                except Exception as e:
                    logger.warning(f"Memory processing failed: {e}")
            
            self.db.commit()
            logger.info(f"Stored conversation turn for session {session_id}")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to store conversation turn: {e}")
            return False
    
    def _update_conversation_context(self, session_id: str, user_message: str, 
                                   ai_response: str, intent: str,
                                   entities: Optional[Dict[str, Any]] = None):
        """Update conversation context in database"""
        context = self.db.query(ConversationContext).filter(
            ConversationContext.session_id == session_id
        ).first()
        
        if not context:
            context = ConversationContext(
                session_id=session_id,
                current_need=intent,
                current_topic="",
                topics_discussed=[],
                places_mentioned=[],
                travel_stage="exploring",
                last_location_discussed="",
                conversation_mood="active"
            )
            self.db.add(context)
        else:
            # Update existing context
            context.current_need = intent
            context.updated_at = datetime.utcnow()
            
            # Extract and update topics
            current_topic = self._extract_topic_from_message(user_message)
            if current_topic and current_topic != context.current_topic:
                if context.current_topic:
                    topics_discussed = context.topics_discussed or []
                    if context.current_topic not in topics_discussed:
                        topics_discussed.append(context.current_topic)
                    context.topics_discussed = topics_discussed
                context.current_topic = current_topic
            
            # Extract and update locations
            locations = self._extract_locations_from_message(user_message)
            if locations:
                places_mentioned = context.places_mentioned or []
                for location in locations:
                    if location not in places_mentioned:
                        places_mentioned.append(location)
                context.places_mentioned = places_mentioned
                context.last_location_discussed = locations[0]  # Most recent location
    
    def _should_expect_followup(self, user_message: str, ai_response: str) -> bool:
        """Determine if a followup question is likely"""
        followup_indicators = [
            'would you like to know more',
            'would you like details',
            'any specific',
            'which one',
            'tell me more',
            'other options',
            'alternatives'
        ]
        
        user_indicators = [
            'and also',
            'what about',
            'tell me about',
            'i also want',
            'besides that'
        ]
        
        return (any(indicator in ai_response.lower() for indicator in followup_indicators) or
                any(indicator in user_message.lower() for indicator in user_indicators))
    
    def _extract_topic_from_message(self, message: str) -> Optional[str]:
        """Extract main topic from user message"""
        message_lower = message.lower()
        
        topics = {
            'restaurants': ['restaurant', 'food', 'eat', 'dining', 'cuisine'],
            'museums': ['museum', 'gallery', 'art', 'exhibition', 'cultural'],
            'transportation': ['transport', 'metro', 'bus', 'taxi', 'ferry', 'travel'],
            'accommodation': ['hotel', 'stay', 'accommodation', 'booking'],
            'shopping': ['shopping', 'market', 'bazaar', 'buy', 'souvenir'],
            'attractions': ['attraction', 'sightseeing', 'visit', 'see', 'landmark']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in message_lower for keyword in keywords):
                return topic
        
        return None
    
    def _extract_locations_from_message(self, message: str) -> List[str]:
        """Extract Istanbul locations from user message"""
        message_lower = message.lower()
        
        locations = [
            'sultanahmet', 'beyoglu', 'taksim', 'galata', 'kadikoy', 'besiktas',
            'eminonu', 'karakoy', 'ortakoy', 'balat', 'fener', 'uskudar',
            'bebek', 'arnavutkoy', 'fatih', 'sisli', 'levent', 'maslak'
        ]
        
        found_locations = []
        for location in locations:
            if location in message_lower:
                found_locations.append(location.title())
        
        return found_locations
    
    def cleanup_old_contexts(self, days_old: int = 2) -> int:
        """Clean up old conversation contexts (keep 48 hours)"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Delete old chat history
        deleted_history = self.db.query(EnhancedChatHistory).filter(
            EnhancedChatHistory.timestamp < cutoff_date
        ).delete()
        
        # Mark old sessions as inactive
        updated_sessions = self.db.query(UserSession).filter(
            UserSession.last_activity < cutoff_date
        ).update({'is_active': False})
        
        self.db.commit()
        logger.info(f"Cleaned up {deleted_history} old chat entries and {updated_sessions} old sessions")
        
        return deleted_history + updated_sessions

class UnifiedPromptSystem:
    """Unified prompt system that resolves conflicts between multiple prompt systems"""
    
    def __init__(self):
        # Initialize enhanced prompts system if available
        self.enhanced_prompts = EnhancedGPTPromptsSystem() if ENHANCED_PROMPTS_AVAILABLE else None
        
        # Track which prompt system to use
        self.use_enhanced_prompts = ENHANCED_PROMPTS_AVAILABLE
        
        logger.info(f"Unified prompt system initialized - Enhanced prompts: {self.use_enhanced_prompts}")
    
    def get_unified_prompt(self, user_input: str, session_context: Dict[str, Any],
                          location_context: Optional[str] = None) -> Tuple[str, int, float, str]:
        """Get unified prompt resolving all system conflicts"""
        
        if self.use_enhanced_prompts and self.enhanced_prompts:
            # Use enhanced prompts system as primary
            try:
                category = self.enhanced_prompts.detect_category_from_query(user_input)
                system_prompt, max_tokens, temperature = self.enhanced_prompts.get_enhanced_prompt(
                    category, user_input, location_context
                )
                
                # Enhance with conversation context
                if session_context.get('has_conversation_history'):
                    context_enhancement = self._build_context_enhancement(session_context)
                    system_prompt = f"{system_prompt}\n\n{context_enhancement}"
                
                return system_prompt, max_tokens, temperature, category.value
                
            except Exception as e:
                logger.warning(f"Enhanced prompts failed, falling back to legacy: {e}")
                return self._get_fallback_prompt(user_input, session_context, location_context)
        else:
            return self._get_fallback_prompt(user_input, session_context, location_context)
    
    def _build_context_enhancement(self, session_context: Dict[str, Any]) -> str:
        """Build context enhancement from conversation history"""
        conversation_history = session_context.get('conversation_history', [])
        context_data = session_context.get('context_data', {})
        
        if not conversation_history:
            return ""
        
        context_enhancement = "\n🧠 CONVERSATION CONTEXT (Use this to provide continuity):\n"
        
        # Add conversation history
        if len(conversation_history) > 0:
            context_enhancement += "RECENT CONVERSATION:\n"
            for i, turn in enumerate(conversation_history[-3:], 1):  # Last 3 turns
                context_enhancement += f"{i}. User: {turn['user_message'][:100]}...\n"
                context_enhancement += f"   AI: {turn['ai_response'][:100]}...\n"
        
        # Add context data
        if context_data:
            context_enhancement += "\nCONVERSATION CONTEXT:\n"
            if context_data.get('current_topic'):
                context_enhancement += f"- Current topic: {context_data['current_topic']}\n"
            if context_data.get('places_mentioned'):
                context_enhancement += f"- Places discussed: {', '.join(context_data['places_mentioned'][-3:])}\n"
            if context_data.get('travel_stage'):
                context_enhancement += f"- Travel stage: {context_data['travel_stage']}\n"

        
        context_enhancement += "\nIMPORTANT: Use this context to provide continuity. Reference previous conversations naturally and build upon what was already discussed. Don't repeat information unless specifically asked.\n"
        
        return context_enhancement
    
    def _get_fallback_prompt(self, user_input: str, session_context: Dict[str, Any],
                           location_context: Optional[str] = None) -> Tuple[str, int, float, str]:
        """Fallback prompt system when enhanced prompts are not available"""
        
        # Build context enhancement
        context_enhancement = ""
        if session_context.get('has_conversation_history'):
            context_enhancement = self._build_context_enhancement(session_context)
        
        # Location focus
        location_focus = ""
        if location_context:
            location_focus = f"\n\nSPECIAL LOCATION FOCUS: The user is asking specifically about {location_context.title()}. Make sure your entire response is focused on this area with specific local details, walking distances to landmarks, and practical information for that neighborhood."
        
        system_prompt = f"""You are an expert Istanbul travel assistant with deep knowledge of the city. Provide comprehensive, informative responses about Istanbul tourism, culture, attractions, restaurants, and travel tips.

CRITICAL RULES:
1. LOCATION FOCUS: Only provide information about ISTANBUL, Turkey. If asked about other cities, redirect to Istanbul.
2. NO PRICING: Never include specific prices, costs, or monetary amounts. Use terms like "affordable", "moderate", "upscale".
3. NO CURRENCY: Avoid all currency symbols or specific cost amounts.
4. DIRECT RELEVANCE: Answer exactly what the user asks - be specific to their query.
5. CONVERSATION CONTINUITY: Build upon previous conversations naturally without repeating information unless asked.

Guidelines:
- Give DIRECT, HELPFUL answers
- Include specific names of places, attractions, districts IN ISTANBUL
- Provide practical information: hours, locations, transportation details
- Be enthusiastic but informative (300-600 words)
- Use conversation context to provide continuity and avoid repetition

Key Istanbul areas to reference when relevant:
- Sultanahmet (historic district), Beyoğlu (modern area), Kadıköy (Asian side)
- Galata (trendy area), Bosphorus (bridges, ferries), Transportation (metro, tram, ferry){location_focus}{context_enhancement}"""
        
        return system_prompt, 500, 0.7, "general"

class UnifiedAISystem:
    """Main unified AI system that resolves all conflicts and implements persistent memory"""
    
    def __init__(self, db: Session):
        self.db = db
        self.context_manager = UnifiedContextManager(db)
        self.prompt_system = UnifiedPromptSystem()
        
        logger.info("Unified AI system initialized successfully")
    
    async def generate_response(self, user_input: str, session_id: Optional[str] = None,
                              user_ip: Optional[str] = None, user_agent: Optional[str] = None) -> Dict[str, Any]:
        """Generate AI response with unified system and persistent context"""
        
        # Ensure persistent session
        session_id = self.context_manager.get_or_create_persistent_session(
            session_id, user_ip, user_agent
        )
        
        # Get comprehensive session context
        session_context = self.context_manager.get_session_context(session_id)
        
        # Extract location context (simplified)
        location_context = self._extract_location_context(user_input)
        
        # Get unified prompt
        system_prompt, max_tokens, temperature, category = self.prompt_system.get_unified_prompt(
            user_input, session_context, location_context
        )
        
        # Generate response with OpenAI
        try:
            from openai import OpenAI
            import os
            
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise Exception("OpenAI API key not found")
            
            client = OpenAI(api_key=openai_api_key, timeout=45.0, max_retries=3)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Store conversation turn for persistent memory
            success = self.context_manager.store_conversation_turn(
                session_id=session_id,
                user_message=user_input,
                ai_response=ai_response,
                intent=category,
                entities={},
                context_data={'location_context': location_context},
                user_ip=user_ip
            )
            
            logger.info(f"Generated response for session {session_id}, memory stored: {success}")
            
            return {
                'success': True,
                'response': ai_response,
                'session_id': session_id,
                'category': category,
                'has_context': session_context.get('has_conversation_history', False),
                'conversation_turns': session_context.get('conversation_turns', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id
            }
    
    def _extract_location_context(self, user_input: str) -> Optional[str]:
        """Extract location context from user input"""
        user_lower = user_input.lower()
        
        locations = {
            'sultanahmet': ['sultanahmet', 'blue mosque', 'hagia sophia', 'topkapi'],
            'beyoglu': ['beyoglu', 'taksim', 'istiklal', 'galata tower'],
            'kadikoy': ['kadikoy', 'moda', 'asian side'],
            'galata': ['galata', 'karakoy'],
            'eminonu': ['eminonu', 'spice bazaar', 'grand bazaar'],
            'besiktas': ['besiktas', 'dolmabahce'],
            'uskudar': ['uskudar', 'maiden tower']
        }
        
        for location, keywords in locations.items():
            if any(keyword in user_lower for keyword in keywords):
                return location
        
        return None
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session information for debugging"""
        session_context = self.context_manager.get_session_context(session_id)
        
        return {
            'session_id': session_id,
            'has_context': session_context.get('has_conversation_history', False),
            'conversation_turns': len(session_context.get('conversation_history', [])),
            'current_topic': session_context.get('context_data', {}).get('current_topic'),
            'places_mentioned': session_context.get('context_data', {}).get('places_mentioned', []),
            'last_activity': datetime.utcnow().isoformat()
        }
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """Cleanup old conversation data"""
        cleaned_contexts = self.context_manager.cleanup_old_contexts()
        
        return {
            'cleaned_entries': cleaned_contexts,
            'retention_hours': self.context_manager.memory_retention_hours
        }

# Global instance (will be initialized by main.py)
unified_ai_system: Optional[UnifiedAISystem] = None

def get_unified_ai_system(db: Session) -> UnifiedAISystem:
    """Get or create the global unified AI system instance"""
    global unified_ai_system
    if unified_ai_system is None:
        unified_ai_system = UnifiedAISystem(db)
    return unified_ai_system

def is_followup_question(user_input: str, session_context: Dict[str, Any]) -> bool:
    """Check if user input is a followup question"""
    if not session_context.get('has_conversation_history'):
        return False
    
    followup_indicators = [
        'what about', 'how about', 'and', 'also', 'besides', 'additionally',
        'more details', 'tell me more', 'continue', 'go on', 'next',
        'other options', 'alternatives', 'similar', 'like that',
        'there', 'it', 'this', 'that', 'they', 'them'  # Pronouns indicating reference
    ]
    
    user_lower = user_input.lower()
    return any(indicator in user_lower for indicator in followup_indicators)
