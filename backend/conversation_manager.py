#!/usr/bin/env python3
"""
Advanced Conversation State Manager
Handles multi-turn conversations with anaphora resolution and context tracking
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ConversationState(Enum):
    """Conversation state types"""
    INITIAL = "initial"
    LOCATION_DISCUSSION = "location_discussion"
    RESTAURANT_SEARCH = "restaurant_search"
    MUSEUM_PLANNING = "museum_planning"
    TRANSPORTATION_QUERY = "transportation_query"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"

@dataclass
class ConversationEntity:
    """Entities mentioned in conversation"""
    entity_type: str  # location, restaurant, museum, etc.
    value: str
    confidence: float
    mentioned_turn: int
    last_referenced: int = 0
    aliases: List[str] = field(default_factory=list)

@dataclass
class ConversationTurn:
    """Individual conversation turn"""
    turn_id: int
    user_message: str
    ai_response: str
    timestamp: datetime
    state: ConversationState
    entities_mentioned: List[ConversationEntity]
    resolved_references: Dict[str, str]  # pronoun -> resolved entity
    intent: str
    confidence: float

class AdvancedConversationManager:
    """Manages multi-turn conversations with anaphora resolution"""
    
    def __init__(self):
        self.conversation_history: Dict[str, List[ConversationTurn]] = {}
        self.entity_tracker: Dict[str, Dict[str, ConversationEntity]] = {}
        self.state_tracker: Dict[str, ConversationState] = {}
        
        # Anaphora resolution patterns
        self.anaphoric_patterns = {
            'location_references': [
                r'\bthere\b', r'\bit\b', r'\bthat place\b', r'\bthis place\b',
                r'\bthe area\b', r'\bthe location\b', r'\bthe district\b'
            ],
            'establishment_references': [
                r'\bthat restaurant\b', r'\bthis restaurant\b', r'\bthe place\b',
                r'\bthat museum\b', r'\bthis museum\b', r'\bthe venue\b'
            ],
            'general_references': [
                r'\bit\b', r'\bthat\b', r'\bthis\b', r'\bthey\b', r'\bthem\b'
            ]
        }
        
        # Intent classification patterns
        self.intent_patterns = {
            'transportation_query': [
                r'how do i get', r'how to get', r'how can i reach', r'directions to',
                r'transport to', r'travel to', r'go to', r'reach', r'arrive at'
            ],
            'hours_query': [
                r'opening hours', r'open time', r'closing time', r'when.*open',
                r'what time.*open', r'hours', r'schedule'
            ],
            'price_query': [
                r'how much', r'price', r'cost', r'expensive', r'cheap', r'budget'
            ],
            'recommendation_query': [
                r'recommend', r'suggest', r'best', r'good', r'worth visiting'
            ]
        }
        
    def process_message(self, session_id: str, user_message: str, 
                       ai_response: str) -> Dict[str, Any]:
        """Process a conversation turn with anaphora resolution"""
        
        # Initialize session if new
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
            self.entity_tracker[session_id] = {}
            self.state_tracker[session_id] = ConversationState.INITIAL
        
        turn_id = len(self.conversation_history[session_id]) + 1
        current_state = self._determine_conversation_state(session_id, user_message)
        
        # Extract entities from current message
        entities = self._extract_entities(user_message, turn_id)
        
        # Resolve anaphoric references
        resolved_refs = self._resolve_anaphora(session_id, user_message)
        
        # Classify intent
        intent, confidence = self._classify_intent(user_message, resolved_refs)
        
        # Create conversation turn
        turn = ConversationTurn(
            turn_id=turn_id,
            user_message=user_message,
            ai_response=ai_response,
            timestamp=datetime.now(),
            state=current_state,
            entities_mentioned=entities,
            resolved_references=resolved_refs,
            intent=intent,
            confidence=confidence
        )
        
        # Update conversation history
        self.conversation_history[session_id].append(turn)
        self.state_tracker[session_id] = current_state
        
        # Update entity tracker
        self._update_entity_tracker(session_id, entities, turn_id)
        
        logger.info(f"Processed turn {turn_id} for session {session_id}: "
                   f"State={current_state.value}, Intent={intent}, "
                   f"Resolved={len(resolved_refs)} references")
        
        return {
            'turn_id': turn_id,
            'conversation_state': current_state.value,
            'intent': intent,
            'confidence': confidence,
            'resolved_references': resolved_refs,
            'entities_mentioned': [e.value for e in entities],
            'context_summary': self._generate_context_summary(session_id)
        }
    
    def _determine_conversation_state(self, session_id: str, 
                                    user_message: str) -> ConversationState:
        """Determine current conversation state"""
        
        # Check for follow-up patterns
        if self._is_follow_up_query(user_message):
            return ConversationState.FOLLOW_UP
        
        # Analyze message content for state
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['restaurant', 'eat', 'food', 'dining']):
            return ConversationState.RESTAURANT_SEARCH
        elif any(word in message_lower for word in ['museum', 'attraction', 'visit', 'see']):
            return ConversationState.MUSEUM_PLANNING
        elif any(word in message_lower for word in ['get', 'transport', 'metro', 'bus', 'taxi']):
            return ConversationState.TRANSPORTATION_QUERY
        elif any(word in message_lower for word in ['sultanahmet', 'galata', 'taksim', 'kadikoy']):
            return ConversationState.LOCATION_DISCUSSION
        else:
            return ConversationState.INITIAL
    
    def _is_follow_up_query(self, message: str) -> bool:
        """Check if message is a follow-up query"""
        follow_up_patterns = [
            r'^how do i get there\??$',
            r'^how to get there\??$',
            r'^directions\??$',
            r'^opening hours\??$',
            r'^how much\??$',
            r'^what about.*\??$',
            r'^and.*\??$',
            r'^what time\??$',
            r'^when\??$',
            r'^where\??$'
        ]
        
        message_clean = message.lower().strip()
        return any(re.match(pattern, message_clean) for pattern in follow_up_patterns)
    
    def _extract_entities(self, message: str, turn_id: int) -> List[ConversationEntity]:
        """Extract entities from message"""
        entities = []
        
        # Location entities
        location_patterns = {
            'sultanahmet': ['sultanahmet', 'old city', 'historic peninsula'],
            'galata': ['galata', 'galata tower', 'karakoy'],
            'taksim': ['taksim', 'taksim square', 'beyoglu'],
            'kadikoy': ['kadikoy', 'asian side'],
            'besiktas': ['besiktas', 'dolmabahce']
        }
        
        message_lower = message.lower()
        for location, variants in location_patterns.items():
            if any(variant in message_lower for variant in variants):
                entities.append(ConversationEntity(
                    entity_type='location',
                    value=location,
                    confidence=0.9,
                    mentioned_turn=turn_id,
                    aliases=variants
                ))
        
        # Restaurant/establishment entities
        establishment_keywords = ['restaurant', 'cafe', 'museum', 'palace', 'mosque']
        for keyword in establishment_keywords:
            if keyword in message_lower:
                entities.append(ConversationEntity(
                    entity_type='establishment',
                    value=keyword,
                    confidence=0.8,
                    mentioned_turn=turn_id
                ))
        
        return entities
    
    def _resolve_anaphora(self, session_id: str, message: str) -> Dict[str, str]:
        """Resolve anaphoric references in message"""
        resolved = {}
        
        if session_id not in self.conversation_history:
            return resolved
        
        # Get recent entities (last 3 turns)
        recent_entities = self._get_recent_entities(session_id, max_turns=3)
        
        # Resolve location references
        for pattern in self.anaphoric_patterns['location_references']:
            if re.search(pattern, message, re.IGNORECASE):
                location_entities = [e for e in recent_entities if e.entity_type == 'location']
                if location_entities:
                    # Use most recently mentioned location
                    latest_location = max(location_entities, key=lambda x: x.last_referenced)
                    resolved[pattern] = latest_location.value
                    logger.debug(f"Resolved '{pattern}' to location: {latest_location.value}")
        
        # Resolve establishment references  
        for pattern in self.anaphoric_patterns['establishment_references']:
            if re.search(pattern, message, re.IGNORECASE):
                establishment_entities = [e for e in recent_entities if e.entity_type == 'establishment']
                if establishment_entities:
                    latest_establishment = max(establishment_entities, key=lambda x: x.last_referenced)
                    resolved[pattern] = latest_establishment.value
                    logger.debug(f"Resolved '{pattern}' to establishment: {latest_establishment.value}")
        
        return resolved
    
    def _get_recent_entities(self, session_id: str, max_turns: int = 3) -> List[ConversationEntity]:
        """Get entities mentioned in recent conversation turns"""
        if session_id not in self.entity_tracker:
            return []
        
        recent_entities = []
        for entity in self.entity_tracker[session_id].values():
            # Include entities mentioned in last max_turns
            current_turn = len(self.conversation_history[session_id])
            if current_turn - entity.last_referenced <= max_turns:
                recent_entities.append(entity)
        
        return recent_entities
    
    def _classify_intent(self, message: str, resolved_refs: Dict[str, str]) -> Tuple[str, float]:
        """Classify the intent of the message"""
        message_lower = message.lower()
        
        # Check for resolved context
        context_aware_message = message_lower
        for ref, resolution in resolved_refs.items():
            context_aware_message += f" {resolution}"
        
        # Find matching intent patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, context_aware_message):
                    confidence = 0.9 if len(resolved_refs) > 0 else 0.7
                    return intent, confidence
        
        return 'general_query', 0.5
    
    def _update_entity_tracker(self, session_id: str, entities: List[ConversationEntity], 
                             turn_id: int):
        """Update entity tracker with new entities"""
        for entity in entities:
            key = f"{entity.entity_type}:{entity.value}"
            
            if key in self.entity_tracker[session_id]:
                # Update existing entity
                self.entity_tracker[session_id][key].last_referenced = turn_id
            else:
                # Add new entity
                entity.last_referenced = turn_id
                self.entity_tracker[session_id][key] = entity
    
    def _generate_context_summary(self, session_id: str) -> str:
        """Generate a summary of current conversation context"""
        if session_id not in self.conversation_history:
            return ""
        
        recent_entities = self._get_recent_entities(session_id, max_turns=5)
        current_state = self.state_tracker.get(session_id, ConversationState.INITIAL)
        
        # Build context summary
        context_parts = []
        
        # Add state context
        if current_state != ConversationState.INITIAL:
            context_parts.append(f"Current focus: {current_state.value.replace('_', ' ')}")
        
        # Add location context
        locations = [e.value for e in recent_entities if e.entity_type == 'location']
        if locations:
            context_parts.append(f"Discussing locations: {', '.join(set(locations))}")
        
        # Add establishment context
        establishments = [e.value for e in recent_entities if e.entity_type == 'establishment']
        if establishments:
            context_parts.append(f"Topics: {', '.join(set(establishments))}")
        
        return " | ".join(context_parts)
    
    def get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation context"""
        if session_id not in self.conversation_history:
            return {}
        
        history = self.conversation_history[session_id]
        recent_entities = self._get_recent_entities(session_id)
        
        return {
            'session_id': session_id,
            'turn_count': len(history),
            'current_state': self.state_tracker.get(session_id, ConversationState.INITIAL).value,
            'recent_entities': [
                {
                    'type': e.entity_type,
                    'value': e.value,
                    'last_mentioned': e.last_referenced
                }
                for e in recent_entities
            ],
            'last_intent': history[-1].intent if history else None,
            'context_summary': self._generate_context_summary(session_id),
            'recent_turns': [
                {
                    'turn_id': turn.turn_id,
                    'user_message': turn.user_message[:100] + '...' if len(turn.user_message) > 100 else turn.user_message,
                    'intent': turn.intent,
                    'resolved_refs': turn.resolved_references
                }
                for turn in history[-3:]  # Last 3 turns
            ]
        }
    
    def resolve_follow_up_query(self, session_id: str, follow_up_message: str) -> Dict[str, Any]:
        """Resolve a follow-up query with full context"""
        
        # Process the message normally
        context = self.process_message(session_id, follow_up_message, "")
        
        # Get enhanced resolution information
        resolved_refs = context['resolved_references']
        recent_entities = self._get_recent_entities(session_id)
        
        # Build enhanced context for AI system
        enhanced_context = {
            'original_query': follow_up_message,
            'resolved_query': self._build_resolved_query(follow_up_message, resolved_refs),
            'conversation_context': self._generate_context_summary(session_id),
            'intent': context['intent'],
            'entities_in_scope': [
                {'type': e.entity_type, 'value': e.value}
                for e in recent_entities
            ],
            'suggested_response_type': self._suggest_response_type(context['intent'], resolved_refs)
        }
        
        return enhanced_context
    
    def _build_resolved_query(self, original_query: str, resolved_refs: Dict[str, str]) -> str:
        """Build a resolved query with anaphora replaced"""
        resolved_query = original_query
        
        for ref_pattern, resolution in resolved_refs.items():
            resolved_query = re.sub(ref_pattern, resolution, resolved_query, flags=re.IGNORECASE)
        
        return resolved_query
    
    def _suggest_response_type(self, intent: str, resolved_refs: Dict[str, str]) -> str:
        """Suggest the type of response needed"""
        if intent == 'transportation_query' and resolved_refs:
            return 'detailed_directions'
        elif intent == 'hours_query' and resolved_refs:
            return 'opening_hours_info'
        elif intent == 'price_query' and resolved_refs:
            return 'pricing_information'
        elif intent == 'recommendation_query':
            return 'personalized_recommendations'
        else:
            return 'contextual_information'

    async def process_conversation_turn(self, session_id: str, user_message: str, 
                                       user_ip: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Process a conversation turn with anaphora resolution and return resolved query
        Returns: (resolved_query, conversation_context)
        """
        try:
            # Initialize session if new
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
                self.entity_tracker[session_id] = {}
                self.state_tracker[session_id] = ConversationState.INITIAL
            
            # Resolve anaphoric references first (this will work if we have conversation history)
            resolved_query = self._resolve_query_anaphora(session_id, user_message)
            
            # Determine conversation state
            current_state = self._determine_conversation_state(session_id, resolved_query)
            
            # Extract entities from resolved query
            entities = self._extract_entities(resolved_query, len(self.conversation_history[session_id]) + 1)
            
            # Classify intent
            intent, confidence = self._classify_intent(resolved_query, {})
            
            # Check if anaphora resolution actually happened
            has_anaphora_resolution = (resolved_query.lower().strip() != user_message.lower().strip() and 
                                     len(self.conversation_history[session_id]) > 0)
            
            # Build conversation context
            conversation_context = {
                'conversation_state': current_state.value,
                'intent': intent,
                'confidence': confidence,
                'entities': {entity.entity_type: entity.value for entity in entities},
                'resolved_references': has_anaphora_resolution,
                'context_summary': self._generate_context_summary(session_id),
                'session_turn': len(self.conversation_history[session_id]) + 1
            }
            
            if has_anaphora_resolution:
                logger.info(f"🧠 ✅ Anaphora resolved for {session_id}: "
                           f"'{user_message}' → '{resolved_query}' "
                           f"(State: {current_state.value})")
            else:
                logger.info(f"🧠 No anaphora resolution needed for {session_id}: "
                           f"'{user_message}' (State: {current_state.value})")
            
            return resolved_query, conversation_context
            
        except Exception as e:
            logger.error(f"❌ Error processing conversation turn: {e}")
            return user_message, {'error': str(e)}
    
    async def update_conversation_state(self, session_id: str, ai_response: str, 
                                      resolved_entities: Dict[str, Any]) -> bool:
        """Update conversation state after AI response"""
        try:
            if session_id not in self.conversation_history:
                return False
            
            # Get the last turn (most recent)
            if self.conversation_history[session_id]:
                last_turn = self.conversation_history[session_id][-1]
                last_turn.ai_response = ai_response
                
                # Update entities with resolved information
                for entity_type, entity_value in resolved_entities.items():
                    if entity_type not in self.entity_tracker[session_id]:
                        entity = ConversationEntity(
                            entity_type=entity_type,
                            value=entity_value,
                            confidence=0.9,
                            mentioned_turn=last_turn.turn_id,
                            last_referenced=last_turn.turn_id
                        )
                        self.entity_tracker[session_id][f"{entity_type}_{entity_value}"] = entity
                
                logger.debug(f"✅ Updated conversation state for session {session_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error updating conversation state: {e}")
            return False
    
    def _resolve_query_anaphora(self, session_id: str, user_message: str) -> str:
        """Resolve anaphoric references in the query"""
        if session_id not in self.conversation_history or not self.conversation_history[session_id]:
            return user_message
        
        resolved_query = user_message.lower()
        
        # Get recent context (last 3 turns)
        recent_turns = self.conversation_history[session_id][-3:]
        
        # Extract locations, establishments, and topics from recent context
        context_locations = set()
        context_establishments = set()
        context_topics = set()
        
        for turn in recent_turns:
            for entity in turn.entities_mentioned:
                if entity.entity_type == 'location':
                    context_locations.add(entity.value)
                elif entity.entity_type in ['restaurant', 'museum', 'attraction']:
                    context_establishments.add(entity.value)
                elif entity.entity_type == 'topic':
                    context_topics.add(entity.value)
        
        # Resolve location references
        for pattern in self.anaphoric_patterns['location_references']:
            if re.search(pattern, resolved_query, re.IGNORECASE):
                if context_locations:
                    most_recent_location = list(context_locations)[-1]  # Get most recent
                    resolved_query = re.sub(pattern, most_recent_location, resolved_query, flags=re.IGNORECASE)
                    logger.debug(f"🔗 Resolved location reference: '{pattern}' → '{most_recent_location}'")
        
        # Resolve establishment references
        for pattern in self.anaphoric_patterns['establishment_references']:
            if re.search(pattern, resolved_query, re.IGNORECASE):
                if context_establishments:
                    most_recent_establishment = list(context_establishments)[-1]
                    resolved_query = re.sub(pattern, most_recent_establishment, resolved_query, flags=re.IGNORECASE)
                    logger.debug(f"🔗 Resolved establishment reference: '{pattern}' → '{most_recent_establishment}'")
        
        # Handle specific common anaphora cases
        if re.search(r'\bhow do i get there\b', resolved_query, re.IGNORECASE):
            if context_locations:
                location = list(context_locations)[-1]
                resolved_query = f"How do I get to {location}"
        elif re.search(r'\bwhat are the opening hours\b', resolved_query, re.IGNORECASE):
            if context_establishments:
                establishment = list(context_establishments)[-1]
                resolved_query = f"What are the opening hours for {establishment}"
            elif context_locations:
                location = list(context_locations)[-1]
                resolved_query = f"What are the opening hours for attractions in {location}"
        elif re.search(r'\bhow much.*cost\b', resolved_query, re.IGNORECASE):
            if context_establishments:
                establishment = list(context_establishments)[-1]
                resolved_query = f"How much does it cost to visit {establishment}"
        
        return resolved_query

# Global conversation manager instance
_conversation_manager = None

def get_conversation_manager() -> AdvancedConversationManager:
    """Get global conversation manager instance"""
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = AdvancedConversationManager()
    return _conversation_manager

if __name__ == "__main__":
    # Test the conversation manager
    manager = AdvancedConversationManager()
    
    print("🧪 Testing Advanced Conversation Manager...")
    
    # Simulate a multi-turn conversation
    session_id = "test_session_123"
    
    # Turn 1: Initial location inquiry
    context1 = manager.process_message(
        session_id,
        "What restaurants are in Sultanahmet?",
        "Here are some great restaurants in Sultanahmet..."
    )
    print(f"Turn 1 - State: {context1['conversation_state']}, Intent: {context1['intent']}")
    
    # Turn 2: Follow-up with anaphora
    context2 = manager.resolve_follow_up_query(
        session_id,
        "How do I get there?"
    )
    print(f"Turn 2 - Resolved: '{context2['resolved_query']}'")
    print(f"Context: {context2['conversation_context']}")
    
    # Turn 3: Another follow-up
    context3 = manager.resolve_follow_up_query(
        session_id,
        "What are the opening hours?"
    )
    print(f"Turn 3 - Intent: {context3['intent']}")
    
    print("✅ Advanced Conversation Manager test complete!")
