#!/usr/bin/env python3
"""
Conversational Memory & Context System
=====================================

Handles multi-turn conversations and contextual understanding
without GPT dependencies. Uses session-based memory and context tracking.
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    timestamp: str
    user_query: str
    normalized_query: str
    intent: str
    entities: Dict[str, Any]
    response: str
    confidence: float

@dataclass
class UserPreferences:
    """User preferences learned from conversation"""
    preferred_districts: List[str]
    preferred_cuisines: List[str]
    budget_level: Optional[str]
    dietary_restrictions: List[str]
    favorite_vibes: List[str]
    visit_frequency: int
    last_active: str

class ConversationalMemory:
    """Manages conversation context and user preferences"""
    
    def __init__(self, max_turns: int = 10, session_timeout_hours: int = 24):
        self.conversations = {}  # session_id -> List[ConversationTurn]
        self.user_preferences = {}  # session_id -> UserPreferences
        self.max_turns = max_turns
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
        # Context patterns for understanding references
        self.reference_patterns = {
            'that_place': ['that place', 'there', 'that one', 'it'],
            'those_places': ['those places', 'them', 'those ones'],
            'similar': ['similar', 'like that', 'something like', 'same type'],
            'different': ['different', 'something else', 'other options'],
            'nearby': ['near there', 'around there', 'close by', 'in that area'],
            'price_related': ['cheaper', 'more expensive', 'same price', 'budget'],
            'time_related': ['later', 'earlier', 'same time', 'tomorrow']
        }
    
    def add_turn(self, session_id: str, turn: ConversationTurn):
        """Add a conversation turn"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add turn
        self.conversations[session_id].append(turn)
        
        # Keep only recent turns
        if len(self.conversations[session_id]) > self.max_turns:
            self.conversations[session_id] = self.conversations[session_id][-self.max_turns:]
        
        # Update user preferences
        self._update_preferences(session_id, turn)
        
        # Clean old sessions
        self._cleanup_old_sessions()
    
    def get_context(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """Get conversation context for current query"""
        if session_id not in self.conversations:
            return {'has_context': False}
        
        turns = self.conversations[session_id]
        if not turns:
            return {'has_context': False}
        
        # Get last few turns
        recent_turns = turns[-3:]  # Last 3 turns for context
        
        # Analyze current query for references
        references = self._detect_references(current_query)
        
        # Build context
        context = {
            'has_context': True,
            'turn_count': len(turns),
            'recent_intents': [turn.intent for turn in recent_turns],
            'recent_entities': self._merge_entities([turn.entities for turn in recent_turns]),
            'references': references,
            'last_response': turns[-1].response if turns else None,
            'conversation_topic': self._infer_topic(recent_turns),
            'user_preferences': self.user_preferences.get(session_id)
        }
        
        return context
    
    def resolve_references(self, current_entities: Dict[str, Any], 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve references in current query using context"""
        if not context.get('has_context'):
            return current_entities
        
        resolved = current_entities.copy()
        references = context.get('references', [])
        recent_entities = context.get('recent_entities', {})
        
        # If current query has references but missing entities, fill from context
        if references and not resolved.get('districts') and recent_entities.get('districts'):
            resolved['districts'] = recent_entities['districts']
            
        if references and not resolved.get('categories') and recent_entities.get('categories'):
            resolved['categories'] = recent_entities['categories']
            
        if references and not resolved.get('cuisines') and recent_entities.get('cuisines'):
            resolved['cuisines'] = recent_entities['cuisines']
        
        # Handle specific reference types
        if 'nearby' in references:
            # Use location from previous turn
            if recent_entities.get('districts'):
                resolved['districts'] = recent_entities['districts']
        
        if 'similar' in references:
            # Keep same category/cuisine from previous
            for key in ['categories', 'cuisines', 'vibes']:
                if recent_entities.get(key) and not resolved.get(key):
                    resolved[key] = recent_entities[key]
        
        if 'different' in references:
            # Change category but keep location
            if recent_entities.get('districts') and not resolved.get('districts'):
                resolved['districts'] = recent_entities['districts']
        
        return resolved
    
    def _detect_references(self, query: str) -> List[str]:
        """Detect references in query"""
        query_lower = query.lower()
        references = []
        
        for ref_type, patterns in self.reference_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                references.append(ref_type)
        
        return references
    
    def _merge_entities(self, entity_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge entities from multiple turns"""
        merged = {
            'districts': [],
            'categories': [],
            'cuisines': [],
            'vibes': [],
            'temporal': None,
            'budget': None
        }
        
        for entities in entity_list:
            for key in ['districts', 'categories', 'cuisines', 'vibes']:
                if entities.get(key):
                    merged[key].extend(entities[key])
            
            # Take most recent temporal/budget
            if entities.get('temporal'):
                merged['temporal'] = entities['temporal']
            if entities.get('budget'):
                merged['budget'] = entities['budget']
        
        # Remove duplicates
        for key in ['districts', 'categories', 'cuisines', 'vibes']:
            merged[key] = list(set(merged[key]))
        
        return merged
    
    def _infer_topic(self, turns: List[ConversationTurn]) -> str:
        """Infer conversation topic from recent turns"""
        if not turns:
            return 'general'
        
        # Count intents
        intent_counts = defaultdict(int)
        for turn in turns:
            intent_counts[turn.intent] += 1
        
        # Return most common intent
        if intent_counts:
            return max(intent_counts.items(), key=lambda x: x[1])[0]
        
        return 'general'
    
    def _update_preferences(self, session_id: str, turn: ConversationTurn):
        """Update user preferences based on turn"""
        if session_id not in self.user_preferences:
            self.user_preferences[session_id] = UserPreferences(
                preferred_districts=[],
                preferred_cuisines=[],
                budget_level=None,
                dietary_restrictions=[],
                favorite_vibes=[],
                visit_frequency=0,
                last_active=datetime.now().isoformat()
            )
        
        prefs = self.user_preferences[session_id]
        prefs.visit_frequency += 1
        prefs.last_active = datetime.now().isoformat()
        
        # Update preferences based on entities
        entities = turn.entities
        
        if entities.get('districts'):
            for district in entities['districts']:
                if district not in prefs.preferred_districts:
                    prefs.preferred_districts.append(district)
        
        if entities.get('cuisines'):
            for cuisine in entities['cuisines']:
                if cuisine not in prefs.preferred_cuisines:
                    prefs.preferred_cuisines.append(cuisine)
        
        if entities.get('budget'):
            prefs.budget_level = entities['budget']
        
        if entities.get('vibes'):
            for vibe in entities['vibes']:
                if vibe not in prefs.favorite_vibes:
                    prefs.favorite_vibes.append(vibe)
        
        # Detect dietary restrictions
        dietary_keywords = {
            'vegetarian': ['vegetarian', 'veg'],
            'vegan': ['vegan'],
            'halal': ['halal'],
            'gluten-free': ['gluten free', 'gluten-free', 'celiac']
        }
        
        query_lower = turn.user_query.lower()
        for dietary, keywords in dietary_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if dietary not in prefs.dietary_restrictions:
                    prefs.dietary_restrictions.append(dietary)
    
    def _cleanup_old_sessions(self):
        """Remove old sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, turns in self.conversations.items():
            if turns:
                last_turn_time = datetime.fromisoformat(turns[-1].timestamp)
                if current_time - last_turn_time > self.session_timeout:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.conversations[session_id]
            if session_id in self.user_preferences:
                del self.user_preferences[session_id]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of user session"""
        if session_id not in self.conversations:
            return {'exists': False}
        
        turns = self.conversations[session_id]
        prefs = self.user_preferences.get(session_id)
        
        return {
            'exists': True,
            'turn_count': len(turns),
            'session_duration': self._calculate_duration(turns),
            'main_topics': self._get_main_topics(turns),
            'preferences': asdict(prefs) if prefs else None,
            'last_active': turns[-1].timestamp if turns else None
        }
    
    def _calculate_duration(self, turns: List[ConversationTurn]) -> str:
        """Calculate session duration"""
        if len(turns) < 2:
            return "Single turn"
        
        start_time = datetime.fromisoformat(turns[0].timestamp)
        end_time = datetime.fromisoformat(turns[-1].timestamp)
        duration = end_time - start_time
        
        if duration.total_seconds() < 60:
            return f"{int(duration.total_seconds())} seconds"
        elif duration.total_seconds() < 3600:
            return f"{int(duration.total_seconds() / 60)} minutes"
        else:
            return f"{duration.total_seconds() / 3600:.1f} hours"
    
    def _get_main_topics(self, turns: List[ConversationTurn]) -> List[str]:
        """Get main conversation topics"""
        intent_counts = defaultdict(int)
        for turn in turns:
            intent_counts[turn.intent] += 1
        
        # Return top 3 intents
        sorted_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
        return [intent for intent, count in sorted_intents[:3]]

# Global memory instance
conversational_memory = ConversationalMemory()

def create_conversation_turn(user_query: str, normalized_query: str, intent: str,
                           entities: Dict[str, Any], response: str, confidence: float) -> ConversationTurn:
    """Create a conversation turn"""
    return ConversationTurn(
        timestamp=datetime.now().isoformat(),
        user_query=user_query,
        normalized_query=normalized_query,
        intent=intent,
        entities=entities,
        response=response,
        confidence=confidence
    )

def process_with_context(query: str, session_id: str) -> Dict[str, Any]:
    """Process query with conversational context"""
    from enhanced_query_understanding import process_enhanced_query
    
    # Get conversation context
    context = conversational_memory.get_context(session_id, query)
    
    # Process query with enhanced understanding
    query_result = process_enhanced_query(query, session_id)
    
    # Resolve references using context
    if context.get('has_context'):
        resolved_entities = conversational_memory.resolve_references(
            query_result['entities'], context
        )
        query_result['entities'] = resolved_entities
        query_result['context_applied'] = True
        query_result['references_resolved'] = len(context.get('references', []))
    else:
        query_result['context_applied'] = False
        query_result['references_resolved'] = 0
    
    # Add context information
    query_result['conversation_context'] = context
    
    return query_result

if __name__ == "__main__":
    # Test conversational memory
    print("💭 Conversational Memory System Test")
    print("=" * 45)
    
    session_id = "test_conversation"
    
    # Simulate a conversation
    conversation_queries = [
        "Find good Turkish restaurants in Sultanahmet",
        "What about something near there?",  # Reference to Sultanahmet
        "Show me cheaper options",  # Reference to restaurants + budget change
        "Any cafes around that area?",  # Reference to location + category change
        "Something more romantic"  # Reference to previous + vibe change
    ]
    
    for i, query in enumerate(conversation_queries, 1):
        print(f"\n🗣️ Turn {i}: '{query}'")
        print("-" * 30)
        
        result = process_with_context(query, session_id)
        
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"Context Applied: {result['context_applied']}")
        print(f"References Resolved: {result['references_resolved']}")
        
        entities = result['entities']
        if entities['districts']:
            print(f"Districts: {', '.join(entities['districts'])}")
        if entities['categories']:
            print(f"Categories: {', '.join(entities['categories'])}")
        if entities['budget']:
            print(f"Budget: {entities['budget']}")
        if entities['vibes']:
            print(f"Vibes: {', '.join(entities['vibes'])}")
        
        # Create mock response and add to memory
        mock_response = f"Found results for {result['intent']} query"
        turn = create_conversation_turn(
            query, result['normalized_query'], result['intent'],
            entities, mock_response, result['confidence']
        )
        conversational_memory.add_turn(session_id, turn)
    
    # Show session summary
    print(f"\n📊 Session Summary")
    print("-" * 20)
    summary = conversational_memory.get_session_summary(session_id)
    print(f"Turns: {summary['turn_count']}")
    print(f"Duration: {summary['session_duration']}")
    print(f"Main Topics: {', '.join(summary['main_topics'])}")
    
    if summary['preferences']:
        prefs = summary['preferences']
        print(f"Preferred Districts: {', '.join(prefs['preferred_districts'])}")
        print(f"Visit Frequency: {prefs['visit_frequency']}")
        if prefs['favorite_vibes']:
            print(f"Favorite Vibes: {', '.join(prefs['favorite_vibes'])}")
