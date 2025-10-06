#!/usr/bin/env python3
"""
Multi-Turn Query Handler for AI Istanbul
========================================

Implements sophisticated multi-turn conversation capabilities with:
- Conversation stack and history buffer
- Pronoun and reference disambiguation  
- Context-aware query resolution
- Entity tracking across turns
- Follow-up query handling

Example conversations:
- "Best cafes near Hagia Sophia?" → Returns list
- "Open now?" → Knows context → checks opening hours of those cafes  
- "Which one is cheapest?" → Uses session entities → ranks previous results by price
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Single conversation turn with query, response, and extracted data"""
    turn_id: str
    user_query: str
    ai_response: str
    intent: str
    entities: Dict[str, List[str]]
    extracted_places: List[Dict[str, Any]]  # Detailed place data from response
    extracted_results: Dict[str, Any]  # Structured results (restaurants, cafes, etc.)
    timestamp: str
    response_type: str  # 'list', 'info', 'directions', etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationTurn':
        return cls(**data)

@dataclass  
class ConversationStack:
    """Conversation history stack with multi-turn capabilities"""
    session_id: str
    turns: List[ConversationTurn]
    current_context: Dict[str, Any]  # Active entities and references
    conversation_topic: str  # Current topic (restaurants, museums, etc.)
    last_results: Dict[str, Any]  # Last structured results for follow-up
    reference_cache: Dict[str, Any]  # Cache for pronoun resolution
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'turns': [turn.to_dict() for turn in self.turns],
            'current_context': self.current_context,
            'conversation_topic': self.conversation_topic,
            'last_results': self.last_results,
            'reference_cache': self.reference_cache
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationStack':
        turns = [ConversationTurn.from_dict(turn_data) for turn_data in data.get('turns', [])]
        return cls(
            session_id=data['session_id'],
            turns=turns,
            current_context=data.get('current_context', {}),
            conversation_topic=data.get('conversation_topic', ''),
            last_results=data.get('last_results', {}),
            reference_cache=data.get('reference_cache', {})
        )

class MultiTurnQueryHandler:
    """Main handler for multi-turn query processing"""
    
    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        self.pronoun_patterns = [
            r'\b(it|them|they|those|these|that|this)\b',
            r'\b(which\s+one|what\s+about|how\s+about)\b',
            r'\b(the\s+(?:first|second|third|last|best|closest))\b'
        ]
        self.reference_patterns = [
            r'\b(open\s+now|opening\s+hours|hours)\b',
            r'\b(cheapest|most\s+expensive|price|cost)\b', 
            r'\b(closest|nearest|distance|far)\b',
            r'\b(best\s+rated|rating|review)\b',
            r'\b(more\s+info|details|tell\s+me\s+more)\b'
        ]
        
    def create_conversation_turn(self, user_query: str, ai_response: str, 
                               intent: str, entities: Dict[str, List[str]]) -> ConversationTurn:
        """Create a new conversation turn with extracted data"""
        
        # Extract places and results from AI response
        extracted_places = self.extract_places_from_response(ai_response)
        extracted_results = self.extract_structured_results(ai_response, intent)
        
        # Determine response type
        response_type = self.classify_response_type(ai_response, intent)
        
        return ConversationTurn(
            turn_id=f"turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(user_query)}",
            user_query=user_query,
            ai_response=ai_response,
            intent=intent,
            entities=entities,
            extracted_places=extracted_places,
            extracted_results=extracted_results,
            timestamp=datetime.now().isoformat(),
            response_type=response_type
        )
    
    def extract_places_from_response(self, response: str) -> List[Dict[str, Any]]:
        """Extract structured place data from AI response"""
        places = []
        
        # Pattern to match restaurant/cafe names with details
        place_patterns = [
            r'(?:•\s*)?(?:\*\*)?([A-Z][^•\n\*]+?)(?:\*\*)?\s*-\s*([^•\n]+)',  # • **Name** - description
            r'(?:•\s*)?([A-Z][A-Za-z\s&\'-]{5,50})\s*(?:in|at|near)\s+([A-Z][A-Za-z\s]+)',  # Name in Location
            r'(?:•\s*)?([A-Z][A-Za-z\s&\'-]{5,50})\s*[–-]\s*([^•\n]{10,100})'  # Name – description
        ]
        
        for pattern in place_patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            for match in matches:
                if len(match) >= 2:
                    places.append({
                        'name': match[0].strip(),
                        'description': match[1].strip()[:200],  # Limit description
                        'extracted_from': 'ai_response'
                    })
        
        # Remove duplicates based on name similarity
        unique_places = []
        for place in places:
            is_duplicate = False
            for existing in unique_places:
                if self.is_similar_name(place['name'], existing['name']):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_places.append(place)
        
        return unique_places[:20]  # Limit to 20 places max
    
    def extract_structured_results(self, response: str, intent: str) -> Dict[str, Any]:
        """Extract structured results based on intent type"""
        results = {
            'type': intent,
            'count': 0,
            'items': [],
            'categories': [],
            'locations': []
        }
        
        if intent in ['restaurant_search', 'cafe_search']:
            # Extract restaurant/cafe information
            results['items'] = self.extract_places_from_response(response)
            results['count'] = len(results['items'])
            results['categories'] = self.extract_cuisine_types(response)
            results['locations'] = self.extract_districts(response)
            
        elif intent == 'museum_inquiry':
            # Extract museum information
            results['items'] = self.extract_places_from_response(response)
            results['count'] = len(results['items'])
            results['categories'] = ['museums', 'historical_sites']
            results['locations'] = self.extract_districts(response)
            
        elif intent == 'place_recommendation':
            # Extract attraction information
            results['items'] = self.extract_places_from_response(response)
            results['count'] = len(results['items'])
            results['categories'] = ['attractions', 'landmarks']
            results['locations'] = self.extract_districts(response)
        
        return results
    
    def extract_cuisine_types(self, response: str) -> List[str]:
        """Extract cuisine types from response"""
        cuisine_keywords = {
            'turkish': r'\b(?:turkish|ottoman|traditional|kebab|meze|baklava)\b',
            'seafood': r'\b(?:seafood|fish|mussels|calamari|sea\s+food)\b',
            'international': r'\b(?:international|italian|chinese|japanese|korean|indian)\b',
            'coffee': r'\b(?:coffee|espresso|latte|cappuccino|specialty\s+coffee)\b',
            'breakfast': r'\b(?:breakfast|brunch|menemen|simit|turkish\s+breakfast)\b'
        }
        
        found_cuisines = []
        response_lower = response.lower()
        
        for cuisine, pattern in cuisine_keywords.items():
            if re.search(pattern, response_lower):
                found_cuisines.append(cuisine)
        
        return found_cuisines
    
    def extract_districts(self, response: str) -> List[str]:
        """Extract Istanbul districts from response"""
        districts = [
            'Sultanahmet', 'Beyoğlu', 'Galata', 'Kadiköy', 'Beşiktaş', 
            'Taksim', 'Eminönü', 'Fatih', 'Şişli', 'Ortaköy', 'Balat',
            'Fener', 'Cihangir', 'Nişantaşı', 'Levent', 'Etiler'
        ]
        
        found_districts = []
        response_lower = response.lower()
        
        for district in districts:
            if district.lower() in response_lower:
                found_districts.append(district)
        
        return list(set(found_districts))  # Remove duplicates
    
    def classify_response_type(self, response: str, intent: str) -> str:
        """Classify the type of response for context handling"""
        response_lower = response.lower()
        
        if re.search(r'(?:•\s*.*){3,}', response):  # Multiple bullet points
            return 'list'
        elif intent in ['restaurant_search', 'cafe_search', 'museum_inquiry']:
            return 'recommendations'
        elif 'direction' in response_lower or 'metro' in response_lower:
            return 'directions'
        elif any(word in response_lower for word in ['open', 'hour', 'close']):
            return 'hours_info'
        elif any(word in response_lower for word in ['cost', 'price', 'fee']):
            return 'pricing_info'
        else:
            return 'general_info'
    
    def is_similar_name(self, name1: str, name2: str, threshold: float = 0.7) -> bool:
        """Check if two place names are similar (simple similarity)"""
        # Simple similarity check based on common words
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def is_follow_up_query(self, query: str) -> Tuple[bool, str]:
        """Detect if query is a follow-up and classify its type"""
        query_lower = query.lower().strip()
        
        # Direct follow-up patterns
        follow_up_patterns = {
            'hours': r'\b(?:open\s+now|opening\s+hours|hours|what\s+time|when\s+open)\b',
            'pricing': r'\b(?:cheap|expensive|price|cost|how\s+much|afford)\b',
            'distance': r'\b(?:close|near|far|distance|walk|metro|transport)\b',
            'details': r'\b(?:more\s+info|details|tell\s+me\s+more|about)\b',
            'comparison': r'\b(?:which\s+one|best|better|compare|difference)\b',
            'selection': r'\b(?:first|second|third|last|that\s+one|this\s+one)\b'
        }
        
        for follow_up_type, pattern in follow_up_patterns.items():
            if re.search(pattern, query_lower):
                return True, follow_up_type
        
        # Check for pronouns and references
        for pattern in self.pronoun_patterns:
            if re.search(pattern, query_lower):
                return True, 'pronoun_reference'
        
        # Short queries are likely follow-ups
        if len(query.split()) <= 3 and any(word in query_lower for word in 
                                         ['open', 'close', 'price', 'cost', 'where', 'how', 'which']):
            return True, 'short_followup'
        
        return False, 'new_query'
    
    def resolve_follow_up_query(self, query: str, conversation_stack: ConversationStack) -> Dict[str, Any]:
        """Resolve follow-up query using conversation context"""
        is_followup, followup_type = self.is_follow_up_query(query)
        
        if not is_followup or not conversation_stack.turns:
            return {
                'resolved': False,
                'type': 'new_query',
                'enhanced_query': query,
                'context_used': {}
            }
        
        # Get the most recent turn with results
        last_turn_with_results = None
        for turn in reversed(conversation_stack.turns):
            if turn.extracted_results.get('count', 0) > 0:
                last_turn_with_results = turn
                break
        
        if not last_turn_with_results:
            return {
                'resolved': False,
                'type': 'no_context',
                'enhanced_query': query,
                'context_used': {}
            }
        
        # Resolve based on follow-up type
        if followup_type == 'hours':
            enhanced_query = self.resolve_hours_query(query, last_turn_with_results)
        elif followup_type == 'pricing':
            enhanced_query = self.resolve_pricing_query(query, last_turn_with_results)
        elif followup_type == 'distance':
            enhanced_query = self.resolve_distance_query(query, last_turn_with_results)
        elif followup_type == 'comparison':
            enhanced_query = self.resolve_comparison_query(query, last_turn_with_results)
        elif followup_type == 'selection':
            enhanced_query = self.resolve_selection_query(query, last_turn_with_results)
        else:
            enhanced_query = self.resolve_general_followup(query, last_turn_with_results)
        
        return {
            'resolved': True,
            'type': followup_type,
            'enhanced_query': enhanced_query,
            'context_used': {
                'previous_query': last_turn_with_results.user_query,
                'previous_intent': last_turn_with_results.intent,
                'results_count': last_turn_with_results.extracted_results.get('count', 0),
                'topic': conversation_stack.conversation_topic
            }
        }
    
    def resolve_hours_query(self, query: str, context_turn: ConversationTurn) -> str:
        """Resolve opening hours follow-up query"""
        places = context_turn.extracted_results.get('items', [])
        
        if places:
            place_names = [place['name'] for place in places[:3]]  # Top 3
            if len(place_names) == 1:
                return f"What are the opening hours for {place_names[0]}?"
            else:
                return f"What are the opening hours for these places: {', '.join(place_names)}?"
        
        # Fallback to topic-based resolution
        topic = context_turn.intent
        if 'restaurant' in topic:
            return "What are the opening hours for the restaurants you mentioned?"
        elif 'museum' in topic:
            return "What are the opening hours for the museums you mentioned?"
        else:
            return f"What are the opening hours for the places you mentioned in Istanbul?"
    
    def resolve_pricing_query(self, query: str, context_turn: ConversationTurn) -> str:
        """Resolve pricing follow-up query"""
        places = context_turn.extracted_results.get('items', [])
        
        if places:
            place_names = [place['name'] for place in places[:3]]
            if 'cheap' in query.lower():
                return f"Which is the most budget-friendly among: {', '.join(place_names)}?"
            elif 'expensive' in query.lower():
                return f"Which is the most expensive among: {', '.join(place_names)}?"
            else:
                return f"What are the price ranges for: {', '.join(place_names)}?"
        
        return "What are the price ranges for the places you mentioned?"
    
    def resolve_distance_query(self, query: str, context_turn: ConversationTurn) -> str:
        """Resolve distance/location follow-up query"""
        places = context_turn.extracted_results.get('items', [])
        locations = context_turn.entities.get('locations', [])
        
        if places and locations:
            place_names = [place['name'] for place in places[:3]]
            location = locations[0] if locations else "your location"
            
            if 'closest' in query.lower() or 'nearest' in query.lower():
                return f"Which is closest to {location}: {', '.join(place_names)}?"
            else:
                return f"How far are these places from {location}: {', '.join(place_names)}?"
        
        return "Which of the mentioned places is closest to you?"
    
    def resolve_comparison_query(self, query: str, context_turn: ConversationTurn) -> str:
        """Resolve comparison follow-up query"""
        places = context_turn.extracted_results.get('items', [])
        
        if places:
            place_names = [place['name'] for place in places[:3]]
            return f"Compare these places in Istanbul: {', '.join(place_names)}. Which one is better?"
        
        return "Compare the places you mentioned. Which one is better?"
    
    def resolve_selection_query(self, query: str, context_turn: ConversationTurn) -> str:
        """Resolve selection follow-up query (first, second, that one, etc.)"""
        places = context_turn.extracted_results.get('items', [])
        
        if not places:
            return "Tell me more about the first place you mentioned."
        
        # Extract ordinal or demonstrative
        query_lower = query.lower()
        
        if 'first' in query_lower and len(places) >= 1:
            return f"Tell me more about {places[0]['name']} in Istanbul."
        elif 'second' in query_lower and len(places) >= 2:
            return f"Tell me more about {places[1]['name']} in Istanbul."
        elif 'third' in query_lower and len(places) >= 3:
            return f"Tell me more about {places[2]['name']} in Istanbul."
        elif 'last' in query_lower and places:
            return f"Tell me more about {places[-1]['name']} in Istanbul."
        else:
            # Default to first place
            return f"Tell me more about {places[0]['name']} in Istanbul."
    
    def resolve_general_followup(self, query: str, context_turn: ConversationTurn) -> str:
        """Resolve general follow-up queries"""
        places = context_turn.extracted_results.get('items', [])
        
        if places:
            place_names = [place['name'] for place in places[:2]]  # Top 2
            return f"{query} (Context: referring to {', '.join(place_names)} in Istanbul)"
        
        return f"{query} (Context: referring to the places mentioned in our previous conversation about Istanbul)"
    
    def update_conversation_stack(self, conversation_stack: ConversationStack, 
                                new_turn: ConversationTurn) -> ConversationStack:
        """Update conversation stack with new turn"""
        
        # Add new turn
        conversation_stack.turns.append(new_turn)
        
        # Limit history to max turns
        if len(conversation_stack.turns) > self.max_history_turns:
            conversation_stack.turns = conversation_stack.turns[-self.max_history_turns:]
        
        # Update current context
        conversation_stack.current_context.update(new_turn.entities)
        
        # Update conversation topic based on recent turns
        recent_intents = [turn.intent for turn in conversation_stack.turns[-3:]]
        if recent_intents:
            # Most common intent in recent turns
            from collections import Counter
            most_common_intent = Counter(recent_intents).most_common(1)[0][0]
            conversation_stack.conversation_topic = most_common_intent
        
        # Update last results if this turn has results
        if new_turn.extracted_results.get('count', 0) > 0:
            conversation_stack.last_results = new_turn.extracted_results
        
        # Update reference cache for pronoun resolution
        if new_turn.extracted_places:
            conversation_stack.reference_cache['last_places'] = new_turn.extracted_places
        
        return conversation_stack

# Export main class
__all__ = ['MultiTurnQueryHandler', 'ConversationTurn', 'ConversationStack']
