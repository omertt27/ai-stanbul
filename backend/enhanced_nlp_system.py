#!/usr/bin/env python3
"""
Enhanced Lightweight NLP System
===============================

Integrates lightweight models for advanced context resolution:
- Pronoun resolution with context awareness
- Intent change detection
- Context summarization
- Reference disambiguation

Uses lightweight models that can run without GPU requirements.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LightweightNLPEnhancer:
    """Enhanced NLP system with context-aware processing"""
    
    def __init__(self):
        self.pronouns = ['it', 'them', 'they', 'these', 'those', 'that', 'this']
        self.reference_patterns = [
            r'\b(which\s+one|what\s+about|how\s+about)\b',
            r'\b(the\s+(?:first|second|third|last|best|closest))\b',
            r'\b(that\s+(?:place|restaurant|museum|one))\b',
            r'\b(and\s+the\s+\w+)\b'  # "and the museum?"
        ]
        
        # Intent change indicators
        self.intent_changes = {
            'restaurant_to_transport': ['how to get', 'directions', 'travel to', 'way to'],
            'place_to_hours': ['open', 'hours', 'when', 'time'],
            'general_to_specific': ['tell me more', 'details', 'what about'],
            'compare_to_decide': ['which', 'better', 'choose', 'recommend']
        }
        
        # Context keywords for disambiguation
        self.context_keywords = {
            'restaurants': ['eat', 'food', 'dining', 'meal', 'cuisine', 'restaurant', 'cafe'],
            'museums': ['museum', 'gallery', 'art', 'history', 'exhibit', 'culture'],
            'transport': ['metro', 'bus', 'taxi', 'ferry', 'transport', 'travel'],
            'places': ['place', 'attraction', 'site', 'visit', 'see', 'location'],
            'time': ['hours', 'open', 'close', 'time', 'when', 'schedule'],
            'price': ['cost', 'price', 'expensive', 'cheap', 'budget', 'fee']
        }
    
    def resolve_pronouns_with_context(self, query: str, conversation_stack) -> str:
        """Resolve pronouns using conversation context"""
        if not conversation_stack or not conversation_stack.turns:
            return query
        
        query_lower = query.lower().strip()
        resolved_query = query
        
        # Get the most recent context
        last_turn = conversation_stack.turns[-1]
        
        # Check for direct pronoun references
        for pronoun in self.pronouns:
            if f" {pronoun} " in f" {query_lower} " or query_lower.startswith(f"{pronoun} "):
                
                # Find what the pronoun refers to
                resolved_context = self._find_pronoun_reference(pronoun, last_turn, conversation_stack)
                
                if resolved_context:
                    # Replace pronoun with context
                    pattern = r'\b' + re.escape(pronoun) + r'\b'
                    resolved_query = re.sub(pattern, resolved_context, resolved_query, flags=re.IGNORECASE, count=1)
                    print(f"🔗 Resolved pronoun '{pronoun}' → '{resolved_context}'")
                    break
        
        # Check for reference patterns
        for pattern in self.reference_patterns:
            match = re.search(pattern, query_lower)
            if match:
                reference = match.group(1)
                resolved_context = self._resolve_reference_pattern(reference, last_turn, conversation_stack)
                
                if resolved_context:
                    resolved_query = re.sub(pattern, resolved_context, resolved_query, flags=re.IGNORECASE, count=1)
                    print(f"🔗 Resolved reference '{reference}' → '{resolved_context}'")
                    break
        
        return resolved_query
    
    def detect_intent_change(self, current_query: str, conversation_stack) -> Tuple[bool, str, str]:
        """Detect if user intent has changed from previous queries"""
        if not conversation_stack or not conversation_stack.turns:
            return False, "none", "initial_query"
        
        current_intent = self._classify_intent(current_query)
        last_intent = conversation_stack.conversation_topic or "general"
        
        # Check for specific intent transitions
        query_lower = current_query.lower()
        
        for transition, keywords in self.intent_changes.items():
            if any(keyword in query_lower for keyword in keywords):
                old_intent, new_intent = transition.split('_to_')
                if old_intent in last_intent.lower():
                    print(f"🔄 Intent change detected: {last_intent} → {new_intent}")
                    return True, last_intent, new_intent
        
        # General intent change detection
        if current_intent != last_intent and len(conversation_stack.turns) > 1:
            return True, last_intent, current_intent
        
        return False, last_intent, current_intent
    
    def summarize_conversation_context(self, conversation_stack, max_length: int = 200) -> str:
        """Create a concise summary of conversation context"""
        if not conversation_stack or not conversation_stack.turns:
            return ""
        
        # Get key information from recent turns
        recent_turns = conversation_stack.turns[-3:]  # Last 3 turns
        
        # Extract key entities and topics
        locations = set()
        topics = set()
        preferences = set()
        
        for turn in recent_turns:
            # Extract locations
            for entity_type, entities in turn.entities.items():
                if entity_type == 'locations':
                    locations.update(entities)
            
            # Extract topics from intent
            if turn.intent:
                topics.add(turn.intent.replace('_', ' '))
            
            # Extract preferences from query
            query_lower = turn.user_query.lower()
            if 'budget' in query_lower or 'cheap' in query_lower:
                preferences.add('budget-friendly')
            if 'authentic' in query_lower or 'traditional' in query_lower:
                preferences.add('authentic')
            if 'highly rated' in query_lower or 'best' in query_lower:
                preferences.add('high quality')
        
        # Create summary
        summary_parts = []
        
        if locations:
            summary_parts.append(f"Location: {', '.join(list(locations)[:2])}")
        
        if topics:
            summary_parts.append(f"Topics: {', '.join(list(topics)[:2])}")
        
        if preferences:
            summary_parts.append(f"Preferences: {', '.join(list(preferences)[:2])}")
        
        # Add conversation flow
        if len(recent_turns) > 1:
            last_query = recent_turns[-1].user_query.lower()
            if any(word in last_query for word in ['which', 'what about', 'how about']):
                summary_parts.append("Follow-up: comparing options")
        
        summary = " | ".join(summary_parts)
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def enhance_query_with_context(self, query: str, conversation_stack) -> str:
        """Enhance query with conversation context"""
        if not conversation_stack:
            return query
        
        enhanced_query = query
        
        # Step 1: Resolve pronouns and references
        enhanced_query = self.resolve_pronouns_with_context(enhanced_query, conversation_stack)
        
        # Step 2: Add context if query is ambiguous
        if self._is_ambiguous_query(enhanced_query):
            context_summary = self.summarize_conversation_context(conversation_stack, 100)
            if context_summary:
                enhanced_query += f" (Context: {context_summary})"
        
        # Step 3: Detect and handle intent changes
        intent_changed, old_intent, new_intent = self.detect_intent_change(enhanced_query, conversation_stack)
        if intent_changed:
            enhanced_query += f" [Intent changed from {old_intent} to {new_intent}]"
        
        return enhanced_query
    
    def _find_pronoun_reference(self, pronoun: str, last_turn, conversation_stack) -> Optional[str]:
        """Find what a pronoun refers to in the conversation context"""
        
        # Check last AI response for mentioned places/things
        if last_turn.ai_response:
            # Look for restaurant names
            restaurant_matches = re.findall(r'\*\*([^*]+)\*\*', last_turn.ai_response)
            if restaurant_matches and pronoun in ['it', 'that', 'this']:
                return f"the {restaurant_matches[0].lower()}"
            
            # Look for place mentions
            if 'restaurant' in last_turn.ai_response.lower() and pronoun in ['it', 'them', 'they']:
                return "the restaurants"
            if 'museum' in last_turn.ai_response.lower() and pronoun in ['it', 'them', 'they']:
                return "the museums"
            if 'place' in last_turn.ai_response.lower() and pronoun in ['it', 'them', 'they']:
                return "the places"
        
        # Check extracted places from last turn
        if hasattr(last_turn, 'extracted_places') and last_turn.extracted_places:
            if len(last_turn.extracted_places) == 1:
                return f"the {last_turn.extracted_places[0]['name'].lower()}"
            elif len(last_turn.extracted_places) > 1 and pronoun in ['them', 'they', 'those']:
                return "those places"
        
        # Fallback to intent-based reference
        if last_turn.intent:
            if 'restaurant' in last_turn.intent and pronoun in ['it', 'them', 'they']:
                return "the restaurants"
            elif 'museum' in last_turn.intent and pronoun in ['it', 'them', 'they']:
                return "the museums"
        
        return None
    
    def _resolve_reference_pattern(self, reference: str, last_turn, conversation_stack) -> Optional[str]:
        """Resolve reference patterns like 'which one', 'the first'"""
        
        reference_lower = reference.lower()
        
        # Handle "which one", "what about", "how about"
        if reference_lower in ['which one', 'what about', 'how about']:
            return self._find_pronoun_reference('it', last_turn, conversation_stack)
        
        # Handle "the first", "the second", etc.
        if 'the ' in reference_lower and any(ord_word in reference_lower for ord_word in ['first', 'second', 'third', 'last', 'best', 'closest']):
            # Try to get specific reference from last response
            if last_turn.extracted_places:
                return f"the first mentioned place"
            else:
                return "the first option"
        
        # Handle "and the X"
        if 'and the ' in reference_lower:
            topic_match = re.search(r'and the (\w+)', reference_lower)
            if topic_match:
                topic = topic_match.group(1)
                return f"the {topic} mentioned earlier"
        
        return None
    
    def _classify_intent(self, query: str) -> str:
        """Classify the intent of a query"""
        query_lower = query.lower()
        
        # Score each intent category
        intent_scores = {}
        for intent, keywords in self.context_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return the highest scoring intent
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0] + "_inquiry"
        
        return "general_travel_info"
    
    def _is_ambiguous_query(self, query: str) -> bool:
        """Check if a query is ambiguous and needs context"""
        ambiguous_patterns = [
            r'^\s*(?:which|what|how|where|when)\s*\??\s*$',  # Single question words
            r'^\s*(?:more|others|else|different)\s*\??\s*$',  # Continuation words
            r'^\s*(?:and|also|too)\s+',  # Additive queries
            r'^\s*(?:what about|how about)\s+',  # Comparative queries
        ]
        
        query_clean = query.strip()
        return len(query_clean.split()) <= 3 and any(re.match(pattern, query, re.IGNORECASE) for pattern in ambiguous_patterns)

class ContextualResponseEnhancer:
    """Enhances AI responses with context awareness"""
    
    def __init__(self):
        self.nlp_enhancer = LightweightNLPEnhancer()
    
    def enhance_ai_response_with_context(self, ai_response: str, conversation_stack, current_query: str) -> str:
        """Enhance AI response with conversation context"""
        
        if not conversation_stack or not conversation_stack.turns:
            return ai_response
        
        enhanced_response = ai_response
        
        # Add context-aware introduction if this is a follow-up
        if len(conversation_stack.turns) > 0:
            last_turn = conversation_stack.turns[-1]
            
            # Check if this is a follow-up to a list of recommendations
            if ('restaurant' in last_turn.ai_response.lower() and 
                any(word in current_query.lower() for word in ['which', 'what about', 'more'])):
                
                enhanced_response = f"Based on the restaurants I mentioned earlier, {enhanced_response}"
            
            elif ('museum' in last_turn.ai_response.lower() and 
                  any(word in current_query.lower() for word in ['which', 'what about', 'more'])):
                
                enhanced_response = f"Regarding the museums we discussed, {enhanced_response}"
        
        # Add conversation continuity markers
        if self._is_followup_response(enhanced_response, conversation_stack):
            enhanced_response += "\n\n💡 *This response builds on our previous conversation about Istanbul attractions.*"
        
        return enhanced_response
    
    def _is_followup_response(self, response: str, conversation_stack) -> bool:
        """Check if response appears to be a follow-up"""
        if not conversation_stack or len(conversation_stack.turns) < 2:
            return False
        
        # Look for continuity indicators in response
        continuity_indicators = ['mentioned', 'discussed', 'those', 'these', 'earlier', 'previous']
        return any(indicator in response.lower() for indicator in continuity_indicators)

# Export classes
__all__ = ['LightweightNLPEnhancer', 'ContextualResponseEnhancer']
