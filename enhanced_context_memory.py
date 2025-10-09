#!/usr/bin/env python3
"""
Enhanced Context Memory System for AI Istanbul
Advanced memory management and context tracking for conversational AI
"""

import json
import redis
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid
import logging
from enum import Enum

class ContextType(Enum):
    """Types of context information"""
    LOCATION = "location"
    PREFERENCE = "preference"
    CONVERSATION = "conversation"
    INTENT = "intent"
    ENTITY = "entity"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"

@dataclass
class ContextItem:
    """Individual context item with metadata"""
    id: str
    type: ContextType
    content: Dict[str, Any]
    confidence: float
    timestamp: datetime
    expiry_time: Optional[datetime] = None
    access_count: int = 0
    relevance_score: float = 1.0
    source: str = "user_input"

@dataclass
class ConversationTurn:
    """Single conversation turn with context"""
    turn_id: str
    user_query: str
    ai_response: str
    extracted_entities: Dict[str, List[str]]
    intent: str
    confidence: float
    timestamp: datetime
    context_used: List[str]  # IDs of context items used

class EnhancedContextMemory:
    """
    Advanced context memory system with multi-layered memory management
    Handles short-term, working, and long-term memory with intelligent decay
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        # Memory layers
        self.short_term_memory = deque(maxlen=10)  # Last 10 interactions
        self.working_memory = {}  # Active context items
        self.long_term_memory = defaultdict(list )  # Persistent patterns
        
        # Redis for persistence (optional)
        self.redis_client = redis_client
        
        # Memory configuration
        self.config = {
            'short_term_capacity': 10,
            'working_memory_capacity': 50,
            'context_decay_hours': 24,
            'min_confidence_threshold': 0.3,
            'max_context_age_days': 30
        }
        
        # Context weights for different types
        self.context_weights = {
            ContextType.LOCATION: 0.9,
            ContextType.PREFERENCE: 0.8,
            ContextType.INTENT: 0.7,
            ContextType.CONVERSATION: 0.6,
            ContextType.ENTITY: 0.5,
            ContextType.TEMPORAL: 0.4,
            ContextType.BEHAVIORAL: 0.3
        }
        
        # Session management
        self.current_session_id = None
        self.session_contexts = {}
        
    def start_new_session(self, user_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        
        self.session_contexts[session_id] = {
            'user_id': user_id,
            'start_time': datetime.now(),
            'conversation_turns': [],
            'active_contexts': {},
            'session_summary': {}
        }
        
        # Load persistent context if user_id provided
        if user_id and self.redis_client:
            self._load_user_context(user_id, session_id)
        
        logging.info(f"✅ New session started: {session_id}")
        return session_id
    
    def add_context_item(self, context_type: ContextType, content: Dict[str, Any],
                        confidence: float = 1.0, source: str = "user_input",
                        expiry_hours: Optional[int] = None) -> str:
        """Add a new context item to memory"""
        
        item_id = str(uuid.uuid4())
        expiry_time = None
        if expiry_hours:
            expiry_time = datetime.now() + timedelta(hours=expiry_hours)
        
        context_item = ContextItem(
            id=item_id,
            type=context_type,
            content=content,
            confidence=confidence,
            timestamp=datetime.now(),
            expiry_time=expiry_time,
            source=source
        )
        
        # Add to working memory
        self.working_memory[item_id] = context_item
        
        # Add to current session if active
        if self.current_session_id:
            session = self.session_contexts[self.current_session_id]
            session['active_contexts'][item_id] = asdict(context_item)
        
        # Maintain memory limits
        self._enforce_memory_limits()
        
        logging.debug(f"Context added: {context_type.value} - {item_id}")
        return item_id
    
    def add_conversation_turn(self, user_query: str, ai_response: str,
                           extracted_entities: Dict[str, List[str]],
                           intent: str, confidence: float) -> str:
        """Add a conversation turn to memory"""
        
        turn_id = str(uuid.uuid4())
        
        # Determine which context items were used
        context_used = self._identify_context_usage(user_query, ai_response)
        
        conversation_turn = ConversationTurn(
            turn_id=turn_id,
            user_query=user_query,
            ai_response=ai_response,
            extracted_entities=extracted_entities,
            intent=intent,
            confidence=confidence,
            timestamp=datetime.now(),
            context_used=context_used
        )
        
        # Add to short-term memory
        self.short_term_memory.append(conversation_turn)
        
        # Add to current session
        if self.current_session_id:
            session = self.session_contexts[self.current_session_id]
            session['conversation_turns'].append(asdict(conversation_turn))
        
        # Extract and add new context items from this turn
        self._extract_context_from_turn(conversation_turn)
        
        # Update context relevance scores
        self._update_context_relevance(context_used)
        
        return turn_id
    
    def get_relevant_context(self, query: str, intent: Optional[str] = None,
                           max_items: int = 10) -> List[ContextItem]:
        """Get most relevant context items for a query"""
        
        # Calculate relevance scores for all context items
        scored_contexts = []
        
        for item_id, context_item in self.working_memory.items():
            # Skip expired items
            if self._is_expired(context_item):
                continue
            
            relevance_score = self._calculate_relevance_score(
                context_item, query, intent
            )
            
            if relevance_score >= self.config['min_confidence_threshold']:
                context_item.relevance_score = relevance_score
                context_item.access_count += 1
                scored_contexts.append(context_item)
        
        # Sort by relevance and return top items
        scored_contexts.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_contexts[:max_items]
    
    def get_conversation_history(self, max_turns: int = 5) -> List[ConversationTurn]:
        """Get recent conversation history"""
        return list(self.short_term_memory)[-max_turns:]
    
    def get_session_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of current or specified session"""
        
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.session_contexts:
            return {}
        
        session = self.session_contexts[session_id]
        
        # Calculate session statistics
        turns = session['conversation_turns']
        contexts = session['active_contexts']
        
        summary = {
            'session_id': session_id,
            'start_time': session['start_time'],
            'duration_minutes': (datetime.now() - session['start_time']).total_seconds() / 60,
            'total_turns': len(turns),
            'active_contexts': len(contexts),
            'dominant_intents': self._get_dominant_intents(turns),
            'key_entities': self._get_key_entities(turns),
            'location_context': self._get_location_context(contexts),
            'user_preferences': self._get_user_preferences(contexts)
        }
        
        return summary
    
    def update_location_context(self, location_data: Dict[str, Any],
                              confidence: float = 0.9) -> str:
        """Update location context with new information"""
        
        # Remove old location contexts
        self._remove_contexts_by_type(ContextType.LOCATION)
        
        # Add new location context
        return self.add_context_item(
            ContextType.LOCATION,
            location_data,
            confidence=confidence,
            source="location_service",
            expiry_hours=self.config['context_decay_hours']
        )
    
    def update_user_preferences(self, preferences: Dict[str, Any],
                              confidence: float = 0.8) -> str:
        """Update user preferences based on interaction patterns"""
        
        # ENHANCEMENT: Handle list values for Redis storage compatibility
        processed_preferences = self._serialize_for_storage(preferences)
        
        # Store in Redis if available
        if self.redis_client and self.current_session_id:
            try:
                user_key = f"user_preferences:{self.current_session_id}"
                self.redis_client.hset(user_key, mapping=processed_preferences)
                logging.debug(f"User preferences saved to Redis: {user_key}")
            except Exception as e:
                logging.error(f"Failed to save preferences to Redis: {e}")
        
        return self.add_context_item(
            ContextType.PREFERENCE,
            preferences,  # Store original data in context memory
            confidence=confidence,
            source="behavioral_analysis"
        )
    
    def _serialize_for_storage(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Serialize complex data types for Redis storage"""
        serialized = {}
        
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                # Convert lists and dicts to JSON strings
                serialized[key] = json.dumps(value)
            else:
                # Keep simple types as strings
                serialized[key] = str(value)
        
        return serialized
    
    def _deserialize_from_storage(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Deserialize data from Redis storage"""
        deserialized = {}
        
        for key, value in data.items():
            try:
                # Try to parse as JSON first
                deserialized[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Keep as string if not valid JSON
                deserialized[key] = value
        
        return deserialized
    
    def _calculate_relevance_score(self, context_item: ContextItem,
                                 query: str, intent: Optional[str] = None) -> float:
        """Calculate relevance score for a context item"""
        
        score = 0.0
        
        # Base confidence score
        score += context_item.confidence * 0.3
        
        # Type-based weight
        type_weight = self.context_weights.get(context_item.type, 0.5)
        score += type_weight * 0.2
        
        # Recency bonus (more recent = higher score)
        age_hours = (datetime.now() - context_item.timestamp).total_seconds() / 3600
        recency_factor = max(0, 1 - (age_hours / self.config['context_decay_hours']))
        score += recency_factor * 0.2
        
        # Access frequency bonus
        access_bonus = min(0.2, context_item.access_count * 0.02)
        score += access_bonus
        
        # Content relevance (basic keyword matching)
        content_relevance = self._calculate_content_relevance(
            context_item, query, intent
        )
        score += content_relevance * 0.3
        
        return min(score, 1.0)
    
    def _calculate_content_relevance(self, context_item: ContextItem,
                                   query: str, intent: Optional[str] = None) -> float:
        """Calculate content-based relevance score"""
        
        relevance = 0.0
        query_lower = query.lower()
        
        # Check content for query keywords
        content_text = json.dumps(context_item.content).lower()
        query_words = query_lower.split()
        
        matching_words = sum(1 for word in query_words if word in content_text)
        if query_words:
            relevance += (matching_words / len(query_words)) * 0.5
        
        # Intent matching
        if intent and context_item.type == ContextType.INTENT:
            if context_item.content.get('intent') == intent:
                relevance += 0.3
        
        # Location relevance
        if context_item.type == ContextType.LOCATION:
            location_keywords = ['where', 'near', 'location', 'place', 'go', 'visit']
            if any(keyword in query_lower for keyword in location_keywords):
                relevance += 0.4
        
        return min(relevance, 1.0)
    
    def _extract_context_from_turn(self, turn: ConversationTurn):
        """Extract context items from a conversation turn"""
        
        # Extract entities as context
        for entity_type, entities in turn.extracted_entities.items():
            for entity in entities:
                self.add_context_item(
                    ContextType.ENTITY,
                    {'type': entity_type, 'value': entity},
                    confidence=turn.confidence * 0.8,
                    source="entity_extraction"
                )
        
        # Extract intent as context
        if turn.intent:
            self.add_context_item(
                ContextType.INTENT,
                {'intent': turn.intent, 'query': turn.user_query},
                confidence=turn.confidence,
                source="intent_detection"
            )
        
        # Extract temporal context
        temporal_keywords = ['today', 'tomorrow', 'morning', 'evening', 'weekend']
        for keyword in temporal_keywords:
            if keyword in turn.user_query.lower():
                self.add_context_item(
                    ContextType.TEMPORAL,
                    {'time_reference': keyword, 'query': turn.user_query},
                    confidence=0.7,
                    source="temporal_extraction"
                )
    
    def _identify_context_usage(self, user_query: str, ai_response: str) -> List[str]:
        """Identify which context items were likely used in generating response"""
        
        used_contexts = []
        response_lower = ai_response.lower()
        
        for item_id, context_item in self.working_memory.items():
            # Check if context content appears in response
            content_text = json.dumps(context_item.content).lower()
            
            # Simple heuristic: if context content words appear in response
            content_words = content_text.split()
            matching_words = sum(1 for word in content_words 
                               if len(word) > 3 and word in response_lower)
            
            if matching_words >= 2:  # At least 2 matching words
                used_contexts.append(item_id)
        
        return used_contexts
    
    def _update_context_relevance(self, used_context_ids: List[str]):
        """Update relevance scores for contexts that were used"""
        
        for context_id in used_context_ids:
            if context_id in self.working_memory:
                context_item = self.working_memory[context_id]
                # Boost relevance for used contexts
                context_item.relevance_score = min(1.0, context_item.relevance_score + 0.1)
    
    def _is_expired(self, context_item: ContextItem) -> bool:
        """Check if a context item has expired"""
        
        if context_item.expiry_time:
            return datetime.now() > context_item.expiry_time
        
        # Default expiry based on type
        max_age = timedelta(days=self.config['max_context_age_days'])
        return datetime.now() - context_item.timestamp > max_age
    
    def _enforce_memory_limits(self):
        """Enforce memory capacity limits"""
        
        # Remove expired items
        expired_ids = [
            item_id for item_id, item in self.working_memory.items()
            if self._is_expired(item)
        ]
        
        for item_id in expired_ids:
            del self.working_memory[item_id]
        
        # Remove least relevant items if over capacity
        if len(self.working_memory) > self.config['working_memory_capacity']:
            items_by_relevance = sorted(
                self.working_memory.items(),
                key=lambda x: (x[1].relevance_score, x[1].access_count)
            )
            
            # Remove oldest, least relevant items
            items_to_remove = len(self.working_memory) - self.config['working_memory_capacity']
            for i in range(items_to_remove):
                item_id = items_by_relevance[i][0]
                del self.working_memory[item_id]
    
    def _remove_contexts_by_type(self, context_type: ContextType):
        """Remove all context items of a specific type"""
        
        to_remove = [
            item_id for item_id, item in self.working_memory.items()
            if item.type == context_type
        ]
        
        for item_id in to_remove:
            del self.working_memory[item_id]
    
    def _get_dominant_intents(self, turns: List[Dict]) -> List[str]:
        """Get most common intents from conversation turns"""
        
        intent_counts = defaultdict(int)
        for turn in turns:
            intent = turn.get('intent')
            if intent:
                intent_counts[intent] += 1
        
        return sorted(intent_counts.keys(), key=intent_counts.get, reverse=True)[:3]
    
    def _get_key_entities(self, turns: List[Dict]) -> Dict[str, List[str]]:
        """Get most frequently mentioned entities"""
        
        entity_counts = defaultdict(lambda: defaultdict(int))
        
        for turn in turns:
            entities = turn.get('extracted_entities', {})
            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_counts[entity_type][entity] += 1
        
        # Get top entities per type
        key_entities = {}
        for entity_type, entities in entity_counts.items():
            top_entities = sorted(entities.keys(), key=entities.get, reverse=True)[:3]
            key_entities[entity_type] = top_entities
        
        return key_entities
    
    def _get_location_context(self, contexts: Dict) -> Optional[Dict[str, Any]]:
        """Get current location context"""
        
        for context_data in contexts.values():
            if context_data.get('type') == ContextType.LOCATION.value:
                return context_data.get('content')
        
        return None
    
    def _get_user_preferences(self, contexts: Dict) -> Dict[str, Any]:
        """Get user preferences from context"""
        
        preferences = {}
        
        for context_data in contexts.values():
            if context_data.get('type') == ContextType.PREFERENCE.value:
                content = context_data.get('content', {})
                preferences.update(content)
        
        return preferences
    
    def _load_user_context(self, user_id: str, session_id: str):
        """Load persistent user context from Redis"""
        
        if not self.redis_client:
            return
        
        try:
            # Load user's long-term context
            user_key = f"user_context:{user_id}"
            context_data = self.redis_client.get(user_key)
            
            if context_data:
                user_context = json.loads(context_data)
                
                # Add persistent preferences to working memory
                for pref_data in user_context.get('preferences', []):
                    self.add_context_item(
                        ContextType.PREFERENCE,
                        pref_data,
                        confidence=0.8,
                        source="persistent_storage"
                    )
                
                logging.info(f"✅ User context loaded for: {user_id}")
        
        except Exception as e:
            logging.error(f"❌ Failed to load user context: {str(e)}")
    
    def save_session_context(self, session_id: Optional[str] = None):
        """Save session context to persistent storage"""
        
        if not self.redis_client:
            return
        
        session_id = session_id or self.current_session_id
        if not session_id or session_id not in self.session_contexts:
            return
        
        try:
            session_data = self.session_contexts[session_id]
            
            # Save session summary
            session_key = f"session:{session_id}"
            self.redis_client.setex(
                session_key,
                timedelta(days=7),  # Keep sessions for 7 days
                json.dumps(session_data, default=str)
            )
            
            logging.info(f"✅ Session context saved: {session_id}")
        
        except Exception as e:
            logging.error(f"❌ Failed to save session context: {str(e)}")

# Example usage and testing
def test_enhanced_context_memory():
    """Test the enhanced context memory system"""
    
    print("🧠 Testing Enhanced Context Memory System...")
    
    memory = EnhancedContextMemory()
    
    # Start a session
    session_id = memory.start_new_session("test_user_123")
    print(f"Session started: {session_id}")
    
    # Add some context items
    location_id = memory.update_location_context({
        'name': 'Sultanahmet',
        'latitude': 41.0082,
        'longitude': 28.9784,
        'district': 'Fatih'
    })
    
    pref_id = memory.update_user_preferences({
        'cuisine_preference': 'Turkish',
        'budget_range': 'medium',
        'activity_type': 'cultural'
    })
    
    # Add conversation turns
    memory.add_conversation_turn(
        user_query="Where can I find good restaurants near me?",
        ai_response="Here are some excellent Turkish restaurants in Sultanahmet area...",
        extracted_entities={'locations': ['Sultanahmet'], 'activities': ['restaurants']},
        intent='restaurant_recommendation',
        confidence=0.9
    )
    
    memory.add_conversation_turn(
        user_query="What about museums?",
        ai_response="The Hagia Sophia and Blue Mosque are very close to your location...",
        extracted_entities={'locations': ['Hagia Sophia', 'Blue Mosque'], 'activities': ['museums']},
        intent='museum_recommendation',
        confidence=0.85
    )
    
    # Test context retrieval
    relevant_contexts = memory.get_relevant_context(
        "Tell me about historical places",
        intent="information_request"
    )
    
    print(f"\nRelevant contexts found: {len(relevant_contexts)}")
    for context in relevant_contexts:
        print(f"  - {context.type.value}: {context.content} (score: {context.relevance_score:.2f})")
    
    # Get session summary
    summary = memory.get_session_summary()
    print(f"\nSession Summary:")
    print(f"  - Duration: {summary['duration_minutes']:.1f} minutes")
    print(f"  - Total turns: {summary['total_turns']}")
    print(f"  - Dominant intents: {summary['dominant_intents']}")
    print(f"  - Key entities: {summary['key_entities']}")

if __name__ == "__main__":
    test_enhanced_context_memory()
