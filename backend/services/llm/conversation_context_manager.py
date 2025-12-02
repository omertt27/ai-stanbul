"""
LLM Conversation Context Manager
=================================

Phase 4.2 of LLM Enhancement: Maintain conversation context and memory.

Gives LLM complete control over:
- Remembering conversation history
- Resolving pronouns and references ("it", "there", "that place")
- Tracking multi-step journeys
- Maintaining user state across session
- Clarifying ambiguities

Example conversations:
- User: "Show me route to Hagia Sophia"
  Bot: [shows route]
  User: "What about restaurants there?"
  → Context Manager resolves "there" = Hagia Sophia

- User: "I'm at Taksim"
  Bot: "How can I help?"
  User: "Take me to the Blue Mosque"
  → Context Manager remembers origin = Taksim

Author: Istanbul AI Team
Date: December 2025
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation"""
    timestamp: datetime
    user_query: str
    bot_response: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    locations: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    """Current state of the conversation"""
    session_id: str
    user_id: Optional[str] = None
    current_location: Optional[Dict[str, float]] = None
    last_mentioned_locations: List[str] = field(default_factory=list)
    last_mentioned_entities: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    active_task: Optional[str] = None  # e.g., "multi_stop_journey", "restaurant_search"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn to history"""
        self.conversation_history.append(turn)
        self.updated_at = datetime.utcnow()
        
        # Update last mentioned locations
        if turn.locations:
            self.last_mentioned_locations = turn.locations[-3:]  # Keep last 3
        
        # Update last mentioned entities
        if turn.entities:
            self.last_mentioned_entities.update(turn.entities)
    
    def get_recent_history(self, n: int = 5) -> List[ConversationTurn]:
        """Get recent conversation turns"""
        return self.conversation_history[-n:] if self.conversation_history else []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'current_location': self.current_location,
            'last_mentioned_locations': self.last_mentioned_locations,
            'last_mentioned_entities': self.last_mentioned_entities,
            'user_preferences': self.user_preferences,
            'active_task': self.active_task,
            'history_length': len(self.conversation_history),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class LLMConversationContextManager:
    """
    LLM-powered conversation context manager.
    
    Uses LLM to understand and resolve context from conversation history,
    giving the LLM complete control over conversation flow and memory.
    """
    
    def __init__(
        self,
        llm_client=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Conversation Context Manager.
        
        Args:
            llm_client: LLM API client for context resolution
            config: Configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'enable_llm': True,
            'fallback_to_rules': True,
            'timeout_seconds': 2,
            'max_history_turns': 10,
            'min_confidence': 0.6,
            **(config or {})
        }
        
        # In-memory session storage (in production, use Redis/database)
        self._sessions: Dict[str, ConversationState] = {}
        
        # Statistics
        self.stats = {
            'total_resolutions': 0,
            'llm_resolutions': 0,
            'fallback_resolutions': 0,
            'sessions_created': 0,
            'average_latency_ms': 0.0
        }
        
        logger.info("✅ LLM Conversation Context Manager initialized")
    
    def get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None
    ) -> ConversationState:
        """Get or create conversation session"""
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(
                session_id=session_id,
                user_id=user_id
            )
            self.stats['sessions_created'] += 1
            logger.info(f"📝 Created new conversation session: {session_id}")
        
        return self._sessions[session_id]
    
    async def resolve_context(
        self,
        current_query: str,
        session_id: str,
        user_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Resolve context for current query using conversation history.
        
        Args:
            current_query: Current user query
            session_id: Session identifier
            user_id: Optional user identifier
            user_location: Optional GPS location
            
        Returns:
            Dict with resolved context
        """
        start_time = datetime.utcnow()
        
        try:
            self.stats['total_resolutions'] += 1
            
            # Get or create session
            session = self.get_or_create_session(session_id, user_id)
            
            # Update location if provided
            if user_location:
                session.current_location = user_location
            
            # Try LLM resolution
            if self.llm_client and self.config['enable_llm']:
                try:
                    resolved_context = await self._llm_resolve_context(
                        current_query, session
                    )
                    self.stats['llm_resolutions'] += 1
                    return resolved_context
                    
                except Exception as e:
                    logger.warning(f"LLM context resolution failed: {e}")
                    if not self.config['fallback_to_rules']:
                        raise
            
            # Fallback to rule-based resolution
            resolved_context = self._rule_based_resolve(current_query, session)
            self.stats['fallback_resolutions'] += 1
            return resolved_context
            
        finally:
            # Update stats
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.stats['average_latency_ms'] = (
                (self.stats['average_latency_ms'] * (self.stats['total_resolutions'] - 1) + latency_ms)
                / self.stats['total_resolutions']
            )
    
    async def _llm_resolve_context(
        self,
        current_query: str,
        session: ConversationState
    ) -> Dict[str, Any]:
        """
        Use LLM to resolve context from conversation history.
        
        This is the PRIMARY method - LLM has FULL control over:
        - Reference resolution (pronouns, "there", "it", "that")
        - Implicit context inference (continuing tasks, remembered locations)
        - Multi-turn conversation understanding
        - Clarification needs detection
        - Context-aware query rewriting
        """
        
        # Build comprehensive prompt for LLM
        prompt = self._build_context_resolution_prompt(current_query, session)
        
        # Call LLM with appropriate method
        if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
            # OpenAI-style client
            response = self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert conversation context analyzer for a travel chatbot in Istanbul. "
                            "Your role is to understand conversation history, resolve references, infer implicit context, "
                            "and rewrite queries to be standalone and clear. You have COMPLETE authority over "
                            "context resolution - use your understanding of natural conversation flow."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for consistent context resolution
                max_tokens=600,   # More tokens for detailed context analysis
                timeout=self.config['timeout_seconds']
            )
            
            llm_output = response.choices[0].message.content.strip()
        else:
            # Fallback: assume generate method (for custom LLM clients)
            llm_output = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=600,
                temperature=0.2
            )
        
        # Parse LLM response
        resolved_context = self._parse_llm_context_response(llm_output, session)
        
        logger.info(
            f"✅ LLM resolved context for '{current_query[:50]}...': "
            f"has_refs={resolved_context.get('has_references', False)}, "
            f"confidence={resolved_context.get('confidence', 0):.2f}, "
            f"resolved={list(resolved_context.get('resolved_references', {}).keys())}"
        )
        
        return resolved_context
    
    def _build_context_resolution_prompt(
        self,
        current_query: str,
        session: ConversationState
    ) -> str:
        """
        Build comprehensive prompt for LLM context resolution.
        
        This prompt gives the LLM COMPLETE control over understanding context.
        """
        
        # Get recent history with more detail
        recent_turns = session.get_recent_history(n=5)
        
        if recent_turns:
            history_text = ""
            for i, turn in enumerate(recent_turns, 1):
                history_text += f"\nTurn {i}:\n"
                history_text += f"  User: {turn.user_query}\n"
                history_text += f"  Bot: {turn.bot_response[:150]}...\n"
                if turn.intent:
                    history_text += f"  Intent: {turn.intent}\n"
                if turn.locations:
                    history_text += f"  Locations: {', '.join(turn.locations)}\n"
        else:
            history_text = "No previous conversation"
        
        # Build context summary
        context_summary = f"""
Session Context:
- Last Mentioned Locations: {', '.join(session.last_mentioned_locations) if session.last_mentioned_locations else 'None'}
- User's GPS Location: {session.current_location if session.current_location else 'Unknown'}
- Active Task: {session.active_task if session.active_task else 'None'}
- User Preferences: {session.user_preferences if session.user_preferences else 'None'}
- Conversation Age: {len(session.conversation_history)} turns
"""
        
        prompt = f"""You are analyzing a conversation to resolve context and references. Your goal is to understand what the user means by considering the full conversation history.

CURRENT QUERY: "{current_query}"

CONVERSATION HISTORY:
{history_text}
{context_summary}

YOUR TASK:
Analyze the current query with COMPLETE authority. You understand natural conversation better than any rule-based system.

1. **Pronouns & References** - Resolve what these refer to:
   - Pronouns: "it", "there", "here", "that", "this", "them"
   - Ordinals: "first one", "second one", "third one", "last one"
   - Demonstratives: "that place", "this restaurant", "those museums"
   - Implicit: "back", "again", "also", "too"

2. **Implicit Context** - What can you infer from history:
   - Is user continuing a previous journey/task?
   - What locations have they shown interest in?
   - What preferences have they expressed?
   - What's their current situation?

3. **Query Rewriting** - Create standalone version:
   - Replace ALL pronouns/references with actual values
   - Add implicit context (origin, destination, preferences)
   - Make query understandable without any history

4. **Clarification Needs** - Detect ambiguity:
   - Is anything still unclear?
   - What additional info is needed?
   - What clarifying question should we ask?

5. **Conversation Flow** - Understand the journey:
   - Multi-step planning?
   - Follow-up questions?
   - Topic changes?

RETURN FORMAT (JSON only, no markdown):
{{
  "has_references": boolean,
  "resolved_references": {{
    "original_text": "what it refers to"
  }},
  "resolved_query": "complete standalone query",
  "implicit_context": {{
    "origin": "location/null",
    "destination": "location/null",
    "continuing_task": "task_type/null",
    "user_intent": "inferred intent",
    "topic": "current topic"
  }},
  "missing_information": ["list of missing info"],
  "confidence": 0.0-1.0,
  "needs_clarification": boolean,
  "clarification_question": "question to ask or null",
  "reasoning": "brief explanation of your analysis"
}}

EXAMPLES:

Query: "What about restaurants there?"
History: User asked about Hagia Sophia
→ {{
  "has_references": true,
  "resolved_references": {{"there": "Hagia Sophia"}},
  "resolved_query": "What restaurants are near Hagia Sophia?",
  "implicit_context": {{"destination": "Hagia Sophia", "user_intent": "find_restaurants", "topic": "dining_near_attraction"}},
  "confidence": 0.95,
  "needs_clarification": false,
  "reasoning": "'There' clearly refers to Hagia Sophia from previous turn"
}}

Query: "Take me to the Blue Mosque"
History: User said "I'm at Taksim Square"
→ {{
  "has_references": false,
  "resolved_references": {{}},
  "resolved_query": "Take me from Taksim Square to the Blue Mosque",
  "implicit_context": {{"origin": "Taksim Square", "destination": "Blue Mosque", "user_intent": "get_directions"}},
  "confidence": 0.92,
  "needs_clarification": false,
  "reasoning": "Origin inferred from previous statement about current location"
}}

Query: "How about the second one?"
History: Showed list of 5 museums
→ {{
  "has_references": true,
  "resolved_references": {{"the second one": "Topkapi Palace"}},
  "resolved_query": "Tell me about Topkapi Palace",
  "implicit_context": {{"topic": "museum_exploration", "continuing_task": "browse_recommendations"}},
  "confidence": 0.88,
  "needs_clarification": false,
  "reasoning": "Second museum from the displayed list"
}}

Query: "What's the weather?"
History: Planning trip to Princes' Islands
→ {{
  "has_references": false,
  "resolved_references": {{}},
  "resolved_query": "What's the weather at Princes' Islands?",
  "implicit_context": {{"destination": "Princes' Islands", "user_intent": "weather_check", "continuing_task": "trip_planning"}},
  "confidence": 0.85,
  "needs_clarification": false,
  "reasoning": "Weather query in context of trip planning - user likely wants weather at destination"
}}

NOW ANALYZE: "{current_query}"

Return ONLY the JSON object, no markdown or explanation outside the JSON."""
        
        return prompt
    
    def _parse_llm_context_response(
        self,
        llm_output: str,
        session: ConversationState
    ) -> Dict[str, Any]:
        """
        Parse LLM JSON response into context dict.
        
        Handles various response formats and extracts maximum information.
        """
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
            
            # Parse JSON
            data = json.loads(llm_output)
            
            # Extract all fields with defaults
            result = {
                'has_references': data.get('has_references', False),
                'resolved_references': data.get('resolved_references', {}),
                'resolved_query': data.get('resolved_query', ''),
                'implicit_context': data.get('implicit_context', {}),
                'missing_information': data.get('missing_information', []),
                'confidence': data.get('confidence', 0.5),
                'needs_clarification': data.get('needs_clarification', False),
                'clarification_question': data.get('clarification_question'),
                'reasoning': data.get('reasoning', ''),  # NEW: LLM's reasoning
                'source': 'llm',
                'session_state': session.to_dict()
            }
            
            # Log reasoning if provided
            if result['reasoning']:
                logger.debug(f"LLM reasoning: {result['reasoning']}")
            
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse LLM context response: {e}")
            logger.debug(f"LLM output was: {llm_output[:500]}...")
            
            # Fall back to rule-based
            logger.warning("⚠️ Falling back to rule-based context resolution due to parse error")
            return self._rule_based_resolve(
                session.conversation_history[-1].user_query if session.conversation_history else "",
                session
            )
    
    def _rule_based_resolve(
        self,
        current_query: str,
        session: ConversationState
    ) -> Dict[str, Any]:
        """
        Rule-based fallback for context resolution.
        
        This is ONLY used when LLM is unavailable.
        """
        
        query_lower = current_query.lower()
        resolved_references = {}
        implicit_context = {}
        
        # Detect references
        has_references = any(ref in query_lower for ref in [
            'there', 'it', 'that place', 'this place',
            'the first', 'the second', 'the third',
            'the other', 'that one', 'this one'
        ])
        
        # Resolve "there" to last mentioned location
        if 'there' in query_lower and session.last_mentioned_locations:
            resolved_references['there'] = session.last_mentioned_locations[-1]
        
        # Resolve "it" to last mentioned entity
        if (' it' in query_lower or query_lower.startswith('it') or query_lower.endswith(' it')) and session.last_mentioned_locations:
            resolved_references['it'] = session.last_mentioned_locations[-1]
        
        # Check if continuing a journey (user previously mentioned origin)
        if session.current_location and any(kw in query_lower for kw in ['take me', 'go to', 'route to']):
            implicit_context['origin'] = session.current_location
        
        # Build resolved query
        resolved_query = current_query
        for ref, value in resolved_references.items():
            resolved_query = resolved_query.replace(ref, value)
        
        logger.info(
            f"⚠️  Fallback context resolution for '{current_query[:50]}...': "
            f"references={list(resolved_references.keys())}"
        )
        
        return {
            'has_references': has_references,
            'resolved_references': resolved_references,
            'resolved_query': resolved_query,
            'implicit_context': implicit_context,
            'missing_information': [],
            'confidence': 0.6,
            'needs_clarification': False,
            'clarification_question': None,
            'source': 'fallback',
            'session_state': session.to_dict()
        }
    
    def record_turn(
        self,
        session_id: str,
        user_query: str,
        bot_response: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        locations: Optional[List[str]] = None
    ):
        """
        Record a conversation turn.
        
        Args:
            session_id: Session identifier
            user_query: User's query
            bot_response: Bot's response
            intent: Detected intent
            entities: Extracted entities
            locations: Mentioned locations
        """
        session = self.get_or_create_session(session_id)
        
        turn = ConversationTurn(
            timestamp=datetime.utcnow(),
            user_query=user_query,
            bot_response=bot_response,
            intent=intent,
            entities=entities or {},
            locations=locations or []
        )
        
        session.add_turn(turn)
        logger.info(f"📝 Recorded conversation turn for session {session_id}")
    
    def clear_session(self, session_id: str):
        """Clear conversation session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"🗑️  Cleared session: {session_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context resolution statistics"""
        return {
            **self.stats,
            'active_sessions': len(self._sessions),
            'llm_usage_rate': (
                self.stats['llm_resolutions'] / self.stats['total_resolutions']
                if self.stats['total_resolutions'] > 0 else 0
            )
        }


# Singleton instance
_context_manager = None


def get_context_manager(
    llm_client=None,
    config: Optional[Dict[str, Any]] = None
) -> LLMConversationContextManager:
    """
    Get or create Conversation Context Manager singleton.
    
    Args:
        llm_client: LLM client (only used on first call)
        config: Configuration overrides
        
    Returns:
        LLMConversationContextManager instance
    """
    global _context_manager
    
    if _context_manager is None:
        _context_manager = LLMConversationContextManager(
            llm_client=llm_client,
            config=config
        )
    
    return _context_manager


# Convenience function
async def resolve_conversation_context(
    current_query: str,
    session_id: str,
    user_id: Optional[str] = None,
    user_location: Optional[Dict[str, float]] = None,
    llm_client=None
) -> Dict[str, Any]:
    """
    Convenience function to resolve conversation context.
    
    Args:
        current_query: Current user query
        session_id: Session identifier
        user_id: Optional user identifier
        user_location: Optional GPS location
        llm_client: Optional LLM client
        
    Returns:
        Resolved context dict
    """
    manager = get_context_manager(llm_client=llm_client)
    return await manager.resolve_context(current_query, session_id, user_id, user_location)
