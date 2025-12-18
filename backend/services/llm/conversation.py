"""
conversation.py - Conversation Management System

Multi-turn conversation support with context and reference resolution.

Features:
- Conversation history storage
- Reference resolution ("it", "there", "that place")
- Context formatting for LLM
- Session management
- Turn tracking
- Topic detection

Author: AI Istanbul Team
Date: November 2025
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Conversation management system for multi-turn dialogues.
    
    Handles:
    - Conversation history per session
    - Reference resolution (pronouns, demonstratives)
    - Context summarization for LLM
    - Topic tracking
    """
    
    def __init__(
        self,
        max_history_length: int = 10,
        enable_reference_resolution: bool = True
    ):
        """
        Initialize conversation manager.
        
        Args:
            max_history_length: Maximum conversation turns to keep
            enable_reference_resolution: Enable reference resolution
        """
        self.max_history = max_history_length
        self.enable_resolution = enable_reference_resolution
        
        # Session storage: session_id -> conversation data
        self.sessions = {}
        
        # Reference patterns for resolution
        self._init_reference_patterns()
        
        logger.info("✅ Conversation Manager initialized")
    
    def _init_reference_patterns(self):
        """Initialize reference resolution patterns."""
        self.reference_patterns = {
            # Pronouns
            'it': ['restaurant', 'place', 'location', 'attraction', 'museum'],
            'there': ['location', 'place', 'area', 'district', 'neighborhood'],
            'that': ['restaurant', 'place', 'attraction', 'museum', 'event'],
            'this': ['restaurant', 'place', 'attraction', 'museum', 'event'],
            
            # Demonstrative phrases
            'that place': ['restaurant', 'location', 'attraction'],
            'this place': ['restaurant', 'location', 'attraction'],
            'the place': ['restaurant', 'location', 'attraction'],
        }
    
    async def get_context(
        self,
        session_id: str,
        current_query: str,
        max_turns: int = 3
    ) -> Dict[str, Any]:
        """
        Get conversation context for current query.
        
        Args:
            session_id: Session identifier
            current_query: Current user query
            max_turns: Maximum turns to include
            
        Returns:
            Dict with:
            - history: List of recent turns
            - needs_resolution: bool
            - topics: List of discussed topics
            - entities: Dict of mentioned entities
        """
        if session_id not in self.sessions:
            return {
                'history': [],
                'needs_resolution': False,
                'topics': [],
                'entities': {}
            }
        
        session = self.sessions[session_id]
        
        # Get recent history
        history = list(session['turns'])[-max_turns:] if max_turns else list(session['turns'])
        
        # Check if current query has references
        needs_resolution = self._has_references(current_query)
        
        return {
            'history': history,
            'needs_resolution': needs_resolution,
            'topics': session.get('topics', []),
            'entities': session.get('entities', {})
        }
    
    async def resolve_references(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve references in query using conversation context.
        
        Args:
            query: User query with potential references
            context: Conversation context
            
        Returns:
            Dict with:
            - resolved: bool
            - resolved_query: str
            - replacements: Dict of reference -> entity
        """
        if not self.enable_resolution or not context.get('history'):
            return {
                'resolved': False,
                'resolved_query': query,
                'replacements': {}
            }
        
        query_lower = query.lower()
        replacements = {}
        resolved_query = query
        
        # Check each reference pattern
        for reference, entity_types in self.reference_patterns.items():
            if reference in query_lower:
                # Find most recent matching entity
                entity = self._find_recent_entity(
                    context['history'],
                    entity_types
                )
                
                if entity:
                    # Replace reference with entity
                    resolved_query = resolved_query.replace(
                        reference,
                        entity,
                        1  # Replace only first occurrence
                    )
                    replacements[reference] = entity
        
        return {
            'resolved': len(replacements) > 0,
            'resolved_query': resolved_query,
            'replacements': replacements
        }
    
    def _has_references(self, query: str) -> bool:
        """Check if query contains references or is context-dependent."""
        query_lower = query.lower()
        
        # Check for explicit reference patterns
        for reference in self.reference_patterns.keys():
            if reference in query_lower:
                return True
        
        # Check for implicit references (context-dependent queries without explicit location)
        # These are short queries that assume previous context
        implicit_patterns = [
            'find attractions',
            'show attractions',
            'what attractions',
            'any attractions',
            'find restaurants',
            'show restaurants',
            'what restaurants',
            'any restaurants',
            'find places',
            'show places',
            'what to do',
            'what to see',
            'things to do',
            'places to visit',
            'how do i get',
            'how can i get',
            'opening hours',
            'when open',
            'how much',
            'ticket price',
        ]
        
        # If query is short and matches an implicit pattern, it likely needs context
        if len(query_lower.split()) <= 4:
            for pattern in implicit_patterns:
                if pattern in query_lower:
                    logger.debug(f"Query '{query}' matches implicit reference pattern: {pattern}")
                    return True
        
        return False
    
    def _find_recent_entity(
        self,
        history: List[Dict[str, Any]],
        entity_types: List[str]
    ) -> Optional[str]:
        """
        Find most recent entity of given types in history.
        
        Args:
            history: Conversation history
            entity_types: Types of entities to look for
            
        Returns:
            Entity name or None
        """
        # Search backwards through history
        for turn in reversed(history):
            # Check both user and assistant messages
            content = turn.get('content', '')  # Keep original case
            metadata = turn.get('metadata', {})
            
            # Try to extract entity name from content
            entity = self._extract_entity_name(content)
            if entity:
                # Filter by entity type if needed (for now accept any proper noun)
                logger.debug(f"Found entity in conversation: {entity}")
                return entity
        
        return None
    
    def _extract_entity_name(self, text: str) -> Optional[str]:
        """Extract entity name from text (simple heuristic)."""
        import re
        
        # Look for capitalized words (likely place names)
        # This is a simple heuristic; production would use NER
        
        # Pattern 1: Multiple capitalized words (e.g., "Blue Mosque", "Galata Tower")
        pattern_multi = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        matches_multi = re.findall(pattern_multi, text)
        
        if matches_multi:
            # Filter out common words that aren't place names
            common_words = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'Is', 'Are', 
                          'Was', 'Were', 'Has', 'Have', 'Can', 'Will', 'Would', 'Should',
                          'Sultan Ahmed Mosque'}  # Alternative name
            
            for match in matches_multi:
                # Skip matches that start with common words
                first_word = match.split()[0]
                if first_word not in common_words:
                    logger.debug(f"Extracted multi-word entity: {match}")
                    return match
        
        # Pattern 2: Single capitalized word (e.g., "Taksim", "Beyoğlu")
        pattern_single = r'\b([A-Z][a-zğüşıöçĞÜŞİÖÇ]+)\b'
        matches_single = re.findall(pattern_single, text)
        
        if matches_single:
            # Filter out common words
            common_single = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'From', 'Is', 'Are',
                           'Turkish', 'Istanbul', 'I', 'You', 'He', 'She', 'It', 'We', 'They'}
            
            for match in matches_single:
                if match not in common_single:
                    logger.debug(f"Extracted single-word entity: {match}")
                    return match
        
        return None
    
    async def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a conversation turn.
        
        Args:
            session_id: Session identifier
            role: 'user' or 'assistant'
            content: Turn content
            metadata: Optional metadata
        """
        # Create session if doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'turns': deque(maxlen=self.max_history),
                'topics': [],
                'entities': {}
            }
        
        session = self.sessions[session_id]
        
        # Add turn
        turn = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        session['turns'].append(turn)
        
        # Update topics and entities
        if role == 'user':
            self._update_topics(session, content, metadata)
    
    def _update_topics(
        self,
        session: Dict[str, Any],
        content: str,
        metadata: Optional[Dict[str, Any]]
    ):
        """Update conversation topics based on signals."""
        if not metadata:
            return
        
        signals = metadata.get('signals', {})
        
        # Add detected signals as topics
        for signal_name, detected in signals.items():
            if detected:
                topic = signal_name.replace('needs_', '')
                if topic not in session['topics']:
                    session['topics'].append(topic)
                    
                    # Keep only last 5 topics
                    if len(session['topics']) > 5:
                        session['topics'] = session['topics'][-5:]
    
    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum turns to return
            
        Returns:
            List of conversation turns
        """
        if session_id not in self.sessions:
            return []
        
        session = self.sessions[session_id]
        turns = list(session['turns'])
        
        if max_turns:
            turns = turns[-max_turns:]
        
        return turns
    
    def format_context_for_llm(
        self,
        session_id: str,
        max_turns: int = 3,
        include_metadata: bool = False
    ) -> str:
        """
        Format conversation context for LLM prompt.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum turns to include
            include_metadata: Include metadata in formatting
            
        Returns:
            Formatted context string
        """
        history = self.get_history(session_id, max_turns)
        
        if not history:
            return ""
        
        formatted = []
        
        for turn in history:
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
            
            if include_metadata and turn.get('metadata'):
                metadata_str = str(turn['metadata'])[:100]
                formatted.append(f"  [Metadata: {metadata_str}]")
        
        return "\n".join(formatted)
    
    def get_context_summary(
        self,
        session_id: str,
        max_length: int = 200
    ) -> str:
        """
        Get a brief summary of conversation context.
        
        Args:
            session_id: Session identifier
            max_length: Maximum summary length
            
        Returns:
            Summary string
        """
        if session_id not in self.sessions:
            return ""
        
        session = self.sessions[session_id]
        topics = session.get('topics', [])
        
        if not topics:
            return ""
        
        # Create summary
        topic_str = ', '.join(topics)
        summary = f"Previous topics discussed: {topic_str}"
        
        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def has_context(self, session_id: str) -> bool:
        """Check if session has conversation context."""
        return (
            session_id in self.sessions and
            len(self.sessions[session_id]['turns']) > 0
        )
    
    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"🗑️ Cleared session: {session_id}")
    
    def clear_old_sessions(self, hours: int = 24):
        """
        Clear sessions older than N hours.
        
        Args:
            hours: Age threshold in hours
        """
        from datetime import datetime, timedelta
        
        cutoff = datetime.now() - timedelta(hours=hours)
        to_remove = []
        
        for session_id, session in self.sessions.items():
            created_str = session.get('created_at', '')
            try:
                created = datetime.fromisoformat(created_str)
                if created < cutoff:
                    to_remove.append(session_id)
            except Exception:
                pass
        
        for session_id in to_remove:
            del self.sessions[session_id]
        
        if to_remove:
            logger.info(f"🗑️ Cleared {len(to_remove)} old sessions")
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Args:
            session_id: Specific session (or None for all)
            
        Returns:
            Statistics dict
        """
        if session_id:
            if session_id not in self.sessions:
                return {'exists': False}
            
            session = self.sessions[session_id]
            
            return {
                'exists': True,
                'session_id': session_id,
                'created_at': session.get('created_at'),
                'turn_count': len(session['turns']),
                'topics': session.get('topics', []),
                'last_topic': session['topics'][-1] if session.get('topics') else None
            }
        else:
            # Global statistics
            total_turns = sum(
                len(session['turns'])
                for session in self.sessions.values()
            )
            
            return {
                'total_sessions': len(self.sessions),
                'total_turns': total_turns,
                'avg_turns_per_session': (
                    total_turns / len(self.sessions)
                    if self.sessions else 0
                )
            }
