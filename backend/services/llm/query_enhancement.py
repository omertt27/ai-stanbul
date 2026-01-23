"""
query_enhancement.py - Query Enhancement System

Advanced query processing to improve understanding and response quality.

Features:
- Spell checking and correction
- Query rewriting and optimization
- Query validation and quality scoring
- Autocomplete suggestions
- Related query recommendations
- Trending queries tracking
- Language detection

Author: AI Istanbul Team
Date: November 2025
"""

import logging
import re
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """
    Query enhancement system for improved understanding.
    
    Enhances queries through:
    - Spell correction
    - Query rewriting
    - Quality validation
    - Suggestions and recommendations
    """
    
    def __init__(
        self,
        enable_spell_check: bool = True,
        enable_rewriting: bool = True,
        enable_validation: bool = True
    ):
        """
        Initialize query enhancer.
        
        Args:
            enable_spell_check: Enable spell checking
            enable_rewriting: Enable query rewriting
            enable_validation: Enable query validation
        """
        self.enable_spell_check = enable_spell_check
        self.enable_rewriting = enable_rewriting
        self.enable_validation = enable_validation
        
        # Query history for suggestions
        self.query_history = defaultdict(int)  # query -> count
        self.query_timestamps = []  # (query, timestamp) for trending
        
        # Common Istanbul terms (for spell check allowlist)
        self.istanbul_terms = {
            'beyoglu', 'galata', 'taksim', 'sultanahmet', 'kadikoy',
            'besiktas', 'ortakoy', 'uskudar', 'bosphorus', 'bosporus',
            'topkapi', 'hagia', 'sophia', 'ayasofya', 'cistern',
            'istanbul', 'turkish', 'turkiye', 'türkiye'
        }
        
        logger.info("✅ Query Enhancer initialized")
    
    async def enhance_query(
        self,
        query: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Enhance a query through all enabled enhancements.
        
        Args:
            query: Original query
            language: Language code
            
        Returns:
            Dict with:
            - query: Enhanced query
            - spell_corrected: bool
            - rewritten: bool
            - validation: validation result dict
        """
        result = {
            'query': query,
            'spell_corrected': False,
            'rewritten': False,
            'validation': None
        }
        
        # Track query
        self._track_query(query)
        
        # Step 1: Spell check
        if self.enable_spell_check:
            corrected = await self.correct_spelling(query, language)
            if corrected['has_correction']:
                result['query'] = corrected['corrected_query']
                result['spell_corrected'] = True
                logger.debug(f"Spell corrected: {query} → {result['query']}")
        
        # Step 2: Query rewriting
        if self.enable_rewriting:
            rewritten = await self.rewrite_query(result['query'], language)
            if rewritten['needs_rewrite']:
                result['query'] = rewritten['rewritten_query']
                result['rewritten'] = True
                logger.debug(f"Rewritten: {query} → {result['query']}")
        
        # Step 3: Validation
        if self.enable_validation:
            validation = await self.validate_query(result['query'], language)
            result['validation'] = validation
        
        return result
    
    async def correct_spelling(
        self,
        query: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Check and correct spelling errors.
        
        Args:
            query: User query
            language: Language code
            
        Returns:
            Dict with has_correction, corrected_query, confidence
        """
        # Simple spell check implementation
        # In production, use a proper library like pyspellchecker or TextBlob
        
        words = query.split()
        corrected_words = []
        has_changes = False
        
        for word in words:
            word_lower = word.lower().strip('.,!?')
            
            # Skip Istanbul-specific terms
            if word_lower in self.istanbul_terms:
                corrected_words.append(word)
                continue
            
            # Check common misspellings
            corrected = self._check_common_misspellings(word_lower)
            
            if corrected != word_lower:
                # Preserve original capitalization
                if word[0].isupper():
                    corrected = corrected.capitalize()
                corrected_words.append(corrected)
                has_changes = True
            else:
                corrected_words.append(word)
        
        return {
            'has_correction': has_changes,
            'corrected_query': ' '.join(corrected_words),
            'confidence': 0.8 if has_changes else 1.0
        }
    
    def _check_common_misspellings(self, word: str) -> str:
        """Check against common misspellings."""
        common_corrections = {
            # Common Istanbul misspellings
            'instanbul': 'istanbul',
            'istambul': 'istanbul',
            'stambul': 'istanbul',
            'haghia': 'hagia',
            'bosphoros': 'bosphorus',
            'galatta': 'galata',
            'takseem': 'taksim',
            # Common travel terms
            'resturant': 'restaurant',
            'restaurent': 'restaurant',
            'restraunt': 'restaurant',
            'restaraunt': 'restaurant',
            'museam': 'museum',
            'musium': 'museum',
            'directionz': 'directions',
            'accomodation': 'accommodation',
            'atractions': 'attractions',
            'atraction': 'attraction',
        }
        
        return common_corrections.get(word, word)
    
    async def rewrite_query(
        self,
        query: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Rewrite query for better understanding.
        
        Args:
            query: User query
            language: Language code
            
        Returns:
            Dict with needs_rewrite, rewritten_query, reason
        """
        query_lower = query.lower()
        
        # Pattern 1: Expand abbreviations
        if self._has_abbreviations(query_lower):
            rewritten = self._expand_abbreviations(query)
            return {
                'needs_rewrite': True,
                'rewritten_query': rewritten,
                'reason': 'Expanded abbreviations'
            }
        
        # Pattern 2: Add context to vague queries
        if self._is_vague_query(query_lower):
            rewritten = self._add_context(query, language)
            return {
                'needs_rewrite': True,
                'rewritten_query': rewritten,
                'reason': 'Added context to vague query'
            }
        
        # Pattern 3: Normalize question format
        if self._needs_normalization(query_lower):
            rewritten = self._normalize_query(query)
            return {
                'needs_rewrite': True,
                'rewritten_query': rewritten,
                'reason': 'Normalized question format'
            }
        
        return {
            'needs_rewrite': False,
            'rewritten_query': query,
            'reason': None
        }
    
    def _has_abbreviations(self, query: str) -> bool:
        """Check if query has common abbreviations."""
        abbreviations = ['pls', 'plz', 'thx', 'u', 'ur', 'r', 'b4']
        return any(abbr in query.split() for abbr in abbreviations)
    
    def _expand_abbreviations(self, query: str) -> str:
        """Expand common abbreviations."""
        expansions = {
            'pls': 'please',
            'plz': 'please',
            'thx': 'thanks',
            'thnx': 'thanks',
            'u': 'you',
            'ur': 'your',
            'r': 'are',
            'b4': 'before',
            'w/': 'with',
            'wo/': 'without'
        }
        
        words = query.split()
        expanded = [expansions.get(word.lower(), word) for word in words]
        return ' '.join(expanded)
    
    def _is_vague_query(self, query: str) -> bool:
        """Check if query is too vague."""
        vague_patterns = [
            r'^(what|where|how)\s+(can|do|is)\s+(i|you)$',
            r'^(show|tell|give)\s+me$',
            r'^(help|info|information)$'
        ]
        return any(re.match(pattern, query) for pattern in vague_patterns)
    
    def _add_context(self, query: str, language: str) -> str:
        """Add context to vague queries."""
        # Simple heuristic: add "in Istanbul" if location not mentioned
        if 'istanbul' not in query.lower():
            if language == 'tr':
                return f"{query} İstanbul'da"
            else:
                return f"{query} in Istanbul"
        return query
    
    def _needs_normalization(self, query: str) -> bool:
        """Check if query needs normalization."""
        # Check for all lowercase or all uppercase
        if query.islower() or query.isupper():
            return True
        return False
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query format."""
        # Capitalize first letter, lowercase rest
        if not query:
            return query
        
        # Handle all caps
        if query.isupper():
            query = query.lower()
        
        # Capitalize first letter
        return query[0].upper() + query[1:]
    
    async def validate_query(
        self,
        query: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Validate query quality.
        
        Args:
            query: User query
            language: Language code
            
        Returns:
            Validation result dict
        """
        issues = []
        quality_score = 1.0
        query_lower = query.lower().strip()
        
        # BYPASS: Greetings and conversational queries should NEVER need clarification
        # These should go directly to LLM for natural response
        greeting_patterns = [
            # Turkish greetings
            'merhaba', 'selam', 'günaydın', 'iyi akşamlar', 'iyi geceler',
            'nasılsın', 'nasilsin', 'nasılsınız', 'naber', 'ne haber',
            'hoşgeldin', 'hosgeldin', 'hoş geldiniz', 'eyvallah', 'sağol',
            'teşekkür', 'tesekkur', 'sağ ol', 'sag ol', 'görüşürüz', 'hoşça kal',
            'kolay gelsin', 'iyi günler', 'hayırlı işler', 'memnun oldum',
            # English greetings
            'hello', 'hi', 'hey', 'good morning', 'good evening', 'good night',
            'how are you', "how's it going", 'what\'s up', 'sup',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'cheers',
            'nice to meet you', 'pleased to meet you', 'welcome',
            # Common short conversational phrases
            'yes', 'no', 'ok', 'okay', 'evet', 'hayır', 'tamam', 'peki',
            'yardım', 'help', 'bilgi', 'info',
        ]
        
        is_greeting = any(pattern in query_lower for pattern in greeting_patterns)
        
        if is_greeting:
            # Greetings are valid and never need clarification
            return {
                'valid': True,
                'quality_score': 1.0,
                'needs_clarification': False,
                'clarification': None,
                'complexity': 'simple',
                'issues': []
            }
        
        # Check 1: Minimum length (only for non-greeting queries)
        if len(query.strip()) < 3:
            issues.append("Query too short")
            quality_score -= 0.5
        
        # Check 2: Maximum length
        if len(query) > 500:
            issues.append("Query very long")
            quality_score -= 0.2
        
        # Check 3: Has actual words (support Turkish characters)
        if not re.search(r'[a-zA-ZğüşıöçĞÜŞİÖÇ]', query):
            issues.append("No readable text found")
            quality_score -= 0.5
        
        # Check 4: Excessive punctuation
        punct_ratio = sum(c in '!?.,;:' for c in query) / max(1, len(query))
        if punct_ratio > 0.2:
            issues.append("Excessive punctuation")
            quality_score -= 0.1
        
        # Determine complexity
        word_count = len(query.split())
        if word_count <= 3:
            complexity = "simple"
        elif word_count <= 10:
            complexity = "medium"
        else:
            complexity = "complex"
        
        # CHANGED: Short queries should NOT automatically need clarification
        # The LLM can understand context and provide helpful responses
        # Only require clarification if quality is very low
        needs_clarification = (
            quality_score < 0.5 and  # Only if quality is very poor
            len(issues) > 1  # And there are multiple issues
        )
        
        clarification = None
        if needs_clarification:
            if language == 'tr':
                clarification = "Lütfen daha açıklayıcı bir soru sorun."
            else:
                clarification = "Could you please provide more details?"
        
        return {
            'valid': quality_score >= 0.5,
            'quality_score': max(0.0, quality_score),
            'needs_clarification': needs_clarification,
            'clarification': clarification,
            'complexity': complexity,
            'issues': issues
        }
    
    async def get_suggestions(
        self,
        partial_query: str,
        language: str = "en",
        max_suggestions: int = 5
    ) -> List[str]:
        """
        Get autocomplete suggestions.
        
        Args:
            partial_query: Partial user input
            language: Language code
            max_suggestions: Maximum suggestions
            
        Returns:
            List of suggested completions
        """
        if len(partial_query) < 2:
            return []
        
        partial_lower = partial_query.lower()
        
        # Find matching queries from history
        matches = []
        for query, count in self.query_history.items():
            if query.lower().startswith(partial_lower):
                matches.append((query, count))
        
        # Sort by popularity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Get popular suggestions for this language
        popular = self._get_popular_suggestions(language)
        
        # Combine and deduplicate
        suggestions = [q for q, _ in matches[:max_suggestions]]
        
        # Fill with popular if needed
        for pop_query in popular:
            if len(suggestions) >= max_suggestions:
                break
            if pop_query.lower().startswith(partial_lower):
                if pop_query not in suggestions:
                    suggestions.append(pop_query)
        
        return suggestions[:max_suggestions]
    
    def _get_popular_suggestions(self, language: str) -> List[str]:
        """Get popular query suggestions."""
        if language == 'tr':
            return [
                "Beyoğlu'nda restoran önerileri",
                "Sultanahmet gezilecek yerler",
                "Taksim'e nasıl gidilir",
                "İstanbul hava durumu",
                "Gizli cennetler İstanbul"
            ]
        else:
            return [
                "Best restaurants in Beyoglu",
                "Things to do in Sultanahmet",
                "How to get to Taksim",
                "Istanbul weather",
                "Hidden gems in Istanbul"
            ]
    
    async def get_trending_queries(
        self,
        language: str = "en",
        hours: int = 24,
        limit: int = 10
    ) -> List[str]:
        """
        Get trending queries from recent activity.
        
        Args:
            language: Language code
            hours: Look back N hours
            limit: Maximum queries
            
        Returns:
            List of trending queries
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Count recent queries
        recent_counts = Counter()
        
        for query, timestamp in self.query_timestamps:
            if timestamp > cutoff_time:
                recent_counts[query] += 1
        
        # Get top queries
        trending = [q for q, _ in recent_counts.most_common(limit)]
        
        return trending
    
    async def get_popular_queries(
        self,
        language: str = "en",
        limit: int = 10
    ) -> List[str]:
        """
        Get all-time popular queries.
        
        Args:
            language: Language code
            limit: Maximum queries
            
        Returns:
            List of popular queries
        """
        # Sort by count
        popular = sorted(
            self.query_history.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [q for q, _ in popular[:limit]]
    
    def _track_query(self, query: str):
        """Track query for suggestions and trending."""
        self.query_history[query] += 1
        self.query_timestamps.append((query, datetime.now()))
        
        # Keep only last 10000 timestamps
        if len(self.query_timestamps) > 10000:
            self.query_timestamps = self.query_timestamps[-10000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        return {
            'total_queries_tracked': sum(self.query_history.values()),
            'unique_queries': len(self.query_history),
            'spell_check_enabled': self.enable_spell_check,
            'rewriting_enabled': self.enable_rewriting,
            'validation_enabled': self.enable_validation
        }
