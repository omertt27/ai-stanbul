#!/usr/bin/env python3
"""
Istanbul Knowledge RAG System
Comprehensive knowledge retrieval for all Istanbul tourism topics

Covers 8 knowledge areas:
1. Neighborhoods - Districts, areas, local tips
2. Scams - Common tourist scams and prevention
3. Turkish Phrases - Essential language guide
4. Transportation - Metro, ferries, taxis, airport transfers
5. Food & Dining - Cuisine, restaurants, prices
6. Attractions - Historical sites, museums, bazaars
7. Practical Info - Money, SIM cards, health, safety
8. Seasonal Events - Weather, festivals, best times

Author: AI Istanbul Team
Date: December 2025
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeResult:
    """A single knowledge retrieval result"""
    category: str
    topic: str
    content: Dict[str, Any]
    relevance_score: float
    source_file: str


class IstanbulKnowledgeRAG:
    """
    Comprehensive knowledge retrieval system for Istanbul tourism.
    
    Loads and indexes all JSON knowledge files for fast retrieval.
    Provides semantic search across all knowledge areas.
    """
    
    # Knowledge categories and their keywords for matching
    CATEGORY_KEYWORDS = {
        'neighborhoods': [
            'neighborhood', 'district', 'area', 'where to stay', 'sultanahmet', 
            'beyoglu', 'kadikoy', 'besiktas', 'taksim', 'galata', 'karakoy',
            'fatih', 'uskudar', 'ortakoy', 'bebek', 'nisantasi', 'cihangir',
            'balat', 'fener', 'moda', 'live', 'stay', 'hipster', 'local',
            'hotel', 'accommodation', 'residential', 'asian side', 'european side'
        ],
        'scams': [
            'scam', 'fraud', 'trick', 'cheat', 'rip off', 'careful', 'avoid',
            'warning', 'danger', 'safe', 'safety', 'tourist trap', 'fake',
            'overcharge', 'shoe shine', 'tea invite', 'carpet', 'taxi meter',
            'watch out', 'beware', 'suspicious', 'common tricks'
        ],
        'turkish_phrases': [
            'turkish', 'language', 'phrase', 'word', 'say', 'speak', 'translate',
            'merhaba', 'tesekkur', 'hello', 'thank you', 'please', 'how to say',
            'pronunciation', 'basic', 'learn', 'communicate', 'local language'
        ],
        'transportation': [
            'metro', 'tram', 'bus', 'ferry', 'taxi', 'uber', 'transport',
            'istanbulkart', 'airport', 'transfer', 'marmaray', 'funicular',
            'how to get', 'route', 'direction', 'travel', 'commute', 'station',
            'get from', 'get to', 'reach', 'public transport'
        ],
        'food_and_dining': [
            'food', 'eat', 'eating', 'restaurant', 'cuisine', 'kebab', 'kebap',
            'breakfast', 'kahvalti', 'dinner', 'lunch', 'brunch', 'meal',
            'cafe', 'coffee', 'tea', 'baklava', 'lokanta', 'meyhane',
            'price', 'menu', 'vegetarian', 'vegan', 'halal', 'kosher',
            'street food', 'doner', 'pide', 'meze', 'raki', 'turkish delight',
            'best food', 'must eat', 'try', 'taste', 'hungry', 'delicious',
            'fish', 'seafood', 'meat', 'dessert', 'sweet', 'drink', 'simit',
            'lahmacun', 'kofte', 'borek', 'menemen', 'ayran', 'cheap eats'
        ],
        'attractions': [
            'visit', 'see', 'attraction', 'museum', 'mosque', 'palace',
            'hagia sophia', 'aya sofya', 'ayasofya', 'blue mosque', 'topkapi', 
            'galata', 'bazaar', 'cistern', 'bosphorus', 'cruise', 'tour', 
            'ticket', 'entrance', 'fee', 'cost', 'price', 'free',
            'opening hours', 'hamam', 'bath', 'dervish', 'historical', 'sight',
            'what to see', 'must see', 'worth visiting', 'landmark'
        ],
        'practical_info': [
            'money', 'currency', 'lira', 'atm', 'exchange', 'sim card', 'phone',
            'wifi', 'internet', 'hospital', 'pharmacy', 'emergency', 'police',
            'visa', 'passport', 'tip', 'tipping', 'plug', 'electricity',
            'toilet', 'restroom', 'dress code', 'what to wear', 'power adapter',
            'bargain', 'haggle', 'negotiate', 'cash', 'credit card'
        ],
        'seasonal_events': [
            'weather', 'season', 'best time', 'when to visit', 'festival',
            'holiday', 'ramadan', 'bayram', 'eid', 'new year', 'tulip',
            'summer', 'winter', 'spring', 'autumn', 'fall', 'crowd', 'busy', 'quiet',
            'temperature', 'rain', 'hot', 'cold', 'pack', 'clothes', 'climate',
            'what to pack', 'best month', 'avoid crowds'
        ]
    }
    
    def __init__(self, knowledge_dir: str = None):
        """
        Initialize the knowledge RAG system.
        
        Args:
            knowledge_dir: Path to the knowledge JSON files directory
        """
        if knowledge_dir is None:
            # Default to the data/knowledge directory
            knowledge_dir = os.path.join(
                os.path.dirname(__file__),
                '../data/knowledge'
            )
        
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_base: Dict[str, Dict] = {}
        self.search_index: Dict[str, List[Tuple[str, str, Any]]] = {}
        
        # Load all knowledge files
        self._load_knowledge_files()
        self._build_search_index()
        
        logger.info(f"✅ Istanbul Knowledge RAG initialized with {len(self.knowledge_base)} knowledge areas")
    
    def _load_knowledge_files(self):
        """Load all JSON knowledge files from the knowledge directory"""
        knowledge_files = {
            'neighborhoods': 'neighborhoods.json',
            'scams': 'scams.json',
            'turkish_phrases': 'turkish_phrases.json',
            'transportation': 'transportation.json',
            'food_and_dining': 'food_and_dining.json',
            'attractions': 'attractions.json',
            'practical_info': 'practical_info.json',
            'seasonal_events': 'seasonal_events.json'
        }
        
        for category, filename in knowledge_files.items():
            filepath = self.knowledge_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.knowledge_base[category] = json.load(f)
                    logger.info(f"   ✓ Loaded {category}: {filename}")
                except Exception as e:
                    logger.error(f"   ✗ Failed to load {filename}: {e}")
            else:
                logger.warning(f"   ⚠ Knowledge file not found: {filename}")
    
    def _build_search_index(self):
        """Build a searchable index from all knowledge content"""
        self.search_index = {}
        
        for category, data in self.knowledge_base.items():
            self.search_index[category] = []
            self._index_dict(category, data, [])
    
    def _index_dict(self, category: str, data: Any, path: List[str]):
        """Recursively index dictionary content"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = path + [key]
                
                # Index the key itself
                self.search_index[category].append((
                    '.'.join(new_path),
                    str(key).lower(),
                    value
                ))
                
                # Recurse into nested structures
                self._index_dict(category, value, new_path)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    self._index_dict(category, item, path + [str(i)])
                elif isinstance(item, str):
                    self.search_index[category].append((
                        '.'.join(path),
                        item.lower(),
                        item
                    ))
    
    def _detect_categories(self, query: str) -> List[str]:
        """Detect which knowledge categories are relevant to a query"""
        query_lower = query.lower()
        matches = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                matches.append((category, score))
        
        # Sort by score (descending) and return category names
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top matches (at least 1, max 3)
        if matches:
            return [m[0] for m in matches[:3]]
        
        # Default to most common categories if no match
        return ['attractions', 'food_and_dining', 'practical_info']
    
    def search(
        self,
        query: str,
        categories: List[str] = None,
        top_k: int = 5
    ) -> List[KnowledgeResult]:
        """
        Search for relevant knowledge across all categories.
        
        Args:
            query: User's query
            categories: Specific categories to search (auto-detected if None)
            top_k: Maximum number of results to return
        
        Returns:
            List of KnowledgeResult objects
        """
        if categories is None:
            categories = self._detect_categories(query)
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []
        
        for category in categories:
            if category not in self.search_index:
                continue
            
            for path, text, content in self.search_index[category]:
                # Calculate relevance score based on word overlap
                text_words = set(text.split())
                overlap = len(query_words & text_words)
                
                # Also check for substring matches
                substring_score = sum(1 for word in query_words if word in text)
                
                score = overlap + (substring_score * 0.5)
                
                if score > 0:
                    results.append(KnowledgeResult(
                        category=category,
                        topic=path,
                        content=content,
                        relevance_score=score,
                        source_file=f"{category}.json"
                    ))
        
        # Sort by relevance and return top_k
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:top_k]
    
    def get_context_for_llm(
        self,
        query: str,
        max_length: int = 2000,
        categories: List[str] = None
    ) -> str:
        """
        Get formatted knowledge context for LLM prompt.
        
        Args:
            query: User's query
            max_length: Maximum context length in characters
            categories: Specific categories (auto-detected if None)
        
        Returns:
            Formatted context string for LLM
        """
        if categories is None:
            categories = self._detect_categories(query)
        
        context_parts = []
        current_length = 0
        
        # Get relevant sections from each detected category
        for category in categories:
            if category not in self.knowledge_base:
                continue
            
            category_context = self._extract_relevant_context(
                category, 
                query, 
                max_length=(max_length - current_length) // len(categories)
            )
            
            if category_context:
                section = f"\n[{category.upper().replace('_', ' ')}]\n{category_context}"
                
                if current_length + len(section) <= max_length:
                    context_parts.append(section)
                    current_length += len(section)
        
        return '\n'.join(context_parts)
    
    def _extract_relevant_context(
        self,
        category: str,
        query: str,
        max_length: int = 500
    ) -> str:
        """Extract the most relevant context from a category using deep search"""
        data = self.knowledge_base.get(category, {})
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Remove common words for better matching
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'how', 'where', 'when', 'should', 'do', 'i', 'in', 'to', 'for', 'of', 'and', 'or'}
        query_keywords = query_words - stop_words
        
        relevant_sections = []
        
        # Deep recursive search for matching content
        self._find_relevant_content(data, query_keywords, [], relevant_sections, max_depth=4)
        
        if not relevant_sections:
            # Fallback: return summary of main sections
            for key, value in list(data.items())[:2]:
                if not key.startswith('metadata'):
                    relevant_sections.append((key, value, 0.5, [key]))
        
        # Sort by score (highest first)
        relevant_sections.sort(key=lambda x: x[2], reverse=True)
        
        # Format the context
        context = ""
        seen_paths = set()
        
        for section_name, section_data, score, path in relevant_sections[:5]:
            path_str = '.'.join(path)
            if path_str in seen_paths:
                continue
            seen_paths.add(path_str)
            
            formatted = self._format_section_deep(section_name, section_data, max_depth=2)
            if len(context) + len(formatted) <= max_length:
                context += formatted + "\n"
            elif len(context) < max_length * 0.3:
                # At least include some content
                context += formatted[:max_length - len(context) - 50] + "...\n"
                break
        
        return context.strip()
    
    def _find_relevant_content(
        self,
        data: Any,
        query_keywords: set,
        path: List[str],
        results: List[Tuple],
        max_depth: int = 4,
        current_depth: int = 0
    ):
        """Recursively find content matching query keywords"""
        if current_depth > max_depth:
            return
        
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower().replace('_', ' ')
                new_path = path + [key]
                
                # Calculate match score for this key
                key_words = set(key_lower.split())
                score = len(query_keywords & key_words) * 2  # Higher weight for key matches
                
                # Also check if query keywords appear in string representation
                if isinstance(value, str):
                    value_lower = value.lower()
                    score += sum(1 for kw in query_keywords if kw in value_lower)
                elif isinstance(value, dict):
                    # Check nested keys and values
                    nested_str = ' '.join(str(k).lower() + ' ' + str(v)[:100].lower() 
                                         for k, v in value.items() if isinstance(v, (str, int, float)))
                    score += sum(0.5 for kw in query_keywords if kw in nested_str)
                
                if score > 0:
                    results.append((key, value, score, new_path))
                
                # Recurse deeper
                self._find_relevant_content(value, query_keywords, new_path, results, max_depth, current_depth + 1)
                
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    self._find_relevant_content(item, query_keywords, path + [str(i)], results, max_depth, current_depth + 1)
                elif isinstance(item, str):
                    item_lower = item.lower()
                    score = sum(1 for kw in query_keywords if kw in item_lower)
                    if score > 0:
                        results.append((path[-1] if path else "item", item, score, path))
    
    def _format_section_deep(self, name: str, data: Any, depth: int = 0, max_depth: int = 2) -> str:
        """Format a section with controlled depth for concise output"""
        indent = "  " * depth
        
        if isinstance(data, dict):
            lines = [f"{indent}**{name.replace('_', ' ').title()}**:"]
            
            for key, value in list(data.items())[:8]:  # More items
                if key.startswith('metadata'):
                    continue
                    
                if isinstance(value, str):
                    # Truncate long strings
                    val_display = value[:150] + "..." if len(value) > 150 else value
                    lines.append(f"{indent}  • {key.replace('_', ' ')}: {val_display}")
                elif isinstance(value, (int, float)):
                    lines.append(f"{indent}  • {key.replace('_', ' ')}: {value}")
                elif isinstance(value, list):
                    if len(value) > 0:
                        if isinstance(value[0], str):
                            # Join short list items
                            items = [str(v)[:50] for v in value[:5]]
                            lines.append(f"{indent}  • {key.replace('_', ' ')}: {', '.join(items)}")
                        elif isinstance(value[0], dict):
                            lines.append(f"{indent}  • {key.replace('_', ' ')}: {len(value)} entries")
                elif isinstance(value, dict) and depth < max_depth:
                    nested = self._format_section_deep(key, value, depth + 1, max_depth)
                    lines.append(nested)
            
            return '\n'.join(lines[:12])  # Limit lines
        
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], str):
                items = [str(d)[:50] for d in data[:6]]
                return f"{indent}**{name.replace('_', ' ').title()}**: {', '.join(items)}"
            elif len(data) > 0 and isinstance(data[0], dict):
                # Format first item as example
                example = self._format_section_deep("Example", data[0], depth, max_depth - 1)
                return f"{indent}**{name.replace('_', ' ').title()}** ({len(data)} items):\n{example}"
            return f"{indent}**{name.replace('_', ' ').title()}**: {len(data)} items"
        
        else:
            val_str = str(data)[:150]
            return f"{indent}**{name.replace('_', ' ').title()}**: {val_str}"
    
    # ==========================================
    # Specific Knowledge Retrieval Methods
    # ==========================================
    
    def get_neighborhood_info(self, neighborhood: str) -> Optional[Dict]:
        """Get detailed information about a specific neighborhood"""
        data = self.knowledge_base.get('neighborhoods', {})
        neighborhood_lower = neighborhood.lower()
        
        # Search in neighborhoods data
        for key, value in data.items():
            if isinstance(value, dict):
                name = value.get('name', key)
                if neighborhood_lower in name.lower() or neighborhood_lower in key.lower():
                    return value
        
        return None
    
    def get_scam_warnings(self, context: str = None) -> List[Dict]:
        """Get relevant scam warnings, optionally filtered by context"""
        data = self.knowledge_base.get('scams', {})
        scams = data.get('common_scams', [])
        
        if context:
            context_lower = context.lower()
            return [s for s in scams if any(
                word in str(s).lower() for word in context_lower.split()
            )]
        
        return scams
    
    def get_food_info(self, food_type: str = None) -> Dict:
        """Get food and dining information"""
        data = self.knowledge_base.get('food_and_dining', {})
        
        if food_type:
            food_lower = food_type.lower()
            # Search for specific food type
            for key, value in data.items():
                if food_lower in key.lower() or food_lower in str(value).lower():
                    return {key: value}
        
        return data
    
    def get_attraction_info(self, attraction: str) -> Optional[Dict]:
        """Get information about a specific attraction"""
        data = self.knowledge_base.get('attractions', {})
        attraction_lower = attraction.lower()
        
        # Search in all attraction categories
        for category, attractions in data.items():
            if isinstance(attractions, dict):
                for key, value in attractions.items():
                    if isinstance(value, dict):
                        name = value.get('name', key)
                        if attraction_lower in name.lower() or attraction_lower in key.lower():
                            return value
        
        return None
    
    def get_practical_info(self, topic: str) -> Optional[Dict]:
        """Get practical information about a specific topic"""
        data = self.knowledge_base.get('practical_info', {})
        topic_lower = topic.lower()
        
        for key, value in data.items():
            if topic_lower in key.lower():
                return {key: value}
        
        return None
    
    def get_weather_info(self, month: str = None) -> Dict:
        """Get weather and seasonal information"""
        data = self.knowledge_base.get('seasonal_events', {})
        
        if month:
            month_lower = month.lower()
            weather = data.get('weather_by_month', {})
            if month_lower in weather:
                return weather[month_lower]
        
        return data.get('weather_by_month', {})
    
    def get_phrase(self, english: str) -> Optional[Dict]:
        """Get Turkish translation for an English phrase"""
        data = self.knowledge_base.get('turkish_phrases', {})
        english_lower = english.lower()
        
        # Search through phrase categories
        for category, phrases in data.items():
            if isinstance(phrases, dict):
                for key, value in phrases.items():
                    if english_lower in key.lower() or english_lower in str(value).lower():
                        return {key: value}
        
        return None
    
    def get_price_guide(self, item_type: str = None) -> Dict:
        """Get price information, optionally for a specific type"""
        food_data = self.knowledge_base.get('food_and_dining', {})
        price_guide = food_data.get('price_guide_2024', {})
        
        if item_type:
            item_lower = item_type.lower()
            return {k: v for k, v in price_guide.items() if item_lower in k.lower()}
        
        return price_guide


# ==========================================
# Singleton Instance
# ==========================================

_knowledge_rag_singleton = None


def get_knowledge_rag() -> IstanbulKnowledgeRAG:
    """
    Get or create Knowledge RAG instance (singleton).
    
    Returns:
        IstanbulKnowledgeRAG instance
    """
    global _knowledge_rag_singleton
    
    if _knowledge_rag_singleton is None:
        logger.info("🚀 Creating Istanbul Knowledge RAG instance...")
        _knowledge_rag_singleton = IstanbulKnowledgeRAG()
    
    return _knowledge_rag_singleton


# ==========================================
# Integration Helper for LLM
# ==========================================

def enhance_prompt_with_knowledge(
    query: str,
    base_prompt: str = "",
    max_context_length: int = 2000
) -> str:
    """
    Enhance an LLM prompt with relevant Istanbul knowledge.
    
    Args:
        query: User's query
        base_prompt: Existing prompt to enhance
        max_context_length: Maximum characters of context to add
    
    Returns:
        Enhanced prompt with knowledge context
    """
    rag = get_knowledge_rag()
    context = rag.get_context_for_llm(query, max_length=max_context_length)
    
    if not context:
        return base_prompt + "\n\n" + query if base_prompt else query
    
    enhanced = f"""{base_prompt}

RELEVANT ISTANBUL KNOWLEDGE:
{context}

USER QUESTION: {query}

Use the knowledge above to provide an accurate, helpful response about Istanbul.
"""
    
    return enhanced


# ==========================================
# CLI for Testing
# ==========================================

if __name__ == "__main__":
    import sys
    
    # Initialize RAG
    rag = get_knowledge_rag()
    
    # Print stats
    print("\n📊 Knowledge Base Statistics:")
    print("-" * 40)
    for category, data in rag.knowledge_base.items():
        count = len(rag.search_index.get(category, []))
        print(f"  {category}: {count} indexed items")
    
    # Test queries
    test_queries = [
        "Where should I stay in Istanbul?",
        "What scams should I watch out for?",
        "How do I say thank you in Turkish?",
        "How do I get from the airport to Sultanahmet?",
        "What is the best kebab in Istanbul?",
        "Is Hagia Sophia free to visit?",
        "What's the best time to visit Istanbul?",
        "How much should I tip at restaurants?"
    ]
    
    print("\n🔍 Test Queries:")
    print("-" * 40)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        categories = rag._detect_categories(query)
        print(f"  Detected categories: {categories}")
        
        results = rag.search(query, top_k=2)
        for r in results:
            print(f"  → {r.category}: {r.topic} (score: {r.relevance_score:.1f})")
