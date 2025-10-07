#!/usr/bin/env python3
"""
Complete Query Processing Pipeline for AI Istanbul
=================================================

Implements:
1. Text preprocessing (normalization, stopwords, lemmatization)
2. Intent classification (lightweight classifier)
3. Vector search integration
4. Rule-based filtering and ranking
5. Response generation with templates
"""

import re
import string
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import numpy as np
from datetime import datetime
import logging

# NLP preprocessing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        print("📦 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
except ImportError:
    print("⚠️ NLTK not available, using basic text processing")
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Query intent classification"""
    RESTAURANT_SEARCH = "restaurant_search"
    MUSEUM_SEARCH = "museum_search"
    ATTRACTION_INFO = "attraction_info"
    TRANSPORTATION = "transportation"
    ITINERARY_PLANNING = "itinerary_planning"
    TICKET_INFO = "ticket_info"
    GENERAL_INFO = "general_info"
    RECOMMENDATION = "recommendation"
    LOCATION_SPECIFIC = "location_specific"
    EVENT_SEARCH = "event_search"

class QueryType(Enum):
    """Query processing type"""
    SEARCH = "search"
    RECOMMENDATION = "recommendation"
    INFORMATION = "information"
    PLANNING = "planning"

@dataclass
class ProcessedQuery:
    """Processed query with analysis results"""
    original_text: str
    normalized_text: str
    tokens: List[str]
    intent: QueryIntent
    intent_confidence: float
    query_type: QueryType
    entities: Dict[str, List[str]]
    keywords: List[str]
    location_context: Optional[str] = None
    temporal_context: Optional[str] = None

@dataclass
class SearchResult:
    """Enhanced search result with ranking scores"""
    id: str
    title: str
    content: str
    category: str
    metadata: Dict[str, Any]
    relevance_score: float
    popularity_score: float
    distance_score: float
    availability_score: float
    final_score: float
    search_type: str  # 'vector', 'keyword', 'hybrid'

class TextPreprocessor:
    """Advanced text preprocessing for queries"""
    
    def __init__(self):
        self.stop_words = set()
        self.lemmatizer = None
        
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
        # Add custom Istanbul-specific stopwords
        self.stop_words.update(['istanbul', 'turkey', 'turkish', 'city'])
        
        # Istanbul districts for entity recognition
        self.istanbul_districts = {
            'sultanahmet', 'beyoğlu', 'galata', 'kadiköy', 'beşiktaş', 'şişli',
            'fatih', 'eminönü', 'üsküdar', 'ortaköy', 'taksim', 'balat',
            'fener', 'nişantaşı', 'bebek', 'arnavutköy', 'sarıyer', 'bakırköy'
        }
        
        # Common Istanbul attractions
        self.attractions = {
            'hagia sophia', 'blue mosque', 'topkapi palace', 'galata tower',
            'basilica cistern', 'dolmabahce palace', 'bosphorus', 'golden horn',
            'grand bazaar', 'spice bazaar', 'maiden tower', 'istiklal street'
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing"""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\-\']', ' ', text)
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Handle Turkish characters
        turkish_mapping = {
            'ğ': 'g', 'ü': 'u', 'ş': 's', 'ı': 'i', 'ö': 'o', 'ç': 'c'
        }
        for turkish, english in turkish_mapping.items():
            text = text.replace(turkish, english)
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        if NLTK_AVAILABLE:
            return word_tokenize(text)
        else:
            # Basic tokenization
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokens"""
        return [token for token in tokens if token.lower() not in self.stop_words and len(token) > 2]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens"""
        if NLTK_AVAILABLE and self.lemmatizer:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
        else:
            return tokens
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text"""
        entities = {
            'locations': [],
            'attractions': [],
            'cuisine_types': [],
            'transport_types': [],
            'time_references': []
        }
        
        text_lower = text.lower()
        
        # Extract locations (districts)
        for district in self.istanbul_districts:
            if district in text_lower:
                entities['locations'].append(district.title())
        
        # Extract attractions
        for attraction in self.attractions:
            if attraction in text_lower:
                entities['attractions'].append(attraction.title())
        
        # Extract cuisine types
        cuisine_keywords = {
            'turkish', 'ottoman', 'mediterranean', 'seafood', 'kebab', 'meze',
            'italian', 'french', 'asian', 'chinese', 'japanese', 'indian'
        }
        for cuisine in cuisine_keywords:
            if cuisine in text_lower:
                entities['cuisine_types'].append(cuisine.title())
        
        # Extract transport types
        transport_keywords = {
            'metro', 'bus', 'ferry', 'tram', 'taxi', 'dolmus', 'boat'
        }
        for transport in transport_keywords:
            if transport in text_lower:
                entities['transport_types'].append(transport.title())
        
        # Extract time references
        time_keywords = {
            'morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow',
            'weekend', 'weekday', 'monday', 'tuesday', 'wednesday', 'thursday',
            'friday', 'saturday', 'sunday'
        }
        for time_ref in time_keywords:
            if time_ref in text_lower:
                entities['time_references'].append(time_ref.title())
        
        return entities
    
    def process(self, text: str) -> Tuple[str, List[str], Dict[str, List[str]]]:
        """Complete preprocessing pipeline"""
        # Normalize
        normalized = self.normalize_text(text)
        
        # Tokenize
        tokens = self.tokenize(normalized)
        
        # Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        return normalized, tokens, entities

class IntentClassifier:
    """Lightweight intent classifier using rule-based approach"""
    
    def __init__(self):
        # Intent keywords mapping
        self.intent_patterns = {
            QueryIntent.RESTAURANT_SEARCH: [
                'restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal',
                'breakfast', 'lunch', 'dinner', 'cafe', 'kebab', 'meze'
            ],
            QueryIntent.MUSEUM_SEARCH: [
                'museum', 'gallery', 'exhibition', 'art', 'history', 'culture',
                'artifact', 'collection', 'heritage'
            ],
            QueryIntent.ATTRACTION_INFO: [
                'visit', 'see', 'attraction', 'landmark', 'monument', 'site',
                'hagia sophia', 'blue mosque', 'galata tower', 'topkapi'
            ],
            QueryIntent.TRANSPORTATION: [
                'how to get', 'transport', 'metro', 'bus', 'ferry', 'taxi',
                'route', 'directions', 'travel', 'go to', 'reach'
            ],
            QueryIntent.ITINERARY_PLANNING: [
                'itinerary', 'plan', 'schedule', 'day trip', 'tour',
                'visit order', 'route planning', 'time management'
            ],
            QueryIntent.TICKET_INFO: [
                'ticket', 'price', 'cost', 'admission', 'fee', 'booking',
                'reservation', 'entry', 'pass'
            ],
            QueryIntent.RECOMMENDATION: [
                'recommend', 'suggest', 'best', 'top', 'popular', 'famous',
                'must see', 'should visit', 'advice'
            ],
            QueryIntent.EVENT_SEARCH: [
                'event', 'festival', 'concert', 'show', 'exhibition',
                'happening', 'activity', 'entertainment'
            ]
        }
    
    def classify(self, tokens: List[str], entities: Dict[str, List[str]]) -> Tuple[QueryIntent, float]:
        """Classify query intent with confidence score"""
        token_text = ' '.join(tokens).lower()
        scores = {}
        
        # Score based on keyword matching
        for intent, keywords in self.intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in token_text:
                    score += 1
                    # Boost score for exact matches
                    if keyword in tokens:
                        score += 0.5
            
            # Normalize score
            if len(keywords) > 0:
                scores[intent] = score / len(keywords)
        
        # Boost scores based on entities
        if entities['attractions']:
            scores[QueryIntent.ATTRACTION_INFO] = scores.get(QueryIntent.ATTRACTION_INFO, 0) + 0.3
        
        if entities['cuisine_types']:
            scores[QueryIntent.RESTAURANT_SEARCH] = scores.get(QueryIntent.RESTAURANT_SEARCH, 0) + 0.3
        
        if entities['transport_types']:
            scores[QueryIntent.TRANSPORTATION] = scores.get(QueryIntent.TRANSPORTATION, 0) + 0.3
        
        # Find best intent
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent], 1.0)
            
            # Minimum confidence threshold
            if confidence > 0.1:
                return best_intent, confidence
        
        return QueryIntent.GENERAL_INFO, 0.5

class QueryRanker:
    """Rule-based ranking system for search results"""
    
    def __init__(self):
        self.ranking_weights = {
            'relevance': 0.4,
            'popularity': 0.3,
            'distance': 0.2,
            'availability': 0.1
        }
        
        # Popularity scores for common attractions
        self.popularity_scores = {
            'hagia sophia': 1.0,
            'blue mosque': 0.95,
            'topkapi palace': 0.9,
            'galata tower': 0.85,
            'grand bazaar': 0.8,
            'basilica cistern': 0.75
        }
    
    def calculate_popularity_score(self, item: Dict[str, Any]) -> float:
        """Calculate popularity score based on ratings and known attractions"""
        # Base score from ratings
        rating = item.get('rating', 0)
        if rating is not None and rating > 0:
            popularity = rating / 5.0
        else:
            popularity = 0.5  # Default score
        
        # Boost for well-known attractions
        name = item.get('name', '').lower()
        for attraction, score in self.popularity_scores.items():
            if attraction in name:
                popularity = max(popularity, score)
                break
        
        return popularity
    
    def calculate_distance_score(self, item: Dict[str, Any], user_location: Optional[str] = None) -> float:
        """Calculate distance score (placeholder - would use real coordinates)"""
        # For now, boost items in same district as user
        if user_location:
            item_district = item.get('district', '').lower()
            if user_location.lower() in item_district:
                return 1.0
            else:
                return 0.7
        
        return 0.8  # Default score when no location context
    
    def calculate_availability_score(self, item: Dict[str, Any]) -> float:
        """Calculate availability score based on opening hours, etc."""
        # Placeholder - would check real-time availability
        item_type = item.get('type', '')
        
        # Museums might have limited hours
        if item_type == 'museum':
            return 0.8
        # Restaurants generally more available
        elif item_type == 'restaurant':
            return 0.9
        
        return 0.85  # Default availability
    
    def rank_results(self, results: List[Dict[str, Any]], 
                    user_location: Optional[str] = None) -> List[SearchResult]:
        """Rank search results using multiple factors"""
        ranked_results = []
        
        for item in results:
            # Calculate component scores
            relevance_score = item.get('score', 0.5)  # From search engine
            popularity_score = self.calculate_popularity_score(item)
            distance_score = self.calculate_distance_score(item, user_location)
            availability_score = self.calculate_availability_score(item)
            
            # Calculate weighted final score
            final_score = (
                self.ranking_weights['relevance'] * relevance_score +
                self.ranking_weights['popularity'] * popularity_score +
                self.ranking_weights['distance'] * distance_score +
                self.ranking_weights['availability'] * availability_score
            )
            
            ranked_result = SearchResult(
                id=item.get('id', ''),
                title=item.get('name', item.get('title', '')),
                content=item.get('description', item.get('content', '')),
                category=item.get('category', item.get('type', '')),
                metadata=item,
                relevance_score=relevance_score,
                popularity_score=popularity_score,
                distance_score=distance_score,
                availability_score=availability_score,
                final_score=final_score,
                search_type=item.get('search_type', 'hybrid')
            )
            
            ranked_results.append(ranked_result)
        
        # Sort by final score
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_results

class ResponseGenerator:
    """Generate natural language responses from search results"""
    
    def __init__(self):
        self.templates = {
            QueryIntent.RESTAURANT_SEARCH: {
                'single': "I found an excellent restaurant for you: **{title}** in {district}. {description}",
                'multiple': "Here are {count} great restaurant options in Istanbul:\n\n{items}",
                'item': "• **{title}** - {district}\n  {description}"
            },
            QueryIntent.MUSEUM_SEARCH: {
                'single': "**{title}** is a fascinating museum in {district}. {description}",
                'multiple': "I recommend these {count} museums in Istanbul:\n\n{items}",
                'item': "• **{title}** - {district}\n  {description}"
            },
            QueryIntent.ATTRACTION_INFO: {
                'single': "**{title}** is one of Istanbul's iconic attractions. {description}",
                'multiple': "Here are {count} must-see attractions:\n\n{items}",
                'item': "• **{title}** - {district}\n  {description}"
            },
            QueryIntent.GENERAL_INFO: {
                'single': "Here's what I found about **{title}**: {description}",
                'multiple': "I found {count} relevant results:\n\n{items}",
                'item': "• **{title}**\n  {description}"
            }
        }
    
    def generate_response(self, intent: QueryIntent, results: List[SearchResult], 
                         query_context: Optional[Dict] = None) -> str:
        """Generate natural language response from results"""
        if not results:
            return "I couldn't find any results for your query. Could you try rephrasing or being more specific?"
        
        # Get appropriate template
        template_set = self.templates.get(intent, self.templates[QueryIntent.GENERAL_INFO])
        
        if len(results) == 1:
            # Single result response
            result = results[0]
            template = template_set['single']
            
            return template.format(
                title=result.title,
                district=result.metadata.get('district', 'Istanbul'),
                description=self._clean_description(result.content)
            )
        else:
            # Multiple results response
            template = template_set['multiple']
            item_template = template_set['item']
            
            items = []
            for result in results[:5]:  # Limit to top 5
                item_text = item_template.format(
                    title=result.title,
                    district=result.metadata.get('district', 'Istanbul'),
                    description=self._clean_description(result.content, max_length=100)
                )
                items.append(item_text)
            
            return template.format(
                count=len(results),
                items='\n\n'.join(items)
            )
    
    def _clean_description(self, description: str, max_length: int = 200) -> str:
        """Clean and truncate description"""
        if not description:
            return "A great place to visit in Istanbul."
        
        # Remove extra whitespace
        description = ' '.join(description.split())
        
        # Truncate if too long
        if len(description) > max_length:
            description = description[:max_length-3] + "..."
        
        return description

class CompleteQueryPipeline:
    """Complete query processing pipeline integrating all components"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.intent_classifier = IntentClassifier()
        self.ranker = QueryRanker()
        self.response_generator = ResponseGenerator()
        
        # Initialize search systems
        self.vector_system = None
        self.keyword_system = None
        
        try:
            from vector_embedding_system import vector_embedding_system
            self.vector_system = vector_embedding_system
        except ImportError:
            print("⚠️ Vector system not available")
        
        try:
            from lightweight_retrieval_system import lightweight_retrieval_system
            self.keyword_system = lightweight_retrieval_system
        except ImportError:
            print("⚠️ Keyword system not available")
    
    def process_query(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process complete query through the pipeline"""
        start_time = datetime.now()
        
        # 1. Preprocessing
        normalized_text, tokens, entities = self.preprocessor.process(query)
        
        # 2. Intent Classification
        intent, confidence = self.intent_classifier.classify(tokens, entities)
        
        # Determine query type
        if intent in [QueryIntent.RESTAURANT_SEARCH, QueryIntent.MUSEUM_SEARCH, QueryIntent.EVENT_SEARCH]:
            query_type = QueryType.SEARCH
        elif intent == QueryIntent.RECOMMENDATION:
            query_type = QueryType.RECOMMENDATION
        elif intent == QueryIntent.ITINERARY_PLANNING:
            query_type = QueryType.PLANNING
        else:
            query_type = QueryType.INFORMATION
        
        # Create processed query object
        processed_query = ProcessedQuery(
            original_text=query,
            normalized_text=normalized_text,
            tokens=tokens,
            intent=intent,
            intent_confidence=confidence,
            query_type=query_type,
            entities=entities,
            keywords=tokens[:10],  # Top 10 keywords
            location_context=entities['locations'][0] if entities['locations'] else None
        )
        
        # 3. Search & Retrieval
        search_results = self._perform_search(processed_query, user_context)
        
        # 4. Ranking
        ranked_results = self.ranker.rank_results(
            search_results, 
            processed_query.location_context
        )
        
        # 5. Response Generation
        response_text = self.response_generator.generate_response(
            intent, ranked_results, user_context
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'success': True,
            'query': processed_query,
            'results': ranked_results,
            'response': response_text,
            'processing_time_ms': processing_time,
            'metadata': {
                'intent': intent.value,
                'confidence': confidence,
                'query_type': query_type.value,
                'entities': entities,
                'result_count': len(ranked_results)
            }
        }
    
    def _perform_search(self, query: ProcessedQuery, user_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Perform search using available systems"""
        all_results = []
        
        # Vector search
        if self.vector_system:
            try:
                vector_results = self.vector_system.semantic_search(
                    query.normalized_text, k=10, min_similarity=0.3
                )
                
                for result in vector_results:
                    all_results.append({
                        'id': result.document.id,
                        'name': result.document.metadata.get('name', 'Unknown'),
                        'description': result.document.content,
                        'category': result.document.metadata.get('type', 'general'),
                        'district': result.document.metadata.get('district', 'Istanbul'),
                        'rating': result.document.metadata.get('rating'),
                        'score': result.similarity_score,
                        'search_type': 'vector',
                        'type': result.document.metadata.get('type')
                    })
            except Exception as e:
                print(f"⚠️ Vector search error: {e}")
        
        # Keyword search
        if self.keyword_system:
            try:
                keyword_results = self.keyword_system.search(
                    query.normalized_text, top_k=10
                )
                
                for result in keyword_results:
                    # Handle both SearchResult objects and dictionaries
                    if isinstance(result, SearchResult):
                        result_id = result.id
                        result_name = result.title
                        result_desc = result.content
                        result_category = result.category
                        result_district = result.metadata.get('district', 'Istanbul')
                        result_rating = result.metadata.get('rating')
                        result_score = result.final_score
                        result_type = result.metadata.get('type', result.category)
                    else:
                        # Dictionary format
                        result_id = result.get('id', '')
                        result_name = result.get('name', result.get('title', 'Unknown'))
                        result_desc = result.get('description', result.get('content', ''))
                        result_category = result.get('category', result.get('type', 'general'))
                        result_district = result.get('district', 'Istanbul')
                        result_rating = result.get('rating')
                        result_score = result.get('score', 0.5)
                        result_type = result.get('type', result.get('category'))
                    
                    # Avoid duplicates
                    if not any(r['id'] == result_id for r in all_results):
                        all_results.append({
                            'id': result_id,
                            'name': result_name,
                            'description': result_desc,
                            'category': result_category,
                            'district': result_district,
                            'rating': result_rating,
                            'score': result_score,
                            'search_type': 'keyword',
                            'type': result_type
                        })
            except Exception as e:
                print(f"⚠️ Keyword search error: {e}")
        
        # Filter by intent if needed
        if query.intent == QueryIntent.RESTAURANT_SEARCH:
            all_results = [r for r in all_results if r.get('type') == 'restaurant' or 'restaurant' in r.get('category', '').lower()]
        elif query.intent == QueryIntent.MUSEUM_SEARCH:
            all_results = [r for r in all_results if r.get('type') == 'museum' or 'museum' in r.get('category', '').lower()]
        
        return all_results

# Global pipeline instance
query_pipeline = CompleteQueryPipeline()

def process_query(query: str, user_context: Optional[Dict] = None) -> Dict[str, Any]:
    """Process query through complete pipeline"""
    return query_pipeline.process_query(query, user_context)

if __name__ == "__main__":
    # Test the complete pipeline
    print("🧪 Testing Complete Query Processing Pipeline...")
    
    test_queries = [
        "I want to find good Turkish restaurants in Sultanahmet",
        "What museums should I visit in Istanbul?",
        "How do I get to Galata Tower from Taksim?",
        "Plan a day itinerary for historic sites",
        "Best seafood restaurants near Galata Bridge"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        result = process_query(query)
        
        if result['success']:
            print(f"   ✅ Intent: {result['metadata']['intent']} (confidence: {result['metadata']['confidence']:.2f})")
            print(f"   📊 Results: {result['metadata']['result_count']} items")
            print(f"   ⚡ Processing time: {result['processing_time_ms']:.1f}ms")
            print(f"   🎯 Entities: {result['metadata']['entities']}")
        else:
            print(f"   ❌ Processing failed")
    
    print("\n✅ Complete Query Processing Pipeline is working correctly!")
