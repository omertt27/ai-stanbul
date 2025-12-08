#!/usr/bin/env python3
"""
Lightweight Retrieval System for Istanbul AI - No Heavy Dependencies
==================================================================

This system implements a retrieval-first design using:
1. TF-IDF based semantic matching
2. Keyword indexing and fuzzy matching
3. Hybrid scoring without heavy ML dependencies
4. Fast content search and ranking
"""

import re
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import math

@dataclass
class SearchResult:
    """Result from retrieval search"""
    content_id: str
    content_type: str  # 'restaurant', 'museum', 'event', 'transport'
    title: str
    description: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class ContentItem:
    """Content item for retrieval search"""
    content_id: str
    content_type: str
    title: str
    description: str
    searchable_text: str
    keywords: List[str]
    metadata: Dict[str, Any]
    tf_idf_vector: Optional[Dict[str, float]] = None

class LightweightRetrievalSystem:
    """Lightweight retrieval system without heavy ML dependencies"""
    
    def __init__(self):
        self.content_items = {}  # content_id -> ContentItem
        self.keyword_index = defaultdict(set)  # keyword -> set of content_ids
        self.term_frequencies = defaultdict(int)  # term -> document frequency
        self.vocabulary = set()
        self.initialized = False
        
        # Initialize the system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the lightweight retrieval system"""
        try:
            print("🚀 Initializing Lightweight Retrieval System...")
            
            # Load and index content
            self._load_and_index_content()
            
            # Build TF-IDF vectors
            self._build_tfidf_vectors()
            
            # Build keyword index
            self._build_keyword_index()
            
            self.initialized = True
            print("✅ Lightweight Retrieval System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing retrieval system: {e}")
            self.initialized = False
    
    def _load_and_index_content(self):
        """Load all content and create searchable index"""
        
        # Load restaurant data
        self._index_restaurants()
        
        # Load museum data  
        self._index_museums()
        
        # Load transport data
        self._index_transport()
        
        # Load event/cultural data
        self._index_events()
        
        print(f"📊 Indexed {len(self.content_items)} content items")
    
    def _index_restaurants(self):
        """Index restaurant data for retrieval"""
        
        # Try to load restaurant database
        # Note: In production, we use PostgreSQL instead of JSON files
        try:
            data_path = Path(__file__).parent / "data" / "restaurants_database.json"
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    restaurants = data.get('restaurants', [])
            else:
                # Fallback to sample data (expected in production)
                restaurants = self._get_sample_restaurants()
        except Exception:
            restaurants = self._get_sample_restaurants()
        
        for i, restaurant in enumerate(restaurants):
            # Create searchable text combining all relevant fields
            searchable_parts = [
                restaurant.get('name', ''),
                restaurant.get('cuisine', ''),
                restaurant.get('district', ''),
                restaurant.get('description', ''),
                ' '.join(restaurant.get('tags', [])),
                str(restaurant.get('atmosphere', '')),
                restaurant.get('specialty', ''),
                restaurant.get('location', '')
            ]
            
            searchable_text = ' '.join(filter(None, searchable_parts)).lower()
            
            # Extract keywords
            keywords = self._extract_keywords(searchable_text)
            keywords.extend([tag.lower() for tag in restaurant.get('tags', [])])
            keywords.append(restaurant.get('cuisine', '').lower())
            keywords.append(restaurant.get('district', '').lower())
            
            content = ContentItem(
                content_id=f"restaurant_{i}",
                content_type="restaurant",
                title=restaurant.get('name', f'Restaurant {i}'),
                description=restaurant.get('description', ''),
                searchable_text=searchable_text,
                keywords=[k for k in keywords if k],
                metadata=restaurant
            )
            
            self.content_items[content.content_id] = content
    
    def _index_museums(self):
        """Index museum data for retrieval"""
        
        try:
            from accurate_museum_database import istanbul_museums
            museums = istanbul_museums.museums
        except ImportError:
            museums = {}
        
        for key, museum in museums.items():
            # Create comprehensive searchable text
            searchable_parts = [
                museum.name,
                museum.location,
                museum.historical_period,
                museum.architectural_style,
                museum.historical_significance,
                ' '.join(museum.key_features),
                ' '.join(museum.must_see_highlights),
                getattr(museum, 'visitor_tips', ''),
                getattr(museum, 'cultural_context', '')
            ]
            
            searchable_text = ' '.join(filter(None, searchable_parts)).lower()
            
            # Extract keywords
            keywords = self._extract_keywords(searchable_text)
            keywords.extend([feature.lower() for feature in museum.key_features])
            keywords.extend([museum.historical_period.lower(), museum.architectural_style.lower()])
            
            # Add location keywords
            location_parts = museum.location.split(',')
            keywords.extend([part.strip().lower() for part in location_parts])
            
            content = ContentItem(
                content_id=f"museum_{key}",
                content_type="museum",
                title=museum.name,
                description=f"{museum.historical_significance} Located in {museum.location}.",
                searchable_text=searchable_text,
                keywords=[k for k in keywords if k],
                metadata={
                    'name': museum.name,
                    'location': museum.location,
                    'historical_period': museum.historical_period,
                    'opening_hours': museum.opening_hours,
                    'entrance_fee': museum.entrance_fee,
                    'key_features': museum.key_features,
                    'must_see_highlights': museum.must_see_highlights
                }
            )
            
            self.content_items[content.content_id] = content
    
    def _index_transport(self):
        """Index transportation data"""
        
        transport_data = [
            {
                'name': 'Istanbul Metro System',
                'type': 'metro',
                'description': 'Comprehensive metro network connecting major districts with M1, M2, M3, M4, M5, M6, M7 lines',
                'keywords': ['metro', 'subway', 'underground', 'public transport', 'istanbulkart', 'rail', 'train']
            },
            {
                'name': 'Bosphorus Ferry System',
                'type': 'ferry', 
                'description': 'Ferry services connecting European and Asian sides via Eminönü-Üsküdar, Karaköy-Kadıköy routes',
                'keywords': ['ferry', 'vapur', 'bosphorus', 'boat', 'sea transport', 'cross continent', 'water', 'scenic']
            },
            {
                'name': 'Istanbul Tram Network',
                'type': 'tram',
                'description': 'Tram lines T1, T3, T4, T5 serving tourist areas and city center including Sultanahmet',
                'keywords': ['tram', 'tramvay', 'rail', 'sultanahmet', 'historic', 'tourist', 'city center']
            },
            {
                'name': 'Metrobus BRT System',
                'type': 'bus',
                'description': 'High-speed bus rapid transit system connecting European and Asian sides',
                'keywords': ['metrobus', 'bus', 'rapid transit', 'brt', 'fast', 'cross continental']
            }
        ]
        
        for i, transport in enumerate(transport_data):
            searchable_text = f"{transport['name']} {transport['description']} {' '.join(transport['keywords'])}".lower()
            
            content = ContentItem(
                content_id=f"transport_{i}",
                content_type="transport",
                title=transport['name'],
                description=transport['description'],
                searchable_text=searchable_text,
                keywords=transport['keywords'],
                metadata=transport
            )
            
            self.content_items[content.content_id] = content
    
    def _index_events(self):
        """Index cultural events and activities"""
        
        events_data = [
            {
                'name': 'Turkish Bath Experience',
                'type': 'cultural_activity',
                'description': 'Traditional hammam experience in historic bathhouses with relaxation and cultural immersion',
                'keywords': ['hammam', 'turkish bath', 'spa', 'relaxation', 'traditional', 'cultural', 'wellness']
            },
            {
                'name': 'Bosphorus Sunset Cruise',
                'type': 'tour',
                'description': 'Evening boat tour with spectacular sunset views over the Bosphorus strait',
                'keywords': ['cruise', 'sunset', 'boat tour', 'bosphorus', 'romantic', 'sightseeing', 'evening', 'scenic']
            },
            {
                'name': 'Turkish Cooking Class',
                'type': 'cultural_activity',
                'description': 'Learn to cook traditional Turkish dishes like kebab, meze, and baklava',
                'keywords': ['cooking', 'food', 'class', 'turkish cuisine', 'hands-on', 'cultural', 'learning', 'kebab']
            },
            {
                'name': 'Istanbul Food Walking Tour',
                'type': 'tour',
                'description': 'Guided tour exploring local food scenes, street food, and authentic restaurants',
                'keywords': ['food tour', 'walking', 'street food', 'local', 'taste', 'guide', 'authentic', 'restaurants']
            }
        ]
        
        for i, event in enumerate(events_data):
            searchable_text = f"{event['name']} {event['description']} {' '.join(event['keywords'])}".lower()
            
            content = ContentItem(
                content_id=f"event_{i}",
                content_type="event",
                title=event['name'],
                description=event['description'],
                searchable_text=searchable_text,
                keywords=event['keywords'],
                metadata=event
            )
            
            self.content_items[content.content_id] = content
    
    def _build_tfidf_vectors(self):
        """Build TF-IDF vectors for all content"""
        
        # Build vocabulary and document frequencies
        for content in self.content_items.values():
            terms = self._tokenize(content.searchable_text)
            unique_terms = set(terms)
            
            for term in unique_terms:
                self.term_frequencies[term] += 1
                self.vocabulary.add(term)
        
        total_docs = len(self.content_items)
        
        # Calculate TF-IDF for each document
        for content in self.content_items.values():
            tf_idf_vector = {}
            terms = self._tokenize(content.searchable_text)
            term_counts = Counter(terms)
            
            for term in set(terms):
                # Term frequency
                tf = term_counts[term] / len(terms)
                
                # Inverse document frequency
                idf = math.log(total_docs / (self.term_frequencies[term] + 1))
                
                # TF-IDF score
                tf_idf_vector[term] = tf * idf
            
            content.tf_idf_vector = tf_idf_vector
        
        print(f"📈 Built TF-IDF vectors with vocabulary size: {len(self.vocabulary)}")
    
    def _build_keyword_index(self):
        """Build keyword index for fast lookup"""
        
        for content_id, content in self.content_items.items():
            # Index explicit keywords
            for keyword in content.keywords:
                self.keyword_index[keyword.lower()].add(content_id)
            
            # Index terms from TF-IDF
            if content.tf_idf_vector:
                for term in content.tf_idf_vector.keys():
                    self.keyword_index[term].add(content_id)
        
        print(f"🔍 Built keyword index with {len(self.keyword_index)} terms")
    
    def search(self, query: str, content_types: Optional[List[str]] = None, 
               top_k: int = 10) -> List[SearchResult]:
        """
        Perform retrieval-first search using TF-IDF and keyword matching
        
        Args:
            query: Search query
            content_types: Filter by content types
            top_k: Number of results to return
        """
        
        if not self.initialized:
            print("⚠️ Retrieval system not initialized")
            return []
        
        # Tokenize query
        query_terms = self._tokenize(query.lower())
        query_counter = Counter(query_terms)
        
        # Calculate scores for all content
        content_scores = {}
        
        for content_id, content in self.content_items.items():
            # Skip if content type filter doesn't match
            if content_types and content.content_type not in content_types:
                continue
            
            # Calculate TF-IDF similarity
            tfidf_score = self._calculate_tfidf_similarity(query_counter, content.tf_idf_vector)
            
            # Calculate keyword match score
            keyword_score = self._calculate_keyword_score(query_terms, content.keywords)
            
            # Calculate fuzzy match score
            fuzzy_score = self._calculate_fuzzy_score(query, content.searchable_text)
            
            # Combine scores (weighted)
            combined_score = (0.5 * tfidf_score) + (0.3 * keyword_score) + (0.2 * fuzzy_score)
            
            if combined_score > 0:
                content_scores[content_id] = combined_score
        
        # Sort by score and create results
        sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for content_id, score in sorted_content[:top_k]:
            content = self.content_items[content_id]
            results.append(SearchResult(
                content_id=content.content_id,
                content_type=content.content_type,
                title=content.title,
                description=content.description,
                score=score,
                metadata=content.metadata
            ))
        
        return results
    
    def _calculate_tfidf_similarity(self, query_counter: Counter, doc_vector: Dict[str, float]) -> float:
        """Calculate TF-IDF cosine similarity"""
        
        if not doc_vector:
            return 0.0
        
        # Calculate query TF-IDF vector
        query_vector = {}
        total_query_terms = sum(query_counter.values())
        
        for term, count in query_counter.items():
            if term in self.vocabulary:
                tf = count / total_query_terms
                idf = math.log(len(self.content_items) / (self.term_frequencies[term] + 1))
                query_vector[term] = tf * idf
        
        # Calculate cosine similarity
        dot_product = sum(query_vector.get(term, 0) * doc_vector.get(term, 0) 
                         for term in set(query_vector.keys()) | set(doc_vector.keys()))
        
        query_magnitude = math.sqrt(sum(score ** 2 for score in query_vector.values()))
        doc_magnitude = math.sqrt(sum(score ** 2 for score in doc_vector.values()))
        
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0.0
        
        return dot_product / (query_magnitude * doc_magnitude)
    
    def _calculate_keyword_score(self, query_terms: List[str], content_keywords: List[str]) -> float:
        """Calculate keyword match score"""
        
        if not content_keywords:
            return 0.0
        
        content_keywords_lower = [k.lower() for k in content_keywords]
        matches = sum(1 for term in query_terms if term in content_keywords_lower)
        
        return matches / max(len(query_terms), 1)
    
    def _calculate_fuzzy_score(self, query: str, content_text: str) -> float:
        """Calculate simple fuzzy matching score"""
        
        query_lower = query.lower()
        content_lower = content_text.lower()
        
        # Simple substring matching
        if query_lower in content_lower:
            return 1.0
        
        # Word overlap scoring
        query_words = set(self._tokenize(query_lower))
        content_words = set(self._tokenize(content_lower))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        return overlap / len(query_words)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = cleaned.split()
        
        # Filter stop words and short tokens
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return [token for token in tokens if len(token) > 2 and token not in stop_words]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        return self._tokenize(text)
    
    def _get_sample_restaurants(self) -> List[Dict[str, Any]]:
        """Sample restaurant data"""
        return [
            {
                'name': 'Hamdi Restaurant',
                'cuisine': 'Turkish',
                'district': 'Eminönü',
                'description': 'Famous for lamb dishes with Golden Horn views',
                'tags': ['kebab', 'meat', 'traditional', 'view'],
                'atmosphere': 'traditional',
                'specialty': 'lamb kebab'
            },
            {
                'name': 'Karaköy Lokantası',
                'cuisine': 'Turkish',
                'district': 'Karaköy',
                'description': 'Contemporary Turkish cuisine in elegant setting',
                'tags': ['fine dining', 'modern', 'seafood'],
                'atmosphere': 'upscale',
                'specialty': 'modern turkish'
            },
            {
                'name': 'Çiya Sofrası',
                'cuisine': 'Turkish',
                'district': 'Kadıköy',
                'description': 'Authentic Anatolian dishes, locals favorite',
                'tags': ['authentic', 'local', 'traditional'],
                'atmosphere': 'casual',
                'specialty': 'anatolian cuisine'
            }
        ]
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the retrieval system"""
        return {
            'initialized': self.initialized,
            'total_content': len(self.content_items),
            'content_types': {
                content_type: sum(1 for c in self.content_items.values() if c.content_type == content_type)
                for content_type in set(c.content_type for c in self.content_items.values())
            },
            'vocabulary_size': len(self.vocabulary),
            'keyword_index_size': len(self.keyword_index)
        }

# Global instance
lightweight_retrieval_system = LightweightRetrievalSystem()

def search_content_lightweight(query: str, content_types: Optional[List[str]] = None, 
                              top_k: int = 5) -> List[SearchResult]:
    """Main function to search content using lightweight retrieval"""
    return lightweight_retrieval_system.search(query, content_types, top_k)

def get_retrieval_system_info() -> Dict[str, Any]:
    """Get information about the retrieval system"""
    return lightweight_retrieval_system.get_system_info()
