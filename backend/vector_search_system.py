#!/usr/bin/env python3
"""
Vector Search System for Istanbul AI - Retrieval-First Design
===========================================================

This system implements semantic search using vector embeddings for:
1. Restaurant content vectorization
2. Museum information encoding  
3. Event and transport data indexing
4. FAISS-based similarity search
5. Hybrid keyword + semantic matching
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
import re
from collections import defaultdict

# Try to import FAISS for vector search
try:
    import faiss
    FAISS_AVAILABLE = True
    print("✅ FAISS available for vector search")
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available - install with: pip install faiss-cpu")

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    print("✅ SentenceTransformers available for embeddings")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ SentenceTransformers not available - install with: pip install sentence-transformers")

@dataclass
class SearchResult:
    """Result from vector search"""
    content_id: str
    content_type: str  # 'restaurant', 'museum', 'event', 'transport'
    title: str
    description: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class VectorizedContent:
    """Content prepared for vector search"""
    content_id: str
    content_type: str
    title: str
    description: str
    searchable_text: str
    keywords: List[str]
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class VectorSearchSystem:
    """Main vector search system implementing retrieval-first design"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = None
        self.faiss_index = None
        self.content_map = {}  # content_id -> VectorizedContent
        self.keyword_index = defaultdict(set)  # keyword -> set of content_ids
        self.initialized = False
        
        # Initialize if dependencies available
        if EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE:
            self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the vector search system"""
        try:
            print("🚀 Initializing Vector Search System...")
            
            # Load embedding model (lightweight, fast model)
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded embedding model: {self.model_name}")
            
            # Load and vectorize content
            self._load_and_vectorize_content()
            
            # Build FAISS index
            self._build_faiss_index()
            
            # Build keyword index
            self._build_keyword_index()
            
            self.initialized = True
            print("✅ Vector Search System initialized successfully!")
            
        except Exception as e:
            print(f"❌ Error initializing vector search: {e}")
            self.initialized = False
    
    def _load_and_vectorize_content(self):
        """Load all content and create vectorized representations"""
        
        # Load restaurant data
        self._vectorize_restaurants()
        
        # Load museum data
        self._vectorize_museums()
        
        # Load transport data
        self._vectorize_transport()
        
        # Load event/cultural data
        self._vectorize_events()
        
        print(f"📊 Vectorized {len(self.content_map)} content items")
    
    def _vectorize_restaurants(self):
        """Vectorize restaurant data for semantic search"""
        
        # Try to load restaurant database
        try:
            data_path = Path(__file__).parent / "data" / "restaurants_database.json"
            if data_path.exists():
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    restaurants = data.get('restaurants', [])
            else:
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
                restaurant.get('atmosphere', ''),
                restaurant.get('specialty', ''),
                restaurant.get('location', '')
            ]
            
            searchable_text = ' '.join(filter(None, searchable_parts))
            
            # Extract keywords
            keywords = self._extract_keywords(searchable_text)
            keywords.extend(restaurant.get('tags', []))
            keywords.append(restaurant.get('cuisine', '').lower())
            keywords.append(restaurant.get('district', '').lower())
            
            content = VectorizedContent(
                content_id=f"restaurant_{i}",
                content_type="restaurant",
                title=restaurant.get('name', f'Restaurant {i}'),
                description=restaurant.get('description', ''),
                searchable_text=searchable_text,
                keywords=[k.lower() for k in keywords if k],
                metadata=restaurant
            )
            
            self.content_map[content.content_id] = content
    
    def _vectorize_museums(self):
        """Vectorize museum data for semantic search"""
        
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
                museum.visitor_tips,
                museum.cultural_context
            ]
            
            searchable_text = ' '.join(filter(None, searchable_parts))
            
            # Extract keywords
            keywords = self._extract_keywords(searchable_text)
            keywords.extend(museum.key_features)
            keywords.extend([museum.historical_period, museum.architectural_style])
            
            # Add location keywords
            location_parts = museum.location.split(',')
            keywords.extend([part.strip().lower() for part in location_parts])
            
            content = VectorizedContent(
                content_id=f"museum_{key}",
                content_type="museum",
                title=museum.name,
                description=f"{museum.historical_significance} Located in {museum.location}.",
                searchable_text=searchable_text,
                keywords=[k.lower() for k in keywords if k],
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
            
            self.content_map[content.content_id] = content
    
    def _vectorize_transport(self):
        """Vectorize transportation data"""
        
        transport_data = [
            {
                'name': 'Istanbul Metro System',
                'type': 'metro',
                'description': 'Comprehensive metro network connecting major districts',
                'routes': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'],
                'keywords': ['metro', 'subway', 'underground', 'public transport', 'istanbulkart']
            },
            {
                'name': 'Bosphorus Ferry System',
                'type': 'ferry',
                'description': 'Ferry services connecting European and Asian sides',
                'routes': ['Eminönü-Üsküdar', 'Karaköy-Kadıköy', 'Beşiktaş-Üsküdar'],
                'keywords': ['ferry', 'vapur', 'bosphorus', 'boat', 'sea transport', 'cross continent']
            },
            {
                'name': 'Istanbul Tram Network',
                'type': 'tram',
                'description': 'Tram lines serving tourist areas and city center',
                'routes': ['T1', 'T3', 'T4', 'T5'],
                'keywords': ['tram', 'tramvay', 'rail', 'sultanahmet', 'historic']
            },
            {
                'name': 'Metrobus BRT',
                'type': 'bus',
                'description': 'High-speed bus rapid transit system',
                'routes': ['Metrobus Main Line'],
                'keywords': ['metrobus', 'bus', 'rapid transit', 'brt', 'fast']
            }
        ]
        
        for i, transport in enumerate(transport_data):
            searchable_text = f"{transport['name']} {transport['description']} {' '.join(transport['routes'])} {' '.join(transport['keywords'])}"
            
            content = VectorizedContent(
                content_id=f"transport_{i}",
                content_type="transport",
                title=transport['name'],
                description=transport['description'],
                searchable_text=searchable_text,
                keywords=transport['keywords'],
                metadata=transport
            )
            
            self.content_map[content.content_id] = content
    
    def _vectorize_events(self):
        """Vectorize cultural events and activities"""
        
        events_data = [
            {
                'name': 'Turkish Bath Experience',
                'type': 'cultural_activity',
                'description': 'Traditional hammam experience in historic bathhouses',
                'keywords': ['hammam', 'turkish bath', 'spa', 'relaxation', 'traditional', 'cultural']
            },
            {
                'name': 'Bosphorus Sunset Cruise',
                'type': 'tour',
                'description': 'Evening boat tour with spectacular sunset views',
                'keywords': ['cruise', 'sunset', 'boat tour', 'bosphorus', 'romantic', 'sightseeing']
            },
            {
                'name': 'Turkish Cooking Class',
                'type': 'cultural_activity',
                'description': 'Learn to cook traditional Turkish dishes',
                'keywords': ['cooking', 'food', 'class', 'turkish cuisine', 'hands-on', 'cultural']
            },
            {
                'name': 'Istanbul Food Walking Tour',
                'type': 'tour',
                'description': 'Guided tour exploring local food scenes',
                'keywords': ['food tour', 'walking', 'street food', 'local', 'taste', 'guide']
            }
        ]
        
        for i, event in enumerate(events_data):
            searchable_text = f"{event['name']} {event['description']} {' '.join(event['keywords'])}"
            
            content = VectorizedContent(
                content_id=f"event_{i}",
                content_type="event",
                title=event['name'],
                description=event['description'],
                searchable_text=searchable_text,
                keywords=event['keywords'],
                metadata=event
            )
            
            self.content_map[content.content_id] = content
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        
        if not FAISS_AVAILABLE or not self.embedding_model:
            print("⚠️ Cannot build FAISS index - dependencies missing")
            return
        
        # Get all searchable texts
        texts = [content.searchable_text for content in self.content_map.values()]
        
        if not texts:
            print("⚠️ No content to vectorize")
            return
        
        print(f"🔄 Generating embeddings for {len(texts)} items...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Store embeddings in content objects
        for i, (content_id, content) in enumerate(self.content_map.items()):
            content.embedding = embeddings[i]
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.faiss_index.add(embeddings.astype('float32'))
        
        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors (dimension: {dimension})")
    
    def _build_keyword_index(self):
        """Build keyword index for hybrid search"""
        
        for content_id, content in self.content_map.items():
            for keyword in content.keywords:
                self.keyword_index[keyword.lower()].add(content_id)
        
        print(f"✅ Keyword index built with {len(self.keyword_index)} unique keywords")
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def search(self, query: str, content_types: Optional[List[str]] = None, 
               top_k: int = 10, hybrid_weight: float = 0.7) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity and keyword matching
        
        Args:
            query: Search query
            content_types: Filter by content types ('restaurant', 'museum', etc.)
            top_k: Number of results to return
            hybrid_weight: Weight for semantic vs keyword (0.0 = all keyword, 1.0 = all semantic)
        """
        
        if not self.initialized:
            print("⚠️ Vector search not initialized, falling back to keyword search only")
            return self._keyword_search_only(query, content_types, top_k)
        
        # Get semantic search results
        semantic_results = self._semantic_search(query, top_k * 2)
        
        # Get keyword search results  
        keyword_results = self._keyword_search(query, top_k * 2)
        
        # Combine and rank results
        combined_results = self._combine_search_results(
            semantic_results, keyword_results, hybrid_weight
        )
        
        # Filter by content type if specified
        if content_types:
            combined_results = [r for r in combined_results if r.content_type in content_types]
        
        return combined_results[:top_k]
    
    def _semantic_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform semantic search using FAISS"""
        
        if not self.faiss_index or not self.embedding_model:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        content_list = list(self.content_map.values())
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(content_list):
                content = content_list[idx]
                results.append(SearchResult(
                    content_id=content.content_id,
                    content_type=content.content_type,
                    title=content.title,
                    description=content.description,
                    score=float(score),
                    metadata=content.metadata
                ))
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform keyword-based search"""
        
        query_keywords = self._extract_keywords(query)
        content_scores = defaultdict(float)
        
        # Score content based on keyword matches
        for keyword in query_keywords:
            if keyword in self.keyword_index:
                for content_id in self.keyword_index[keyword]:
                    content_scores[content_id] += 1.0
        
        # Get top scoring content
        sorted_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for content_id, score in sorted_content[:top_k]:
            if content_id in self.content_map:
                content = self.content_map[content_id]
                results.append(SearchResult(
                    content_id=content.content_id,
                    content_type=content.content_type,
                    title=content.title,
                    description=content.description,
                    score=score,
                    metadata=content.metadata
                ))
        
        return results
    
    def _keyword_search_only(self, query: str, content_types: Optional[List[str]], 
                           top_k: int) -> List[SearchResult]:
        """Fallback keyword-only search when vector search unavailable"""
        
        results = self._keyword_search(query, top_k * 2)
        
        if content_types:
            results = [r for r in results if r.content_type in content_types]
        
        return results[:top_k]
    
    def _combine_search_results(self, semantic_results: List[SearchResult], 
                              keyword_results: List[SearchResult], 
                              hybrid_weight: float) -> List[SearchResult]:
        """Combine semantic and keyword search results"""
        
        # Create combined scoring
        result_scores = {}
        
        # Add semantic scores
        for result in semantic_results:
            result_scores[result.content_id] = {
                'semantic': result.score,
                'keyword': 0.0,
                'result': result
            }
        
        # Add keyword scores
        for result in keyword_results:
            if result.content_id not in result_scores:
                result_scores[result.content_id] = {
                    'semantic': 0.0,
                    'keyword': result.score,
                    'result': result
                }
            else:
                result_scores[result.content_id]['keyword'] = result.score
        
        # Calculate hybrid scores
        for content_id, scores in result_scores.items():
            # Normalize scores (simple min-max normalization)
            semantic_norm = scores['semantic'] / max(1.0, max(r.score for r in semantic_results)) if semantic_results else 0
            keyword_norm = scores['keyword'] / max(1.0, max(r.score for r in keyword_results)) if keyword_results else 0
            
            # Combine with hybrid weight
            hybrid_score = (hybrid_weight * semantic_norm) + ((1 - hybrid_weight) * keyword_norm)
            scores['hybrid'] = hybrid_score
            scores['result'].score = hybrid_score
        
        # Sort by hybrid score
        sorted_results = sorted(result_scores.values(), key=lambda x: x['hybrid'], reverse=True)
        
        return [item['result'] for item in sorted_results]
    
    def _get_sample_restaurants(self) -> List[Dict[str, Any]]:
        """Sample restaurant data for testing"""
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
        """Get information about the vector search system"""
        return {
            'initialized': self.initialized,
            'faiss_available': FAISS_AVAILABLE,
            'embeddings_available': EMBEDDINGS_AVAILABLE,
            'model_name': self.model_name,
            'total_content': len(self.content_map),
            'content_types': {
                content_type: sum(1 for c in self.content_map.values() if c.content_type == content_type)
                for content_type in set(c.content_type for c in self.content_map.values())
            },
            'total_keywords': len(self.keyword_index),
            'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0
        }

# Global instance
vector_search_system = VectorSearchSystem()

def search_content(query: str, content_types: Optional[List[str]] = None, 
                  top_k: int = 5) -> List[SearchResult]:
    """Main function to search content using vector/hybrid search"""
    return vector_search_system.search(query, content_types, top_k)

def get_search_system_info() -> Dict[str, Any]:
    """Get information about the search system"""
    return vector_search_system.get_system_info()
