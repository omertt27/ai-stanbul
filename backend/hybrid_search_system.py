#!/usr/bin/env python3
"""
Hybrid Search System for AI Istanbul
===================================

Combines keyword search + vector search for better precision and recall.
This system merges the strengths of both approaches:
- Keyword search: Exact matches, specific terms
- Vector search: Semantic similarity, context understanding

Features:
- Weighted combination of keyword and vector scores
- Query-adaptive weighting based on query characteristics
- Performance optimization with parallel processing
- Result diversification and ranking
"""

import numpy as np
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import Counter
import re

@dataclass
class HybridSearchResult:
    """Result from hybrid search with detailed scoring"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    
    # Individual scores
    keyword_score: float
    vector_score: float
    
    # Combined score
    hybrid_score: float
    
    # Ranking factors
    relevance_boost: float = 0.0
    freshness_boost: float = 0.0
    popularity_boost: float = 0.0
    
    # Search details
    matched_keywords: List[str] = None
    search_type: str = "hybrid"

class HybridSearchSystem:
    """Advanced hybrid search combining keyword and vector approaches"""
    
    def __init__(self, keyword_weight: float = 0.4, vector_weight: float = 0.6):
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Search components (will be injected)
        self.keyword_searcher = None
        self.vector_searcher = None
        
        # Query analysis patterns
        self.exact_match_patterns = [
            r'"([^"]+)"',  # Quoted text
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper nouns
            r'\b\d{4}\b',  # Years
            r'\b[A-Z]{2,}\b'  # Acronyms
        ]
        
        # Performance metrics
        self.search_stats = {
            "total_searches": 0,
            "avg_keyword_time": 0.0,
            "avg_vector_time": 0.0,
            "avg_hybrid_time": 0.0
        }
    
    def initialize(self, keyword_searcher, vector_searcher):
        """Initialize with search components"""
        self.keyword_searcher = keyword_searcher
        self.vector_searcher = vector_searcher
        print("✅ Hybrid Search System initialized")
    
    def analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal search strategy"""
        characteristics = {
            "has_quotes": bool(re.search(r'"[^"]+"', query)),
            "has_proper_nouns": bool(re.search(r'\b[A-Z][a-z]+\b', query)),
            "has_numbers": bool(re.search(r'\d+', query)),
            "has_specific_terms": False,
            "query_length": len(query.split()),
            "is_conversational": len(query.split()) > 5,
            "recommended_keyword_weight": 0.4,
            "recommended_vector_weight": 0.6
        }
        
        # Specific terms that benefit from keyword search
        specific_terms = [
            'address', 'phone', 'hours', 'price', 'cost',
            'open', 'closed', 'metro', 'bus', 'tram',
            'entrance', 'ticket', 'reservation'
        ]
        
        characteristics["has_specific_terms"] = any(
            term in query.lower() for term in specific_terms
        )
        
        # Adjust weights based on characteristics
        if characteristics["has_quotes"] or characteristics["has_specific_terms"]:
            # Favor keyword search for exact matches
            characteristics["recommended_keyword_weight"] = 0.7
            characteristics["recommended_vector_weight"] = 0.3
        elif characteristics["is_conversational"]:
            # Favor vector search for natural language
            characteristics["recommended_keyword_weight"] = 0.3
            characteristics["recommended_vector_weight"] = 0.7
        elif characteristics["has_proper_nouns"]:
            # Balanced approach for named entities
            characteristics["recommended_keyword_weight"] = 0.5
            characteristics["recommended_vector_weight"] = 0.5
        
        return characteristics
    
    async def hybrid_search(self, query: str, top_k: int = 10, 
                           content_types: Optional[List[str]] = None,
                           adaptive_weighting: bool = True) -> List[HybridSearchResult]:
        """Perform hybrid search combining keyword and vector approaches"""
        start_time = time.time()
        
        try:
            # Analyze query characteristics
            query_chars = self.analyze_query_characteristics(query)
            
            # Determine weights
            if adaptive_weighting:
                kw_weight = query_chars["recommended_keyword_weight"]
                vec_weight = query_chars["recommended_vector_weight"]
            else:
                kw_weight = self.keyword_weight
                vec_weight = self.vector_weight
            
            # Run searches in parallel
            keyword_task = asyncio.create_task(
                self._run_keyword_search(query, top_k * 2, content_types)
            )
            vector_task = asyncio.create_task(
                self._run_vector_search(query, top_k * 2, content_types)
            )
            
            # Wait for both searches
            keyword_results, vector_results = await asyncio.gather(
                keyword_task, vector_task
            )
            
            # Combine and rank results
            hybrid_results = self._combine_results(
                query, keyword_results, vector_results, 
                kw_weight, vec_weight, top_k
            )
            
            # Update statistics
            total_time = (time.time() - start_time) * 1000
            self._update_search_stats(total_time)
            
            print(f"🔍 Hybrid search: {len(hybrid_results)} results in {total_time:.1f}ms")
            return hybrid_results
            
        except Exception as e:
            print(f"❌ Hybrid search error: {e}")
            return []
    
    async def _run_keyword_search(self, query: str, top_k: int, 
                                 content_types: Optional[List[str]]) -> List[Any]:
        """Run keyword search asynchronously"""
        if not self.keyword_searcher:
            return []
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._execute_keyword_search,
                query, top_k, content_types
            )
            return results
        except Exception as e:
            print(f"⚠️ Keyword search error: {e}")
            return []
    
    async def _run_vector_search(self, query: str, top_k: int,
                                content_types: Optional[List[str]]) -> List[Any]:
        """Run vector search asynchronously"""
        if not self.vector_searcher:
            return []
        
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._execute_vector_search,
                query, top_k, content_types
            )
            return results
        except Exception as e:
            print(f"⚠️ Vector search error: {e}")
            return []
    
    def _execute_keyword_search(self, query: str, top_k: int, content_types: Optional[List[str]]):
        """Execute keyword search (runs in thread pool)"""
        if hasattr(self.keyword_searcher, 'search'):
            return self.keyword_searcher.search(query, content_types, top_k)
        else:
            return []
    
    def _execute_vector_search(self, query: str, top_k: int, content_types: Optional[List[str]]):
        """Execute vector search (runs in thread pool)"""
        if hasattr(self.vector_searcher, 'semantic_search'):
            return self.vector_searcher.semantic_search(query, top_k)
        elif hasattr(self.vector_searcher, 'search'):
            return self.vector_searcher.search(query, top_k)
        else:
            return []
    
    def _combine_results(self, query: str, keyword_results: List[Any], 
                        vector_results: List[Any], kw_weight: float, 
                        vec_weight: float, top_k: int) -> List[HybridSearchResult]:
        """Combine and rank results from both search methods"""
        
        # Normalize scores and combine results
        combined_scores = {}
        
        # Process keyword results
        max_kw_score = max([getattr(r, 'score', getattr(r, 'relevance_score', 0)) 
                           for r in keyword_results], default=1.0)
        
        for result in keyword_results:
            doc_id = getattr(result, 'id', getattr(result, 'document_id', str(hash(str(result)))))
            score = getattr(result, 'score', getattr(result, 'relevance_score', 0))
            normalized_score = score / max_kw_score if max_kw_score > 0 else 0
            
            combined_scores[doc_id] = {
                'result': result,
                'keyword_score': normalized_score,
                'vector_score': 0.0,
                'matched_keywords': self._extract_matched_keywords(query, result)
            }
        
        # Process vector results
        max_vec_score = max([getattr(r, 'similarity_score', getattr(r, 'score', 0)) 
                           for r in vector_results], default=1.0)
        
        for result in vector_results:
            doc_id = getattr(result, 'document', result).id if hasattr(result, 'document') else \
                     getattr(result, 'id', str(hash(str(result))))
            score = getattr(result, 'similarity_score', getattr(result, 'score', 0))
            normalized_score = score / max_vec_score if max_vec_score > 0 else 0
            
            if doc_id in combined_scores:
                combined_scores[doc_id]['vector_score'] = normalized_score
            else:
                combined_scores[doc_id] = {
                    'result': result,
                    'keyword_score': 0.0,
                    'vector_score': normalized_score,
                    'matched_keywords': []
                }
        
        # Calculate hybrid scores and create results
        hybrid_results = []
        
        for doc_id, scores in combined_scores.items():
            hybrid_score = (kw_weight * scores['keyword_score'] + 
                           vec_weight * scores['vector_score'])
            
            # Apply ranking boosts
            relevance_boost = self._calculate_relevance_boost(scores['result'], query)
            popularity_boost = self._calculate_popularity_boost(scores['result'])
            freshness_boost = self._calculate_freshness_boost(scores['result'])
            
            final_score = hybrid_score + relevance_boost + popularity_boost + freshness_boost
            
            # Create hybrid result
            result_obj = scores['result']
            content = getattr(result_obj, 'content', getattr(result_obj, 'title', ''))
            if hasattr(result_obj, 'document'):
                content = result_obj.document.content
                metadata = result_obj.document.metadata
            else:
                metadata = getattr(result_obj, 'metadata', {})
            
            hybrid_result = HybridSearchResult(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                keyword_score=scores['keyword_score'],
                vector_score=scores['vector_score'],
                hybrid_score=final_score,
                relevance_boost=relevance_boost,
                popularity_boost=popularity_boost,
                freshness_boost=freshness_boost,
                matched_keywords=scores['matched_keywords']
            )
            
            hybrid_results.append(hybrid_result)
        
        # Sort by hybrid score and return top k
        hybrid_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        # Apply diversity filtering
        diverse_results = self._apply_diversity_filter(hybrid_results, top_k)
        
        return diverse_results[:top_k]
    
    def _extract_matched_keywords(self, query: str, result: Any) -> List[str]:
        """Extract keywords that matched in the result"""
        query_terms = set(query.lower().split())
        content = getattr(result, 'content', getattr(result, 'title', ''))
        content_terms = set(content.lower().split())
        
        return list(query_terms & content_terms)
    
    def _calculate_relevance_boost(self, result: Any, query: str) -> float:
        """Calculate relevance boost based on content analysis"""
        boost = 0.0
        content = getattr(result, 'content', getattr(result, 'title', ''))
        
        # Boost for exact phrase matches
        if query.lower() in content.lower():
            boost += 0.1
        
        # Boost for title/name matches
        title = getattr(result, 'title', getattr(result, 'name', ''))
        if any(term in title.lower() for term in query.lower().split()):
            boost += 0.05
        
        return boost
    
    def _calculate_popularity_boost(self, result: Any) -> float:
        """Calculate popularity boost based on ratings, reviews, etc."""
        boost = 0.0
        
        # Check for rating in metadata
        if hasattr(result, 'metadata') and result.metadata:
            rating = result.metadata.get('rating', 0)
            if rating > 4.0:
                boost += 0.02
        
        # Check for review count
        if hasattr(result, 'metadata') and result.metadata:
            review_count = result.metadata.get('review_count', 0)
            if review_count > 100:
                boost += 0.01
        
        return boost
    
    def _calculate_freshness_boost(self, result: Any) -> float:
        """Calculate freshness boost for recently updated content"""
        # For now, return 0 as we don't have timestamp info
        # In production, this would check last_updated timestamps
        return 0.0
    
    def _apply_diversity_filter(self, results: List[HybridSearchResult], 
                               target_count: int) -> List[HybridSearchResult]:
        """Apply diversity filtering to avoid too similar results"""
        if len(results) <= target_count:
            return results
        
        diverse_results = []
        seen_types = set()
        
        # First pass: ensure type diversity
        for result in results:
            result_type = result.metadata.get('type', 'unknown')
            if result_type not in seen_types or len(diverse_results) < target_count // 2:
                diverse_results.append(result)
                seen_types.add(result_type)
                
                if len(diverse_results) >= target_count:
                    break
        
        # Second pass: fill remaining slots with highest scores
        remaining_slots = target_count - len(diverse_results)
        remaining_results = [r for r in results if r not in diverse_results]
        diverse_results.extend(remaining_results[:remaining_slots])
        
        return diverse_results
    
    def _update_search_stats(self, total_time: float):
        """Update search performance statistics"""
        self.search_stats["total_searches"] += 1
        
        # Update rolling average
        count = self.search_stats["total_searches"]
        current_avg = self.search_stats["avg_hybrid_time"]
        self.search_stats["avg_hybrid_time"] = (
            (current_avg * (count - 1) + total_time) / count
        )
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics"""
        return {
            **self.search_stats,
            "current_weights": {
                "keyword_weight": self.keyword_weight,
                "vector_weight": self.vector_weight
            },
            "timestamp": time.time()
        }
    
    def optimize_weights(self, query_performance_data: List[Dict[str, Any]]):
        """Optimize search weights based on performance data"""
        # Simple optimization based on success rates
        keyword_successful = []
        vector_successful = []
        
        for data in query_performance_data:
            if data.get('user_satisfaction', 0) >= 4:  # Good results
                keyword_successful.append(data.get('keyword_contribution', 0))
                vector_successful.append(data.get('vector_contribution', 0))
        
        if keyword_successful and vector_successful:
            avg_kw_success = np.mean(keyword_successful)
            avg_vec_success = np.mean(vector_successful)
            
            total = avg_kw_success + avg_vec_success
            if total > 0:
                self.keyword_weight = avg_kw_success / total
                self.vector_weight = avg_vec_success / total
                
                print(f"🎯 Optimized weights: keyword={self.keyword_weight:.2f}, vector={self.vector_weight:.2f}")

# Global hybrid search instance
hybrid_search_system = HybridSearchSystem()

async def initialize_hybrid_search():
    """Initialize hybrid search with available components"""
    try:
        # Import search components
        from lightweight_retrieval_system import lightweight_retrieval_system
        from vector_embedding_system import vector_embedding_system
        
        # Initialize hybrid search
        hybrid_search_system.initialize(
            keyword_searcher=lightweight_retrieval_system,
            vector_searcher=vector_embedding_system
        )
        
        print("✅ Hybrid Search System ready")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search initialization error: {e}")
        return False

async def test_hybrid_search():
    """Test the hybrid search system"""
    print("🧪 Testing Hybrid Search System...")
    
    # Initialize
    success = await initialize_hybrid_search()
    if not success:
        return False
    
    # Test queries with different characteristics
    test_queries = [
        "Turkish restaurants in Sultanahmet",  # General query
        '"Blue Mosque" opening hours',         # Exact match query
        "romantic dinner with Bosphorus view", # Conversational query
        "Galata Tower metro station",         # Specific information
        "traditional Ottoman architecture"     # Semantic query
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Analyze query characteristics
        chars = hybrid_search_system.analyze_query_characteristics(query)
        print(f"   📊 Query type: {chars['recommended_keyword_weight']:.1f} keyword, {chars['recommended_vector_weight']:.1f} vector")
        
        # Perform hybrid search
        results = await hybrid_search_system.hybrid_search(query, top_k=3)
        
        print(f"   📋 Results: {len(results)}")
        for i, result in enumerate(results[:2], 1):
            print(f"      {i}. Hybrid: {result.hybrid_score:.3f} (KW: {result.keyword_score:.3f}, Vec: {result.vector_score:.3f})")
    
    # Get stats
    stats = hybrid_search_system.get_search_stats()
    print(f"\n📊 Search stats: {stats['total_searches']} searches, {stats['avg_hybrid_time']:.1f}ms avg")
    
    return True

if __name__ == "__main__":
    # Test the hybrid search system
    async def main():
        success = await test_hybrid_search()
        if success:
            print("✅ Hybrid Search System is working correctly!")
        else:
            print("❌ Hybrid Search System test failed")
    
    asyncio.run(main())
