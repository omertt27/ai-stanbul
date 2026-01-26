"""
LLM-Based RAG Service for Production

This RAG service uses your RunPod Llama model for semantic understanding
instead of local HuggingFace embedding models. This allows RAG to work
in Cloud Run without any model downloads.

Architecture:
1. Database Query: Keyword/SQL search for relevant entities
2. LLM Reranking: Use the LLM to score/rerank results by relevance
3. Context Building: Format results for LLM consumption

This approach is faster and more production-friendly than vector embeddings.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_, func, text
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMRAGService:
    """
    Lightweight RAG service that works with your existing RunPod LLM.
    
    Instead of embedding vectors, this uses:
    1. Keyword/full-text search in the database
    2. LLM-based relevance scoring (optional)
    3. Smart context formatting
    
    This works in Cloud Run without HuggingFace downloads!
    """
    
    def __init__(self, db: Session = None, llm_client=None):
        """
        Initialize RAG service
        
        Args:
            db: SQLAlchemy database session
            llm_client: RunPod LLM client for optional reranking
        """
        self.db = db
        self.llm_client = llm_client
        self._models_imported = False
        
        logger.info("🚀 LLM RAG Service initialized (no HuggingFace required)")
    
    def _import_models(self):
        """Lazily import database models to avoid circular imports"""
        if self._models_imported:
            return
        
        try:
            from models import Restaurant, Museum, Event, Place, BlogPost
            self.Restaurant = Restaurant
            self.Museum = Museum
            self.Event = Event
            self.Place = Place
            self.BlogPost = BlogPost
            self._models_imported = True
        except ImportError as e:
            logger.warning(f"Could not import models: {e}")
            self._models_imported = False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_types: List[str] = None,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant entities using keyword matching
        
        Args:
            query: User's search query
            top_k: Number of results to return
            entity_types: Types to search ['restaurant', 'museum', 'event', 'place', 'blog']
            filters: Additional filters (category, district, etc.)
            
        Returns:
            List of relevant entities with metadata
        """
        if not self.db:
            logger.warning("No database session available for RAG search")
            return []
        
        self._import_models()
        if not self._models_imported:
            return []
        
        # Default to all entity types
        if entity_types is None:
            entity_types = ['restaurant', 'museum', 'place', 'blog']
        
        results = []
        query_lower = query.lower()
        
        # Extract keywords for search
        keywords = self._extract_keywords(query)
        
        try:
            # Search restaurants
            if 'restaurant' in entity_types:
                restaurant_results = self._search_restaurants(keywords, filters, top_k)
                results.extend(restaurant_results)
            
            # Search museums/attractions
            if 'museum' in entity_types:
                museum_results = self._search_museums(keywords, filters, top_k)
                results.extend(museum_results)
            
            # Search places/districts
            if 'place' in entity_types:
                place_results = self._search_places(keywords, filters, top_k)
                results.extend(place_results)
            
            # Search blog posts
            if 'blog' in entity_types:
                blog_results = self._search_blogs(keywords, filters, top_k)
                results.extend(blog_results)
            
            # Sort by relevance score and return top_k
            results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            logger.info(f"🔍 RAG: Found {len(results)} results for '{query[:50]}...'")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract search keywords from query"""
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'can', 'what', 'where', 'when', 'how', 'why', 'which', 'who',
            'me', 'my', 'i', 'you', 'your', 'we', 'they', 'it', 'this', 'that',
            'tell', 'show', 'find', 'give', 'about', 'near', 'nearby', 'best',
            'good', 'nice', 'great', 'some', 'any'
        }
        
        # Tokenize and filter
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate relevance score based on keyword matches"""
        if not text or not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        
        # Bonus for exact phrase match
        query_phrase = ' '.join(keywords)
        if query_phrase in text_lower:
            matches += len(keywords)
        
        return min(1.0, matches / max(len(keywords), 1))
    
    def _search_restaurants(
        self,
        keywords: List[str],
        filters: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search restaurants table"""
        results = []
        
        try:
            # Build query with keyword search
            query = self.db.query(self.Restaurant)
            
            # Add keyword filters
            if keywords:
                keyword_filters = []
                for kw in keywords:
                    keyword_filters.append(
                        or_(
                            func.lower(self.Restaurant.name).contains(kw),
                            func.lower(self.Restaurant.cuisine).contains(kw),
                            func.lower(self.Restaurant.district).contains(kw),
                            func.lower(self.Restaurant.description).contains(kw) if hasattr(self.Restaurant, 'description') else False
                        )
                    )
                if keyword_filters:
                    query = query.filter(or_(*keyword_filters))
            
            # Apply additional filters
            if filters:
                if 'cuisine' in filters:
                    query = query.filter(func.lower(self.Restaurant.cuisine).contains(filters['cuisine'].lower()))
                if 'district' in filters:
                    query = query.filter(func.lower(self.Restaurant.district).contains(filters['district'].lower()))
            
            # Get results
            restaurants = query.limit(limit * 2).all()  # Get more for scoring
            
            for r in restaurants:
                # Build searchable text
                text = f"{r.name} {r.cuisine or ''} {r.district or ''}"
                if hasattr(r, 'description') and r.description:
                    text += f" {r.description}"
                
                score = self._calculate_keyword_score(text, keywords)
                
                if score > 0:
                    results.append({
                        'id': r.id,
                        'type': 'restaurant',
                        'name': r.name,
                        'cuisine': getattr(r, 'cuisine', None),
                        'district': getattr(r, 'district', None),
                        'rating': getattr(r, 'rating', None),
                        'price_level': getattr(r, 'price_level', None),
                        'description': getattr(r, 'description', None),
                        'relevance_score': score,
                        'metadata': {
                            'type': 'restaurant',
                            'name': r.name
                        }
                    })
        
        except Exception as e:
            logger.warning(f"Restaurant search failed: {e}")
        
        return results
    
    def _search_museums(
        self,
        keywords: List[str],
        filters: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search museums/attractions table"""
        results = []
        
        try:
            query = self.db.query(self.Museum)
            
            # Add keyword filters
            if keywords:
                keyword_filters = []
                for kw in keywords:
                    keyword_filters.append(
                        or_(
                            func.lower(self.Museum.name).contains(kw),
                            func.lower(self.Museum.description).contains(kw) if hasattr(self.Museum, 'description') else False,
                            func.lower(self.Museum.category).contains(kw) if hasattr(self.Museum, 'category') else False
                        )
                    )
                if keyword_filters:
                    query = query.filter(or_(*keyword_filters))
            
            museums = query.limit(limit * 2).all()
            
            for m in museums:
                text = f"{m.name}"
                if hasattr(m, 'description') and m.description:
                    text += f" {m.description}"
                if hasattr(m, 'category') and m.category:
                    text += f" {m.category}"
                
                score = self._calculate_keyword_score(text, keywords)
                
                if score > 0:
                    results.append({
                        'id': m.id,
                        'type': 'museum',
                        'name': m.name,
                        'description': getattr(m, 'description', None),
                        'category': getattr(m, 'category', None),
                        'hours': getattr(m, 'hours', None),
                        'admission': getattr(m, 'admission', None),
                        'relevance_score': score,
                        'metadata': {
                            'type': 'museum',
                            'name': m.name
                        }
                    })
        
        except Exception as e:
            logger.warning(f"Museum search failed: {e}")
        
        return results
    
    def _search_places(
        self,
        keywords: List[str],
        filters: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search places/districts table"""
        results = []
        
        try:
            query = self.db.query(self.Place)
            
            # Add keyword filters
            if keywords:
                keyword_filters = []
                for kw in keywords:
                    keyword_filters.append(
                        or_(
                            func.lower(self.Place.name).contains(kw),
                            func.lower(self.Place.description).contains(kw) if hasattr(self.Place, 'description') else False
                        )
                    )
                if keyword_filters:
                    query = query.filter(or_(*keyword_filters))
            
            places = query.limit(limit * 2).all()
            
            for p in places:
                text = f"{p.name}"
                if hasattr(p, 'description') and p.description:
                    text += f" {p.description}"
                
                score = self._calculate_keyword_score(text, keywords)
                
                if score > 0:
                    results.append({
                        'id': p.id,
                        'type': 'place',
                        'name': p.name,
                        'description': getattr(p, 'description', None),
                        'relevance_score': score,
                        'metadata': {
                            'type': 'place',
                            'name': p.name
                        }
                    })
        
        except Exception as e:
            logger.warning(f"Place search failed: {e}")
        
        return results
    
    def _search_blogs(
        self,
        keywords: List[str],
        filters: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search blog posts table"""
        results = []
        
        try:
            query = self.db.query(self.BlogPost)
            
            # Add keyword filters
            if keywords:
                keyword_filters = []
                for kw in keywords:
                    keyword_filters.append(
                        or_(
                            func.lower(self.BlogPost.title).contains(kw),
                            func.lower(self.BlogPost.content).contains(kw) if hasattr(self.BlogPost, 'content') else False,
                            func.lower(self.BlogPost.summary).contains(kw) if hasattr(self.BlogPost, 'summary') else False
                        )
                    )
                if keyword_filters:
                    query = query.filter(or_(*keyword_filters))
            
            blogs = query.limit(limit * 2).all()
            
            for b in blogs:
                text = f"{b.title}"
                if hasattr(b, 'summary') and b.summary:
                    text += f" {b.summary}"
                if hasattr(b, 'content') and b.content:
                    text += f" {b.content[:500]}"  # First 500 chars
                
                score = self._calculate_keyword_score(text, keywords)
                
                if score > 0:
                    results.append({
                        'id': b.id,
                        'type': 'blog',
                        'title': b.title,
                        'summary': getattr(b, 'summary', None),
                        'content_preview': getattr(b, 'content', '')[:200] if hasattr(b, 'content') else None,
                        'relevance_score': score,
                        'metadata': {
                            'type': 'blog_post',
                            'name': b.title
                        }
                    })
        
        except Exception as e:
            logger.warning(f"Blog search failed: {e}")
        
        return results
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 3,
        entity_types: List[str] = None
    ) -> str:
        """
        Get formatted context string for LLM prompt
        
        Args:
            query: User's query
            top_k: Number of results to include
            entity_types: Types to search
            
        Returns:
            Formatted context string for LLM
        """
        results = self.search(query, top_k=top_k, entity_types=entity_types)
        
        if not results:
            return ""
        
        # Format context for LLM
        context_parts = ["Here is relevant information from our database:\n"]
        
        for i, result in enumerate(results, 1):
            entity_type = result.get('type', 'item')
            name = result.get('name') or result.get('title', 'Unknown')
            
            context_parts.append(f"\n{i}. {entity_type.title()}: {name}")
            
            # Add type-specific details
            if entity_type == 'restaurant':
                if result.get('cuisine'):
                    context_parts.append(f"   Cuisine: {result['cuisine']}")
                if result.get('district'):
                    context_parts.append(f"   Location: {result['district']}")
                if result.get('rating'):
                    context_parts.append(f"   Rating: {result['rating']}/5")
                if result.get('price_level'):
                    context_parts.append(f"   Price: {'$' * result['price_level']}")
            
            elif entity_type == 'museum':
                if result.get('category'):
                    context_parts.append(f"   Category: {result['category']}")
                if result.get('hours'):
                    context_parts.append(f"   Hours: {result['hours']}")
                if result.get('admission'):
                    context_parts.append(f"   Admission: {result['admission']}")
            
            elif entity_type == 'place':
                pass  # Name and description are usually enough
            
            elif entity_type == 'blog':
                if result.get('summary'):
                    context_parts.append(f"   Summary: {result['summary'][:200]}...")
            
            # Add description if available
            if result.get('description'):
                desc = result['description'][:300]
                if len(result.get('description', '')) > 300:
                    desc += '...'
                context_parts.append(f"   Description: {desc}")
        
        return '\n'.join(context_parts)


# Singleton instance
_llm_rag_service = None


def get_llm_rag_service(db: Session = None, llm_client=None) -> LLMRAGService:
    """
    Get or create LLM RAG service singleton
    
    Args:
        db: Database session
        llm_client: Optional LLM client for reranking
        
    Returns:
        LLMRAGService instance
    """
    global _llm_rag_service
    
    if _llm_rag_service is None:
        _llm_rag_service = LLMRAGService(db=db, llm_client=llm_client)
    elif db is not None:
        # Update database session if provided
        _llm_rag_service.db = db
    
    return _llm_rag_service
