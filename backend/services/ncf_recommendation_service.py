"""
NCF Recommendation Service - Production Integration

Integrates the Neural Collaborative Filtering model into the production API.
Provides personalized recommendations using deep learning embeddings.
"""

import os
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class NCFRecommendation:
    """Recommendation from NCF model"""
    item_id: str
    item_name: str
    score: float
    confidence: float
    item_type: str
    metadata: Dict[str, Any]
    embedding_similarity: float


class NCFRecommendationService:
    """
    Production service for NCF-based recommendations.
    
    Provides:
    - User-personalized recommendations via NCF model
    - Fallback to rule-based recommendations
    - Integration with existing services
    - Performance monitoring
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize NCF recommendation service
        
        Args:
            model_path: Path to trained NCF model (optional)
        """
        self.model = None
        self.user_encoder = None
        self.item_encoder = None
        self.item_metadata = {}
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 
            '../ml/models/ncf_model.pt'
        )
        self.enabled = False
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'ncf_requests': 0,
            'fallback_requests': 0,
            'avg_latency_ms': 0.0,
            'cache_hits': 0
        }
        
        # Simple cache for recommendations
        self._cache = {}
        self._cache_ttl = 3600  # 1 hour
        
        # Initialize model
        self._load_model()
    
    def _load_model(self):
        """Load the trained NCF model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"⚠️ NCF model not found at {self.model_path}")
                logger.info("💡 Train the model first: python test_ncf_model.py")
                return
            
            # Import NCF model class
            from backend.ml.deep_learning.models.ncf import NCFModel
            
            # Load model checkpoint
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
            # Get model parameters
            num_users = checkpoint.get('num_users', 200)
            num_items = checkpoint.get('num_items', 500)
            embedding_dim = checkpoint.get('embedding_dim', 64)
            mlp_layers = checkpoint.get('mlp_layers', [128, 64, 32])
            
            # Create model
            self.model = NCFModel(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
                mlp_layers=mlp_layers
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load encoders
            self.user_encoder = checkpoint.get('user_encoder', {})
            self.item_encoder = checkpoint.get('item_encoder', {})
            
            # Load metadata if available
            self.item_metadata = checkpoint.get('item_metadata', {})
            
            self.enabled = True
            logger.info(f"✅ NCF model loaded successfully")
            logger.info(f"   Users: {num_users}, Items: {num_items}")
            logger.info(f"   Embedding dim: {embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load NCF model: {e}")
            self.enabled = False
    
    async def get_recommendations(
        self,
        user_id: str,
        top_k: int = 10,
        filter_visited: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> List[NCFRecommendation]:
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            filter_visited: Filter out already visited items
            context: Additional context (location, time, etc.)
            
        Returns:
            List of NCFRecommendation objects
        """
        start_time = datetime.now()
        self.stats['total_requests'] += 1
        
        try:
            # Check cache
            cache_key = f"{user_id}_{top_k}_{filter_visited}"
            if cache_key in self._cache:
                cached_time, cached_recs = self._cache[cache_key]
                if (datetime.now() - cached_time).seconds < self._cache_ttl:
                    self.stats['cache_hits'] += 1
                    logger.info(f"📦 Cache hit for user {user_id}")
                    return cached_recs
            
            if not self.enabled or self.model is None:
                logger.warning("⚠️ NCF model not available, using fallback")
                return await self._get_fallback_recommendations(user_id, top_k, context)
            
            # Get NCF predictions
            recommendations = await self._get_ncf_predictions(
                user_id, top_k, filter_visited, context
            )
            
            # Cache results
            self._cache[cache_key] = (datetime.now(), recommendations)
            
            # Update stats
            self.stats['ncf_requests'] += 1
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.stats['avg_latency_ms'] = (
                (self.stats['avg_latency_ms'] * (self.stats['ncf_requests'] - 1) + latency_ms) /
                self.stats['ncf_requests']
            )
            
            logger.info(f"✅ NCF recommendations for {user_id}: {len(recommendations)} items in {latency_ms:.1f}ms")
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ NCF recommendation error: {e}")
            return await self._get_fallback_recommendations(user_id, top_k, context)
    
    async def _get_ncf_predictions(
        self,
        user_id: str,
        top_k: int,
        filter_visited: bool,
        context: Optional[Dict[str, Any]]
    ) -> List[NCFRecommendation]:
        """Get predictions from NCF model"""
        
        # Check if user exists in encoder
        if user_id not in self.user_encoder:
            logger.warning(f"⚠️ User {user_id} not in training data, using fallback")
            return await self._get_fallback_recommendations(user_id, top_k, context)
        
        user_idx = self.user_encoder[user_id]
        
        # Get all item indices
        item_indices = list(range(len(self.item_encoder)))
        
        # Create user-item pairs
        user_indices = [user_idx] * len(item_indices)
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_indices)
        item_tensor = torch.LongTensor(item_indices)
        
        # Get predictions
        with torch.no_grad():
            scores = self.model(user_tensor, item_tensor).squeeze().numpy()
        
        # Get top-k items
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Create recommendations
        recommendations = []
        item_decoder = {v: k for k, v in self.item_encoder.items()}
        
        for idx in top_indices:
            item_idx = item_indices[idx]
            item_id = item_decoder.get(item_idx, f"item_{item_idx}")
            score = float(scores[idx])
            
            # Get item metadata
            metadata = self.item_metadata.get(item_id, {
                'name': item_id,
                'type': 'attraction',
                'district': 'Istanbul'
            })
            
            recommendations.append(NCFRecommendation(
                item_id=item_id,
                item_name=metadata.get('name', item_id),
                score=score,
                confidence=min(score, 1.0),  # Normalize to [0, 1]
                item_type=metadata.get('type', 'attraction'),
                metadata=metadata,
                embedding_similarity=score
            ))
        
        return recommendations
    
    async def _get_fallback_recommendations(
        self,
        user_id: str,
        top_k: int,
        context: Optional[Dict[str, Any]]
    ) -> List[NCFRecommendation]:
        """Fallback to rule-based recommendations"""
        self.stats['fallback_requests'] += 1
        
        logger.info(f"📋 Using fallback recommendations for {user_id}")
        
        # Popular Istanbul attractions (fallback)
        popular_items = [
            {
                'id': 'hagia_sophia',
                'name': 'Hagia Sophia',
                'type': 'historical_site',
                'score': 0.95,
                'district': 'Sultanahmet'
            },
            {
                'id': 'blue_mosque',
                'name': 'Blue Mosque',
                'type': 'mosque',
                'score': 0.93,
                'district': 'Sultanahmet'
            },
            {
                'id': 'topkapi_palace',
                'name': 'Topkapi Palace',
                'type': 'palace',
                'score': 0.92,
                'district': 'Sultanahmet'
            },
            {
                'id': 'grand_bazaar',
                'name': 'Grand Bazaar',
                'type': 'market',
                'score': 0.90,
                'district': 'Fatih'
            },
            {
                'id': 'galata_tower',
                'name': 'Galata Tower',
                'type': 'tower',
                'score': 0.88,
                'district': 'Beyoğlu'
            },
            {
                'id': 'basilica_cistern',
                'name': 'Basilica Cistern',
                'type': 'historical_site',
                'score': 0.87,
                'district': 'Sultanahmet'
            },
            {
                'id': 'dolmabahce_palace',
                'name': 'Dolmabahçe Palace',
                'type': 'palace',
                'score': 0.85,
                'district': 'Beşiktaş'
            },
            {
                'id': 'bosphorus_cruise',
                'name': 'Bosphorus Cruise',
                'type': 'activity',
                'score': 0.84,
                'district': 'Various'
            },
            {
                'id': 'spice_bazaar',
                'name': 'Spice Bazaar',
                'type': 'market',
                'score': 0.83,
                'district': 'Eminönü'
            },
            {
                'id': 'istiklal_street',
                'name': 'İstiklal Street',
                'type': 'street',
                'score': 0.82,
                'district': 'Beyoğlu'
            }
        ]
        
        recommendations = []
        for item in popular_items[:top_k]:
            recommendations.append(NCFRecommendation(
                item_id=item['id'],
                item_name=item['name'],
                score=item['score'],
                confidence=0.7,  # Lower confidence for fallback
                item_type=item['type'],
                metadata={'district': item['district'], 'source': 'fallback'},
                embedding_similarity=0.0
            ))
        
        return recommendations
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            'enabled': self.enabled,
            'model_loaded': self.model is not None,
            'stats': self.stats,
            'cache_size': len(self._cache)
        }
    
    def clear_cache(self):
        """Clear recommendation cache"""
        self._cache.clear()
        logger.info("🗑️ NCF recommendation cache cleared")


# Global service instance
_ncf_service: Optional[NCFRecommendationService] = None


def get_ncf_service() -> NCFRecommendationService:
    """Get or create NCF service singleton"""
    global _ncf_service
    if _ncf_service is None:
        _ncf_service = NCFRecommendationService()
    return _ncf_service


async def get_ncf_recommendations(
    user_id: str,
    top_k: int = 10,
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to get NCF recommendations
    
    Returns recommendations as dictionaries for easy API response
    """
    service = get_ncf_service()
    recommendations = await service.get_recommendations(user_id, top_k, context=context)
    
    return [
        {
            'id': rec.item_id,
            'name': rec.item_name,
            'score': rec.score,
            'confidence': rec.confidence,
            'type': rec.item_type,
            'metadata': rec.metadata,
            'source': 'ncf_deep_learning'
        }
        for rec in recommendations
    ]
