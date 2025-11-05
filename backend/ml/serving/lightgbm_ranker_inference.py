"""
LightGBM Ranker Inference Service
Week 7-8: Production-ready inference for ranking

Features:
- Fast batch inference
- Feature caching
- Integration with NCF embeddings
- Thread-safe operation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import threading
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lightgbm_ranker import LightGBMRanker


class RankerInferenceService:
    """
    Production inference service for LightGBM ranker
    """
    
    def __init__(
        self,
        model_path: str,
        feature_stats_path: str,
        ncf_embeddings_path: Optional[str] = None,
        cache_size: int = 1000
    ):
        """
        Initialize inference service
        
        Args:
            model_path: Path to trained LightGBM model
            feature_stats_path: Path to feature statistics
            ncf_embeddings_path: Path to NCF embeddings (optional)
            cache_size: Size of feature cache
        """
        print(f"🚀 Initializing LightGBM Ranker Inference Service...")
        
        # Load model
        self.ranker = LightGBMRanker()
        self.ranker.load(model_path)
        print(f"✅ Model loaded from {model_path}")
        
        # Load feature stats
        with open(feature_stats_path, 'rb') as f:
            self.feature_stats = pickle.load(f)
        print(f"✅ Feature stats loaded from {feature_stats_path}")
        
        # Load NCF embeddings if available
        self.ncf_embeddings = None
        if ncf_embeddings_path and Path(ncf_embeddings_path).exists():
            with open(ncf_embeddings_path, 'rb') as f:
                self.ncf_embeddings = pickle.load(f)
            print(f"✅ NCF embeddings loaded from {ncf_embeddings_path}")
        
        # Feature cache for fast repeated queries
        self.cache_size = cache_size
        self.feature_cache: Dict = {}
        self.cache_lock = threading.Lock()
        
        # Statistics cache (user/item stats)
        self.user_stats_cache: Dict = {}
        self.item_stats_cache: Dict = {}
        
        print("✅ Inference service ready!")
    
    def rank_items_for_user(
        self,
        user_id: int,
        item_ids: List[int],
        user_stats: Optional[Dict] = None,
        item_stats: Optional[Dict[int, Dict]] = None,
        current_time: Optional[datetime] = None
    ) -> List[Tuple[int, float]]:
        """
        Rank items for a specific user
        
        Args:
            user_id: User ID
            item_ids: List of candidate item IDs
            user_stats: Optional precomputed user statistics
            item_stats: Optional precomputed item statistics (dict of item_id -> stats)
            current_time: Optional current timestamp
            
        Returns:
            List of (item_id, score) tuples, sorted by score descending
        """
        if not item_ids:
            return []
        
        # Extract features for all user-item pairs
        features = self._extract_features(
            user_id,
            item_ids,
            user_stats,
            item_stats,
            current_time
        )
        
        # Get predictions
        scores = self.ranker.predict(features)
        
        # Sort items by score
        ranked_items = sorted(
            zip(item_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked_items
    
    def rank_items_batch(
        self,
        user_ids: List[int],
        item_ids_per_user: List[List[int]],
        user_stats_batch: Optional[List[Dict]] = None,
        item_stats_batch: Optional[List[Dict[int, Dict]]] = None,
        current_time: Optional[datetime] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        Batch ranking for multiple users
        
        Args:
            user_ids: List of user IDs
            item_ids_per_user: List of item ID lists (one per user)
            user_stats_batch: Optional batch of user statistics
            item_stats_batch: Optional batch of item statistics
            current_time: Optional current timestamp
            
        Returns:
            List of ranked item lists (one per user)
        """
        results = []
        
        for i, (user_id, item_ids) in enumerate(zip(user_ids, item_ids_per_user)):
            user_stats = user_stats_batch[i] if user_stats_batch else None
            item_stats = item_stats_batch[i] if item_stats_batch else None
            
            ranked = self.rank_items_for_user(
                user_id,
                item_ids,
                user_stats,
                item_stats,
                current_time
            )
            results.append(ranked)
        
        return results
    
    def _extract_features(
        self,
        user_id: int,
        item_ids: List[int],
        user_stats: Optional[Dict],
        item_stats: Optional[Dict[int, Dict]],
        current_time: Optional[datetime]
    ) -> np.ndarray:
        """
        Extract features for user-item pairs
        
        Returns:
            Feature matrix (n_items, n_features)
        """
        n_items = len(item_ids)
        feature_list = []
        
        # Get user stats (from cache or provided)
        if user_stats is None:
            user_stats = self.user_stats_cache.get(user_id, {})
        
        # Get item stats (from cache or provided)
        if item_stats is None:
            item_stats = {
                item_id: self.item_stats_cache.get(item_id, {})
                for item_id in item_ids
            }
        
        # Get current time
        if current_time is None:
            current_time = datetime.now()
        
        # 1. User features
        user_features = np.array([
            user_stats.get('n_interactions', 0),
            user_stats.get('avg_rating', 3.0),
            user_stats.get('std_rating', 0.0),
            user_stats.get('days_since_last_interaction', 0),
        ])
        
        # Repeat for all items
        user_features_matrix = np.tile(user_features, (n_items, 1))
        feature_list.append(user_features_matrix)
        
        # 2. Item features
        item_features_list = []
        for item_id in item_ids:
            stats = item_stats.get(item_id, {})
            item_feats = np.array([
                stats.get('popularity', 0),
                stats.get('log_popularity', 0),
                stats.get('avg_rating', 3.0),
                stats.get('std_rating', 0.0),
            ])
            item_features_list.append(item_feats)
        
        item_features_matrix = np.array(item_features_list)
        feature_list.append(item_features_matrix)
        
        # 3. Temporal features
        temporal_features = np.array([
            current_time.hour,
            current_time.weekday(),
            int(current_time.weekday() in [5, 6]),  # is_weekend
            current_time.month,
        ])
        
        temporal_features_matrix = np.tile(temporal_features, (n_items, 1))
        feature_list.append(temporal_features_matrix)
        
        # 4. User-Item interaction features
        # Rating deviation from user mean
        user_avg_rating = user_stats.get('avg_rating', 3.0)
        item_avg_ratings = np.array([
            item_stats.get(item_id, {}).get('avg_rating', 3.0)
            for item_id in item_ids
        ])
        
        # Use item average as proxy for expected rating
        user_rating_dev = item_avg_ratings - user_avg_rating
        item_rating_dev = np.zeros_like(item_avg_ratings)  # Placeholder
        
        feature_list.append(user_rating_dev.reshape(-1, 1))
        feature_list.append(item_rating_dev.reshape(-1, 1))
        
        # 5. NCF embeddings (if available)
        if self.ncf_embeddings is not None:
            ncf_features = self._get_ncf_features(user_id, item_ids)
            if ncf_features is not None:
                feature_list.append(ncf_features)
        
        # Concatenate all features
        X = np.hstack(feature_list).astype(np.float32)
        
        # Normalize using stored statistics
        X = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        return X
    
    def _get_ncf_features(
        self,
        user_id: int,
        item_ids: List[int]
    ) -> Optional[np.ndarray]:
        """
        Extract NCF embedding features
        
        Returns:
            NCF features (n_items, emb_dim + 2)
        """
        if self.ncf_embeddings is None:
            return None
        
        user_emb_dict = self.ncf_embeddings.get('user', {})
        item_emb_dict = self.ncf_embeddings.get('item', {})
        
        if user_id not in user_emb_dict:
            return None
        
        user_emb = user_emb_dict[user_id]
        
        features = []
        for item_id in item_ids:
            if item_id in item_emb_dict:
                item_emb = item_emb_dict[item_id]
                
                # Compute embedding features
                dot_product = np.dot(user_emb, item_emb)
                
                # Cosine similarity
                user_norm = np.linalg.norm(user_emb)
                item_norm = np.linalg.norm(item_emb)
                cosine_sim = dot_product / (user_norm * item_norm + 1e-8)
                
                # Element-wise product
                hadamard = user_emb * item_emb
                
                emb_features = np.concatenate([
                    [dot_product, cosine_sim],
                    hadamard
                ])
                
                features.append(emb_features)
            else:
                # Use zero features if item not found
                emb_dim = len(user_emb)
                features.append(np.zeros(emb_dim + 2))
        
        return np.array(features, dtype=np.float32)
    
    def update_user_stats(self, user_id: int, stats: Dict):
        """Update cached user statistics"""
        with self.cache_lock:
            self.user_stats_cache[user_id] = stats
            
            # Limit cache size
            if len(self.user_stats_cache) > self.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self.user_stats_cache))
                del self.user_stats_cache[oldest_key]
    
    def update_item_stats(self, item_id: int, stats: Dict):
        """Update cached item statistics"""
        with self.cache_lock:
            self.item_stats_cache[item_id] = stats
            
            # Limit cache size
            if len(self.item_stats_cache) > self.cache_size:
                oldest_key = next(iter(self.item_stats_cache))
                del self.item_stats_cache[oldest_key]
    
    def batch_update_stats(
        self,
        user_stats: Optional[Dict[int, Dict]] = None,
        item_stats: Optional[Dict[int, Dict]] = None
    ):
        """Batch update statistics cache"""
        with self.cache_lock:
            if user_stats:
                self.user_stats_cache.update(user_stats)
            
            if item_stats:
                self.item_stats_cache.update(item_stats)
    
    def get_cache_info(self) -> Dict:
        """Get cache statistics"""
        return {
            'user_stats_cache_size': len(self.user_stats_cache),
            'item_stats_cache_size': len(self.item_stats_cache),
            'max_cache_size': self.cache_size,
        }


def test_inference_service():
    """
    Test the inference service with synthetic data
    """
    print("="*60)
    print("🧪 Testing LightGBM Ranker Inference Service")
    print("="*60)
    
    # First, check if model exists, otherwise train it
    model_path = Path('models/ranker/lightgbm_ranker.pkl')
    feature_stats_path = Path('models/ranker/feature_stats.pkl')
    
    if not model_path.exists() or not feature_stats_path.exists():
        print("⚠️  Model not found. Training model first...")
        sys.path.append(str(Path(__file__).parent.parent))
        from training.train_lightgbm_ranker import train_ranker
        
        train_ranker(
            data_dir='data/ranker',
            output_dir='models/ranker',
            use_synthetic=True,
            n_estimators=50
        )
    
    # Initialize inference service
    service = RankerInferenceService(
        model_path=str(model_path),
        feature_stats_path=str(feature_stats_path),
        cache_size=100
    )
    
    print("\n" + "="*60)
    print("🎯 Testing Single User Ranking")
    print("="*60)
    
    # Test single user ranking
    user_id = 42
    candidate_items = [1, 5, 10, 20, 50, 100, 150]
    
    # Mock user stats
    user_stats = {
        'n_interactions': 10,
        'avg_rating': 4.2,
        'std_rating': 0.5,
        'days_since_last_interaction': 2.0
    }
    
    # Mock item stats
    item_stats = {
        item_id: {
            'popularity': np.random.randint(10, 100),
            'log_popularity': np.log1p(np.random.randint(10, 100)),
            'avg_rating': np.random.uniform(3.5, 4.8),
            'std_rating': np.random.uniform(0.3, 0.8),
        }
        for item_id in candidate_items
    }
    
    # Rank items
    ranked_items = service.rank_items_for_user(
        user_id=user_id,
        item_ids=candidate_items,
        user_stats=user_stats,
        item_stats=item_stats
    )
    
    print(f"👤 User ID: {user_id}")
    print(f"📦 Candidate items: {len(candidate_items)}")
    print("\n📊 Ranked items:")
    for rank, (item_id, score) in enumerate(ranked_items, 1):
        print(f"  {rank}. Item {item_id}: {score:.4f}")
    
    # Test batch ranking
    print("\n" + "="*60)
    print("🎯 Testing Batch Ranking")
    print("="*60)
    
    user_ids = [10, 20, 30]
    item_ids_per_user = [
        [1, 2, 3, 4, 5],
        [10, 20, 30],
        [100, 101, 102, 103, 104, 105]
    ]
    
    batch_results = service.rank_items_batch(
        user_ids=user_ids,
        item_ids_per_user=item_ids_per_user
    )
    
    for user_id, ranked_items in zip(user_ids, batch_results):
        print(f"\n👤 User {user_id}:")
        for rank, (item_id, score) in enumerate(ranked_items[:3], 1):
            print(f"  {rank}. Item {item_id}: {score:.4f}")
    
    # Cache info
    print("\n" + "="*60)
    print("📊 Cache Statistics")
    print("="*60)
    cache_info = service.get_cache_info()
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Inference service test completed successfully!")


if __name__ == "__main__":
    test_inference_service()
