"""
Lightweight NCF Inference Service
Efficient inference with caching and batch processing

Week 5-6 Implementation - Budget-Optimized Roadmap
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle
import time

from backend.ml.models.lightweight_ncf import LightweightNCF

logger = logging.getLogger(__name__)


class LightweightNCFInference:
    """
    Efficient inference service for NCF
    
    Features:
    - FP16 inference for 2x speedup
    - Precomputed item embeddings
    - Batch processing
    - Caching
    """
    
    def __init__(
        self,
        model_path: str,
        mappings_path: str,
        device: str = 'cuda',
        use_fp16: bool = True
    ):
        """
        Initialize inference service
        
        Args:
            model_path: Path to trained model checkpoint
            mappings_path: Path to ID mappings
            device: Device to run inference on
            use_fp16: Use FP16 for faster inference
        """
        self.device = device
        self.use_fp16 = use_fp16 and (device == 'cuda')
        
        logger.info(f"🔧 Initializing NCF inference service...")
        logger.info(f"   Device: {device}")
        logger.info(f"   FP16: {self.use_fp16}")
        
        # Load model
        self.model = LightweightNCF.load(model_path, device)
        self.model.eval()
        
        # Convert to FP16 if enabled
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("✅ Model converted to FP16")
        
        # Load mappings
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        
        self.user_id_map = mappings['user_id_map']
        self.item_id_map = mappings['item_id_map']
        self.reverse_user_map = mappings['reverse_user_map']
        self.reverse_item_map = mappings['reverse_item_map']
        self.num_users = mappings['num_users']
        self.num_items = mappings['num_items']
        
        logger.info(f"✅ Loaded {self.num_users} users and {self.num_items} items")
        
        # Precompute item embeddings
        self._precompute_item_embeddings()
        
        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        logger.info("✅ Inference service ready!")
    
    def _precompute_item_embeddings(self):
        """Precompute all item embeddings for faster inference"""
        logger.info("🔄 Precomputing item embeddings...")
        
        with torch.no_grad():
            # Create tensor of all item IDs
            all_item_ids = torch.arange(self.num_items, dtype=torch.long, device=self.device)
            
            # Get embeddings
            item_emb_gmf = self.model.item_embedding_gmf(all_item_ids)
            item_emb_mlp = self.model.item_embedding_mlp(all_item_ids)
            
            if self.use_fp16:
                item_emb_gmf = item_emb_gmf.half()
                item_emb_mlp = item_emb_mlp.half()
            
            self.precomputed_item_emb_gmf = item_emb_gmf
            self.precomputed_item_emb_mlp = item_emb_mlp
        
        logger.info(f"✅ Precomputed embeddings for {self.num_items} items")
    
    @torch.no_grad()
    def predict_for_user(
        self,
        user_id: str,
        candidate_items: List[str],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top-k recommendations for a user
        
        Args:
            user_id: User ID (original, not mapped)
            candidate_items: List of candidate item IDs
            top_k: Number of recommendations to return
        
        Returns:
            List of (item_id, score) tuples
        """
        start_time = time.time()
        
        # Map user ID
        if user_id not in self.user_id_map:
            logger.warning(f"⚠️ Unknown user: {user_id}")
            return []
        
        user_idx = self.user_id_map[user_id]
        
        # Map item IDs
        item_indices = []
        original_items = []
        
        for item_id in candidate_items:
            if item_id in self.item_id_map:
                item_indices.append(self.item_id_map[item_id])
                original_items.append(item_id)
        
        if not item_indices:
            logger.warning(f"⚠️ No valid candidate items")
            return []
        
        # Create tensors
        user_tensor = torch.tensor([user_idx] * len(item_indices), dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        
        if self.use_fp16:
            # Get embeddings with FP16
            user_emb_gmf = self.model.user_embedding_gmf(user_tensor).half()
            user_emb_mlp = self.model.user_embedding_mlp(user_tensor).half()
        else:
            user_emb_gmf = self.model.user_embedding_gmf(user_tensor)
            user_emb_mlp = self.model.user_embedding_mlp(user_tensor)
        
        # Use precomputed item embeddings
        item_emb_gmf = self.precomputed_item_emb_gmf[item_tensor]
        item_emb_mlp = self.precomputed_item_emb_mlp[item_tensor]
        
        # GMF
        gmf_output = user_emb_gmf * item_emb_gmf
        
        # MLP
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.model.mlp(mlp_input)
        
        # Fusion and prediction
        fusion = torch.cat([gmf_output, mlp_output], dim=-1)
        scores = torch.sigmoid(self.model.output(fusion)).squeeze()
        
        # Convert to numpy
        scores_np = scores.cpu().float().numpy()
        
        # Get top-k
        if len(scores_np) <= top_k:
            top_indices = np.argsort(scores_np)[::-1]
        else:
            top_indices = np.argpartition(scores_np, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores_np[top_indices])[::-1]]
        
        # Create results
        results = [
            (original_items[idx], float(scores_np[idx]))
            for idx in top_indices
        ]
        
        # Update statistics
        elapsed = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += elapsed
        
        return results
    
    @torch.no_grad()
    def predict_batch(
        self,
        user_items: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Batch prediction for multiple (user, item) pairs
        
        Args:
            user_items: List of (user_id, item_id) tuples
        
        Returns:
            Array of scores
        """
        # Map IDs
        user_indices = []
        item_indices = []
        
        for user_id, item_id in user_items:
            if user_id in self.user_id_map and item_id in self.item_id_map:
                user_indices.append(self.user_id_map[user_id])
                item_indices.append(self.item_id_map[item_id])
        
        if not user_indices:
            return np.array([])
        
        # Create tensors
        user_tensor = torch.tensor(user_indices, dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
        
        # Forward pass
        scores = self.model(user_tensor, item_tensor)
        
        return scores.cpu().float().numpy()
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        avg_time = (self.total_inference_time / self.inference_count * 1000) if self.inference_count > 0 else 0
        
        return {
            'total_inferences': self.inference_count,
            'avg_latency_ms': avg_time,
            'total_time_seconds': self.total_inference_time,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'device': self.device,
            'fp16_enabled': self.use_fp16
        }
    
    def reset_statistics(self):
        """Reset inference statistics"""
        self.inference_count = 0
        self.total_inference_time = 0.0


# Global inference service instance
_inference_service = None


def get_ncf_inference(
    model_path: Optional[str] = None,
    mappings_path: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> LightweightNCFInference:
    """
    Get or create NCF inference service (singleton pattern)
    
    Args:
        model_path: Path to model checkpoint
        mappings_path: Path to ID mappings
        device: Device to run on
    
    Returns:
        NCF inference service
    """
    global _inference_service
    
    if _inference_service is None:
        if model_path is None:
            model_path = './checkpoints/ncf/best_model.pth'
        if mappings_path is None:
            mappings_path = './data/ncf/mappings.pkl'
        
        _inference_service = LightweightNCFInference(
            model_path=model_path,
            mappings_path=mappings_path,
            device=device
        )
    
    return _inference_service


if __name__ == "__main__":
    """Test inference service"""
    
    print("🧪 Testing NCF Inference Service...")
    
    # Test with dummy data
    try:
        service = get_ncf_inference(
            model_path='./checkpoints/ncf/best_model.pth',
            mappings_path='./data/ncf/mappings.pkl'
        )
        
        print("✅ Service initialized")
        
        # Test prediction
        # Note: Replace with actual user/item IDs from your data
        # results = service.predict_for_user(
        #     user_id='user_123',
        #     candidate_items=['item_1', 'item_2', 'item_3'],
        #     top_k=5
        # )
        
        # print(f"✅ Predictions: {results}")
        
        stats = service.get_statistics()
        print(f"📊 Statistics: {stats}")
        
    except Exception as e:
        print(f"⚠️ Test failed (expected if no trained model exists): {e}")
