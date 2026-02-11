#!/usr/bin/env python3
"""
ONNX Inference Engine for NCF Model

Provides fast inference using ONNX runtime (3-5x faster than PyTorch).

Features:
- ONNX runtime inference
- Batch prediction support
- Performance optimizations
- Drop-in replacement for PyTorch model

Usage:
    from backend.ml.deep_learning.onnx_inference import ONNXNCFPredictor
    
    predictor = ONNXNCFPredictor('models/ncf_model.onnx')
    scores = predictor.predict(user_ids=[1, 2], item_ids=[10, 20])
"""

import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class ONNXNCFPredictor:
    """
    ONNX-based NCF prediction engine for production deployment.
    """
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        session_options: Optional[ort.SessionOptions] = None
    ):
        """
        Initialize ONNX predictor.
        
        Args:
            model_path: Path to ONNX model file
            providers: ONNX execution providers (default: ['CPUExecutionProvider'])
            session_options: ONNX session configuration
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        
        # Set execution providers (CPU, CUDA, etc.)
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Configure session options for best performance
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 4  # Tune based on CPU cores
        
        # Create ONNX runtime session
        logger.info(f"Loading ONNX model from {self.model_path}")
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=providers
        )
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Cache input/output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"✅ ONNX model loaded successfully")
        logger.info(f"   Input names: {self.input_names}")
        logger.info(f"   Output names: {self.output_names}")
        logger.info(f"   Execution providers: {self.session.get_providers()}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from JSON file."""
        metadata_path = self.model_path.with_suffix('.json')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            return {}
    
    def predict(
        self,
        user_ids: List[int],
        item_ids: List[int]
    ) -> np.ndarray:
        """
        Predict interaction scores for user-item pairs.
        
        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs (same length as user_ids)
            
        Returns:
            Array of prediction scores (shape: [batch_size, 1])
        """
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have same length")
        
        # Convert to numpy arrays (batch_size, 1)
        user_array = np.array(user_ids, dtype=np.int64).reshape(-1, 1)
        item_array = np.array(item_ids, dtype=np.int64).reshape(-1, 1)
        
        # Run inference
        ort_inputs = {
            'user_ids': user_array,
            'item_ids': item_array
        }
        
        outputs = self.session.run(self.output_names, ort_inputs)
        
        return outputs[0]
    
    def predict_for_user(
        self,
        user_id: int,
        item_ids: List[int],
        top_k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Get top-K recommendations for a single user.
        
        Args:
            user_id: User ID
            item_ids: List of candidate item IDs
            top_k: Number of top recommendations to return (default: all)
            
        Returns:
            List of (item_id, score) tuples, sorted by score descending
        """
        # Predict scores for all items
        user_ids = [user_id] * len(item_ids)
        scores = self.predict(user_ids, item_ids)
        
        # Combine item_ids with scores
        item_scores = list(zip(item_ids, scores.flatten()))
        
        # Sort by score descending
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-K
        if top_k is not None:
            item_scores = item_scores[:top_k]
        
        return item_scores
    
    def predict_batch(
        self,
        batch_user_ids: List[List[int]],
        batch_item_ids: List[List[int]]
    ) -> List[np.ndarray]:
        """
        Batch prediction for multiple users.
        
        Args:
            batch_user_ids: List of user ID lists
            batch_item_ids: List of item ID lists
            
        Returns:
            List of prediction arrays
        """
        results = []
        
        for user_ids, item_ids in zip(batch_user_ids, batch_item_ids):
            scores = self.predict(user_ids, item_ids)
            results.append(scores)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_path': str(self.model_path),
            'num_users': self.metadata.get('num_users'),
            'num_items': self.metadata.get('num_items'),
            'embedding_dim': self.metadata.get('embedding_dim'),
            'mlp_layers': self.metadata.get('mlp_layers'),
            'training_metrics': self.metadata.get('training_metrics'),
            'onnx_info': {
                'opset_version': self.metadata.get('onnx_export', {}).get('opset_version'),
                'model_size_bytes': self.metadata.get('onnx_export', {}).get('model_size_bytes'),
                'comparison_metrics': self.metadata.get('onnx_export', {}).get('comparison_metrics')
            },
            'execution_providers': self.session.get_providers()
        }
    
    def benchmark(
        self,
        num_iterations: int = 1000,
        batch_sizes: List[int] = [1, 10, 50, 100]
    ) -> Dict[int, Dict[str, float]]:
        """
        Benchmark inference performance.
        
        Args:
            num_iterations: Number of iterations per batch size
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary mapping batch_size -> metrics
        """
        import time
        
        results = {}
        
        max_user = self.metadata.get('num_users', 1000)
        max_item = self.metadata.get('num_items', 1000)
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch_size={batch_size}...")
            
            latencies = []
            
            for _ in range(num_iterations):
                # Generate random inputs
                user_ids = np.random.randint(0, max_user, size=batch_size).tolist()
                item_ids = np.random.randint(0, max_item, size=batch_size).tolist()
                
                # Time inference
                start = time.perf_counter()
                self.predict(user_ids, item_ids)
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # Convert to ms
            
            # Calculate statistics
            latencies_np = np.array(latencies)
            
            results[batch_size] = {
                'mean_ms': float(np.mean(latencies_np)),
                'median_ms': float(np.median(latencies_np)),
                'p95_ms': float(np.percentile(latencies_np, 95)),
                'p99_ms': float(np.percentile(latencies_np, 99)),
                'min_ms': float(np.min(latencies_np)),
                'max_ms': float(np.max(latencies_np)),
                'throughput_qps': 1000.0 / np.mean(latencies_np)
            }
            
            logger.info(f"  Mean latency: {results[batch_size]['mean_ms']:.2f}ms")
            logger.info(f"  P95 latency: {results[batch_size]['p95_ms']:.2f}ms")
            logger.info(f"  Throughput: {results[batch_size]['throughput_qps']:.0f} QPS")
        
        return results


def main():
    """Example usage and benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ONNX NCF Inference Engine')
    parser.add_argument(
        '--model',
        type=str,
        default='backend/ml/deep_learning/models/ncf_model.onnx',
        help='Path to ONNX model'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run prediction test'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ONNXNCFPredictor(args.model)
    
    # Show model info
    info = predictor.get_model_info()
    print("\n📊 Model Information:")
    print(f"  Users: {info['num_users']}")
    print(f"  Items: {info['num_items']}")
    print(f"  Embedding dim: {info['embedding_dim']}")
    print(f"  Execution providers: {info['execution_providers']}")
    
    # Run test prediction
    if args.test:
        print("\n🔮 Test Prediction:")
        scores = predictor.predict(
            user_ids=[0, 1, 2],
            item_ids=[10, 20, 30]
        )
        print(f"  Scores: {scores.flatten()}")
        
        # Top-K recommendations
        print("\n🏆 Top-5 Recommendations for User 0:")
        top_items = predictor.predict_for_user(
            user_id=0,
            item_ids=list(range(100)),
            top_k=5
        )
        for item_id, score in top_items:
            print(f"  Item {item_id}: {score:.4f}")
    
    # Run benchmark
    if args.benchmark:
        print("\n⚡ Performance Benchmark:")
        results = predictor.benchmark(
            num_iterations=100,
            batch_sizes=[1, 10, 50, 100]
        )
        
        print("\n| Batch Size | Mean (ms) | P95 (ms) | Throughput (QPS) |")
        print("|------------|-----------|----------|------------------|")
        for batch_size, metrics in results.items():
            print(f"| {batch_size:10d} | {metrics['mean_ms']:9.2f} | "
                  f"{metrics['p95_ms']:8.2f} | {metrics['throughput_qps']:16.0f} |")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
