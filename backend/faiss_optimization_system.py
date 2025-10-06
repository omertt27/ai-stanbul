#!/usr/bin/env python3
"""
FAISS Vector Index Optimization System
=====================================

Implements advanced FAISS indexing strategies for production-scale vector search:
1. IVF (Inverted File) indexes for faster search on large datasets
2. PQ (Product Quantization) for memory efficiency
3. Automatic index selection based on data size
4. Index training and optimization
5. Performance benchmarking
"""

import numpy as np
import faiss
import json
import pickle
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IndexConfig:
    """Configuration for FAISS index optimization"""
    min_docs_for_ivf: int = 1000  # Minimum documents to use IVF
    min_docs_for_pq: int = 5000   # Minimum documents to use PQ
    nlist: int = 100              # Number of clusters for IVF
    m: int = 8                    # Number of sub-quantizers for PQ
    nbits: int = 8                # Bits per sub-quantizer for PQ
    nprobe: int = 10              # Number of clusters to search

class FAISSOptimizer:
    """Advanced FAISS index optimization system"""
    
    def __init__(self, embedding_dim: int = 384, config: IndexConfig = None):
        self.embedding_dim = embedding_dim
        self.config = config or IndexConfig()
        self.current_index = None
        self.index_type = "flat"
        self.is_trained = False
        self.lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            "index_type": "flat",
            "total_vectors": 0,
            "index_size_mb": 0.0,
            "last_search_time_ms": 0.0,
            "avg_search_time_ms": 0.0,
            "search_count": 0,
            "memory_usage_mb": 0.0
        }
    
    def build_optimized_index(self, vectors: np.ndarray, force_rebuild: bool = False) -> bool:
        """Build optimized FAISS index based on dataset size"""
        try:
            with self.lock:
                n_vectors, dim = vectors.shape
                
                if dim != self.embedding_dim:
                    raise ValueError(f"Vector dimension {dim} doesn't match expected {self.embedding_dim}")
                
                print(f"🔧 Building optimized FAISS index for {n_vectors} vectors...")
                
                # Normalize vectors for cosine similarity
                vectors_normalized = vectors.copy().astype(np.float32)
                faiss.normalize_L2(vectors_normalized)
                
                # Choose index type based on dataset size
                if n_vectors < self.config.min_docs_for_ivf:
                    # Use flat index for small datasets
                    index = self._build_flat_index(vectors_normalized)
                    self.index_type = "flat"
                    
                elif n_vectors < self.config.min_docs_for_pq:
                    # Use IVF index for medium datasets
                    index = self._build_ivf_index(vectors_normalized)
                    self.index_type = "ivf"
                    
                else:
                    # Use IVF+PQ index for large datasets
                    index = self._build_ivf_pq_index(vectors_normalized)
                    self.index_type = "ivf_pq"
                
                # Validate index
                if index.ntotal != n_vectors:
                    raise ValueError(f"Index size mismatch: expected {n_vectors}, got {index.ntotal}")
                
                self.current_index = index
                self.is_trained = True
                
                # Update metrics
                self._update_metrics(n_vectors)
                
                print(f"✅ Built {self.index_type.upper()} index: {n_vectors} vectors, {self.metrics['index_size_mb']:.1f}MB")
                return True
                
        except Exception as e:
            print(f"❌ Index building error: {e}")
            return False
    
    def _build_flat_index(self, vectors: np.ndarray) -> faiss.Index:
        """Build flat (brute force) index - best accuracy, slower for large datasets"""
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(vectors)
        return index
    
    def _build_ivf_index(self, vectors: np.ndarray) -> faiss.Index:
        """Build IVF index - faster search with slight accuracy trade-off"""
        n_vectors = vectors.shape[0]
        
        # Adjust nlist based on dataset size
        nlist = min(self.config.nlist, max(int(np.sqrt(n_vectors)), 10))
        
        # Create IVF index
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
        
        # Train the index
        print(f"🎯 Training IVF index with {nlist} clusters...")
        index.train(vectors)
        
        # Add vectors
        index.add(vectors)
        
        # Set search parameters
        index.nprobe = min(self.config.nprobe, nlist)
        
        return index
    
    def _build_ivf_pq_index(self, vectors: np.ndarray) -> faiss.Index:
        """Build IVF+PQ index - memory efficient for very large datasets"""
        n_vectors = vectors.shape[0]
        
        # Adjust parameters based on dataset size
        nlist = min(self.config.nlist * 2, max(int(np.sqrt(n_vectors)), 50))
        
        # Ensure m divides the dimension
        m = self.config.m
        while self.embedding_dim % m != 0 and m > 4:
            m -= 1
        
        # Create IVF+PQ index
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, m, self.config.nbits)
        
        # Train the index
        print(f"🎯 Training IVF+PQ index with {nlist} clusters, m={m}...")
        index.train(vectors)
        
        # Add vectors
        index.add(vectors)
        
        # Set search parameters
        index.nprobe = min(self.config.nprobe * 2, nlist // 2)
        
        return index
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Optimized search with performance tracking"""
        if self.current_index is None:
            raise ValueError("No index available - build index first")
        
        start_time = time.time()
        
        try:
            # Normalize query vector
            query_normalized = query_vector.copy().astype(np.float32)
            if query_normalized.ndim == 1:
                query_normalized = query_normalized.reshape(1, -1)
            faiss.normalize_L2(query_normalized)
            
            # Search
            similarities, indices = self.current_index.search(query_normalized, k)
            
            # Update performance metrics
            search_time = (time.time() - start_time) * 1000
            self._update_search_metrics(search_time)
            
            return similarities, indices
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            raise
    
    def _update_metrics(self, n_vectors: int):
        """Update index metrics"""
        self.metrics.update({
            "index_type": self.index_type,
            "total_vectors": n_vectors,
            "index_size_mb": self._estimate_index_size_mb(n_vectors),
            "memory_usage_mb": self._get_memory_usage_mb()
        })
    
    def _update_search_metrics(self, search_time_ms: float):
        """Update search performance metrics"""
        self.metrics["last_search_time_ms"] = search_time_ms
        self.metrics["search_count"] += 1
        
        # Update rolling average
        current_avg = self.metrics["avg_search_time_ms"]
        count = self.metrics["search_count"]
        self.metrics["avg_search_time_ms"] = (
            (current_avg * (count - 1) + search_time_ms) / count
        )
    
    def _estimate_index_size_mb(self, n_vectors: int) -> float:
        """Estimate index size in MB"""
        if self.index_type == "flat":
            # Flat index: n_vectors * embedding_dim * 4 bytes (float32)
            return (n_vectors * self.embedding_dim * 4) / (1024 * 1024)
        
        elif self.index_type == "ivf":
            # IVF index: similar to flat but with some overhead
            return (n_vectors * self.embedding_dim * 4 * 1.1) / (1024 * 1024)
        
        elif self.index_type == "ivf_pq":
            # IVF+PQ index: much smaller due to quantization
            # Approximately n_vectors * m bytes + overhead
            return (n_vectors * self.config.m * 1.2) / (1024 * 1024)
        
        return 0.0
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage (simplified estimation)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return self._estimate_index_size_mb(self.metrics["total_vectors"])
    
    def benchmark_search_performance(self, test_vectors: np.ndarray, k: int = 10, 
                                   num_queries: int = 100) -> Dict[str, Any]:
        """Benchmark search performance"""
        if self.current_index is None:
            raise ValueError("No index available")
        
        print(f"🏃 Benchmarking search performance ({num_queries} queries)...")
        
        # Generate random test queries
        n_test = min(num_queries, test_vectors.shape[0])
        query_indices = np.random.choice(test_vectors.shape[0], n_test, replace=False)
        test_queries = test_vectors[query_indices]
        
        search_times = []
        accuracy_scores = []
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            try:
                similarities, indices = self.search(query, k=k)
                search_time = (time.time() - start_time) * 1000
                search_times.append(search_time)
                
                # Simple accuracy check (if we find the query itself)
                if query_indices[i] in indices[0]:
                    accuracy_scores.append(1.0)
                else:
                    accuracy_scores.append(0.0)
                
            except Exception as e:
                print(f"   ⚠️ Query {i} failed: {e}")
                search_times.append(float('inf'))
                accuracy_scores.append(0.0)
        
        # Calculate statistics
        valid_times = [t for t in search_times if t != float('inf')]
        
        benchmark_results = {
            "index_type": self.index_type,
            "total_vectors": self.metrics["total_vectors"],
            "queries_tested": len(valid_times),
            "k": k,
            "avg_search_time_ms": np.mean(valid_times) if valid_times else 0,
            "min_search_time_ms": np.min(valid_times) if valid_times else 0,
            "max_search_time_ms": np.max(valid_times) if valid_times else 0,
            "p95_search_time_ms": np.percentile(valid_times, 95) if valid_times else 0,
            "p99_search_time_ms": np.percentile(valid_times, 99) if valid_times else 0,
            "accuracy": np.mean(accuracy_scores) if accuracy_scores else 0,
            "index_size_mb": self.metrics["index_size_mb"],
            "memory_usage_mb": self.metrics["memory_usage_mb"],
            "failed_queries": num_queries - len(valid_times)
        }
        
        print(f"📊 Benchmark Results:")
        print(f"   ⚡ Avg search time: {benchmark_results['avg_search_time_ms']:.2f}ms")
        print(f"   📊 P95 search time: {benchmark_results['p95_search_time_ms']:.2f}ms")
        print(f"   🎯 Accuracy: {benchmark_results['accuracy']:.1%}")
        print(f"   💾 Index size: {benchmark_results['index_size_mb']:.1f}MB")
        
        return benchmark_results
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for index optimization"""
        recommendations = []
        
        n_vectors = self.metrics["total_vectors"]
        avg_search_time = self.metrics["avg_search_time_ms"]
        
        # Size-based recommendations
        if n_vectors > self.config.min_docs_for_pq and self.index_type != "ivf_pq":
            recommendations.append({
                "type": "index_upgrade",
                "priority": "high",
                "message": f"Consider IVF+PQ index for {n_vectors} vectors to reduce memory usage",
                "estimated_benefit": "60-80% memory reduction"
            })
        
        elif n_vectors > self.config.min_docs_for_ivf and self.index_type == "flat":
            recommendations.append({
                "type": "index_upgrade",
                "priority": "medium",
                "message": f"Consider IVF index for {n_vectors} vectors to improve search speed",
                "estimated_benefit": "2-5x faster search"
            })
        
        # Performance-based recommendations
        if avg_search_time > 100:  # > 100ms average
            recommendations.append({
                "type": "performance_tuning",
                "priority": "medium",
                "message": f"Average search time ({avg_search_time:.1f}ms) is high",
                "suggested_actions": [
                    "Increase nprobe for better speed/accuracy trade-off",
                    "Consider GPU acceleration for large datasets",
                    "Add caching layer for frequent queries"
                ]
            })
        
        # Memory-based recommendations
        memory_usage = self.metrics["memory_usage_mb"]
        if memory_usage > 1000:  # > 1GB
            recommendations.append({
                "type": "memory_optimization",
                "priority": "medium",
                "message": f"High memory usage ({memory_usage:.0f}MB)",
                "suggested_actions": [
                    "Use PQ quantization to reduce memory",
                    "Consider distributed indexing",
                    "Implement index sharding"
                ]
            })
        
        return recommendations
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current optimization metrics"""
        return {
            **self.metrics,
            "is_trained": self.is_trained,
            "config": {
                "min_docs_for_ivf": self.config.min_docs_for_ivf,
                "min_docs_for_pq": self.config.min_docs_for_pq,
                "nlist": self.config.nlist,
                "nprobe": self.config.nprobe
            },
            "timestamp": datetime.now().isoformat()
        }

def test_faiss_optimization():
    """Test FAISS optimization system"""
    print("🧪 Testing FAISS Optimization System...")
    
    # Create test data
    n_vectors = 2000  # Medium size to test IVF
    embedding_dim = 384
    
    print(f"📊 Generating {n_vectors} test vectors (dim={embedding_dim})...")
    test_vectors = np.random.random((n_vectors, embedding_dim)).astype(np.float32)
    
    # Initialize optimizer
    optimizer = FAISSOptimizer(embedding_dim=embedding_dim)
    
    # Build optimized index
    success = optimizer.build_optimized_index(test_vectors)
    if not success:
        return False
    
    # Test search
    query_vector = test_vectors[0]  # Use first vector as query
    similarities, indices = optimizer.search(query_vector, k=5)
    
    print(f"🔍 Search test: Found {len(indices[0])} results")
    print(f"   Top similarity: {similarities[0][0]:.4f}")
    
    # Benchmark performance
    benchmark_results = optimizer.benchmark_search_performance(test_vectors, num_queries=50)
    
    # Get recommendations
    recommendations = optimizer.get_optimization_recommendations()
    if recommendations:
        print(f"💡 Optimization recommendations: {len(recommendations)}")
        for rec in recommendations:
            print(f"   • {rec['message']}")
    
    # Get metrics
    metrics = optimizer.get_metrics()
    print(f"📊 Final metrics: {metrics['index_type']} index, {metrics['avg_search_time_ms']:.2f}ms avg search")
    
    return True

if __name__ == "__main__":
    # Test the FAISS optimization system
    success = test_faiss_optimization()
    if success:
        print("✅ FAISS Optimization System is working correctly!")
    else:
        print("❌ FAISS Optimization System test failed")
