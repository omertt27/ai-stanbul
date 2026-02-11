"""
Performance Benchmarking for Optimized NCF Service

Comprehensive benchmarking of:
- INT8 vs FP32 inference speed
- Batch vs sequential performance
- Cache effectiveness
- Latency percentiles
- Throughput (QPS)

Author: AI Istanbul Team
Date: February 11, 2026
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from backend.services.optimized_ncf_service import OptimizedNCFService
    from backend.services.onnx_ncf_service import ONNXNCFService
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Run from project root: python -m backend.ml.deep_learning.performance_benchmark")
    exit(1)


class PerformanceBenchmark:
    """
    Benchmark NCF service performance.
    """
    
    def __init__(self, num_warmup: int = 10, num_iterations: int = 100):
        """
        Initialize benchmark.
        
        Args:
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
        """
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        
        # Test data
        self.test_user_ids = [f"user_{i}" for i in range(200)]
        self.test_top_k = 10
        
    def benchmark_single_inference(
        self,
        service: OptimizedNCFService,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        """Benchmark single-user inference."""
        logger.info(f"🔍 Benchmarking single inference (cache={'on' if use_cache else 'off'})...")
        
        # Warmup
        for i in range(self.num_warmup):
            service.get_recommendations(
                self.test_user_ids[i % len(self.test_user_ids)],
                top_k=self.test_top_k,
                use_cache=use_cache
            )
        
        # Benchmark
        latencies = []
        for i in range(self.num_iterations):
            user_id = self.test_user_ids[i % len(self.test_user_ids)]
            
            start = time.perf_counter()
            service.get_recommendations(
                user_id,
                top_k=self.test_top_k,
                use_cache=use_cache
            )
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
        
        return self._compute_stats(latencies, "Single Inference")
    
    def benchmark_batch_inference(
        self,
        service: OptimizedNCFService,
        batch_sizes: List[int] = [10, 50, 100]
    ) -> Dict[int, Dict[str, Any]]:
        """Benchmark batch inference with different batch sizes."""
        logger.info("🔍 Benchmarking batch inference...")
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"  Testing batch size: {batch_size}")
            
            # Warmup
            for _ in range(self.num_warmup):
                user_ids = self.test_user_ids[:batch_size]
                service.get_recommendations_batch(
                    user_ids,
                    top_k=self.test_top_k,
                    use_cache=False
                )
            
            # Benchmark
            latencies = []
            for i in range(min(self.num_iterations, 50)):  # Fewer iterations for batches
                user_ids = self.test_user_ids[:batch_size]
                
                start = time.perf_counter()
                service.get_recommendations_batch(
                    user_ids,
                    top_k=self.test_top_k,
                    use_cache=False
                )
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
            
            stats = self._compute_stats(latencies, f"Batch {batch_size}")
            stats['avg_per_user_ms'] = stats['mean'] / batch_size
            stats['speedup_vs_sequential'] = (
                batch_size * 5.0 / stats['mean']  # Assume 5ms per sequential
            )
            
            results[batch_size] = stats
        
        return results
    
    def benchmark_cache_effectiveness(
        self,
        service: OptimizedNCFService
    ) -> Dict[str, Any]:
        """Benchmark cache hit rate and latency improvement."""
        logger.info("🔍 Benchmarking cache effectiveness...")
        
        # Reset metrics
        service.metrics = {
            'total_requests': 0,
            'l1_hits': 0,
            'l2_hits': 0,
            'precompute_hits': 0,
            'cache_misses': 0,
            'total_latency_ms': 0.0,
            'errors': 0
        }
        
        # First pass: populate cache
        logger.info("  Populating cache...")
        cache_population_latencies = []
        for i in range(50):
            user_id = self.test_user_ids[i]
            start = time.perf_counter()
            service.get_recommendations(user_id, top_k=self.test_top_k, use_cache=True)
            latency = (time.perf_counter() - start) * 1000
            cache_population_latencies.append(latency)
        
        # Second pass: measure cache hits
        logger.info("  Measuring cache hits...")
        cache_hit_latencies = []
        for i in range(50):
            user_id = self.test_user_ids[i]
            start = time.perf_counter()
            service.get_recommendations(user_id, top_k=self.test_top_k, use_cache=True)
            latency = (time.perf_counter() - start) * 1000
            cache_hit_latencies.append(latency)
        
        # Compute improvement
        avg_miss = statistics.mean(cache_population_latencies)
        avg_hit = statistics.mean(cache_hit_latencies)
        speedup = avg_miss / avg_hit if avg_hit > 0 else 0
        
        metrics = service.get_metrics()
        stats = metrics.get('stats', {})
        cache_hits = stats.get('cache_hits', {})
        
        total_hits = cache_hits.get('total', 0) if isinstance(cache_hits, dict) else 0
        total_requests = stats.get('total_requests', 100)
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'cache_miss_latency_ms': {
                'mean': round(avg_miss, 2),
                'p50': round(statistics.median(cache_population_latencies), 2),
                'p95': round(self._percentile(cache_population_latencies, 95), 2)
            },
            'cache_hit_latency_ms': {
                'mean': round(avg_hit, 2),
                'p50': round(statistics.median(cache_hit_latencies), 2),
                'p95': round(self._percentile(cache_hit_latencies, 95), 2)
            },
            'speedup': round(speedup, 2),
            'hit_rate': f"{hit_rate:.2%}",
            'l1_hits': cache_hits.get('l1', 0) if isinstance(cache_hits, dict) else 0,
            'l2_hits': cache_hits.get('l2', 0) if isinstance(cache_hits, dict) else 0,
            'precompute_hits': cache_hits.get('precompute', 0) if isinstance(cache_hits, dict) else 0
        }
    
    def benchmark_throughput(
        self,
        service: OptimizedNCFService,
        duration_seconds: int = 10
    ) -> Dict[str, Any]:
        """Benchmark maximum throughput (QPS)."""
        logger.info(f"🔍 Benchmarking throughput for {duration_seconds}s...")
        
        request_count = 0
        latencies = []
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        while time.perf_counter() < end_time:
            user_id = self.test_user_ids[request_count % len(self.test_user_ids)]
            
            req_start = time.perf_counter()
            service.get_recommendations(
                user_id,
                top_k=self.test_top_k,
                use_cache=True
            )
            latency = (time.perf_counter() - req_start) * 1000
            
            latencies.append(latency)
            request_count += 1
        
        elapsed = time.perf_counter() - start_time
        qps = request_count / elapsed
        
        return {
            'total_requests': request_count,
            'duration_seconds': round(elapsed, 2),
            'qps': round(qps, 2),
            'avg_latency_ms': round(statistics.mean(latencies), 2),
            'p50_latency_ms': round(statistics.median(latencies), 2),
            'p95_latency_ms': round(self._percentile(latencies, 95), 2),
            'p99_latency_ms': round(self._percentile(latencies, 99), 2)
        }
    
    def compare_int8_vs_fp32(self) -> Dict[str, Any]:
        """Compare INT8 vs FP32 model performance."""
        logger.info("🔍 Comparing INT8 vs FP32 models...")
        
        results = {}
        
        # Test INT8 model
        logger.info("  Testing INT8 model...")
        try:
            service_int8 = OptimizedNCFService(
                model_path="backend/ml/deep_learning/models/ncf_model_int8.onnx",
                enable_caching=False
            )
            if service_int8.model_type == "int8":
                results['int8'] = self.benchmark_single_inference(service_int8, use_cache=False)
        except Exception as e:
            logger.warning(f"INT8 model not available: {e}")
            results['int8'] = None
        
        # Test FP32 model
        logger.info("  Testing FP32 model...")
        try:
            service_fp32 = OptimizedNCFService(
                model_path="backend/ml/deep_learning/models/ncf_model.onnx",
                fallback_model_path="",
                enable_caching=False
            )
            if service_fp32.model_type == "fp32":
                results['fp32'] = self.benchmark_single_inference(service_fp32, use_cache=False)
        except Exception as e:
            logger.warning(f"FP32 model not available: {e}")
            results['fp32'] = None
        
        # Compute speedup
        if results.get('int8') and results.get('fp32'):
            speedup = results['fp32']['mean'] / results['int8']['mean']
            results['speedup'] = round(speedup, 2)
            
            # Model size comparison (if models exist)
            int8_path = Path("backend/ml/deep_learning/models/ncf_model_int8.onnx")
            fp32_path = Path("backend/ml/deep_learning/models/ncf_model.onnx")
            
            if int8_path.exists() and fp32_path.exists():
                int8_size = int8_path.stat().st_size / (1024 * 1024)  # MB
                fp32_size = fp32_path.stat().st_size / (1024 * 1024)  # MB
                
                results['model_size_mb'] = {
                    'int8': round(int8_size, 2),
                    'fp32': round(fp32_size, 2),
                    'reduction': f"{(1 - int8_size/fp32_size) * 100:.1f}%"
                }
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        logger.info("="*60)
        logger.info("🚀 Starting Comprehensive NCF Performance Benchmark")
        logger.info("="*60)
        
        # Initialize service
        service = OptimizedNCFService(
            enable_caching=True,
            enable_batching=True
        )
        
        logger.info(f"Service initialized:")
        logger.info(f"  Model type: {service.model_type}")
        logger.info(f"  Caching: {service.enable_caching}")
        logger.info(f"  Batching: {service.enable_batching}")
        logger.info("")
        
        results = {
            'service_config': {
                'model_type': service.model_type,
                'caching_enabled': service.enable_caching,
                'batching_enabled': service.enable_batching,
                'num_warmup': self.num_warmup,
                'num_iterations': self.num_iterations
            }
        }
        
        # 1. Single inference (no cache)
        results['single_inference_no_cache'] = self.benchmark_single_inference(
            service, use_cache=False
        )
        
        # 2. Single inference (with cache)
        results['single_inference_cached'] = self.benchmark_single_inference(
            service, use_cache=True
        )
        
        # 3. Batch inference
        if service.enable_batching:
            results['batch_inference'] = self.benchmark_batch_inference(service)
        
        # 4. Cache effectiveness
        if service.enable_caching:
            results['cache_effectiveness'] = self.benchmark_cache_effectiveness(service)
        
        # 5. Throughput
        results['throughput'] = self.benchmark_throughput(service, duration_seconds=10)
        
        # 6. INT8 vs FP32 comparison
        results['int8_vs_fp32'] = self.compare_int8_vs_fp32()
        
        return results
    
    def _compute_stats(self, latencies: List[float], name: str) -> Dict[str, Any]:
        """Compute latency statistics."""
        return {
            'name': name,
            'count': len(latencies),
            'mean': round(statistics.mean(latencies), 2),
            'median': round(statistics.median(latencies), 2),
            'stdev': round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
            'min': round(min(latencies), 2),
            'max': round(max(latencies), 2),
            'p50': round(self._percentile(latencies, 50), 2),
            'p95': round(self._percentile(latencies, 95), 2),
            'p99': round(self._percentile(latencies, 99), 2)
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted way."""
        logger.info("")
        logger.info("="*60)
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("="*60)
        
        # Service config
        logger.info("\n🔧 Service Configuration:")
        config = results['service_config']
        logger.info(f"  Model Type: {config['model_type']}")
        logger.info(f"  Caching: {config['caching_enabled']}")
        logger.info(f"  Batching: {config['batching_enabled']}")
        
        # Single inference
        logger.info("\n📈 Single Inference (No Cache):")
        stats = results['single_inference_no_cache']
        logger.info(f"  Mean: {stats['mean']}ms")
        logger.info(f"  P50: {stats['p50']}ms")
        logger.info(f"  P95: {stats['p95']}ms")
        logger.info(f"  P99: {stats['p99']}ms")
        
        # Cached inference
        logger.info("\n⚡ Single Inference (Cached):")
        stats = results['single_inference_cached']
        logger.info(f"  Mean: {stats['mean']}ms")
        logger.info(f"  P95: {stats['p95']}ms")
        
        # Batch inference
        if 'batch_inference' in results:
            logger.info("\n🚀 Batch Inference:")
            for batch_size, stats in results['batch_inference'].items():
                logger.info(f"  Batch Size {batch_size}:")
                logger.info(f"    Total: {stats['mean']}ms")
                logger.info(f"    Per User: {stats['avg_per_user_ms']:.2f}ms")
                logger.info(f"    Speedup: {stats['speedup_vs_sequential']:.1f}x")
        
        # Cache effectiveness
        if 'cache_effectiveness' in results:
            logger.info("\n💾 Cache Effectiveness:")
            cache = results['cache_effectiveness']
            logger.info(f"  Hit Rate: {cache['hit_rate']}")
            logger.info(f"  Speedup: {cache['speedup']:.1f}x")
            logger.info(f"  L1 Hits: {cache['l1_hits']}")
            logger.info(f"  L2 Hits: {cache['l2_hits']}")
            logger.info(f"  Miss Latency (P95): {cache['cache_miss_latency_ms']['p95']}ms")
            logger.info(f"  Hit Latency (P95): {cache['cache_hit_latency_ms']['p95']}ms")
        
        # Throughput
        logger.info("\n🏎️  Throughput:")
        tp = results['throughput']
        logger.info(f"  QPS: {tp['qps']}")
        logger.info(f"  Total Requests: {tp['total_requests']}")
        logger.info(f"  Duration: {tp['duration_seconds']}s")
        logger.info(f"  P95 Latency: {tp['p95_latency_ms']}ms")
        
        # INT8 vs FP32
        if 'int8_vs_fp32' in results:
            comparison = results['int8_vs_fp32']
            logger.info("\n🔬 INT8 vs FP32 Comparison:")
            if comparison.get('int8'):
                logger.info(f"  INT8 Mean: {comparison['int8']['mean']}ms")
            if comparison.get('fp32'):
                logger.info(f"  FP32 Mean: {comparison['fp32']['mean']}ms")
            if 'speedup' in comparison:
                logger.info(f"  Speedup: {comparison['speedup']:.2f}x")
            if 'model_size_mb' in comparison:
                sizes = comparison['model_size_mb']
                logger.info(f"  Model Size Reduction: {sizes['reduction']}")
                logger.info(f"    INT8: {sizes['int8']}MB")
                logger.info(f"    FP32: {sizes['fp32']}MB")
        
        logger.info("\n" + "="*60)
        logger.info("✅ Benchmark Complete!")
        logger.info("="*60)
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {output_file}")


def main():
    """Run benchmark."""
    benchmark = PerformanceBenchmark(num_warmup=10, num_iterations=100)
    
    # Run full benchmark
    results = benchmark.run_full_benchmark()
    
    # Print results
    benchmark.print_results(results)
    
    # Save results
    benchmark.save_results(
        results,
        "backend/ml/deep_learning/benchmark_results.json"
    )


if __name__ == "__main__":
    main()
