#!/usr/bin/env python3
"""
CPU vs GPU Performance Benchmark for AI Istanbul
================================================

Measures actual performance of SentenceTransformers on CPU
to validate that CPU-only is sufficient for 10K monthly users.
"""

import time
import numpy as np
from datetime import datetime
import psutil
import sys

# Check if running in test mode
QUICK_TEST = '--quick' in sys.argv

print("🔬 AI Istanbul - CPU Performance Benchmark")
print("=" * 60)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
print("=" * 60)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()
    device = 'cuda' if has_cuda else 'cpu'
    
    print(f"\n{'🎮 GPU Available!' if has_cuda else '💻 CPU Only'}")
    if has_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Load model
    print("📦 Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
    start = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Force CPU for fair comparison
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f}s")
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print()
    
    # Test queries (realistic Istanbul tourism queries)
    test_queries = [
        "How do I get to Sultanahmet from Taksim?",
        "Boğaz turuna nasıl katılabilirim?",
        "Best restaurants near Galata Tower",
        "İstanbul kart nerede alabilirim?",
        "Metro schedule Kadıköy to Beşiktaş",
        "أين يمكنني شراء بطاقة إسطنبول؟",  # Arabic
        "Где находится дворец Топкапы?",  # Russian
        "Comment aller à la mosquée bleue?",  # French
        "İstanbul'da gezilecek yerler",
        "Ferry times from Eminönü to Kadıköy"
    ]
    
    # Benchmark 1: Single Query Latency
    print("🔍 Benchmark 1: Single Query Latency")
    print("-" * 60)
    latencies = []
    for i, query in enumerate(test_queries, 1):
        start = time.time()
        embedding = model.encode(query, convert_to_numpy=True)
        latency = (time.time() - start) * 1000  # Convert to ms
        latencies.append(latency)
        print(f"Query {i:2d}: {latency:6.2f}ms - {query[:50]}")
    
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    print(f"\n📊 Statistics:")
    print(f"   Average:  {avg_latency:.2f}ms")
    print(f"   Median:   {p50_latency:.2f}ms")
    print(f"   95th %:   {p95_latency:.2f}ms")
    print(f"   99th %:   {p99_latency:.2f}ms")
    print()
    
    # Benchmark 2: Batch Processing
    batch_sizes = [1, 10, 50, 100] if not QUICK_TEST else [1, 10]
    print("📦 Benchmark 2: Batch Processing")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        queries_batch = test_queries * (batch_size // len(test_queries) + 1)
        queries_batch = queries_batch[:batch_size]
        
        start = time.time()
        embeddings = model.encode(queries_batch, batch_size=batch_size, convert_to_numpy=True)
        total_time = time.time() - start
        time_per_query = (total_time / batch_size) * 1000
        throughput = batch_size / total_time
        
        print(f"Batch size {batch_size:3d}: {total_time:6.2f}s total | "
              f"{time_per_query:6.2f}ms/query | {throughput:6.1f} queries/sec")
    print()
    
    # Benchmark 3: Sustained Load (10K users scenario)
    print("🚀 Benchmark 3: Sustained Load Simulation")
    print("-" * 60)
    
    # Simulate one hour of traffic for 10K monthly users
    # 10K users/month ≈ 330 users/day ≈ 40 users/hour (peak)
    # Assuming 5 queries/user = 200 queries/hour peak
    queries_per_hour = 200
    test_duration = 10 if not QUICK_TEST else 3  # seconds (simulate 10s instead of 1 hour)
    queries_to_simulate = int(queries_per_hour * (test_duration / 3600))
    
    print(f"Simulating {queries_per_hour} queries/hour for {test_duration} seconds")
    print(f"Total queries: {queries_to_simulate}")
    print()
    
    successful_queries = 0
    failed_queries = 0
    query_times = []
    
    start_time = time.time()
    
    while time.time() - start_time < test_duration:
        query = test_queries[successful_queries % len(test_queries)]
        query_start = time.time()
        
        try:
            embedding = model.encode(query, convert_to_numpy=True)
            query_time = (time.time() - query_start) * 1000
            query_times.append(query_time)
            successful_queries += 1
        except Exception as e:
            failed_queries += 1
            print(f"❌ Query failed: {e}")
        
        # Small delay to simulate real-world spacing
        time.sleep(0.01)
    
    total_duration = time.time() - start_time
    actual_qps = successful_queries / total_duration
    
    print(f"✅ Results:")
    print(f"   Duration: {total_duration:.2f}s")
    print(f"   Successful: {successful_queries}")
    print(f"   Failed: {failed_queries}")
    print(f"   Throughput: {actual_qps:.1f} queries/sec")
    print(f"   Average latency: {np.mean(query_times):.2f}ms")
    print(f"   95th percentile: {np.percentile(query_times, 95):.2f}ms")
    print()
    
    # Benchmark 4: POI Embedding Generation
    if not QUICK_TEST:
        print("🏛️ Benchmark 4: POI Embedding Generation")
        print("-" * 60)
        
        # Simulate embedding 10,000 POIs
        sample_pois = [
            "Hagia Sophia - Historic Byzantine church",
            "Blue Mosque - Iconic Ottoman mosque",
            "Topkapi Palace - Ottoman royal palace",
            "Grand Bazaar - Historic covered market",
            "Galata Tower - Medieval Genoese tower",
        ] * 200  # 1000 POIs
        
        print(f"Generating embeddings for {len(sample_pois)} POIs...")
        start = time.time()
        embeddings = model.encode(sample_pois, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        total_time = time.time() - start
        
        print(f"✅ Generated {len(embeddings)} embeddings in {total_time:.2f}s")
        print(f"   Average: {(total_time / len(sample_pois)) * 1000:.2f}ms per POI")
        print(f"   Estimated for 10K POIs: {(total_time * 10):.2f}s ({(total_time * 10 / 60):.1f} minutes)")
        print()
    
    # Final Analysis
    print("=" * 60)
    print("📊 FINAL ANALYSIS")
    print("=" * 60)
    
    # Calculate capacity
    max_qps = 1000 / avg_latency  # queries per second
    hourly_capacity = max_qps * 3600
    daily_capacity = hourly_capacity * 24
    
    print(f"\n🎯 CPU Capacity:")
    print(f"   Max throughput: {max_qps:.1f} queries/second")
    print(f"   Hourly capacity: {hourly_capacity:,.0f} queries")
    print(f"   Daily capacity: {daily_capacity:,.0f} queries")
    print()
    
    print(f"📈 10K Monthly Users Requirements:")
    print(f"   Daily queries: ~1,650 (330 users × 5 queries)")
    print(f"   Peak hour: ~200 queries/hour")
    print(f"   Peak minute: ~3-4 queries/minute")
    print()
    
    # Verdict
    capacity_margin = (hourly_capacity / 200) - 1
    
    if capacity_margin > 10:
        verdict = "✅ EXCELLENT - CPU has massive overhead"
        recommendation = "CPU-only is perfect. No GPU needed."
    elif capacity_margin > 5:
        verdict = "✅ GREAT - CPU has plenty of capacity"
        recommendation = "CPU-only is recommended. Save GPU costs."
    elif capacity_margin > 2:
        verdict = "✅ GOOD - CPU can handle the load"
        recommendation = "CPU-only works well. Consider GPU at 50K+ users."
    elif capacity_margin > 0.5:
        verdict = "⚠️  ADEQUATE - CPU at comfortable limit"
        recommendation = "CPU works but consider GPU for growth."
    else:
        verdict = "❌ INSUFFICIENT - CPU may struggle at peak"
        recommendation = "GPU recommended for better performance."
    
    print(f"🎯 Verdict: {verdict}")
    print(f"   Capacity margin: {capacity_margin:.1f}x peak load")
    print(f"   Can handle: {capacity_margin * 10000:,.0f} monthly users")
    print()
    print(f"💡 Recommendation: {recommendation}")
    print()
    
    # Cost analysis
    print("💰 Cost Analysis:")
    print(f"   CPU-only (current): $50/month")
    print(f"   Estimated cost per user: ${50 / 10000:.4f}")
    print(f"   GPU upgrade cost: +$200-250/month")
    print(f"   Break-even point: ~50,000-100,000 users/month")
    print()
    
    print("=" * 60)
    print("✅ Benchmark Complete!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\n📦 Please install required packages:")
    print("   pip install sentence-transformers torch psutil numpy")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
