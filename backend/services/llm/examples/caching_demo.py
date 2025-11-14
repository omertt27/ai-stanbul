"""
Caching demonstration for Pure LLM Handler.

This script demonstrates the dual-layer caching system and its benefits.
"""
import asyncio
import time
from backend.services.llm.core import PureLLMHandler


async def main():
    """Run caching demonstration."""
    config = {
        'openai': {
            'api_key': 'your-api-key-here',
            'model': 'gpt-3.5-turbo'
        },
        'caching': {
            'enable_cache': True,
            'l1_ttl': 300,  # 5 minutes
            'l2_ttl': 3600,  # 1 hour
            'l1_max_size': 100,
            'l2_max_size': 1000
        }
    }
    
    handler = PureLLMHandler(config)
    
    print("=" * 60)
    print("Pure LLM Handler - Caching Demonstration")
    print("=" * 60)
    
    # Popular query that will be cached
    popular_query = "What are the best restaurants in Sultanahmet?"
    context = {'user_id': 'cache_demo_user'}
    
    print("\n1. First Request (Cache Miss)")
    print("-" * 60)
    start_time = time.time()
    response1 = await handler.process_query(popular_query, context)
    elapsed1 = time.time() - start_time
    
    print(f"Query: {popular_query}")
    print(f"Cache Hit: {response1.get('cache_hit', False)}")
    print(f"Cache Level: {response1.get('cache_level', 'none')}")
    print(f"Response Time: {elapsed1:.3f}s")
    print(f"Response: {response1['response'][:100]}...")
    
    # Same query - should hit L1 cache
    print("\n2. Second Request (L1 Cache Hit)")
    print("-" * 60)
    start_time = time.time()
    response2 = await handler.process_query(popular_query, context)
    elapsed2 = time.time() - start_time
    
    print(f"Query: {popular_query}")
    print(f"Cache Hit: {response2.get('cache_hit', False)}")
    print(f"Cache Level: {response2.get('cache_level', 'none')}")
    print(f"Response Time: {elapsed2:.3f}s")
    print(f"Speedup: {elapsed1 / elapsed2:.1f}x faster")
    
    # Fill L1 cache with other queries
    print("\n3. Filling Cache with Multiple Queries")
    print("-" * 60)
    
    other_queries = [
        "Best hotels in Taksim?",
        "Things to do in Kadikoy?",
        "How to get to Galata Tower?",
        "Turkish breakfast places near me?",
        "Shopping malls in Istanbul?"
    ]
    
    for query in other_queries:
        await handler.process_query(query, context)
        print(f"✓ Cached: {query}")
    
    # Original query again - might be in L2 if L1 evicted it
    print("\n4. Third Request (Potential L2 Cache Hit)")
    print("-" * 60)
    start_time = time.time()
    response3 = await handler.process_query(popular_query, context)
    elapsed3 = time.time() - start_time
    
    print(f"Query: {popular_query}")
    print(f"Cache Hit: {response3.get('cache_hit', False)}")
    print(f"Cache Level: {response3.get('cache_level', 'none')}")
    print(f"Response Time: {elapsed3:.3f}s")
    
    # Cache statistics
    print("\n5. Cache Statistics")
    print("-" * 60)
    
    cache_stats = handler.get_cache_stats()
    if cache_stats:
        print(f"Total Requests: {cache_stats.get('total_requests', 0)}")
        print(f"Cache Hits: {cache_stats.get('cache_hits', 0)}")
        print(f"Cache Misses: {cache_stats.get('cache_misses', 0)}")
        print(f"Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
        print(f"L1 Size: {cache_stats.get('l1_size', 0)} / {config['caching']['l1_max_size']}")
        print(f"L2 Size: {cache_stats.get('l2_size', 0)} / {config['caching']['l2_max_size']}")
        print(f"Avg Lookup Time: {cache_stats.get('avg_lookup_time', 0):.3f}s")
    
    # Demonstrate cache invalidation
    print("\n6. Cache Invalidation")
    print("-" * 60)
    
    print("Invalidating cache for Sultanahmet restaurants...")
    handler.invalidate_cache(pattern="sultanahmet*restaurant*")
    print("✓ Cache invalidated")
    
    start_time = time.time()
    response4 = await handler.process_query(popular_query, context)
    elapsed4 = time.time() - start_time
    
    print(f"\nQuery: {popular_query}")
    print(f"Cache Hit: {response4.get('cache_hit', False)} (should be False)")
    print(f"Response Time: {elapsed4:.3f}s (should be slower)")
    
    # Demonstrate semantic similarity caching
    print("\n7. Semantic Similarity Caching")
    print("-" * 60)
    
    similar_queries = [
        "What are the best restaurants in Sultanahmet?",
        "Can you recommend good restaurants in Sultanahmet?",
        "Top rated restaurants in Sultanahmet area?",
        "Where to eat in Sultanahmet?"
    ]
    
    print("Testing semantically similar queries:")
    for query in similar_queries:
        start_time = time.time()
        response = await handler.process_query(query, context)
        elapsed = time.time() - start_time
        
        print(f"\nQuery: {query}")
        print(f"Cache Hit: {response.get('cache_hit', False)}")
        print(f"Similarity Score: {response.get('cache_similarity', 0):.2f}")
        print(f"Response Time: {elapsed:.3f}s")
    
    print("\n" + "=" * 60)
    print("Caching demonstration completed!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- L1 cache provides fastest lookups for recent queries")
    print("- L2 cache provides good performance for popular queries")
    print("- Semantic similarity can match similar queries")
    print("- Cache invalidation allows for fresh data when needed")


if __name__ == "__main__":
    asyncio.run(main())
