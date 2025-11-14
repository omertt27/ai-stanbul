#!/usr/bin/env python3
"""
Semantic Cache Monitoring Script

Monitor and analyze semantic cache performance metrics.
Provides real-time statistics and insights.

Usage:
    python monitor_semantic_cache.py [--interval SECONDS] [--json]
"""

import asyncio
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    import redis
except ImportError:
    print("❌ Error: redis library not installed")
    print("Install with: pip install redis")
    sys.exit(1)


class SemanticCacheMonitor:
    """Monitor semantic cache performance and statistics."""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379, redis_db: int = 0):
        """Initialize monitor."""
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self.cache_prefix = "semantic_cache:"
        self.stats_history = []
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics."""
        try:
            # Get all semantic cache keys
            keys = self.redis_client.keys(f"{self.cache_prefix}*")
            
            # Calculate statistics
            total_entries = len(keys)
            
            # Get Redis stats
            redis_info = self.redis_client.info('stats')
            keyspace_hits = redis_info.get('keyspace_hits', 0)
            keyspace_misses = redis_info.get('keyspace_misses', 0)
            
            # Calculate hit rate
            total_ops = keyspace_hits + keyspace_misses
            hit_rate = (keyspace_hits / total_ops * 100) if total_ops > 0 else 0
            
            # Get memory usage
            memory_info = self.redis_client.info('memory')
            used_memory = memory_info.get('used_memory_human', 'N/A')
            
            # Sample cache entries for analysis
            sample_keys = keys[:10] if len(keys) > 0 else []
            sample_data = []
            
            for key in sample_keys:
                try:
                    data = json.loads(self.redis_client.get(key))
                    sample_data.append({
                        'query': data.get('query', '')[:50],
                        'timestamp': data.get('timestamp', 0),
                        'age_hours': (time.time() - data.get('timestamp', time.time())) / 3600
                    })
                except:
                    pass
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_entries': total_entries,
                'keyspace_hits': keyspace_hits,
                'keyspace_misses': keyspace_misses,
                'hit_rate': round(hit_rate, 2),
                'used_memory': used_memory,
                'sample_entries': sample_data,
                'redis_connected': True
            }
            
        except redis.ConnectionError:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': 'Redis connection failed',
                'redis_connected': False
            }
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'redis_connected': False
            }
    
    def print_stats(self, stats: Dict[str, Any], json_output: bool = False):
        """Print statistics to console."""
        if json_output:
            print(json.dumps(stats, indent=2))
            return
        
        if not stats.get('redis_connected'):
            print(f"❌ {stats.get('error', 'Unknown error')}")
            return
        
        # Clear screen (optional)
        # print("\033[2J\033[H")
        
        print("=" * 60)
        print(f"🧠 Semantic Cache Monitor - {stats['timestamp']}")
        print("=" * 60)
        print()
        
        # Cache statistics
        print("📊 Cache Statistics:")
        print(f"  Total Entries:    {stats['total_entries']:,}")
        print(f"  Cache Hit Rate:   {stats['hit_rate']:.2f}%")
        print(f"  Keyspace Hits:    {stats['keyspace_hits']:,}")
        print(f"  Keyspace Misses:  {stats['keyspace_misses']:,}")
        print(f"  Memory Used:      {stats['used_memory']}")
        print()
        
        # Performance assessment
        hit_rate = stats['hit_rate']
        if hit_rate >= 40:
            status = "✅ EXCELLENT"
            color = "\033[92m"  # Green
        elif hit_rate >= 30:
            status = "✓ GOOD"
            color = "\033[93m"  # Yellow
        elif hit_rate >= 20:
            status = "⚠ FAIR"
            color = "\033[93m"  # Yellow
        else:
            status = "⚠ LOW"
            color = "\033[91m"  # Red
        
        reset = "\033[0m"
        print(f"Performance: {color}{status}{reset} (Target: >40%)")
        print()
        
        # Sample entries
        if stats['sample_entries']:
            print("📝 Sample Cache Entries:")
            for i, entry in enumerate(stats['sample_entries'][:5], 1):
                age = entry['age_hours']
                age_str = f"{age:.1f}h" if age < 24 else f"{age/24:.1f}d"
                print(f"  {i}. [{age_str}] {entry['query']}...")
            print()
        
        # Recommendations
        print("💡 Recommendations:")
        if hit_rate < 20:
            print("  ⚠ Low hit rate - Consider:")
            print("    • Decreasing similarity threshold (current: 0.85)")
            print("    • Increasing top-K candidates")
            print("    • Analyzing query diversity")
        elif hit_rate > 70:
            print("  ⚠ Very high hit rate - Verify:")
            print("    • Response quality is maintained")
            print("    • Queries aren't too repetitive")
        else:
            print("  ✓ Performance within expected range")
        
        print()
    
    def monitor_continuous(self, interval: int = 10, json_output: bool = False):
        """Monitor cache continuously."""
        print(f"Starting continuous monitoring (interval: {interval}s)")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            while True:
                stats = self.get_cache_stats()
                self.print_stats(stats, json_output)
                
                # Store in history
                self.stats_history.append(stats)
                
                # Keep only last 100 entries
                if len(self.stats_history) > 100:
                    self.stats_history = self.stats_history[-100:]
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n⏹ Monitoring stopped")
            self.print_summary()
    
    def print_summary(self):
        """Print summary of monitoring session."""
        if len(self.stats_history) < 2:
            return
        
        print("\n" + "=" * 60)
        print("📈 Monitoring Summary")
        print("=" * 60)
        
        # Calculate trends
        first = self.stats_history[0]
        last = self.stats_history[-1]
        
        entries_growth = last.get('total_entries', 0) - first.get('total_entries', 0)
        hit_rate_change = last.get('hit_rate', 0) - first.get('hit_rate', 0)
        
        print(f"Duration:           {len(self.stats_history)} samples")
        print(f"Cache Growth:       {entries_growth:+,} entries")
        print(f"Hit Rate Change:    {hit_rate_change:+.2f}%")
        print(f"Final Hit Rate:     {last.get('hit_rate', 0):.2f}%")
        print(f"Total Entries:      {last.get('total_entries', 0):,}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Monitor semantic cache performance')
    parser.add_argument('--interval', '-i', type=int, default=10,
                       help='Monitoring interval in seconds (default: 10)')
    parser.add_argument('--json', action='store_true',
                       help='Output in JSON format')
    parser.add_argument('--redis-host', default='localhost',
                       help='Redis host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--redis-db', type=int, default=0,
                       help='Redis database (default: 0)')
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SemanticCacheMonitor(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_db=args.redis_db
    )
    
    # Run monitoring
    if args.once:
        stats = monitor.get_cache_stats()
        monitor.print_stats(stats, args.json)
    else:
        monitor.monitor_continuous(interval=args.interval, json_output=args.json)


if __name__ == '__main__':
    main()
