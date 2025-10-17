#!/usr/bin/env python3
"""
Cache Management & Warming System for AI Istanbul
Provides cache warming, maintenance, and optimization utilities
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheManager:
    """Comprehensive cache management system"""
    
    def __init__(self):
        self.ml_cache = None
        self.edge_cache = None
        self.multi_intent_handler = None
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize cache and handler systems"""
        try:
            from ml_result_cache import get_ml_cache
            from edge_cache_system import get_edge_cache
            from multi_intent_query_handler import MultiIntentQueryHandler
            
            self.ml_cache = get_ml_cache()
            self.edge_cache = get_edge_cache()
            self.multi_intent_handler = MultiIntentQueryHandler()
            
            logger.info("✅ Cache management systems initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize systems: {e}")
    
    def warm_ml_cache(self, queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Warm ML cache with common queries"""
        
        if not queries:
            # Common tourist queries for Istanbul
            queries = [
                "best restaurants in sultanahmet",
                "things to do near blue mosque",
                "route from taksim to galata tower",
                "museums in beyoglu",
                "turkish breakfast places",
                "romantic restaurants with bosphorus view",
                "nightlife in kadikoy",
                "shopping in grand bazaar area",
                "events this weekend in istanbul",
                "transportation from airport to city center",
                "neighborhoods for young professionals",
                "family-friendly activities",
                "budget hotels in sultanahmet",
                "vegetarian restaurants in istanbul",
                "art galleries in galata",
                "traditional turkish baths",
                "ferry routes to asian side",
                "rooftop bars with view",
                "halal restaurants near blue mosque",
                "cultural events in october"
            ]
        
        warming_results = {
            'started_at': datetime.now().isoformat(),
            'total_queries': len(queries),
            'successful_warming': 0,
            'failed_warming': 0,
            'cache_entries_added': 0,
            'processing_time': 0.0,
            'warmed_queries': []
        }
        
        start_time = time.time()
        
        print(f"🔥 Starting cache warming with {len(queries)} queries...")
        
        for i, query in enumerate(queries, 1):
            try:
                print(f"  {i:2d}. Warming: '{query[:50]}...'")
                
                # Check if already cached
                existing_cache = self.ml_cache.get(query, {}, ["general"])
                if existing_cache:
                    print(f"      ✅ Already cached")
                    continue
                
                # Process query to warm cache
                result = self.multi_intent_handler.analyze_query(query, {
                    'cache_warming': True,
                    'user_type': 'tourist'
                })
                
                if result and hasattr(result, 'ml_enhanced') and result.ml_enhanced:
                    warming_results['successful_warming'] += 1
                    warming_results['cache_entries_added'] += len(result.ml_enhancements) if hasattr(result, 'ml_enhancements') else 1
                    print(f"      🔥 Warmed successfully")
                else:
                    warming_results['successful_warming'] += 1  # Still count as success
                    print(f"      ⚡ Processed (no ML)")
                
                warming_results['warmed_queries'].append({
                    'query': query,
                    'success': True,
                    'ml_enhanced': hasattr(result, 'ml_enhanced') and result.ml_enhanced
                })
                
                # Small delay to avoid overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                warming_results['failed_warming'] += 1
                warming_results['warmed_queries'].append({
                    'query': query,
                    'success': False,
                    'error': str(e)
                })
                print(f"      ❌ Failed: {e}")
        
        warming_results['processing_time'] = time.time() - start_time
        warming_results['completed_at'] = datetime.now().isoformat()
        
        print(f"\n📊 Cache Warming Results:")
        print(f"  • Total queries: {warming_results['total_queries']}")
        print(f"  • Successful: {warming_results['successful_warming']}")
        print(f"  • Failed: {warming_results['failed_warming']}")
        print(f"  • Cache entries added: {warming_results['cache_entries_added']}")
        print(f"  • Processing time: {warming_results['processing_time']:.2f}s")
        print(f"  • Final cache size: {len(self.ml_cache.memory_cache)} entries")
        
        return warming_results
    
    def warm_edge_cache(self) -> Dict[str, Any]:
        """Warm edge cache with static data"""
        
        warming_results = {
            'started_at': datetime.now().isoformat(),
            'attractions_cached': False,
            'events_cached': False,
            'cache_entries_before': len(self.edge_cache.cache_entries),
            'cache_entries_after': 0,
            'errors': []
        }
        
        print("🌐 Starting edge cache warming...")
        
        # Cache attractions data
        try:
            print("  📍 Caching attractions data...")
            attractions_key = self.edge_cache.cache_attractions_data()
            if attractions_key:
                warming_results['attractions_cached'] = True
                print("      ✅ Attractions cached successfully")
            else:
                warming_results['errors'].append("Failed to cache attractions")
                print("      ❌ Failed to cache attractions")
        except Exception as e:
            warming_results['errors'].append(f"Attractions error: {e}")
            print(f"      ❌ Attractions error: {e}")
        
        # Cache events data
        try:
            print("  🎭 Caching events data...")
            events_key = self.edge_cache.cache_events_data()
            if events_key:
                warming_results['events_cached'] = True
                print("      ✅ Events cached successfully")
            else:
                warming_results['errors'].append("Failed to cache events")
                print("      ❌ Failed to cache events")
        except Exception as e:
            warming_results['errors'].append(f"Events error: {e}")
            print(f"      ❌ Events error: {e}")
        
        warming_results['cache_entries_after'] = len(self.edge_cache.cache_entries)
        warming_results['completed_at'] = datetime.now().isoformat()
        
        print(f"\n📊 Edge Cache Warming Results:")
        print(f"  • Attractions cached: {'✅' if warming_results['attractions_cached'] else '❌'}")
        print(f"  • Events cached: {'✅' if warming_results['events_cached'] else '❌'}")
        print(f"  • Cache entries: {warming_results['cache_entries_before']} → {warming_results['cache_entries_after']}")
        print(f"  • Errors: {len(warming_results['errors'])}")
        
        return warming_results
    
    def cleanup_expired_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries"""
        
        cleanup_results = {
            'started_at': datetime.now().isoformat(),
            'ml_cache_before': len(self.ml_cache.memory_cache),
            'ml_cache_after': 0,
            'edge_cache_before': len(self.edge_cache.cache_entries),
            'edge_cache_after': 0,
            'expired_entries_removed': 0
        }
        
        print("🧹 Starting cache cleanup...")
        
        # Clean ML cache
        print("  🧠 Cleaning ML cache...")
        current_time = datetime.now()
        expired_keys = []
        
        for key, entry in self.ml_cache.memory_cache.items():
            if entry.expires_at < current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.ml_cache.memory_cache[key]
            cleanup_results['expired_entries_removed'] += 1
        
        cleanup_results['ml_cache_after'] = len(self.ml_cache.memory_cache)
        
        # Clean edge cache (entries have their own expiration logic)
        print("  🌐 Checking edge cache...")
        expired_edge_keys = []
        
        for key, entry in self.edge_cache.cache_entries.items():
            if entry.expires_at < current_time:
                expired_edge_keys.append(key)
        
        for key in expired_edge_keys:
            del self.edge_cache.cache_entries[key]
            cleanup_results['expired_entries_removed'] += 1
        
        cleanup_results['edge_cache_after'] = len(self.edge_cache.cache_entries)
        cleanup_results['completed_at'] = datetime.now().isoformat()
        
        print(f"\n📊 Cache Cleanup Results:")
        print(f"  • ML cache: {cleanup_results['ml_cache_before']} → {cleanup_results['ml_cache_after']} entries")
        print(f"  • Edge cache: {cleanup_results['edge_cache_before']} → {cleanup_results['edge_cache_after']} entries")
        print(f"  • Expired entries removed: {cleanup_results['expired_entries_removed']}")
        
        return cleanup_results
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'ml_cache': {
                'total_entries': len(self.ml_cache.memory_cache),
                'memory_usage_mb': round(len(self.ml_cache.memory_cache) * 2.0 / 1024, 2),
                'entries_by_system': {},
                'avg_confidence': 0.0,
                'hit_counts': {}
            },
            'edge_cache': {
                'total_entries': len(self.edge_cache.cache_entries),
                'entries_by_type': {},
                'compression_enabled': self.edge_cache.enable_compression,
                'total_access_count': 0
            }
        }
        
        # Analyze ML cache entries
        total_confidence = 0
        confidence_count = 0
        
        for entry in self.ml_cache.memory_cache.values():
            total_confidence += entry.confidence_score
            confidence_count += 1
            
            for system in entry.enhancement_systems:
                stats['ml_cache']['entries_by_system'][system] = \
                    stats['ml_cache']['entries_by_system'].get(system, 0) + 1
            
            stats['ml_cache']['hit_counts'][entry.query_hash] = entry.hit_count
        
        if confidence_count > 0:
            stats['ml_cache']['avg_confidence'] = round(total_confidence / confidence_count, 3)
        
        # Analyze edge cache entries
        for entry in self.edge_cache.cache_entries.values():
            content_type = getattr(entry, 'content_type', 'unknown')
            stats['edge_cache']['entries_by_type'][content_type] = \
                stats['edge_cache']['entries_by_type'].get(content_type, 0) + 1
            
            stats['edge_cache']['total_access_count'] += entry.access_count
        
        return stats
    
    def optimize_cache_settings(self) -> Dict[str, Any]:
        """Analyze and optimize cache settings"""
        
        optimization = {
            'timestamp': datetime.now().isoformat(),
            'current_settings': {
                'ml_cache_max_size': self.ml_cache.max_cache_size,
                'edge_cache_compression': self.edge_cache.enable_compression
            },
            'recommendations': [],
            'optimizations_applied': []
        }
        
        stats = self.get_cache_statistics()
        
        # Analyze ML cache optimization opportunities
        ml_entries = stats['ml_cache']['total_entries']
        ml_max_size = self.ml_cache.max_cache_size
        
        if ml_entries > ml_max_size * 0.8:
            optimization['recommendations'].append(
                f"ML cache is {ml_entries/ml_max_size*100:.1f}% full - consider increasing max size"
            )
        
        if stats['ml_cache']['avg_confidence'] < 0.7:
            optimization['recommendations'].append(
                "Average ML confidence is low - review cache quality threshold"
            )
        
        # Analyze edge cache optimization
        if not self.edge_cache.enable_compression:
            optimization['recommendations'].append(
                "Enable compression for edge cache to reduce bandwidth"
            )
        
        if stats['edge_cache']['total_entries'] < 5:
            optimization['recommendations'].append(
                "Edge cache has few entries - implement more static data caching"
            )
        
        # General recommendations
        optimization['recommendations'].extend([
            "Implement cache warming scheduler for peak hours",
            "Add monitoring alerts for cache hit ratio drops",
            "Consider implementing distributed caching for high availability"
        ])
        
        return optimization

def main():
    """Run comprehensive cache management operations"""
    
    print("🚀 AI Istanbul Cache Management System")
    print("=" * 50)
    
    cache_manager = CacheManager()
    
    if not cache_manager.ml_cache or not cache_manager.edge_cache:
        print("❌ Cache systems not available")
        return
    
    # Show initial cache status
    print("\n📊 Initial Cache Status:")
    initial_stats = cache_manager.get_cache_statistics()
    print(f"  • ML Cache: {initial_stats['ml_cache']['total_entries']} entries")
    print(f"  • Edge Cache: {initial_stats['edge_cache']['total_entries']} entries")
    
    # Warm ML cache
    print(f"\n{'='*20} ML CACHE WARMING {'='*20}")
    ml_warming = cache_manager.warm_ml_cache()
    
    # Warm edge cache
    print(f"\n{'='*20} EDGE CACHE WARMING {'='*20}")
    edge_warming = cache_manager.warm_edge_cache()
    
    # Show final cache status
    print(f"\n{'='*20} FINAL STATUS {'='*20}")
    final_stats = cache_manager.get_cache_statistics()
    print(f"📊 Final Cache Statistics:")
    print(f"  • ML Cache: {final_stats['ml_cache']['total_entries']} entries")
    print(f"  • Average ML Confidence: {final_stats['ml_cache']['avg_confidence']}")
    print(f"  • Edge Cache: {final_stats['edge_cache']['total_entries']} entries")
    print(f"  • Memory Usage: {final_stats['ml_cache']['memory_usage_mb']} MB")
    
    # Get optimization recommendations
    print(f"\n💡 Optimization Recommendations:")
    optimization = cache_manager.optimize_cache_settings()
    for i, rec in enumerate(optimization['recommendations'][:5], 1):
        print(f"  {i}. {rec}")
    
    # Cache performance summary
    total_warmed = ml_warming['successful_warming']
    total_errors = ml_warming['failed_warming'] + len(edge_warming['errors'])
    
    print(f"\n🎯 Cache Management Summary:")
    print(f"  • Queries warmed: {total_warmed}")
    print(f"  • ML cache entries: {initial_stats['ml_cache']['total_entries']} → {final_stats['ml_cache']['total_entries']}")
    print(f"  • Edge cache entries: {initial_stats['edge_cache']['total_entries']} → {final_stats['edge_cache']['total_entries']}")
    print(f"  • Total errors: {total_errors}")
    print(f"  • Cache system: {'🟢 OPTIMIZED' if total_errors == 0 else '🟡 NEEDS ATTENTION'}")
    
    # Save management report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"cache_management_report_{timestamp}.json"
    
    management_report = {
        'initial_stats': initial_stats,
        'ml_warming': ml_warming,
        'edge_warming': edge_warming,
        'final_stats': final_stats,
        'optimization': optimization
    }
    
    with open(report_filename, 'w') as f:
        json.dump(management_report, f, indent=2, default=str)
    
    print(f"\n📋 Management report saved: {report_filename}")

if __name__ == "__main__":
    main()
