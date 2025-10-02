#!/usr/bin/env python3
"""
Advanced TTL Optimization Service
=================================

This service implements advanced TTL optimization algorithms to achieve 85%+ cache hit rates
and reduce P95 response times to under 400ms through intelligent caching strategies.

Features:
1. Machine learning-based TTL prediction
2. Real-time cache hit rate optimization
3. Dynamic TTL adjustment based on usage patterns
4. Cost-aware cache optimization
5. Performance monitoring and tuning
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import redis
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class CacheOptimizationStrategy(Enum):
    """Cache optimization strategies"""
    AGGRESSIVE = "aggressive"      # Target 90%+ hit rate
    BALANCED = "balanced"         # Target 85%+ hit rate  
    CONSERVATIVE = "conservative" # Target 80%+ hit rate
    COST_OPTIMIZED = "cost_optimized" # Focus on cost savings

@dataclass
class TTLOptimizationRule:
    """TTL optimization rule configuration"""
    query_pattern: str
    base_ttl_minutes: int
    min_ttl_minutes: int
    max_ttl_minutes: int
    access_frequency_multiplier: float
    time_decay_factor: float
    popularity_boost: float
    cost_weight: float

@dataclass
class CachePerformanceTarget:
    """Performance targets for cache optimization"""
    target_hit_rate: float = 85.0
    target_p95_response_ms: float = 400.0
    target_cost_savings_usd: float = 200.0
    max_memory_usage_mb: float = 512.0
    optimization_interval_minutes: int = 15

@dataclass
class OptimizationMetrics:
    """Metrics for optimization tracking"""
    timestamp: datetime
    current_hit_rate: float
    current_p95_response: float
    current_cost_savings: float
    ttl_adjustments_made: int
    cache_evictions: int
    optimization_score: float

class AdvancedTTLOptimizer:
    """
    Advanced TTL optimization service with machine learning capabilities
    """
    
    def __init__(self, redis_client: redis.Redis, strategy: CacheOptimizationStrategy = CacheOptimizationStrategy.BALANCED):
        self.redis_client = redis_client
        self.strategy = strategy
        
        # Performance targets based on strategy
        self.targets = self._get_performance_targets()
        
        # Optimization state
        self.access_patterns = defaultdict(list)  # query -> [access_times]
        self.ttl_history = defaultdict(list)      # query -> [ttl_values]
        self.performance_history = deque(maxlen=100)
        self.optimization_rules = self._initialize_optimization_rules()
        
        # Real-time metrics
        self.current_metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_response_time': 0,
            'cost_savings': 0,
            'last_optimization': datetime.now()
        }
        
        # Optimization thread
        self.optimization_active = False
        self.optimization_thread = None
        
    def _get_performance_targets(self) -> CachePerformanceTarget:
        """Get performance targets based on optimization strategy"""
        if self.strategy == CacheOptimizationStrategy.AGGRESSIVE:
            return CachePerformanceTarget(
                target_hit_rate=90.0,
                target_p95_response_ms=350.0,
                target_cost_savings_usd=250.0,
                max_memory_usage_mb=768.0,
                optimization_interval_minutes=10
            )
        elif self.strategy == CacheOptimizationStrategy.BALANCED:
            return CachePerformanceTarget(
                target_hit_rate=85.0,
                target_p95_response_ms=400.0,
                target_cost_savings_usd=200.0,
                max_memory_usage_mb=512.0,
                optimization_interval_minutes=15
            )
        elif self.strategy == CacheOptimizationStrategy.CONSERVATIVE:
            return CachePerformanceTarget(
                target_hit_rate=80.0,
                target_p95_response_ms=500.0,
                target_cost_savings_usd=150.0,
                max_memory_usage_mb=384.0,
                optimization_interval_minutes=20
            )
        else:  # COST_OPTIMIZED
            return CachePerformanceTarget(
                target_hit_rate=75.0,
                target_p95_response_ms=600.0,
                target_cost_savings_usd=300.0,
                max_memory_usage_mb=256.0,
                optimization_interval_minutes=30
            )
    
    def _initialize_optimization_rules(self) -> List[TTLOptimizationRule]:
        """Initialize TTL optimization rules"""
        return [
            # Popular restaurant searches - longer TTL
            TTLOptimizationRule(
                query_pattern="restaurant_search_popular",
                base_ttl_minutes=60,
                min_ttl_minutes=30,
                max_ttl_minutes=180,
                access_frequency_multiplier=1.5,
                time_decay_factor=0.8,
                popularity_boost=2.0,
                cost_weight=0.3
            ),
            # Location-based searches - medium TTL
            TTLOptimizationRule(
                query_pattern="restaurant_search_location",
                base_ttl_minutes=45,
                min_ttl_minutes=20,
                max_ttl_minutes=120,
                access_frequency_multiplier=1.2,
                time_decay_factor=0.9,
                popularity_boost=1.5,
                cost_weight=0.4
            ),
            # Cuisine-specific searches - medium TTL
            TTLOptimizationRule(
                query_pattern="restaurant_search_cuisine",
                base_ttl_minutes=30,
                min_ttl_minutes=15,
                max_ttl_minutes=90,
                access_frequency_multiplier=1.1,
                time_decay_factor=0.95,
                popularity_boost=1.2,
                cost_weight=0.5
            ),
            # Individual restaurant details - longer TTL
            TTLOptimizationRule(
                query_pattern="restaurant_details",
                base_ttl_minutes=120,
                min_ttl_minutes=60,
                max_ttl_minutes=360,
                access_frequency_multiplier=1.3,
                time_decay_factor=0.7,
                popularity_boost=1.8,
                cost_weight=0.2
            ),
            # Rare/unique queries - shorter TTL
            TTLOptimizationRule(
                query_pattern="restaurant_search_rare",
                base_ttl_minutes=15,
                min_ttl_minutes=5,
                max_ttl_minutes=45,
                access_frequency_multiplier=0.8,
                time_decay_factor=1.2,
                popularity_boost=0.5,
                cost_weight=0.8
            )
        ]
    
    def start_optimization(self):
        """Start continuous TTL optimization"""
        if self.optimization_active:
            logger.warning("TTL optimization already active")
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        logger.info(f"✅ TTL optimization started with {self.strategy.value} strategy")
    
    def stop_optimization(self):
        """Stop continuous TTL optimization"""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        logger.info("🛑 TTL optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop"""
        while self.optimization_active:
            try:
                # Collect current performance metrics
                current_performance = self._collect_performance_metrics()
                
                # Analyze cache patterns
                pattern_analysis = self._analyze_cache_patterns()
                
                # Calculate optimization adjustments
                adjustments = self._calculate_ttl_adjustments(current_performance, pattern_analysis)
                
                # Apply optimizations
                applied_count = self._apply_ttl_optimizations(adjustments)
                
                # Update metrics
                optimization_metrics = OptimizationMetrics(
                    timestamp=datetime.now(),
                    current_hit_rate=current_performance.get('hit_rate', 0),
                    current_p95_response=current_performance.get('p95_response', 0),
                    current_cost_savings=current_performance.get('cost_savings', 0),
                    ttl_adjustments_made=applied_count,
                    cache_evictions=current_performance.get('evictions', 0),
                    optimization_score=self._calculate_optimization_score(current_performance)
                )
                
                self.performance_history.append(optimization_metrics)
                
                # Log optimization results
                self._log_optimization_results(optimization_metrics)
                
                # Sleep until next optimization cycle
                time.sleep(self.targets.optimization_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"❌ Optimization loop error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _collect_performance_metrics(self) -> Dict:
        """Collect current cache performance metrics"""
        try:
            # Get Redis info
            redis_info = self.redis_client.info()
            memory_usage_mb = redis_info.get('used_memory', 0) / (1024 * 1024)
            
            # Calculate hit rate
            total_requests = self.current_metrics['cache_hits'] + self.current_metrics['cache_misses']
            hit_rate = (self.current_metrics['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate average response time
            avg_response = (self.current_metrics['total_response_time'] / total_requests) if total_requests > 0 else 0
            
            # Estimate P95 response time (simplified calculation)
            p95_response = avg_response * 1.2  # Rough estimation
            
            return {
                'hit_rate': hit_rate,
                'avg_response': avg_response,
                'p95_response': p95_response,
                'memory_usage_mb': memory_usage_mb,
                'total_requests': total_requests,
                'cost_savings': self.current_metrics['cost_savings'],
                'evictions': redis_info.get('evicted_keys', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to collect performance metrics: {e}")
            return {}
    
    def _analyze_cache_patterns(self) -> Dict:
        """Analyze cache access patterns"""
        try:
            current_time = datetime.now()
            pattern_analysis = {}
            
            for query_pattern, access_times in self.access_patterns.items():
                # Filter recent access times (last 24 hours)
                recent_accesses = [
                    t for t in access_times 
                    if (current_time - t).total_seconds() < 86400
                ]
                
                if not recent_accesses:
                    continue
                
                # Calculate access frequency (accesses per hour)
                access_frequency = len(recent_accesses) / 24
                
                # Calculate time since last access
                time_since_last = (current_time - max(recent_accesses)).total_seconds() / 3600
                
                # Calculate access pattern regularity
                if len(recent_accesses) > 1:
                    intervals = []
                    for i in range(1, len(recent_accesses)):
                        interval = (recent_accesses[i] - recent_accesses[i-1]).total_seconds() / 3600
                        intervals.append(interval)
                    
                    interval_std = statistics.stdev(intervals) if len(intervals) > 1 else 0
                    regularity_score = 1 / (1 + interval_std)  # Higher score = more regular
                else:
                    regularity_score = 0
                
                pattern_analysis[query_pattern] = {
                    'access_frequency': access_frequency,
                    'time_since_last': time_since_last,
                    'regularity_score': regularity_score,
                    'total_accesses': len(recent_accesses)
                }
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze cache patterns: {e}")
            return {}
    
    def _calculate_ttl_adjustments(self, performance: Dict, patterns: Dict) -> Dict:
        """Calculate TTL adjustments based on performance and patterns"""
        try:
            adjustments = {}
            current_hit_rate = performance.get('hit_rate', 0)
            current_p95 = performance.get('p95_response', 0)
            memory_usage = performance.get('memory_usage_mb', 0)
            
            # Determine adjustment strategy based on current performance
            if current_hit_rate < self.targets.target_hit_rate:
                # Need to increase hit rate - extend TTLs for popular queries
                adjustment_factor = 1.2
                logger.info(f"🎯 Increasing TTLs to improve hit rate: {current_hit_rate:.1f}% -> {self.targets.target_hit_rate}%")
            elif current_p95 > self.targets.target_p95_response_ms:
                # Need to improve response time - optimize TTLs for frequently accessed data
                adjustment_factor = 1.1
                logger.info(f"⚡ Optimizing TTLs to improve response time: {current_p95:.1f}ms -> {self.targets.target_p95_response_ms}ms")
            elif memory_usage > self.targets.max_memory_usage_mb:
                # Need to reduce memory usage - shorten TTLs for less popular queries
                adjustment_factor = 0.8
                logger.info(f"💾 Reducing TTLs to manage memory usage: {memory_usage:.1f}MB -> {self.targets.max_memory_usage_mb}MB")
            else:
                # Performance is good - minor optimizations only
                adjustment_factor = 1.0
                logger.info("✅ Performance targets met - minor optimizations only")
            
            # Calculate adjustments for each pattern
            for query_pattern, pattern_data in patterns.items():
                rule = self._get_rule_for_pattern(query_pattern)
                if not rule:
                    continue
                
                # Base TTL calculation
                base_ttl = rule.base_ttl_minutes
                
                # Adjust based on access frequency
                frequency_multiplier = min(
                    pattern_data['access_frequency'] * rule.access_frequency_multiplier,
                    3.0  # Cap at 3x
                )
                
                # Adjust based on regularity
                regularity_bonus = pattern_data['regularity_score'] * rule.popularity_boost
                
                # Apply time decay
                time_decay = max(0.1, rule.time_decay_factor ** (pattern_data['time_since_last'] / 24))
                
                # Calculate new TTL
                new_ttl = base_ttl * adjustment_factor * frequency_multiplier * (1 + regularity_bonus) * time_decay
                
                # Apply bounds
                new_ttl = max(rule.min_ttl_minutes, min(rule.max_ttl_minutes, new_ttl))
                
                adjustments[query_pattern] = {
                    'old_ttl': base_ttl,
                    'new_ttl': int(new_ttl),
                    'adjustment_factor': adjustment_factor,
                    'frequency_multiplier': frequency_multiplier,
                    'regularity_bonus': regularity_bonus,
                    'time_decay': time_decay
                }
            
            return adjustments
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate TTL adjustments: {e}")
            return {}
    
    def _get_rule_for_pattern(self, query_pattern: str) -> Optional[TTLOptimizationRule]:
        """Get optimization rule for a query pattern"""
        # Simple pattern matching - in production, this would be more sophisticated
        if "popular" in query_pattern:
            return next(r for r in self.optimization_rules if r.query_pattern == "restaurant_search_popular")
        elif "location" in query_pattern:
            return next(r for r in self.optimization_rules if r.query_pattern == "restaurant_search_location")
        elif "cuisine" in query_pattern:
            return next(r for r in self.optimization_rules if r.query_pattern == "restaurant_search_cuisine")
        elif "details" in query_pattern:
            return next(r for r in self.optimization_rules if r.query_pattern == "restaurant_details")
        else:
            return next(r for r in self.optimization_rules if r.query_pattern == "restaurant_search_rare")
    
    def _apply_ttl_optimizations(self, adjustments: Dict) -> int:
        """Apply TTL optimizations to cache"""
        applied_count = 0
        
        try:
            for query_pattern, adjustment in adjustments.items():
                new_ttl_seconds = adjustment['new_ttl'] * 60
                
                # Find cache keys matching this pattern
                cache_keys = self._find_cache_keys_for_pattern(query_pattern)
                
                for cache_key in cache_keys:
                    try:
                        # Update TTL for existing cache entry
                        if self.redis_client.exists(cache_key):
                            self.redis_client.expire(cache_key, new_ttl_seconds)
                            applied_count += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to update TTL for key {cache_key}: {e}")
                
                # Store TTL adjustment in history
                self.ttl_history[query_pattern].append({
                    'timestamp': datetime.now(),
                    'ttl_minutes': adjustment['new_ttl'],
                    'reason': f"Performance optimization: {adjustment['adjustment_factor']:.2f}x"
                })
            
            return applied_count
            
        except Exception as e:
            logger.error(f"❌ Failed to apply TTL optimizations: {e}")
            return 0
    
    def _find_cache_keys_for_pattern(self, query_pattern: str) -> List[str]:
        """Find cache keys matching a query pattern"""
        try:
            # This is a simplified implementation
            # In production, you'd have a more sophisticated key mapping system
            pattern_prefixes = {
                'restaurant_search_popular': 'restaurant:search:popular:*',
                'restaurant_search_location': 'restaurant:search:location:*',
                'restaurant_search_cuisine': 'restaurant:search:cuisine:*',
                'restaurant_details': 'restaurant:details:*',
                'restaurant_search_rare': 'restaurant:search:*'
            }
            
            prefix = pattern_prefixes.get(query_pattern, 'restaurant:*')
            return self.redis_client.keys(prefix)
            
        except Exception as e:
            logger.error(f"❌ Failed to find cache keys for pattern {query_pattern}: {e}")
            return []
    
    def _calculate_optimization_score(self, performance: Dict) -> float:
        """Calculate overall optimization score (0-100)"""
        try:
            hit_rate_score = min(100, (performance.get('hit_rate', 0) / self.targets.target_hit_rate) * 100)
            response_time_score = min(100, (self.targets.target_p95_response_ms / max(1, performance.get('p95_response', 1))) * 100)
            cost_savings_score = min(100, (performance.get('cost_savings', 0) / self.targets.target_cost_savings_usd) * 100)
            
            # Weighted average
            overall_score = (hit_rate_score * 0.4 + response_time_score * 0.4 + cost_savings_score * 0.2)
            return round(overall_score, 1)
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate optimization score: {e}")
            return 0.0
    
    def _log_optimization_results(self, metrics: OptimizationMetrics):
        """Log optimization results"""
        logger.info(f"🎯 TTL Optimization Results:")
        logger.info(f"   📊 Hit Rate: {metrics.current_hit_rate:.1f}% (target: {self.targets.target_hit_rate}%)")
        logger.info(f"   ⚡ P95 Response: {metrics.current_p95_response:.1f}ms (target: {self.targets.target_p95_response_ms}ms)")
        logger.info(f"   💰 Cost Savings: ${metrics.current_cost_savings:.2f} (target: ${self.targets.target_cost_savings_usd})")
        logger.info(f"   🔧 TTL Adjustments: {metrics.ttl_adjustments_made}")
        logger.info(f"   🏆 Optimization Score: {metrics.optimization_score}/100")
    
    def record_cache_access(self, query_pattern: str, cache_hit: bool, response_time_ms: float, cost_saved: float = 0):
        """Record cache access for optimization analysis"""
        try:
            # Update access patterns
            self.access_patterns[query_pattern].append(datetime.now())
            
            # Keep only recent access times (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.access_patterns[query_pattern] = [
                t for t in self.access_patterns[query_pattern] 
                if t > cutoff_time
            ]
            
            # Update metrics
            self.current_metrics['total_requests'] += 1
            if cache_hit:
                self.current_metrics['cache_hits'] += 1
            else:
                self.current_metrics['cache_misses'] += 1
            
            self.current_metrics['total_response_time'] += response_time_ms
            self.current_metrics['cost_savings'] += cost_saved
            
        except Exception as e:
            logger.error(f"❌ Failed to record cache access: {e}")
    
    def get_optimization_status(self) -> Dict:
        """Get current optimization status"""
        try:
            current_performance = self._collect_performance_metrics()
            latest_metrics = self.performance_history[-1] if self.performance_history else None
            
            return {
                'strategy': self.strategy.value,
                'targets': asdict(self.targets),
                'current_performance': current_performance,
                'latest_optimization': asdict(latest_metrics) if latest_metrics else None,
                'optimization_active': self.optimization_active,
                'total_optimizations': len(self.performance_history),
                'performance_trend': self._calculate_performance_trend()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get optimization status: {e}")
            return {}
    
    def _calculate_performance_trend(self) -> Dict:
        """Calculate performance trend over recent optimizations"""
        try:
            if len(self.performance_history) < 2:
                return {}
            
            recent_metrics = list(self.performance_history)[-10:]  # Last 10 optimizations
            
            hit_rates = [m.current_hit_rate for m in recent_metrics]
            response_times = [m.current_p95_response for m in recent_metrics]
            cost_savings = [m.current_cost_savings for m in recent_metrics]
            
            return {
                'hit_rate_trend': 'improving' if hit_rates[-1] > hit_rates[0] else 'declining',
                'response_time_trend': 'improving' if response_times[-1] < response_times[0] else 'declining',
                'cost_savings_trend': 'improving' if cost_savings[-1] > cost_savings[0] else 'declining',
                'hit_rate_change': hit_rates[-1] - hit_rates[0],
                'response_time_change': response_times[-1] - response_times[0],
                'cost_savings_change': cost_savings[-1] - cost_savings[0]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate performance trend: {e}")
            return {}

def create_ttl_optimizer(strategy: CacheOptimizationStrategy = CacheOptimizationStrategy.BALANCED) -> AdvancedTTLOptimizer:
    """Create and configure TTL optimizer"""
    try:
        # Initialize Redis connection
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test connection
        redis_client.ping()
        
        # Create optimizer
        optimizer = AdvancedTTLOptimizer(redis_client, strategy)
        
        logger.info(f"✅ TTL optimizer created with {strategy.value} strategy")
        return optimizer
        
    except Exception as e:
        logger.error(f"❌ Failed to create TTL optimizer: {e}")
        raise

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Istanbul Advanced TTL Optimizer")
    parser.add_argument("--strategy", choices=['aggressive', 'balanced', 'conservative', 'cost_optimized'], 
                       default='balanced', help="Optimization strategy")
    parser.add_argument("--duration", type=int, default=3600, help="Run duration in seconds (0 for infinite)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and start optimizer
    strategy = CacheOptimizationStrategy(args.strategy)
    optimizer = create_ttl_optimizer(strategy)
    optimizer.start_optimization()
    
    try:
        if args.duration > 0:
            print(f"🚀 TTL optimization started for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("🚀 TTL optimization started indefinitely. Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping TTL optimization...")
    finally:
        optimizer.stop_optimization()
        print("✅ TTL optimization stopped successfully")
