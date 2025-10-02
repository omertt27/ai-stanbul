#!/usr/bin/env python3
"""
TTL Fine-Tuning Configuration
============================

Dynamic TTL optimization based on real usage patterns and production metrics.
Automatically adjusts cache TTL values based on hit rates, usage patterns, and performance data.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import redis
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class CachePattern(Enum):
    """Cache access patterns for optimization"""
    HIGH_FREQUENCY = "high_frequency"      # Accessed very frequently
    STEADY_USAGE = "steady_usage"          # Regular, consistent access
    BURSTY = "bursty"                      # Periodic high usage, then quiet
    DECLINING = "declining"                # Usage decreasing over time
    SEASONAL = "seasonal"                  # Seasonal access patterns
    UNKNOWN = "unknown"                    # Insufficient data

@dataclass
class TTLConfiguration:
    """TTL configuration for a cache type"""
    cache_type: str
    base_ttl_seconds: int
    min_ttl_seconds: int
    max_ttl_seconds: int
    current_multiplier: float
    target_hit_rate_min: float
    target_hit_rate_max: float
    volatility_factor: float
    last_optimized: datetime
    optimization_count: int = 0
    
    def get_current_ttl(self) -> int:
        """Calculate current TTL with multiplier"""
        return max(
            self.min_ttl_seconds,
            min(int(self.base_ttl_seconds * self.current_multiplier), self.max_ttl_seconds)
        )

@dataclass
class UsageMetrics:
    """Usage metrics for TTL optimization"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    request_timestamps: List[datetime] = None
    access_pattern: CachePattern = CachePattern.UNKNOWN
    
    def __post_init__(self):
        if self.request_timestamps is None:
            self.request_timestamps = []
    
    def calculate_hit_rate(self) -> float:
        """Calculate current hit rate"""
        if self.total_requests > 0:
            return (self.cache_hits / self.total_requests) * 100
        return 0.0
    
    def add_request(self, is_hit: bool, response_time_ms: float):
        """Add a new request to metrics"""
        self.total_requests += 1
        if is_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        self.hit_rate = self.calculate_hit_rate()
        
        # Update average response time (exponential moving average)
        alpha = 0.1  # Smoothing factor
        self.avg_response_time_ms = (alpha * response_time_ms + 
                                   (1 - alpha) * self.avg_response_time_ms)
        
        # Track timestamp
        self.request_timestamps.append(datetime.now())
        
        # Keep only last 1000 timestamps to prevent memory bloat
        if len(self.request_timestamps) > 1000:
            self.request_timestamps = self.request_timestamps[-1000:]

class TTLOptimizer:
    """
    Intelligent TTL optimizer that fine-tunes cache TTL values
    based on real usage patterns and performance metrics
    """
    
    def __init__(self):
        self.redis_client = self._initialize_redis()
        
        # TTL configurations for different cache types
        self.ttl_configs = {
            'restaurant_basic_info': TTLConfiguration(
                cache_type='restaurant_basic_info',
                base_ttl_seconds=604800,  # 7 days
                min_ttl_seconds=3600,     # 1 hour
                max_ttl_seconds=1209600,  # 14 days
                current_multiplier=1.0,
                target_hit_rate_min=80.0,
                target_hit_rate_max=95.0,
                volatility_factor=0.1,    # Very low volatility
                last_optimized=datetime.now()
            ),
            'restaurant_details': TTLConfiguration(
                cache_type='restaurant_details',
                base_ttl_seconds=86400,   # 24 hours
                min_ttl_seconds=3600,     # 1 hour
                max_ttl_seconds=172800,   # 2 days
                current_multiplier=1.0,
                target_hit_rate_min=70.0,
                target_hit_rate_max=90.0,
                volatility_factor=0.2,    # Low volatility
                last_optimized=datetime.now()
            ),
            'opening_hours': TTLConfiguration(
                cache_type='opening_hours',
                base_ttl_seconds=14400,   # 4 hours
                min_ttl_seconds=1800,     # 30 minutes
                max_ttl_seconds=43200,    # 12 hours
                current_multiplier=1.0,
                target_hit_rate_min=60.0,
                target_hit_rate_max=85.0,
                volatility_factor=0.3,    # Medium volatility
                last_optimized=datetime.now()
            ),
            'real_time_status': TTLConfiguration(
                cache_type='real_time_status',
                base_ttl_seconds=1800,    # 30 minutes
                min_ttl_seconds=300,      # 5 minutes
                max_ttl_seconds=7200,     # 2 hours
                current_multiplier=1.0,
                target_hit_rate_min=40.0,
                target_hit_rate_max=70.0,
                volatility_factor=0.5,    # High volatility
                last_optimized=datetime.now()
            ),
            'live_pricing': TTLConfiguration(
                cache_type='live_pricing',
                base_ttl_seconds=300,     # 5 minutes
                min_ttl_seconds=60,       # 1 minute
                max_ttl_seconds=3600,     # 1 hour
                current_multiplier=1.0,
                target_hit_rate_min=30.0,
                target_hit_rate_max=60.0,
                volatility_factor=0.7,    # Very high volatility
                last_optimized=datetime.now()
            ),
            'location_search': TTLConfiguration(
                cache_type='location_search',
                base_ttl_seconds=21600,   # 6 hours
                min_ttl_seconds=3600,     # 1 hour
                max_ttl_seconds=86400,    # 24 hours
                current_multiplier=1.0,
                target_hit_rate_min=75.0,
                target_hit_rate_max=92.0,
                volatility_factor=0.15,   # Low volatility
                last_optimized=datetime.now()
            ),
            'preference_search': TTLConfiguration(
                cache_type='preference_search',
                base_ttl_seconds=7200,    # 2 hours
                min_ttl_seconds=1800,     # 30 minutes
                max_ttl_seconds=21600,    # 6 hours
                current_multiplier=1.0,
                target_hit_rate_min=65.0,
                target_hit_rate_max=88.0,
                volatility_factor=0.25,   # Medium-low volatility
                last_optimized=datetime.now()
            )
        }
        
        # Usage metrics tracking
        self.usage_metrics = {}
        for cache_type in self.ttl_configs.keys():
            self.usage_metrics[cache_type] = UsageMetrics()
        
        # Optimization settings
        self.optimization_interval_hours = 2  # Optimize every 2 hours
        self.min_requests_for_optimization = 50  # Minimum requests before optimizing
        self.adaptation_rate = 0.1  # How quickly to adapt TTL (0.1 = 10% adjustment)
        
        # Time-based multipliers for different periods
        self.time_multipliers = {
            'peak_hours': {
                'multiplier': 0.8,  # Shorter TTL during peak
                'hours': [11, 12, 13, 14, 18, 19, 20, 21]  # Meal times
            },
            'off_peak': {
                'multiplier': 1.2,  # Longer TTL during off-peak
                'hours': [15, 16, 17, 22, 23, 0]
            },
            'late_night': {
                'multiplier': 1.5,  # Much longer TTL late night
                'hours': [1, 2, 3, 4, 5, 6]  
            },
            'morning': {
                'multiplier': 1.0,  # Normal TTL in morning
                'hours': [7, 8, 9, 10]
            }
        }
        
        # Load saved configurations
        self._load_configurations()
        
        logger.info("🔧 TTL Optimizer initialized with dynamic fine-tuning")
    
    def _initialize_redis(self) -> Optional[redis.Redis]:
        """Initialize Redis for TTL optimization"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/4')  # Use DB 4 for TTL optimization
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for TTL optimization: {e}")
            return None
    
    def _load_configurations(self):
        """Load TTL configurations from Redis or file"""
        try:
            if self.redis_client:
                for cache_type in self.ttl_configs.keys():
                    config_key = f"ttl_config:{cache_type}"
                    config_data = self.redis_client.get(config_key)
                    
                    if config_data:
                        saved_config = json.loads(config_data)
                        # Update current configuration with saved values
                        self.ttl_configs[cache_type].current_multiplier = saved_config.get(
                            'current_multiplier', 1.0
                        )
                        self.ttl_configs[cache_type].optimization_count = saved_config.get(
                            'optimization_count', 0
                        )
                        
                        logger.debug(f"🔧 Loaded TTL config for {cache_type}: multiplier={saved_config.get('current_multiplier', 1.0)}")
        
        except Exception as e:
            logger.error(f"❌ Error loading TTL configurations: {e}")
    
    def _save_configurations(self):
        """Save TTL configurations to Redis"""
        try:
            if self.redis_client:
                for cache_type, config in self.ttl_configs.items():
                    config_key = f"ttl_config:{cache_type}"
                    config_data = {
                        'current_multiplier': config.current_multiplier,
                        'optimization_count': config.optimization_count,
                        'last_optimized': config.last_optimized.isoformat()
                    }
                    
                    self.redis_client.setex(
                        config_key,
                        86400 * 7,  # Keep for 7 days
                        json.dumps(config_data)
                    )
        
        except Exception as e:
            logger.error(f"❌ Error saving TTL configurations: {e}")
    
    def record_cache_access(self, cache_type: str, is_hit: bool, response_time_ms: float):
        """Record a cache access for optimization"""
        if cache_type not in self.usage_metrics:
            self.usage_metrics[cache_type] = UsageMetrics()
        
        self.usage_metrics[cache_type].add_request(is_hit, response_time_ms)
        
        # Detect access pattern
        self._detect_access_pattern(cache_type)
        
        # Check if optimization is needed
        if self._should_optimize(cache_type):
            self._optimize_cache_type(cache_type)
    
    def _detect_access_pattern(self, cache_type: str):
        """Detect access pattern for better TTL optimization"""
        metrics = self.usage_metrics[cache_type]
        
        if len(metrics.request_timestamps) < 10:
            metrics.access_pattern = CachePattern.UNKNOWN
            return
        
        # Analyze recent request timestamps
        recent_timestamps = metrics.request_timestamps[-50:]  # Last 50 requests
        
        if len(recent_timestamps) < 10:
            return
        
        # Calculate time intervals between requests
        intervals = []
        for i in range(1, len(recent_timestamps)):
            interval = (recent_timestamps[i] - recent_timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        coefficient_of_variation = std_interval / avg_interval if avg_interval > 0 else 0
        
        # Classify pattern based on statistical analysis
        if avg_interval < 60 and coefficient_of_variation < 0.5:
            # High frequency, low variation
            metrics.access_pattern = CachePattern.HIGH_FREQUENCY
        elif avg_interval < 300 and coefficient_of_variation < 1.0:
            # Steady usage
            metrics.access_pattern = CachePattern.STEADY_USAGE
        elif coefficient_of_variation > 2.0:
            # High variation - bursty
            metrics.access_pattern = CachePattern.BURSTY
        elif len(recent_timestamps) >= 20:
            # Check for declining usage
            first_half = recent_timestamps[:len(recent_timestamps)//2]
            second_half = recent_timestamps[len(recent_timestamps)//2:]
            
            first_half_rate = len(first_half) / ((first_half[-1] - first_half[0]).total_seconds() / 3600)
            second_half_rate = len(second_half) / ((second_half[-1] - second_half[0]).total_seconds() / 3600)
            
            if second_half_rate < first_half_rate * 0.7:
                metrics.access_pattern = CachePattern.DECLINING
            else:
                metrics.access_pattern = CachePattern.STEADY_USAGE
        else:
            metrics.access_pattern = CachePattern.UNKNOWN
    
    def _should_optimize(self, cache_type: str) -> bool:
        """Determine if cache type should be optimized"""
        config = self.ttl_configs.get(cache_type)
        metrics = self.usage_metrics.get(cache_type)
        
        if not config or not metrics:
            return False
        
        # Check minimum requests threshold
        if metrics.total_requests < self.min_requests_for_optimization:
            return False
        
        # Check time since last optimization
        time_since_optimization = datetime.now() - config.last_optimized
        if time_since_optimization.total_seconds() < (self.optimization_interval_hours * 3600):
            return False
        
        # Check if hit rate is outside target range
        if (metrics.hit_rate < config.target_hit_rate_min or 
            metrics.hit_rate > config.target_hit_rate_max):
            return True
        
        return False
    
    def _optimize_cache_type(self, cache_type: str):
        """Optimize TTL for a specific cache type"""
        config = self.ttl_configs.get(cache_type)
        metrics = self.usage_metrics.get(cache_type)
        
        if not config or not metrics:
            return
        
        old_multiplier = config.current_multiplier
        old_ttl = config.get_current_ttl()
        
        # Calculate new multiplier based on hit rate and access pattern
        if metrics.hit_rate < config.target_hit_rate_min:
            # Hit rate too low - increase TTL
            adjustment = self._calculate_ttl_adjustment(config, metrics, increase=True)
            config.current_multiplier = min(
                config.current_multiplier * (1 + adjustment),
                config.max_ttl_seconds / config.base_ttl_seconds
            )
            
        elif metrics.hit_rate > config.target_hit_rate_max:
            # Hit rate too high - decrease TTL for freshness
            adjustment = self._calculate_ttl_adjustment(config, metrics, increase=False)
            config.current_multiplier = max(
                config.current_multiplier * (1 - adjustment),
                config.min_ttl_seconds / config.base_ttl_seconds
            )
        
        # Apply access pattern adjustments
        pattern_adjustment = self._get_pattern_adjustment(metrics.access_pattern)
        config.current_multiplier *= pattern_adjustment
        
        # Ensure multiplier stays within bounds
        max_multiplier = config.max_ttl_seconds / config.base_ttl_seconds
        min_multiplier = config.min_ttl_seconds / config.base_ttl_seconds
        
        config.current_multiplier = max(min_multiplier, min(config.current_multiplier, max_multiplier))
        
        # Update optimization metadata
        config.optimization_count += 1
        config.last_optimized = datetime.now()
        
        new_ttl = config.get_current_ttl()
        
        # Save configuration
        self._save_configurations()
        
        logger.info(f"🔧 TTL Optimization - {cache_type}: "
                   f"Hit rate: {metrics.hit_rate:.1f}% → "
                   f"TTL: {old_ttl}s → {new_ttl}s "
                   f"(multiplier: {old_multiplier:.3f} → {config.current_multiplier:.3f})")
    
    def _calculate_ttl_adjustment(self, config: TTLConfiguration, metrics: UsageMetrics, increase: bool) -> float:
        """Calculate TTL adjustment amount"""
        
        # Base adjustment rate
        base_adjustment = self.adaptation_rate
        
        # Adjust based on how far we are from target
        if increase:
            hit_rate_distance = (config.target_hit_rate_min - metrics.hit_rate) / config.target_hit_rate_min
        else:
            hit_rate_distance = (metrics.hit_rate - config.target_hit_rate_max) / config.target_hit_rate_max
        
        # Scale adjustment by distance from target
        adjustment = base_adjustment * (1 + hit_rate_distance)
        
        # Apply volatility factor - more volatile data needs smaller adjustments
        adjustment *= (1 - config.volatility_factor)
        
        # Response time consideration - if responses are slow, be more conservative
        if metrics.avg_response_time_ms > 500:  # If slower than 500ms
            adjustment *= 0.7
        
        return min(adjustment, 0.3)  # Cap at 30% adjustment
    
    def _get_pattern_adjustment(self, pattern: CachePattern) -> float:
        """Get TTL adjustment based on access pattern"""
        pattern_adjustments = {
            CachePattern.HIGH_FREQUENCY: 1.2,    # Increase TTL for frequent access
            CachePattern.STEADY_USAGE: 1.0,      # No adjustment
            CachePattern.BURSTY: 0.9,           # Decrease TTL for bursty access
            CachePattern.DECLINING: 0.8,        # Decrease TTL for declining usage
            CachePattern.SEASONAL: 1.1,         # Slightly increase for seasonal
            CachePattern.UNKNOWN: 1.0           # No adjustment
        }
        
        return pattern_adjustments.get(pattern, 1.0)
    
    def get_optimized_ttl(self, cache_type: str) -> int:
        """Get current optimized TTL for cache type"""
        config = self.ttl_configs.get(cache_type)
        if not config:
            return 3600  # Default 1 hour
        
        base_ttl = config.get_current_ttl()
        
        # Apply time-of-day multiplier
        current_hour = datetime.now().hour
        time_multiplier = 1.0
        
        for period, settings in self.time_multipliers.items():
            if current_hour in settings['hours']:
                time_multiplier = settings['multiplier']
                break
        
        # Weekend adjustment
        if datetime.now().weekday() >= 5:  # Weekend
            time_multiplier *= 1.1
        
        final_ttl = int(base_ttl * time_multiplier)
        
        # Ensure within bounds
        return max(config.min_ttl_seconds, min(final_ttl, config.max_ttl_seconds))
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'cache_types': {},
            'summary': {
                'total_cache_types': len(self.ttl_configs),
                'total_optimizations': sum(c.optimization_count for c in self.ttl_configs.values()),
                'avg_hit_rate': 0.0,
                'total_requests': sum(m.total_requests for m in self.usage_metrics.values())
            }
        }
        
        total_hit_rate = 0
        active_cache_types = 0
        
        for cache_type, config in self.ttl_configs.items():
            metrics = self.usage_metrics.get(cache_type, UsageMetrics())
            
            if metrics.total_requests > 0:
                total_hit_rate += metrics.hit_rate
                active_cache_types += 1
            
            report['cache_types'][cache_type] = {
                'configuration': {
                    'base_ttl_seconds': config.base_ttl_seconds,
                    'current_ttl_seconds': config.get_current_ttl(),
                    'current_multiplier': round(config.current_multiplier, 3),
                    'target_hit_rate_range': [config.target_hit_rate_min, config.target_hit_rate_max],
                    'volatility_factor': config.volatility_factor
                },
                'metrics': {
                    'total_requests': metrics.total_requests,
                    'hit_rate': round(metrics.hit_rate, 2),
                    'avg_response_time_ms': round(metrics.avg_response_time_ms, 1),
                    'access_pattern': metrics.access_pattern.value
                },
                'optimization': {
                    'optimization_count': config.optimization_count,
                    'last_optimized': config.last_optimized.isoformat(),
                    'hours_since_optimization': round(
                        (datetime.now() - config.last_optimized).total_seconds() / 3600, 1
                    )
                }
            }
        
        if active_cache_types > 0:
            report['summary']['avg_hit_rate'] = round(total_hit_rate / active_cache_types, 2)
        
        return report
    
    def force_optimize_all(self) -> Dict[str, Any]:
        """Force optimization of all cache types (for testing/manual optimization)"""
        results = {}
        
        for cache_type in self.ttl_configs.keys():
            metrics = self.usage_metrics.get(cache_type)
            
            if metrics and metrics.total_requests >= 10:  # Lower threshold for forced optimization
                old_multiplier = self.ttl_configs[cache_type].current_multiplier
                
                # Temporarily reduce optimization interval to allow forced optimization
                original_last_optimized = self.ttl_configs[cache_type].last_optimized
                self.ttl_configs[cache_type].last_optimized = datetime.now() - timedelta(hours=3)
                
                self._optimize_cache_type(cache_type)
                
                new_multiplier = self.ttl_configs[cache_type].current_multiplier
                
                results[cache_type] = {
                    'optimized': True,
                    'old_multiplier': round(old_multiplier, 3),
                    'new_multiplier': round(new_multiplier, 3),
                    'hit_rate': round(metrics.hit_rate, 2)
                }
            else:
                results[cache_type] = {
                    'optimized': False,
                    'reason': 'Insufficient data for optimization'
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'total_optimized': sum(1 for r in results.values() if r.get('optimized', False))
        }

# Global TTL optimizer instance
ttl_optimizer = TTLOptimizer()

def get_optimized_ttl(cache_type: str) -> int:
    """Get optimized TTL for cache type"""
    return ttl_optimizer.get_optimized_ttl(cache_type)

def record_cache_access(cache_type: str, is_hit: bool, response_time_ms: float):
    """Record cache access for TTL optimization"""
    ttl_optimizer.record_cache_access(cache_type, is_hit, response_time_ms)

def get_ttl_optimization_report() -> Dict[str, Any]:
    """Get TTL optimization report"""
    return ttl_optimizer.get_optimization_report()

def force_ttl_optimization() -> Dict[str, Any]:
    """Force TTL optimization for all cache types"""
    return ttl_optimizer.force_optimize_all()

if __name__ == "__main__":
    # Test TTL optimization
    print("🔧 Testing TTL Optimization System")
    print("=" * 50)
    
    # Simulate cache accesses
    test_scenarios = [
        ('restaurant_basic_info', [(True, 150)] * 80 + [(False, 300)] * 20),  # High hit rate
        ('opening_hours', [(True, 200)] * 40 + [(False, 400)] * 60),          # Low hit rate
        ('real_time_status', [(True, 100)] * 30 + [(False, 200)] * 70),       # Very low hit rate
    ]
    
    for cache_type, accesses in test_scenarios:
        print(f"\n📊 Simulating {len(accesses)} accesses for {cache_type}")
        for is_hit, response_time in accesses:
            ttl_optimizer.record_cache_access(cache_type, is_hit, response_time)
    
    # Force optimization
    print(f"\n🔧 Running forced optimization...")
    optimization_results = ttl_optimizer.force_optimize_all()
    
    for cache_type, result in optimization_results['results'].items():
        if result['optimized']:
            print(f"✅ {cache_type}: {result['old_multiplier']} → {result['new_multiplier']} "
                  f"(hit rate: {result['hit_rate']}%)")
        else:
            print(f"⏸️ {cache_type}: {result['reason']}")
    
    # Show optimization report
    print(f"\n📈 Optimization Report:")
    report = ttl_optimizer.get_optimization_report()
    print(f"Total requests: {report['summary']['total_requests']}")
    print(f"Average hit rate: {report['summary']['avg_hit_rate']}%")
    print(f"Total optimizations: {report['summary']['total_optimizations']}")
