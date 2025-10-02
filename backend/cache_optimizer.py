#!/usr/bin/env python3
"""
Advanced Cache Optimization Module
=================================

This module implements advanced cache optimization strategies to achieve higher cache hit rates,
lower response times, and increased cost savings for the AI Istanbul system.

Features:
1. Dynamic TTL optimization based on access patterns
2. Intelligent cache warming for popular queries
3. Cache performance tuning and analysis
4. Cost-based cache optimization
5. Predictive caching for trending queries
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter, deque
import redis
import threading
import pickle
from enum import Enum
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache optimization strategies"""
    ACCESS_FREQUENCY = "access_frequency"
    TIME_BASED = "time_based"
    COST_BASED = "cost_based"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"

@dataclass
class CacheOptimizationConfig:
    """Configuration for cache optimization"""
    target_hit_rate: float = 85.0  # Target cache hit rate %
    target_response_time_p95: float = 400.0  # Target P95 response time ms
    target_cost_savings: float = 200.0  # Target monthly cost savings $
    
    # TTL optimization settings
    min_ttl_seconds: int = 300  # 5 minutes
    max_ttl_seconds: int = 86400  # 24 hours
    ttl_adjustment_factor: float = 1.2  # Multiplier for TTL adjustments
    
    # Cache warming settings
    warm_cache_enabled: bool = True
    warm_cache_queries: int = 50  # Number of popular queries to warm
    warm_cache_frequency_hours: int = 6  # How often to warm cache
    
    # Predictive caching settings
    predictive_enabled: bool = True
    prediction_horizon_hours: int = 4  # How far ahead to predict
    prediction_confidence_threshold: float = 0.7  # Minimum confidence for predictions

@dataclass
class QueryPattern:
    """Query access pattern analysis"""
    query_hash: str
    access_count: int
    last_access: datetime
    average_response_time: float
    cost_impact: float
    seasonal_pattern: List[float]  # Hourly access pattern (24 values)
    current_ttl: int
    optimal_ttl: int
    hit_rate: float

class AdvancedCacheOptimizer:
    """
    Advanced cache optimizer with multiple optimization strategies
    """
    
    def __init__(self, redis_client: redis.Redis, config: CacheOptimizationConfig = None):
        self.redis_client = redis_client
        self.config = config or CacheOptimizationConfig()
        
        # Query pattern tracking
        self.query_patterns = {}
        self.access_log = deque(maxlen=10000)  # Track last 10k accesses
        self.response_times = defaultdict(list)
        self.cost_data = defaultdict(float)
        
        # Optimization state
        self.optimization_active = False
        self.optimization_thread = None
        self.last_optimization = datetime.min
        
        # Predictive model (simple linear regression for demo)
        self.prediction_model = None
        self.model_features = ['hour', 'day_of_week', 'access_frequency', 'recent_trend']
        
        # Performance metrics
        self.optimization_results = deque(maxlen=100)
        
    def start_optimization(self, interval_minutes: int = 30):
        """Start continuous cache optimization"""
        if self.optimization_active:
            logger.warning("Cache optimization already active")
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            args=(interval_minutes,),
            daemon=True
        )
        self.optimization_thread.start()
        logger.info(f"✅ Cache optimization started (interval: {interval_minutes} minutes)")
    
    def stop_optimization(self):
        """Stop cache optimization"""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        logger.info("🛑 Cache optimization stopped")
    
    def _optimization_loop(self, interval_minutes: int):
        """Main optimization loop"""
        while self.optimization_active:
            try:
                # Run optimization cycle
                self._run_optimization_cycle()
                
                # Sleep for interval
                time.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"❌ Optimization loop error: {e}")
                time.sleep(60)  # Sleep 1 minute on error
    
    def _run_optimization_cycle(self):
        """Run a complete optimization cycle"""
        start_time = time.time()
        logger.info("🔄 Starting cache optimization cycle...")
        
        try:
            # Step 1: Analyze current cache performance
            performance_metrics = self._analyze_cache_performance()
            
            # Step 2: Update query patterns
            self._update_query_patterns()
            
            # Step 3: Optimize TTL values
            ttl_improvements = self._optimize_ttl_values()
            
            # Step 4: Warm cache with popular queries
            if self.config.warm_cache_enabled:
                cache_warming_results = self._warm_popular_queries()
            else:
                cache_warming_results = {"warmed_queries": 0}
            
            # Step 5: Predictive caching
            if self.config.predictive_enabled:
                prediction_results = self._run_predictive_caching()
            else:
                prediction_results = {"predicted_queries": 0}
            
            # Step 6: Apply optimizations
            applied_optimizations = self._apply_optimizations(ttl_improvements)
            
            # Record optimization results
            cycle_duration = time.time() - start_time
            results = {
                'timestamp': datetime.now(),
                'cycle_duration_seconds': cycle_duration,
                'performance_metrics': performance_metrics,
                'ttl_improvements': len(ttl_improvements),
                'cache_warming': cache_warming_results,
                'predictions': prediction_results,
                'applied_optimizations': applied_optimizations
            }
            
            self.optimization_results.append(results)
            self.last_optimization = datetime.now()
            
            logger.info(f"✅ Optimization cycle completed in {cycle_duration:.1f}s - "
                       f"TTL: {len(ttl_improvements)}, "
                       f"Warmed: {cache_warming_results.get('warmed_queries', 0)}, "
                       f"Predicted: {prediction_results.get('predicted_queries', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Optimization cycle failed: {e}")
    
    def _analyze_cache_performance(self) -> Dict:
        """Analyze current cache performance"""
        try:
            # Get cache statistics from Redis
            redis_info = self.redis_client.info()
            
            # Calculate hit rates by pattern
            pattern_performance = {}
            total_hits = 0
            total_requests = 0
            
            for query_hash, pattern in self.query_patterns.items():
                hits = pattern.access_count * pattern.hit_rate
                total_hits += hits
                total_requests += pattern.access_count
                
                pattern_performance[query_hash] = {
                    'hit_rate': pattern.hit_rate,
                    'response_time': pattern.average_response_time,
                    'cost_impact': pattern.cost_impact,
                    'access_frequency': pattern.access_count
                }
            
            overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'overall_hit_rate': overall_hit_rate,
                'total_patterns': len(self.query_patterns),
                'redis_memory_usage': redis_info.get('used_memory', 0),
                'redis_keys': redis_info.get('db0', {}).get('keys', 0),
                'pattern_performance': pattern_performance
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze cache performance: {e}")
            return {}
    
    def _update_query_patterns(self):
        """Update query access patterns from recent activity"""
        try:
            # Simulate pattern updates (in production, this would read from actual logs)
            current_time = datetime.now()
            
            # Update existing patterns with simulated data
            for query_hash, pattern in self.query_patterns.items():
                # Simulate access frequency decay
                time_since_access = (current_time - pattern.last_access).total_seconds()
                decay_factor = max(0.1, 1.0 - (time_since_access / 86400))  # Decay over 24 hours
                pattern.access_count = int(pattern.access_count * decay_factor)
                
                # Update seasonal patterns (simplified)
                hour = current_time.hour
                pattern.seasonal_pattern[hour] = min(1.0, pattern.seasonal_pattern[hour] + 0.1)
            
            # Add new patterns (simulated)
            if len(self.query_patterns) < 100:  # Limit for demo
                new_query_hash = f"query_{len(self.query_patterns)}_{int(time.time())}"
                self.query_patterns[new_query_hash] = QueryPattern(
                    query_hash=new_query_hash,
                    access_count=np.random.randint(1, 50),
                    last_access=current_time,
                    average_response_time=np.random.uniform(100, 800),
                    cost_impact=np.random.uniform(0.01, 0.50),
                    seasonal_pattern=[np.random.uniform(0, 1) for _ in range(24)],
                    current_ttl=np.random.randint(300, 3600),
                    optimal_ttl=0,  # Will be calculated
                    hit_rate=np.random.uniform(0.5, 0.95)
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to update query patterns: {e}")
    
    def _optimize_ttl_values(self) -> List[Dict]:
        """Optimize TTL values based on access patterns"""
        ttl_improvements = []
        
        try:
            for query_hash, pattern in self.query_patterns.items():
                # Calculate optimal TTL based on multiple factors
                
                # Factor 1: Access frequency (higher frequency = longer TTL)
                frequency_factor = min(2.0, pattern.access_count / 10.0)
                
                # Factor 2: Response time (slower queries = longer TTL)
                response_time_factor = min(2.0, pattern.average_response_time / 500.0)
                
                # Factor 3: Cost impact (expensive queries = longer TTL)
                cost_factor = min(2.0, pattern.cost_impact / 0.10)
                
                # Factor 4: Current hit rate (low hit rate = longer TTL)
                hit_rate_factor = max(0.5, 2.0 - pattern.hit_rate)
                
                # Factor 5: Seasonal pattern (peak hours = longer TTL)
                current_hour = datetime.now().hour
                seasonal_factor = 1.0 + pattern.seasonal_pattern[current_hour]
                
                # Calculate optimal TTL
                base_ttl = self.config.min_ttl_seconds
                optimization_multiplier = (frequency_factor + response_time_factor + 
                                         cost_factor + hit_rate_factor + seasonal_factor) / 5.0
                
                optimal_ttl = int(base_ttl * optimization_multiplier)
                optimal_ttl = max(self.config.min_ttl_seconds, 
                                min(self.config.max_ttl_seconds, optimal_ttl))
                
                # Check if improvement is significant
                improvement_threshold = 0.2  # 20% change required
                current_ttl = pattern.current_ttl
                change_ratio = abs(optimal_ttl - current_ttl) / current_ttl
                
                if change_ratio > improvement_threshold:
                    ttl_improvements.append({
                        'query_hash': query_hash,
                        'current_ttl': current_ttl,
                        'optimal_ttl': optimal_ttl,
                        'improvement_ratio': change_ratio,
                        'factors': {
                            'frequency': frequency_factor,
                            'response_time': response_time_factor,
                            'cost': cost_factor,
                            'hit_rate': hit_rate_factor,
                            'seasonal': seasonal_factor
                        }
                    })
                    
                    # Update the pattern
                    pattern.optimal_ttl = optimal_ttl
            
            # Sort by improvement potential
            ttl_improvements.sort(key=lambda x: x['improvement_ratio'], reverse=True)
            
            return ttl_improvements[:20]  # Return top 20 improvements
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize TTL values: {e}")
            return []
    
    def _warm_popular_queries(self) -> Dict:
        """Warm cache with popular queries"""
        try:
            # Get most popular queries
            popular_queries = sorted(
                self.query_patterns.values(),
                key=lambda p: p.access_count * p.cost_impact,
                reverse=True
            )[:self.config.warm_cache_queries]
            
            warmed_count = 0
            failed_count = 0
            
            for pattern in popular_queries:
                try:
                    # Check if query is already cached
                    cache_key = f"warmed:{pattern.query_hash}"
                    if not self.redis_client.exists(cache_key):
                        # Simulate cache warming (in production, this would make actual API calls)
                        warm_data = {
                            'query_hash': pattern.query_hash,
                            'warmed_at': datetime.now().isoformat(),
                            'access_count': pattern.access_count,
                            'cost_impact': pattern.cost_impact
                        }
                        
                        # Store in cache with optimal TTL
                        self.redis_client.setex(
                            cache_key,
                            pattern.optimal_ttl or pattern.current_ttl,
                            json.dumps(warm_data)
                        )
                        warmed_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Failed to warm query {pattern.query_hash}: {e}")
                    failed_count += 1
            
            return {
                'warmed_queries': warmed_count,
                'failed_queries': failed_count,
                'total_popular_queries': len(popular_queries)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to warm popular queries: {e}")
            return {'warmed_queries': 0, 'failed_queries': 0}
    
    def _run_predictive_caching(self) -> Dict:
        """Run predictive caching based on patterns"""
        try:
            # Train or update prediction model
            self._update_prediction_model()
            
            if not self.prediction_model:
                return {'predicted_queries': 0, 'model_accuracy': 0}
            
            predicted_count = 0
            current_time = datetime.now()
            future_time = current_time + timedelta(hours=self.config.prediction_horizon_hours)
            
            # Make predictions for each query pattern
            for query_hash, pattern in self.query_patterns.items():
                try:
                    # Prepare features for prediction
                    features = self._extract_features(pattern, future_time)
                    
                    # Make prediction
                    prediction = self.prediction_model.predict([features])[0]
                    
                    # If prediction confidence is high and query not cached, cache it
                    if prediction > self.config.prediction_confidence_threshold:
                        cache_key = f"predicted:{query_hash}"
                        if not self.redis_client.exists(cache_key):
                            # Pre-cache the query
                            pred_data = {
                                'query_hash': query_hash,
                                'predicted_at': current_time.isoformat(),
                                'prediction_score': float(prediction),
                                'predicted_for': future_time.isoformat()
                            }
                            
                            self.redis_client.setex(
                                cache_key,
                                self.config.prediction_horizon_hours * 3600,
                                json.dumps(pred_data)
                            )
                            predicted_count += 1
                            
                except Exception as e:
                    logger.error(f"❌ Failed to predict for query {query_hash}: {e}")
            
            return {
                'predicted_queries': predicted_count,
                'model_accuracy': getattr(self.prediction_model, 'score_', 0.0),
                'total_patterns_analyzed': len(self.query_patterns)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to run predictive caching: {e}")
            return {'predicted_queries': 0}
    
    def _update_prediction_model(self):
        """Update the predictive model with recent data"""
        try:
            # Prepare training data from query patterns
            X = []  # Features
            y = []  # Target (access probability)
            
            for pattern in self.query_patterns.values():
                if pattern.access_count > 5:  # Only use patterns with sufficient data
                    # Create training examples for different times
                    for hour in range(24):
                        features = self._extract_features(pattern, datetime.now().replace(hour=hour))
                        target = pattern.seasonal_pattern[hour]
                        
                        X.append(features)
                        y.append(target)
            
            if len(X) >= 10:  # Minimum samples for training
                # Train simple linear regression model
                self.prediction_model = LinearRegression()
                self.prediction_model.fit(X, y)
                
                logger.info(f"✅ Prediction model updated with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"❌ Failed to update prediction model: {e}")
    
    def _extract_features(self, pattern: QueryPattern, target_time: datetime) -> List[float]:
        """Extract features for prediction model"""
        return [
            target_time.hour,  # Hour of day
            target_time.weekday(),  # Day of week
            min(1.0, pattern.access_count / 100.0),  # Normalized access frequency
            pattern.seasonal_pattern[target_time.hour]  # Current seasonal pattern
        ]
    
    def _apply_optimizations(self, ttl_improvements: List[Dict]) -> Dict:
        """Apply optimization recommendations"""
        try:
            applied_count = 0
            failed_count = 0
            
            for improvement in ttl_improvements:
                try:
                    query_hash = improvement['query_hash']
                    new_ttl = improvement['optimal_ttl']
                    
                    # Update TTL in pattern
                    if query_hash in self.query_patterns:
                        self.query_patterns[query_hash].current_ttl = new_ttl
                        applied_count += 1
                    
                    # In production, this would update actual cache entries
                    # For now, we'll just log the optimization
                    logger.debug(f"Applied TTL optimization: {query_hash} -> {new_ttl}s")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to apply optimization for {improvement.get('query_hash')}: {e}")
                    failed_count += 1
            
            return {
                'applied_optimizations': applied_count,
                'failed_optimizations': failed_count,
                'total_recommendations': len(ttl_improvements)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to apply optimizations: {e}")
            return {'applied_optimizations': 0, 'failed_optimizations': 0}
    
    def get_optimization_report(self) -> Dict:
        """Get comprehensive optimization report"""
        try:
            # Calculate current performance metrics
            total_patterns = len(self.query_patterns)
            avg_hit_rate = np.mean([p.hit_rate for p in self.query_patterns.values()]) if total_patterns > 0 else 0
            avg_response_time = np.mean([p.average_response_time for p in self.query_patterns.values()]) if total_patterns > 0 else 0
            total_cost_impact = sum([p.cost_impact for p in self.query_patterns.values()])
            
            # Get recent optimization results
            recent_results = list(self.optimization_results)[-10:] if self.optimization_results else []
            
            # Calculate improvement trends
            improvement_trend = []
            if len(recent_results) >= 2:
                for i in range(1, len(recent_results)):
                    prev_metrics = recent_results[i-1]['performance_metrics']
                    curr_metrics = recent_results[i]['performance_metrics']
                    
                    hit_rate_improvement = curr_metrics.get('overall_hit_rate', 0) - prev_metrics.get('overall_hit_rate', 0)
                    improvement_trend.append(hit_rate_improvement)
            
            return {
                'optimization_status': 'active' if self.optimization_active else 'inactive',
                'last_optimization': self.last_optimization.isoformat() if self.last_optimization != datetime.min else None,
                'current_performance': {
                    'total_query_patterns': total_patterns,
                    'average_hit_rate': round(avg_hit_rate * 100, 1),
                    'average_response_time_ms': round(avg_response_time, 1),
                    'total_cost_impact_usd': round(total_cost_impact, 2)
                },
                'targets': {
                    'target_hit_rate': self.config.target_hit_rate,
                    'target_response_time_p95': self.config.target_response_time_p95,
                    'target_cost_savings': self.config.target_cost_savings
                },
                'recent_optimizations': len(recent_results),
                'improvement_trend': improvement_trend,
                'optimization_results': recent_results
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate optimization report: {e}")
            return {}

def create_cache_optimizer(redis_host: str = 'localhost', redis_port: int = 6379) -> AdvancedCacheOptimizer:
    """Create and configure a cache optimizer instance"""
    
    # Connect to Redis
    redis_client = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    
    # Configure optimization for production targets
    config = CacheOptimizationConfig(
        target_hit_rate=85.0,           # Target 85% cache hit rate
        target_response_time_p95=400.0, # Target P95 < 400ms
        target_cost_savings=200.0,      # Target $200+ monthly savings
        
        # Aggressive optimization settings
        min_ttl_seconds=300,            # 5 minutes minimum
        max_ttl_seconds=86400,          # 24 hours maximum
        ttl_adjustment_factor=1.3,      # 30% adjustment factor
        
        # Cache warming every 4 hours
        warm_cache_frequency_hours=4,
        warm_cache_queries=75,          # Warm top 75 queries
        
        # Predictive caching enabled
        predictive_enabled=True,
        prediction_horizon_hours=2,     # Predict 2 hours ahead
        prediction_confidence_threshold=0.75  # 75% confidence threshold
    )
    
    optimizer = AdvancedCacheOptimizer(redis_client, config)
    
    return optimizer

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Istanbul Advanced Cache Optimizer")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--interval", type=int, default=30, help="Optimization interval in minutes")
    parser.add_argument("--duration", type=int, default=3600, help="Optimization duration in seconds (0 for infinite)")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create and start cache optimizer
    optimizer = create_cache_optimizer(args.redis_host, args.redis_port)
    optimizer.start_optimization(args.interval)
    
    try:
        if args.duration > 0:
            print(f"🚀 Cache optimization started for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("🚀 Cache optimization started indefinitely. Press Ctrl+C to stop...")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping optimization...")
    finally:
        optimizer.stop_optimization()
        print("✅ Cache optimization stopped successfully")
