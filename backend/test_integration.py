#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite
===================================

Tests all components of the integrated cache system:
- Time-aware caching
- TTL optimization
- Production monitoring
- Cache warming
- API integration
"""

import asyncio
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationTestSuite:
    """Comprehensive test suite for integrated cache system"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
        
    def log_test_result(self, test_name: str, success: bool, details: str = "", duration_ms: float = 0):
        """Log test result"""
        result = {
            'test_name': test_name,
            'success': success,
            'details': details,
            'duration_ms': duration_ms,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        duration_str = f" ({duration_ms:.0f}ms)" if duration_ms > 0 else ""
        print(f"{status} {test_name}{duration_str}")
        if details:
            print(f"    {details}")
    
    async def test_google_api_integration(self):
        """Test Google API integration with field optimization"""
        test_start = time.time()
        
        try:
            from google_api_integration import (
                GoogleApiFieldOptimizer, 
                QueryIntent, 
                search_restaurants_with_optimization
            )
            
            # Test 1: Component initialization
            optimizer = GoogleApiFieldOptimizer()
            self.log_test_result(
                "Google API Optimizer Initialization", 
                True, 
                f"Initialized with {len(optimizer.field_costs)} field mappings"
            )
            
            # Test 2: Query intent classification
            test_queries = [
                ("quick restaurant nearby", QueryIntent.QUICK_RECOMMENDATION),
                ("tell me about Pandeli restaurant", QueryIntent.DETAILED_INFO),
                ("how to get to Hamdi Restaurant", QueryIntent.NAVIGATION),
                ("vegetarian restaurants", QueryIntent.DINING_DECISION)
            ]
            
            for query, expected_intent in test_queries:
                detected_intent = optimizer.classify_query_intent(query)
                self.log_test_result(
                    f"Intent Classification: '{query[:30]}...'",
                    detected_intent == expected_intent,
                    f"Expected: {expected_intent.value}, Got: {detected_intent.value}"
                )
            
            # Test 3: Field optimization
            for intent in [QueryIntent.BASIC_SEARCH, QueryIntent.DETAILED_INFO, QueryIntent.QUICK_RECOMMENDATION]:
                fields = optimizer.get_optimized_fields(intent)
                cost = optimizer.calculate_request_cost(fields)
                
                self.log_test_result(
                    f"Field Optimization: {intent.value}",
                    len(fields) > 0 and cost > 0,
                    f"Fields: {len(fields)}, Cost score: {cost}"
                )
            
            # Test 4: Mock restaurant search (without API key)
            try:
                # This will use mock data if no API key is available
                result = search_restaurants_with_optimization(
                    "Turkish restaurants in Sultanahmet",
                    budget_mode=True
                )
                
                self.log_test_result(
                    "Restaurant Search (Mock/Real)",
                    result.get('success', False),
                    f"Found {len(result.get('restaurants', []))} restaurants"
                )
                
            except Exception as e:
                self.log_test_result(
                    "Restaurant Search",
                    False,
                    f"Search failed: {str(e)}"
                )
            
        except ImportError as e:
            self.log_test_result(
                "Google API Integration Import",
                False,
                f"Import failed: {str(e)}"
            )
        except Exception as e:
            self.log_test_result(
                "Google API Integration",
                False,
                f"Test failed: {str(e)}"
            )
        
        duration = (time.time() - test_start) * 1000
        return duration
    
    async def test_time_aware_cache(self):
        """Test time-aware caching system"""
        test_start = time.time()
        
        try:
            from google_api_integration import TimeAwareCacheManager, QueryIntent
            
            # Test 1: Cache manager initialization
            cache_manager = TimeAwareCacheManager()
            self.log_test_result(
                "Time-Aware Cache Manager Init",
                True,
                f"Initialized with {len(cache_manager.cache_strategies)} cache strategies"
            )
            
            # Test 2: Cache type classification
            test_cases = [
                ("restaurant open now", "real_time_status"),
                ("best Turkish restaurants", "restaurant_basic_info"),
                ("restaurant prices", "live_pricing"),
                ("vegetarian restaurants in Beyoğlu", "preference_search")
            ]
            
            for query, expected_type in test_cases:
                intent = QueryIntent.BASIC_SEARCH
                cache_type = cache_manager.classify_cache_type(query, intent, [])
                
                self.log_test_result(
                    f"Cache Type Classification: '{query[:25]}...'",
                    cache_type == expected_type,
                    f"Expected: {expected_type}, Got: {cache_type}"
                )
            
            # Test 3: TTL calculation
            for cache_type in cache_manager.cache_strategies.keys():
                ttl = cache_manager.get_time_aware_ttl(cache_type)
                strategy = cache_manager.cache_strategies[cache_type]
                
                self.log_test_result(
                    f"TTL Calculation: {cache_type}",
                    60 <= ttl <= 1209600,  # Between 1 minute and 14 days
                    f"TTL: {ttl}s (base: {strategy.ttl_seconds}s)"
                )
            
            # Test 4: Cache key generation
            test_query = "Turkish restaurants in Sultanahmet"
            cache_key = cache_manager.generate_cache_key(
                test_query, "Istanbul, Turkey", QueryIntent.BASIC_SEARCH, ['name', 'rating']
            )
            
            self.log_test_result(
                "Cache Key Generation",
                len(cache_key) > 10 and ":" in cache_key,
                f"Generated key: {cache_key[:50]}..."
            )
            
            # Test 5: Cache analytics
            analytics = cache_manager.get_cache_analytics()
            self.log_test_result(
                "Cache Analytics",
                'overall_hit_rate_percent' in analytics,
                f"Analytics keys: {list(analytics.keys())}"
            )
            
        except ImportError as e:
            self.log_test_result(
                "Time-Aware Cache Import",
                False,
                f"Import failed: {str(e)}"
            )
        except Exception as e:
            self.log_test_result(
                "Time-Aware Cache",
                False,
                f"Test failed: {str(e)}"
            )
        
        duration = (time.time() - test_start) * 1000
        return duration
    
    async def test_ttl_optimization(self):
        """Test TTL fine-tuning system"""
        test_start = time.time()
        
        try:
            from ttl_fine_tuning import (
                TTLOptimizer, 
                get_optimized_ttl, 
                record_cache_access,
                get_ttl_optimization_report
            )
            
            # Test 1: TTL optimizer initialization
            optimizer = TTLOptimizer()
            self.log_test_result(
                "TTL Optimizer Initialization",
                len(optimizer.ttl_configs) > 0,
                f"Initialized with {len(optimizer.ttl_configs)} cache type configs"
            )
            
            # Test 2: TTL optimization for different cache types
            for cache_type in optimizer.ttl_configs.keys():
                ttl = get_optimized_ttl(cache_type)
                config = optimizer.ttl_configs[cache_type]
                
                self.log_test_result(
                    f"TTL Optimization: {cache_type}",
                    config.min_ttl_seconds <= ttl <= config.max_ttl_seconds,
                    f"TTL: {ttl}s (range: {config.min_ttl_seconds}-{config.max_ttl_seconds}s)"
                )
            
            # Test 3: Cache access recording
            test_cache_type = "restaurant_basic_info"
            
            # Simulate cache accesses
            for i in range(10):
                is_hit = i % 3 == 0  # 33% hit rate
                response_time = 150.0 if is_hit else 500.0
                record_cache_access(test_cache_type, is_hit, response_time)
            
            self.log_test_result(
                "Cache Access Recording",
                True,
                f"Recorded 10 cache accesses for {test_cache_type}"
            )
            
            # Test 4: Optimization report
            report = get_ttl_optimization_report()
            self.log_test_result(
                "TTL Optimization Report",
                'cache_types' in report and 'summary' in report,
                f"Report contains {len(report.get('cache_types', {}))} cache types"
            )
            
        except ImportError as e:
            self.log_test_result(
                "TTL Optimization Import",
                False,
                f"Import failed: {str(e)}"
            )
        except Exception as e:
            self.log_test_result(
                "TTL Optimization",
                False,
                f"Test failed: {str(e)}"
            )
        
        duration = (time.time() - test_start) * 1000
        return duration
    
    async def test_integrated_cache_system(self):
        """Test integrated cache system"""
        test_start = time.time()
        
        try:
            from integrated_cache_system import (
                IntegratedCacheSystem,
                search_restaurants_with_integrated_cache,
                get_integrated_analytics
            )
            
            # Test 1: Integrated system initialization
            integrated_system = IntegratedCacheSystem()
            self.log_test_result(
                "Integrated Cache System Init",
                integrated_system is not None,
                "System components initialized"
            )
            
            # Test 2: Integrated restaurant search
            try:
                result = await search_restaurants_with_integrated_cache(
                    query="Turkish restaurants in Sultanahmet",
                    location="Istanbul, Turkey"
                )
                
                self.log_test_result(
                    "Integrated Restaurant Search",
                    result.get('success', False),
                    f"Search result keys: {list(result.keys())}"
                )
                
                # Check performance metrics
                if 'cache_performance' in result:
                    perf = result['cache_performance']
                    self.log_test_result(
                        "Performance Metrics",
                        'cache_hit' in perf and 'response_time_ms' in perf,
                        f"Cache hit: {perf.get('cache_hit')}, Time: {perf.get('response_time_ms', 0):.0f}ms"
                    )
                
            except Exception as e:
                self.log_test_result(
                    "Integrated Restaurant Search",
                    False,
                    f"Search failed: {str(e)}"
                )
            
            # Test 3: System analytics
            try:
                analytics = get_integrated_analytics()
                
                expected_keys = ['cache_optimization', 'time_aware_cache', 'production_monitoring']
                has_all_keys = all(key in analytics for key in expected_keys)
                
                self.log_test_result(
                    "Integrated Analytics",
                    has_all_keys,
                    f"Analytics keys: {list(analytics.keys())}"
                )
                
            except Exception as e:
                self.log_test_result(
                    "Integrated Analytics",
                    False,
                    f"Analytics failed: {str(e)}"
                )
            
            # Test 4: Cache warming
            try:
                warming_result = await integrated_system.warm_cache_for_query(
                    "vegetarian restaurants in Beyoğlu"
                )
                
                self.log_test_result(
                    "Cache Warming",
                    warming_result in [True, False],  # Should return boolean
                    f"Warming result: {warming_result}"
                )
                
            except Exception as e:
                self.log_test_result(
                    "Cache Warming",
                    False,
                    f"Warming failed: {str(e)}"
                )
            
        except ImportError as e:
            self.log_test_result(
                "Integrated Cache System Import",
                False,
                f"Import failed: {str(e)}"
            )
        except Exception as e:
            self.log_test_result(
                "Integrated Cache System",
                False,
                f"Test failed: {str(e)}"
            )
        
        duration = (time.time() - test_start) * 1000
        return duration
    
    async def test_unified_ai_integration(self):
        """Test unified AI system integration"""
        test_start = time.time()
        
        try:
            from unified_ai_system import UnifiedAISystem
            from database import SessionLocal
            
            # Test 1: Check if unified AI system can be imported
            self.log_test_result(
                "Unified AI System Import",
                True,
                "Successfully imported UnifiedAISystem"
            )
            
            # Test 2: Mock initialization (without database)
            try:
                # This would normally require a database session
                # We'll just test that the class exists and has the right methods
                
                expected_methods = [
                    'search_restaurants_enhanced',
                    'get_cache_performance_analytics',
                    'warm_cache_for_popular_queries'
                ]
                
                has_all_methods = all(hasattr(UnifiedAISystem, method) for method in expected_methods)
                
                self.log_test_result(
                    "Unified AI System Methods",
                    has_all_methods,
                    f"Has enhanced methods: {expected_methods}"
                )
                
            except Exception as e:
                self.log_test_result(
                    "Unified AI System Methods",
                    False,
                    f"Method check failed: {str(e)}"
                )
            
        except ImportError as e:
            self.log_test_result(
                "Unified AI Integration Import",
                False,
                f"Import failed: {str(e)}"
            )
        except Exception as e:
            self.log_test_result(
                "Unified AI Integration",
                False,
                f"Test failed: {str(e)}"
            )
        
        duration = (time.time() - test_start) * 1000
        return duration
    
    async def test_monitoring_routes(self):
        """Test cache monitoring routes"""
        test_start = time.time()
        
        try:
            from routes.cache_monitoring import router
            
            # Test 1: Router import
            self.log_test_result(
                "Cache Monitoring Routes Import",
                router is not None,
                "Successfully imported monitoring router"
            )
            
            # Test 2: Check router configuration
            self.log_test_result(
                "Router Configuration",
                router.prefix == "/api/cache",
                f"Router prefix: {router.prefix}"
            )
            
            # Test 3: Check available routes
            route_paths = [route.path for route in router.routes]
            expected_paths = ['/analytics', '/performance', '/health', '/ttl/report']
            
            has_expected_routes = any(
                any(expected in path for expected in expected_paths)
                for path in route_paths
            )
            
            self.log_test_result(
                "Monitoring Route Paths",
                has_expected_routes,
                f"Available routes: {len(route_paths)}"
            )
            
        except ImportError as e:
            self.log_test_result(
                "Cache Monitoring Routes Import",
                False,
                f"Import failed: {str(e)}"
            )
        except Exception as e:
            self.log_test_result(
                "Cache Monitoring Routes",
                False,
                f"Test failed: {str(e)}"
            )
        
        duration = (time.time() - test_start) * 1000
        return duration
    
    async def test_performance_benchmark(self):
        """Performance benchmark test"""
        test_start = time.time()
        
        try:
            # Test multiple queries to measure performance
            test_queries = [
                "best Turkish restaurants in Sultanahmet",
                "vegetarian restaurants in Beyoğlu", 
                "seafood restaurants near Bosphorus",
                "quick lunch places in Kadıköy",
                "romantic dinner restaurants Istanbul"
            ]
            
            response_times = []
            
            for query in test_queries:
                query_start = time.time()
                
                try:
                    from integrated_cache_system import search_restaurants_with_integrated_cache
                    
                    result = await search_restaurants_with_integrated_cache(
                        query=query,
                        location="Istanbul, Turkey"
                    )
                    
                    query_time = (time.time() - query_start) * 1000
                    response_times.append(query_time)
                    
                    self.log_test_result(
                        f"Performance: '{query[:25]}...'",
                        result.get('success', False),
                        f"Response time: {query_time:.0f}ms"
                    )
                    
                except Exception as e:
                    self.log_test_result(
                        f"Performance: '{query[:25]}...'",
                        False,
                        f"Query failed: {str(e)}"
                    )
            
            # Calculate performance metrics
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                min_response_time = min(response_times)
                
                self.log_test_result(
                    "Performance Benchmark Summary",
                    avg_response_time < 2000,  # Should be under 2 seconds average
                    f"Avg: {avg_response_time:.0f}ms, Min: {min_response_time:.0f}ms, Max: {max_response_time:.0f}ms"
                )
            
        except ImportError as e:
            self.log_test_result(
                "Performance Benchmark Import",
                False,
                f"Import failed: {str(e)}"
            )
        except Exception as e:
            self.log_test_result(
                "Performance Benchmark",
                False,
                f"Test failed: {str(e)}"
            )
        
        duration = (time.time() - test_start) * 1000
        return duration
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        total_duration = (time.time() - self.start_time) * 1000
        
        passed_tests = [r for r in self.test_results if r['success']]
        failed_tests = [r for r in self.test_results if not r['success']]
        
        print("\n" + "=" * 80)
        print("🧪 INTEGRATION TEST RESULTS")
        print("=" * 80)
        
        print(f"📊 SUMMARY:")
        print(f"   Total Tests: {len(self.test_results)}")
        print(f"   Passed: {len(passed_tests)} ✅")
        print(f"   Failed: {len(failed_tests)} ❌")
        print(f"   Success Rate: {(len(passed_tests) / len(self.test_results) * 100):.1f}%")
        print(f"   Total Duration: {total_duration:.0f}ms")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   • {test['test_name']}: {test['details']}")
        
        print(f"\n📈 COMPONENT STATUS:")
        
        components = {
            'Google API Integration': any('Google API' in t['test_name'] for t in passed_tests),
            'Time-Aware Cache': any('Time-Aware Cache' in t['test_name'] or 'TTL' in t['test_name'] for t in passed_tests),
            'Integrated Cache System': any('Integrated' in t['test_name'] for t in passed_tests),
            'Unified AI Integration': any('Unified AI' in t['test_name'] for t in passed_tests),
            'Monitoring Routes': any('Monitoring' in t['test_name'] for t in passed_tests),
            'Performance': any('Performance' in t['test_name'] for t in passed_tests)
        }
        
        for component, status in components.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if len(failed_tests) == 0:
            print("   ✅ All tests passed! System is ready for production.")
        elif len(failed_tests) <= 3:
            print("   ⚠️  Minor issues detected. Review failed tests before production.")
        else:
            print("   ❌ Multiple issues detected. Address failures before deployment.")
        
        # Save detailed report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'success_rate': (len(passed_tests) / len(self.test_results) * 100),
                'total_duration_ms': total_duration
            },
            'test_results': self.test_results,
            'component_status': components
        }
        
        with open('integration_test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: integration_test_report.json")
        print("=" * 80)
        
        return len(failed_tests) == 0

async def main():
    """Run comprehensive integration tests"""
    print("🚀 Starting AI Istanbul Integration Test Suite")
    print("=" * 80)
    
    test_suite = IntegrationTestSuite()
    
    # Run all test categories
    test_categories = [
        ("Google API Integration", test_suite.test_google_api_integration),
        ("Time-Aware Cache", test_suite.test_time_aware_cache),
        ("TTL Optimization", test_suite.test_ttl_optimization),
        ("Integrated Cache System", test_suite.test_integrated_cache_system),
        ("Unified AI Integration", test_suite.test_unified_ai_integration),
        ("Monitoring Routes", test_suite.test_monitoring_routes),
        ("Performance Benchmark", test_suite.test_performance_benchmark)
    ]
    
    for category_name, test_function in test_categories:
        print(f"\n🧪 Testing {category_name}...")
        try:
            duration = await test_function()
            print(f"   Completed in {duration:.0f}ms")
        except Exception as e:
            print(f"   ❌ Category failed: {str(e)}")
    
    # Generate final report
    success = test_suite.generate_test_report()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
