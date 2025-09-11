#!/usr/bin/env python3
"""
Load Testing Script for AIstanbul Chatbot
Tests concurrent user interactions and system robustness under load.
"""

import asyncio
import aiohttp
import json
import time
import random
from typing import List, Dict, Any
from dataclasses import dataclass
from statistics import mean, median
import sys

@dataclass
class TestResult:
    user_id: int
    query: str
    response_time: float
    status_code: int
    success: bool
    response_length: int
    error_message: str = ""

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[TestResult] = []
        
        # Diverse test queries simulating real user behavior
        self.test_queries = [
            # Restaurant queries
            "restaurants in Kadıköy",
            "best Turkish food in Sultanahmet", 
            "seafood restaurants Galata",
            "romantic dinner spots Beyoğlu",
            "family friendly restaurants",
            "cheap eats in Taksim",
            
            # Attraction queries
            "museums in Istanbul",
            "places to visit Sultanahmet",
            "things to do in Galata",
            "Hagia Sophia information",
            "Blue Mosque details",
            "attractions for families",
            
            # Location queries
            "best neighborhoods Istanbul",
            "what to do in Kadıköy",
            "Sultanahmet district guide",
            "Asian side attractions",
            "European side restaurants",
            
            # Special interest queries
            "romantic places couples",
            "rainy day activities",
            "budget friendly attractions",
            "nightlife in Beyoğlu",
            "shopping Grand Bazaar",
            "Turkish bath experience",
            
            # Transportation queries
            "how to get around Istanbul",
            "metro system guide",
            "ferry schedules",
            "taxi vs metro",
            "airport to city center",
            
            # General queries
            "hello",
            "good morning",
            "weather in Istanbul",
            "best time to visit",
            "Turkish culture",
            "Ottoman history",
            
            # Challenging/typo queries
            "resturants kadikoy",
            "musuems istambul", 
            "gud plases to visit",
            "whre to eat",
            "familey atraction",
            "romantik spots",
            "cheep food",
            "",  # Empty query
            "???",  # Invalid query
            "abcdefghijklmnop",  # Nonsense query
        ]
    
    async def make_request(self, session: aiohttp.ClientSession, user_id: int, query: str) -> TestResult:
        """Make a single API request and measure performance"""
        start_time = time.time()
        
        try:
            payload = {"message": query}
            async with session.post(
                f"{self.base_url}/ai",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    response_data = await response.json()
                    response_message = response_data.get("message", "")
                    
                    return TestResult(
                        user_id=user_id,
                        query=query,
                        response_time=response_time,
                        status_code=response.status,
                        success=True,
                        response_length=len(response_message),
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        user_id=user_id,
                        query=query,
                        response_time=response_time,
                        status_code=response.status,
                        success=False,
                        response_length=0,
                        error_message=f"HTTP {response.status}: {error_text[:100]}"
                    )
                    
        except asyncio.TimeoutError:
            return TestResult(
                user_id=user_id,
                query=query,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                response_length=0,
                error_message="Request timeout"
            )
        except Exception as e:
            return TestResult(
                user_id=user_id,
                query=query,
                response_time=time.time() - start_time,
                status_code=0,
                success=False,
                response_length=0,
                error_message=str(e)
            )
    
    async def simulate_user(self, session: aiohttp.ClientSession, user_id: int, queries_per_user: int) -> List[TestResult]:
        """Simulate a single user making multiple requests"""
        user_results = []
        
        for i in range(queries_per_user):
            # Random delay between requests (1-5 seconds) to simulate real user behavior
            if i > 0:
                await asyncio.sleep(random.uniform(1, 5))
            
            # Select a random query
            query = random.choice(self.test_queries)
            
            print(f"User {user_id:2d} | Request {i+1:2d} | Query: '{query[:30]:<30}' | ", end="")
            
            result = await self.make_request(session, user_id, query)
            user_results.append(result)
            
            status = "✅ OK" if result.success else "❌ FAIL"
            print(f"{status} | {result.response_time:.2f}s | {result.response_length} chars")
            
            if not result.success:
                print(f"    Error: {result.error_message}")
        
        return user_results
    
    async def run_load_test(self, concurrent_users: int = 10, queries_per_user: int = 5) -> Dict[str, Any]:
        """Run the load test with specified parameters"""
        print(f"\n🚀 Starting Load Test")
        print(f"📊 Concurrent Users: {concurrent_users}")
        print(f"📊 Queries per User: {queries_per_user}")
        print(f"📊 Total Requests: {concurrent_users * queries_per_user}")
        print(f"📊 Server: {self.base_url}")
        print("=" * 70)
        
        start_time = time.time()
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=concurrent_users * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            # Create tasks for all users
            tasks = []
            for user_id in range(concurrent_users):
                task = self.simulate_user(session, user_id, queries_per_user)
                tasks.append(task)
            
            # Run all users concurrently
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results
            for user_results in all_results:
                if isinstance(user_results, list):
                    self.results.extend(user_results)
                else:
                    print(f"User task failed: {user_results}")
        
        total_time = time.time() - start_time
        
        return self.analyze_results(total_time)
    
    def analyze_results(self, total_time: float) -> Dict[str, Any]:
        """Analyze test results and generate report"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time for r in successful_results]
        response_lengths = [r.response_length for r in successful_results]
        
        # Calculate metrics
        success_rate = len(successful_results) / len(self.results) * 100
        
        analysis = {
            "summary": {
                "total_requests": len(self.results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": round(success_rate, 1),
                "total_test_time": round(total_time, 2),
                "requests_per_second": round(len(self.results) / total_time, 2)
            },
            "performance": {
                "avg_response_time": round(mean(response_times), 3) if response_times else 0,
                "median_response_time": round(median(response_times), 3) if response_times else 0,
                "min_response_time": round(min(response_times), 3) if response_times else 0,
                "max_response_time": round(max(response_times), 3) if response_times else 0,
                "avg_response_length": round(mean(response_lengths), 0) if response_lengths else 0
            },
            "errors": {}
        }
        
        # Analyze error patterns
        if failed_results:
            error_counts = {}
            for result in failed_results:
                error_type = result.error_message.split(':')[0] if result.error_message else "Unknown"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            analysis["errors"] = error_counts
        
        return analysis
    
    def print_report(self, analysis: Dict[str, Any]):
        """Print a comprehensive test report"""
        print("\n" + "="*70)
        print("📊 LOAD TEST RESULTS")
        print("="*70)
        
        # Summary
        s = analysis["summary"]
        print(f"📈 Total Requests: {s['total_requests']}")
        print(f"✅ Successful: {s['successful_requests']} ({s['success_rate']}%)")
        print(f"❌ Failed: {s['failed_requests']}")
        print(f"⏱️  Total Time: {s['total_test_time']}s")
        print(f"🚀 RPS (Requests/sec): {s['requests_per_second']}")
        
        # Performance
        print(f"\n⚡ PERFORMANCE METRICS")
        p = analysis["performance"]
        print(f"📊 Average Response Time: {p['avg_response_time']}s")
        print(f"📊 Median Response Time: {p['median_response_time']}s")
        print(f"📊 Min Response Time: {p['min_response_time']}s")
        print(f"📊 Max Response Time: {p['max_response_time']}s")
        print(f"📊 Average Response Length: {p['avg_response_length']} chars")
        
        # Errors
        if analysis["errors"]:
            print(f"\n❌ ERROR ANALYSIS")
            for error_type, count in analysis["errors"].items():
                print(f"  • {error_type}: {count} occurrences")
        
        # Overall Assessment
        print(f"\n🎯 ASSESSMENT")
        success_rate = s['success_rate']
        avg_time = p['avg_response_time']
        
        if success_rate >= 95 and avg_time <= 3.0:
            print("🏆 EXCELLENT: System handles load very well")
        elif success_rate >= 90 and avg_time <= 5.0:
            print("✅ GOOD: System performs well under load")
        elif success_rate >= 80 and avg_time <= 10.0:
            print("⚠️  ACCEPTABLE: Some performance degradation")
        else:
            print("❌ POOR: System struggles under load")
        
        print("="*70)

async def main():
    """Main function to run load tests"""
    if len(sys.argv) < 3:
        print("Usage: python test_load_concurrent.py <concurrent_users> <queries_per_user>")
        print("Example: python test_load_concurrent.py 10 5")
        concurrent_users = 5
        queries_per_user = 3
        print(f"Using defaults: {concurrent_users} users, {queries_per_user} queries each")
    else:
        concurrent_users = int(sys.argv[1])
        queries_per_user = int(sys.argv[2])
    
    # Run different test scenarios
    test_scenarios = [
        {"name": "Light Load", "users": concurrent_users, "queries": queries_per_user},
    ]
    
    # If running heavy tests, add more scenarios
    if concurrent_users >= 10:
        test_scenarios.extend([
            {"name": "Medium Load", "users": concurrent_users * 2, "queries": queries_per_user},
            {"name": "Heavy Load", "users": concurrent_users * 3, "queries": max(1, queries_per_user // 2)},
        ])
    
    all_test_results = []
    
    for scenario in test_scenarios:
        print(f"\n🎯 Running {scenario['name']} Test...")
        
        tester = LoadTester()
        analysis = await tester.run_load_test(
            concurrent_users=scenario['users'],
            queries_per_user=scenario['queries']
        )
        
        analysis['scenario'] = scenario['name']
        all_test_results.append(analysis)
        
        tester.print_report(analysis)
        
        # Brief pause between scenarios
        if len(test_scenarios) > 1:
            print("\n⏳ Waiting 10 seconds before next test...")
            await asyncio.sleep(10)
    
    # Final summary
    if len(all_test_results) > 1:
        print(f"\n📋 FINAL SUMMARY")
        print("="*50)
        for result in all_test_results:
            scenario = result['scenario']
            success_rate = result['summary']['success_rate']
            avg_time = result['performance']['avg_response_time']
            rps = result['summary']['requests_per_second']
            print(f"{scenario:12} | Success: {success_rate:5.1f}% | Avg: {avg_time:5.2f}s | RPS: {rps:5.1f}")

if __name__ == "__main__":
    # Run the load test
    asyncio.run(main())
