#!/usr/bin/env python3
"""
Comprehensive AI Istanbul Chatbot Test Suite
Tests all areas with challenging inputs and validates answer correctness
"""

import json
import requests
import time
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class ChatbotTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        
    def make_request(self, query: str, session_id: str = "test") -> Dict:
        """Make a request to the chatbot API"""
        try:
            response = requests.post(
                f"{self.base_url}/ai",
                json={"query": query, "session_id": session_id},
                timeout=30
            )
            if response.status_code == 200:
                return {
                    "status": "success",
                    "response": response.json().get("message", ""),
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "response": ""
                }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "response": ""
            }
    
    def validate_response(self, response: str, expected_keywords: List[str], 
                         forbidden_keywords: Optional[List[str]] = None, 
                         min_length: int = 20) -> Tuple[bool, List[str]]:
        """Validate if response contains expected content and avoids forbidden content"""
        issues = []
        
        # Check minimum length
        if len(response) < min_length:
            issues.append(f"Response too short ({len(response)} chars)")
        
        # Check for expected keywords (at least 50% should be present)
        found_keywords = [kw for kw in expected_keywords if kw.lower() in response.lower()]
        if len(found_keywords) < len(expected_keywords) * 0.5:
            issues.append(f"Missing key content. Found: {found_keywords}")
        
        # Check for forbidden keywords
        if forbidden_keywords:
            found_forbidden = [kw for kw in forbidden_keywords if kw.lower() in response.lower()]
            if found_forbidden:
                issues.append(f"Contains forbidden content: {found_forbidden}")
        
        # Check for common error patterns
        error_patterns = [
            "sorry, i couldn't",
            "i don't understand",
            "error occurred",
            "something went wrong",
            "unable to process"
        ]
        found_errors = [pattern for pattern in error_patterns if pattern in response.lower()]
        if found_errors:
            issues.append(f"Contains error patterns: {found_errors}")
        
        return len(issues) == 0, issues

    def test_restaurant_queries(self):
        """Test restaurant-related queries with challenging inputs"""
        print("\n🍽️  Testing Restaurant Queries")
        print("-" * 40)
        
        test_cases = [
            # Basic restaurant queries
            {
                "query": "restaurants in beyoglu",
                "expected": ["restaurant", "beyoglu", "dining", "food"],
                "forbidden": ["museum", "transport", "metro"],
                "description": "Basic restaurant query"
            },
            {
                "query": "where to eat in sultanahmet",
                "expected": ["restaurant", "sultanahmet", "food", "eat"],
                "forbidden": ["museum", "attraction"],
                "description": "Alternative phrasing for restaurants"
            },
            # Misspelled queries
            {
                "query": "resturants in galata",
                "expected": ["restaurant", "galata", "dining"],
                "forbidden": ["museum", "transport"],
                "description": "Misspelled restaurant query"
            },
            {
                "query": "gud restarnts in taksim",
                "expected": ["restaurant", "taksim", "food"],
                "forbidden": ["museum"],
                "description": "Multiple misspellings"
            },
            # Mixed content queries (tricky)
            {
                "query": "restaurants near hagia sophia",
                "expected": ["restaurant", "hagia sophia", "near"],
                "forbidden": ["museum opening hours", "entrance fee"],
                "description": "Restaurant near landmark (should not give museum info)"
            },
            {
                "query": "food at topkapi palace",
                "expected": ["restaurant", "food", "topkapi"],
                "forbidden": ["museum ticket", "palace history"],
                "description": "Food near palace (should focus on restaurants)"
            },
            # Specific cuisine queries
            {
                "query": "turkish food in kadikoy",
                "expected": ["restaurant", "turkish", "kadikoy", "food"],
                "forbidden": ["museum", "transport"],
                "description": "Specific cuisine request"
            },
            {
                "query": "vegetarian restaurants istanbul",
                "expected": ["restaurant", "vegetarian", "istanbul"],
                "forbidden": ["museum", "transport"],
                "description": "Dietary restriction query"
            },
            # Problematic inputs
            {
                "query": "restaurants restaurants restaurants",
                "expected": ["restaurant", "istanbul"],
                "forbidden": ["error", "unable"],
                "description": "Repeated words"
            },
            {
                "query": "restrnt",
                "expected": ["restaurant", "istanbul"],
                "forbidden": ["museum", "transport"],
                "description": "Heavily abbreviated"
            }
        ]
        
        for test_case in test_cases:
            self.run_single_test("Restaurant", test_case)
    
    def test_transportation_queries(self):
        """Test transportation queries with edge cases"""
        print("\n🚇 Testing Transportation Queries")
        print("-" * 40)
        
        test_cases = [
            # Route queries
            {
                "query": "how can I go kadikoy from beyoglu",
                "expected": ["ferry", "metro", "transport", "kadikoy", "beyoglu"],
                "forbidden": ["restaurant", "museum"],
                "description": "Direct route query"
            },
            {
                "query": "from sultanahmet to taksim",
                "expected": ["metro", "tram", "transport", "sultanahmet", "taksim"],
                "forbidden": ["restaurant", "museum opening"],
                "description": "Route with landmarks"
            },
            # Transportation mode queries
            {
                "query": "metro to airport",
                "expected": ["metro", "airport", "m1", "transport"],
                "forbidden": ["restaurant", "museum"],
                "description": "Specific transport mode"
            },
            {
                "query": "ferry routes istanbul",
                "expected": ["ferry", "routes", "bosphorus"],
                "forbidden": ["restaurant menu", "museum ticket"],
                "description": "Ferry information"
            },
            # Practical questions
            {
                "query": "istanbul metro cost",
                "expected": ["metro", "cost", "istanbulkart", "price"],
                "forbidden": ["restaurant", "museum"],
                "description": "Cost information"
            },
            {
                "query": "where to buy istanbulkart",
                "expected": ["istanbulkart", "buy", "metro", "station"],
                "forbidden": ["restaurant", "museum"],
                "description": "Card purchase info"
            },
            # Misspelled transport queries
            {
                "query": "metroe from kadıköy",
                "expected": ["metro", "kadikoy", "transport"],
                "forbidden": ["restaurant", "museum"],
                "description": "Misspelled metro"
            },
            {
                "query": "how go from galata to uskudar",
                "expected": ["ferry", "metro", "transport", "galata", "uskudar"],
                "forbidden": ["restaurant", "museum"],
                "description": "Simplified grammar"
            },
            # Confusing queries
            {
                "query": "transport museum istanbul",
                "expected": ["transport", "museum", "rahmi"],
                "forbidden": ["metro route", "ferry schedule"],
                "description": "Transport museum (should give museum info, not transport routes)"
            },
            {
                "query": "how to get to transport",
                "expected": ["transport", "metro", "bus"],
                "forbidden": ["museum", "restaurant"],
                "description": "Ambiguous transport query"
            }
        ]
        
        for test_case in test_cases:
            self.run_single_test("Transportation", test_case)
    
    def test_museum_queries(self):
        """Test museum queries with various complexities"""
        print("\n🏛️  Testing Museum Queries")
        print("-" * 40)
        
        test_cases = [
            # Basic museum queries
            {
                "query": "museums in istanbul",
                "expected": ["museum", "hagia sophia", "topkapi", "istanbul"],
                "forbidden": ["restaurant", "metro route"],
                "description": "General museum query"
            },
            {
                "query": "best museums to visit",
                "expected": ["museum", "topkapi", "hagia sophia", "visit"],
                "forbidden": ["restaurant", "transport"],
                "description": "Museum recommendations"
            },
            # Specific museums
            {
                "query": "topkapi palace",
                "expected": ["topkapi", "palace", "ottoman", "museum"],
                "forbidden": ["restaurant", "metro"],
                "description": "Specific museum"
            },
            {
                "query": "hagia sophia opening hours",
                "expected": ["hagia sophia", "hours", "open"],
                "forbidden": ["restaurant", "transport"],
                "description": "Museum practical info"
            },
            # Location-based museum queries
            {
                "query": "museums in sultanahmet",
                "expected": ["museum", "sultanahmet", "hagia sophia", "topkapi"],
                "forbidden": ["restaurant", "transport"],
                "description": "Museums in specific area"
            },
            {
                "query": "museums near galata tower",
                "expected": ["museum", "galata", "pera", "istanbul modern"],
                "forbidden": ["restaurant", "metro"],
                "description": "Museums near landmark"
            },
            # Art and history queries
            {
                "query": "art museums istanbul",
                "expected": ["museum", "art", "pera", "istanbul modern"],
                "forbidden": ["restaurant", "transport"],
                "description": "Art museum specific"
            },
            {
                "query": "history museums",
                "expected": ["museum", "history", "archaeological", "topkapi"],
                "forbidden": ["restaurant", "metro"],
                "description": "History museum specific"
            },
            # Misspelled museum queries
            {
                "query": "musems in istanbul",
                "expected": ["museum", "istanbul", "hagia sophia"],
                "forbidden": ["restaurant", "transport"],
                "description": "Misspelled museum"
            },
            {
                "query": "topkapı palase",
                "expected": ["topkapi", "palace", "museum"],
                "forbidden": ["restaurant", "metro"],
                "description": "Misspelled palace name"
            },
            # Mixed queries (tricky)
            {
                "query": "museum restaurant",
                "expected": ["museum", "restaurant"],
                "forbidden": ["metro route"],
                "description": "Museum restaurant query (ambiguous)"
            }
        ]
        
        for test_case in test_cases:
            self.run_single_test("Museum", test_case)
    
    def test_general_queries(self):
        """Test general conversation and help queries"""
        print("\n💬 Testing General Queries")
        print("-" * 40)
        
        test_cases = [
            # Greetings
            {
                "query": "hello",
                "expected": ["hello", "help", "istanbul"],
                "forbidden": ["error", "sorry"],
                "description": "Basic greeting"
            },
            {
                "query": "hi there",
                "expected": ["hello", "hi", "help", "istanbul"],
                "forbidden": ["error"],
                "description": "Casual greeting"
            },
            # Help queries
            {
                "query": "what can you do",
                "expected": ["help", "restaurants", "museums", "transport"],
                "forbidden": ["error", "unable"],
                "description": "Capability inquiry"
            },
            {
                "query": "help me plan istanbul",
                "expected": ["help", "istanbul", "plan", "attractions"],
                "forbidden": ["error"],
                "description": "Planning help"
            },
            # Weather and general info
            {
                "query": "weather in istanbul",
                "expected": ["weather", "istanbul", "temperature"],
                "forbidden": ["restaurant", "museum"],
                "description": "Weather query"
            },
            {
                "query": "about istanbul",
                "expected": ["istanbul", "city", "turkey"],
                "forbidden": ["error"],
                "description": "General city info"
            }
        ]
        
        for test_case in test_cases:
            self.run_single_test("General", test_case)
    
    def test_problematic_inputs(self):
        """Test problematic and edge case inputs"""
        print("\n⚠️  Testing Problematic Inputs")
        print("-" * 40)
        
        test_cases = [
            # Empty and minimal inputs
            {
                "query": "",
                "expected": ["help", "istanbul"],
                "forbidden": ["error"],
                "description": "Empty query",
                "min_length": 10
            },
            {
                "query": "a",
                "expected": ["help", "istanbul"],
                "forbidden": ["error"],
                "description": "Single character",
                "min_length": 10
            },
            # Repetitive inputs
            {
                "query": "istanbul istanbul istanbul istanbul",
                "expected": ["istanbul", "help"],
                "forbidden": ["error"],
                "description": "Repeated words"
            },
            {
                "query": "aaaaaaaaaaaaaaa",
                "expected": ["help", "istanbul"],
                "forbidden": ["error"],
                "description": "Repeated characters"
            },
            # Special characters
            {
                "query": "!@#$%^&*()",
                "expected": ["help", "istanbul"],
                "forbidden": ["error"],
                "description": "Special characters only"
            },
            {
                "query": "restaurant!!!! beyoglu???",
                "expected": ["restaurant", "beyoglu"],
                "forbidden": ["error"],
                "description": "Query with excessive punctuation"
            },
            # Numbers and mixed content
            {
                "query": "123456789",
                "expected": ["help", "istanbul"],
                "forbidden": ["error"],
                "description": "Numbers only"
            },
            {
                "query": "restaurant 123 in beyoglu 456",
                "expected": ["restaurant", "beyoglu"],
                "forbidden": ["error"],
                "description": "Mixed numbers and text"
            },
            # Very long inputs
            {
                "query": "restaurant " * 100,
                "expected": ["restaurant", "istanbul"],
                "forbidden": ["error"],
                "description": "Very long repeated query"
            },
            # Non-English inputs
            {
                "query": "مطاعم في إسطنبول",
                "expected": ["help", "istanbul"],
                "forbidden": ["error"],
                "description": "Arabic script"
            },
            {
                "query": "ресторан в стамбуле",
                "expected": ["help", "istanbul"],
                "forbidden": ["error"],
                "description": "Cyrillic script"
            },
            # SQL injection attempts
            {
                "query": "'; DROP TABLE restaurants; --",
                "expected": ["help", "istanbul"],
                "forbidden": ["error", "drop", "table"],
                "description": "SQL injection attempt"
            },
            # Script injection
            {
                "query": "<script>alert('xss')</script>restaurants",
                "expected": ["restaurant", "istanbul"],
                "forbidden": ["script", "alert"],
                "description": "XSS attempt with restaurant query"
            },
            # Nonsense but structured
            {
                "query": "flibber jabberwocky wompus in istanbul",
                "expected": ["istanbul", "help"],
                "forbidden": ["error"],
                "description": "Nonsense words with location"
            }
        ]
        
        for test_case in test_cases:
            min_length = test_case.get("min_length", 20)
            self.run_single_test("Problematic", test_case, min_length=min_length)
    
    def test_multilingual_and_typos(self):
        """Test multilingual queries and common typos"""
        print("\n🌍 Testing Multilingual & Typo Handling")
        print("-" * 40)
        
        test_cases = [
            # Turkish queries
            {
                "query": "beyoğlu'nda restoranlar",
                "expected": ["restaurant", "beyoglu"],
                "forbidden": ["error"],
                "description": "Turkish restaurant query"
            },
            {
                "query": "sultanahmet müzeleri",
                "expected": ["museum", "sultanahmet"],
                "forbidden": ["restaurant"],
                "description": "Turkish museum query"
            },
            # Common English typos
            {
                "query": "resturant in istambul",
                "expected": ["restaurant", "istanbul"],
                "forbidden": ["error"],
                "description": "Common misspellings"
            },
            {
                "query": "how to go galata towar",
                "expected": ["transport", "galata", "tower"],
                "forbidden": ["error"],
                "description": "Typo in landmark name"
            },
            {
                "query": "beuatiful musem",
                "expected": ["museum", "istanbul"],
                "forbidden": ["error"],
                "description": "Multiple typos"
            },
            # Phonetic spellings
            {
                "query": "bazilika sistern",
                "expected": ["basilica", "cistern"],
                "forbidden": ["error"],
                "description": "Phonetic spelling"
            },
            {
                "query": "hagya sofya",
                "expected": ["hagia sophia"],
                "forbidden": ["error"],
                "description": "Alternative spelling"
            }
        ]
        
        for test_case in test_cases:
            self.run_single_test("Multilingual", test_case)
    
    def test_contextual_queries(self):
        """Test queries that require contextual understanding"""
        print("\n🧠 Testing Contextual Understanding")
        print("-" * 40)
        
        test_cases = [
            # Ambiguous location references
            {
                "query": "restaurants there",
                "expected": ["restaurant", "istanbul"],
                "forbidden": ["error"],
                "description": "Vague location reference"
            },
            {
                "query": "how to get there",
                "expected": ["transport", "istanbul"],
                "forbidden": ["error"],
                "description": "Vague destination"
            },
            # Time-sensitive queries
            {
                "query": "restaurants open now",
                "expected": ["restaurant", "open", "hours"],
                "forbidden": ["error"],
                "description": "Time-dependent query"
            },
            {
                "query": "museums open today",
                "expected": ["museum", "open", "hours"],
                "forbidden": ["error"],
                "description": "Current day query"
            },
            # Comparative queries
            {
                "query": "better restaurants than beyoglu",
                "expected": ["restaurant", "beyoglu"],
                "forbidden": ["error"],
                "description": "Comparative restaurant query"
            },
            {
                "query": "more interesting museums",
                "expected": ["museum", "interesting"],
                "forbidden": ["error"],
                "description": "Subjective museum query"
            },
            # Intent mixing
            {
                "query": "restaurant near metro station",
                "expected": ["restaurant", "metro", "near"],
                "forbidden": ["error"],
                "description": "Mixed restaurant and transport"
            },
            {
                "query": "transport to museum",
                "expected": ["transport", "museum"],
                "forbidden": ["error"],
                "description": "Transport to attraction"
            }
        ]
        
        for test_case in test_cases:
            self.run_single_test("Contextual", test_case)
    
    def run_single_test(self, category: str, test_case: Dict, min_length: int = 20):
        """Run a single test case"""
        self.total_tests += 1
        query = test_case["query"]
        expected = test_case["expected"]
        forbidden = test_case.get("forbidden", [])
        description = test_case["description"]
        
        print(f"Test {self.total_tests:3d}: {description}")
        print(f"         Query: '{query}'")
        
        # Make request
        start_time = time.time()
        result = self.make_request(query, f"test_{self.total_tests}")
        response_time = time.time() - start_time
        
        if result["status"] != "success":
            print(f"         ❌ FAILED - {result['error']}")
            self.failed_tests += 1
            self.results.append({
                "test_id": self.total_tests,
                "category": category,
                "query": query,
                "description": description,
                "status": "FAILED",
                "error": result["error"],
                "response": "",
                "response_time": response_time,
                "issues": [result["error"]]
            })
            return
        
        response = result["response"]
        
        # Validate response
        is_valid, issues = self.validate_response(response, expected, forbidden, min_length)
        
        if is_valid:
            print(f"         ✅ PASSED - {len(response)} chars, {response_time:.2f}s")
            self.passed_tests += 1
            status = "PASSED"
        else:
            print(f"         ❌ FAILED - Issues: {', '.join(issues)}")
            self.failed_tests += 1
            status = "FAILED"
        
        # Preview response
        preview = response[:100] + "..." if len(response) > 100 else response
        print(f"         Response: {preview}")
        print()
        
        # Store result
        self.results.append({
            "test_id": self.total_tests,
            "category": category,
            "query": query,
            "description": description,
            "status": status,
            "response": response,
            "response_time": response_time,
            "response_length": len(response),
            "expected_keywords": expected,
            "forbidden_keywords": forbidden,
            "issues": issues if not is_valid else []
        })
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE CHATBOT TEST REPORT")
        print("=" * 80)
        
        # Overall statistics
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Total Tests: {self.total_tests}")
        print(f"   Passed: {self.passed_tests} ({success_rate:.1f}%)")
        print(f"   Failed: {self.failed_tests}")
        
        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0, "failed": 0}
            categories[cat]["total"] += 1
            if result["status"] == "PASSED":
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1
        
        print(f"\n📋 RESULTS BY CATEGORY:")
        for category, stats in categories.items():
            cat_success = (stats["passed"] / stats["total"]) * 100
            print(f"   {category:15} {stats['passed']:3d}/{stats['total']:3d} ({cat_success:5.1f}%)")
        
        # Performance statistics
        response_times = [r["response_time"] for r in self.results if r["status"] == "PASSED"]
        avg_time = 0.0
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            print(f"\n⚡ PERFORMANCE:")
            print(f"   Average Response Time: {avg_time:.2f}s")
            print(f"   Fastest Response: {min(response_times):.2f}s")
            print(f"   Slowest Response: {max(response_times):.2f}s")
        
        # Failed tests analysis
        failed_tests = [r for r in self.results if r["status"] == "FAILED"]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ANALYSIS:")
            
            # Common failure reasons
            all_issues = []
            for test in failed_tests:
                all_issues.extend(test["issues"])
            
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            
            print(f"   Common Issues:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"     • {issue}: {count} occurrences")
            
            print(f"\n   Failed Test Details:")
            for test in failed_tests[:10]:  # Show first 10 failed tests
                print(f"     Test {test['test_id']:3d}: {test['description']}")
                print(f"              Query: '{test['query']}'")
                print(f"              Issues: {', '.join(test['issues'])}")
        
        # Quality assessment
        print(f"\n🎖️  QUALITY ASSESSMENT:")
        if success_rate >= 90:
            print("   🌟 EXCELLENT - Chatbot is highly reliable")
        elif success_rate >= 80:
            print("   ✅ GOOD - Chatbot performs well with minor issues")
        elif success_rate >= 70:
            print("   ⚠️  ACCEPTABLE - Some areas need improvement")
        elif success_rate >= 60:
            print("   ❌ POOR - Significant issues require attention")
        else:
            print("   🚨 CRITICAL - Major problems need immediate fixing")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if self.failed_tests > 0:
            print("   • Review failed test cases for pattern analysis")
            print("   • Improve input validation and error handling")
            print("   • Enhance fuzzy matching for typos and misspellings")
            print("   • Add better context understanding for ambiguous queries")
        else:
            print("   • Maintain current quality standards")
            print("   • Consider adding more edge case handling")
            print("   • Monitor real user interactions for new patterns")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_test_results_{timestamp}.json"
        
        report_data = {
            "test_summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": success_rate,
                "test_timestamp": timestamp
            },
            "category_breakdown": categories,
            "performance_stats": {
                "average_response_time": avg_time,
                "min_response_time": min(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0
            },
            "detailed_results": self.results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {filename}")
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 STARTING COMPREHENSIVE CHATBOT TEST SUITE")
        print("=" * 60)
        
        # Check if server is available
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code != 200:
                print("❌ Server not responding correctly")
                return
        except:
            print("❌ Server not accessible")
            print("Please start the backend server first:")
            print("cd /Users/omer/Desktop/ai-stanbul/backend && uvicorn main:app --reload --port 8000")
            return
        
        print("✅ Server is accessible, starting tests...")
        
        # Run all test categories
        start_time = time.time()
        
        self.test_restaurant_queries()
        self.test_transportation_queries()
        self.test_museum_queries()
        self.test_general_queries()
        self.test_problematic_inputs()
        self.test_multilingual_and_typos()
        self.test_contextual_queries()
        
        total_time = time.time() - start_time
        
        print(f"\n⏱️  Total test execution time: {total_time:.2f} seconds")
        
        # Generate comprehensive report
        self.generate_report()

if __name__ == "__main__":
    tester = ChatbotTester()
    tester.run_all_tests()
