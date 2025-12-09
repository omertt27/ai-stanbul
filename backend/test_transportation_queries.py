#!/usr/bin/env python3
"""
Transportation Test Queries for Istanbul Chatbot
Test script with 25 diverse transportation-related queries
"""

import json
import requests
import time
from typing import List, Dict

# Test queries covering various transportation scenarios
TRANSPORTATION_TEST_QUERIES = [
    # Direct route queries
    "how can I go kadikoy from beyoglu",
    "how to get from sultanahmet to taksim",
    "from galata to uskudar",
    "beyoglu to kadikoy",
    "how do I travel from besiktas to fatih",
    
    # Alternative phrasings
    "how can I go from eminonu to karakoy",
    "what's the best way to get to kadikoy from taksim",
    "I need to go from ortakoy to sultanahmet",
    "can you tell me how to travel from sisli to uskudar",
    "directions from bakirkoy to bebek",
    
    # Transportation mode specific
    "metro from kadikoy to vezneciler",
    "ferry from karakoy to kadikoy",
    "bus route to taksim square",
    "how to take metro to airport",
    "which ferry goes to uskudar",
    
    # General transportation questions
    "transportation in istanbul",
    "how to get around istanbul",
    "public transport options",
    "istanbul metro system",
    "ferry routes in istanbul",
    
    # Practical questions
    "how much does metro cost",
    "where to buy istanbulkart",
    "metro hours in istanbul",
    "is there uber in istanbul",
    "taxi prices in istanbul"
]

# Expected response patterns to validate
EXPECTED_PATTERNS = [
    "ferry", "metro", "bus", "taxi", "uber", "transport", "istanbul",
    "minutes", "cost", "route", "station", "terminal", "bridge",
    "bosphorus", "european", "asian", "istanbulkart", "citymapper"
]

def test_single_query(query: str, session_id: str = "test_session") -> Dict:
    """Test a single transportation query"""
    
    url = "http://localhost:8000/api/chat"
    payload = {
        "query": query,
        "session_id": session_id
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", "")
            
            # Check if response contains transportation-related keywords
            contains_transport_keywords = any(
                keyword.lower() in message.lower() 
                for keyword in EXPECTED_PATTERNS
            )
            
            return {
                "query": query,
                "status": "success",
                "response_time": round(response_time, 2),
                "response_length": len(message),
                "contains_transport_keywords": contains_transport_keywords,
                "response": message[:200] + "..." if len(message) > 200 else message
            }
        else:
            return {
                "query": query,
                "status": "error",
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "query": query,
            "status": "error", 
            "error": f"Request failed: {str(e)}"
        }

def run_transportation_tests() -> Dict:
    """Run all transportation tests and return summary"""
    
    print("🚀 Starting Istanbul Transportation Chatbot Tests")
    print(f"📝 Testing {len(TRANSPORTATION_TEST_QUERIES)} queries...")
    print("-" * 60)
    
    results = []
    successful_tests = 0
    failed_tests = 0
    total_response_time = 0
    
    for i, query in enumerate(TRANSPORTATION_TEST_QUERIES, 1):
        print(f"Test {i:2d}/25: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        result = test_single_query(query, f"test_session_{i}")
        results.append(result)
        
        if result["status"] == "success":
            successful_tests += 1
            total_response_time += result["response_time"]
            
            # Print summary of response
            keywords_found = "✅" if result["contains_transport_keywords"] else "❌"
            print(f"         Status: ✅ Success | Time: {result['response_time']}s | Keywords: {keywords_found}")
            
        else:
            failed_tests += 1
            print(f"         Status: ❌ Failed - {result['error']}")
        
        print()
        time.sleep(0.5)  # Brief delay between requests
    
    # Calculate summary statistics
    avg_response_time = total_response_time / successful_tests if successful_tests > 0 else 0
    success_rate = (successful_tests / len(TRANSPORTATION_TEST_QUERIES)) * 100
    
    keyword_matches = sum(1 for r in results if r.get("contains_transport_keywords", False))
    keyword_accuracy = (keyword_matches / successful_tests) * 100 if successful_tests > 0 else 0
    
    summary = {
        "total_tests": len(TRANSPORTATION_TEST_QUERIES),
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "success_rate": round(success_rate, 1),
        "average_response_time": round(avg_response_time, 2),
        "keyword_accuracy": round(keyword_accuracy, 1),
        "keyword_matches": keyword_matches,
        "detailed_results": results
    }
    
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']} ({summary['success_rate']}%)")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Average Response Time: {summary['average_response_time']}s")
    print(f"Transportation Keyword Accuracy: {summary['keyword_accuracy']}%")
    print(f"Responses with Transport Keywords: {summary['keyword_matches']}/{summary['successful_tests']}")
    
    return summary

def save_test_results(results: Dict, filename: str = "transportation_test_results.json"):
    """Save test results to JSON file"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to {filename}")

def test_specific_routes():
    """Test specific route combinations"""
    
    print("\n🗺️  Testing Specific Route Combinations")
    print("-" * 40)
    
    route_tests = [
        ("beyoglu", "kadikoy", "Ferry should be recommended"),
        ("kadikoy", "beyoglu", "Ferry should be recommended"),
        ("sultanahmet", "taksim", "Metro/tram combination"),
        ("airport", "sultanahmet", "Metro M1 should be mentioned"),
        ("galata", "uskudar", "Ferry or metro options")
    ]
    
    for origin, destination, expectation in route_tests:
        query = f"how to get from {origin} to {destination}"
        result = test_single_query(query)
        
        print(f"Route: {origin.title()} → {destination.title()}")
        print(f"Expected: {expectation}")
        print(f"Status: {'✅' if result['status'] == 'success' else '❌'}")
        if result['status'] == 'success':
            print(f"Response preview: {result['response'][:100]}...")
        print()

def run_full_test_suite():
    """Run complete test suite"""
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            return
    except:
        print("❌ Server not accessible at http://localhost:8000")
        print("Please start the backend server first:")
        print("cd /Users/omer/Desktop/ai-stanbul/backend && uvicorn main:app --reload --port 8000")
        return
    
    print("✅ Server is running")
    
    # Run main tests
    results = run_transportation_tests()
    
    # Save results
    save_test_results(results)
    
    # Test specific routes
    test_specific_routes()
    
    # Performance analysis
    if results['successful_tests'] > 0:
        print("\n⚡ PERFORMANCE ANALYSIS")
        print("-" * 30)
        
        response_times = [r['response_time'] for r in results['detailed_results'] if r['status'] == 'success']
        if response_times:
            print(f"Fastest Response: {min(response_times):.2f}s")
            print(f"Slowest Response: {max(response_times):.2f}s")
            print(f"Average Response: {sum(response_times)/len(response_times):.2f}s")
        
        # Quality analysis
        print(f"\n🎯 QUALITY ANALYSIS")
        print("-" * 20)
        
        keyword_hits = [r for r in results['detailed_results'] if r.get('contains_transport_keywords', False)]
        print(f"Responses with relevant keywords: {len(keyword_hits)}")
        
        avg_length = sum(r['response_length'] for r in results['detailed_results'] if r['status'] == 'success') / results['successful_tests']
        print(f"Average response length: {avg_length:.0f} characters")

if __name__ == "__main__":
    run_full_test_suite()
