#!/usr/bin/env python3
"""
Museum Test Queries for Istanbul Chatbot
Test script with diverse museum-related queries
"""

import json
import requests
import time
from typing import List, Dict

# Test queries covering various museum scenarios
MUSEUM_TEST_QUERIES = [
    # General museum queries
    "museums in istanbul",
    "best museums to visit",
    "show me museums",
    "list museums",
    "museum recommendations",
    
    # Specific museums
    "topkapi palace",
    "hagia sophia",
    "dolmabahce palace",
    "istanbul modern",
    "pera museum",
    "archaeological museum",
    "basilica cistern",
    
    # Location-based museum queries
    "museums in sultanahmet",
    "museums near galata tower",
    "museums in beyoglu",
    "museums around taksim",
    
    # Museum information queries
    "hagia sophia opening hours",
    "topkapi palace entrance fee",
    "how to get to istanbul modern",
    "dolmabahce palace history",
    "museum pass istanbul",
    
    # Art and culture queries
    "art museums istanbul",
    "history museums",
    "cultural sites",
    "ottoman museums",
    "byzantine museums"
]

# Expected museum-related keywords
EXPECTED_MUSEUM_PATTERNS = [
    "museum", "topkapi", "hagia sophia", "dolmabahce", "istanbul modern",
    "pera", "archaeological", "basilica cistern", "art", "history",
    "culture", "ottoman", "byzantine", "gallery", "exhibition",
    "sultanahmet", "beyoglu", "galata", "opening hours", "entrance fee"
]

def test_single_museum_query(query: str, session_id: str = "museum_test") -> Dict:
    """Test a single museum query"""
    
    url = "http://localhost:8000/ai"
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
            
            # Check if response contains museum-related keywords
            contains_museum_keywords = any(
                keyword.lower() in message.lower() 
                for keyword in EXPECTED_MUSEUM_PATTERNS
            )
            
            # Check if it's giving restaurant results instead of museum results
            restaurant_indicators = ["restaurant", "food", "dining", "eat", "meal"]
            is_restaurant_response = any(
                indicator.lower() in message.lower()
                for indicator in restaurant_indicators
            ) and not any(
                keyword.lower() in message.lower()
                for keyword in EXPECTED_MUSEUM_PATTERNS
            )
            
            return {
                "query": query,
                "status": "success",
                "response_time": round(response_time, 2),
                "response_length": len(message),
                "contains_museum_keywords": contains_museum_keywords,
                "is_restaurant_response": is_restaurant_response,
                "response": message[:300] + "..." if len(message) > 300 else message
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

def run_museum_tests() -> Dict:
    """Run all museum tests and return summary"""
    
    print("🏛️ Starting Istanbul Museum Chatbot Tests")
    print(f"📝 Testing {len(MUSEUM_TEST_QUERIES)} museum queries...")
    print("-" * 60)
    
    results = []
    successful_tests = 0
    failed_tests = 0
    total_response_time = 0
    museum_responses = 0
    restaurant_responses = 0
    
    for i, query in enumerate(MUSEUM_TEST_QUERIES, 1):
        print(f"Test {i:2d}/{len(MUSEUM_TEST_QUERIES)}: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        result = test_single_museum_query(query, f"museum_test_{i}")
        results.append(result)
        
        if result["status"] == "success":
            successful_tests += 1
            total_response_time += result["response_time"]
            
            # Check response type
            if result["contains_museum_keywords"]:
                museum_responses += 1
                keywords_status = "✅ Museum"
            elif result["is_restaurant_response"]:
                restaurant_responses += 1
                keywords_status = "❌ Restaurant"
            else:
                keywords_status = "⚠️ Other"
            
            print(f"         Status: ✅ Success | Time: {result['response_time']}s | Type: {keywords_status}")
            
        else:
            failed_tests += 1
            print(f"         Status: ❌ Failed - {result['error']}")
        
        print()
        time.sleep(0.5)  # Brief delay between requests
    
    # Calculate summary statistics
    avg_response_time = total_response_time / successful_tests if successful_tests > 0 else 0
    success_rate = (successful_tests / len(MUSEUM_TEST_QUERIES)) * 100
    museum_accuracy = (museum_responses / successful_tests) * 100 if successful_tests > 0 else 0
    
    summary = {
        "total_tests": len(MUSEUM_TEST_QUERIES),
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "success_rate": round(success_rate, 1),
        "average_response_time": round(avg_response_time, 2),
        "museum_responses": museum_responses,
        "restaurant_responses": restaurant_responses,
        "museum_accuracy": round(museum_accuracy, 1),
        "detailed_results": results
    }
    
    print("=" * 60)
    print("📊 MUSEUM TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']} ({summary['success_rate']}%)")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Average Response Time: {summary['average_response_time']}s")
    print(f"Museum Responses: {summary['museum_responses']}")
    print(f"Restaurant Responses: {summary['restaurant_responses']}")
    print(f"Museum Query Accuracy: {summary['museum_accuracy']}%")
    
    return summary

def test_specific_museum_queries():
    """Test some specific museum queries in detail"""
    
    print("\n🏛️  Testing Specific Museum Queries")
    print("-" * 40)
    
    specific_tests = [
        ("hagia sophia", "Should return info about Hagia Sophia"),
        ("topkapi palace", "Should return info about Topkapi Palace"),
        ("museums in sultanahmet", "Should list museums in Sultanahmet"),
        ("istanbul modern", "Should return info about Istanbul Modern"),
        ("museum pass", "Should provide info about museum passes")
    ]
    
    for query, expectation in specific_tests:
        print(f"🔍 Query: {query}")
        print(f"Expected: {expectation}")
        
        result = test_single_museum_query(query)
        
        if result['status'] == 'success':
            if result['contains_museum_keywords']:
                print(f"✅ SUCCESS - Museum response detected")
            elif result['is_restaurant_response']:
                print(f"❌ ISSUE - Got restaurant response instead")
            else:
                print(f"⚠️  UNCLEAR - Response type unclear")
            
            print(f"Response preview: {result['response'][:150]}...")
        else:
            print(f"❌ FAILED - {result['error']}")
        
        print("-" * 40)

def save_museum_test_results(results: Dict, filename: str = "museum_test_results.json"):
    """Save museum test results to JSON file"""
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Museum test results saved to {filename}")

def run_museum_test_suite():
    """Run complete museum test suite"""
    
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
    results = run_museum_tests()
    
    # Save results
    save_museum_test_results(results)
    
    # Test specific queries
    test_specific_museum_queries()
    
    # Analysis
    if results['successful_tests'] > 0:
        print("\n📈 MUSEUM QUERY ANALYSIS")
        print("-" * 30)
        
        if results['restaurant_responses'] > 0:
            print(f"⚠️  ISSUE DETECTED:")
            print(f"   {results['restaurant_responses']} museum queries returned restaurant results")
            print(f"   This suggests keyword classification needs improvement")
        
        if results['museum_responses'] == results['successful_tests']:
            print(f"✅ PERFECT: All museum queries returned museum responses")
        else:
            print(f"⚠️  {results['museum_responses']}/{results['successful_tests']} queries returned proper museum responses")
        
        # Performance analysis
        response_times = [r['response_time'] for r in results['detailed_results'] if r['status'] == 'success']
        if response_times:
            print(f"\n⚡ Performance:")
            print(f"   Fastest: {min(response_times):.2f}s")
            print(f"   Slowest: {max(response_times):.2f}s")
            print(f"   Average: {sum(response_times)/len(response_times):.2f}s")

if __name__ == "__main__":
    run_museum_test_suite()
