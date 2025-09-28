#!/usr/bin/env python3
"""
Quick Test Subset - 10 Representative Tests
==========================================

Test a small subset to validate our improvements before running the full suite.
"""

import requests
import json
import time
from datetime import datetime

# Test endpoint
API_URL = "http://localhost:8001/ai/chat"

# Representative test cases from each category that had issues
quick_test_cases = [
    # Transportation (was struggling)
    {
        "input": "How do I get from the airport to Sultanahmet?",
        "expected_elements": ["metro", "bus", "cost", "time", "directions"],
        "category": "Transportation"
    },
    {
        "input": "What's the best way to travel between the European and Asian sides?",
        "expected_elements": ["ferry", "metro", "bridge", "cost", "time"],
        "category": "Transportation"
    },
    
    # General Tips & Practical (lowest scoring)
    {
        "input": "What are the most important cultural etiquette rules I should follow?",
        "expected_elements": ["etiquette", "respect", "mosque", "dress", "behavior"],
        "category": "General Tips & Practical"
    },
    {
        "input": "How widespread is English and how can I communicate effectively?",
        "expected_elements": ["english", "communication", "phrases", "apps", "help"],
        "category": "General Tips & Practical"
    },
    
    # Districts & Neighborhoods (second lowest)
    {
        "input": "I want to experience local life away from tourists. Which neighborhoods should I explore?",
        "expected_elements": ["neighborhoods", "local", "authentic", "walking", "atmosphere"],
        "category": "Districts & Neighborhoods"
    },
    {
        "input": "What makes Beyoğlu district special and how should I explore it?",
        "expected_elements": ["beyoglu", "character", "attractions", "walking", "culture"],
        "category": "Districts & Neighborhoods"
    },
    
    # Museums & Cultural Sites
    {
        "input": "Which are the must-see historical sites and how should I plan my visits?",
        "expected_elements": ["sites", "history", "planning", "tickets", "hours"],
        "category": "Museums & Cultural Sites"
    },
    {
        "input": "What should I know about visiting mosques as a tourist?",
        "expected_elements": ["mosques", "etiquette", "dress", "respect", "visiting"],
        "category": "Museums & Cultural Sites"
    },
    
    # Restaurant & Food
    {
        "input": "What are the must-try traditional Turkish dishes and where can I find them?",
        "expected_elements": ["dishes", "traditional", "restaurants", "location", "recommendations"],
        "category": "Restaurant & Food"
    },
    {
        "input": "I have dietary restrictions. How can I navigate Turkish cuisine?",
        "expected_elements": ["dietary", "restrictions", "options", "communication", "help"],
        "category": "Restaurant & Food"
    }
]

def test_single_query(query_data):
    """Test a single query and analyze the response"""
    try:
        response = requests.post(
            API_URL,
            json={"message": query_data["input"]},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get('response', '')
            
            # Quick analysis
            word_count = len(ai_response.split())
            elements_found = sum(1 for elem in query_data["expected_elements"] 
                               if elem.lower() in ai_response.lower())
            coverage = elements_found / len(query_data["expected_elements"]) * 100
            
            # Check for structure
            has_structure = any(marker in ai_response.upper() for marker in 
                              ['IMMEDIATE', 'ACTIONABLE', 'ESSENTIAL', 'PRACTICAL'])
            
            return {
                "success": True,
                "category": query_data["category"],
                "query": query_data["input"][:50] + "...",
                "response_length": word_count,
                "elements_coverage": f"{elements_found}/{len(query_data['expected_elements'])} ({coverage:.1f}%)",
                "has_structure": has_structure,
                "response": ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "category": query_data["category"]
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "category": query_data["category"]
        }

def run_quick_test():
    """Run the quick test suite"""
    print("🚀 AI Istanbul Quick Test Suite (10 tests)")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for i, test_case in enumerate(quick_test_cases, 1):
        print(f"\n🔍 Test {i}/10: {test_case['category']}")
        print(f"Query: {test_case['input'][:60]}...")
        
        result = test_single_query(test_case)
        results.append(result)
        
        if result["success"]:
            print(f"✅ Success - {result['response_length']} words, {result['elements_coverage']} coverage")
            print(f"   Structure: {'Yes' if result['has_structure'] else 'No'}")
        else:
            print(f"❌ Failed - {result['error']}")
        
        # Small delay between requests
        time.sleep(1)
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    
    print("\n" + "=" * 60)
    print("📊 QUICK TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Success Rate: {successful/len(results)*100:.1f}%")
    print(f"Execution Time: {total_time:.1f} seconds")
    
    # Category breakdown
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"success": 0, "total": 0}
        categories[cat]["total"] += 1
        if result["success"]:
            categories[cat]["success"] += 1
    
    print("\n📈 CATEGORY BREAKDOWN:")
    for cat, stats in categories.items():
        rate = stats["success"] / stats["total"] * 100
        print(f"   {cat}: {stats['success']}/{stats['total']} ({rate:.1f}%)")
    
    # Show successful responses
    print("\n✅ SUCCESSFUL RESPONSES:")
    for i, result in enumerate([r for r in results if r["success"]], 1):
        print(f"\n{i}. {result['category']} - {result['elements_coverage']} coverage")
        print(f"   {result['response']}")
    
    return results

if __name__ == "__main__":
    results = run_quick_test()
