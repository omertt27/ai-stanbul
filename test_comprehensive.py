#!/usr/bin/env python3
"""
Comprehensive chatbot test using the 50 Istanbul guide test inputs
"""

import requests
import json
import time
from datetime import datetime

# Test configurations
BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/ai"

# The 50 test inputs from the frontend test suite
test_inputs = [
    {
        "id": 1,
        "category": "Restaurants",
        "input": "Best traditional Turkish restaurants in Sultanahmet",
        "expectedTopics": ["Turkish cuisine", "Sultanahmet", "traditional food", "restaurants"]
    },
    {
        "id": 2,
        "category": "Restaurants", 
        "input": "Where can I find authentic kebab in Beyoğlu?",
        "expectedTopics": ["kebab", "Beyoğlu", "authentic", "Turkish food"]
    },
    {
        "id": 3,
        "category": "Restaurants",
        "input": "Seafood restaurants with Bosphorus view",
        "expectedTopics": ["seafood", "Bosphorus", "view", "restaurants"]
    },
    {
        "id": 4,
        "category": "Restaurants",
        "input": "Budget-friendly local food in Kadıköy",
        "expectedTopics": ["budget", "local food", "Kadıköy", "cheap eats"]
    },
    {
        "id": 5,
        "category": "Restaurants",
        "input": "Best breakfast places in Beşiktaş",
        "expectedTopics": ["breakfast", "Beşiktaş", "morning food", "kahvaltı"]
    },
    {
        "id": 6,
        "category": "Museums",
        "input": "Best museums in Istanbul",
        "expectedTopics": ["museum", "Istanbul"]
    },
    {
        "id": 7,
        "category": "Attractions",
        "input": "Must-see historical sites in Istanbul",
        "expectedTopics": ["historical sites", "must-see", "landmarks", "history"]
    },
    {
        "id": 8,
        "category": "Attractions",
        "input": "How to visit Hagia Sophia and Blue Mosque in one day",
        "expectedTopics": ["Hagia Sophia", "Blue Mosque", "one day", "itinerary"]
    },
    {
        "id": 9,
        "category": "Districts",
        "input": "What to do in Karaköy district",
        "expectedTopics": ["Karaköy", "activities", "district guide", "neighborhood"]
    },
    {
        "id": 10,
        "category": "Transportation",
        "input": "How to get from airport to city center",
        "expectedTopics": ["airport", "city center", "transportation", "transfer"]
    }
]

def test_single_query(test_case):
    """Test a single query and return results"""
    print(f"\n🔍 Testing #{test_case['id']}: {test_case['input'][:50]}...")
    
    try:
        payload = {"user_input": test_case["input"]}
        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", "")
            
            # Check if response contains expected topics
            topic_matches = sum(1 for topic in test_case["expectedTopics"] 
                              if topic.lower() in message.lower())
            topic_coverage = topic_matches / len(test_case["expectedTopics"])
            
            # Quality checks
            is_meaningful = len(message) > 20 and not message.startswith("Sorry, I couldn't")
            is_substantial = len(message) > 100
            
            status = "✅" if is_meaningful else "⚠️"
            print(f"{status} {len(message)} chars, {topic_matches}/{len(test_case['expectedTopics'])} topics")
            
            return {
                "test_id": test_case["id"],
                "category": test_case["category"],
                "input": test_case["input"],
                "success": True,
                "response": message,
                "response_length": len(message),
                "topic_matches": topic_matches,
                "topic_coverage": topic_coverage,
                "is_meaningful": is_meaningful,
                "is_substantial": is_substantial,
                "status_code": response.status_code,
                "timestamp": datetime.now().isoformat()
            }
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return {
                "test_id": test_case["id"],
                "success": False,
                "error": f"HTTP {response.status_code}",
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return {
            "test_id": test_case["id"],
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_comprehensive_test():
    """Run the first 10 test cases as a comprehensive sample"""
    print("🚀 Starting AIstanbul Chatbot Comprehensive Test")
    print(f"📊 Testing {len(test_inputs)} sample queries from the 50-input test suite")
    print(f"🌐 Backend: {BASE_URL}")
    
    results = []
    
    for i, test_case in enumerate(test_inputs, 1):
        print(f"\n[{i}/{len(test_inputs)}] {test_case['category']}")
        result = test_single_query(test_case)
        results.append(result)
        
        # Delay between requests to avoid overwhelming the API
        time.sleep(2)
    
    # Generate comprehensive analysis
    successful = len([r for r in results if r.get("success", False)])
    meaningful = len([r for r in results if r.get("is_meaningful", False)])
    substantial = len([r for r in results if r.get("is_substantial", False)])
    
    # Topic coverage analysis
    total_topic_coverage = sum(r.get("topic_coverage", 0) for r in results if r.get("success", False))
    avg_topic_coverage = total_topic_coverage / successful if successful > 0 else 0
    
    print(f"\n" + "="*70)
    print("📈 COMPREHENSIVE TEST SUMMARY")
    print(f"="*70)
    print(f"Total Tests: {len(test_inputs)}")
    print(f"Successful Responses: {successful}/{len(test_inputs)} ({successful/len(test_inputs)*100:.1f}%)")
    print(f"Meaningful Responses: {meaningful}/{len(test_inputs)} ({meaningful/len(test_inputs)*100:.1f}%)")
    print(f"Substantial Responses: {substantial}/{len(test_inputs)} ({substantial/len(test_inputs)*100:.1f}%)")
    print(f"Average Topic Coverage: {avg_topic_coverage*100:.1f}%")
    
    # Category-wise analysis
    print(f"\n📊 PERFORMANCE BY CATEGORY:")
    categories = {}
    for result in results:
        cat = result.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = {
                "total": 0, "success": 0, "meaningful": 0, 
                "topic_coverage": 0, "topic_count": 0
            }
        categories[cat]["total"] += 1
        if result.get("success", False):
            categories[cat]["success"] += 1
            categories[cat]["topic_coverage"] += result.get("topic_coverage", 0)
            categories[cat]["topic_count"] += 1
        if result.get("is_meaningful", False):
            categories[cat]["meaningful"] += 1
    
    for category, stats in categories.items():
        success_rate = stats["success"] / stats["total"] * 100
        meaningful_rate = stats["meaningful"] / stats["total"] * 100
        avg_topics = stats["topic_coverage"] / stats["topic_count"] * 100 if stats["topic_count"] > 0 else 0
        print(f"  {category}: {stats['meaningful']}/{stats['total']} meaningful ({meaningful_rate:.1f}%), "
              f"topics {avg_topics:.1f}%")
    
    # Quality issues analysis
    print(f"\n🔍 QUALITY ANALYSIS:")
    failed_meaningful = [r for r in results if r.get("success", False) and not r.get("is_meaningful", False)]
    if failed_meaningful:
        print(f"❌ {len(failed_meaningful)} responses were not meaningful:")
        for result in failed_meaningful:
            print(f"   Test #{result['test_id']}: {result['input'][:40]}...")
            print(f"      Response: {result.get('response', 'No response')[:80]}...")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'/Users/omer/Desktop/ai-stanbul/comprehensive_test_results_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed results saved to {filename}")
    
    return results

if __name__ == "__main__":
    try:
        # Test server connection first
        print("🔍 Testing server connection...")
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running!")
            results = run_comprehensive_test()
        else:
            print(f"❌ Backend server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to backend server: {e}")
        print("🔧 Make sure the backend is running on http://127.0.0.1:8000")
