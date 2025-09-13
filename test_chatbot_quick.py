#!/usr/bin/env python3
"""
Quick chatbot test script to verify core functionality
"""

import requests
import json
import time
from datetime import datetime

# Test configurations
BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/ai"

# Sample test cases covering key categories
test_cases = [
    {
        "id": 1,
        "category": "Restaurants",
        "input": "Best Turkish restaurants in Sultanahmet",
        "expected_keywords": ["restaurant", "Turkish", "Sultanahmet"]
    },
    {
        "id": 2,
        "category": "Museums",
        "input": "Best museums in Istanbul",
        "expected_keywords": ["museum", "Istanbul"]
    },
    {
        "id": 3,
        "category": "Districts",
        "input": "Kadikoy",
        "expected_keywords": ["Kadikoy", "place"]
    },
    {
        "id": 4,
        "category": "Attractions",
        "input": "things to do in Istanbul",
        "expected_keywords": ["Istanbul", "place", "visit"]
    },
    {
        "id": 5,
        "category": "Transportation",
        "input": "How to get from airport to city center",
        "expected_keywords": ["airport", "transport", "city"]
    }
]

def test_single_query(test_case):
    """Test a single query and return results"""
    print(f"\n🔍 Testing: {test_case['input']}")
    
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
            
            # Check if response contains expected keywords
            contains_keywords = any(
                keyword.lower() in message.lower() 
                for keyword in test_case["expected_keywords"]
            )
            
            # Basic quality checks
            is_meaningful = len(message) > 10 and not message.startswith("Sorry, I couldn't")
            
            print(f"✅ Response received ({len(message)} chars)")
            print(f"📝 Keywords found: {contains_keywords}")
            print(f"💭 Meaningful response: {is_meaningful}")
            print(f"📄 Response preview: {message[:100]}...")
            
            return {
                "test_id": test_case["id"],
                "category": test_case["category"],
                "input": test_case["input"],
                "success": True,
                "response": message,
                "response_length": len(message),
                "contains_keywords": contains_keywords,
                "is_meaningful": is_meaningful,
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

def run_all_tests():
    """Run all test cases and generate report"""
    print("🚀 Starting AIstanbul Chatbot Quick Test")
    print(f"📊 Testing {len(test_cases)} queries")
    print(f"🌐 Backend: {BASE_URL}")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing {test_case['category']}")
        result = test_single_query(test_case)
        results.append(result)
        
        # Small delay between requests
        time.sleep(1)
    
    # Generate summary
    successful = len([r for r in results if r.get("success", False)])
    meaningful = len([r for r in results if r.get("is_meaningful", False)])
    with_keywords = len([r for r in results if r.get("contains_keywords", False)])
    
    print(f"\n" + "="*60)
    print("📈 TEST SUMMARY")
    print(f"="*60)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Successful: {successful}/{len(test_cases)} ({successful/len(test_cases)*100:.1f}%)")
    print(f"Meaningful: {meaningful}/{len(test_cases)} ({meaningful/len(test_cases)*100:.1f}%)")
    print(f"With Keywords: {with_keywords}/{len(test_cases)} ({with_keywords/len(test_cases)*100:.1f}%)")
    
    # Category breakdown
    print(f"\n📊 BY CATEGORY:")
    categories = {}
    for result in results:
        cat = result.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "success": 0, "meaningful": 0}
        categories[cat]["total"] += 1
        if result.get("success", False):
            categories[cat]["success"] += 1
        if result.get("is_meaningful", False):
            categories[cat]["meaningful"] += 1
    
    for category, stats in categories.items():
        success_rate = stats["success"] / stats["total"] * 100
        meaningful_rate = stats["meaningful"] / stats["total"] * 100
        print(f"  {category}: {stats['success']}/{stats['total']} success ({success_rate:.1f}%), "
              f"{stats['meaningful']}/{stats['total']} meaningful ({meaningful_rate:.1f}%)")
    
    # Save detailed results
    with open('/Users/omer/Desktop/ai-stanbul/test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to test_results.json")
    
    return results

if __name__ == "__main__":
    try:
        # Test server connection first
        print("🔍 Testing server connection...")
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running!")
            results = run_all_tests()
        else:
            print(f"❌ Backend server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to backend server: {e}")
        print("🔧 Make sure the backend is running on http://127.0.0.1:8000")
