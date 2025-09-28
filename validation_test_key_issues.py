#!/usr/bin/env python3
"""
Key Issues Validation Test
=========================

Focused test script to validate the key issues identified from the comprehensive test:
1. Transportation response length and specific elements
2. Hagia Sophia mosque status and current information
3. Response format improvements

This script tests the 5 most critical failing areas.
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8001"
CHAT_ENDPOINT = f"{API_BASE_URL}/ai/chat"

# Critical test cases that were failing
KEY_VALIDATION_TESTS = [
    {
        "id": "transport_airport",
        "category": "Transportation",
        "input": "How do I get from Istanbul Airport to Sultanahmet?",
        "expected_elements": ["HAVAIST", "M1A", "Kabataş", "T1 tram", "time", "transfer"],
        "expected_length_range": (150, 250),
        "key_requirement": "Must mention HAVAIST airport bus and M1A metro with Kabataş transfer"
    },
    {
        "id": "transport_taksim_bazaar", 
        "category": "Transportation",
        "input": "What's the best way to get from Taksim to Grand Bazaar?",
        "expected_elements": ["M2 metro", "Vezneciler", "walking", "15 minutes", "tram"],
        "expected_length_range": (150, 250),
        "key_requirement": "Must include M2 metro to Vezneciler and walking distance"
    },
    {
        "id": "hagia_sophia_critical",
        "category": "Museums & Cultural Sites", 
        "input": "What are the opening hours and ticket prices for Hagia Sophia?",
        "expected_elements": ["mosque", "free entry", "prayer times", "2020", "active mosque"],
        "expected_length_range": (150, 250),
        "key_requirement": "CRITICAL: Must mention it's an active mosque since 2020 with free entry"
    },
    {
        "id": "blue_mosque_features",
        "category": "Museums & Cultural Sites",
        "input": "What's the difference between the Blue Mosque and other mosques in Istanbul?",
        "expected_elements": ["six minarets", "blue tiles", "Iznik", "Sultan Ahmed", "unique"],
        "expected_length_range": (150, 250), 
        "key_requirement": "Must highlight six minarets and blue Iznik tiles as unique features"
    },
    {
        "id": "ferry_schedule",
        "category": "Transportation",
        "input": "I want to take a ferry from Eminönü to Üsküdar. What's the schedule and cost?",
        "expected_elements": ["Şehir Hatları", "frequency", "15-20 minutes", "schedule", "affordable"],
        "expected_length_range": (150, 250),
        "key_requirement": "Must mention ferry company and travel duration"
    }
]

def test_single_query(test_case):
    """Test a single query and validate response"""
    print(f"\n🧪 Testing: {test_case['id']}")
    print(f"Input: {test_case['input']}")
    print("-" * 60)
    
    try:
        payload = {"message": test_case["input"], "session_id": f"validation_{test_case['id']}"}
        response = requests.post(CHAT_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result.get("response", "")
            
            print(f"✅ Response received ({len(ai_response)} chars)")
            print(f"Response:\n{ai_response}")
            
            # Validate response
            word_count = len(ai_response.split())
            min_words, max_words = test_case["expected_length_range"]
            
            validation_results = {
                "length_ok": min_words <= word_count <= max_words,
                "word_count": word_count,
                "expected_elements_found": [],
                "missing_elements": [],
                "key_requirement_met": False
            }
            
            # Check expected elements
            response_lower = ai_response.lower()
            for element in test_case["expected_elements"]:
                if element.lower() in response_lower:
                    validation_results["expected_elements_found"].append(element)
                else:
                    validation_results["missing_elements"].append(element)
            
            # Check key requirement
            key_req = test_case["key_requirement"].lower()
            if any(word in response_lower for word in ["havaist", "m1a", "mosque", "six minarets", "ferry"]):
                validation_results["key_requirement_met"] = True
            
            # Print validation results
            print(f"\n📊 VALIDATION RESULTS:")
            print(f"   Length: {word_count} words ({'✅' if validation_results['length_ok'] else '❌'} Target: {min_words}-{max_words})")
            print(f"   Elements found: {len(validation_results['expected_elements_found'])}/{len(test_case['expected_elements'])}")
            print(f"   ✅ Found: {', '.join(validation_results['expected_elements_found'])}")
            if validation_results["missing_elements"]:
                print(f"   ❌ Missing: {', '.join(validation_results['missing_elements'])}")
            print(f"   Key requirement: {'✅' if validation_results['key_requirement_met'] else '❌'}")
            
            # Overall assessment
            elements_score = len(validation_results["expected_elements_found"]) / len(test_case["expected_elements"])
            overall_score = (
                (0.3 if validation_results["length_ok"] else 0) +
                (elements_score * 0.5) +
                (0.2 if validation_results["key_requirement_met"] else 0)
            ) * 100
            
            print(f"   Overall Score: {overall_score:.1f}/100")
            
            return {
                "test_id": test_case["id"],
                "success": True,
                "score": overall_score,
                "validation": validation_results,
                "response": ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
            }
            
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return {"test_id": test_case["id"], "success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return {"test_id": test_case["id"], "success": False, "error": str(e)}

def run_validation_tests():
    """Run all validation tests"""
    print("🔍 KEY ISSUES VALIDATION TEST")
    print("=" * 60)
    print("Testing the 5 most critical issues from comprehensive test")
    print(f"Testing endpoint: {CHAT_ENDPOINT}")
    
    results = []
    successful_tests = 0
    
    for test_case in KEY_VALIDATION_TESTS:
        result = test_single_query(test_case)
        results.append(result)
        
        if result.get("success") and result.get("score", 0) >= 70:
            successful_tests += 1
        
        time.sleep(1)  # Brief pause between requests
    
    # Summary
    print("\n" + "=" * 60)
    print("🏆 VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(KEY_VALIDATION_TESTS)
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests (≥70%): {successful_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if successful_tests == total_tests:
        print("🎉 ALL CRITICAL ISSUES RESOLVED!")
    elif successful_tests >= total_tests * 0.8:
        print("✅ Most critical issues resolved - good progress!")
    elif successful_tests >= total_tests * 0.6:
        print("⚠️ Some improvements made, but more work needed")
    else:
        print("❌ Critical issues still need attention")
    
    # Show individual results
    print(f"\n📋 INDIVIDUAL TEST RESULTS:")
    for result in results:
        if result.get("success"):
            status = "✅" if result.get("score", 0) >= 70 else "⚠️"
            print(f"   {status} {result['test_id']}: {result.get('score', 0):.1f}%")
        else:
            print(f"   ❌ {result['test_id']}: {result.get('error', 'Failed')}")
    
    return results

if __name__ == "__main__":
    run_validation_tests()
