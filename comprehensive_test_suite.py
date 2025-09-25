#!/usr/bin/env python3
"""
Comprehensive AI Istanbul Test Suite - 75 Test Cases
Tests all categories: general advice, transportation, districts, museums, restaurants, and follow-ups
"""

import requests
import json
import time
from datetime import datetime
import sys

# Backend URL
BASE_URL = "http://localhost:8000"

# Test categories with inputs
test_cases = [
    # ========== GENERAL ADVICE (15 tests) ==========
    {
        "category": "general_advice",
        "input": "Hi! I'm visiting Istanbul for the first time. What should I know?",
        "expected_topics": ["attractions", "tips", "neighborhoods", "culture"]
    },
    {
        "category": "general_advice", 
        "input": "What are the must-see attractions in Istanbul?",
        "expected_topics": ["Hagia Sophia", "Blue Mosque", "Topkapi Palace", "Grand Bazaar"]
    },
    {
        "category": "general_advice",
        "input": "How many days do I need to explore Istanbul properly?",
        "expected_topics": ["days", "itinerary", "time", "planning"]
    },
    {
        "category": "general_advice",
        "input": "What's the best time of year to visit Istanbul?",
        "expected_topics": ["weather", "season", "spring", "fall"]
    },
    {
        "category": "general_advice",
        "input": "Is Istanbul safe for tourists?",
        "expected_topics": ["safety", "tips", "precautions"]
    },
    {
        "category": "general_advice",
        "input": "What should I pack for my Istanbul trip?",
        "expected_topics": ["clothing", "shoes", "weather", "mosque"]
    },
    {
        "category": "general_advice",
        "input": "How much money should I budget for Istanbul?",
        "expected_topics": ["budget", "cost", "accommodation", "food"]
    },
    {
        "category": "general_advice",
        "input": "Do I need to know Turkish to visit Istanbul?",
        "expected_topics": ["language", "English", "communication"]
    },
    {
        "category": "general_advice",
        "input": "What are some cultural customs I should know about?",
        "expected_topics": ["culture", "customs", "mosque", "respect"]
    },
    {
        "category": "general_advice",
        "input": "Is Istanbul family-friendly with children?",
        "expected_topics": ["family", "children", "activities", "attractions"]
    },
    {
        "category": "general_advice",
        "input": "What's special about Istanbul compared to other cities?",
        "expected_topics": ["unique", "Europe", "Asia", "history", "culture"]
    },
    {
        "category": "general_advice",
        "input": "Can you recommend a romantic itinerary for couples?",
        "expected_topics": ["romantic", "couples", "sunset", "restaurants"]
    },
    {
        "category": "general_advice",
        "input": "What are some free things to do in Istanbul?",
        "expected_topics": ["free", "budget", "parks", "walking", "mosques"]
    },
    {
        "category": "general_advice",
        "input": "Tell me about Istanbul's history in brief",
        "expected_topics": ["history", "Byzantine", "Ottoman", "Constantinople"]
    },
    {
        "category": "general_advice",
        "input": "What makes Istanbul culturally unique?",
        "expected_topics": ["culture", "East meets West", "diversity", "traditions"]
    },

    # ========== TRANSPORTATION (15 tests) ==========
    {
        "category": "transportation",
        "input": "How do I get from the airport to the city center?",
        "expected_topics": ["airport", "metro", "taxi", "IST", "transport"]
    },
    {
        "category": "transportation",
        "input": "What is the Istanbul Card and how does it work?",
        "expected_topics": ["Istanbulkart", "public transport", "metro", "bus", "ferry"]
    },
    {
        "category": "transportation",
        "input": "How do I get from Sultanahmet to Galata Tower?",
        "expected_topics": ["metro", "tram", "walking", "Golden Horn"]
    },
    {
        "category": "transportation",
        "input": "What's the best way to cross from Europe to Asia side?",
        "expected_topics": ["ferry", "bridge", "metro", "Bosphorus"]
    },
    {
        "category": "transportation",
        "input": "Are taxis expensive in Istanbul?",
        "expected_topics": ["taxi", "meter", "BiTaksi", "Uber", "cost"]
    },
    {
        "category": "transportation",
        "input": "How does the metro system work in Istanbul?",
        "expected_topics": ["metro", "lines", "M1", "M2", "connections"]
    },
    {
        "category": "transportation",
        "input": "Can I use Uber in Istanbul?",
        "expected_topics": ["Uber", "BiTaksi", "taxi apps", "available"]
    },
    {
        "category": "transportation",
        "input": "What are the ferry routes in Istanbul?",
        "expected_topics": ["ferry", "Bosphorus", "Golden Horn", "routes", "scenic"]
    },
    {
        "category": "transportation",
        "input": "How do I get to Kadıköy from Beyoğlu?",
        "expected_topics": ["ferry", "metro", "Karaköy", "Eminönü"]
    },
    {
        "category": "transportation",
        "input": "Is public transportation safe at night?",
        "expected_topics": ["night", "safety", "metro", "taxi", "late hours"]
    },
    {
        "category": "transportation",
        "input": "How much does a taxi cost from Taksim to Sultanahmet?",
        "expected_topics": ["taxi", "cost", "distance", "meter"]
    },
    {
        "category": "transportation",
        "input": "What's the difference between metro and metrobus?",
        "expected_topics": ["metro", "metrobus", "BRT", "difference"]
    },
    {
        "category": "transportation",
        "input": "Can I walk between major attractions in Sultanahmet?",
        "expected_topics": ["walking", "distance", "Sultanahmet", "attractions"]
    },
    {
        "category": "transportation",
        "input": "How do I get to the Asian side from the airport?",
        "expected_topics": ["airport", "Asian side", "metro", "connection"]
    },
    {
        "category": "transportation",
        "input": "What's the best transport app for Istanbul?",
        "expected_topics": ["app", "Moovit", "Google Maps", "BiTaksi"]
    },

    # ========== DISTRICTS (15 tests) ==========
    {
        "category": "districts",
        "input": "Tell me about Sultanahmet district",
        "expected_topics": ["historic", "Hagia Sophia", "Blue Mosque", "Topkapi"]
    },
    {
        "category": "districts",
        "input": "What's special about Beyoğlu?",
        "expected_topics": ["modern", "nightlife", "Istiklal", "Galata Tower"]
    },
    {
        "category": "districts",
        "input": "Should I visit Kadıköy on the Asian side?",
        "expected_topics": ["Asian side", "local", "authentic", "market", "Moda"]
    },
    {
        "category": "districts",
        "input": "What can I do in Galata area?",
        "expected_topics": ["Galata Tower", "views", "trendy", "cafes", "art"]
    },
    {
        "category": "districts",
        "input": "Is Beşiktaş worth visiting?",
        "expected_topics": ["modern", "shopping", "Dolmabahçe", "waterfront"]
    },
    {
        "category": "districts",
        "input": "Tell me about the Fatih district",
        "expected_topics": ["conservative", "mosques", "traditional", "history"]
    },
    {
        "category": "districts",
        "input": "What's Balat neighborhood like?",
        "expected_topics": ["colorful", "historic", "Jewish quarter", "Instagram"]
    },
    {
        "category": "districts",
        "input": "Should I stay in Taksim area?",
        "expected_topics": ["central", "nightlife", "transport", "hotels"]
    },
    {
        "category": "districts",
        "input": "What's the difference between Üsküdar and Kadıköy?",
        "expected_topics": ["Asian side", "conservative", "modern", "atmosphere"]
    },
    {
        "category": "districts",
        "input": "Is Ortaköy good for dining?",
        "expected_topics": ["waterfront", "restaurants", "Bosphorus", "romantic"]
    },
    {
        "category": "districts",
        "input": "Tell me about Karaköy neighborhood",
        "expected_topics": ["trendy", "galleries", "cafes", "design", "hipster"]
    },
    {
        "category": "districts",
        "input": "What's Cihangir known for?",
        "expected_topics": ["bohemian", "cafes", "artists", "alternative"]
    },
    {
        "category": "districts",
        "input": "Should I visit Eminönü?",
        "expected_topics": ["Spice Bazaar", "ferry terminal", "historic", "busy"]
    },
    {
        "category": "districts",
        "input": "What's special about Arnavutköy?",
        "expected_topics": ["Bosphorus", "Ottoman houses", "seafood", "quiet"]
    },
    {
        "category": "districts",
        "input": "Which district is best for families with kids?",
        "expected_topics": ["family", "parks", "safe", "activities", "Sultanahmet"]
    },

    # ========== MUSEUMS (15 tests) ==========
    {
        "category": "museums",
        "input": "What are the top museums in Istanbul?",
        "expected_topics": ["Topkapi Palace", "Hagia Sophia", "Archaeological", "Istanbul Modern"]
    },
    {
        "category": "museums",
        "input": "Tell me about Topkapi Palace Museum",
        "expected_topics": ["Ottoman", "palace", "treasury", "harem", "sultans"]
    },
    {
        "category": "museums",
        "input": "Is Hagia Sophia worth visiting?",
        "expected_topics": ["Byzantine", "Ottoman", "mosaics", "architecture", "historic"]
    },
    {
        "category": "museums",
        "input": "What can I see at the Archaeological Museums?",
        "expected_topics": ["archaeology", "ancient", "artifacts", "sarcophagus"]
    },
    {
        "category": "museums",
        "input": "Tell me about Istanbul Modern Art Museum",
        "expected_topics": ["contemporary", "Turkish art", "modern", "exhibitions"]
    },
    {
        "category": "museums",
        "input": "What's the Museum Pass Istanbul?",
        "expected_topics": ["pass", "skip lines", "museums", "discount", "value"]
    },
    {
        "category": "museums",
        "input": "Should I visit the Basilica Cistern?",
        "expected_topics": ["underground", "cistern", "columns", "atmospheric", "Byzantine"]
    },
    {
        "category": "museums",
        "input": "What's in the Turkish and Islamic Arts Museum?",
        "expected_topics": ["Islamic art", "carpets", "manuscripts", "decorative"]
    },
    {
        "category": "museums",
        "input": "Is Dolmabahçe Palace worth seeing?",
        "expected_topics": ["palace", "European style", "Ottoman", "luxury", "tours"]
    },
    {
        "category": "museums",
        "input": "Tell me about the Pera Museum",
        "expected_topics": ["European art", "orientalist", "exhibitions", "Beyoğlu"]
    },
    {
        "category": "museums",
        "input": "What's special about the Chora Church?",
        "expected_topics": ["Byzantine", "mosaics", "frescoes", "Christian art"]
    },
    {
        "category": "museums",
        "input": "Are there any science museums in Istanbul?",
        "expected_topics": ["science", "interactive", "children", "technology"]
    },
    {
        "category": "museums",
        "input": "What are the best art galleries in Istanbul?",
        "expected_topics": ["galleries", "contemporary", "local artists", "Karaköy"]
    },
    {
        "category": "museums",
        "input": "How much time do I need for Topkapi Palace?",
        "expected_topics": ["time", "hours", "visit", "sections", "planning"]
    },
    {
        "category": "museums",
        "input": "Can I take photos in Istanbul museums?",
        "expected_topics": ["photography", "rules", "restrictions", "policy"]
    },

    # ========== RESTAURANTS & FOOD (15 tests) ==========
    {
        "category": "restaurants",
        "input": "What are the must-try Turkish dishes in Istanbul?",
        "expected_topics": ["kebab", "döner", "meze", "baklava", "Turkish delight"]
    },
    {
        "category": "restaurants",
        "input": "Where can I find the best kebab in Istanbul?",
        "expected_topics": ["kebab", "restaurants", "recommendations", "authentic"]
    },
    {
        "category": "restaurants",
        "input": "What's the best area for seafood restaurants?",
        "expected_topics": ["seafood", "Kumkapı", "Karaköy", "Bosphorus", "fish"]
    },
    {
        "category": "restaurants",
        "input": "Tell me about Turkish breakfast culture",
        "expected_topics": ["breakfast", "kahvaltı", "cheese", "olives", "tea"]
    },
    {
        "category": "restaurants",
        "input": "Where can I eat with a Bosphorus view?",
        "expected_topics": ["Bosphorus", "view", "restaurants", "waterfront", "romantic"]
    },
    {
        "category": "restaurants",
        "input": "What's street food like in Istanbul?",
        "expected_topics": ["street food", "simit", "balık ekmek", "döner", "midye"]
    },
    {
        "category": "restaurants",
        "input": "Are there good vegetarian restaurants in Istanbul?",
        "expected_topics": ["vegetarian", "vegan", "options", "restaurants"]
    },
    {
        "category": "restaurants",
        "input": "What should I try at the Grand Bazaar food court?",
        "expected_topics": ["Grand Bazaar", "food court", "local", "snacks"]
    },
    {
        "category": "restaurants",
        "input": "Where can I find authentic Ottoman cuisine?",
        "expected_topics": ["Ottoman", "traditional", "historic", "recipes"]
    },
    {
        "category": "restaurants",
        "input": "What's the tipping culture in Istanbul restaurants?",
        "expected_topics": ["tipping", "service charge", "culture", "percentage"]
    },
    {
        "category": "restaurants",
        "input": "Best rooftop restaurants in Istanbul?",
        "expected_topics": ["rooftop", "views", "dining", "skyline", "atmosphere"]
    },
    {
        "category": "restaurants",
        "input": "Where should I go for Turkish tea and coffee?",
        "expected_topics": ["tea", "coffee", "çay", "Turkish coffee", "culture"]
    },
    {
        "category": "restaurants",
        "input": "What's the food scene like in Kadıköy?",
        "expected_topics": ["Kadıköy", "local", "authentic", "market", "diverse"]
    },
    {
        "category": "restaurants",
        "input": "Are there any food tours in Istanbul?",
        "expected_topics": ["food tours", "guided", "tasting", "culinary"]
    },
    {
        "category": "restaurants",
        "input": "What desserts should I try in Istanbul?",
        "expected_topics": ["desserts", "baklava", "Turkish delight", "künefe", "sweets"]
    }
]

def test_ai_response(input_text, session_id="test_session"):
    """Send request to AI endpoint and return response"""
    try:
        response = requests.post(
            f"{BASE_URL}/ai",
            json={
                "user_input": input_text,
                "session_id": session_id
            },
            headers={
                "Content-Type": "application/json",
                "Accept-Language": "en-US,en;q=0.9"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response")
        else:
            return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def check_response_quality(response, expected_topics):
    """Check if response contains expected topics"""
    response_lower = response.lower()
    found_topics = []
    missing_topics = []
    
    for topic in expected_topics:
        if topic.lower() in response_lower:
            found_topics.append(topic)
        else:
            missing_topics.append(topic)
    
    coverage = len(found_topics) / len(expected_topics) * 100
    return coverage, found_topics, missing_topics

def run_comprehensive_test():
    """Run all 75 test cases and generate report"""
    print("🚀 Starting Comprehensive AI Istanbul Test Suite")
    print("=" * 80)
    print(f"Total test cases: {len(test_cases)}")
    print(f"Testing backend: {BASE_URL}")
    print("=" * 80)
    
    results = []
    category_stats = {}
    
    for i, test_case in enumerate(test_cases, 1):
        category = test_case["category"]
        input_text = test_case["input"]
        expected_topics = test_case["expected_topics"]
        
        print(f"\n[{i:2d}/75] Testing {category}: {input_text[:50]}...")
        
        # Send request
        response = test_ai_response(input_text, f"test_session_{i}")
        
        # Check quality
        coverage, found, missing = check_response_quality(response, expected_topics)
        
        # Store result
        result = {
            "test_id": i,
            "category": category,
            "input": input_text,
            "response": response,
            "expected_topics": expected_topics,
            "found_topics": found,
            "missing_topics": missing,
            "coverage": coverage,
            "status": "PASS" if coverage >= 60 else "FAIL"  # 60% threshold
        }
        results.append(result)
        
        # Update category stats
        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0, "coverage_sum": 0}
        
        category_stats[category]["total"] += 1
        category_stats[category]["coverage_sum"] += coverage
        if result["status"] == "PASS":
            category_stats[category]["passed"] += 1
        
        # Print immediate result
        status_emoji = "✅" if result["status"] == "PASS" else "❌"
        print(f"    {status_emoji} {result['status']} - Coverage: {coverage:.1f}% - Found: {len(found)}/{len(expected_topics)} topics")
        
        if missing:
            print(f"    Missing: {', '.join(missing)}")
        
        # Brief delay to not overwhelm server
        time.sleep(0.1)
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS REPORT")
    print("=" * 80)
    
    # Overall stats
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["status"] == "PASS")
    overall_coverage = sum(r["coverage"] for r in results) / total_tests
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"   Failed: {total_tests - passed_tests} ({(total_tests-passed_tests)/total_tests*100:.1f}%)")
    print(f"   Average Coverage: {overall_coverage:.1f}%")
    
    # Category breakdown
    print(f"\n📋 CATEGORY BREAKDOWN:")
    for category, stats in category_stats.items():
        avg_coverage = stats["coverage_sum"] / stats["total"]
        pass_rate = stats["passed"] / stats["total"] * 100
        print(f"   {category.upper().replace('_', ' ')}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%) - Avg Coverage: {avg_coverage:.1f}%")
    
    # Failed tests details
    failed_tests = [r for r in results if r["status"] == "FAIL"]
    if failed_tests:
        print(f"\n❌ FAILED TESTS ({len(failed_tests)} tests):")
        for test in failed_tests:
            print(f"   [{test['test_id']:2d}] {test['category']} - {test['coverage']:.1f}% coverage")
            print(f"       Input: {test['input'][:60]}...")
            if test['missing_topics']:
                print(f"       Missing: {', '.join(test['missing_topics'])}")
            print()
    
    # Top performing tests
    top_tests = sorted(results, key=lambda x: x["coverage"], reverse=True)[:5]
    print(f"\n🏆 TOP PERFORMING TESTS:")
    for test in top_tests:
        print(f"   [{test['test_id']:2d}] {test['category']} - {test['coverage']:.1f}% coverage")
        print(f"       {test['input'][:70]}...")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "overall_coverage": overall_coverage,
                "category_stats": category_stats
            },
            "test_results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {filename}")
    print("=" * 80)
    
    return results, category_stats

if __name__ == "__main__":
    print("AI Istanbul Comprehensive Test Suite")
    print("Testing 75 scenarios across 5 categories...")
    
    # Check if backend is running
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Backend is running and accessible")
        else:
            print("❌ Backend returned unexpected status")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("Please ensure the backend is running on http://localhost:8000")
        sys.exit(1)
    
    # Run tests
    results, stats = run_comprehensive_test()
    
    print("\n🎉 Test suite completed!")
