#!/usr/bin/env python3
"""
Comprehensive chatbot test using the extended 100+ test inputs
This script tests all categories with detailed analysis and reporting
"""

import requests
import json
import time
from datetime import datetime
import random

# Test configurations
BASE_URL = "http://127.0.0.1:8000"
ENDPOINT = "/ai"

# Extended test inputs (100 additional test cases)
extended_test_inputs = [
    # RESTAURANTS (25 additional)
    {"id": 51, "category": "Restaurants", "input": "Halal restaurants near Grand Bazaar", "expectedTopics": ["halal", "Grand Bazaar", "restaurants", "Islamic food"]},
    {"id": 52, "category": "Restaurants", "input": "Best Turkish breakfast in Çukurcuma", "expectedTopics": ["Turkish breakfast", "Çukurcuma", "kahvaltı", "morning"]},
    {"id": 53, "category": "Restaurants", "input": "Vegan restaurants in Cihangir", "expectedTopics": ["vegan", "Cihangir", "plant-based", "healthy"]},
    {"id": 54, "category": "Restaurants", "input": "Late night food delivery in Şişli", "expectedTopics": ["late night", "food delivery", "Şişli", "24 hour"]},
    {"id": 55, "category": "Restaurants", "input": "Authentic Ottoman cuisine restaurants", "expectedTopics": ["Ottoman cuisine", "authentic", "historical", "traditional"]},
    {"id": 56, "category": "Restaurants", "input": "Fish restaurants in Kumkapı", "expectedTopics": ["fish", "seafood", "Kumkapı", "fresh"]},
    {"id": 57, "category": "Restaurants", "input": "Michelin starred restaurants Istanbul", "expectedTopics": ["Michelin", "fine dining", "starred", "luxury"]},
    {"id": 58, "category": "Restaurants", "input": "Student-friendly cheap eats Beyazıt", "expectedTopics": ["student", "cheap", "budget", "Beyazıt"]},
    {"id": 59, "category": "Restaurants", "input": "Gluten-free restaurants Nişantaşı", "expectedTopics": ["gluten-free", "Nişantaşı", "dietary", "healthy"]},
    {"id": 60, "category": "Restaurants", "input": "Best meze restaurants Beyoğlu", "expectedTopics": ["meze", "Beyoğlu", "appetizers", "Turkish"]},
    
    # ATTRACTIONS (20 additional)
    {"id": 76, "category": "Attractions", "input": "Secret underground passages Istanbul", "expectedTopics": ["underground", "secret", "passages", "hidden"]},
    {"id": 77, "category": "Attractions", "input": "Byzantine ruins and walls", "expectedTopics": ["Byzantine", "ruins", "walls", "ancient"]},
    {"id": 78, "category": "Attractions", "input": "Bosphorus cruise best times", "expectedTopics": ["Bosphorus cruise", "best times", "boat", "tour"]},
    {"id": 79, "category": "Attractions", "input": "Photography spots sunrise sunset", "expectedTopics": ["photography", "sunrise", "sunset", "scenic"]},
    {"id": 80, "category": "Attractions", "input": "Free attractions and activities", "expectedTopics": ["free", "attractions", "budget", "no cost"]},
    {"id": 81, "category": "Attractions", "input": "Islamic architecture examples", "expectedTopics": ["Islamic architecture", "mosques", "design", "religious"]},
    {"id": 82, "category": "Attractions", "input": "Ottoman palaces to visit", "expectedTopics": ["Ottoman palaces", "royal", "architecture", "history"]},
    {"id": 83, "category": "Attractions", "input": "Historic hammam Turkish baths", "expectedTopics": ["hammam", "Turkish bath", "historic", "traditional"]},
    {"id": 84, "category": "Attractions", "input": "Princes Islands day trip", "expectedTopics": ["Princes Islands", "day trip", "ferry", "escape"]},
    {"id": 85, "category": "Attractions", "input": "City walls and fortifications", "expectedTopics": ["city walls", "fortifications", "Byzantine", "defense"]},
    
    # DISTRICTS (15 additional)
    {"id": 96, "category": "Districts", "input": "Explore Fener and Balat neighborhoods", "expectedTopics": ["Fener", "Balat", "colorful", "historic"]},
    {"id": 97, "category": "Districts", "input": "Trendy areas for young professionals", "expectedTopics": ["trendy", "young professionals", "modern", "lifestyle"]},
    {"id": 98, "category": "Districts", "input": "Family-friendly neighborhoods", "expectedTopics": ["family-friendly", "neighborhoods", "safe", "residential"]},
    {"id": 99, "category": "Districts", "input": "Bohemian artistic districts", "expectedTopics": ["bohemian", "artistic", "creative", "alternative"]},
    {"id": 100, "category": "Districts", "input": "Luxury shopping districts", "expectedTopics": ["luxury shopping", "upscale", "boutiques", "high-end"]},
    {"id": 101, "category": "Districts", "input": "Historic Jewish quarter", "expectedTopics": ["Jewish quarter", "historic", "synagogue", "heritage"]},
    {"id": 102, "category": "Districts", "input": "Student areas near universities", "expectedTopics": ["student areas", "universities", "academic", "young"]},
    {"id": 103, "category": "Districts", "input": "Waterfront promenade areas", "expectedTopics": ["waterfront", "promenade", "walking", "seaside"]},
    
    # TRANSPORTATION & PRACTICAL (20 additional)
    {"id": 111, "category": "Transportation", "input": "Metro lines and connections map", "expectedTopics": ["metro", "lines", "connections", "map"]},
    {"id": 112, "category": "Transportation", "input": "Taxi rates and ride sharing apps", "expectedTopics": ["taxi", "rates", "ride sharing", "apps"]},
    {"id": 113, "category": "Transportation", "input": "Bus routes to main attractions", "expectedTopics": ["bus routes", "attractions", "public transport", "stops"]},
    {"id": 114, "category": "Transportation", "input": "Dolmuş shared taxis explained", "expectedTopics": ["dolmuş", "shared taxi", "local transport", "routes"]},
    {"id": 115, "category": "Transportation", "input": "Car rental and driving tips", "expectedTopics": ["car rental", "driving", "traffic", "parking"]},
    {"id": 121, "category": "Practical", "input": "ATMs and currency exchange locations", "expectedTopics": ["ATM", "currency exchange", "money", "banking"]},
    {"id": 122, "category": "Practical", "input": "Free WiFi hotspots around city", "expectedTopics": ["free WiFi", "internet", "hotspots", "connectivity"]},
    {"id": 123, "category": "Practical", "input": "Tourist information centers", "expectedTopics": ["tourist information", "help centers", "assistance", "guides"]},
    {"id": 124, "category": "Practical", "input": "Pharmacy and medical services", "expectedTopics": ["pharmacy", "medical", "healthcare", "doctor"]},
    {"id": 125, "category": "Practical", "input": "Turkish language basic phrases", "expectedTopics": ["Turkish language", "phrases", "basic", "communication"]},
    
    # CULTURE (10 additional)
    {"id": 131, "category": "Culture", "input": "Traditional Turkish music venues", "expectedTopics": ["Turkish music", "traditional", "venues", "cultural"]},
    {"id": 132, "category": "Culture", "input": "Folk dance performances", "expectedTopics": ["folk dance", "performances", "traditional", "cultural"]},
    {"id": 133, "category": "Culture", "input": "Whirling dervish ceremonies", "expectedTopics": ["whirling dervish", "ceremony", "spiritual", "Sufi"]},
    {"id": 134, "category": "Culture", "input": "Istanbul Biennial art festival", "expectedTopics": ["Istanbul Biennial", "art festival", "contemporary", "international"]},
    {"id": 135, "category": "Culture", "input": "Ramadan celebrations and traditions", "expectedTopics": ["Ramadan", "celebrations", "traditions", "Islamic"]},
    
    # SHOPPING (10 additional)
    {"id": 141, "category": "Shopping", "input": "Local markets for authentic goods", "expectedTopics": ["local markets", "authentic", "handmade", "traditional"]},
    {"id": 142, "category": "Shopping", "input": "Vintage and second-hand shops", "expectedTopics": ["vintage", "second-hand", "thrift", "unique"]},
    {"id": 143, "category": "Shopping", "input": "Turkish carpet buying guide", "expectedTopics": ["Turkish carpet", "buying guide", "authentic", "quality"]},
    {"id": 144, "category": "Shopping", "input": "Spice market and Turkish delights", "expectedTopics": ["spice market", "Turkish delight", "food", "traditional"]},
    {"id": 145, "category": "Shopping", "input": "Modern malls and shopping centers", "expectedTopics": ["modern malls", "shopping centers", "retail", "brands"]},
]

def test_single_query(test_case):
    """Test a single query and return detailed results"""
    print(f"🔍 Testing #{test_case['id']}: {test_case['input'][:60]}...")
    
    try:
        payload = {"user_input": test_case["input"]}
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}{ENDPOINT}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=45
        )
        
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", "")
            
            # Advanced analysis
            topic_matches = sum(1 for topic in test_case["expectedTopics"] 
                              if topic.lower() in message.lower())
            topic_coverage = topic_matches / len(test_case["expectedTopics"])
            
            # Quality metrics
            is_meaningful = len(message) > 50 and not message.lower().startswith("sorry, i couldn't")
            is_detailed = len(message) > 200
            is_substantial = len(message) > 100
            contains_apology = "sorry" in message.lower()
            
            # Response quality scoring (0-100)
            quality_score = 0
            if is_meaningful: quality_score += 30
            if is_detailed: quality_score += 20
            if topic_coverage > 0.5: quality_score += 25
            if topic_coverage > 0.8: quality_score += 15
            if not contains_apology: quality_score += 10
            
            # Status determination
            if quality_score >= 80:
                status = "🟢 EXCELLENT"
            elif quality_score >= 60:
                status = "🟡 GOOD"
            elif quality_score >= 40:
                status = "🟠 FAIR"
            else:
                status = "🔴 POOR"
            
            print(f"{status} | {len(message)} chars | {topic_matches}/{len(test_case['expectedTopics'])} topics | {response_time:.1f}s")
            
            return {
                "test_id": test_case["id"],
                "category": test_case["category"],
                "input": test_case["input"],
                "success": True,
                "response": message,
                "response_length": len(message),
                "response_time": response_time,
                "topic_matches": topic_matches,
                "topic_coverage": topic_coverage,
                "is_meaningful": is_meaningful,
                "is_detailed": is_detailed,
                "is_substantial": is_substantial,
                "contains_apology": contains_apology,
                "quality_score": quality_score,
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

def run_comprehensive_test(max_tests=50, randomize=True, filter_category=None):
    """Run comprehensive test with extended test cases"""
    print("🚀 Starting AIstanbul Chatbot EXTENDED Test Suite")
    print(f"📊 Available test cases: {len(extended_test_inputs)}")
    
    # Filter by category if specified
    test_cases = extended_test_inputs
    if filter_category:
        test_cases = [t for t in extended_test_inputs if t['category'].lower() == filter_category.lower()]
        print(f"🎯 Filtering by category: {filter_category} ({len(test_cases)} tests)")
    
    # Randomize and limit
    if randomize:
        test_cases = random.sample(test_cases, min(len(test_cases), max_tests))
        print(f"🔀 Randomized selection of {len(test_cases)} tests")
    else:
        test_cases = test_cases[:max_tests]
        print(f"📋 Sequential selection of {len(test_cases)} tests")
    
    print(f"🌐 Backend: {BASE_URL}")
    
    results = []
    start_time = datetime.now()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {test_case['category']}")
        result = test_single_query(test_case)
        results.append(result)
        
        # Progressive delay (longer delays for more tests to avoid rate limiting)
        delay = min(3.0, 1.0 + (i / 10) * 0.1)
        time.sleep(delay)
    
    # Comprehensive analysis
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    successful = [r for r in results if r.get("success", False)]
    meaningful = [r for r in results if r.get("is_meaningful", False)]
    detailed = [r for r in results if r.get("is_detailed", False)]
    excellent = [r for r in results if r.get("quality_score", 0) >= 80]
    good = [r for r in results if 60 <= r.get("quality_score", 0) < 80]
    
    avg_quality = sum(r.get("quality_score", 0) for r in successful) / len(successful) if successful else 0
    avg_response_time = sum(r.get("response_time", 0) for r in successful) / len(successful) if successful else 0
    avg_topic_coverage = sum(r.get("topic_coverage", 0) for r in successful) / len(successful) if successful else 0
    
    print(f"\n" + "="*80)
    print("📈 EXTENDED TEST SUITE RESULTS")
    print(f"="*80)
    print(f"Test Duration: {total_time:.1f} seconds")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Successful Responses: {len(successful)}/{len(test_cases)} ({len(successful)/len(test_cases)*100:.1f}%)")
    print(f"Meaningful Responses: {len(meaningful)}/{len(test_cases)} ({len(meaningful)/len(test_cases)*100:.1f}%)")
    print(f"Detailed Responses: {len(detailed)}/{len(test_cases)} ({len(detailed)/len(test_cases)*100:.1f}%)")
    print(f"Excellent Quality (80+): {len(excellent)}/{len(test_cases)} ({len(excellent)/len(test_cases)*100:.1f}%)")
    print(f"Good Quality (60-79): {len(good)}/{len(test_cases)} ({len(good)/len(test_cases)*100:.1f}%)")
    print(f"Average Quality Score: {avg_quality:.1f}/100")
    print(f"Average Response Time: {avg_response_time:.2f}s")
    print(f"Average Topic Coverage: {avg_topic_coverage*100:.1f}%")
    
    # Category breakdown
    print(f"\n📊 PERFORMANCE BY CATEGORY:")
    categories = {}
    for result in results:
        cat = result.get("category", "Unknown")
        if cat not in categories:
            categories[cat] = {
                "total": 0, "successful": 0, "meaningful": 0, "detailed": 0,
                "quality_sum": 0, "quality_count": 0, "topic_coverage_sum": 0
            }
        
        stats = categories[cat]
        stats["total"] += 1
        if result.get("success", False):
            stats["successful"] += 1
            stats["quality_sum"] += result.get("quality_score", 0)
            stats["quality_count"] += 1
            stats["topic_coverage_sum"] += result.get("topic_coverage", 0)
        if result.get("is_meaningful", False):
            stats["meaningful"] += 1
        if result.get("is_detailed", False):
            stats["detailed"] += 1
    
    for category, stats in categories.items():
        meaningful_rate = stats["meaningful"] / stats["total"] * 100
        avg_quality = stats["quality_sum"] / stats["quality_count"] if stats["quality_count"] > 0 else 0
        avg_topics = stats["topic_coverage_sum"] / stats["quality_count"] * 100 if stats["quality_count"] > 0 else 0
        
        print(f"  {category}: {stats['meaningful']}/{stats['total']} meaningful ({meaningful_rate:.1f}%), "
              f"Quality: {avg_quality:.1f}, Topics: {avg_topics:.1f}%")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'extended_test_results_{timestamp}.json'
    
    # Prepare detailed report
    report = {
        "test_info": {
            "timestamp": start_time.isoformat(),
            "duration_seconds": total_time,
            "total_tests": len(test_cases),
            "filter_category": filter_category,
            "randomized": randomize
        },
        "summary": {
            "successful": len(successful),
            "meaningful": len(meaningful),
            "detailed": len(detailed),
            "excellent": len(excellent),
            "good": len(good),
            "avg_quality_score": avg_quality,
            "avg_response_time": avg_response_time,
            "avg_topic_coverage": avg_topic_coverage
        },
        "categories": categories,
        "results": results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Comprehensive report saved to {filename}")
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run extended chatbot tests')
    parser.add_argument('--max-tests', type=int, default=25, help='Maximum number of tests to run')
    parser.add_argument('--category', type=str, help='Filter by category (Restaurants, Attractions, etc.)')
    parser.add_argument('--no-randomize', action='store_true', help='Disable randomization')
    parser.add_argument('--all', action='store_true', help='Run all available tests')
    
    args = parser.parse_args()
    
    max_tests = len(extended_test_inputs) if args.all else args.max_tests
    randomize = not args.no_randomize
    
    try:
        # Test server connection first
        print("🔍 Testing server connection...")
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running!")
            results = run_comprehensive_test(
                max_tests=max_tests,
                randomize=randomize,
                filter_category=args.category
            )
        else:
            print(f"❌ Backend server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to backend server: {e}")
        print("🔧 Make sure the backend is running on http://127.0.0.1:8000")
