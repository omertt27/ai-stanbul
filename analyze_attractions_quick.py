#!/usr/bin/env python3
"""
Quick Attractions Analysis - 10 Key Tests
Tests the core functionality of attractions system
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Updated to use unified root-level main_system.py
from istanbul_ai.main_system import IstanbulDailyTalkAI

def analyze_response(response: str, query: str) -> dict:
    """Analyze response for key features"""
    response_lower = response.lower()
    query_lower = query.lower()
    
    features = {
        'museums': any(w in response_lower for w in ['museum', 'palace', 'gallery', 'exhibition', 'topkapi', 'hagia sophia']),
        'parks': any(w in response_lower for w in ['park', 'garden', 'emirgan', 'gulhane']),
        'religious': any(w in response_lower for w in ['mosque', 'church', 'blue mosque', 'prayer']),
        'monuments': any(w in response_lower for w in ['tower', 'galata', 'monument', 'landmark']),
        'location_aware': any(loc in response_lower for loc in ['beyoğlu', 'sultanahmet', 'kadıköy', 'taksim', 'fatih']),
        'family_friendly': any(w in response_lower for w in ['family', 'children', 'kids', 'child']),
        'romantic': any(w in response_lower for w in ['romantic', 'couple', 'sunset', 'view', 'bosphorus']),
        'budget_free': any(w in response_lower for w in ['free', 'no fee', 'no cost', 'entrance free']),
        'weather_aware': any(w in response_lower for w in ['rain', 'weather', 'indoor', 'outdoor', 'sunny']),
        'has_recommendations': len(response) > 200
    }
    
    return features

def test_query(ai, num, query, desc):
    """Test a single query"""
    print(f"\n{'='*80}")
    print(f"TEST #{num}: {desc}")
    print(f"Query: '{query}'")
    
    start = time.time()
    try:
        response = ai.process_message(query, "quick_test_attractions")
        elapsed = time.time() - start
        
        features = analyze_response(response, query)
        detected = [k for k, v in features.items() if v]
        
        print(f"✅ Success ({elapsed:.2f}s) - Features: {', '.join(detected) if detected else 'None'}")
        print(f"Response preview: {response[:150]}...")
        
        return True, features
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return False, {}

def main():
    print("="*80)
    print("🏛️  QUICK ATTRACTIONS ANALYSIS (10 Tests)")
    print("="*80)
    
    ai = IstanbulDailyTalkAI()
    results = []
    
    # Test 1: General museums
    results.append(test_query(ai, 1, "Best museums in Istanbul", "General museum query"))
    
    # Test 2: Contemporary art
    results.append(test_query(ai, 2, "Contemporary art museums in Beyoğlu", "Art museums + location"))
    
    # Test 3: Mosques
    results.append(test_query(ai, 3, "Beautiful mosques to visit", "Religious sites"))
    
    # Test 4: Parks for kids
    results.append(test_query(ai, 4, "Parks for children in Istanbul", "Family-friendly parks"))
    
    # Test 5: Romantic places
    results.append(test_query(ai, 5, "Romantic places for couples", "Romantic spots"))
    
    # Test 6: Free attractions
    results.append(test_query(ai, 6, "Free things to do in Istanbul", "Budget-friendly"))
    
    # Test 7: Rainy day
    results.append(test_query(ai, 7, "Indoor attractions for rainy day", "Weather-appropriate"))
    
    # Test 8: Galata Tower
    results.append(test_query(ai, 8, "How to visit Galata Tower?", "Specific monument"))
    
    # Test 9: Hagia Sophia
    results.append(test_query(ai, 9, "Tell me about Hagia Sophia", "Specific attraction"))
    
    # Test 10: Family day
    results.append(test_query(ai, 10, "Plan a family day with attractions", "Multi-intent family"))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 QUICK ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    success_count = sum(1 for r in results if r[0])
    print(f"Success Rate: {success_count}/10 ({success_count*10}%)")
    
    # Feature analysis
    all_features = {}
    for success, features in results:
        if success:
            for feature, detected in features.items():
                if detected:
                    all_features[feature] = all_features.get(feature, 0) + 1
    
    print(f"\nFeature Detection:")
    for feature, count in sorted(all_features.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {feature}: {count}/10 tests ({count*10}%)")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
