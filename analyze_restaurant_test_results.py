#!/usr/bin/env python3
"""
Quick Restaurant Test Analysis
Runs 5 key tests to validate the enhanced features
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Updated to use unified root-level main_system.py
from istanbul_ai.main_system import IstanbulDailyTalkAI

def test_single_query(ai, test_num, query, gps=None, description=""):
    """Test a single query and analyze the response"""
    print(f"\n{'='*80}")
    print(f"TEST #{test_num}: {description}")
    print(f"{'='*80}")
    print(f"📝 Query: '{query}'")
    if gps:
        print(f"📍 GPS: {gps.get('name')}")
    
    try:
        response = ai.process_message(
            user_input=query,
            user_id="quick_test",
            gps_location=gps
        )
        
        print(f"\n✅ SUCCESS - Response received")
        print(f"Response length: {len(response)} characters")
        
        # Analyze response
        response_lower = response.lower()
        features_detected = []
        
        if any(loc in response_lower for loc in ['beyoğlu', 'sultanahmet', 'kadıköy', 'taksim', 'besiktas']):
            features_detected.append("✅ Location-aware")
        
        if any(word in response_lower for word in ['vegetarian', 'vegan', 'halal', 'gluten']):
            features_detected.append("✅ Dietary support")
        
        if any(word in response_lower for word in ['budget', 'cheap', 'expensive', 'luxury', 'tl', 'price']):
            features_detected.append("✅ Price indication")
        
        if 'conflict' in response_lower or '⚔️' in response:
            features_detected.append("✅ Conflict detection")
        
        if 'tell me more' in response_lower or 'could you' in response_lower or 'which area' in response_lower:
            features_detected.append("✅ Ambiguity detection")
        
        if any(cuisine in response_lower for cuisine in ['turkish', 'seafood', 'italian', 'cuisine']):
            features_detected.append("✅ Cuisine filtering")
        
        print(f"\n🎯 Features Detected: {', '.join(features_detected) if features_detected else 'None'}")
        
        # Show first 200 chars of response
        preview = response[:200] + "..." if len(response) > 200 else response
        print(f"\n📄 Response Preview:\n{preview}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        return False

def main():
    print("="*80)
    print("🍽️  QUICK RESTAURANT SYSTEM ANALYSIS")
    print("="*80)
    print("Testing 5 key scenarios...")
    print()
    
    ai = IstanbulDailyTalkAI()
    results = []
    
    # Test 1: Simple location search
    results.append(test_single_query(
        ai, 1,
        "Best restaurants in Beyoğlu",
        description="Simple location-based search"
    ))
    
    # Test 2: Typo correction
    results.append(test_single_query(
        ai, 2,
        "Good resturants in beyoglu with vegiterian options",
        description="Multiple typos (resturants, beyoglu, vegiterian)"
    ))
    
    # Test 3: GPS location
    results.append(test_single_query(
        ai, 3,
        "Find seafood restaurants nearby",
        gps={'latitude': 41.0082, 'longitude': 28.9784, 'name': 'Sultanahmet'},
        description="GPS-based search with cuisine filter"
    ))
    
    # Test 4: Conflict detection
    results.append(test_single_query(
        ai, 4,
        "Looking for cheap luxury restaurants",
        description="Price conflict (cheap + luxury)"
    ))
    
    # Test 5: Ambiguity detection
    results.append(test_single_query(
        ai, 5,
        "Where to eat?",
        description="Ambiguous query"
    ))
    
    # Summary
    print(f"\n\n{'='*80}")
    print("📊 QUICK ANALYSIS SUMMARY")
    print(f"{'='*80}")
    success_count = sum(results)
    total_count = len(results)
    print(f"Tests Passed: {success_count}/{total_count}")
    print(f"Success Rate: {(success_count/total_count*100):.0f}%")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
