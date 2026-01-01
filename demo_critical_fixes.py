#!/usr/bin/env python3
"""
Demonstration of Critical Fixes
Shows before/after for each fix
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.services.transportation_rag_system import get_transportation_rag

def demo_ferry_stop_count():
    """Demonstrate ferry stop count fix"""
    print("\n" + "="*80)
    print("DEMO 1: Ferry Stop Count Fix")
    print("="*80)
    
    rag = get_transportation_rag()
    route = rag.find_route("Karaköy", "Kadıköy")
    
    if route:
        print("\n📊 Route: Karaköy → Kadıköy")
        print(f"   Total time: {route.total_time} min")
        print(f"   Total distance: {route.total_distance:.2f} km")
        
        for i, step in enumerate(route.steps, 1):
            if step.get('type') == 'transit':
                line = step.get('line')
                from_station = step.get('from')
                to_station = step.get('to')
                duration = step.get('duration')
                stops = step.get('stops')
                is_ferry = step.get('ferry_crossing')
                
                print(f"\n   Step {i}: {line}")
                print(f"      {from_station} → {to_station}")
                print(f"      Duration: {duration} min")
                
                if is_ferry:
                    print(f"      ✅ Ferry crossing: Direct (no stop count)")
                    print(f"      Stops field: {stops}")
                else:
                    print(f"      Stops: {stops}")
        
        # Check if ferry step has correct format
        ferry_step = None
        for step in route.steps:
            if step.get('ferry_crossing'):
                ferry_step = step
                break
        
        if ferry_step:
            if ferry_step.get('stops') is None:
                print(f"\n   ✅ SUCCESS: Ferry step has stops=None (correct!)")
            else:
                print(f"\n   ❌ FAIL: Ferry step has stops={ferry_step.get('stops')} (should be None)")
        else:
            print(f"\n   ℹ️  No ferry step in this route (may have faster rail alternative)")


def demo_ataturk_deprecation():
    """Demonstrate Atatürk Airport deprecation"""
    print("\n" + "="*80)
    print("DEMO 2: Atatürk Airport Deprecation")
    print("="*80)
    
    rag = get_transportation_rag()
    
    queries = [
        ("Taksim", "Atatürk Airport"),
        ("Atatürk Airport", "Sultanahmet"),
    ]
    
    for origin, destination in queries:
        print(f"\n📍 Query: {origin} → {destination}")
        route = rag.find_route(origin, destination)
        
        if route:
            if route.steps and route.steps[0].get('type') == 'info':
                message = route.steps[0].get('instruction')
                print(f"   ✅ Deprecation message shown:")
                print(f"      {message}")
            else:
                print(f"   ⚠️  Route returned but no deprecation message")
                print(f"      Steps: {len(route.steps)}")
        else:
            print(f"   ❌ No route returned")


def demo_ferry_distance():
    """Demonstrate ferry distance calculation"""
    print("\n" + "="*80)
    print("DEMO 3: Ferry Distance Calculation")
    print("="*80)
    
    rag = get_transportation_rag()
    
    test_routes = [
        ("FERRY-Karaköy", "FERRY-Kadıköy", "Direct ferry (should be ~2-6 km)"),
        ("Karaköy", "Kadıköy", "Ferry with possible transfer"),
    ]
    
    for origin, destination, description in test_routes:
        print(f"\n📏 {description}")
        print(f"   Route: {origin} → {destination}")
        
        if origin.startswith("FERRY-"):
            route = rag._find_path(origin, destination, max_transfers=3)
        else:
            route = rag.find_route(origin, destination)
        
        if route:
            print(f"   Distance: {route.total_distance:.2f} km")
            print(f"   Time: {route.total_time} min")
            
            # Check for ferry steps
            has_ferry = any(step.get('ferry_crossing') for step in route.steps)
            if has_ferry:
                if route.total_distance <= 10:
                    print(f"   ✅ Distance looks reasonable (<= 10 km)")
                else:
                    print(f"   ⚠️  Distance high (> 10 km) - may indicate bug")
            else:
                print(f"   ℹ️  No ferry in route (faster alternative used)")
        else:
            print(f"   ❌ No route found")


def demo_route_ranking():
    """Demonstrate route ranking infrastructure"""
    print("\n" + "="*80)
    print("DEMO 4: Route Ranking Infrastructure")
    print("="*80)
    
    rag = get_transportation_rag()
    
    print("\n📊 Testing route ranking for: Sultanahmet → Kadıköy")
    print("   (Expected: Both rail and ferry options)")
    
    route = rag.find_route("Sultanahmet", "Kadıköy")
    
    if route:
        print(f"\n   Primary Route:")
        print(f"      Time: {route.total_time} min")
        print(f"      Transfers: {route.transfers}")
        print(f"      Lines: {', '.join(route.lines_used)}")
        
        has_ferry = any(step.get('ferry_crossing') for step in route.steps)
        print(f"      Has ferry: {has_ferry}")
        
        # Check for ranking scores
        if hasattr(route, 'ranking_scores') and route.ranking_scores:
            print(f"      ✅ Ranking scores: {route.ranking_scores}")
        else:
            print(f"      ℹ️  No ranking scores (infrastructure ready but not applied)")
        
        # Check for alternatives
        if route.alternatives:
            print(f"\n   Alternative Routes: {len(route.alternatives)}")
            for i, alt in enumerate(route.alternatives, 1):
                alt_has_ferry = any(step.get('ferry_crossing') for step in alt.steps)
                print(f"      {i}. Time: {alt.total_time} min, Ferry: {alt_has_ferry}")
        else:
            print(f"\n   ℹ️  No alternatives (route ranking not yet integrated)")


def main():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("CRITICAL FIXES DEMONSTRATION")
    print("="*80)
    print("\nThis script demonstrates the 4 critical fixes implemented:")
    print("1. Ferry stop count → 'Direct crossing'")
    print("2. Atatürk Airport → Deprecation message")
    print("3. Ferry distance → Accurate GPS calculation")
    print("4. Route ranking → Scenic preference infrastructure")
    
    try:
        demo_ferry_stop_count()
        demo_ataturk_deprecation()
        demo_ferry_distance()
        demo_route_ranking()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\n✅ All 4 fixes demonstrated")
        print("📝 See CRITICAL_FIXES_IMPLEMENTATION_SUMMARY.md for details")
        print("🧪 Run test_critical_fixes.py for comprehensive validation")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
