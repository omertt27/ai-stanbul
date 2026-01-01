#!/usr/bin/env python3
"""
Analyze the timing model to find absurd durations.
"""
import sys
sys.path.insert(0, '/Users/omer/Desktop/ai-stanbul/backend')

from services.transportation_rag_system import get_transportation_rag

rag = get_transportation_rag()

print("="*80)
print("TIMING MODEL ANALYSIS")
print("="*80)

test_cases = [
    {
        "name": "Beşiktaş → Üsküdar (Ferry)",
        "origin": "Beşiktaş",
        "destination": "Üsküdar",
        "expected_time_min": 15,
        "expected_time_max": 20,
        "expected_distance_km": 2.0,
        "transport": "FERRY"
    },
    {
        "name": "Zeytinburnu → Bostancı (MARMARAY)",
        "origin": "Zeytinburnu",
        "destination": "Bostancı",
        "expected_time_min": 35,
        "expected_time_max": 40,
        "expected_distance_km": 20.0,
        "transport": "MARMARAY"
    }
]

for test in test_cases:
    print(f"\n{'='*80}")
    print(f"TEST: {test['name']}")
    print(f"{'='*80}")
    
    route = rag.find_route(test['origin'], test['destination'])
    
    if not route:
        print("❌ No route found!")
        continue
    
    print(f"\n📊 ACTUAL vs EXPECTED:")
    print(f"   Time:     {route.total_time} min (expected: {test['expected_time_min']}-{test['expected_time_max']} min)")
    print(f"   Distance: {route.total_distance} km (expected: ~{test['expected_distance_km']} km)")
    
    # Check if within reasonable bounds
    time_ok = test['expected_time_min'] <= route.total_time <= test['expected_time_max']
    dist_ok = route.total_distance >= test['expected_distance_km'] * 0.8
    
    if time_ok:
        print(f"   ✅ Time is reasonable")
    else:
        print(f"   ❌ Time is {route.total_time / test['expected_time_min']:.1f}x too {'fast' if route.total_time < test['expected_time_min'] else 'slow'}")
    
    if dist_ok:
        print(f"   ✅ Distance is reasonable")
    else:
        print(f"   ❌ Distance is {route.total_distance / test['expected_distance_km']:.1%} of expected")
    
    print(f"\n📋 Route Details:")
    for step in route.steps:
        if step['type'] == 'transit':
            stops = step.get('stops', 0)
            duration = step.get('duration', 0)
            
            # Calculate speed
            if duration > 0 and stops > 0:
                min_per_stop = duration / stops if stops > 0 else 0
                print(f"   {step['line']}: {step['from']} → {step['to']}")
                print(f"      • {stops} stops in {duration} min = {min_per_stop:.2f} min/stop")
                
                # Check for absurdities
                if min_per_stop < 0.5:
                    print(f"      ❌ ABSURD: {min_per_stop:.2f} min/stop is impossibly fast!")
                elif min_per_stop > 10:
                    print(f"      ⚠️  WARNING: {min_per_stop:.2f} min/stop is very slow")
                else:
                    print(f"      ✅ {min_per_stop:.2f} min/stop is reasonable")

print("\n" + "="*80)
print("CHECKING _create_direct_route() LOGIC")
print("="*80)

# Check the direct route code
print("\nThe _create_direct_route() method calculates:")
print("  travel_time from database OR stops * 2.5 min")
print("  distance = (travel_time / 10.0) * 1.5 km")
print("\nThis explains:")
print("  ❌ Ferry routes get wrong timing (treated like metro)")
print("  ❌ Distance calculated from TIME not actual route geometry")
print("  ❌ No transport-type-specific timing rules")
