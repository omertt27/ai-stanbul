#!/usr/bin/env python3
"""
Trace the actual path taken in the route
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.transportation_rag_system import get_transportation_rag

# Get RAG system
rag = get_transportation_rag()

# Manually trace the route using the routing algorithm
print("Finding path: MARMARAY-Sirkeci → MARMARAY-Ayrılık Çeşmesi")
print("=" * 80)

# Use the internal _find_path method
origin_id = "MARMARAY-Sirkeci"
dest_id = "MARMARAY-Ayrılık Çeşmesi"

# This will give us the route
route = rag.find_route("Sirkeci", "Ayrılık Çeşmesi", max_transfers=0)

if route:
    print(f"Route found: {route.origin} → {route.destination}")
    print(f"Time: {route.total_time} min")
    print(f"Distance: {route.total_distance:.1f} km")
    print(f"\nSteps:")
    for i, step in enumerate(route.steps, 1):
        if step.get('type') == 'transit':
            print(f"{i}. {step['line']}: {step['from']} → {step['to']}")
            print(f"   Stops: {step['stops']}, Duration: {step['duration']} min")
else:
    print("No route found")

print("\n" + "=" * 80)
print("Finding path: M4-Ayrılık Çeşmesi → M4-Kadıköy")
print("=" * 80)

# Check M4 path
route2 = rag.find_route("M4-Ayrılık Çeşmesi", "M4-Kadıköy", max_transfers=0)

if route2:
    print(f"Route found: {route2.origin} → {route2.destination}")
    print(f"Time: {route2.total_time} min")
    print(f"Distance: {route2.total_distance:.1f} km")
    print(f"\nSteps:")
    for i, step in enumerate(route2.steps, 1):
        if step.get('type') == 'transit':
            print(f"{i}. {step['line']}: {step['from']} → {step['to']}")
            print(f"   Stops: {step['stops']}, Duration: {step['duration']} min")
else:
    print("No route found")
