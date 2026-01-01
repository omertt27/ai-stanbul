#!/usr/bin/env python3
"""
Debug the actual Dijkstra path
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.transportation_rag_system import get_transportation_rag

# Get RAG
rag = get_transportation_rag()

# Manually call _find_path_dijkstra with debug
print("Testing Dijkstra: MARMARAY-Sirkeci → MARMARAY-Ayrılık Çeşmesi")
print("=" * 80)

route = rag._find_path_dijkstra(
    "MARMARAY-Sirkeci",
    "MARMARAY-Ayrılık Çeşmesi",
    max_transfers=3
)

if route:
    print(f"Route found!")
    print(f"Origin: {route.origin}")
    print(f"Destination: {route.destination}")
    print(f"Time: {route.total_time} min")
    print(f"Transfers: {route.transfers}")
    print(f"\nSteps ({len(route.steps)}):")
    for i, step in enumerate(route.steps, 1):
        if step.get('type') == 'transit':
            print(f"{i}. {step['line']}: {step['from']} → {step['to']}")
            print(f"   Stops: {step['stops']}, Duration: {step['duration']} min")
else:
    print("No route found")
