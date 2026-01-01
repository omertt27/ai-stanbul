#!/usr/bin/env python3
"""
Check the actual Marmaray stations between Sirkeci and Ayrılık Çeşmesi
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.transportation_rag_system import get_transportation_rag

# Get RAG system
rag = get_transportation_rag()

# Check Marmaray stations
print("MARMARAY stations in the graph:")
print("=" * 80)

marmaray_stations = [sid for sid in rag.station_graph.keys() if sid.startswith('MARMARAY-')]
marmaray_stations_sorted = sorted(marmaray_stations)

for station_id in marmaray_stations_sorted:
    station = rag.station_graph[station_id]
    print(f"{station_id}: {station.name}")

print(f"\nTotal MARMARAY stations: {len(marmaray_stations)}")

# Check M4 stations
print("\n" + "=" * 80)
print("M4 stations in the graph:")
print("=" * 80)

m4_stations = [sid for sid in rag.station_graph.keys() if sid.startswith('M4-')]
m4_stations_sorted = sorted(m4_stations)

for station_id in m4_stations_sorted:
    station = rag.station_graph[station_id]
    print(f"{station_id}: {station.name}")

print(f"\nTotal M4 stations: {len(m4_stations)}")

# Check connections from Sirkeci
print("\n" + "=" * 80)
print("Connections from MARMARAY-Sirkeci:")
print("=" * 80)

sirkeci_id = "MARMARAY-Sirkeci"
if sirkeci_id in rag.station_graph:
    sirkeci = rag.station_graph[sirkeci_id]
    for conn in sirkeci.connections:
        print(f"  → {conn}")

# Check connections from Ayrılık Çeşmesi (both MARMARAY and M4)
print("\n" + "=" * 80)
print("Connections from MARMARAY-Ayrılık Çeşmesi:")
print("=" * 80)

ayrilik_marmaray_id = "MARMARAY-Ayrılık Çeşmesi"
if ayrilik_marmaray_id in rag.station_graph:
    ayrilik_m = rag.station_graph[ayrilik_marmaray_id]
    for conn in ayrilik_m.connections:
        print(f"  → {conn}")

print("\n" + "=" * 80)
print("Connections from M4-Ayrılık Çeşmesi:")
print("=" * 80)

ayrilik_m4_id = "M4-Ayrılık Çeşmesi"
if ayrilik_m4_id in rag.station_graph:
    ayrilik_m4 = rag.station_graph[ayrilik_m4_id]
    for conn in ayrilik_m4.connections:
        print(f"  → {conn}")
