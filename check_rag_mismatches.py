#!/usr/bin/env python3
"""
Comprehensive RAG System Mismatch Checker
Compares canonical station IDs with RAG system aliases and station graph
"""

import sys
sys.path.insert(0, 'backend')

from services.transportation_station_normalization import StationNormalizer
from services.transportation_rag_system import IstanbulTransportationRAG

def print_header(text):
    print(f"\n{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}\n")

def print_section(text):
    print(f"\n{'-'*80}")
    print(f"{text}")
    print(f"{'-'*80}")

# Initialize systems
print("Initializing systems...")
canonical_normalizer = StationNormalizer()
rag_system = IstanbulTransportationRAG()

print(f"✅ Canonical system has {len(canonical_normalizer.stations)} stations")
print(f"✅ RAG system has {len(rag_system.stations)} stations")

# Check 1: Station ID mismatches
print_header("CHECK 1: Station ID Mismatches")

canonical_ids = {station.canonical_id for station in canonical_normalizer.stations}
rag_ids = set(rag_system.stations.keys())

print(f"Canonical IDs count: {len(canonical_ids)}")
print(f"RAG IDs count: {len(rag_ids)}")

# IDs in canonical but not in RAG
missing_in_rag = canonical_ids - rag_ids
if missing_in_rag:
    print(f"\n❌ {len(missing_in_rag)} stations in Canonical but NOT in RAG:")
    for station_id in sorted(missing_in_rag)[:20]:  # Show first 20
        station = next(s for s in canonical_normalizer.stations if s.canonical_id == station_id)
        print(f"   {station_id}: {station.name_tr} ({station.line_id})")
    if len(missing_in_rag) > 20:
        print(f"   ... and {len(missing_in_rag) - 20} more")
else:
    print("✅ All canonical stations are in RAG system")

# IDs in RAG but not in canonical
extra_in_rag = rag_ids - canonical_ids
if extra_in_rag:
    print(f"\n❌ {len(extra_in_rag)} stations in RAG but NOT in Canonical:")
    for station_id in sorted(extra_in_rag)[:20]:
        station = rag_system.stations[station_id]
        print(f"   {station_id}: {station.name} ({station.line})")
    if len(extra_in_rag) > 20:
        print(f"   ... and {len(extra_in_rag) - 20} more")
else:
    print("✅ No extra stations in RAG system")

# Check 2: Alias mismatches
print_header("CHECK 2: Station Alias Mismatches")

alias_issues = []
for alias, station_ids in rag_system.station_aliases.items():
    for station_id in station_ids:
        if station_id not in rag_system.stations:
            alias_issues.append((alias, station_id, "ID not found in RAG stations"))

if alias_issues:
    print(f"❌ Found {len(alias_issues)} alias issues:")
    for alias, station_id, issue in alias_issues[:30]:
        print(f"   Alias '{alias}' → '{station_id}': {issue}")
    if len(alias_issues) > 30:
        print(f"   ... and {len(alias_issues) - 30} more")
else:
    print("✅ All aliases point to valid station IDs")

# Check 3: Key stations that should have aliases
print_header("CHECK 3: Important Stations - Alias Coverage")

important_stations = [
    ("Mecidiyeköy", ["M7", "M2"]),
    ("Olimpiyat", ["M3", "M9"]),
    ("İkitelli Sanayi", ["M3", "M9"]),
    ("Taksim", ["M2"]),
    ("Gayrettepe", ["M2", "M11"]),
    ("Istanbul Airport", ["M11"]),
    ("Kadıköy", ["M4"]),
    ("Üsküdar", ["M5"]),
]

for station_name, expected_lines in important_stations:
    normalized = rag_system._normalize_station_name(station_name)
    
    # Check if it has an alias
    has_alias = normalized in rag_system.station_aliases
    
    # Check if we can find it via _get_stations_for_location
    found_stations = rag_system._get_stations_for_location(station_name)
    
    print(f"\n{station_name}:")
    print(f"   Normalized: '{normalized}'")
    print(f"   Has alias: {has_alias}")
    if has_alias:
        print(f"   Alias maps to: {rag_system.station_aliases[normalized]}")
    print(f"   Can find via lookup: {len(found_stations) > 0}")
    if found_stations:
        for sid in found_stations:
            if sid in rag_system.stations:
                station = rag_system.stations[sid]
                print(f"   ✅ Found: {sid} ({station.name}, Line: {station.line})")
            else:
                print(f"   ❌ Found ID but not in stations: {sid}")
    
    # Check canonical system
    canonical_station = canonical_normalizer.normalize_station(station_name)
    print(f"   Canonical ID: {canonical_station.canonical_id}")
    print(f"   Canonical Lines: {canonical_station.transfers + [canonical_station.line_id]}")

# Check 4: Line-specific checks
print_header("CHECK 4: New Lines (M9, M11, T4, T5) Station Coverage")

new_lines = {
    "M9": ["Olimpiyat", "İkitelli Sanayi"],
    "M11": ["Gayrettepe", "Kağıthane", "İstanbul Havalimanı"],
    "T4": ["Topkapı", "Mescid-i Selam"],
    "T5": ["Cibali", "Alibeyköy"]
}

for line_id, sample_stations in new_lines.items():
    print(f"\n{line_id} Line:")
    
    # Count stations in canonical
    canonical_line_stations = [s for s in canonical_normalizer.stations if s.line_id == line_id]
    print(f"   Canonical system: {len(canonical_line_stations)} stations")
    
    # Count stations in RAG
    rag_line_stations = [s for s in rag_system.stations.values() if s.line == line_id]
    print(f"   RAG system: {len(rag_line_stations)} stations")
    
    # Check sample stations
    for station_name in sample_stations[:3]:
        found = rag_system._get_stations_for_location(station_name)
        line_match = any(rag_system.stations.get(sid, type('obj', (), {'line': None})).line == line_id for sid in found if sid in rag_system.stations)
        status = "✅" if found and line_match else "❌"
        print(f"   {status} {station_name}: {'Found' if found else 'Not found'}")

# Check 5: Neighborhood mappings
print_header("CHECK 5: Neighborhood Mapping Issues")

neighborhood_issues = []
for neighborhood, station_ids in rag_system.neighborhoods.items():
    for station_id in station_ids:
        if station_id not in rag_system.stations:
            neighborhood_issues.append((neighborhood, station_id))

if neighborhood_issues:
    print(f"❌ Found {len(neighborhood_issues)} neighborhood mapping issues:")
    for neighborhood, station_id in neighborhood_issues[:20]:
        print(f"   '{neighborhood}' → '{station_id}': ID not found in RAG stations")
    if len(neighborhood_issues) > 20:
        print(f"   ... and {len(neighborhood_issues) - 20} more")
else:
    print("✅ All neighborhood mappings point to valid station IDs")

# Summary
print_header("SUMMARY")

total_issues = len(missing_in_rag) + len(extra_in_rag) + len(alias_issues) + len(neighborhood_issues)

if total_issues == 0:
    print("🎉 NO MISMATCHES FOUND! System is fully aligned.")
else:
    print(f"⚠️  Found {total_issues} total issues:")
    print(f"   - Stations in Canonical but not RAG: {len(missing_in_rag)}")
    print(f"   - Stations in RAG but not Canonical: {len(extra_in_rag)}")
    print(f"   - Alias mapping issues: {len(alias_issues)}")
    print(f"   - Neighborhood mapping issues: {len(neighborhood_issues)}")
    print("\n🔧 These mismatches need to be fixed to ensure proper routing!")

