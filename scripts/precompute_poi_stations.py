#!/usr/bin/env python3
"""
POI-Station Connection Precomputation Script
============================================

This script precomputes the nearest 3-5 transit stations for each POI in the database.
It calculates:
- Distance in km (haversine formula)
- Walking time in minutes (assuming 5 km/h walking speed)
- Stores results in POI database

Phase 1.2 of POI Enhanced Route Planning Implementation
"""

import json
import math
from typing import List, Dict, Tuple
from pathlib import Path

# Configuration
POI_DATABASE_PATH = Path(__file__).parent.parent / "data" / "istanbul_pois.json"
TRANSIT_MAP_PATH = Path(__file__).parent.parent / "data" / "istanbul_metro_map.geojson"
MAX_STATIONS_PER_POI = 5
WALKING_SPEED_KM_H = 5.0  # Average walking speed
MAX_WALKING_DISTANCE_KM = 2.0  # Don't include stations beyond 2km


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def calculate_walking_time(distance_km: float) -> int:
    """Calculate walking time in minutes based on distance."""
    return int((distance_km / WALKING_SPEED_KM_H) * 60)


def load_transit_stations() -> List[Dict]:
    """Load all transit stations from GeoJSON file."""
    print(f"Loading transit stations from: {TRANSIT_MAP_PATH}")
    
    with open(TRANSIT_MAP_PATH, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    stations = []
    for feature in geojson_data.get('features', []):
        if feature['geometry']['type'] == 'Point':
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            
            station = {
                'station_id': props.get('stop_id', ''),
                'name': props.get('name', ''),
                'name_en': props.get('name_en', props.get('name', '')),
                'lon': coords[0],  # GeoJSON is [lon, lat]
                'lat': coords[1],
                'type': props.get('stop_type', 'metro')
            }
            stations.append(station)
    
    print(f"✓ Loaded {len(stations)} transit stations")
    return stations


def find_nearest_stations(poi: Dict, stations: List[Dict]) -> List[Tuple[str, float, int]]:
    """
    Find the nearest 3-5 transit stations for a POI.
    
    Returns:
        List of tuples: (station_id, distance_km, walk_time_min)
    """
    poi_lat = poi['location']['lat']
    poi_lon = poi['location']['lon']
    
    # Calculate distances to all stations
    distances = []
    for station in stations:
        distance_km = haversine_distance(
            poi_lat, poi_lon,
            station['lat'], station['lon']
        )
        
        # Only include stations within reasonable walking distance
        if distance_km <= MAX_WALKING_DISTANCE_KM:
            walk_time = calculate_walking_time(distance_km)
            distances.append((
                station['station_id'],
                round(distance_km, 3),
                walk_time
            ))
    
    # Sort by distance and take the nearest 3-5 stations
    distances.sort(key=lambda x: x[1])
    return distances[:MAX_STATIONS_PER_POI]


def precompute_all_connections():
    """Main function to precompute POI-station connections."""
    print("=" * 70)
    print("POI-Station Connection Precomputation")
    print("=" * 70)
    print()
    
    # Load data
    print("Step 1: Loading data...")
    stations = load_transit_stations()
    
    print(f"Loading POI database from: {POI_DATABASE_PATH}")
    with open(POI_DATABASE_PATH, 'r', encoding='utf-8') as f:
        poi_data = json.load(f)
    
    pois = poi_data.get('pois', [])
    print(f"✓ Loaded {len(pois)} POIs")
    print()
    
    # Precompute connections
    print("Step 2: Computing nearest stations for each POI...")
    print("-" * 70)
    
    updated_count = 0
    for i, poi in enumerate(pois):
        poi_name = poi.get('name_en', poi.get('name', 'Unknown'))
        print(f"\n[{i+1}/{len(pois)}] Processing: {poi_name}")
        
        # Find nearest stations
        nearest = find_nearest_stations(poi, stations)
        
        if nearest:
            poi['nearest_stations'] = nearest
            updated_count += 1
            
            print(f"  Found {len(nearest)} nearby stations:")
            for station_id, distance, walk_time in nearest:
                # Find station name
                station = next((s for s in stations if s['station_id'] == station_id), None)
                station_name = station['name'] if station else station_id
                print(f"    • {station_name:30s} - {distance:.2f}km ({walk_time}min walk)")
        else:
            print(f"  ⚠ No stations within {MAX_WALKING_DISTANCE_KM}km")
            poi['nearest_stations'] = []
    
    print()
    print("-" * 70)
    print(f"✓ Updated {updated_count} POIs with station connections")
    print()
    
    # Save updated database
    print("Step 3: Saving updated POI database...")
    with open(POI_DATABASE_PATH, 'w', encoding='utf-8') as f:
        json.dump(poi_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved to: {POI_DATABASE_PATH}")
    print()
    
    # Generate statistics
    print("=" * 70)
    print("PRECOMPUTATION STATISTICS")
    print("=" * 70)
    
    total_connections = sum(len(poi.get('nearest_stations', [])) for poi in pois)
    pois_with_connections = sum(1 for poi in pois if poi.get('nearest_stations'))
    
    print(f"Total POIs:                    {len(pois)}")
    print(f"POIs with station connections: {pois_with_connections}")
    print(f"POIs without connections:      {len(pois) - pois_with_connections}")
    print(f"Total station connections:     {total_connections}")
    print(f"Average connections per POI:   {total_connections / len(pois):.1f}")
    print()
    
    # Show POIs without connections
    pois_without = [poi for poi in pois if not poi.get('nearest_stations')]
    if pois_without:
        print("POIs without nearby stations (>2km from transit):")
        for poi in pois_without:
            poi_name = poi.get('name_en', poi.get('name', 'Unknown'))
            print(f"  - {poi_name}")
        print()
    
    print("=" * 70)
    print("✓ Phase 1.2 Complete: POI-Station Connections Precomputed!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Verify connections: python scripts/verify_poi_connections.py")
    print("  2. Test POI database service with new connections")
    print("  3. Proceed to Phase 1.3: Integrate POIs into route planner")


if __name__ == '__main__':
    try:
        precompute_all_connections()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
