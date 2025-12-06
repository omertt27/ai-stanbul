#!/usr/bin/env python3
"""
Quick validation script to check if the map integration fix is correct.
This validates the logic without needing to run the full backend.
"""

import re
from typing import Dict, Any, Optional, List

def _generate_map_from_context(
    context: Dict[str, Any],
    signals: Dict[str, bool],
    user_location: Optional[Dict[str, float]]
) -> Optional[Dict[str, Any]]:
    """
    Generate basic map_data from database context for location-based queries.
    This is the NEW method added to core.py
    """
    try:
        markers = []
        locations = []
        
        # Try to extract locations from database context string
        db_context = context.get('database', '')
        
        if not db_context:
            print("   ❌ No database context found")
            return None
        
        print(f"   📄 Database context length: {len(db_context)} chars")
        
        # Pattern to find coordinates in the database context
        coord_patterns = [
            r'Coordinates:\s*\(([0-9.]+),\s*([0-9.]+)\)',
            r'lat(?:itude)?:\s*([0-9.]+)[,\s]+lon(?:gitude)?:\s*([0-9.]+)',
            r'\(([0-9.]+),\s*([0-9.]+)\)',  # Simple tuple format
        ]
        
        # Pattern to find location names (before coordinates)
        name_pattern = r'([A-ZÇĞİÖŞÜ][A-Za-zçğıöşü\s&\'-]+?)(?:\s*[-:]\s*|\s+Coordinates:)'
        
        # Extract all coordinate pairs
        for pattern in coord_patterns:
            for match in re.finditer(pattern, db_context):
                try:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    
                    # Find the name before this coordinate
                    name = "Location"
                    text_before = db_context[:match.start()]
                    name_match = re.search(name_pattern, text_before[-200:])
                    if name_match:
                        name = name_match.group(1).strip()
                    
                    locations.append({
                        'name': name,
                        'lat': lat,
                        'lon': lon
                    })
                    print(f"   ✅ Extracted: {name} at ({lat}, {lon})")
                except (ValueError, IndexError):
                    continue
        
        # If we found locations, create markers
        if locations:
            for loc in locations:
                markers.append({
                    "position": {"lat": loc['lat'], "lng": loc['lon']},
                    "label": loc['name'],
                    "type": "restaurant" if signals.get('needs_restaurant') else "attraction"
                })
            
            # Calculate center point
            avg_lat = sum(loc['lat'] for loc in locations) / len(locations)
            avg_lon = sum(loc['lon'] for loc in locations) / len(locations)
            
            # Add user location marker if available
            if user_location:
                markers.append({
                    "position": {"lat": user_location['lat'], "lng": user_location['lon']},
                    "label": "Your Location",
                    "type": "user"
                })
                # Recalculate center to include user location
                avg_lat = (avg_lat * len(locations) + user_location['lat']) / (len(locations) + 1)
                avg_lon = (avg_lon * len(locations) + user_location['lon']) / (len(locations) + 1)
            
            map_data = {
                "type": "markers",
                "markers": markers,
                "center": {"lat": avg_lat, "lng": avg_lon},
                "zoom": 13,
                "has_origin": user_location is not None,
                "has_destination": False,
                "origin_name": "Your Location" if user_location else None,
                "destination_name": None,
                "locations_count": len(locations)
            }
            
            print(f"   ✅ Generated map_data with {len(markers)} markers")
            return map_data
        else:
            print(f"   ⚠️ No locations found in database context")
        
        return None
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

# Test cases
test_cases = [
    {
        "name": "Restaurant Context with Coordinates",
        "context": {
            "database": """=== RESTAURANTS ===

Dürümçün Kebap - Traditional Kebab Restaurant in Kadıköy
Coordinates: (40.9886, 29.0318)
Address: Caferağa Mahallesi, Moda Cad. No:134, Kadıköy
Cuisine: Turkish Kebab
Price Range: $$

Çiya Sofrası - Authentic Anatolian Cuisine
Coordinates: (40.9919, 29.0262)
Address: Caferağa Mahallesi, Güneslibahçe Sk. No:43, Kadıköy
Cuisine: Turkish Regional
Price Range: $$$"""
        },
        "signals": {"needs_restaurant": True},
        "user_location": {"lat": 41.0082, "lon": 28.9784}
    },
    {
        "name": "Attraction Context",
        "context": {
            "database": """=== ATTRACTIONS ===

Blue Mosque (Sultan Ahmed Mosque)
Coordinates: (41.0054, 28.9768)
Description: Historic mosque with six minarets
Address: Sultanahmet Square"""
        },
        "signals": {"needs_attraction": True},
        "user_location": None
    },
    {
        "name": "No Coordinates",
        "context": {
            "database": """=== RESTAURANTS ===

Some restaurant without coordinates
Just text description"""
        },
        "signals": {"needs_restaurant": True},
        "user_location": None
    }
]

print("=" * 80)
print("🧪 MAP GENERATION FROM CONTEXT - VALIDATION TEST")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['name']}")
    print("-" * 80)
    
    result = _generate_map_from_context(
        context=test['context'],
        signals=test['signals'],
        user_location=test['user_location']
    )
    
    if result:
        print(f"\n✅ SUCCESS")
        print(f"   Type: {result['type']}")
        print(f"   Markers: {len(result['markers'])}")
        print(f"   Center: ({result['center']['lat']:.4f}, {result['center']['lng']:.4f})")
        print(f"   Locations: {result.get('locations_count', 0)}")
        if result.get('has_origin'):
            print(f"   Origin: {result['origin_name']}")
    else:
        print(f"\n❌ FAILED - No map_data generated")

print("\n" + "=" * 80)
print("Validation Complete")
print("=" * 80)
