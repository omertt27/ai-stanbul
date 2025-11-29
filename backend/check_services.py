"""
Quick fix for service imports in LLM Service Registry
Run this to see which services exist and update the registry
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, '/Users/omer/Desktop/ai-stanbul/backend')

# Check which service files exist
services_dir = '/Users/omer/Desktop/ai-stanbul/services'
backend_services_dir = '/Users/omer/Desktop/ai-stanbul/backend/services'

print("🔍 Checking service files...\n")

service_files = {
    "Restaurant": [
        "restaurant_service.py",
        "restaurant_integration_service.py",
        "google_maps_restaurant_service.py"
    ],
    "Transportation": [
        "transportation_service.py",
        "live_ibb_transportation_service.py",
        "enhanced_bus_route_service.py"
    ],
    "Navigation": [
        "osrm_routing_service.py",
        "walking_directions.py",
        "intelligent_route_finder.py"
    ],
    "POI": [
        "poi_database_service.py",
    ],
    "Weather": [
        "weather_service.py",
        "weather_cache_service.py"
    ]
}

print("=" * 70)
print("EXISTING SERVICE FILES")
print("=" * 70 + "\n")

for category, files in service_files.items():
    print(f"\n📁 {category}:")
    for filename in files:
        # Check in services/ directory
        path1 = os.path.join(services_dir, filename)
        # Check in backend/services/ directory
        path2 = os.path.join(backend_services_dir, filename)
        
        if os.path.exists(path1):
            print(f"   ✅ /services/{filename}")
        elif os.path.exists(path2):
            print(f"   ✅ /backend/services/{filename}")
        else:
            print(f"   ❌ {filename} - NOT FOUND")

print("\n" + "=" * 70)
print("\n💡 Recommendation:")
print("   Update llm_service_registry.py to use the correct import paths")
print("   based on the files marked with ✅ above")
print("\n" + "=" * 70)
