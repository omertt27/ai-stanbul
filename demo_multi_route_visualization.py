#!/usr/bin/env python3
"""
Interactive demonstration of multi-route visualization.

Shows how the system now displays multiple route alternatives with different colors.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.transportation_rag_system import IstanbulTransportationRAG
import json

def demo_multi_route_visualization():
    """Demo showing multi-route visualization in action."""
    
    print("\n" + "="*80)
    print("MULTI-ROUTE VISUALIZATION DEMO")
    print("="*80)
    print("\nThis demo shows how the chat frontend now displays ALL route alternatives")
    print("on the map simultaneously with different colors.\n")
    
    transport_rag = IstanbulTransportationRAG()
    
    # Example query
    origin = "Taksim"
    destination = "Kadıköy"
    
    print(f"Query: '{origin} to {destination}'\n")
    
    # Get routes
    route = transport_rag.find_route(origin, destination)
    
    if not route:
        print("❌ No route found")
        return
    
    print(f"✅ Found route with {len(route.alternatives) if route.alternatives else 0} alternatives\n")
    
    # Get map data
    map_data = transport_rag.get_map_data_for_last_route()
    
    if not map_data:
        print("❌ No map data generated")
        return
    
    # Display what the frontend will show
    print("="*80)
    print("WHAT THE FRONTEND WILL DISPLAY")
    print("="*80)
    
    routes = map_data.get('routes', [])
    
    print(f"\n📍 Map Center: {map_data.get('center')}")
    print(f"🔍 Zoom Level: {map_data.get('zoom')}")
    print(f"\n🗺️  Total Routes to Display: {len(routes)}\n")
    
    for idx, route_data in enumerate(routes, 1):
        is_main = route_data.get('isMain', False)
        route_type = "MAIN ROUTE" if is_main else f"ALTERNATIVE {idx - 1}"
        
        print("─" * 80)
        print(f"{route_type}")
        print("─" * 80)
        print(f"Description: {route_data.get('description')}")
        print(f"Color:       {route_data.get('color')} {'■' * 10}")
        print(f"Weight:      {route_data.get('weight')} (line thickness)")
        print(f"Opacity:     {route_data.get('opacity')} (visibility)")
        print(f"Style:       {'Solid Line' if is_main else 'Dashed Line (10, 5)'}")
        print(f"Coordinates: {len(route_data.get('coordinates', []))} waypoints")
        
        # Show first and last coordinates
        coords = route_data.get('coordinates', [])
        if coords:
            print(f"  Start: {coords[0]}")
            print(f"  End:   {coords[-1]}")
        print()
    
    # Show markers
    markers = map_data.get('markers', [])
    if markers:
        print("="*80)
        print(f"📌 MARKERS ({len(markers)})")
        print("="*80)
        for marker in markers:
            print(f"  {marker.get('type', 'poi').upper()}: {marker.get('label')} at ({marker.get('lat'):.4f}, {marker.get('lon'):.4f})")
        print()
    
    # Show metadata
    metadata = map_data.get('metadata', {})
    print("="*80)
    print("📊 METADATA")
    print("="*80)
    print(f"  Total Time:     {metadata.get('total_time')} minutes")
    print(f"  Total Distance: {metadata.get('total_distance'):.2f} km")
    print(f"  Transfers:      {metadata.get('transfers')}")
    print(f"  Lines Used:     {', '.join(metadata.get('lines_used', []))}")
    print(f"  Alternatives:   {metadata.get('alternatives_count')}")
    print()
    
    # Show JSON structure (for developers)
    print("="*80)
    print("JSON STRUCTURE (for developers)")
    print("="*80)
    print(json.dumps({
        'type': map_data.get('type'),
        'markers': f"{len(markers)} markers",
        'routes': [
            {
                'description': r.get('description'),
                'color': r.get('color'),
                'isMain': r.get('isMain', False),
                'coordinates': f"{len(r.get('coordinates', []))} points"
            }
            for r in routes
        ],
        'center': map_data.get('center'),
        'zoom': map_data.get('zoom'),
        'metadata': {
            'alternatives_count': metadata.get('alternatives_count')
        }
    }, indent=2))
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION READY FOR FRONTEND")
    print("="*80)
    print("\nThe chat frontend will now display:")
    print(f"  • {len(routes)} colored routes on the map")
    print("  • Main route in bold blue")
    print("  • Alternative routes in red/yellow/green with dashed lines")
    print("  • All routes visible simultaneously for easy comparison")
    print()

if __name__ == '__main__':
    demo_multi_route_visualization()
