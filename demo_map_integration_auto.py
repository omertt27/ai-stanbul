"""
🗺️ Istanbul AI - Map Integration Demo (Non-Interactive)
Automated demonstration of all map-enabled handlers
"""

import sys
sys.path.insert(0, '/Users/omer/Desktop/ai-stanbul')

# Import the demo functions
from demo_map_integration import (
    print_banner,
    demo_restaurants,
    demo_attractions,
    demo_transportation,
    demo_route_planning,
    demo_hidden_gems,
    demo_summary
)


def main():
    """Run all demos automatically"""
    print_banner("🗺️  ISTANBUL AI - MAP INTEGRATION DEMO (AUTO)")
    print("""
This automated demo will show you:
- Restaurant recommendations with maps
- Attraction suggestions with maps  
- Google Maps-quality transportation routing
- Multi-stop route planning
- Hidden gems visualization

Running all demos automatically...
    """)
    
    # Run all demos
    demo_restaurants()
    print("\n" + "="*80 + "\n")
    
    demo_attractions()
    print("\n" + "="*80 + "\n")
    
    demo_transportation()
    print("\n" + "="*80 + "\n")
    
    demo_route_planning()
    print("\n" + "="*80 + "\n")
    
    demo_hidden_gems()
    print("\n" + "="*80 + "\n")
    
    demo_summary()
    
    print("\n✅ Demo complete! Check the generated JSON files in /tmp/")
    print("\n📁 Generated files:")
    print("   - /tmp/demo_restaurant_map.json")
    print("   - /tmp/demo_attraction_map.json")
    print("   - /tmp/demo_transportation_map.json")
    print("   - /tmp/demo_route_planning_map.json")
    print("   - /tmp/demo_hidden_gems_map.json")


if __name__ == "__main__":
    main()
