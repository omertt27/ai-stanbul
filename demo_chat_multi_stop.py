"""
Multi-Stop Chat Integration Demo
=================================

Interactive demo showing how the chat integration works with real queries.

Run this to see:
1. Query analysis and signal detection
2. POI extraction from natural language
3. Itinerary generation
4. Formatted response display

Author: Istanbul AI Team
Date: November 30, 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'services'))
sys.path.insert(0, os.path.dirname(__file__))

from ai_chat_route_integration import get_chat_route_handler
import logging

# Setup logging (reduce noise)
logging.basicConfig(level=logging.WARNING)

def print_separator(char="=", length=80):
    print(char * length)

def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)

def analyze_query(handler, query):
    """Analyze a query and show detection results"""
    print(f"\n📝 Query: \"{query}\"\n")
    
    # Step 1: Is it a route request?
    is_route = handler._is_route_request(query)
    print(f"   ✓ Route Request: {'YES' if is_route else 'NO'}")
    
    if not is_route:
        print("   → Will be handled by general chat system")
        return
    
    # Step 2: Single or multi-stop?
    is_multi = handler._is_multi_stop_request(query)
    query_type = "Multi-Stop Itinerary" if is_multi else "Single Route"
    print(f"   ✓ Query Type: {query_type}")
    
    # Step 3: Extract POIs
    pois = handler._extract_poi_names(query)
    if pois:
        print(f"   ✓ Locations Found: {len(pois)}")
        for i, poi in enumerate(pois, 1):
            print(f"      {i}. {poi}")
    else:
        print("   ✗ No locations identified")
    
    return is_multi, pois


def demo_query(handler, query, show_full_response=False):
    """Process a query and show results"""
    print_section(f"DEMO: {query[:60]}...")
    
    # Analyze
    analysis = analyze_query(handler, query)
    
    if not analysis:
        return
    
    is_multi, pois = analysis
    
    # Process
    print(f"\n🔄 Processing request...")
    
    try:
        result = handler.handle_route_request(query)
        
        if result:
            result_type = result['type']
            print(f"   ✓ Response Type: {result_type}")
            
            if result_type == 'multi_stop_itinerary':
                route_data = result['route_data']
                summary = route_data['summary']
                
                print(f"\n📊 Itinerary Summary:")
                print(f"   • Stops: {summary['total_stops']}")
                print(f"   • Distance: {summary['total_distance_km']:.2f} km")
                print(f"   • Travel Time: {summary['total_travel_time_min']} min")
                print(f"   • Visit Time: {summary['total_visit_time_min']} min")
                print(f"   • Total Time: {summary['total_time_min']} min (~{summary['total_time_min']/60:.1f} hours)")
                print(f"   • Cost: ₺{summary['total_cost_tl']:.2f}")
                print(f"   • Strategy: {summary['strategy']}")
                print(f"   • Accessible: {'Yes' if summary['accessibility_friendly'] else 'No'}")
                
                print(f"\n📍 Route Order:")
                for i, stop in enumerate(route_data['stops'], 1):
                    print(f"   {i}. {stop['name']} ({stop['duration']} min, {stop['category']})")
                
                if show_full_response:
                    print(f"\n💬 User Response:")
                    print_separator("-")
                    print(result['message'])
                    print_separator("-")
            
            elif result_type == 'error':
                print(f"\n⚠️ Error: {result['message']}")
            
            else:
                print(f"\n✓ Single route planned (not showing details)")
        
        else:
            print("   ✗ No result (not a route request)")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run interactive demo"""
    print_section("🗺️ MULTI-STOP CHAT INTEGRATION DEMO")
    print("\nInitializing chat handler...")
    
    handler = get_chat_route_handler()
    
    if not handler.multi_stop_planner:
        print("\n⚠️ WARNING: Multi-stop planner not fully initialized")
        print("Some features may not work correctly in demo mode.\n")
    
    print("✅ Ready!\n")
    
    # Demo Queries
    demo_queries = [
        # Multi-stop examples
        ("Plan a tour of Hagia Sophia, Blue Mosque, and Grand Bazaar", True),
        ("Visit Topkapi Palace, Basilica Cistern, and Dolmabahce Palace", True),
        ("I want to see Galata Tower, Istiklal Street, and Taksim today", True),
        ("Create an accessible itinerary for museums in Sultanahmet", False),
        
        # Single route examples
        ("Route from Sultanahmet to Galata Tower", False),
        ("How do I get to Taksim Square?", False),
        
        # Edge cases
        ("Plan a day trip", False),
        ("Tell me about Istanbul", False),
    ]
    
    for i, (query, show_full) in enumerate(demo_queries, 1):
        demo_query(handler, query, show_full_response=show_full)
        
        if i < len(demo_queries):
            input(f"\n{'─'*80}\nPress Enter to continue to next demo...")
    
    # Interactive mode
    print_section("🎮 INTERACTIVE MODE")
    print("\nNow you can try your own queries!")
    print("Type 'quit' or 'exit' to end the demo.\n")
    
    while True:
        try:
            query = input("\n💬 Your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for trying the demo!")
                break
            
            demo_query(handler, query, show_full_response=True)
        
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Goodbye!")
            break
        except EOFError:
            break
    
    print_section("DEMO COMPLETE")
    print("\n📚 For more information, see:")
    print("   • MULTI_STOP_CHAT_INTEGRATION_COMPLETE.md")
    print("   • MULTI_STOP_CHAT_INTEGRATION_QUICK_REFERENCE.md")
    print("   • test_chat_multi_stop_integration.py\n")


if __name__ == "__main__":
    main()
