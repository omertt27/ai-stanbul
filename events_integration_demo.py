#!/usr/bin/env python3
"""
Comprehensive Demo: Location Detection with Istanbul Events Integration
Shows how the system detects location and recommends nearby events from İKSV and other sources
"""
import asyncio
import json
from datetime import datetime
from backend.services.intelligent_location_detector import detect_user_location, get_events_for_location, LocationConfidence

def format_event_info(event):
    """Format event information for display"""
    info_lines = []
    info_lines.append(f"    🎭 {event.title}")
    info_lines.append(f"       📍 Venue: {event.venue}")
    info_lines.append(f"       🏛️ District: {event.district or 'Unknown'}")
    info_lines.append(f"       🎨 Category: {event.category.value}")
    info_lines.append(f"       🏢 Organizer: {event.organizer or 'Unknown'}")
    
    if event.start_date:
        info_lines.append(f"       📅 Date: {event.start_date.strftime('%Y-%m-%d %H:%M')}")
    
    if event.description:
        info_lines.append(f"       📝 Description: {event.description[:100]}...")
    
    if event.is_free:
        info_lines.append(f"       💰 Free Event")
    
    if event.url:
        info_lines.append(f"       🔗 URL: {event.url}")
    
    return "\n".join(info_lines)

async def demo_events_integration():
    """Demonstrate location detection with events integration"""
    
    print("\n🎭 === ISTANBUL EVENTS INTEGRATION DEMO ===\n")
    print("Demonstrating location detection with nearby events from İKSV and local venues\n")
    
    test_scenarios = [
        {
            "name": "🏛️ Tourist in Sultanahmet",
            "description": "Tourist near major attractions wants cultural events",
            "text": "I'm visiting Hagia Sophia, what cultural events are happening nearby?",
            "context": None,
            "ip": None
        },
        {
            "name": "🎨 Art Lover in Beyoğlu",
            "description": "Art enthusiast in cultural district",
            "text": "I'm in Galata area, looking for art exhibitions",
            "context": None,
            "ip": None
        },
        {
            "name": "🎵 Music Fan in Beşiktaş",
            "description": "Music lover near major venues",
            "text": "GPS: 41.0553, 29.0275, any concerts nearby?",
            "context": None,
            "ip": None
        },
        {
            "name": "🌐 Local in Kadıköy",
            "description": "Local resident on Asian side",
            "text": "What's happening in my neighborhood?",
            "context": {"last_location": {"lat": 41.0066, "lng": 29.0297}},
            "ip": None
        },
        {
            "name": "📍 Central Location",
            "description": "Direct coordinate search for events",
            "text": "Show me events",
            "context": None,
            "ip": None,
            "direct_coords": (41.0369, 28.9850)  # Taksim
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"{'='*70}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*70}")
        print(f"📝 Description: {scenario['description']}")
        print(f"💬 User Query: \"{scenario['text']}\"")
        
        if scenario.get('context'):
            print(f"🔍 Context: {scenario['context']}")
        
        print(f"\n🔄 Processing location detection with events...")
        
        try:
            if 'direct_coords' in scenario:
                # Direct coordinate search
                lat, lng = scenario['direct_coords']
                events = await get_events_for_location(lat, lng)
                print(f"\n📊 DIRECT LOCATION SEARCH:")
                print(f"   📍 Coordinates: {lat}, {lng}")
                print(f"   🎭 Found {len(events)} events nearby")
                
                if events:
                    print(f"\n🎪 NEARBY EVENTS:")
                    for event in events[:5]:  # Show top 5
                        print(format_event_info(event))
                        print()
            else:
                # Standard location detection with events
                location = await detect_user_location(
                    text=scenario['text'],
                    user_context=scenario['context'],
                    ip_address=scenario['ip'],
                    include_events=True
                )
                
                print(f"\n📊 LOCATION DETECTION RESULTS:")
                print(f"   🎯 Confidence: {location.confidence.value.upper()}")
                print(f"   📍 Source: {location.source}")
                
                if location.latitude and location.longitude:
                    print(f"   🌍 Coordinates: {location.latitude:.6f}, {location.longitude:.6f}")
                
                if location.name:
                    print(f"   🏛️ Location: {location.name}")
                
                if location.neighborhood:
                    print(f"   🏘️ Neighborhood: {location.neighborhood}")
                    
                if location.district:
                    print(f"   🏛️ District: {location.district}")
                
                if location.nearby_events:
                    print(f"\n🎪 NEARBY EVENTS ({len(location.nearby_events)} found):")
                    for event in location.nearby_events:
                        print(format_event_info(event))
                        print()
                else:
                    print(f"\n🎪 No events found in this area")
                
                # AI Response Simulation
                print(f"\n🤖 AI PERSONALIZED RESPONSE:")
                if location.nearby_events:
                    event_types = set(e.category.value for e in location.nearby_events)
                    location_desc = location.name or location.neighborhood or location.district or "your area"
                    print(f"   🎭 Great! I found {len(location.nearby_events)} events near {location_desc}!")
                    print(f"   🎨 Event types available: {', '.join(event_types)}")
                    print(f"   🎪 Top recommendation: {location.nearby_events[0].title}")
                    print(f"   📍 At {location.nearby_events[0].venue}")
                else:
                    print(f"   🎭 I couldn't find specific events in your area, but Istanbul")
                    print(f"      always has something happening! Check İKSV's website for updates.")
        
        except Exception as e:
            print(f"❌ Error in scenario: {e}")
        
        print(f"\n{'='*70}\n")
    
    # Test İKSV direct integration
    print(f"🎭 === İKSV EVENTS DIRECT TEST ===")
    try:
        from backend.services.intelligent_location_detector import intelligent_location_detector
        iksv_events = await intelligent_location_detector.fetch_iksv_events()
        
        print(f"📊 İKSV Events Fetched: {len(iksv_events)}")
        
        if iksv_events:
            print(f"\n🎪 SAMPLE İKSV EVENTS:")
            for event in iksv_events[:3]:  # Show first 3
                print(format_event_info(event))
                print()
        else:
            print("⚠️ No İKSV events fetched (website might be down or structure changed)")
            
    except Exception as e:
        print(f"❌ İKSV integration error: {e}")
    
    print(f"\n🎉 === EVENTS INTEGRATION DEMO COMPLETE ===")
    print(f"")
    print(f"📊 CAPABILITIES DEMONSTRATED:")
    print(f"   🎭 Real-time event fetching from İKSV website")
    print(f"   📍 Location-based event recommendations")
    print(f"   🎨 Event categorization (music, theater, art, etc.)")
    print(f"   🏛️ Venue-based event mapping")
    print(f"   🗓️ Static/recurring event integration")
    print(f"   🤖 AI personalization with event context")
    print(f"")
    print(f"✅ The Istanbul Events system is now integrated with")
    print(f"   the Intelligent Location Detection Service!")

if __name__ == "__main__":
    # Set up logging to see detailed process
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s:%(levelname)s:%(message)s')
    
    # Run the events integration demonstration
    asyncio.run(demo_events_integration())
