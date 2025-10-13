#!/usr/bin/env python3
"""
Integration example showing how other AI modules can use the Monthly Events Scheduler
"""

import asyncio
from monthly_events_scheduler import fetch_monthly_events, get_cached_events, check_if_fetch_needed

async def example_integration():
    """Example showing how other modules can integrate with the events scheduler"""
    
    print("🔄 AI Istanbul - Events Integration Example")
    print("=" * 50)
    
    # 1. Check if we need to fetch new events
    needs_fetch = check_if_fetch_needed()
    print(f"📅 Fetch needed: {needs_fetch}")
    
    # 2. Try to get cached events first
    cached_events = get_cached_events()
    print(f"📂 Cached events available: {len(cached_events)}")
    
    # 3. If we need fresh data or don't have cached events, fetch new ones
    if needs_fetch or len(cached_events) == 0:
        print("\n🌐 Fetching fresh events from İKSV...")
        fetch_result = await fetch_monthly_events()
        print(f"✅ Fetch completed: {fetch_result.get('success', False)}")
        print(f"📊 Events fetched: {fetch_result.get('events_fetched', 0)}")
        
        # Get the updated cached events
        cached_events = get_cached_events()
    
    # 4. Use the events in your AI module
    print(f"\n🎭 Using {len(cached_events)} events for AI processing...")
    
    # Example: Filter events by category
    theatre_events = [e for e in cached_events if e.get('category') == 'Theatre']
    music_events = [e for e in cached_events if e.get('category') in ['Music', 'Salon İKSV']]
    art_events = [e for e in cached_events if e.get('category') == 'Art']
    
    print(f"🎭 Theatre events: {len(theatre_events)}")
    print(f"🎵 Music events: {len(music_events)}")
    print(f"🎨 Art events: {len(art_events)}")
    
    # Example: Show upcoming events
    print(f"\n📋 Sample upcoming events:")
    for i, event in enumerate(cached_events[:5], 1):
        print(f"   {i}. {event['title'][:50]}...")
        print(f"      📅 {event.get('date_str', 'Date TBA')}")
        print(f"      📍 {event.get('venue', 'Venue TBA')}")
        print()
    
    return cached_events

def example_event_filtering(events):
    """Example of how to filter and process events for different AI use cases"""
    
    print("🔍 Event Filtering Examples for AI Integration")
    print("=" * 50)
    
    # Filter by date (events in October)
    october_events = [e for e in events if 'october' in e.get('date_str', '').lower()]
    print(f"📅 October events: {len(october_events)}")
    
    # Filter by venue type
    zorlu_events = [e for e in events if 'zorlu' in e.get('venue', '').lower()]
    salon_events = [e for e in events if 'salon' in e.get('venue', '').lower()]
    
    print(f"🏢 Zorlu PSM events: {len(zorlu_events)}")
    print(f"🎪 Salon İKSV events: {len(salon_events)}")
    
    # Filter by event type keywords
    performance_events = [e for e in events if any(keyword in e['title'].lower() 
                                                  for keyword in ['performance', 'ballet', 'dance', 'theatre'])]
    
    print(f"🎭 Performance events: {len(performance_events)}")
    
    # Extract event info for AI responses
    event_summaries = []
    for event in events[:3]:
        summary = {
            'title': event['title'],
            'when': event.get('date_str', 'Date TBA'),
            'where': event.get('venue', 'Venue TBA'),
            'type': event.get('category', 'Event'),
            'description': f"A {event.get('category', 'cultural')} event at {event.get('venue', 'İKSV venue')}"
        }
        event_summaries.append(summary)
    
    print(f"\n📊 Processed {len(event_summaries)} events for AI responses")
    return event_summaries

if __name__ == "__main__":
    async def main():
        # Test the integration
        events = await example_integration()
        
        # Test filtering
        summaries = example_event_filtering(events)
        
        print("\n✅ Integration test completed!")
        print("💡 This demonstrates how other AI modules can:")
        print("   - Check for fresh event data")
        print("   - Load cached events efficiently")
        print("   - Filter events by category, venue, date")
        print("   - Process events for AI responses")
        print("   - Integrate with chatbots, recommendations, etc.")
    
    asyncio.run(main())
