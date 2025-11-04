#!/usr/bin/env python3
"""
Export IKSV events database to semantic search indexes
Integrates the IKSV events into ML system for event queries
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_systems.semantic_search_engine import SemanticSearchEngine
from backend.data.events_database import get_iksv_events_only, SEASONAL_EVENTS
import json
from pathlib import Path

def export_iksv_events_to_semantic_index():
    """Export all IKSV events to semantic search"""
    print("\n" + "="*70)
    print("🎭 Exporting IKSV Events Database to Semantic Search")
    print("="*70)
    
    # Get all IKSV events (static festivals + live events)
    iksv_events = get_iksv_events_only()
    
    # Convert to semantic search format
    events = []
    for event in iksv_events:
        # Handle both formats (static and live)
        name = event.get('name', {})
        if isinstance(name, dict):
            name_str = name.get('en', name.get('tr', 'Unknown Event'))
        else:
            name_str = name
        
        description = event.get('description', {})
        if isinstance(description, dict):
            desc_str = description.get('en', description.get('tr', ''))
        else:
            desc_str = str(description)
        
        # Get location
        location = event.get('location', event.get('venue', 'Various locations'))
        
        # Get date info
        date_str = event.get('date_str', '')
        month = event.get('month', '')
        if month and not date_str:
            date_str = f"Month: {month}"
        
        # Get event type
        event_type = event.get('type', 'cultural')
        category = event.get('category', event_type)
        
        # Build comprehensive description
        full_description = f"{desc_str} "
        if date_str:
            full_description += f"Date: {date_str}. "
        if event.get('time'):
            full_description += f"Time: {event['time']}. "
        
        # Get cost info
        cost = event.get('cost', {})
        if isinstance(cost, dict):
            cost_str = cost.get('en', cost.get('tr', 'Price varies'))
        else:
            cost_str = str(cost)
        
        event_item = {
            "id": f"iksv_{event.get('id', event.get('event_number', ''))}",
            "name": name_str,
            "type": "event",
            "category": category,
            "description": full_description[:500],  # Limit length
            "location": location,
            "date": date_str,
            "time": event.get('time', ''),
            "price": cost_str,
            "tags": [
                category,
                "iksv",
                "istanbul",
                event_type,
                "cultural"
            ] + event.get('tags', []),
            "website": event.get('website', 'www.iksv.org'),
            "ticket_url": event.get('ticket_url', 'https://www.iksv.org/tr/bilet'),
            "source": event.get('source', 'IKSV'),
            "is_live": event.get('is_live', False)
        }
        events.append(event_item)
    
    print(f"📦 Exported {len(events)} IKSV events from database")
    
    # Show breakdown
    static_events = [e for e in events if not e.get('is_live')]
    live_events = [e for e in events if e.get('is_live')]
    print(f"   • Static major festivals: {len(static_events)}")
    print(f"   • Live/current events: {len(live_events)}")
    
    # Create semantic search engine
    search_engine = SemanticSearchEngine()
    
    # Index events
    search_engine.index_items(events, save_path="./data/events_index.bin")
    
    print("✅ IKSV events indexed successfully!")
    print(f"   File: ./data/events_index.bin")
    print(f"   Items: {len(events)}")
    
    # Show sample
    print("\n📋 Sample events indexed:")
    categories = {}
    for event in events:
        cat = event['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n📊 Events by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"   • {cat}: {count}")
    
    print("\n🎭 Sample events:")
    for i, event in enumerate(events[:5], 1):
        date_info = f" - {event['date']}" if event['date'] else ""
        print(f"  {i}. {event['name']} ({event['category']}){date_info}")
    
    return events

def test_events_search():
    """Test searching the events database"""
    print("\n" + "="*70)
    print("🧪 Testing Events Search")
    print("="*70)
    
    search_engine = SemanticSearchEngine()
    search_engine.load_collection("events", "./data/events_index.bin")
    
    test_queries = [
        "theatre performances this week",
        "music concerts and jazz",
        "workshops and cultural activities",
        "Istanbul film festival",
        "contemporary art exhibitions"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = search_engine.search(query, top_k=3, collection="events")
        for i, r in enumerate(results, 1):
            date_info = f" - {r.get('date', '')}" if r.get('date') else ""
            print(f"  {i}. {r['name']} ({r['category']}){date_info} (score: {r['similarity_score']:.3f})")

def main():
    print("\n" + "="*70)
    print("🚀 Integrating IKSV Events Database into ML System")
    print("="*70)
    print("\nThis will:")
    print("  ✅ Export 38+ IKSV events from database")
    print("  ✅ Include both major festivals and current events")
    print("  ✅ Create semantic search index for event queries")
    print("  ✅ Enable accurate event information responses")
    print("\n" + "="*70)
    
    # Export events
    events = export_iksv_events_to_semantic_index()
    
    # Test search
    test_events_search()
    
    print("\n" + "="*70)
    print("✅ IKSV EVENTS DATABASE INTEGRATION COMPLETE!")
    print("="*70)
    print("\nWhat changed:")
    print("  🎭 Events index now has 38+ IKSV cultural events")
    print("  🎭 Includes theatre, music, workshops, festivals")
    print("  🎭 Real event dates and venues")
    print("  🎭 Both major annual festivals and current events")
    print("\nNext steps:")
    print("  1. Restart ML service to load new index")
    print("  2. Test with event queries")
    print("  3. Verify: Accurate IKSV event information in responses")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
