#!/usr/bin/env python3
"""
Index all database items for semantic search
Run this once locally, then upload to T4
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_systems.semantic_search_engine import SemanticSearchEngine
import json

def load_from_json():
    """Load all items from JSON files"""
    all_items = []
    
    # Load restaurants from JSON
    try:
        print("📥 Loading restaurants from JSON...")
        restaurants_path = 'backend/data/restaurants_database.json'
        if os.path.exists(restaurants_path):
            with open(restaurants_path, 'r', encoding='utf-8') as f:
                restaurants_data = json.load(f)
            
            # Handle both dict with 'restaurants' key or direct list
            if isinstance(restaurants_data, dict) and 'restaurants' in restaurants_data:
                restaurants = restaurants_data['restaurants']
            elif isinstance(restaurants_data, list):
                restaurants = restaurants_data
            else:
                restaurants = []
            
            for i, restaurant in enumerate(restaurants):
                item = {
                    'id': restaurant.get('id', f'restaurant_{i}'),
                    'type': 'restaurant',
                    'category': 'restaurant',
                    'name': restaurant.get('name', 'Unknown'),
                    'description': restaurant.get('description', ''),
                    'location': restaurant.get('location', restaurant.get('neighborhood', '')),
                    'cuisine': restaurant.get('cuisine', ''),
                    'price_level': restaurant.get('price_level', '')
                }
                all_items.append(item)
            
            print(f"✅ Loaded {len(restaurants)} restaurants")
        else:
            print(f"⚠️  Restaurants file not found: {restaurants_path}")
    except Exception as e:
        print(f"⚠️  Could not load restaurants: {e}")
    
    # Load attractions from JSON
    try:
        print("📥 Loading attractions from JSON...")
        attractions_path = 'backend/data/attractions_database.json'
        if os.path.exists(attractions_path):
            with open(attractions_path, 'r', encoding='utf-8') as f:
                attractions_data = json.load(f)
            
            # Handle both dict with 'attractions' key or direct list
            if isinstance(attractions_data, dict) and 'attractions' in attractions_data:
                attractions = attractions_data['attractions']
            elif isinstance(attractions_data, list):
                attractions = attractions_data
            else:
                attractions = []
            
            for i, attraction in enumerate(attractions):
                item = {
                    'id': attraction.get('id', f'attraction_{i}'),
                    'type': 'attraction',
                    'category': attraction.get('category', 'attraction'),
                    'name': attraction.get('name', 'Unknown'),
                    'description': attraction.get('description', ''),
                    'location': attraction.get('location', attraction.get('neighborhood', ''))
                }
                all_items.append(item)
            
            print(f"✅ Loaded {len(attractions)} attractions")
        else:
            print(f"⚠️  Attractions file not found: {attractions_path}")
    except Exception as e:
        print(f"⚠️  Could not load attractions: {e}")
    
    print(f"\n✅ Total items to index: {len(all_items)}")
    return all_items
    
    print(f"\n✅ Total items to index: {len(all_items)}")
    return all_items

def main():
    print("🚀 Starting indexing process...")
    print("=" * 60)
    
    # Load data from JSON files
    items = load_from_json()
    
    if not items:
        print("\n❌ No items found to index!")
        print("   Make sure your database has data in restaurants and/or attractions tables")
        return
    
    # Create search engine
    search_engine = SemanticSearchEngine()
    
    # Index items
    search_engine.index_items(items, save_path="./data/semantic_index.bin")
    
    # Test search
    print("\n🧪 Testing search...")
    print("-" * 60)
    
    test_queries = [
        "authentic Turkish restaurant",
        "museums in Istanbul",
        "romantic place with view"
    ]
    
    for query in test_queries:
        results = search_engine.search(query, top_k=3)
        print(f"\nQuery: '{query}'")
        if results:
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r.get('name', 'Unknown')} - {r.get('location', 'N/A')} (score: {r.get('similarity_score', 0):.3f})")
        else:
            print("  No results found")
    
    print("\n" + "=" * 60)
    print("✅ Indexing complete!")
    print("\n📦 Files created:")
    print("  - ./data/semantic_index.bin")
    print("  - ./data/semantic_index.bin.items")
    print("\n📌 Next step: Run 'python scripts/test_local_system.py'")

if __name__ == "__main__":
    main()
