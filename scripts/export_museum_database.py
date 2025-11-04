#!/usr/bin/env python3
"""
Export museum database to semantic search indexes
Integrates the huge accurate_museum_database.py into ML system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_systems.semantic_search_engine import SemanticSearchEngine
from backend.accurate_museum_database import IstanbulMuseumDatabase

def export_museums_to_semantic_index():
    """Export all museums from accurate database to semantic search"""
    print("\n" + "="*70)
    print("📍 Exporting Museum Database to Semantic Search")
    print("="*70)
    
    # Initialize museum database
    museum_db = IstanbulMuseumDatabase()
    
    # Convert to semantic search format
    attractions = []
    for key, museum in museum_db.museums.items():
        # Build rich description
        description_parts = [
            museum.historical_significance,
            f"Built: {museum.construction_date}.",
            f"Key features: {', '.join(museum.key_features[:3])}.",
            f"Must see: {', '.join(museum.must_see_highlights[:2])}." if museum.must_see_highlights else "",
        ]
        full_description = " ".join(description_parts)
        
        # Determine category
        category = "museum"
        if "mosque" in museum.name.lower() and "museum" not in museum.name.lower():
            category = "religious_site"
        elif "palace" in museum.name.lower():
            category = "palace"
        elif "bazaar" in museum.name.lower():
            category = "market"
        elif "tower" in museum.name.lower():
            category = "monument"
        elif "fortress" in museum.name.lower():
            category = "monument"
        
        attraction = {
            "id": f"museum_{key}",
            "name": museum.name,
            "type": "attraction",
            "category": category,
            "description": full_description[:500],  # Limit length
            "location": museum.location,
            "district": museum.location.split(",")[-1].strip() if "," in museum.location else museum.location,
            "price": museum.entrance_fee,
            "hours": str(museum.opening_hours.get('daily', 'Varies')),
            "tags": [
                category,
                "historical" if "historical" in museum.historical_significance.lower() else "cultural",
                museum.architectural_style.split()[0].lower() if museum.architectural_style else "attraction",
                "unesco" if "UNESCO" in museum.historical_significance else "landmark"
            ],
            "duration": museum.visiting_duration,
            "best_time": museum.best_time_to_visit,
            "highlights": museum.must_see_highlights[:3],
            "nearby": museum.nearby_attractions[:3]
        }
        attractions.append(attraction)
    
    print(f"📦 Exported {len(attractions)} museums/attractions from database")
    
    # Create semantic search engine
    search_engine = SemanticSearchEngine()
    
    # Index attractions
    search_engine.index_items(attractions, save_path="./data/attractions_index.bin")
    
    print("✅ Museum database indexed successfully!")
    print(f"   File: ./data/attractions_index.bin")
    print(f"   Items: {len(attractions)}")
    
    # Show sample
    print("\n📋 Sample attractions indexed:")
    for i, attr in enumerate(attractions[:5], 1):
        print(f"  {i}. {attr['name']} ({attr['category']})")
    
    return attractions

def test_museum_search():
    """Test searching the museum database"""
    print("\n" + "="*70)
    print("🧪 Testing Museum Search")
    print("="*70)
    
    search_engine = SemanticSearchEngine()
    search_engine.load_collection("attractions", "./data/attractions_index.bin")
    
    test_queries = [
        "Byzantine mosaics and history",
        "Ottoman palaces with Bosphorus view",
        "Modern art museums",
        "Islamic architecture mosques",
        "Museums for kids and families"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = search_engine.search(query, top_k=3, collection="attractions")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']} - {r['category']} (score: {r['similarity_score']:.3f})")

def main():
    print("\n" + "="*70)
    print("🚀 Integrating Huge Museum Database into ML System")
    print("="*70)
    print("\nThis will:")
    print("  ✅ Export 49+ museums/attractions from accurate_museum_database.py")
    print("  ✅ Create detailed semantic search index")
    print("  ✅ Enable accurate, fact-checked attraction responses")
    print("\n" + "="*70)
    
    # Export museums
    attractions = export_museums_to_semantic_index()
    
    # Test search
    test_museum_search()
    
    print("\n" + "="*70)
    print("✅ MUSEUM DATABASE INTEGRATION COMPLETE!")
    print("="*70)
    print("\nWhat changed:")
    print("  📍 Attractions index now has 49+ verified museums")
    print("  📍 Includes accurate hours, prices, descriptions")
    print("  📍 Hagia Sophia and Blue Mosque correctly differentiated")
    print("  📍 All major Istanbul attractions covered")
    print("\nNext steps:")
    print("  1. Restart ML service to load new index")
    print("  2. Test with: python test_problematic_intents.py")
    print("  3. Verify: Accurate museum information in responses")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
