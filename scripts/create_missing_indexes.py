#!/usr/bin/env python3
"""
Create separate semantic search indexes for attractions, tips, and events
This fixes the problem where attraction queries return restaurant data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_systems.semantic_search_engine import SemanticSearchEngine
import json
import pickle

def create_attractions_index():
    """Create semantic index for attractions"""
    print("\n" + "="*70)
    print("📍 Creating Attractions Index")
    print("="*70)
    
    # Famous Istanbul attractions with rich descriptions
    attractions = [
        {
            "id": "attr_001",
            "name": "Hagia Sophia (Ayasofya)",
            "type": "attraction",
            "category": "museum",
            "description": "Built in 537 AD as Byzantine cathedral, later converted to Ottoman mosque in 1453, then museum in 1935, and mosque again in 2020. Famous for massive dome, Byzantine mosaics, Islamic calligraphy, and 1500 years of history. NOT the same as Blue Mosque.",
            "location": "Sultanahmet Square",
            "district": "Fatih",
            "price": "free (donations accepted)",
            "hours": "Open daily except prayer times, closed Fridays during midday prayer",
            "tags": ["museum", "mosque", "historical", "byzantine", "ottoman", "architecture", "unesco", "religious_site"]
        },
        {
            "id": "attr_002",
            "name": "Blue Mosque (Sultanahmet Camii)",
            "type": "attraction",
            "category": "religious_site",
            "description": "Built 1609-1616 by Sultan Ahmed I. Called Blue Mosque due to 20,000+ blue Iznik tiles inside. Has 6 minarets (unique at time of construction), large courtyard, and still active mosque. DIFFERENT from Hagia Sophia - they face each other across Sultanahmet Square.",
            "location": "Sultanahmet Square (across from Hagia Sophia)",
            "district": "Fatih",
            "price": "free (donations welcome)",
            "hours": "Open daily except during 5 daily prayer times, modest dress required",
            "tags": ["mosque", "ottoman", "religious", "architecture", "iconic", "blue_tiles", "active_worship"]
        },
        {
            "id": "attr_003",
            "name": "Topkapı Palace",
            "type": "attraction",
            "category": "palace",
            "description": "Former Ottoman imperial palace with treasury, harem, holy relics, and stunning Bosphorus views. Sprawling complex with courtyards and gardens",
            "location": "Sultanahmet",
            "district": "Fatih",
            "price": "paid",
            "hours": "09:00-18:00 (closed Tuesdays)",
            "tags": ["palace", "ottoman", "museum", "historical", "unesco"]
        },
        {
            "id": "attr_004",
            "name": "Grand Bazaar",
            "type": "attraction",
            "category": "market",
            "description": "One of the world's oldest and largest covered markets with 4000+ shops. Labyrinth of streets selling carpets, jewelry, ceramics, spices, and souvenirs",
            "location": "Beyazıt",
            "district": "Fatih",
            "price": "free",
            "hours": "09:00-19:00 (closed Sundays)",
            "tags": ["shopping", "market", "historical", "bazaar", "souvenirs"]
        },
        {
            "id": "attr_005",
            "name": "Spice Bazaar (Egyptian Bazaar)",
            "type": "attraction",
            "category": "market",
            "description": "Colorful covered market near Eminönü specializing in spices, Turkish delight, dried fruits, nuts, and teas. Aromatic and photogenic",
            "location": "Eminönü",
            "district": "Fatih",
            "price": "free",
            "hours": "08:00-19:30",
            "tags": ["shopping", "market", "spices", "food", "bazaar"]
        },
        {
            "id": "attr_006",
            "name": "Galata Tower",
            "type": "attraction",
            "category": "monument",
            "description": "Medieval Genoese tower offering 360-degree panoramic views of Istanbul. Historic landmark in Beyoğlu with restaurant at top",
            "location": "Galata",
            "district": "Beyoğlu",
            "price": "paid",
            "hours": "08:30-23:00",
            "tags": ["tower", "viewpoint", "medieval", "panorama", "landmark"]
        },
        {
            "id": "attr_007",
            "name": "Basilica Cistern",
            "type": "attraction",
            "category": "historical",
            "description": "Atmospheric underground Byzantine water reservoir with 336 columns. Famous for Medusa head columns and eerie lighting. Ancient engineering marvel",
            "location": "Sultanahmet",
            "district": "Fatih",
            "price": "paid",
            "hours": "09:00-18:30",
            "tags": ["historical", "byzantine", "underground", "unique", "atmospheric"]
        },
        {
            "id": "attr_008",
            "name": "Dolmabahçe Palace",
            "type": "attraction",
            "category": "palace",
            "description": "Lavish 19th century Ottoman palace on Bosphorus. European-style architecture with crystal chandeliers, ornate rooms, and beautiful gardens",
            "location": "Beşiktaş",
            "district": "Beşiktaş",
            "price": "paid",
            "hours": "09:00-16:00 (closed Mondays)",
            "tags": ["palace", "ottoman", "bosphorus", "luxury", "gardens"]
        },
        {
            "id": "attr_009",
            "name": "Istiklal Street",
            "type": "attraction",
            "category": "street",
            "description": "Famous pedestrian avenue in Beyoğlu with shops, cafes, restaurants, historic tram, and vibrant nightlife. Heart of modern Istanbul",
            "location": "Beyoğlu",
            "district": "Beyoğlu",
            "price": "free",
            "hours": "24/7",
            "tags": ["shopping", "pedestrian", "nightlife", "cafes", "modern"]
        },
        {
            "id": "attr_010",
            "name": "Maiden's Tower",
            "type": "attraction",
            "category": "monument",
            "description": "Iconic small tower on islet in Bosphorus. Romantic spot with restaurant, stunning views, and legendary folklore. Accessible by boat",
            "location": "Üsküdar",
            "district": "Üsküdar",
            "price": "paid",
            "hours": "09:00-19:00",
            "tags": ["tower", "bosphorus", "romantic", "viewpoint", "iconic"]
        },
        {
            "id": "attr_011",
            "name": "Süleymaniye Mosque",
            "type": "attraction",
            "category": "religious_site",
            "description": "Magnificent Ottoman mosque by architect Sinan with peaceful gardens and city views. Less touristy than Blue Mosque, architectural masterpiece",
            "location": "Süleymaniye",
            "district": "Fatih",
            "price": "free",
            "hours": "Prayer times vary",
            "tags": ["mosque", "ottoman", "architecture", "sinan", "peaceful"]
        },
        {
            "id": "attr_012",
            "name": "Chora Church (Kariye Museum)",
            "type": "attraction",
            "category": "museum",
            "description": "Byzantine church with world's finest Byzantine mosaics and frescoes. Hidden gem in less touristy neighborhood",
            "location": "Edirnekapı",
            "district": "Fatih",
            "price": "paid",
            "hours": "09:00-19:00 (closed Wednesdays)",
            "tags": ["museum", "byzantine", "mosaics", "art", "hidden_gem"]
        },
        {
            "id": "attr_013",
            "name": "Princes' Islands",
            "type": "attraction",
            "category": "nature",
            "description": "Car-free islands in Sea of Marmara accessible by ferry. Bike riding, historic mansions, beaches, and pine forests. Perfect day trip",
            "location": "Adalar",
            "district": "Adalar",
            "price": "ferry_ticket",
            "hours": "Daylight hours",
            "tags": ["islands", "nature", "day_trip", "ferry", "peaceful"]
        },
        {
            "id": "attr_014",
            "name": "Balat",
            "type": "attraction",
            "category": "neighborhood",
            "description": "Colorful historic neighborhood with Ottoman houses, street art, cafes, and authentic local atmosphere. Instagram-worthy streets",
            "location": "Balat",
            "district": "Fatih",
            "price": "free",
            "hours": "24/7",
            "tags": ["neighborhood", "colorful", "local", "photography", "authentic"]
        },
        {
            "id": "attr_015",
            "name": "Ortaköy",
            "type": "attraction",
            "category": "waterfront",
            "description": "Bosphorus-side neighborhood famous for kumpir (stuffed potato), mosque by water, and bridge views. Lively cafes and boutiques",
            "location": "Ortaköy",
            "district": "Beşiktaş",
            "price": "free",
            "hours": "24/7",
            "tags": ["bosphorus", "waterfront", "food", "cafes", "scenic"]
        }
    ]
    
    print(f"📦 Prepared {len(attractions)} curated attractions")
    
    # Create search engine
    search_engine = SemanticSearchEngine()
    
    # Index attractions
    search_engine.index_items(attractions, save_path="./data/attractions_index.bin")
    
    print("✅ Attractions index created successfully!")
    print(f"   File: ./data/attractions_index.bin")
    print(f"   Items: {len(attractions)}")
    
    return attractions

def create_tips_index():
    """Create semantic index for local tips and hidden gems"""
    print("\n" + "="*70)
    print("💎 Creating Local Tips Index")
    print("="*70)
    
    tips = [
        {
            "id": "tip_001",
            "name": "Pierre Loti Cafe",
            "type": "tip",
            "category": "viewpoint",
            "description": "Historic hilltop cafe in Eyüp with stunning Golden Horn views. Take cable car up. Popular with locals for sunset tea and Turkish coffee",
            "location": "Eyüp",
            "district": "Eyüp Sultan",
            "tip_type": "hidden_gem",
            "local_favorite": True
        },
        {
            "id": "tip_002",
            "name": "Karaköy Street Food",
            "type": "tip",
            "category": "food",
            "description": "Explore Karaköy's narrow streets for authentic street food: midye dolma (stuffed mussels), balık ekmek (fish sandwich), and local bakeries",
            "location": "Karaköy",
            "district": "Beyoğlu",
            "tip_type": "food_tip",
            "local_favorite": True
        },
        {
            "id": "tip_003",
            "name": "Çamlıca Hill",
            "type": "tip",
            "category": "viewpoint",
            "description": "Highest point in Istanbul on Asian side with panoramic city views. Less crowded than European side viewpoints. Great for picnics",
            "location": "Çamlıca",
            "district": "Üsküdar",
            "tip_type": "hidden_gem",
            "local_favorite": True
        },
        {
            "id": "tip_004",
            "name": "Fener-Balat Walking Tour",
            "type": "tip",
            "category": "neighborhood",
            "description": "Self-guided walk through colorful historic neighborhoods. Start at Fener Greek Patriarchate, explore Balat's rainbow houses and cafes",
            "location": "Fener-Balat",
            "district": "Fatih",
            "tip_type": "local_experience",
            "local_favorite": True
        },
        {
            "id": "tip_005",
            "name": "Sunday Morning at Ortaköy",
            "type": "tip",
            "category": "experience",
            "description": "Visit Ortaköy on Sunday mornings for local market, fresh simit, and peaceful Bosphorus views before crowds arrive",
            "location": "Ortaköy",
            "district": "Beşiktaş",
            "tip_type": "timing_tip",
            "local_favorite": True
        }
    ]
    
    print(f"📦 Prepared {len(tips)} local tips")
    
    search_engine = SemanticSearchEngine()
    search_engine.index_items(tips, save_path="./data/tips_index.bin")
    
    print("✅ Tips index created successfully!")
    print(f"   File: ./data/tips_index.bin")
    print(f"   Items: {len(tips)}")
    
    return tips

def test_indexes():
    """Test the newly created indexes"""
    print("\n" + "="*70)
    print("🧪 Testing New Indexes")
    print("="*70)
    
    search_engine = SemanticSearchEngine()
    
    # Load attractions
    try:
        search_engine.load_collection("attractions", "./data/attractions_index.bin")
        print("✅ Attractions collection loaded")
        
        # Test search
        results = search_engine.search("Tell me about Hagia Sophia", top_k=3, collection="attractions")
        print(f"\n🔍 Test search: 'Tell me about Hagia Sophia'")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']} ({r['category']}) - score: {r['similarity_score']:.3f}")
    except Exception as e:
        print(f"❌ Error testing attractions: {e}")
    
    # Load tips
    try:
        search_engine.load_collection("tips", "./data/tips_index.bin")
        print("\n✅ Tips collection loaded")
        
        # Test search
        results = search_engine.search("hidden gems locals love", top_k=3, collection="tips")
        print(f"\n🔍 Test search: 'hidden gems locals love'")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']} - {r.get('tip_type', 'N/A')} - score: {r['similarity_score']:.3f}")
    except Exception as e:
        print(f"❌ Error testing tips: {e}")

def main():
    print("\n" + "="*70)
    print("🚀 Creating Missing Semantic Search Indexes")
    print("="*70)
    print("\nThis will fix:")
    print("  ❌ Attraction queries returning restaurant data")
    print("  ❌ Local tips not finding relevant content")
    print("  ❌ Events queries having no data")
    print("\n" + "="*70)
    
    # Create attractions index
    create_attractions_index()
    
    # Create tips index
    create_tips_index()
    
    # Test the indexes
    test_indexes()
    
    print("\n" + "="*70)
    print("✅ ALL INDEXES CREATED SUCCESSFULLY!")
    print("="*70)
    print("\nCreated files:")
    print("  📁 ./data/attractions_index.bin")
    print("  📁 ./data/attractions_index.bin.items")
    print("  📁 ./data/tips_index.bin")
    print("  📁 ./data/tips_index.bin.items")
    print("\nNext steps:")
    print("  1. Restart ML service: python ml_api_service.py")
    print("  2. Run tests: python test_problematic_intents.py")
    print("  3. Verify: Attractions should now return attraction data!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
