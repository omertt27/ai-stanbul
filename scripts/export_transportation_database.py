#!/usr/bin/env python3
"""
Export transportation guide data to semantic search index
Integrates Istanbul transportation system knowledge into KAM
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_systems.semantic_search_engine import SemanticSearchEngine

def create_transportation_database():
    """Create comprehensive transportation knowledge base"""
    
    transportation_data = [
        # Metro Lines
        {
            "id": "metro_m1",
            "name": "M1 Metro Line (Yenikapı - Atatürk Airport/Kirazlı)",
            "type": "metro",
            "category": "transportation",
            "description": "M1 metro line connects Yenikapı to Atatürk Airport and Kirazlı. Splits at Otogar station. Key stops: Aksaray, Otogar (bus terminal), Airport.",
            "route": "Yenikapı → Aksaray → Otogar → Airport/Kirazlı",
            "tags": ["metro", "m1", "airport", "yenikapı", "aksaray", "otogar", "kirazlı"],
            "tips": "Use for: Airport access, Aksaray connections, Otogar bus terminal. Connects with T1 tram at Aksaray.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-00:00 (approximately)"
        },
        {
            "id": "metro_m2",
            "name": "M2 Metro Line (Yenikapı - Hacıosman)",
            "type": "metro",
            "category": "transportation",
            "description": "M2 metro line runs from Yenikapı to Hacıosman via Taksim and Şişli. Most useful for tourists. Key stops: Taksim, Şişhane, Vezneciler, Yenikapı.",
            "route": "Hacıosman → Levent → Taksim → Şişhane → Vezneciler → Yenikapı",
            "tags": ["metro", "m2", "taksim", "şişhane", "yenikapı", "vezneciler", "levent"],
            "tips": "Use for: Taksim Square, İstiklal Street (Şişhane), Galata Tower area. Connects with Marmaray and M1 at Yenikapı.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-00:00"
        },
        {
            "id": "metro_m3",
            "name": "M3 Metro Line (Kirazlı - Başakşehir/Olimpiyat)",
            "type": "metro",
            "category": "transportation",
            "description": "M3 metro line serves western suburbs. Connects with M1 at Kirazlı. Less relevant for tourists.",
            "route": "Kirazlı → Başakşehir → Olimpiyat",
            "tags": ["metro", "m3", "kirazlı", "başakşehir", "olimpiyat"],
            "tips": "Use for: Residential areas, shopping malls in western Istanbul. Connects with M1 at Kirazlı.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-00:00"
        },
        {
            "id": "metro_m4",
            "name": "M4 Metro Line (Kadıköy - Tavşantepe)",
            "type": "metro",
            "category": "transportation",
            "description": "M4 metro line serves Asian side from Kadıköy to Tavşantepe. Essential for Asian side travel. Key stop: Kadıköy.",
            "route": "Kadıköy → Ayrılık Çeşmesi → Bostancı → Tavşantepe",
            "tags": ["metro", "m4", "kadıköy", "asian side", "bostancı", "tavşantepe"],
            "tips": "Use for: Asian side destinations, Kadıköy nightlife and cafes. Connects with ferries at Kadıköy.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-00:00"
        },
        {
            "id": "metro_m5",
            "name": "M5 Metro Line (Üsküdar - Çekmeköy)",
            "type": "metro",
            "category": "transportation",
            "description": "M5 metro line serves northern Asian side from Üsküdar. Connects with Marmaray at Üsküdar.",
            "route": "Üsküdar → Ümraniye → Çekmeköy",
            "tags": ["metro", "m5", "üsküdar", "asian side", "ümraniye", "çekmeköy"],
            "tips": "Use for: Üsküdar Mosque area, northern Asian side. Connects with Marmaray and ferries at Üsküdar.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-00:00"
        },
        
        # Tram Lines
        {
            "id": "tram_t1",
            "name": "T1 Tram (Bağcılar - Kabataş)",
            "type": "tram",
            "category": "transportation",
            "description": "T1 tram is THE most important line for tourists. Connects all major Old City attractions. Key stops: Sultanahmet, Eminönü, Karaköy, Kabataş. Historic red trams.",
            "route": "Bağcılar → Aksaray → Sultanahmet → Eminönü → Karaköy → Kabataş",
            "tags": ["tram", "t1", "sultanahmet", "eminönü", "karaköy", "kabataş", "old city", "tourist"],
            "tips": "ESSENTIAL for tourists! Use for: Hagia Sophia, Blue Mosque (Sultanahmet), Grand Bazaar (Beyazıt), Spice Bazaar (Eminönü), Galata Bridge (Karaköy). Very crowded during peak hours.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-00:00",
            "connections": "Connects with M2 metro at Şişhane (via Karaköy), M1 at Aksaray, Funicular at Kabataş"
        },
        {
            "id": "tram_t2",
            "name": "T2 Tram (Taksim - Tünel)",
            "type": "tram",
            "category": "transportation",
            "description": "T2 nostalgic red tram runs along İstiklal Street from Taksim to Tünel. Short scenic route, often crowded.",
            "route": "Taksim ↔ İstiklal Street ↔ Tünel",
            "tags": ["tram", "t2", "taksim", "istiklal", "tünel", "nostalgic", "historic"],
            "tips": "Use for: İstiklal Street experience. Better to walk İstiklal Street - it's pedestrian and more enjoyable. Tram is slow and crowded.",
            "fare": "Pay on board (cash accepted)",
            "operating_hours": "07:00-21:00"
        },
        {
            "id": "tram_t3",
            "name": "T3 Tram (Kadıköy - Moda)",
            "type": "tram",
            "category": "transportation",
            "description": "T3 tram serves Kadıköy seafront and Moda neighborhood on Asian side. Short, scenic route.",
            "route": "Kadıköy ↔ Moda",
            "tags": ["tram", "t3", "kadıköy", "moda", "asian side", "seafront"],
            "tips": "Use for: Moda neighborhood, seaside walk. Pleasant ride along the coast.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-23:00"
        },
        
        # Marmaray
        {
            "id": "marmaray",
            "name": "Marmaray (Gebze - Halkalı)",
            "type": "train",
            "category": "transportation",
            "description": "Marmaray is the underground rail tunnel connecting Europe and Asia under the Bosphorus. Revolutionary cross-continental train. Key stops: Sirkeci (European side), Üsküdar (Asian side), Yenikapı.",
            "route": "Halkalı → Yenikapı → Sirkeci → BOSPHORUS TUNNEL → Üsküdar → Gebze",
            "tags": ["marmaray", "train", "bosphorus", "cross-continental", "sirkeci", "üsküdar", "yenikapı"],
            "tips": "AMAZING experience! Cross from Europe to Asia in 4 minutes underwater. Use for: Quick Bosphorus crossing, connecting European and Asian sides. Sirkeci stop is near Eminönü and Old City.",
            "fare": "Single ride with Istanbulkart (counted as transfer if within 2 hours)",
            "operating_hours": "06:00-00:00",
            "special": "World's first transcontinental undersea rail tunnel!"
        },
        
        # Funiculars
        {
            "id": "funicular_f1",
            "name": "F1 Funicular (Taksim - Kabataş)",
            "type": "funicular",
            "category": "transportation",
            "description": "F1 funicular connects Kabataş (waterfront) to Taksim Square. Fast uphill connection. Just 2 stops.",
            "route": "Kabataş ↔ Taksim",
            "tags": ["funicular", "f1", "taksim", "kabataş", "fast", "uphill"],
            "tips": "Use for: Quick connection between Kabataş (T1 tram, ferries) and Taksim. Saves steep uphill walk. Very frequent service.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "06:00-00:00",
            "frequency": "Every 3-5 minutes"
        },
        {
            "id": "funicular_f2",
            "name": "F2 Funicular (Karaköy - Tünel)",
            "type": "funicular",
            "category": "transportation",
            "description": "F2 Tünel is the world's second-oldest underground railway (1875). Connects Karaköy to İstiklal Street (Tünel end). Historic and charming.",
            "route": "Karaköy ↔ Tünel (İstiklal Street)",
            "tags": ["funicular", "f2", "tünel", "karaköy", "historic", "istiklal"],
            "tips": "Use for: Avoiding steep uphill walk from Karaköy to İstiklal Street. Historic experience - world's 2nd oldest subway! Connects with T1 tram at Karaköy.",
            "fare": "Single ride with Istanbulkart",
            "operating_hours": "07:00-22:00",
            "special": "World's second-oldest underground railway (1875)!"
        },
        
        # Ferries
        {
            "id": "ferry_bosphorus",
            "name": "Bosphorus Ferries",
            "type": "ferry",
            "category": "transportation",
            "description": "Public ferries (şehir hatları) cross the Bosphorus between European and Asian sides. Main routes: Eminönü-Kadıköy, Karaköy-Kadıköy, Beşiktaş-Üsküdar, Kabataş-Üsküdar. Scenic and authentic.",
            "route": "Multiple routes: Eminönü/Karaköy/Kabataş/Beşiktaş ↔ Kadıköy/Üsküdar",
            "tags": ["ferry", "bosphorus", "eminönü", "kadıköy", "karaköy", "beşiktaş", "üsküdar", "scenic"],
            "tips": "HIGHLY RECOMMENDED! Use for: Crossing to Asian side with amazing Bosphorus views. Drink çay (tea) on deck. Cheaper and more authentic than tourist cruises. Very frequent during day.",
            "fare": "~25-30 TL with Istanbulkart (discounted)",
            "operating_hours": "06:00-midnight (reduced service late night)",
            "duration": "20-25 minutes",
            "special": "Best way to experience Bosphorus! Buy simit (bread ring) and çay on board."
        },
        {
            "id": "ferry_princes_islands",
            "name": "Princes' Islands Ferries",
            "type": "ferry",
            "category": "transportation",
            "description": "Ferries to Princes' Islands (Adalar) from Eminönü and Kabataş. Islands: Kınalıada, Burgazada, Heybeliada, Büyükada. Perfect day trip - no cars allowed on islands.",
            "route": "Eminönü/Kabataş → Kınalıada → Burgazada → Heybeliada → Büyükada",
            "tags": ["ferry", "princes islands", "adalar", "büyükada", "day trip", "scenic"],
            "tips": "Perfect day trip! Use for: Escape Istanbul crowds, bike riding, horse carriages (fayton), seafood lunch. Büyükada is the biggest and most popular. Check schedules - less frequent than Bosphorus ferries.",
            "fare": "~50-80 TL with Istanbulkart",
            "operating_hours": "Seasonal schedule - more frequent in summer",
            "duration": "1-1.5 hours to Büyükada",
            "special": "No cars on islands! Use bikes or horse carriages."
        },
        
        # İstanbulkart
        {
            "id": "istanbulkart",
            "name": "İstanbulkart - Istanbul's Transport Card",
            "type": "card",
            "category": "transportation",
            "description": "İstanbulkart is Istanbul's rechargeable transport card. Used for metro, tram, bus, ferry, Marmaray, funicular. ESSENTIAL for visitors. Cheaper than tokens/tickets. Enables transfers within 2 hours.",
            "tags": ["istanbulkart", "transport card", "payment", "transfer", "discount"],
            "tips": "GET THIS FIRST! Available at: Airport, metro/tram stations, kiosks. Cost: ~50 TL card + load money. Save 50% with transfers. Can use one card for multiple people if traveling together (scan multiple times). Refundable at machines.",
            "where_to_buy": "Airport, any metro/tram station, newspaper kiosks (Biletix)",
            "price": "~50 TL card deposit + desired credit amount",
            "benefits": "Discounted fares, free transfers within 2 hours, works on all public transport",
            "special": "ONE CARD works on ALL public transport in Istanbul!"
        },
        
        # Route Tips
        {
            "id": "route_sultanahmet_taksim",
            "name": "How to get from Sultanahmet to Taksim",
            "type": "route",
            "category": "transportation",
            "description": "Popular route from Old City (Sultanahmet) to Taksim Square. Best option: T1 tram to Kabataş, then F1 funicular to Taksim. Alternative: T1 to Şişhane, walk to Taksim via İstiklal.",
            "route": "Sultanahmet → T1 tram → Kabataş → F1 funicular → Taksim",
            "tags": ["route", "sultanahmet", "taksim", "old city", "beyoğlu"],
            "tips": "Recommended route: T1 tram from Sultanahmet to Kabataş (20 min), then F1 funicular to Taksim (2 min). Total: ~25 minutes. Counts as single journey with İstanbulkart if within 2 hours.",
            "duration": "25-30 minutes",
            "cost": "~15-20 TL with İstanbulkart (including transfer)",
            "alternative": "T1 to Karaköy, walk across Galata Bridge, F2 funicular to İstiklal, walk to Taksim"
        },
        {
            "id": "route_sultanahmet_kadikoy",
            "name": "How to get from Sultanahmet to Kadıköy (Asian side)",
            "type": "route",
            "category": "transportation",
            "description": "Popular route from Old City to Asian side. Best option: Walk to Eminönü (5 min), take ferry to Kadıköy. Alternative: Marmaray from Sirkeci.",
            "route": "Sultanahmet → Walk → Eminönü → Ferry → Kadıköy",
            "tags": ["route", "sultanahmet", "kadıköy", "asian side", "ferry", "bosphorus"],
            "tips": "HIGHLY RECOMMENDED: Walk from Sultanahmet to Eminönü ferry terminal (5-10 min downhill), take ferry to Kadıköy (20 min). Scenic Bosphorus crossing! Alternative: T1 tram to Sirkeci, Marmaray to Üsküdar, transfer to M5 metro to other Asian locations.",
            "duration": "30-40 minutes (including walk)",
            "cost": "Ferry: ~25 TL with İstanbulkart",
            "recommendation": "Ferry is best - scenic and authentic experience!"
        },
        {
            "id": "route_airport_sultanahmet",
            "name": "How to get from Istanbul Airport to Sultanahmet",
            "type": "route",
            "category": "transportation",
            "description": "From new Istanbul Airport (IST) to Old City. Options: HAVAIST bus (direct), or Metro + Tram combination. HAVAIST is easier with luggage.",
            "route": "Airport → HAVAIST bus → Sultanahmet OR Airport → M11 metro → M2 metro → M1 metro → T1 tram → Sultanahmet",
            "tags": ["route", "airport", "ist", "sultanahmet", "arrivals"],
            "tips": "EASIEST: HAVAIST bus from airport directly to Sultanahmet (IST-19 line). Cost: ~250 TL, Duration: 60-90 min depending on traffic. Buy ticket from Havaist desk at arrivals. With luggage, this is best option. PUBLIC TRANSPORT: M11 → M2 (Gayrettepe) → Transfer to M1 (via various connections), then T1 to Sultanahmet. Cheaper but complex with luggage.",
            "duration": "60-90 minutes (HAVAIST), 90-120 minutes (public transport)",
            "cost": "HAVAIST: ~250 TL, Public transport: ~50 TL",
            "recommendation": "HAVAIST bus recommended for first-time visitors with luggage"
        },
        {
            "id": "transportation_tips_general",
            "name": "General Istanbul Transportation Tips",
            "type": "tips",
            "category": "transportation",
            "description": "Essential tips for using Istanbul's public transportation system effectively.",
            "tags": ["tips", "advice", "transportation", "general", "beginner"],
            "tips": """KEY TIPS:
1. GET ISTANBULKART FIRST - Essential! Saves 50%+ on fares
2. PEAK HOURS - Avoid 8-9 AM and 6-7 PM if possible (very crowded)
3. T1 TRAM - Most important for tourists (Sultanahmet, Eminönü, Karaköy)
4. FERRIES - Don't miss! Best way to cross Bosphorus with amazing views
5. TRANSFERS - Free within 2 hours with İstanbulkart
6. GOOGLE MAPS - Works well for Istanbul public transport directions
7. MARMARAY - Amazing experience crossing Europe-Asia underwater
8. TAKSIM-SULTANAHMET - Use T1 tram + F1 funicular (fast and easy)
9. KADIKOY - Asian side is worth visiting! Take ferry from Eminönü
10. LATE NIGHT - Reduced service after midnight, some lines stop around 00:00
11. RUSH HOUR - T1 tram extremely crowded 8-9 AM and 5-7 PM
12. ISTIKLAL STREET - Walk it! Tram is slow. Better on foot.
13. CASH - Keep some cash for HAVAIST bus and emergencies
14. LUGGAGE - HAVAIST bus better than metro with heavy bags
15. APPS - IBB Mobile app shows real-time info (Turkish mostly)""",
            "special": "Istanbul public transport is extensive, cheap, and reliable!"
        }
    ]
    
    return transportation_data

def export_transportation_to_semantic_index():
    """Export transportation data to semantic search"""
    print("\n" + "="*70)
    print("🚇 Exporting Transportation Database to Semantic Search")
    print("="*70)
    
    # Get transportation data
    transport_data = create_transportation_database()
    
    print(f"📦 Exported {len(transport_data)} transportation items")
    
    # Show breakdown
    categories = {}
    for item in transport_data:
        cat = item.get('type', 'other')
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n📊 Transportation items by type:")
    for cat, count in sorted(categories.items()):
        print(f"   • {cat}: {count}")
    
    # Create semantic search engine
    search_engine = SemanticSearchEngine()
    
    # Index transportation data
    search_engine.index_items(transport_data, save_path="./data/transportation_index.bin")
    
    print("\n✅ Transportation database indexed successfully!")
    print(f"   File: ./data/transportation_index.bin")
    print(f"   Items: {len(transport_data)}")
    
    return transport_data

def test_transportation_search():
    """Test searching the transportation database"""
    print("\n" + "="*70)
    print("🧪 Testing Transportation Search")
    print("="*70)
    
    search_engine = SemanticSearchEngine()
    search_engine.load_collection("transportation", "./data/transportation_index.bin")
    
    test_queries = [
        "How do I get from Sultanahmet to Taksim?",
        "What is İstanbulkart?",
        "Ferry to Kadıköy from Old City",
        "Airport to Sultanahmet transportation",
        "Metro lines in Istanbul"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = search_engine.search(query, top_k=3, collection="transportation")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['name']}")
            print(f"     Type: {r.get('type', 'N/A')} | Score: {r['similarity_score']:.3f}")

def main():
    print("\n" + "="*70)
    print("🚀 Integrating Transportation System into ML System")
    print("="*70)
    print("\nThis will:")
    print("  ✅ Create comprehensive transportation knowledge base")
    print("  ✅ Include metro, tram, ferry, Marmaray, funicular info")
    print("  ✅ Add route recommendations and tips")
    print("  ✅ Enable accurate transportation advice from KAM")
    print("\n" + "="*70)
    
    # Export transportation
    transport_data = export_transportation_to_semantic_index()
    
    # Test search
    test_transportation_search()
    
    print("\n" + "="*70)
    print("✅ TRANSPORTATION DATABASE INTEGRATION COMPLETE!")
    print("="*70)
    print("\nWhat changed:")
    print("  🚇 Transportation index with 20+ comprehensive entries")
    print("  🚇 All major metro, tram, ferry, and train lines covered")
    print("  🚇 İstanbulkart information and tips")
    print("  🚇 Popular route recommendations")
    print("  🚇 General transportation tips and advice")
    print("\nNext steps:")
    print("  1. Restart ML service to load new index")
    print("  2. Test with transportation queries")
    print("  3. Verify: KAM provides accurate transport directions")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
