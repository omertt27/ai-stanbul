"""
Hidden Gems Database for Istanbul
Curated list of secret spots, local favorites, and off-the-beaten-path locations
"""

from typing import Dict, List

HIDDEN_GEMS_DATABASE = {
    'sarıyer': [
        {
            'name': 'Kilyos Hidden Beach (Kumköy)',
            'type': 'nature',
            'category': 'beach',
            'description': 'Secluded beach on the Black Sea coast known only to locals',
            'how_to_find': 'Take bus 151 from Sarıyer to Kilyos, then walk 15 minutes north along the coast',
            'local_tip': 'Best visited on weekdays to avoid crowds. Bring your own food as restaurants are limited',
            'best_time': 'June-September, early morning or sunset',
            'cost': 'Free',
            'hidden_factor': 9,  # 1-10 scale
            'coordinates': {'lat': 41.2356, 'lng': 29.0489},
            'tags': ['beach', 'nature', 'peaceful', 'swimming', 'sunset']
        },
        {
            'name': 'Rumelifeneri Lighthouse Café',
            'type': 'cafe',
            'category': 'food',
            'description': 'Charming café next to historic lighthouse at the northernmost point of European Istanbul',
            'how_to_find': 'Take bus 150 to end of line, walk 5 minutes to lighthouse. Look for small café with red umbrellas',
            'local_tip': 'Order Turkish breakfast and sit outside for stunning Black Sea views',
            'best_time': 'Saturday/Sunday breakfast (09:00-12:00)',
            'cost': '150-250 TL per person',
            'hidden_factor': 8,
            'coordinates': {'lat': 41.2314, 'lng': 29.0756},
            'tags': ['cafe', 'breakfast', 'view', 'lighthouse', 'black-sea']
        },
        {
            'name': 'Bahçeköy Nature Park Secret Trail',
            'type': 'nature',
            'category': 'hiking',
            'description': 'Off-trail hiking path through Belgrade Forest with hidden waterfall',
            'how_to_find': 'Enter Belgrade Forest from Bahçeköy entrance. Follow red markers for 2km, then look for unmarked path on right',
            'local_tip': 'Locals come here for wild blackberries in late summer. Bring water and snacks',
            'best_time': 'April-October, weekday mornings',
            'cost': 'Free',
            'hidden_factor': 10,
            'coordinates': {'lat': 41.1678, 'lng': 28.9834},
            'tags': ['hiking', 'nature', 'waterfall', 'forest', 'adventure']
        },
        {
            'name': 'Zekeriyaköy Village Breakfast Houses',
            'type': 'cafe',
            'category': 'food',
            'description': 'Traditional village breakfast spots where locals go on weekends',
            'how_to_find': 'Take minibus from Maslak to Zekeriyaköy. Ask for "köy kahvaltısı" - villagers will point you to best spots',
            'local_tip': 'Try "menemen" and homemade jams. These places don\'t have signs - look for gardens with tables',
            'best_time': 'Sunday morning (08:00-11:00)',
            'cost': '200-350 TL per person',
            'hidden_factor': 9,
            'coordinates': {'lat': 41.1523, 'lng': 28.9912},
            'tags': ['breakfast', 'village', 'authentic', 'nature', 'traditional']
        },
        {
            'name': 'Garipçe Fish Restaurants Waterfront',
            'type': 'restaurant',
            'category': 'food',
            'description': 'Tiny fishing village with authentic seafood restaurants right on Bosphorus',
            'how_to_find': 'Bus 150E to Garipçe village. Walk to waterfront - 5-6 family-run restaurants with no tourists',
            'local_tip': 'Ask for "günün balığı" (catch of the day) and eat on terrace overlooking fishing boats',
            'best_time': 'Lunch or sunset dinner',
            'cost': '400-600 TL per person',
            'hidden_factor': 8,
            'coordinates': {'lat': 41.2156, 'lng': 29.1023},
            'tags': ['seafood', 'bosphorus', 'authentic', 'fishing-village', 'romantic']
        }
    ],
    
    'beşiktaş': [
        {
            'name': 'Akaretler Row Houses',
            'type': 'historical',
            'category': 'architecture',
            'description': 'Beautifully restored 19th-century Ottoman row houses now hosting art galleries and boutiques',
            'how_to_find': 'Behind Çırağan Palace. Enter from Salhane Street. Most tourists walk right past it',
            'local_tip': 'Perfect for afternoon walk. Check out W Hotel\'s hidden garden bar at the end',
            'best_time': 'Any afternoon, especially weekdays',
            'cost': 'Free to walk around',
            'hidden_factor': 7,
            'coordinates': {'lat': 41.0478, 'lng': 29.0089},
            'tags': ['architecture', 'history', 'galleries', 'boutique', 'photography']
        },
        {
            'name': 'Yıldız Park Hidden Tea Garden',
            'type': 'cafe',
            'category': 'nature',
            'description': 'Secret tea garden deep inside Yıldız Park, known only to locals',
            'how_to_find': 'Enter Yıldız Park from Beşiktaş side. Follow uphill path past Malta Pavilion, continue 10 minutes',
            'local_tip': 'Brings own snacks (allowed). Best view of Bosphorus from here',
            'best_time': 'Sunset (one hour before dark)',
            'cost': '50-100 TL',
            'hidden_factor': 9,
            'coordinates': {'lat': 41.0523, 'lng': 29.0145},
            'tags': ['tea-garden', 'park', 'view', 'peaceful', 'nature']
        },
        {
            'name': 'Ortaköy Secret Stairway to Bosphorus',
            'type': 'viewpoint',
            'category': 'scenic',
            'description': 'Hidden stairway leading to private fishing spots on Bosphorus shore',
            'how_to_find': 'From Ortaköy Square, walk north 200m. Look for narrow stone stairs between buildings #47 and #49',
            'local_tip': 'Local fishermen gather here at dawn. Great photo spot for Bosphorus Bridge',
            'best_time': 'Early morning or sunset',
            'cost': 'Free',
            'hidden_factor': 8,
            'coordinates': {'lat': 41.0556, 'lng': 29.0275},
            'tags': ['viewpoint', 'photography', 'bosphorus', 'secret', 'romantic']
        },
        {
            'name': 'Abbasağa Park Underground Spring',
            'type': 'nature',
            'category': 'historical',
            'description': 'Ancient Byzantine spring still flowing in park. Locals fill bottles with fresh water',
            'how_to_find': 'Enter Abbasağa Park from Beşiktaş side. Follow path downhill to northeast corner',
            'local_tip': 'Bring empty bottle - locals believe this water is blessed. Free and safe to drink',
            'best_time': 'Any time, less crowded mornings',
            'cost': 'Free',
            'hidden_factor': 7,
            'coordinates': {'lat': 41.0428, 'lng': 28.9943},
            'tags': ['history', 'nature', 'spring', 'byzantine', 'local-tradition']
        },
        {
            'name': 'Beşiktaş Fish Market Hidden Breakfast',
            'type': 'cafe',
            'category': 'food',
            'description': 'Tiny breakfast place inside fish market, known only to market workers',
            'how_to_find': 'Enter Beşiktaş fish market from Barbaros Boulevard. Walk to back, look for stairs going down',
            'local_tip': 'Order "tam kahvaltı" (full breakfast). Cash only, no menu - they tell you what\'s fresh',
            'best_time': '07:00-10:00 (closes when food runs out)',
            'cost': '150-200 TL',
            'hidden_factor': 10,
            'coordinates': {'lat': 41.0423, 'lng': 29.0067},
            'tags': ['breakfast', 'authentic', 'market', 'local', 'cheap-eats']
        }
    ],
    
    'beyoğlu': [
        {
            'name': 'Çukurcuma Antique District',
            'type': 'shopping',
            'category': 'antiques',
            'description': 'Maze of narrow streets filled with antique shops, vintage furniture, and hidden galleries',
            'how_to_find': 'From Galatasaray High School, walk down Turnacıbaşı Street. Turn left at Çukurcuma Street',
            'local_tip': 'Best on Saturday afternoons. Shop owners are friendly and love to chat about pieces',
            'best_time': 'Saturday 14:00-18:00',
            'cost': 'Free to browse',
            'hidden_factor': 6,
            'coordinates': {'lat': 41.0334, 'lng': 28.9789},
            'tags': ['antiques', 'shopping', 'vintage', 'galleries', 'unique']
        },
        {
            'name': 'Asmalımescit Hidden Meyhanes',
            'type': 'restaurant',
            'category': 'nightlife',
            'description': 'Unmarked traditional meyhanes (taverns) in back alleys where locals drink rakı',
            'how_to_find': 'From Istiklal, enter Asmalımescit Street. Look for doors with only numbers, no signs',
            'local_tip': 'Ring doorbell to enter. Order mezze platter and rakı. Live music after 21:00',
            'best_time': 'Friday/Saturday night after 20:00',
            'cost': '500-800 TL per person',
            'hidden_factor': 9,
            'coordinates': {'lat': 41.0312, 'lng': 28.9756},
            'tags': ['meyhane', 'nightlife', 'rakı', 'live-music', 'authentic']
        },
        {
            'name': 'Galata Tower Secret Terrace',
            'type': 'viewpoint',
            'category': 'scenic',
            'description': 'Lesser-known viewing platform on building next to Galata Tower - same view, no crowds',
            'how_to_find': 'Walk past Galata Tower entrance. Building #42 has rooftop café - take elevator to 6th floor',
            'local_tip': 'Sunset views rival Galata Tower. Order Turkish coffee and sit for hours',
            'best_time': 'Sunset (one hour before dark)',
            'cost': '100-150 TL (coffee mandatory)',
            'hidden_factor': 8,
            'coordinates': {'lat': 41.0256, 'lng': 28.9739},
            'tags': ['viewpoint', 'cafe', 'sunset', 'alternative', 'photography']
        },
        {
            'name': 'French Street (Cezayir Sokağı) Hidden Passage',
            'type': 'architecture',
            'category': 'scenic',
            'description': 'Beautiful covered passage connecting French Street to Nevizade, covered in flowers',
            'how_to_find': 'On French Street, look for archway between restaurants #5 and #7. Passage is lit with fairy lights',
            'local_tip': 'Instagram heaven! Best photos in late afternoon when light comes through flower vines',
            'best_time': '16:00-18:00 for best light',
            'cost': 'Free',
            'hidden_factor': 7,
            'coordinates': {'lat': 41.0323, 'lng': 28.9745},
            'tags': ['photography', 'architecture', 'instagram', 'flowers', 'romantic']
        },
        {
            'name': 'Pera Palace Hidden Bar',
            'type': 'cafe',
            'category': 'historical',
            'description': 'Secret Agatha Christie-themed bar in historic Pera Palace Hotel basement',
            'how_to_find': 'Enter Pera Palace Hotel lobby. Take stairs down (not elevator). Look for unmarked door',
            'local_tip': 'Order "Orient Express" cocktail. Walls have Agatha Christie memorabilia',
            'best_time': 'Evening after 19:00',
            'cost': '300-500 TL per person',
            'hidden_factor': 8,
            'coordinates': {'lat': 41.0312, 'lng': 28.9767},
            'tags': ['bar', 'historical', 'luxury', 'cocktails', 'themed']
        },
        {
            'name': 'Balat Street Art Walls',
            'type': 'art',
            'category': 'cultural',
            'description': 'Colorful murals and street art by local artists in Balat\'s back streets',
            'how_to_find': 'From Balat ferry station, walk uphill. Turn into any narrow street - murals everywhere',
            'local_tip': 'Best between 10:00-16:00 for good lighting. Local artists sometimes paint on weekends',
            'best_time': 'Morning or early afternoon',
            'cost': 'Free',
            'hidden_factor': 5,
            'coordinates': {'lat': 41.0289, 'lng': 28.9489},
            'tags': ['street-art', 'photography', 'colorful', 'instagram', 'walking']
        },
        {
            'name': 'Tünel Passage Bookshops',
            'type': 'shopping',
            'category': 'books',
            'description': 'Hidden passage with second-hand bookshops, vintage records, and old postcards',
            'how_to_find': 'From Tünel Square, take Tünel Passage towards İstiklal. Shops are inside the passage',
            'local_tip': 'Owners speak multiple languages. Great for finding rare Istanbul books and maps',
            'best_time': 'Any afternoon',
            'cost': 'Free to browse',
            'hidden_factor': 7,
            'coordinates': {'lat': 41.0289, 'lng': 28.9734},
            'tags': ['books', 'vintage', 'shopping', 'cultural', 'nostalgic']
        }
    ],
    
    'kadıköy': [
        {
            'name': 'Moda Abandoned Pier',
            'type': 'viewpoint',
            'category': 'scenic',
            'description': 'Old abandoned pier where locals fish and watch sunset over European side',
            'how_to_find': 'Walk to end of Moda coastline. Continue past last café - pier is behind fence (but accessible)',
            'local_tip': 'Bring drinks and snacks. Locals gather here for sunset. Perfect picnic spot',
            'best_time': 'Sunset',
            'cost': 'Free',
            'hidden_factor': 8,
            'coordinates': {'lat': 40.9867, 'lng': 29.0289},
            'tags': ['viewpoint', 'sunset', 'fishing', 'romantic', 'picnic']
        },
        {
            'name': 'Yeldeğirmeni Street Art Quarter',
            'type': 'art',
            'category': 'cultural',
            'description': 'Entire neighborhood transformed into open-air gallery with massive murals',
            'how_to_find': 'From Kadıköy pier, walk 15 minutes inland. Start at Yeldegirmeni Street and explore',
            'local_tip': 'New murals added each year during art festival. Cafes have maps showing all artworks',
            'best_time': 'Morning for best light',
            'cost': 'Free',
            'hidden_factor': 6,
            'coordinates': {'lat': 40.9889, 'lng': 29.0323},
            'tags': ['street-art', 'murals', 'photography', 'walking', 'instagram']
        },
        {
            'name': 'Çarşı Market Hidden Food Stalls',
            'type': 'food',
            'category': 'street-food',
            'description': 'Tiny food stalls inside Kadıköy market serving local specialties',
            'how_to_find': 'Enter main market from seafront. Walk to very back - food stalls are in last alley',
            'local_tip': 'Try "kokoreç" and "midye dolma". Only locals eat here - very authentic',
            'best_time': 'Lunch (12:00-14:00)',
            'cost': '50-150 TL',
            'hidden_factor': 9,
            'coordinates': {'lat': 40.9878, 'lng': 29.0245},
            'tags': ['street-food', 'market', 'authentic', 'cheap-eats', 'local']
        },
        {
            'name': 'Fenerbahçe Park Secret Beach',
            'type': 'nature',
            'category': 'beach',
            'description': 'Small rocky beach hidden at end of Fenerbahçe Park, known to few',
            'how_to_find': 'Take tram to Fenerbahçe. Walk through park to southern tip. Beach is past lighthouse',
            'local_tip': 'Bring towel and snacks. Great for swimming away from crowds',
            'best_time': 'Summer mornings',
            'cost': 'Free',
            'hidden_factor': 8,
            'coordinates': {'lat': 40.9623, 'lng': 29.0467},
            'tags': ['beach', 'swimming', 'peaceful', 'nature', 'hidden']
        },
        {
            'name': 'Kadıköy Vinyl Record Shops',
            'type': 'shopping',
            'category': 'music',
            'description': 'Cluster of vintage vinyl shops in back streets of Kadıköy',
            'how_to_find': 'From Kadıköy square, head to Altıyol. Look for "Plakçılar Çarşısı" signs',
            'local_tip': 'Owners are passionate collectors. Ask for Turkish rock/arabesque recommendations',
            'best_time': 'Afternoon',
            'cost': 'Varies by record',
            'hidden_factor': 7,
            'coordinates': {'lat': 40.9890, 'lng': 29.0267},
            'tags': ['vinyl', 'music', 'vintage', 'shopping', 'culture']
        }
    ],
    
    'üsküdar': [
        {
            'name': 'Kuzguncuk Wooden Houses',
            'type': 'neighborhood',
            'category': 'architecture',
            'description': 'Perfectly preserved Ottoman-era wooden houses in tiny village-like neighborhood',
            'how_to_find': 'Ferry to Üsküdar, then bus to Kuzguncuk. Walk uphill from waterfront',
            'local_tip': 'Perfect for photography. Grab börek from corner bakery and explore narrow streets',
            'best_time': 'Morning (fewer tourists)',
            'cost': 'Free',
            'hidden_factor': 6,
            'coordinates': {'lat': 41.0212, 'lng': 29.0623},
            'tags': ['architecture', 'ottoman', 'photography', 'peaceful', 'village']
        },
        {
            'name': 'Çamlıca Hill Secret Tea Garden',
            'type': 'cafe',
            'category': 'viewpoint',
            'description': 'Hidden tea garden on Çamlıca Hill with panoramic city views, missed by most visitors',
            'how_to_find': 'Take bus to Çamlıca Hill. Walk past main tower - tea garden is 300m behind it',
            'local_tip': 'Brings own cheese/olive platter (allowed). Best sunset view in Asian side',
            'best_time': 'Sunset',
            'cost': '50-100 TL',
            'hidden_factor': 7,
            'coordinates': {'lat': 41.0234, 'lng': 29.0678},
            'tags': ['tea-garden', 'viewpoint', 'sunset', 'panoramic', 'romantic']
        },
        {
            'name': 'Maiden\'s Tower Secret Boat Access',
            'type': 'transportation',
            'category': 'experience',
            'description': 'Private boats to Maiden\'s Tower from Üsküdar shore, cheaper than official tours',
            'how_to_find': 'From Üsküdar ferry terminal, walk south along coast. Local boatmen near Salacak pier',
            'local_tip': 'Negotiate price (should be 100-150 TL return). Same experience as official boats',
            'best_time': 'Late afternoon',
            'cost': '100-150 TL per person',
            'hidden_factor': 8,
            'coordinates': {'lat': 41.0134, 'lng': 29.0045},
            'tags': ['boat', 'experience', 'landmark', 'budget', 'adventure']
        },
        {
            'name': 'Üsküdar Backstreet Künefe Shop',
            'type': 'food',
            'category': 'dessert',
            'description': 'Tiny family-run künefe (cheese dessert) shop, best in Istanbul according to locals',
            'how_to_find': 'From Üsküdar square, walk toward Valide Sultan Mosque. Shop is on small street behind mosque',
            'local_tip': 'Order "fıstıklı künefe" (with pistachios). Eat immediately while hot',
            'best_time': 'After dinner (20:00-22:00)',
            'cost': '100-150 TL',
            'hidden_factor': 9,
            'coordinates': {'lat': 41.0245, 'lng': 29.0089},
            'tags': ['dessert', 'authentic', 'local', 'künefe', 'family-run']
        }
    ],
    
    'fatih': [
        {
            'name': 'Süleymaniye Mosque Hidden Courtyard',
            'type': 'historical',
            'category': 'architecture',
            'description': 'Secret inner courtyard of Süleymaniye with peaceful garden and stunning views',
            'how_to_find': 'Enter mosque complex. Instead of main mosque, look for smaller door on right side',
            'local_tip': 'Perfect for quiet reflection. Few tourists find this spot. Free entry',
            'best_time': 'Early morning or late afternoon',
            'cost': 'Free',
            'hidden_factor': 7,
            'coordinates': {'lat': 41.0167, 'lng': 28.9645},
            'tags': ['mosque', 'courtyard', 'peaceful', 'viewpoint', 'architecture']
        },
        {
            'name': 'Balat Juice Street',
            'type': 'food',
            'category': 'drinks',
            'description': 'Hidden alley in Balat where locals get fresh-squeezed juice for 20 TL',
            'how_to_find': 'Walk uphill from Balat pier. Turn left at rainbow stairs, then first right',
            'local_tip': 'Try "karışık meyve suyu" (mixed fruit juice). Friendliest vendors in Istanbul',
            'best_time': 'Morning',
            'cost': '20-40 TL',
            'hidden_factor': 8,
            'coordinates': {'lat': 41.0278, 'lng': 28.9478},
            'tags': ['juice', 'fresh', 'cheap', 'local', 'healthy']
        },
        {
            'name': 'Fener Greek Orthodox Patriarchate Garden',
            'type': 'historical',
            'category': 'cultural',
            'description': 'Beautiful hidden garden at Greek Patriarchate, open to respectful visitors',
            'how_to_find': 'Walk to Fener neighborhood. Patriarchate has red building - ring bell at gate',
            'local_tip': 'Dress modestly. Guards will show you garden if you ask nicely',
            'best_time': 'Weekday mornings',
            'cost': 'Free (donations welcome)',
            'hidden_factor': 9,
            'coordinates': {'lat': 41.0289, 'lng': 28.9512},
            'tags': ['historical', 'garden', 'cultural', 'religious', 'peaceful']
        }
    ]
}


def get_all_neighborhoods() -> List[str]:
    """Get list of all neighborhoods in database"""
    return list(HIDDEN_GEMS_DATABASE.keys())


def get_gems_by_neighborhood(neighborhood: str) -> List[Dict]:
    """Get all hidden gems for a specific neighborhood"""
    return HIDDEN_GEMS_DATABASE.get(neighborhood.lower(), [])


def get_all_hidden_gems() -> List[Dict]:
    """Get all hidden gems across all neighborhoods"""
    all_gems = []
    for gems in HIDDEN_GEMS_DATABASE.values():
        all_gems.extend(gems)
    return all_gems


def get_gems_by_type(gem_type: str) -> List[Dict]:
    """Get hidden gems filtered by type (cafe, nature, historical, etc.)"""
    all_gems = get_all_hidden_gems()
    return [gem for gem in all_gems if gem['type'].lower() == gem_type.lower()]


def get_gems_by_category(category: str) -> List[Dict]:
    """Get hidden gems filtered by category"""
    all_gems = get_all_hidden_gems()
    return [gem for gem in all_gems if gem['category'].lower() == category.lower()]


def get_top_hidden_gems(limit: int = 10) -> List[Dict]:
    """Get top hidden gems sorted by hidden_factor"""
    all_gems = get_all_hidden_gems()
    sorted_gems = sorted(all_gems, key=lambda x: x['hidden_factor'], reverse=True)
    return sorted_gems[:limit]


# Statistics
TOTAL_HIDDEN_GEMS = len(get_all_hidden_gems())
NEIGHBORHOODS_COVERED = len(get_all_neighborhoods())

print(f"✅ Hidden Gems Database Loaded: {TOTAL_HIDDEN_GEMS} gems across {NEIGHBORHOODS_COVERED} neighborhoods")
