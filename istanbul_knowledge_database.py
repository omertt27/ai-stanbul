#!/usr/bin/env python3
"""
Istanbul Knowledge Database
==========================

Comprehensive knowledge base for Istanbul attractions, districts, and practical information
to enhance the AI Istanbul system with detailed local insights.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class AttractionInfo:
    """Detailed attraction information"""
    name: str
    turkish_name: str
    district: str
    category: str
    description: str
    opening_hours: Dict[str, str]
    entrance_fee: str
    transportation: List[str]
    nearby_attractions: List[str]
    duration: str
    best_time: str
    cultural_significance: str
    practical_tips: List[str]

@dataclass
class DistrictProfile:
    """Comprehensive district profile"""
    name: str
    turkish_name: str
    description: str
    character: str
    main_attractions: List[str]
    hidden_gems: List[str]
    local_specialties: List[str]
    transportation_hubs: List[str]
    walking_areas: List[str]
    dining_scene: str
    shopping: List[str]
    cultural_context: str
    local_tips: List[str]
    nearby_districts: List[str]

class IstanbulKnowledgeDatabase:
    """Comprehensive Istanbul knowledge database"""
    
    def __init__(self):
        self.attractions = self._load_attractions()
        self.districts = self._load_districts()
        self.turkish_phrases = self._load_turkish_phrases()
        self.cultural_context = self._load_cultural_context()
        self.practical_info = self._load_practical_info()
        # NEW: Critical additions for identified gaps
        self.free_attractions = self._load_free_attractions()
        self.alternative_culture_venues = self._load_alternative_culture_venues()
        self.enhanced_transportation = self._load_enhanced_transportation()
        self.detailed_practical_info = self._load_detailed_practical_info()
    
    def _load_attractions(self) -> Dict[str, AttractionInfo]:
        """Load comprehensive attraction database"""
        return {
            'hagia_sophia': AttractionInfo(
                name="Hagia Sophia",
                turkish_name="Ayasofya",
                district="sultanahmet",
                category="historic_monument",
                description="Former Byzantine cathedral and Ottoman mosque, now a mosque again",
                opening_hours={
                    "monday": "Closed to tourists during prayer times",
                    "tuesday": "09:00-18:00 (except prayer times)",
                    "wednesday": "09:00-18:00 (except prayer times)",
                    "thursday": "09:00-18:00 (except prayer times)",
                    "friday": "14:00-18:00 (morning prayers)",
                    "saturday": "09:00-18:00 (except prayer times)",
                    "sunday": "09:00-18:00 (except prayer times)"
                },
                entrance_fee="Free (functioning mosque)",
                transportation=[
                    "Sultanahmet Tram Station (5-min walk)",
                    "Eminönü Ferry Terminal (10-min walk)",
                    "Gülhane Metro Station (8-min walk)"
                ],
                nearby_attractions=["Blue Mosque", "Topkapi Palace", "Basilica Cistern"],
                duration="1-2 hours",
                best_time="Early morning or late afternoon",
                cultural_significance="UNESCO World Heritage Site, symbol of Istanbul's layered history",
                practical_tips=[
                    "Dress modestly (head covering for women)",
                    "Remove shoes before entering",
                    "Free entrance but donations appreciated",
                    "Photography allowed in most areas"
                ]
            ),
            'blue_mosque': AttractionInfo(
                name="Blue Mosque",
                turkish_name="Sultanahmet Camii",
                district="sultanahmet",
                category="religious_site",
                description="17th-century mosque famous for its blue İznik tiles",
                opening_hours={
                    "daily": "08:30-12:00, 14:00-16:30, 17:30-18:30 (outside prayer times)"
                },
                entrance_fee="Free",
                transportation=[
                    "Sultanahmet Tram Station (3-min walk)",
                    "Eminönü Ferry Terminal (8-min walk)"
                ],
                nearby_attractions=["Hagia Sophia", "Hippodrome", "Grand Bazaar"],
                duration="45 minutes - 1 hour",
                best_time="Early morning for fewer crowds",
                cultural_significance="Active mosque, masterpiece of Ottoman architecture",
                practical_tips=[
                    "Mandatory modest dress and head covering",
                    "Free shoe bags provided",
                    "No entrance during prayer times",
                    "Tourist entrance separate from worshippers"
                ]
            ),
            'topkapi_palace': AttractionInfo(
                name="Topkapi Palace",
                turkish_name="Topkapı Sarayı",
                district="sultanahmet",
                category="museum",
                description="Former Ottoman imperial palace with treasury and harem",
                opening_hours={
                    "tuesday": "Closed",
                    "other_days": "09:00-18:00 (winter), 09:00-19:00 (summer)"
                },
                entrance_fee="Moderate (separate fee for Harem)",
                transportation=[
                    "Sultanahmet Tram Station (7-min walk)",
                    "Gülhane Metro Station (5-min walk)"
                ],
                nearby_attractions=["Hagia Sophia", "Archaeological Museum", "Gülhane Park"],
                duration="2-4 hours",
                best_time="Early morning to avoid crowds",
                cultural_significance="Center of Ottoman Empire for 400 years",
                practical_tips=[
                    "Buy tickets online to skip lines",
                    "Harem requires separate ticket",
                    "Audio guide recommended",
                    "Comfortable shoes essential"
                ]
            ),
            'galata_tower': AttractionInfo(
                name="Galata Tower",
                turkish_name="Galata Kulesi",
                district="galata",
                category="historic_monument",
                description="Medieval tower offering panoramic city views",
                opening_hours={
                    "daily": "08:30-22:00"
                },
                entrance_fee="Moderate",
                transportation=[
                    "Karaköy Metro Station (8-min walk)",
                    "Galata Bridge (10-min uphill walk)",
                    "Tünel historic funicular (3-min walk)"
                ],
                nearby_attractions=["Galata Bridge", "İstiklal Street", "Pera Museum"],
                duration="1-2 hours",
                best_time="Sunset for best views",
                cultural_significance="Byzantine-era watchtower, symbol of Galata district",
                practical_tips=[
                    "Book timed entry tickets online",
                    "Elevator available but expect queues",
                    "Restaurant at top level",
                    "360-degree observation deck"
                ]
            ),
            'grand_bazaar': AttractionInfo(
                name="Grand Bazaar",
                turkish_name="Kapalıçarşı",
                district="eminönü",
                category="shopping",
                description="Historic covered market with 4,000 shops",
                opening_hours={
                    "monday_saturday": "09:00-19:00",
                    "sunday": "Closed"
                },
                entrance_fee="Free",
                transportation=[
                    "Beyazıt-Kapalıçarşı Tram Station (direct access)",
                    "Eminönü Ferry Terminal (10-min walk)"
                ],
                nearby_attractions=["Spice Bazaar", "Süleymaniye Mosque", "Blue Mosque"],
                duration="2-4 hours",
                best_time="Late morning for better selection",
                cultural_significance="One of world's oldest and largest covered markets",
                practical_tips=[
                    "Bargaining expected and encouraged",
                    "Bring cash, some accept cards",
                    "Keep valuables secure",
                    "Try Turkish tea offered by shopkeepers"
                ]
            ),
            'dolmabahce_palace': AttractionInfo(
                name="Dolmabahçe Palace",
                turkish_name="Dolmabahçe Sarayı",
                district="beşiktaş",
                category="museum",
                description="Opulent 19th-century Ottoman palace on the Bosphorus",
                opening_hours={
                    "monday_tuesday": "Closed",
                    "other_days": "09:00-16:00 (winter), 09:00-17:00 (summer)"
                },
                entrance_fee="Moderate (separate tickets for different sections)",
                transportation=[
                    "Dolmabahçe-Kabataş Tram Station (5-min walk)",
                    "Kabataş Ferry Terminal (7-min walk)",
                    "Beşiktaş Ferry Terminal (15-min walk)"
                ],
                nearby_attractions=["Beşiktaş Square", "Yıldız Park", "Naval Museum"],
                duration="2-3 hours",
                best_time="Morning for better lighting",
                cultural_significance="Last residence of Ottoman sultans, where Atatürk died",
                practical_tips=[
                    "Guided tours mandatory (included in ticket)",
                    "No photography inside",
                    "Separate tickets for Selamlık and Harem",
                    "Crystal staircase and world's largest chandelier"
                ]
            ),
            # DIVERSIFIED ATTRACTIONS - Hidden Gems & Lesser-Known Sites
            'basilica_cistern': AttractionInfo(
                name="Basilica Cistern",
                turkish_name="Yerebatan Sarnıcı",
                district="sultanahmet",
                category="historic_monument",
                description="Ancient underground cistern with mystical atmosphere",
                opening_hours={"daily": "09:00-18:00"},
                entrance_fee="Moderate",
                transportation=["Sultanahmet Tram Station (2-min walk)"],
                nearby_attractions=["Hagia Sophia", "Blue Mosque"],
                duration="45 minutes",
                best_time="Early morning or late afternoon",
                cultural_significance="Byzantine engineering marvel, atmospheric underground experience",
                practical_tips=["Cool temperature year-round", "Photography allowed", "Wheelchair accessible"]
            ),
            'suleymaniye_mosque': AttractionInfo(
                name="Süleymaniye Mosque",
                turkish_name="Süleymaniye Camii",
                district="eminönü",
                category="religious_site",
                description="Magnificent mosque complex by architect Sinan, less crowded than Blue Mosque",
                opening_hours={"daily": "Outside prayer times"},
                entrance_fee="Free",
                transportation=["Eminönü Ferry Terminal (10-min walk)", "Beyazıt-Kapalıçarşı Tram (8-min walk)"],
                nearby_attractions=["Grand Bazaar", "Spice Bazaar"],
                duration="1 hour",
                best_time="Late afternoon for golden light",
                cultural_significance="Masterpiece of Ottoman architecture, stunning city views",
                practical_tips=["Less touristy alternative to Blue Mosque", "Beautiful cemetery with Bosphorus views"]
            ),
            'chora_church': AttractionInfo(
                name="Chora Church",
                turkish_name="Kariye Müzesi",
                district="fatih",
                category="museum",
                description="Hidden gem with world's finest Byzantine mosaics and frescoes",
                opening_hours={"wednesday": "Closed", "other_days": "09:00-17:00"},
                entrance_fee="Moderate",
                transportation=["Bus from Eminönü (30 min)", "Taxi recommended"],
                nearby_attractions=["Eyüp Sultan Mosque", "Golden Horn"],
                duration="1-2 hours",
                best_time="Morning for better lighting",
                cultural_significance="UNESCO candidate, Byzantine art masterpiece",
                practical_tips=["Off the beaten path", "Bring good camera", "Combine with Eyüp visit"]
            ),
            'pierre_loti_hill': AttractionInfo(
                name="Pierre Loti Hill",
                turkish_name="Pierre Loti Tepesi",
                district="eyüp",
                category="viewpoint",
                description="Panoramic Golden Horn views with cable car access",
                opening_hours={"daily": "08:00-24:00 (cable car until 23:00)"},
                entrance_fee="Budget (cable car fee)",
                transportation=["Cable car from Eyüp", "Bus to Eyüp then cable car"],
                nearby_attractions=["Eyüp Sultan Mosque", "Golden Horn"],
                duration="1-2 hours",
                best_time="Sunset for spectacular views",
                cultural_significance="Named after French writer, authentic local atmosphere",
                practical_tips=["Traditional tea gardens", "Great for sunset photos", "Less touristy viewpoint"]
            ),
            'maiden_tower': AttractionInfo(
                name="Maiden's Tower",
                turkish_name="Kız Kulesi",
                district="üsküdar",
                category="historic_monument",
                description="Iconic tower on small island with restaurant and museum",
                opening_hours={"daily": "09:00-18:00"},
                entrance_fee="Moderate (includes boat transfer)",
                transportation=["Boat from Üsküdar or Kabataş"],
                nearby_attractions=["Üsküdar waterfront", "Salacak shore"],
                duration="2-3 hours",
                best_time="Sunset dinner for romantic experience",
                cultural_significance="Symbol of Istanbul, legendary love stories",
                practical_tips=["Book restaurant in advance", "Boat ride included", "Perfect for proposals"]
            ),
            # FAMILY-FRIENDLY ATTRACTIONS
            'miniaturk': AttractionInfo(
                name="Miniaturk",
                turkish_name="Miniaturk",
                district="beyoğlu",
                category="family_attraction",
                description="Miniature park with scale models of Turkey's landmarks",
                opening_hours={"daily": "09:00-19:00"},
                entrance_fee="Budget",
                transportation=["Golden Horn ferry", "Bus from city center"],
                nearby_attractions=["Golden Horn", "Fener district"],
                duration="2-3 hours",
                best_time="Morning for family visits",
                cultural_significance="Educational overview of Turkish heritage",
                practical_tips=["Great for kids", "Wheelchair accessible", "Cafe on-site", "Interactive exhibits"]
            ),
            'rahmi_koc_museum': AttractionInfo(
                name="Rahmi M. Koç Museum",
                turkish_name="Rahmi M. Koç Müzesi",
                district="beyoğlu",
                category="family_attraction",
                description="Industrial museum with submarines, trains, and interactive exhibits",
                opening_hours={"monday": "Closed", "other_days": "10:00-17:00"},
                entrance_fee="Moderate",
                transportation=["Golden Horn ferry", "Bus connections"],
                nearby_attractions=["Miniaturk", "Golden Horn"],
                duration="3-4 hours",
                best_time="Weekday mornings",
                cultural_significance="Turkey's first industrial museum",
                practical_tips=["Perfect for children", "Hands-on exhibits", "Submarine tour", "Cafe with Golden Horn view"]
            ),
            # ROMANTIC ATTRACTIONS
            'galata_bridge_sunset': AttractionInfo(
                name="Galata Bridge Sunset",
                turkish_name="Galata Köprüsü Gün Batımı",
                district="eminönü",
                category="romantic_spot",
                description="Iconic bridge famous for fishermen and sunset views",
                opening_hours={"daily": "24 hours"},
                entrance_fee="Free",
                transportation=["Eminönü Tram Station", "Karaköy Metro"],
                nearby_attractions=["Spice Bazaar", "Galata Tower"],
                duration="1 hour",
                best_time="Sunset (golden hour)",
                cultural_significance="Historic Golden Horn crossing, local fishing culture",
                practical_tips=["Evening stroll recommended", "Fish restaurants below", "Street musicians", "Perfect for couples"]
            ),
            'bosphorus_sunset_cruise': AttractionInfo(
                name="Bosphorus Sunset Cruise",
                turkish_name="Boğaz Gün Batımı Turu",
                district="eminönü",
                category="romantic_experience",
                description="Romantic boat cruise between Europe and Asia at sunset",
                opening_hours={"daily": "Various departure times"},
                entrance_fee="Moderate to upscale",
                transportation=["Eminönü Ferry Terminal", "Kabataş Ferry Terminal"],
                nearby_attractions=["Dolmabahçe Palace", "Bosphorus shores"],
                duration="2-3 hours",
                best_time="Sunset cruises",
                cultural_significance="Two continents connection, Ottoman palaces from water",
                practical_tips=["Book dinner cruise", "Bring jacket", "Photography opportunity", "Most romantic experience"]
            ),
            # BUDGET-FRIENDLY ATTRACTIONS
            'gulhane_park': AttractionInfo(
                name="Gülhane Park",
                turkish_name="Gülhane Parkı",
                district="sultanahmet",
                category="park",
                description="Historic park with tulip gardens and Bosphorus views",
                opening_hours={"daily": "24 hours"},
                entrance_fee="Free",
                transportation=["Gülhane Metro Station (direct)", "Sultanahmet Tram (5-min walk)"],
                nearby_attractions=["Topkapi Palace", "Archaeological Museum"],
                duration="1-2 hours",
                best_time="Spring for tulips, any time for picnic",
                cultural_significance="Former Ottoman palace gardens",
                practical_tips=["Free admission", "Great for picnics", "Spring tulip festival", "Safe for families"]
            ),
            'kumkapi_fish_market': AttractionInfo(
                name="Kumkapı Fish Market",
                turkish_name="Kumkapı Balık Pazarı",
                district="fatih",
                category="local_experience",
                description="Authentic fish market and meyhane district",
                opening_hours={"daily": "Market: 06:00-20:00, Restaurants: 18:00-02:00"},
                entrance_fee="Free (food costs vary)",
                transportation=["Kumkapı Train Station", "Aksaray Metro then bus"],
                nearby_attractions=["Sea of Marmara shore", "Little Hagia Sophia"],
                duration="2-3 hours",
                best_time="Evening for meyhane culture",
                cultural_significance="Traditional fishing community, authentic meyhane experience",
                practical_tips=["Budget-friendly dining", "Live music evenings", "Authentic local experience", "Try seasonal fish"]
            )
        }
    
    def _load_districts(self) -> Dict[str, DistrictProfile]:
        """Load comprehensive district profiles"""
        return {
            'sultanahmet': DistrictProfile(
                name="Sultanahmet",
                turkish_name="Sultanahmet",
                description="Historic heart of Istanbul with Byzantine and Ottoman monuments",
                character="Tourist-focused historic peninsula with world-class museums and monuments",
                main_attractions=["Hagia Sophia", "Blue Mosque", "Topkapi Palace", "Basilica Cistern"],
                hidden_gems=[
                    "Soğukçeşme Street (Ottoman houses)",
                    "Turkish and Islamic Arts Museum",
                    "Carpet Museum (in Blue Mosque complex)",
                    "Arasta Bazaar (quieter than Grand Bazaar)"
                ],
                local_specialties=[
                    "Turkish breakfast at historic hotels",
                    "Traditional Ottoman cuisine",
                    "Turkish delight and baklava shops",
                    "Çay (tea) gardens with Bosphorus views"
                ],
                transportation_hubs=["Sultanahmet Tram Station", "Eminönü Ferry Terminal"],
                walking_areas=["Hippodrome Square", "Sultanahmet Park", "Kennedy Avenue waterfront"],
                dining_scene="Tourist-oriented restaurants, some authentic gems in side streets",
                shopping=["Arasta Bazaar", "Carpet shops", "Antique dealers", "Souvenir stores"],
                cultural_context="Sacred to both Christianity and Islam, UNESCO World Heritage area",
                local_tips=[
                    "Visit early morning or late afternoon for fewer crowds",
                    "Many restaurants are tourist traps - ask locals for recommendations",
                    "Dress modestly when visiting mosques",
                    "Free public WiFi in most squares"
                ],
                nearby_districts=["Eminönü", "Fatih", "Beyazıt"]
            ),
            'beyoğlu': DistrictProfile(
                name="Beyoğlu",
                turkish_name="Beyoğlu",
                description="Modern cultural center with European architecture and vibrant nightlife",
                character="Cosmopolitan district blending historic Pera with modern Istanbul lifestyle",
                main_attractions=["İstiklal Street", "Galata Tower", "Pera Museum", "Taksim Square"],
                hidden_gems=[
                    "Çiçek Pasajı (Flower Passage)",
                    "French Street (Fransız Sokağı)",
                    "Galatasaray Fish Market",
                    "Atlas Pasajı (vintage cinema and bars)",
                    "Nevizade Street (meyhane alley)"
                ],
                local_specialties=[
                    "Meyhane culture (Turkish taverns)",
                    "International cuisine",
                    "Craft cocktail bars",
                    "Street food (wet hamburger, kokoreç)"
                ],
                transportation_hubs=["Taksim Metro Station", "Karaköy Metro", "Tünel historic funicular"],
                walking_areas=["İstiklal Street", "Galata neighborhood", "Cihangir slopes"],
                dining_scene="Diverse, from street food to fine dining, strong meyhane tradition",
                shopping=["İstiklal Street shops", "Galata antique stores", "Independent boutiques"],
                cultural_context="Historic Pera district, center of Ottoman-era European community",
                local_tips=[
                    "İstiklal Street best walked in evening",
                    "Try meyhane (tavern) culture in Nevizade",
                    "Tünel is world's second-oldest underground railway",
                    "Many art galleries in Galata area"
                ],
                nearby_districts=["Galata", "Karaköy", "Taksim", "Cihangir"]
            ),
            'kadıköy': DistrictProfile(
                name="Kadıköy",
                turkish_name="Kadıköy",
                description="Hip Asian-side district known for alternative culture and food scene",
                character="Bohemian, local, and authentic - Istanbul's cultural alternative heart",
                main_attractions=["Moda waterfront", "Kadıköy Market", "Barış Manço House"],
                hidden_gems=[
                    "Moda Park and seaside promenade",
                    "Yeldeğirmeni street art district",
                    "Kriton Curi vintage market",
                    "Özgürlük Park",
                    "Historical train station"
                ],
                local_specialties=[
                    "Third-wave coffee culture",
                    "Craft beer bars",
                    "Experimental restaurants",
                    "Fresh produce markets"
                ],
                transportation_hubs=["Kadıköy Ferry Terminal", "Kadıköy Metro Station"],
                walking_areas=["Moda coast", "Bahariye Street", "Market area"],
                dining_scene="Authentic and innovative, away from tourist prices",
                shopping=["Tuesday market", "Independent bookstores", "Vintage shops"],
                cultural_context="Traditionally Greek area, now center of Istanbul's alternative scene",
                local_tips=[
                    "Take ferry from European side for scenic approach",
                    "Best district for authentic local dining",
                    "Great for coffee culture and craft beer",
                    "Less touristy, more affordable"
                ],
                nearby_districts=["Moda", "Fenerbahçe", "Üsküdar"]
            ),
            'galata': DistrictProfile(
                name="Galata",
                turkish_name="Galata",
                description="Historic port district with medieval tower and modern art scene",
                character="Trendy area mixing medieval history with contemporary culture",
                main_attractions=["Galata Tower", "Galata Bridge", "Ottoman Bank Museum"],
                hidden_gems=[
                    "Galata Mevlevi Lodge (Whirling Dervish museum)",
                    "SALT Galata (contemporary art)",
                    "Kamondo Steps",
                    "Galata House rooftop bar"
                ],
                local_specialties=["Art galleries", "Design studios", "Boutique hotels", "Rooftop bars"],
                transportation_hubs=["Karaköy Metro", "Galata Bridge ferry stops"],
                walking_areas=["Around Galata Tower", "Karaköy waterfront", "Art gallery district"],
                dining_scene="Mix of traditional and contemporary, many rooftop restaurants",
                shopping=["Art galleries", "Design boutiques", "Antique shops"],
                cultural_context="Historic Genoese quarter, now trendy arts district",
                local_tips=[
                    "Climb narrow streets for best tower views",
                    "Many galleries offer free exhibitions",
                    "Great area for architectural photography",
                    "Connect to Beyoğlu via historic Tünel"
                ],
                nearby_districts=["Karaköy", "Beyoğlu", "Eminönü"]
            ),
            'üsküdar': DistrictProfile(
                name="Üsküdar",
                turkish_name="Üsküdar",
                description="Conservative Asian-side district with Ottoman mosques and Maiden's Tower",
                character="Traditional, religious, authentic Istanbul experience",
                main_attractions=["Maiden's Tower", "Mihrimah Sultan Mosque", "Çamlıca Hill"],
                hidden_gems=[
                    "Şemsi Pasha Mosque (waterfront)",
                    "Yeni Valide Mosque complex",
                    "Çinili Mosque (tiled mosque)",
                    "Historic Üsküdar ferry terminal"
                ],
                local_specialties=["Traditional Turkish breakfast", "Ottoman sweets", "Prayer bead shops"],
                transportation_hubs=["Üsküdar Ferry Terminal", "Metro connections"],
                walking_areas=["Waterfront promenade", "Mosque courtyards", "Traditional bazaar"],
                dining_scene="Traditional and conservative, excellent Turkish breakfast spots",
                shopping=["Religious items", "Traditional crafts", "Local markets"],
                cultural_context="Historic Asian gateway to Istanbul, deeply religious character",
                local_tips=[
                    "Dress conservatively",
                    "Great views back to European side",
                    "Authentic Turkish breakfast culture",
                    "Less touristy, more traditional"
                ],
                nearby_districts=["Kadıköy", "Beylerbeyi", "Çengelköy"]
            ),
            'balat': DistrictProfile(
                name="Balat",
                turkish_name="Balat",
                description="Historic multicultural neighborhood with colorful houses and Jewish heritage",
                character="Bohemian, Instagram-famous, multicultural historic quarter",
                main_attractions=["Colorful houses", "Ahrida Synagogue", "Fener Greek Patriarchate"],
                hidden_gems=[
                    "Balat Sahil Park",
                    "Historic cisterns",
                    "Traditional coffee houses",
                    "Antique markets"
                ],
                local_specialties=["Turkish coffee", "Traditional sweets", "Antique hunting"],
                transportation_hubs=["Ferry connections", "Bus routes from Eminönü"],
                walking_areas=["Colorful house streets", "Golden Horn waterfront"],
                dining_scene="Traditional Turkish, some trendy cafes emerging",
                shopping=["Antiques", "Vintage items", "Local crafts"],
                cultural_context="Historic Jewish and Greek quarter, UNESCO recognition pending",
                local_tips=[
                    "Early morning best for photography",
                    "Respect local residents when taking photos",
                    "Great for walking and discovering",
                    "Combine with Fener for full experience"
                ],
                nearby_districts=["Fener", "Ayvansaray", "Fatih"]
            )
        }
    
    def _load_turkish_phrases(self) -> Dict[str, Dict[str, str]]:
        """Load useful Turkish phrases and cultural terms"""
        return {
            'basic_phrases': {
                'hello': 'Merhaba',
                'thank_you': 'Teşekkür ederim',
                'please': 'Lütfen',
                'excuse_me': 'Özür dilerim',
                'where_is': 'Nerede?',
                'how_much': 'Ne kadar?',
                'yes': 'Evet',
                'no': 'Hayır'
            },
            'travel_terms': {
                'tram': 'tramvay',
                'metro': 'metro',
                'ferry': 'vapur',
                'bus': 'otobüs',
                'taxi': 'taksi',
                'airport': 'havaalanı',
                'hotel': 'otel',
                'restaurant': 'restoran'
            },
            'cultural_terms': {
                'mosque': 'cami',
                'palace': 'saray',
                'market': 'çarşı',
                'bridge': 'köprü',
                'tower': 'kule',
                'street': 'sokak',
                'square': 'meydan',
                'district': 'semt'
            },
            'food_terms': {
                'breakfast': 'kahvaltı',
                'tea': 'çay',
                'coffee': 'kahve',
                'water': 'su',
                'bread': 'ekmek',
                'meat': 'et',
                'fish': 'balık',
                'vegetarian': 'vejetaryen'
            }
        }
    
    def _load_cultural_context(self) -> Dict[str, str]:
        """Load cultural context and etiquette information"""
        return {
            'mosque_etiquette': "Remove shoes, dress modestly, women cover hair, no photography during prayer",
            'greeting_culture': "Handshakes common, close friends kiss cheeks, respect for elders important",
            'dining_etiquette': "Wait for host to start, bread sacred (don't waste), tea culture central to socializing",
            'bazaar_culture': "Bargaining expected, relationship-building important, accept offered tea",
            'religious_considerations': "Respect prayer times, Friday prayers especially important, Ramadan affects schedules",
            'tipping_culture': "10-15% in restaurants, round up for taxis, small tips for services appreciated"
        }
    
    def _load_practical_info(self) -> Dict[str, Any]:
        """Load practical information for visitors"""
        return {
            'transportation': {
                'istanbulkart': "Rechargeable card for all public transport, available at stations",
                'tram_lines': "T1: Airport-Zeytinburnu-Eminönü-Beyazıt-Sultanahmet",
                'metro_lines': "M2: Yenikapi-Şişli-Levent, M1: Airport-Zeytinburnu",
                'ferry_routes': "Eminönü-Kadıköy, Karaköy-Üsküdar, Bosphorus tours",
                'taxi_tips': "Use BiTaksi or Uber apps, meter should be on, short rides common"
            },
            'opening_hours': {
                'museums': "Usually 09:00-17:00, closed Mondays typically",
                'mosques': "Open except during prayer times, Friday mornings restricted",
                'bazaars': "09:00-19:00, closed Sundays usually"
            },
            'pricing_levels': {
                'budget': "Street food, public transport, local markets",
                'moderate': "Mid-range restaurants, attractions, taxis",
                'upscale': "Fine dining, luxury hotels, private tours"
            }
        }
    
    def _load_free_attractions(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive free attractions database - addresses 21.1/100 score issue"""
        return {
            'completely_free': {
                'description': 'Attractions with absolutely no cost',
                'attractions': [
                    {
                        'name': 'Blue Mosque (Sultan Ahmed Mosque)',
                        'turkish_name': 'Sultan Ahmet Camii',
                        'location': 'Sultanahmet',
                        'hours': 'Daily except during prayer times (5 times daily)',
                        'description': 'Stunning 17th-century mosque with blue Iznik tiles',
                        'transportation': ['Sultanahmet Tram T1', 'Eminönü Ferry + 8min walk'],
                        'tips': ['Dress modestly', 'Free shoe bags provided', 'Tourist entrance separate']
                    },
                    {
                        'name': 'Galata Bridge Walk',
                        'turkish_name': 'Galata Köprüsü Yürüyüşü',
                        'location': 'Eminönü-Karaköy',
                        'hours': '24/7 open access',
                        'description': 'Historic bridge with Golden Horn views and fishermen',
                        'transportation': ['Eminönü Tram T1', 'Karaköy Metro M2'],
                        'tips': ['Best at sunset', 'Watch fishermen', 'Fish restaurants below bridge']
                    },
                    {
                        'name': 'İstiklal Avenue Stroll',
                        'turkish_name': 'İstiklal Caddesi Gezisi',
                        'location': 'Beyoğlu',
                        'hours': '24/7 pedestrian access',
                        'description': '1.4km pedestrian street with historic architecture',
                        'transportation': ['Taksim Metro M2', 'Tünel Historic Funicular'],
                        'tips': ['Historic red tram still runs', 'Street performers', 'Free window shopping']
                    },
                    {
                        'name': 'Gülhane Park',
                        'turkish_name': 'Gülhane Parkı',
                        'location': 'Sultanahmet',
                        'hours': '24/7 open',
                        'description': 'Historic Ottoman palace gardens with tulips',
                        'transportation': ['Gülhane Metro M2 direct', 'Sultanahmet Tram T1'],
                        'tips': ['Spring tulip festival', 'Great for picnics', 'Bosphorus views']
                    },
                    {
                        'name': 'Süleymaniye Mosque',
                        'turkish_name': 'Süleymaniye Camii',
                        'location': 'Eminönü',
                        'hours': 'Outside prayer times',
                        'description': 'Sinan\'s masterpiece with panoramic city views',
                        'transportation': ['Eminönü Ferry + 10min walk', 'Beyazıt Tram T1'],
                        'tips': ['Less crowded than Blue Mosque', 'Cemetery has Bosphorus views', 'Architectural masterpiece']
                    },
                    {
                        'name': 'Golden Horn Waterfront Walk',
                        'turkish_name': 'Haliç Sahil Yürüyüşü',
                        'location': 'Eminönü to Eyüp',
                        'hours': '24/7 access',
                        'description': 'Scenic waterfront promenade along historic inlet',
                        'transportation': ['Multiple ferry stops', 'Coastal bus routes'],
                        'tips': ['Great for jogging', 'Multiple coffee stops', 'Historic landmarks en route']
                    }
                ]
            },
            'minimal_cost_experiences': {
                'description': 'Very low-cost activities under 20 TL',
                'attractions': [
                    {
                        'name': 'Bosphorus Public Ferry',
                        'turkish_name': 'Boğaz Vapur',
                        'cost': '15-25 TL with İstanbulkart',
                        'duration': '90 minutes full route',
                        'description': 'Public ferry tour between Europe and Asia',
                        'tips': ['Cheaper than tourist cruises', 'Same Bosphorus views', 'Local experience']
                    },
                    {
                        'name': 'Grand Bazaar Browse',
                        'turkish_name': 'Kapalıçarşı Gezinti',
                        'cost': 'Free entry, tea 10-15 TL',
                        'duration': '2-3 hours',
                        'description': 'Historic covered market exploration',
                        'tips': ['Free to walk around', 'Accept tea offers', 'No obligation to buy']
                    },
                    {
                        'name': 'Balat Colorful Houses Walk',
                        'turkish_name': 'Balat Renkli Evler Turu',
                        'cost': 'Free walking, coffee 20-30 TL',
                        'duration': '2-3 hours',
                        'description': 'Instagram-famous colorful neighborhood',
                        'tips': ['Early morning best light', 'Respect residents', 'Combine with Fener']
                    }
                ]
            },
            'free_viewpoints': {
                'description': 'Free panoramic viewpoints across Istanbul',
                'locations': [
                    {
                        'name': 'Galata Bridge Upper Level',
                        'views': 'Golden Horn, Historic Peninsula',
                        'cost': 'Free',
                        'best_time': 'Sunset'
                    },
                    {
                        'name': 'Eminönü Waterfront',
                        'views': 'Galata Tower, Golden Horn',
                        'cost': 'Free',
                        'best_time': 'Any time'
                    },
                    {
                        'name': 'Süleymaniye Mosque Courtyard',
                        'views': 'Golden Horn, Bosphorus',
                        'cost': 'Free',
                        'best_time': 'Late afternoon'
                    }
                ]
            }
        }

    def _load_alternative_culture_venues(self) -> Dict[str, Dict[str, Any]]:
        """Alternative culture venues database - addresses 21.8/100 score issue"""
        return {
            'kadikoy_alternative': {
                'description': 'Asian-side bohemian culture hub',
                'venues': [
                    {
                        'name': 'Karga Bar',
                        'type': 'Underground music venue',
                        'address': 'Kadıköy, near Moda',
                        'specialty': 'Live indie rock, alternative music',
                        'hours': '20:00-02:00 (closed Mon-Tue)',
                        'vibe': 'Grungy, authentic, local crowd',
                        'price_range': 'Budget-friendly drinks'
                    },
                    {
                        'name': 'Arkaoda',
                        'type': 'Independent theater and performance space',
                        'address': 'Moda Caddesi, Kadıköy',
                        'specialty': 'Experimental theater, poetry nights',
                        'hours': 'Event-based schedule',
                        'vibe': 'Intellectual, artistic community',
                        'price_range': 'Affordable ticket prices'
                    },
                    {
                        'name': 'Yeldeğirmeni Street Art District',
                        'type': 'Open-air street art gallery',
                        'address': 'Yeldeğirmeni neighborhood, Kadıköy',
                        'specialty': 'Murals, graffiti, local artists',
                        'hours': '24/7 street access',
                        'vibe': 'Creative, evolving, Instagram-worthy',
                        'price_range': 'Free exploration'
                    },
                    {
                        'name': 'Kriton Curi',
                        'type': 'Vintage market and cultural space',
                        'address': 'Moda, Kadıköy',
                        'specialty': 'Vintage items, local designers, events',
                        'hours': 'Tue-Sun 11:00-19:00',
                        'vibe': 'Hipster, curated, trendy',
                        'price_range': 'Mid-range vintage prices'
                    }
                ]
            },
            'galata_arts_scene': {
                'description': 'Historic district with contemporary art',
                'venues': [
                    {
                        'name': 'SALT Galata',
                        'type': 'Contemporary art and research center',
                        'address': 'Bankalar Caddesi, Galata',
                        'specialty': 'Cutting-edge exhibitions, research library',
                        'hours': 'Tue-Sun 10:00-20:00',
                        'vibe': 'Intellectual, modern, international',
                        'price_range': 'Free exhibitions'
                    },
                    {
                        'name': 'Galata Mevlevi Lodge',
                        'type': 'Whirling Dervish museum and performance',
                        'address': 'Galip Dede Caddesi, Galata',
                        'specialty': 'Sufi culture, traditional ceremonies',
                        'hours': 'Wed-Mon 09:00-17:00',
                        'vibe': 'Spiritual, traditional, peaceful',
                        'price_range': 'Museum entry fee'
                    },
                    {
                        'name': 'Kamondo Steps',
                        'type': 'Historic Art Nouveau staircase',
                        'address': 'Bankalar Caddesi, Galata',
                        'specialty': 'Architecture photography, city views',
                        'hours': '24/7 access',
                        'vibe': 'Romantic, historic, photogenic',
                        'price_range': 'Free access'
                    }
                ]
            },
            'cihangir_bohemian': {
                'description': 'Bohemian neighborhood with cafes and culture',
                'venues': [
                    {
                        'name': 'Smyrna Café',
                        'type': 'Bohemian coffee house',
                        'address': 'Cihangir neighborhood',
                        'specialty': 'Literary atmosphere, local intellectuals',
                        'hours': '08:00-24:00',
                        'vibe': 'Intellectual, cozy, book-lined',
                        'price_range': 'Reasonable coffee prices'
                    },
                    {
                        'name': '5. Kat',
                        'type': 'Rooftop bar with city views',
                        'address': 'Cihangir, near Taksim',
                        'specialty': 'Panoramic Istanbul views, cocktails',
                        'hours': '18:00-02:00',
                        'vibe': 'Trendy, romantic, city views',
                        'price_range': 'Mid-range cocktail prices'
                    }
                ]
            },
            'beyoglu_underground': {
                'description': 'Underground culture in historic Pera',
                'venues': [
                    {
                        'name': 'Nevizade Street',
                        'type': 'Traditional meyhane (tavern) alley',
                        'address': 'Off İstiklal Avenue, Beyoğlu',
                        'specialty': 'Traditional Turkish tavern culture',
                        'hours': '18:00-02:00',
                        'vibe': 'Authentic, musical, social',
                        'price_range': 'Traditional meyhane prices'
                    },
                    {
                        'name': 'Atlas Pasajı',
                        'type': 'Historic passage with vintage cinema',
                        'address': 'İstiklal Avenue, Beyoğlu',
                        'specialty': 'Art house cinema, vintage bars',
                        'hours': 'Various venue hours',
                        'vibe': 'Nostalgic, cinematic, cultural',
                        'price_range': 'Cinema tickets + bar prices'
                    }
                ]
            }
        }

    def _load_enhanced_transportation(self) -> Dict[str, Dict[str, Any]]:
        """Enhanced transportation system with detailed practical information"""
        return {
            'metro_lines': {
                'M1A': {
                    'route': 'Yenikapı - Atatürk Airport',
                    'key_stops': ['Zeytinburnu (T1 connection)', 'Bakırköy', 'Atatürk Airport'],
                    'operation': '06:00-24:00',
                    'frequency': '3-8 minutes',
                    'journey_time': '45 minutes end-to-end'
                },
                'M1B': {
                    'route': 'Yenikapı - Kirazlı',
                    'key_stops': ['Zeytinburnu (T1 connection)', 'Esenler'],
                    'operation': '06:00-24:00',
                    'frequency': '3-8 minutes',
                    'journey_time': '30 minutes end-to-end'
                },
                'M2': {
                    'route': 'Yenikapı - Hacıosman',
                    'key_stops': ['Vezneciler (Sultanahmet area)', 'Şişhane (Galata)', 'Taksim', 'Levent'],
                    'operation': '06:00-24:00',
                    'frequency': '2-5 minutes',
                    'journey_time': '35 minutes end-to-end'
                },
                'M3': {
                    'route': 'Kirazlı - Başakşehir',
                    'key_stops': ['Metrokent', 'İkitelli'],
                    'operation': '06:00-24:00',
                    'frequency': '5-10 minutes',
                    'journey_time': '25 minutes end-to-end'
                }
            },
            'tram_lines': {
                'T1': {
                    'route': 'Bağcılar - Kabataş',
                    'key_stops': ['Sultanahmet', 'Eminönü', 'Karaköy', 'Kabataş'],
                    'operation': '06:00-24:00',
                    'frequency': '5-10 minutes',
                    'journey_time': '55 minutes end-to-end',
                    'tourist_relevance': 'Main tourist route - all major attractions'
                },
                'T4': {
                    'route': 'Topkapı - Mescid-i Selam',
                    'key_stops': ['Edirnekapı', 'Sultançiftliği'],
                    'operation': '06:00-24:00',
                    'frequency': '8-12 minutes',
                    'journey_time': '22 minutes end-to-end'
                }
            },
            'ferry_routes': {
                'bosphorus_tours': {
                    'short_tour': {
                        'route': 'Eminönü - Anadolu Kavağı',
                        'duration': '90 minutes each way',
                        'frequency': 'Every 2 hours',
                        'price': '25-40 TL with İstanbulkart',
                        'highlights': 'Bosphorus palaces, fortresses, European and Asian shores'
                    },
                    'full_tour': {
                        'route': 'Eminönü - Anadolu Kavağı (with stops)',
                        'duration': '6 hours total',
                        'frequency': '2-3 times daily',
                        'price': '35-50 TL',
                        'highlights': 'All Bosphorus attractions, lunch break in fishing village'
                    }
                },
                'city_ferries': {
                    'golden_horn': {
                        'routes': ['Eminönü - Eyüp', 'Karaköy - Sütlüce'],
                        'frequency': '20-30 minutes',
                        'price': '7-15 TL',
                        'scenic_value': 'Historic Golden Horn views'
                    },
                    'cross_bosphorus': {
                        'routes': ['Eminönü - Üsküdar', 'Karaköy - Üsküdar', 'Beşiktaş - Üsküdar'],
                        'frequency': '15-20 minutes',
                        'price': '7-15 TL',
                        'practical_use': 'Europe to Asia crossing'
                    }
                }
            },
            'istanbulkart_info': {
                'purchase_locations': ['Metro stations', 'Ferry terminals', 'Bus stops', 'Some kiosks'],
                'card_cost': '13 TL (6 TL refundable)',
                'load_amounts': 'Minimum 5 TL, maximum 300 TL',
                'discounts': '50% student discount with valid ID',
                'validity': 'All public transport (metro, tram, bus, ferry)',
                'tips': ['Always validate when boarding', 'Can be shared for multiple people', 'Check balance at machines']
            }
        }

    def _load_detailed_practical_info(self) -> Dict[str, Dict[str, Any]]:
        """Detailed practical information for all major attractions"""
        return {
            'opening_hours_comprehensive': {
                'major_museums': {
                    'standard_hours': 'Tuesday-Sunday 09:00-17:00 (winter), 09:00-19:00 (summer)',
                    'closed_days': 'Most museums closed Mondays',
                    'exceptions': {
                        'Topkapi Palace': 'Closed Tuesdays',
                        'Dolmabahçe Palace': 'Closed Monday-Tuesday',
                        'Hagia Sophia': 'Open daily (mosque schedule)'
                    },
                    'ramadan_changes': 'Reduced hours during Ramadan month',
                    'last_entry': 'Usually 1 hour before closing'
                },
                'mosques': {
                    'general_access': 'Open daily except during prayer times',
                    'prayer_times': 'Five daily prayers - check local schedule',
                    'friday_restrictions': 'Limited access Friday 11:30-14:30',
                    'tourist_hours': 'Best access: 09:00-11:30, 14:30-16:30, 17:30-sunset',
                    'closed_periods': '30 minutes before and during each prayer'
                },
                'markets_bazaars': {
                    'grand_bazaar': 'Monday-Saturday 09:00-19:00, Closed Sundays',
                    'spice_bazaar': 'Daily 08:00-19:30',
                    'local_markets': 'Vary by district, usually Tuesday and Friday',
                    'saturday_markets': 'Weekend markets in Kadıköy, Beşiktaş'
                }
            },
            'entrance_fees_detailed': {
                'museums': {
                    'topkapi_palace': '₺100 (main palace), ₺70 (Harem separate)',
                    'dolmabahce_palace': '₺90 (Selamlık), ₺60 (Harem)',
                    'basilica_cistern': '₺30',
                    'galata_tower': '₺100',
                    'chora_church': '₺45'
                },
                'discounts': {
                    'students': '50% discount with valid international student ID',
                    'seniors': '50% discount for 65+ with ID',
                    'children': 'Under 12 free at most attractions',
                    'museum_pass': '5-day pass ₺325, covers 12+ major sites'
                },
                'free_attractions': {
                    'mosques': 'All mosques free (Blue Mosque, Süleymaniye, etc.)',
                    'parks': 'Gülhane Park, Emirgan Park, all public parks',
                    'bridges_streets': 'Galata Bridge, İstiklal Avenue, waterfront walks',
                    'markets': 'Grand Bazaar and Spice Bazaar browsing free'
                }
            },
            'transportation_to_attractions': {
                'sultanahmet_area': {
                    'tram': 'T1 Sultanahmet Station (Blue Mosque, Hagia Sophia)',
                    'metro': 'M2 Vezneciler Station + 10min walk',
                    'ferry': 'Eminönü Terminal + 15min walk',
                    'taxi': '₺25-40 from Taksim, ₺40-60 from airports'
                },
                'galata_beyoglu': {
                    'metro': 'M2 Şişhane or Taksim stations',
                    'tram': 'T1 Karaköy Station + uphill walk to Galata Tower',
                    'funicular': 'Historic Tünel from Karaköy to Galata',
                    'taxi': '₺20-35 from Sultanahmet'
                },
                'kadikoy_asian_side': {
                    'ferry': 'From Eminönü or Karaköy (20-30min, scenic)',
                    'metro': 'M4 Kadıköy-Kartal line',
                    'marmaray': 'Undersea rail tunnel from European side',
                    'bus': 'Various routes from Taksim, Beşiktaş'
                }
            }
        }
    
    def get_attraction_info(self, attraction_key: str) -> Optional[AttractionInfo]:
        """Get detailed attraction information"""
        return self.attractions.get(attraction_key)
    
    def get_district_profile(self, district_key: str) -> Optional[DistrictProfile]:
        """Get comprehensive district profile"""
        return self.districts.get(district_key)
    
    def get_cultural_context(self, context_type: str) -> Optional[str]:
        """Get cultural context information"""
        return self.cultural_context.get(context_type)
    
    def get_practical_info(self, info_type: str) -> Optional[Any]:
        """Get practical information"""
        return self.practical_info.get(info_type)
    
    def search_attractions_by_district(self, district: str) -> List[AttractionInfo]:
        """Find all attractions in a specific district"""
        return [attr for attr in self.attractions.values() if attr.district == district]
    
    def get_nearby_attractions(self, attraction_key: str) -> List[str]:
        """Get nearby attractions for the given attraction"""
        attraction = self.attractions.get(attraction_key)
        return attraction.nearby_attractions if attraction else []
    
    def get_turkish_translation(self, category: str, term: str) -> Optional[str]:
        """Get Turkish translation for English terms"""
        category_terms = self.turkish_phrases.get(category, {})
        return category_terms.get(term)
    
    def get_attractions_by_category(self, category: str) -> List[AttractionInfo]:
        """Get attractions filtered by category"""
        return [attr for attr in self.attractions.values() if attr.category == category]
    
    def get_family_friendly_attractions(self) -> List[AttractionInfo]:
        """Get attractions suitable for families with children"""
        family_categories = ['family_attraction', 'park', 'museum']
        family_attractions = []
        
        for attraction in self.attractions.values():
            if (attraction.category in family_categories or 
                any(tip.lower() in ['great for kids', 'perfect for children', 'family-friendly', 'wheelchair accessible'] 
                    for tip in attraction.practical_tips)):
                family_attractions.append(attraction)
        
        return family_attractions
    
    def get_romantic_attractions(self) -> List[AttractionInfo]:
        """Get attractions perfect for romantic experiences"""
        romantic_categories = ['romantic_spot', 'romantic_experience', 'viewpoint']
        romantic_attractions = []
        
        for attraction in self.attractions.values():
            if (attraction.category in romantic_categories or 
                attraction.best_time == "Sunset for spectacular views" or
                attraction.best_time == "Sunset (golden hour)" or
                any(keyword in attraction.description.lower() 
                    for keyword in ['romantic', 'sunset', 'panoramic', 'bosphorus view', 'atmospheric'])):
                romantic_attractions.append(attraction)
        
        return romantic_attractions
    
    def get_budget_friendly_attractions(self) -> List[AttractionInfo]:
        """Get budget-friendly and free attractions"""
        budget_attractions = []
        
        for attraction in self.attractions.values():
            if (attraction.entrance_fee in ['Free', 'Budget', 'Budget (cable car fee)'] or
                attraction.category in ['park', 'local_experience'] or
                'free' in attraction.entrance_fee.lower()):
                budget_attractions.append(attraction)
        
        return budget_attractions
    
    def get_hidden_gems(self) -> List[AttractionInfo]:
        """Get lesser-known attractions and hidden gems"""
        hidden_gems = []
        
        for attraction in self.attractions.values():
            if (any(keyword in attraction.description.lower() 
                    for keyword in ['hidden gem', 'less crowded', 'off the beaten path', 'authentic', 'local']) or
                any(tip.lower() in ['less touristy', 'authentic local experience', 'off the beaten path']
                    for tip in attraction.practical_tips)):
                hidden_gems.append(attraction)
        
        return hidden_gems
    
    def get_attractions_by_audience(self, audience: str) -> Dict[str, List[AttractionInfo]]:
        """Get attractions categorized by specific audience needs"""
        audience_map = {
            'family': {
                'main_attractions': self.get_family_friendly_attractions(),
                'tips': [
                    "Look for attractions with interactive exhibits",
                    "Plan shorter visits (1-2 hours max) for young children",
                    "Bring snacks and water",
                    "Check if strollers are allowed"
                ]
            },
            'romantic': {
                'main_attractions': self.get_romantic_attractions(),
                'tips': [
                    "Plan sunset timing for best romantic atmosphere",
                    "Book dinner cruises in advance",
                    "Bring a jacket for evening activities",
                    "Consider private tours for intimate experience"
                ]
            },
            'budget': {
                'main_attractions': self.get_budget_friendly_attractions(),
                'tips': [
                    "Many mosques are free but dress modestly",
                    "Parks and markets offer authentic experiences",
                    "Use public transport with İstanbulkart",
                    "Street food is delicious and affordable"
                ]
            },
            'cultural': {
                'main_attractions': [attr for attr in self.attractions.values() 
                                   if attr.category in ['museum', 'historic_monument', 'religious_site']],
                'tips': [
                    "Hire local guides for deeper cultural insight",
                    "Respect photography rules in religious sites",
                    "Learn basic Turkish greetings",
                    "Visit during less crowded times"
                ]
            },
            'adventure': {
                'main_attractions': [attr for attr in self.attractions.values() 
                                   if 'cable car' in attr.description or 'climb' in attr.description.lower()],
                'tips': [
                    "Wear comfortable walking shoes",
                    "Bring water and snacks",
                    "Check weather conditions",
                    "Plan extra time for exploration"
                ]
            }
        }
        
        return audience_map.get(audience, {})
    
    def get_personalized_recommendations(self, preferences: Dict[str, Any]) -> List[AttractionInfo]:
        """Get personalized attraction recommendations based on user preferences"""
        recommendations = []
        
        # Filter by audience type
        audience = preferences.get('audience', 'general')
        if audience != 'general':
            audience_data = self.get_attractions_by_audience(audience)
            if audience_data and 'main_attractions' in audience_data:
                recommendations.extend(audience_data['main_attractions'][:5])
        
        # Filter by interests
        interests = preferences.get('interests', [])
        for interest in interests:
            if interest == 'history':
                recommendations.extend([attr for attr in self.attractions.values() 
                                     if attr.category in ['historic_monument', 'museum']][:3])
            elif interest == 'culture':
                recommendations.extend([attr for attr in self.attractions.values() 
                                     if attr.category == 'religious_site'][:2])
            elif interest == 'nature':
                recommendations.extend([attr for attr in self.attractions.values() 
                                     if attr.category in ['park', 'viewpoint']][:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for attr in recommendations:
            if attr.name not in seen:
                seen.add(attr.name)
                unique_recommendations.append(attr)
        
        return unique_recommendations[:8]  # Return top 8 personalized recommendations
