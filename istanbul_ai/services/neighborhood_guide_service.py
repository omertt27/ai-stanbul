"""
Istanbul Neighborhood Guide Service

Provides comprehensive, district-specific recommendations with local insights,
character descriptions, and practical tips for Istanbul neighborhoods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import logging
from datetime import datetime, time
import re

logger = logging.getLogger(__name__)

class DistrictType(Enum):
    """Types of Istanbul districts"""
    OLD_CITY = "old_city"
    MODERN_BUSINESS = "modern_business"
    TRENDY_CULTURAL = "trendy_cultural"
    TRADITIONAL_RESIDENTIAL = "traditional_residential"
    COASTAL = "coastal"
    HISTORICAL = "historical"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"
    LOCAL_AUTHENTIC = "local_authentic"

class TimeOfDay(Enum):
    """Time periods for recommendations"""
    MORNING = "morning"
    AFTERNOON = "afternoon"  
    EVENING = "evening"
    NIGHT = "night"

@dataclass
class NeighborhoodCharacter:
    """Character and atmosphere description of a neighborhood"""
    vibe: str
    crowd: str
    atmosphere: str
    best_time: List[TimeOfDay]
    local_saying: Optional[str] = None
    insider_tip: Optional[str] = None

@dataclass  
class Recommendation:
    """Individual recommendation within a neighborhood"""
    name: str
    type: str  # restaurant, cafe, attraction, shop, etc.
    description: str
    address: str
    opening_hours: Optional[str] = None
    insider_tip: Optional[str] = None
    local_favorite: bool = False
    tourist_friendly: bool = True
    coordinates: Optional[Tuple[float, float]] = None

@dataclass
class DistrictGuide:
    """Comprehensive guide for a specific district/neighborhood"""
    name: str
    district_type: DistrictType
    character: NeighborhoodCharacter
    getting_there: Dict[str, str]  # transport methods and details
    recommendations: Dict[str, List[Recommendation]]  # categorized recommendations
    walking_routes: List[str]
    local_customs: List[str]
    practical_tips: List[str]
    avoid_when: List[str]
    budget_estimate: str
    safety_notes: str


class NeighborhoodGuideService:
    """
    Comprehensive Istanbul Neighborhood Guide Service
    
    Provides detailed, district-specific recommendations with local insights,
    character descriptions, and practical tips for Istanbul neighborhoods.
    """
    
    def __init__(self):
        """Initialize the service with comprehensive district data"""
        self.districts = self._initialize_district_data()
        self.keywords_map = self._create_keyword_mapping()
        logger.info(f"Initialized NeighborhoodGuideService with {len(self.districts)} districts")
    
    def _initialize_district_data(self) -> Dict[str, DistrictGuide]:
        """Initialize comprehensive data for all major Istanbul districts"""
        districts = {}
        
        # Beşiktaş - Modern Business & Nightlife
        districts["besiktas"] = DistrictGuide(
            name="Beşiktaş",
            district_type=DistrictType.MODERN_BUSINESS,
            character=NeighborhoodCharacter(
                vibe="Dynamic blend of business, sports culture, and vibrant nightlife",
                crowd="Young professionals, university students, football fans, and party-goers",
                atmosphere="Energetic during day, electric at night, especially on match days",
                best_time=[TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT],
                local_saying="'Beşiktaş'ta her gece festival!' (Every night is a festival in Beşiktaş!)",
                insider_tip="Visit on a Beşiktaş match day to experience the true spirit of the neighborhood"
            ),
            getting_there={
                "metro": "M6 Levent-Haliç line to Levent, then bus or taxi",
                "bus": "Multiple lines from Taksim, Kadıköy, and other major hubs",
                "ferry": "Ferry from Kadıköy, Üsküdar, or Eminönü to Beşiktaş pier",
                "car": "Limited parking, better to use public transport"
            },
            recommendations={
                "restaurants": [
                    Recommendation(
                        name="Pandeli",
                        type="Ottoman cuisine",
                        description="Historic restaurant serving traditional Ottoman dishes since 1901",
                        address="Eminönü Mısır Çarşısı No:1",
                        opening_hours="12:00-15:00, 19:00-24:00",
                        insider_tip="Try the lamb tandir, it's been their signature for over a century",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Çiya Sofrası",
                        type="Anatolian cuisine",
                        description="Authentic regional Turkish dishes from all over Anatolia",
                        address="Güneşlibahçe Sk. No:43, Kadıköy",
                        opening_hours="11:00-22:00",
                        insider_tip="Don't miss the seasonal vegetables and regional specialties",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "cafes": [
                    Recommendation(
                        name="Starbucks Barbaros",
                        type="International coffee chain",
                        description="Spacious Starbucks with outdoor seating and city views",
                        address="Barbaros Blv. No:145, Beşiktaş",
                        opening_hours="07:00-24:00",
                        insider_tip="Great spot for working with laptop, reliable WiFi",
                        local_favorite=False,
                        tourist_friendly=True
                    )
                ],
                "nightlife": [
                    Recommendation(
                        name="Blackk",
                        type="Night
                        description="Upscale nightclub with international DJs and sophisticated crowd",
                        address="Salhane Sk. No:1, Ortaköy",
                        opening_hours="23:00-06:00 (Thu-Sat)",
                        insider_tip="Dress code enforced, reservations recommended",
                        local_favorite=False,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Çırağan Palace Bar",
                        type="Luxury hotel bar",
                        description="Elegant bar with Bosphorus views in historic palace setting",
                        address="Çırağan Cd. No:32, Beşiktaş",
                        opening_hours="18:00-02:00",
                        insider_tip="Perfect for sunset cocktails with palace atmosphere",
                        local_favorite=False,
                        tourist_friendly=True
                    )
                ],
                "attractions": [
                    Recommendation(
                        name="Dolmabahçe Palace",
                        type="Historical palace",
                        description="19th-century Ottoman palace showcasing European architectural influence",
                        address="Dolmabahçe Cd., Beşiktaş",
                        opening_hours="09:00-16:00 (Tue-Sun)",
                        insider_tip="Visit early morning to avoid crowds, photography restrictions inside",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Beşiktaş Fish Market",
                        type="Local market",
                        description="Traditional fish market with fresh seafood and local atmosphere",
                        address="Beşiktaş Balık Pazarı, Beşiktaş",
                        opening_hours="06:00-20:00",
                        insider_tip="Best selection in early morning, try street food vendors nearby",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ]
            },
            walking_routes=[
                "Dolmabahçe Palace → Beşiktaş Square → Fish Market → Barbaros Boulevard",
                "Ortaköy → Çırağan Palace → Dolmabahçe → Beşiktaş Pier",
                "Beşiktaş Stadium → Akaretler → Dolmabahçe Gardens"
            ],
            local_customs=[
                "Join the pre-match excitement at Beşiktaş Square on game days",
                "Respect the passionate football culture - avoid wearing rival team colors",
                "Try fresh fish sandwiches from street vendors"
            ],
            practical_tips=[
                "Use ferry for scenic arrival and departure",
                "Parking is limited, public transport recommended",
                "Match days create heavy traffic and crowds",
                "Nightlife peaks Thursday-Saturday"
            ],
            avoid_when=[
                "Major football match days if you want quiet atmosphere",
                "Rush hours (8-9 AM, 6-8 PM) for heavy traffic"
            ],
            budget_estimate="$50-100 per day for dining and activities",
            safety_notes="Generally safe, be aware of pickpockets in crowded areas and match days"
        )
        
        # Şişli - Modern Business & Shopping
        districts["sisli"] = DistrictGuide(
            name="Şişli",
            district_type=DistrictType.MODERN_BUSINESS,
            character=NeighborhoodCharacter(
                vibe="Upscale business district with luxury shopping and fine dining",
                crowd="Business professionals, wealthy locals, shoppers, and expats",
                atmosphere="Sophisticated and cosmopolitan, busiest during business hours",
                best_time=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON, TimeOfDay.EVENING],
                local_saying="'Şişli'de para konuşur' (Money talks in Şişli)",
                insider_tip="Visit during weekdays to see the true business atmosphere"
            ),
            getting_there={
                "metro": "M2 line to Şişli-Mecidiyeköy or Gayrettepe stations",
                "bus": "Extensive bus network from all parts of the city",
                "taxi": "Easy to find, but expect traffic during rush hours",
                "car": "Modern parking facilities available but expensive"
            },
            recommendations={
                "restaurants": [
                    Recommendation(
                        name="Nusr-Et Steakhouse",
                        type="Premium steakhouse",
                        description="World-famous steakhouse by Salt Bae with theatrical service",
                        address="Nispetiye Cd. No:87, Levent",
                        opening_hours="12:00-02:00",
                        insider_tip="Expensive but Instagram-worthy experience, book ahead",
                        local_favorite=False,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Seasons Restaurant",
                        type="International fine dining",
                        description="Elegant restaurant in Four Seasons with seasonal menu",
                        address="Teşvikiye Mh., Nispetiye Cd. No:20",
                        opening_hours="18:30-24:00",
                        insider_tip="Perfect for business dinners, dress code required",
                        local_favorite=False,
                        tourist_friendly=True
                    )
                ],
                "shopping": [
                    Recommendation(
                        name="Cevahir Mall",
                        type="Shopping mall",
                        description="One of Europe's largest shopping malls with 300+ stores",
                        address="Büyükdere Cd. No:22, Şişli",
                        opening_hours="10:00-22:00",
                        insider_tip="Visit the top floor for panoramic city views",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Nişantaşı Shopping District",
                        type="Luxury shopping area",
                        description="High-end boutiques and international luxury brands",
                        address="Nişantaşı, Şişli",
                        opening_hours="10:00-20:00",
                        insider_tip="Best window shopping in Istanbul, café breaks at Mudo City",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "cafes": [
                    Recommendation(
                        name="Mudo City Café",
                        type="Trendy café",
                        description="Stylish café in Nişantaşı perfect for people watching",
                        address="Teşvikiye Mh., Abdi İpekçi Cd.",
                        opening_hours="08:00-24:00",
                        insider_tip="Prime location for seeing Istanbul's fashionable crowd",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ]
            },
            walking_routes=[
                "Nişantaşı → Teşvikiye Mosque → Abdi İpekçi Street → City's Mall",
                "Şişli Metro → Cevahir Mall → Harbiye → Military Museum"
            ],
            local_customs=[
                "Dress well - this is Istanbul's fashion district",
                "Business lunch culture is strong here",
                "Tipping expected in upscale venues"
            ],
            practical_tips=[
                "Credit cards widely accepted",
                "Valet parking available at most venues",
                "Business district - quieter on weekends",
                "Many venues have dress codes"
            ],
            avoid_when=[
                "Rush hours for heavy traffic",
                "Sundays when many shops are closed"
            ],
            budget_estimate="$80-150 per day for shopping and dining",
            safety_notes="Very safe, well-patrolled area with good lighting"
        )
        
        # Üsküdar - Traditional Residential & Historical
        districts["uskudar"] = DistrictGuide(
            name="Üsküdar",
            district_type=DistrictType.TRADITIONAL_RESIDENTIAL,
            character=NeighborhoodCharacter(
                vibe="Traditional, religious, and family-oriented with stunning Bosphorus views",
                crowd="Conservative locals, families, and pilgrims visiting religious sites",
                atmosphere="Peaceful and spiritual, especially around mosques and waterfront",
                best_time=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON, TimeOfDay.EVENING],
                local_saying="'Üsküdar'da huzur var' (There's peace in Üsküdar)",
                insider_tip="Best sunset views of the European side from the waterfront"
            ),
            getting_there={
                "ferry": "Most scenic route from Eminönü, Karaköy, or Beşiktaş",
                "metro": "M5 line to Üsküdar station",
                "bus": "Multiple lines from Asian and European sides",
                "marmaray": "Cross-continental rail line connects to European side"
            },
            recommendations={
                "attractions": [
                    Recommendation(
                        name="Mihrimah Sultan Mosque",
                        type="Historical mosque",
                        description="16th-century mosque by Mimar Sinan with beautiful architecture",
                        address="İskele Mh., Üsküdar",
                        opening_hours="Always open (prayer times respected)",
                        insider_tip="Visit during non-prayer times, dress modestly",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Maiden's Tower",
                        type="Historic landmark",
                        description="Iconic tower on small islet with restaurant and city views",
                        address="Salacak Mh., Üsküdar",
                        opening_hours="09:00-19:00",
                        insider_tip="Take boat from Üsküdar pier, sunset dinner reservations essential",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Çamlıca Hill",
                        type="Scenic viewpoint",
                        description="Highest hill in Istanbul with panoramic city views",
                        address="Çamlıca Tepesi, Üsküdar",
                        opening_hours="24/7",
                        insider_tip="Best views at sunset, bring a picnic and arrive early",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "restaurants": [
                    Recommendation(
                        name="İskenderpaşa Konağı",
                        type="Traditional Ottoman",
                        description="Historic mansion serving authentic Ottoman cuisine",
                        address="İcadiye Mh., Üsküdar",
                        opening_hours="12:00-24:00",
                        insider_tip="Try the historical recipes, beautiful garden seating",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "cafes": [
                    Recommendation(
                        name="Çinili Köşk Café",
                        type="Traditional tea garden",
                        description="Historic tea garden with Bosphorus views and local atmosphere",
                        address="Salacak Mh., Üsküdar",
                        opening_hours="08:00-24:00",
                        insider_tip="Perfect for Turkish tea and backgammon, very local feel",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ]
            },
            walking_routes=[
                "Üsküdar Pier → Mihrimah Sultan Mosque → Şemsi Paşa Mosque → Maiden's Tower viewpoint",
                "Çamlıca Hill → Büyük Çamlıca Mosque → TV Tower → Kısıklı"
            ],
            local_customs=[
                "Dress modestly, especially around mosques",
                "Remove shoes when entering mosques",
                "Respect prayer times and religious observances",
                "Try traditional Turkish tea culture"
            ],
            practical_tips=[
                "Ferry is the most scenic transport option",
                "Conservative dress recommended",
                "Many venues close during prayer times",
                "Great for photography enthusiasts"
            ],
            avoid_when=[
                "Friday prayers (12:00-14:00) around mosques",
                "Very early morning or late night for women traveling alone"
            ],
            budget_estimate="$30-60 per day for modest dining and activities",
            safety_notes="Very safe, traditional neighborhood with strong community feel"
        )
        
        # Kadıköy - Trendy Cultural Hub
        districts["kadikoy"] = DistrictGuide(
            name="Kadıköy",
            district_type=DistrictType.TRENDY_CULTURAL,
            character=NeighborhoodCharacter(
                vibe="Bohemian, artistic, and vibrant with strong alternative culture",
                crowd="Artists, students, young professionals, and creative types",
                atmosphere="Energetic and diverse, with strong arts and music scene",
                best_time=[TimeOfDay.AFTERNOON, TimeOfDay.EVENING, TimeOfDay.NIGHT],
                local_saying="'Kadıköy'de her köşe sanat' (Every corner is art in Kadıköy)",
                insider_tip="Thursday-Sunday evenings are when the cultural scene really comes alive"
            ),
            getting_there={
                "ferry": "Scenic ferry ride from Eminönü or Karaköy",
                "metro": "M4 line to Kadıköy-Müteferrika",
                "bus": "Extensive bus network from both sides of the city",
                "marmaray": "Quick connection from European side"
            },
            recommendations={
                "restaurants": [
                    Recommendation(
                        name="Çiya Sofrası",
                        type="Anatolian cuisine",
                        description="Famous for authentic regional dishes from all over Turkey",
                        address="Güneşlibahçe Sk. No:43, Kadıköy",
                        opening_hours="11:00-22:00",
                        insider_tip="Try different regional specialties, changes daily menu",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Kırmızı Balık",
                        type="Seafood",
                        description="Fresh seafood with modern presentation and local atmosphere",
                        address="Muvakkithane Cd. No:22, Kadıköy",
                        opening_hours="12:00-24:00",
                        insider_tip="Daily catch is always the best choice",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "nightlife": [
                    Recommendation(
                        name="Arkaoda",
                        type="Alternative bar/club",
                        description="Underground venue with indie music and alternative crowd",
                        address="Tellalzade Sk. No:3, Kadıköy",
                        opening_hours="20:00-04:00",
                        insider_tip="Check their events calendar for live music and DJ sets",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Karga Bar",
                        type="Rock bar",
                        description="Legendary rock bar with live music and authentic atmosphere",
                        address="Kadife Sk. No:16, Kadıköy",
                        opening_hours="18:00-04:00",
                        insider_tip="Classic Istanbul rock scene, cash only",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "cultural": [
                    Recommendation(
                        name="Kadıköy Market",
                        type="Traditional market",
                        description="Vibrant local market with fresh produce and street food",
                        address="Güneşlibahçe Mh., Kadıköy",
                        opening_hours="08:00-19:00",
                        insider_tip="Tuesday and Friday are the busiest days with best selection",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Moda Street",
                        type="Bohemian quarter",
                        description="Trendy street with vintage shops, cafés, and galleries",
                        address="Moda Cd., Kadıköy",
                        opening_hours="10:00-22:00",
                        insider_tip="Perfect for vintage shopping and people watching",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ]
            },
            walking_routes=[
                "Kadıköy Pier → Fish Market → Çiya Restaurant → Moda Street",
                "Moda Park → Moda Coast → Fenerbahçe → Bağdat Avenue",
                "Kadife Street → Barlar Sokağı → Alternative venues tour"
            ],
            local_customs=[
                "Support local artists and musicians",
                "Try street food from market vendors",
                "Embrace the alternative, laid-back culture",
                "Join outdoor drinking culture in summer"
            ],
            practical_tips=[
                "Cash preferred in many local venues",
                "Ferry provides beautiful city views",
                "Peak nightlife is Thursday-Sunday",
                "Many venues have live music schedules"
            ],
            avoid_when=[
                "Monday evenings when many venues are closed",
                "Early mornings if you want the vibrant atmosphere"
            ],
            budget_estimate="$40-80 per day for dining and nightlife",
            safety_notes="Generally safe, popular with young crowds, standard urban precautions"
        )
        
        # Fatih - Historical Heart
        districts["fatih"] = DistrictGuide(
            name="Fatih",
            district_type=DistrictType.HISTORICAL,
            character=NeighborhoodCharacter(
                vibe="Historic and deeply traditional with layers of Byzantine and Ottoman heritage",
                crowd="Religious pilgrims, history enthusiasts, conservative locals, and tourists",
                atmosphere="Spiritual and historic, bustling around major monuments",
                best_time=[TimeOfDay.MORNING, TimeOfDay.AFTERNOON],
                local_saying="'Fatih'te tarih nefes alır' (History breathes in Fatih)",
                insider_tip="Early morning visits to major sites avoid crowds and provide better photography"
            ),
            getting_there={
                "tram": "T1 tram line serves major attractions",
                "metro": "M2 line to Vezneciler-İstanbul Üniversitesi",
                "bus": "Extensive network from all parts of the city",
                "walking": "Many attractions walkable from each other"
            },
            recommendations={
                "attractions": [
                    Recommendation(
                        name="Blue Mosque (Sultan Ahmed)",
                        type="Historical mosque",
                        description="Iconic 17th-century mosque with six minarets and blue tiles",
                        address="Sultanahmet Mh., Fatih",
                        opening_hours="Outside prayer times",
                        insider_tip="Visit early morning for best photos, dress modestly",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Hagia Sophia",
                        type="Historical monument",
                        description="Former church and mosque, now museum showcasing Byzantine architecture",
                        address="Sultanahmet Mh., Fatih",
                        opening_hours="09:00-17:00",
                        insider_tip="Entry can be crowded, early morning or late afternoon best",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Grand Bazaar",
                        type="Historic market",
                        description="One of the oldest covered markets with 4,000 shops",
                        address="Beyazıt Mh., Fatih",
                        opening_hours="09:00-19:00 (Mon-Sat)",
                        insider_tip="Haggling expected, compare prices, avoid rush hours",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Topkapi Palace",
                        type="Historical palace",
                        description="Former Ottoman palace with treasury and harem sections",
                        address="Cankurtaran Mh., Fatih",
                        opening_hours="09:00-18:00 (Wed-Mon)",
                        insider_tip="Harem requires separate ticket, allow full day visit",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "restaurants": [
                    Recommendation(
                        name="Sultanahmet Fish House",
                        type="Traditional seafood",
                        description="Historic fish restaurant with Ottoman recipes",
                        address="Prof. Kazım İsmail Gürkan Cd., Sultanahmet",
                        opening_hours="11:00-24:00",
                        insider_tip="Try the Ottoman fish preparations, terrace has good views",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Hamdi Restaurant",
                        type="Traditional Turkish",
                        description="Famous for lamb dishes and traditional preparation methods",
                        address="Kalçın Sk. No:17, Eminönü",
                        opening_hours="11:00-24:00",
                        insider_tip="Try the tandır lamb, upper floors have Golden Horn views",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ]
            },
            walking_routes=[
                "Blue Mosque → Hagia Sophia → Topkapi Palace → Archaeological Museums",
                "Grand Bazaar → Süleymaniye Mosque → Spice Bazaar → Galata Bridge",
                "Sultanahmet Square → Hippodrome → Basilica Cistern → Turkish Bath"
            ],
            local_customs=[
                "Dress modestly for mosque visits",
                "Remove shoes when entering mosques",
                "Respect prayer times and religious observances",
                "Haggling is expected in bazaars"
            ],
            practical_tips=[
                "Buy museum pass for multiple attractions",
                "Avoid Fridays around mosques during prayer times",
                "Comfortable walking shoes essential",
                "Guided tours available for deeper historical context"
            ],
            avoid_when=[
                "Peak summer months for extreme crowds",
                "Friday prayers around major mosques",
                "Grand Bazaar on Sundays (closed)"
            ],
            budget_estimate="$50-90 per day including attractions and dining",
            safety_notes="Tourist police present, watch for pickpockets in crowded areas"
        )
        
        # Sultanahmet - Tourist Heart (Part of Fatih but distinct character)
        districts["sultanahmet"] = DistrictGuide(
            name="Sultanahmet",
            district_type=DistrictType.HISTORICAL,
            character=NeighborhoodCharacter(
                vibe="Historic center with concentrated Byzantine and Ottoman landmarks",
                crowd="International tourists, history enthusiasts, and local vendors",
                atmosphere="Bustling with tourists during day, magical at sunset",
                best_time=[TimeOfDay.MORNING, TimeOfDay.EVENING],
                local_saying="'Sultanahmet dünyaya açılan kapı' (Sultanahmet is the gateway to the world)",
                insider_tip="Golden hour around sunset provides the most magical atmosphere"
            ),
            getting_there={
                "tram": "T1 tram to Sultanahmet stop",
                "metro": "M2 to Vezneciler, then short walk or tram",
                "taxi": "Easy access but traffic can be heavy",
                "walking": "Walkable from Galata Bridge and Eminönü"
            },
            recommendations={
                "attractions": [
                    Recommendation(
                        name="Basilica Cistern",
                        type="Historical underground site",
                        description="6th-century underground cistern with atmospheric lighting",
                        address="Yerebatan Cd. 1/3, Sultanahmet",
                        opening_hours="09:00-18:30",
                        insider_tip="Cool retreat in summer, stunning lighting effects",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Hippodrome of Constantinople",
                        type="Historical site",
                        description="Ancient chariot racing arena with original monuments",
                        address="Sultanahmet Mh., Fatih",
                        opening_hours="24/7",
                        insider_tip="Great for understanding Byzantine city layout",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "cultural": [
                    Recommendation(
                        name="Traditional Turkish Bath (Cağaloğlu Hamamı)",
                        type="Historic bathhouse",
                        description="16th-century Turkish bath offering traditional experience",
                        address="Prof. Kazım İsmail Gürkan Cd. No:34",
                        opening_hours="06:00-24:00",
                        insider_tip="Book massage in advance, bring flip-flops",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "restaurants": [
                    Recommendation(
                        name="Seven Hills Restaurant",
                        type="International with Turkish",
                        description="Rooftop restaurant with views of major monuments",
                        address="Tevkifhane Sk. No:8/A, Sultanahmet",
                        opening_hours="07:00-24:00",
                        insider_tip="Reserve terrace table for sunset dinner",
                        local_favorite=False,
                        tourist_friendly=True
                    )
                ]
            },
            walking_routes=[
                "Sultanahmet Square → Blue Mosque → Hagia Sophia → Basilica Cistern",
                "Topkapi Palace → Archaeological Museums → Gülhane Park → Sirkeci"
            ],
            local_customs=[
                "Very tourist-oriented, prices higher than local areas",
                "Mosque etiquette strictly observed",
                "Street vendors common, polite refusal accepted",
                "Photography restrictions in some indoor sites"
            ],
            practical_tips=[
                "Istanbul Museum Pass saves money and time",
                "Book restaurant reservations for sunset views",
                "Comfortable shoes for cobblestone streets",
                "Audio guides enhance historical understanding"
            ],
            avoid_when=[
                "Midday in summer for extreme heat and crowds",
                "Cruise ship arrival times for peak crowding"
            ],
            budget_estimate="$60-120 per day including major attractions",
            safety_notes="Tourist police present, generally very safe, standard tourist precautions"
        )
        
        # Sarıyer - Coastal Retreat
        districts["sariyer"] = DistrictGuide(
            name="Sarıyer",
            district_type=DistrictType.COASTAL,
            character=NeighborhoodCharacter(
                vibe="Upscale coastal district with natural beauty and fresh seafood",
                crowd="Wealthy locals, weekend visitors, and seafood enthusiasts",
                atmosphere="Relaxed and scenic, especially along the Bosphorus coast",
                best_time=[TimeOfDay.AFTERNOON, TimeOfDay.EVENING],
                local_saying="'Sarıyer'de deniz, doğa ve lezzet' (Sea, nature and flavor in Sarıyer)",
                insider_tip="Weekend fish markets and seafood restaurants are the main attractions"
            ),
            getting_there={
                "bus": "Bus lines from Taksim, Beşiktaş, and other major centers",
                "car": "Scenic coastal drive, parking available",
                "ferry": "Weekend ferry services from city center",
                "dolmuş": "Shared minibus from metro stations"
            },
            recommendations={
                "restaurants": [
                    Recommendation(
                        name="Körfez Restaurant",
                        type="Seafood",
                        description="Waterfront seafood restaurant with fresh daily catch",
                        address="Sarıyer İskele Cd., Sarıyer",
                        opening_hours="12:00-24:00",
                        insider_tip="Try the grilled fish with seasonal vegetables",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Uskumru Restaurant",
                        type="Fish speciality",
                        description="Famous for mackerel and other Bosphorus fish",
                        address="Rumeli Kavağı, Sarıyer",
                        opening_hours="11:00-22:00",
                        insider_tip="Best mackerel sandwiches in Istanbul",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ],
                "attractions": [
                    Recommendation(
                        name="Rumeli Fortress",
                        type="Historical fortress",
                        description="15th-century fortress with Bosphorus views",
                        address="Yahya Kemal Cd. No:42, Sarıyer",
                        opening_hours="09:00-16:30 (Thu-Tue)",
                        insider_tip="Climb to top for panoramic Bosphorus views",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Emirgan Park",
                        type="Public park",
                        description="Large park famous for tulips and scenic picnic areas",
                        address="Emirgan Mh., Sarıyer",
                        opening_hours="24/7",
                        insider_tip="Visit during tulip season (April-May) for spectacular displays",
                        local_favorite=True,
                        tourist_friendly=True
                    ),
                    Recommendation(
                        name="Sadberk Hanım Museum",
                        type="Private museum",
                        description="Ottoman and Turkish Islamic art in waterfront mansion",
                        address="Büyükdere Cd. No:27-29, Sarıyer",
                        opening_hours="10:00-17:00 (Thu-Tue)",
                        insider_tip="Beautiful mansion setting with garden overlooking Bosphorus",
                        local_favorite=True,
                        tourist_friendly=True
                    )
                ]
            },
            walking_routes=[
                "Sarıyer Fish Market → Waterfront restaurants → Rumeli Fortress",
                "Emirgan Park → Bosphorus coast walk → Büyükdere",
                "Sadberk Hanım Museum → Yeniköy → Tarabya coastal walk"
            ],
            local_customs=[
                "Weekend fish market is social gathering place",
                "Respect local fishing spots and fishermen",
                "Tipping appreciated in seafood restaurants",
                "Seasonal visits align with fish seasons"
            ],
            practical_tips=[
                "Best visited on weekends for full atmosphere",
                "Car recommended for exploring multiple coastal spots",
                "Fresh fish prices vary by season",
                "Beautiful photography opportunities along coast"
            ],
            avoid_when=[
                "Weekdays when many seasonal restaurants are closed",
                "Winter months for limited outdoor dining"
            ],
            budget_estimate="$50-100 per day for seafood dining and activities",
            safety_notes="Very safe, upscale area with good infrastructure"
        )
        
        return districts
    
    def _create_keyword_mapping(self) -> Dict[str, List[str]]:
        """Create keyword mapping for district identification"""
        return {
            "besiktas": ["besiktas", "beşiktaş", "ortakoy", "ortaköy", "dolmabahce", "dolmabahçe", "barbaros", "akaretler", "football", "nightlife", "black eagle"],
            "sisli": ["sisli", "şişli", "nisantasi", "nişantaşı", "mecidiyekoy", "mecidiyeköy", "gayrettepe", "levent", "cevahir", "shopping", "business", "luxury"],
            "uskudar": ["uskudar", "üsküdar", "camlica", "çamlıca", "maiden tower", "kiz kulesi", "kız kulesi", "mihrimah", "traditional", "asian side", "conservative"],
            "kadikoy": ["kadikoy", "kadıköy", "moda", "fenerbahce", "fenerbahçe", "ciya", "çiya", "alternative", "bohemian", "cultural", "artistic", "bars", "nightlife"],
            "fatih": ["fatih", "sultanahmet", "eminonu", "eminönü", "grand bazaar", "kapalıçarşı", "süleymaniye", "historical", "ottoman", "byzantine", "mosque"],
            "sultanahmet": ["sultanahmet", "blue mosque", "hagia sophia", "ayasofya", "topkapi", "topkapı", "basilica cistern", "yerebatan", "hippodrome", "tourist", "historical"],
            "sariyer": ["sariyer", "sarıyer", "rumeli", "emirgan", "tarabya", "yeniköy", "büyükdere", "coastal", "bosphorus", "seafood", "fish", "nature"]
        }
    
    def get_district_guide(self, district_name: str) -> Optional[DistrictGuide]:
        """Get complete guide for a specific district"""
        district_key = district_name.lower().replace(" ", "").replace("ş", "s").replace("ç", "c").replace("ğ", "g").replace("ü", "u").replace("ö", "o").replace("ı", "i")
        return self.districts.get(district_key)
    
    def find_districts_by_keywords(self, query: str) -> List[str]:
        """Find districts matching keywords in the query"""
        query_lower = query.lower()
        matched_districts = []
        
        for district, keywords in self.keywords_map.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_districts.append(district)
        
        return matched_districts
    
    def get_recommendations_by_category(self, district_name: str, category: str) -> List[Recommendation]:
        """Get recommendations for a specific category in a district"""
        district = self.get_district_guide(district_name)
        if district and category in district.recommendations:
            return district.recommendations[category]
        return []
    
    def search_recommendations(self, query: str, district_name: Optional[str] = None) -> List[Tuple[str, Recommendation]]:
        """Search for recommendations across districts or within a specific district"""
        results = []
        query_lower = query.lower()
        
        districts_to_search = [district_name] if district_name else list(self.districts.keys())
        
        for district_key in districts_to_search:
            district = self.districts.get(district_key)
            if not district:
                continue
                
            for category, recommendations in district.recommendations.items():
                for rec in recommendations:
                    if (query_lower in rec.name.lower() or 
                        query_lower in rec.type.lower() or 
                        query_lower in rec.description.lower() or
                        (rec.insider_tip and query_lower in rec.insider_tip.lower())):
                        results.append((district.name, rec))
        
        return results
    
    def get_neighborhood_character(self, district_name: str) -> Optional[NeighborhoodCharacter]:
        """Get character description for a district"""
        district = self.get_district_guide(district_name)
        return district.character if district else None
    
    def get_practical_info(self, district_name: str) -> Dict[str, Any]:
        """Get practical information for visiting a district"""
        district = self.get_district_guide(district_name)
        if not district:
            return {}
            
        return {
            "getting_there": district.getting_there,
            "walking_routes": district.walking_routes,
            "local_customs": district.local_customs,
            "practical_tips": district.practical_tips,
            "avoid_when": district.avoid_when,
            "budget_estimate": district.budget_estimate,
            "safety_notes": district.safety_notes
        }
