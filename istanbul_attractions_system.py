#!/usr/bin/env python3
"""
Istanbul Attractions Deep Learning System
Advanced AI system for 78+ curated Istanbul attractions with multi-intent support
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttractionCategory(Enum):
    """Categories for Istanbul attractions"""
    HISTORICAL_MONUMENT = "historical_monument"
    MUSEUM = "museum"
    RELIGIOUS_SITE = "religious_site"
    PARK_GARDEN = "park_garden"
    VIEWPOINT = "viewpoint"
    NEIGHBORHOOD = "neighborhood"
    SHOPPING = "shopping"
    ENTERTAINMENT = "entertainment"
    FAMILY_ATTRACTION = "family_attraction"
    ROMANTIC_SPOT = "romantic_spot"
    NIGHTLIFE = "nightlife"
    CULTURAL_CENTER = "cultural_center"
    MARKET_BAZAAR = "market_bazar"
    MARKET_SHOPPING = "market_shopping"  # Added for compatibility
    WATERFRONT = "waterfront"
    HIDDEN_GEM = "hidden_gem"
    PALACE_MANSION = "palace_mansion"
    WELLNESS_SPA = "wellness_spa"
    THEME_PARK = "theme_park"
    BEACH_RECREATION = "beach_recreation"
    ISLAND_NATURE = "island_nature"
    NATURE_RECREATION = "nature_recreation"
    SHOPPING_DISTRICT = "shopping_district"

class WeatherPreference(Enum):
    """Weather preferences for attractions"""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    COVERED = "covered"
    ALL_WEATHER = "all_weather"

class BudgetCategory(Enum):
    """Budget categories for attractions"""
    FREE = "free"
    BUDGET = "budget"        # 0-50 TL
    MODERATE = "moderate"    # 50-150 TL
    EXPENSIVE = "expensive"  # 150+ TL

@dataclass
class AttractionData:
    """Comprehensive attraction data structure"""
    id: str
    name: str
    turkish_name: str
    district: str
    category: AttractionCategory
    description: str
    
    # Practical information
    opening_hours: Dict[str, str]
    entrance_fee: BudgetCategory
    estimated_cost: str
    duration: str
    
    # Location and access
    transportation: List[str]
    coordinates: Optional[Tuple[float, float]] = None
    address: Optional[str] = None
    
    # Experience details
    best_time: str = ""
    weather_preference: WeatherPreference = WeatherPreference.ALL_WEATHER
    nearby_attractions: List[str] = field(default_factory=list)
    
    # Categorization
    is_family_friendly: bool = True
    is_romantic: bool = False
    is_hidden_gem: bool = False
    difficulty_level: str = "easy"  # easy, moderate, challenging
    
    # Cultural information
    cultural_significance: str = ""
    practical_tips: List[str] = field(default_factory=list)
    
    # AI enhancement fields
    keywords: List[str] = field(default_factory=list)
    sentiment_tags: List[str] = field(default_factory=list)
    seasonal_info: Dict[str, str] = field(default_factory=dict)

class IstanbulAttractionsSystem:
    """Advanced attractions recommendation system with deep learning integration"""
    
    def __init__(self):
        """Initialize the attractions system"""
        self.attractions = self._load_comprehensive_attractions()
        self.districts = self._load_district_data()
        self.categories = list(AttractionCategory)
        logger.info(f"🏛️ Loaded {len(self.attractions)} attractions across {len(self.districts)} districts")
    
    def _load_comprehensive_attractions(self) -> Dict[str, AttractionData]:
        """Load comprehensive database of 78+ Istanbul attractions"""
        attractions = {}
        
        # HISTORICAL MONUMENTS (15 attractions)
        attractions.update({
            'hagia_sophia': AttractionData(
                id='hagia_sophia',
                name="Hagia Sophia",
                turkish_name="Ayasofya",
                district="Sultanahmet",
                category=AttractionCategory.HISTORICAL_MONUMENT,
                description="Former Byzantine cathedral and Ottoman mosque, now a mosque again. A symbol of Istanbul's rich history.",
                opening_hours={"daily": "09:00-18:00 (except prayer times)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (donations appreciated)",
                duration="1-2 hours",
                transportation=["Sultanahmet Tram", "Eminönü Ferry", "Gülhane Metro"],
                best_time="Early morning or late afternoon",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Blue Mosque", "Topkapi Palace", "Basilica Cistern"],
                is_family_friendly=True,
                is_romantic=True,
                cultural_significance="UNESCO World Heritage Site, architectural marvel spanning Byzantine and Ottoman periods",
                practical_tips=[
                    "Dress modestly with head covering for women",
                    "Remove shoes before entering",
                    "Photography allowed in most areas",
                    "Respect prayer times and worshippers"
                ],
                keywords=["byzantine", "ottoman", "mosque", "cathedral", "unesco", "history", "architecture"],
                sentiment_tags=["awe-inspiring", "spiritual", "historic", "iconic"],
                coordinates=(41.0086, 28.9802)
            ),
            
            'blue_mosque': AttractionData(
                id='blue_mosque',
                name="Blue Mosque",
                turkish_name="Sultanahmet Camii",
                district="Sultanahmet",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="17th-century mosque famous for its blue İznik tiles and six minarets.",
                opening_hours={"daily": "08:30-12:00, 14:00-16:30, 17:30-18:30 (outside prayer times)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="45 minutes - 1 hour",
                transportation=["Sultanahmet Tram", "Eminönü Ferry"],
                best_time="Early morning for fewer crowds",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Hagia Sophia", "Hippodrome", "Grand Bazaar"],
                is_family_friendly=True,
                is_romantic=True,
                cultural_significance="Active mosque, masterpiece of Ottoman architecture with unique six minarets",
                practical_tips=[
                    "Mandatory modest dress and head covering",
                    "Free shoe bags provided",
                    "No entrance during prayer times",
                    "Tourist entrance separate from worshippers"
                ],
                keywords=["mosque", "ottoman", "blue", "tiles", "minarets", "islamic", "architecture"],
                sentiment_tags=["serene", "beautiful", "spiritual", "architectural marvel"],
                coordinates=(41.0054, 28.9768)
            ),
            
            'topkapi_palace': AttractionData(
                id='topkapi_palace',
                name="Topkapi Palace",
                turkish_name="Topkapı Sarayı",
                district="Sultanahmet",
                category=AttractionCategory.MUSEUM,
                description="Former Ottoman imperial palace with treasury and harem sections.",
                opening_hours={"tuesday": "Closed", "other_days": "09:00-18:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="100-150 TL (extra for Harem)",
                duration="2-4 hours",
                transportation=["Sultanahmet Tram", "Gülhane Metro"],
                best_time="Early morning to avoid crowds",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Hagia Sophia", "Archaeological Museum", "Gülhane Park"],
                is_family_friendly=True,
                is_romantic=False,
                cultural_significance="Center of Ottoman Empire for 400 years, incredible treasury and imperial artifacts",
                practical_tips=[
                    "Buy tickets online to skip lines",
                    "Harem requires separate ticket",
                    "Audio guide highly recommended",
                    "Comfortable shoes essential for extensive walking"
                ],
                keywords=["palace", "ottoman", "sultan", "treasury", "harem", "imperial", "museum"],
                sentiment_tags=["majestic", "historical", "luxurious", "educational"],
                coordinates=(41.0115, 28.9833)
            ),
            
            'galata_tower': AttractionData(
                id='galata_tower',
                name="Galata Tower",
                turkish_name="Galata Kulesi",
                district="Galata",
                category=AttractionCategory.VIEWPOINT,
                description="Medieval tower offering panoramic 360-degree city views.",
                opening_hours={"daily": "08:30-22:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="120 TL",
                duration="1-2 hours",
                transportation=["Karaköy Metro", "Galata Bridge walk", "Tünel funicular"],
                best_time="Sunset for spectacular views",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Galata Bridge", "İstiklal Street", "Pera Museum"],
                is_family_friendly=True,
                is_romantic=True,
                cultural_significance="Byzantine-era watchtower, symbol of Galata district and Istanbul skyline",
                practical_tips=[
                    "Book timed entry tickets online",
                    "Elevator available but expect queues",
                    "Restaurant at top level with city views",
                    "360-degree observation deck perfect for photos"
                ],
                keywords=["tower", "view", "panoramic", "medieval", "galata", "observation", "skyline"],
                sentiment_tags=["breathtaking", "romantic", "panoramic", "iconic"],
                coordinates=(41.0256, 28.9744)
            ),
                        
            'basilica_cistern': AttractionData(
                id='basilica_cistern',
                name="Basilica Cistern",
                turkish_name="Yerebatan Sarnıcı",
                district="Sultanahmet",
                category=AttractionCategory.HISTORICAL_MONUMENT,
                description="Ancient underground cistern with mystical atmosphere and iconic Medusa columns.",
                opening_hours={"daily": "09:00-18:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="90 TL",
                duration="45 minutes",
                transportation=["Sultanahmet Tram"],
                best_time="Any time - underground",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Hagia Sophia", "Blue Mosque"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Byzantine engineering marvel with mystical atmosphere and famous Medusa head columns",
                practical_tips=[
                    "Cool temperature year-round",
                    "Photography allowed",
                    "Wheelchair accessible",
                    "Mystical lighting creates unique atmosphere"
                ],
                keywords=["cistern", "underground", "byzantine", "medusa", "columns", "mystical", "engineering"],
                sentiment_tags=["mysterious", "atmospheric", "ancient", "cool"],
                coordinates=(41.0084, 28.9778)
            ),
            
            'dolmabahce_palace': AttractionData(
                id='dolmabahce_palace',
                name="Dolmabahçe Palace",
                turkish_name="Dolmabahçe Sarayı",
                district="Beşiktaş",
                category=AttractionCategory.MUSEUM,
                description="Opulent 19th-century Ottoman palace on the Bosphorus with crystal staircase.",
                opening_hours={"monday_tuesday": "Closed", "other_days": "09:00-16:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="120 TL (separate sections)",
                duration="2-3 hours",
                transportation=["Kabataş Ferry", "Dolmabahçe Tram"],
                best_time="Morning for better lighting",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Beşiktaş Square", "Yıldız Park", "Naval Museum"],
                is_family_friendly=True,
                is_romantic=True,
                cultural_significance="Last residence of Ottoman sultans, where Atatürk died, European-influenced architecture",
                practical_tips=[
                    "Guided tours mandatory",
                    "No photography inside",
                    "World's largest chandelier",
                    "Crystal staircase is highlight"
                ],
                keywords=["palace", "ottoman", "bosphorus", "crystal", "chandelier", "ataturk", "luxury"],
                sentiment_tags=["opulent", "luxurious", "historic", "elegant"],
                coordinates=(41.0391, 28.9998)
            )
        })
        
        # HIDDEN GEMS & LESSER-KNOWN SITES (20 attractions)
        attractions.update({
            'chora_church': AttractionData(
                id='chora_church',
                name="Chora Church",
                turkish_name="Kariye Müzesi",
                district="Fatih",
                category=AttractionCategory.HIDDEN_GEM,
                description="Hidden gem with world's finest Byzantine mosaics and frescoes.",
                opening_hours={"wednesday": "Closed", "other_days": "09:00-17:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="80 TL",
                duration="1-2 hours",
                transportation=["Bus from Eminönü", "Taxi recommended"],
                best_time="Morning for better lighting",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Eyüp Sultan Mosque", "Golden Horn"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="UNESCO candidate with incredible medieval Byzantine artwork and mosaics",
                practical_tips=[
                    "Off the beaten path",
                    "Bring good camera",
                    "Combine with Eyüp visit",
                    "World-class Byzantine art"
                ],
                keywords=["byzantine", "mosaics", "frescoes", "hidden", "art", "medieval", "church"],
                sentiment_tags=["hidden treasure", "artistic", "peaceful", "authentic"],
                coordinates=(41.0308, 28.9381)
            ),
            
            'pierre_loti_hill': AttractionData(
                id='pierre_loti_hill',
                name="Pierre Loti Hill",
                turkish_name="Pierre Loti Tepesi",
                district="Eyüp",
                category=AttractionCategory.VIEWPOINT,
                description="Panoramic Golden Horn views with cable car access and authentic tea gardens.",
                opening_hours={"daily": "08:00-24:00 (cable car until 23:00)"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="25 TL (cable car)",
                duration="1-2 hours",
                transportation=["Cable car from Eyüp", "Bus to Eyüp then cable car"],
                best_time="Sunset for spectacular views",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Eyüp Sultan Mosque", "Golden Horn"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Named after French writer, authentic local atmosphere with traditional tea culture",
                practical_tips=[
                    "Traditional tea gardens",
                    "Great for sunset photos",
                    "Less touristy viewpoint",
                    "Cable car ride is scenic"
                ],
                keywords=["viewpoint", "golden horn", "cable car", "tea garden", "sunset", "panoramic"],
                sentiment_tags=["peaceful", "authentic", "scenic", "romantic"],
                coordinates=(41.0458, 28.9356)
            ),
            
            'maiden_tower': AttractionData(
                id='maiden_tower',
                name="Maiden's Tower",
                turkish_name="Kız Kulesi",
                district="Üsküdar",
                category=AttractionCategory.ROMANTIC_SPOT,
                description="Iconic tower on small island with restaurant and museum.",
                opening_hours={"daily": "09:00-18:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="100 TL (includes boat)",
                duration="2-3 hours",
                transportation=["Boat from Üsküdar or Kabataş"],
                best_time="Sunset dinner for romantic experience",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Üsküdar waterfront", "Salacak shore"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Symbol of Istanbul with legendary love stories and island mystique",
                practical_tips=[
                    "Book restaurant in advance",
                    "Boat ride included in price",
                    "Perfect for proposals",
                    "Sunset views spectacular"
                ],
                keywords=["tower", "island", "romantic", "boat", "legend", "bosphorus", "sunset"],
                sentiment_tags=["romantic", "legendary", "isolated", "mystical"],
                coordinates=(41.0214, 29.0042)
            ),
            
            'balat_colorful_houses': AttractionData(
                id='balat_colorful_houses',
                name="Balat Colorful Houses",
                turkish_name="Balat Renkli Evler",
                district="Fatih",
                category=AttractionCategory.NEIGHBORHOOD,
                description="Instagram-famous colorful houses in historic Jewish quarter.",
                opening_hours={"daily": "24 hours (street viewing)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="2-3 hours walking",
                transportation=["Ferry to Golden Horn", "Metro to Vezneciler then bus"],
                best_time="Morning for best light and fewer crowds",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Fener Greek Patriarchate", "Golden Horn"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Historic multicultural neighborhood with Jewish and Greek heritage",
                practical_tips=[
                    "Respect residents when photographing",
                    "Great for Instagram",
                    "Authentic local atmosphere",
                    "Combine with coffee shop visits"
                ],
                keywords=["colorful", "houses", "neighborhood", "jewish", "instagram", "authentic", "walking"],
                sentiment_tags=["colorful", "authentic", "photogenic", "multicultural"],
                coordinates=(41.0299, 28.9493)
            ),
            
            'rumeli_fortress': AttractionData(
                id='rumeli_fortress',
                name="Rumeli Fortress",
                turkish_name="Rumeli Hisarı",
                district="Sarıyer",
                category=AttractionCategory.HISTORICAL_MONUMENT,
                description="Ottoman fortress built to control Bosphorus, with stunning waterfront views.",
                opening_hours={"tuesday_sunday": "09:00-19:00 (closed Monday)"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="30 TL",
                duration="1-2 hours",
                transportation=["Bus to Sarıyer", "Ferry to Sarıyer"],
                best_time="Late afternoon for golden hour",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Fatih Sultan Mehmet Bridge", "Emirgan Park"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Strategic Ottoman fortress controlling the Bosphorus strait",
                practical_tips=[
                    "Great for photography",
                    "Combine with Bosphorus ferry ride",
                    "Wear comfortable shoes for climbing",
                    "Best views at sunset"
                ],
                keywords=["fortress", "ottoman", "bosphorus", "history", "views", "military"],
                sentiment_tags=["historic", "scenic", "impressive", "strategic"],
                coordinates=(41.0835, 29.0565)
            ),
            
            'suleymaniye_mosque': AttractionData(
                id='suleymaniye_mosque',
                name="Süleymaniye Mosque",
                turkish_name="Süleymaniye Camii",
                district="Fatih",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="Magnificent Ottoman mosque by architect Sinan with panoramic city views.",
                opening_hours={"daily": "Prayer times permitting, best 10:00-17:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (donations welcome)",
                duration="45 minutes - 1 hour",
                transportation=["Tram to Beyazıt", "Metro to Vezneciler", "Walk from Grand Bazaar"],
                best_time="Mid-morning or late afternoon",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Grand Bazaar", "Istanbul University"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Masterpiece of Ottoman architecture by the great architect Sinan",
                practical_tips=[
                    "Dress modestly (scarves for women)",
                    "Remove shoes before entering",
                    "Avoid prayer times",
                    "Terrace offers best Golden Horn views"
                ],
                keywords=["mosque", "sinan", "ottoman", "architecture", "views", "religious"],
                sentiment_tags=["majestic", "peaceful", "architectural", "spiritual"],
                coordinates=(41.0166, 28.9642)
            ),
            
            'bosphorus_bridge': AttractionData(
                id='bosphorus_bridge',
                name="Bosphorus Bridge",
                turkish_name="Boğaziçi Köprüsü",
                district="Beşiktaş",
                category=AttractionCategory.VIEWPOINT,
                description="Iconic suspension bridge connecting Europe and Asia, best viewed from waterfront.",
                opening_hours={"daily": "24 hours (viewing areas)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free to view",
                duration="30 minutes - 1 hour",
                transportation=["Ferry for best views", "Bus to Ortaköy", "Walking along Bosphorus"],
                best_time="Sunset and evening when illuminated",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Ortaköy", "Dolmabahçe Palace"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="First bridge connecting Europe and Asia, opened 1973",
                practical_tips=[
                    "Best viewed from Ortaköy waterfront",
                    "Beautiful when lit at night",
                    "Great photo opportunities",
                    "Combine with Bosphorus ferry ride"
                ],
                keywords=["bridge", "bosphorus", "europe", "asia", "suspension", "iconic"],
                sentiment_tags=["impressive", "iconic", "scenic", "connecting"],
                coordinates=(41.0404, 29.0136)
            ),
            
            'grand_bazaar': AttractionData(
                id='grand_bazaar',
                name="Grand Bazaar",
                turkish_name="Kapalıçarşı",
                district="Fatih",
                category=AttractionCategory.MARKET_SHOPPING,
                description="One of world's oldest covered markets with 4,000 shops selling everything from carpets to spices.",
                opening_hours={"monday_saturday": "09:00-19:00 (closed Sunday)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free entry, shopping budget varies",
                duration="1-3 hours",
                transportation=["Tram to Beyazıt", "Metro to Vezneciler", "Walk from Sultanahmet"],
                best_time="Morning for less crowds, afternoon for atmosphere",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Süleymaniye Mosque", "Spice Bazaar"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Historic marketplace dating back to 1461, one of the first shopping malls",
                practical_tips=[
                    "Negotiate prices (start at 30% of asking)",
                    "Bring cash",
                    "Watch for pickpockets",
                    "Try Turkish delight samples"
                ],
                keywords=["bazaar", "shopping", "carpets", "jewelry", "spices", "historic", "covered"],
                sentiment_tags=["bustling", "authentic", "traditional", "overwhelming"],
                coordinates=(41.0108, 28.9682)
            ),
            
            'spice_bazaar': AttractionData(
                id='spice_bazaar',
                name="Spice Bazaar",
                turkish_name="Mısır Çarşısı",
                district="Fatih",
                category=AttractionCategory.MARKET_SHOPPING,
                description="Historic Egyptian Bazaar filled with aromatic spices, Turkish delight, and local delicacies.",
                opening_hours={"monday_saturday": "08:00-19:30 (closed Sunday)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free entry, shopping budget varies",
                duration="1-2 hours",
                transportation=["Tram to Eminönü", "Ferry to Eminönü", "Walk from Galata Bridge"],
                best_time="Morning for freshest products",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Galata Bridge", "New Mosque", "Golden Horn"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Ottoman-era spice market, center of spice trade for centuries",
                practical_tips=[
                    "Sample before buying",
                    "Compare prices between stalls",
                    "Buy saffron and Turkish delight",
                    "Negotiate for bulk purchases"
                ],
                keywords=["spices", "turkish delight", "bazaar", "egyptian", "market", "aromatic"],
                sentiment_tags=["aromatic", "colorful", "traditional", "sensory"],
                coordinates=(41.0166, 28.9706)
            ),
            
            'new_mosque': AttractionData(
                id='new_mosque',
                name="New Mosque",
                turkish_name="Yeni Cami",
                district="Fatih",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="Beautiful Ottoman mosque near Galata Bridge with impressive courtyard and interior.",
                opening_hours={"daily": "Prayer times permitting, best 09:00-18:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (donations welcome)",
                duration="30-45 minutes",
                transportation=["Tram to Eminönü", "Ferry to Eminönü"],
                best_time="Mid-morning or afternoon",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Spice Bazaar", "Galata Bridge"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="17th century Ottoman mosque, despite its name",
                practical_tips=[
                    "Combine with Spice Bazaar visit",
                    "Dress modestly",
                    "Beautiful courtyard for photos",
                    "Feed pigeons in the square"
                ],
                keywords=["mosque", "ottoman", "courtyard", "galata bridge", "religious"],
                sentiment_tags=["peaceful", "architectural", "traditional", "spiritual"],
                coordinates=(41.0164, 28.9706)
            ),
            
            'turkish_baths_cagaloglu': AttractionData(
                id='turkish_baths_cagaloglu',
                name="Cağaloğlu Hamam",
                turkish_name="Cağaloğlu Hamam",
                district="Fatih",
                category=AttractionCategory.WELLNESS_SPA,
                description="Historic Ottoman bathhouse offering traditional Turkish bath experience since 1741.",
                opening_hours={"daily": "06:00-24:00 (men), 08:00-20:00 (women)"},
                entrance_fee=BudgetCategory.EXPENSIVE,
                estimated_cost="200-400 TL depending on services",
                duration="1-2 hours",
                transportation=["Tram to Beyazıt", "Walk from Sultanahmet", "Metro to Vezneciler"],
                best_time="Afternoon for relaxation",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Grand Bazaar", "Hagia Sophia"],
                is_family_friendly=False,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Oldest functioning Turkish bath in Istanbul, established 1741",
                practical_tips=[
                    "Bring swimwear or rent towels",
                    "Stay hydrated",
                    "Book massage in advance",
                    "Authentic Ottoman architecture"
                ],
                keywords=["hamam", "turkish bath", "spa", "massage", "ottoman", "wellness"],
                sentiment_tags=["relaxing", "authentic", "luxurious", "historic"],
                coordinates=(41.0087, 28.9719)
            ),
            
            'fatih_mosque': AttractionData(
                id='fatih_mosque',
                name="Fatih Mosque",
                turkish_name="Fatih Camii",
                district="Fatih",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="Grand mosque built on the site of the former Church of the Holy Apostles.",
                opening_hours={"daily": "Prayer times permitting, best 09:00-17:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (donations welcome)",
                duration="45 minutes",
                transportation=["Metro to Vezneciler", "Bus to Fatih"],
                best_time="Morning or late afternoon",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Süleymaniye Mosque", "Fener-Balat"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="Built by Fatih Sultan Mehmet (Conqueror of Constantinople)",
                practical_tips=[
                    "Less touristy than other mosques",
                    "Beautiful Ottoman architecture",
                    "Large courtyard and gardens",
                    "Wednesday market nearby"
                ],
                keywords=["mosque", "fatih", "ottoman", "conqueror", "architecture"],
                sentiment_tags=["grand", "peaceful", "historic", "authentic"],
                coordinates=(41.0200, 28.9497)
            ),
            
            'princes_islands': AttractionData(
                id='princes_islands',
                name="Princes' Islands",
                turkish_name="Adalar",
                district="Adalar",
                category=AttractionCategory.ISLAND_NATURE,
                description="Chain of nine islands in Sea of Marmara, perfect for day trips with horse carriages and beaches.",
                opening_hours={"daily": "Ferry schedule dependent, usually 07:00-21:00"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="50-100 TL (ferry + activities)",
                duration="Full day (6-8 hours)",
                transportation=["Ferry from Kabataş", "Ferry from Bostancı"],
                best_time="Spring through autumn, weekdays less crowded",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Sea of Marmara", "Historic mansions"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic retreat for Byzantine royalty and Ottoman elite",
                practical_tips=[
                    "Take ferry to Büyükada (largest island)",
                    "Rent bicycles or take horse carriage",
                    "Bring swimming gear in summer",
                    "Try island seafood restaurants"
                ],
                keywords=["islands", "ferry", "bicycles", "beaches", "nature", "day trip"],
                sentiment_tags=["peaceful", "scenic", "relaxing", "natural"],
                coordinates=(40.8833, 29.1167)
            ),
            
            'rahmi_koc_museum': AttractionData(
                id='rahmi_koc_museum',
                name="Rahmi M. Koç Museum",
                turkish_name="Rahmi M. Koç Müzesi",
                district="Beyoğlu",
                category=AttractionCategory.MUSEUM,
                description="Industrial museum with vintage cars, trains, planes, and interactive exhibits.",
                opening_hours={"tuesday_friday": "10:00-17:00", "weekends": "10:00-18:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="25 TL adults, 15 TL students",
                duration="2-3 hours",
                transportation=["Metro to Haliç", "Bus to Hasköy", "Ferry to Golden Horn"],
                best_time="Weekday mornings for fewer crowds",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Golden Horn", "Miniaturk"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="Turkey's first industrial museum showcasing transportation history",
                practical_tips=[
                    "Great for kids who love vehicles",
                    "Interactive exhibits and simulators",
                    "Submarine and vintage cars to explore",
                    "Cafe with Golden Horn views"
                ],
                keywords=["museum", "industrial", "cars", "trains", "interactive", "family"],
                sentiment_tags=["educational", "interactive", "fun", "nostalgic"],
                coordinates=(41.0322, 28.9711)
            ),
            
            'miniaturk': AttractionData(
                id='miniaturk',
                name="Miniaturk",
                turkish_name="Miniaturk",
                district="Beyoğlu",
                category=AttractionCategory.THEME_PARK,
                description="Miniature park featuring scaled models of famous Turkish landmarks and monuments.",
                opening_hours={"daily": "09:00-19:00 (until 18:00 in winter)"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="25 TL adults, 15 TL children",
                duration="2-3 hours",
                transportation=["Metro to Haliç", "Bus to Sütlüce"],
                best_time="Morning or late afternoon",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Golden Horn", "Rahmi Koç Museum"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Educational park showcasing Turkey's architectural heritage",
                practical_tips=[
                    "Perfect for families with children",
                    "See all of Turkey's landmarks in one place",
                    "Interactive exhibits and activities",
                    "Restaurant and cafe on site"
                ],
                keywords=["miniatures", "models", "family", "educational", "turkey", "landmarks"],
                sentiment_tags=["fun", "educational", "family-friendly", "unique"],
                coordinates=(41.0392, 28.9503)
            ),
            
            'florya_beach': AttractionData(
                id='florya_beach',
                name="Florya Beach",
                turkish_name="Florya Plajı",
                district="Bakırköy",
                category=AttractionCategory.BEACH_RECREATION,
                description="Popular urban beach on the Marmara Sea with restaurants and recreational facilities.",
                opening_hours={"daily": "24 hours (facilities vary)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free beach access, paid facilities available",
                duration="Half day (3-4 hours)",
                transportation=["Metro to Florya", "Bus to Florya"],
                best_time="Summer months, weekdays less crowded",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Atatürk Marine Mansion", "Aqua Club Dolphin"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic summer retreat area for Istanbul residents",
                practical_tips=[
                    "Bring sun protection",
                    "Beach clubs offer amenities",
                    "Good seafood restaurants nearby",
                    "Combine with Yeşilköy area visit"
                ],
                keywords=["beach", "marmara", "swimming", "summer", "recreation", "families"],
                sentiment_tags=["relaxing", "summery", "recreational", "popular"],
                coordinates=(40.9667, 28.7833)
            ),
            
            'beylerbeyi_palace': AttractionData(
                id='beylerbeyi_palace',
                name="Beylerbeyi Palace",
                turkish_name="Beylerbeyi Sarayı",
                district="Üsküdar",
                category=AttractionCategory.PALACE_MANSION,
                description="19th century Ottoman summer palace with ornate interiors and Bosphorus gardens.",
                opening_hours={"tuesday_sunday": "09:00-17:00 (closed Monday)"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="40 TL",
                duration="1-2 hours",
                transportation=["Bus to Beylerbeyi", "Ferry to Üsküdar then bus"],
                best_time="Morning for best lighting",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Bosphorus Bridge", "Çamlıca Hill"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Summer residence of Ottoman sultans with European influences",
                practical_tips=[
                    "Less crowded than Dolmabahçe",
                    "Beautiful Bosphorus views",
                    "Ornate European-style interiors",
                    "Lovely palace gardens"
                ],
                keywords=["palace", "ottoman", "bosphorus", "summer", "gardens", "luxury"],
                sentiment_tags=["elegant", "peaceful", "ornate", "historic"],
                coordinates=(41.0417, 29.0400)
            ),
            
            'camlica_hill': AttractionData(
                id='camlica_hill',
                name="Çamlıca Hill",
                turkish_name="Çamlıca Tepesi",
                district="Üsküdar",
                category=AttractionCategory.VIEWPOINT,
                description="Highest point in Istanbul with panoramic city views and traditional tea gardens.",
                opening_hours={"daily": "24 hours (tea gardens 08:00-24:00)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (tea garden prices vary)",
                duration="1-2 hours",
                transportation=["Metro to Üsküdar then bus", "Taxi from Üsküdar"],
                best_time="Sunset for spectacular views",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Maiden's Tower", "Beylerbeyi Palace"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Traditional picnic and recreation area for Istanbul families",
                practical_tips=[
                    "Best panoramic views in Istanbul",
                    "Traditional tea gardens",
                    "Great for sunset photography",
                    "Can be crowded on weekends"
                ],
                keywords=["hill", "viewpoint", "panoramic", "tea garden", "sunset", "highest"],
                sentiment_tags=["scenic", "peaceful", "romantic", "elevated"],
                coordinates=(41.0167, 29.0667)
            ),
            
            'archaeology_museum': AttractionData(
                id='archaeology_museum',
                name="Istanbul Archaeology Museum",
                turkish_name="İstanbul Arkeoloji Müzesi",
                district="Sultanahmet",
                category=AttractionCategory.MUSEUM,
                description="World-class collection of ancient artifacts, sarcophagi, and classical antiquities.",
                opening_hours={"tuesday_sunday": "09:00-19:00 (closed Monday)"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="30 TL",
                duration="2-3 hours",
                transportation=["Tram to Gülhane", "Walk from Sultanahmet"],
                best_time="Morning for quiet exploration",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Topkapi Palace", "Gülhane Park"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="One of world's great archaeological museums with unique artifacts",
                practical_tips=[
                    "Don't miss Alexander Sarcophagus",
                    "Three museums in complex",
                    "Ancient Orient Museum included",
                    "Museum of Islamic Art nearby"
                ],
                keywords=["archaeology", "museum", "sarcophagi", "ancient", "artifacts", "classical"],
                sentiment_tags=["educational", "impressive", "historical", "cultural"],
                coordinates=(41.0117, 28.9819)
            ),
            
            'sadberk_hanim_museum': AttractionData(
                id='sadberk_hanim_museum',
                name="Sadberk Hanım Museum",
                turkish_name="Sadberk Hanım Müzesi",
                district="Sarıyer",
                category=AttractionCategory.MUSEUM,
                description="Private museum in Ottoman mansion showcasing Turkish and Islamic arts.",
                opening_hours={"thursday_tuesday": "10:00-17:00 (closed Wednesday)"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="25 TL",
                duration="1-2 hours",
                transportation=["Bus to Büyükdere", "Ferry to Sarıyer"],
                best_time="Afternoon for peaceful visit",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Rumeli Fortress", "Emirgan Park"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="First private museum in Turkey, housed in historic waterfront mansion",
                practical_tips=[
                    "Beautiful Bosphorus mansion setting",
                    "Excellent Islamic art collection",
                    "Less crowded than major museums",
                    "Combine with Bosphorus ferry trip"
                ],
                keywords=["museum", "private", "islamic art", "mansion", "bosphorus", "collection"],
                sentiment_tags=["intimate", "cultural", "elegant", "peaceful"],
                coordinates=(41.1086, 29.0564)
            ),
            
            'gulhane_park': AttractionData(
                id='gulhane_park',
                name="Gülhane Park",
                turkish_name="Gülhane Parkı",
                district="Sultanahmet",
                category=AttractionCategory.PARK_GARDEN,
                description="Historic park near Topkapi Palace with beautiful gardens and city views.",
                opening_hours={"daily": "07:00-22:30"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="1-2 hours",
                transportation=["Tram to Gülhane", "Walk from Sultanahmet"],
                best_time="Spring for tulips, evening for cooler weather",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Topkapi Palace", "Archaeology Museum"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Former outer gardens of Topkapi Palace, now public park",
                practical_tips=[
                    "Famous for tulip displays in spring",
                    "Great for picnics",
                    "Historical plane trees",
                    "Tea garden with Bosphorus views"
                ],
                keywords=["park", "gardens", "tulips", "topkapi", "historic", "picnic"],
                sentiment_tags=["peaceful", "green", "historic", "romantic"],
                coordinates=(41.0133, 28.9817)
            ),
            
            'bebek_neighborhood': AttractionData(
                id='bebek_neighborhood',
                name="Bebek Neighborhood",
                turkish_name="Bebek Mahallesi",
                district="Beşiktaş",
                category=AttractionCategory.NEIGHBORHOOD,
                description="Upscale Bosphorus neighborhood with trendy cafes, waterfront parks, and scenic views.",
                opening_hours={"daily": "24 hours (businesses vary)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free to walk, dining varies",
                duration="2-3 hours",
                transportation=["Bus to Bebek", "Ferry to Bebek"],
                best_time="Afternoon and evening",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Rumeli Fortress", "Emirgan Park"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic wealthy neighborhood, popular with locals and visitors",
                practical_tips=[
                    "Expensive dining and cafes",
                    "Beautiful waterfront walks",
                    "Great people watching",
                    "Combine with Bosphorus ferry ride"
                ],
                keywords=["neighborhood", "upscale", "cafes", "bosphorus", "waterfront", "trendy"],
                sentiment_tags=["elegant", "trendy", "scenic", "expensive"],
                coordinates=(41.0833, 29.0433)
            ),
            
            'ortakoy_square': AttractionData(
                id='ortakoy_square',
                name="Ortaköy Square",
                turkish_name="Ortaköy Meydanı",
                district="Beşiktaş",
                category=AttractionCategory.NEIGHBORHOOD,
                description="Picturesque waterfront square with mosque, craft market, and Bosphorus Bridge views.",
                opening_hours={"daily": "24 hours (market usually 10:00-sunset)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (shopping and dining extra)",
                duration="1-2 hours",
                transportation=["Bus to Ortaköy", "Dolmuş from Beşiktaş"],
                best_time="Sunset for bridge views",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Bosphorus Bridge", "Dolmabahçe Palace"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic multicultural neighborhood with mosque, church, and synagogue",
                practical_tips=[
                    "Famous for kumpir (stuffed baked potato)",
                    "Artisan craft stalls",
                    "Best Bosphorus Bridge photo spot",
                    "Can be very crowded weekends"
                ],
                keywords=["square", "mosque", "bosphorus bridge", "crafts", "kumpir", "waterfront"],
                sentiment_tags=["picturesque", "bustling", "romantic", "touristy"],
                coordinates=(41.0475, 29.0269)
            ),
            
            'buyukcekmece_lake': AttractionData(
                id='buyukcekmece_lake',
                name="Büyükçekmece Lake",
                turkish_name="Büyükçekmece Gölü",
                district="Büyükçekmece",
                category=AttractionCategory.NATURE_RECREATION,
                description="Large freshwater lake with parks, walking paths, and recreational activities.",
                opening_hours={"daily": "24 hours (facilities vary)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (activities extra)",
                duration="Half day (3-4 hours)",
                transportation=["Metrobus to Büyükçekmece", "Train to Büyükçekmece"],
                best_time="Spring and autumn for mild weather",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Historic Büyükçekmece Bridge", "Silivri beaches"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Important freshwater source and recreational area for western Istanbul",
                practical_tips=[
                    "Great for cycling and walking",
                    "Bird watching opportunities",
                    "Picnic areas available",
                    "Less crowded than city center"
                ],
                keywords=["lake", "nature", "cycling", "walking", "recreation", "freshwater"],
                sentiment_tags=["peaceful", "natural", "spacious", "refreshing"],
                coordinates=(41.0167, 28.5833)
            ),
            
            'maiden_tower_bosphorus': AttractionData(
                id='maiden_tower_bosphorus',
                name="Maiden's Tower (Kız Kulesi)",
                turkish_name="Kız Kulesi",
                district="Üsküdar",
                category=AttractionCategory.HISTORICAL_MONUMENT,
                description="Iconic tower on small islet in Bosphorus with restaurant and panoramic views.",
                opening_hours={"daily": "09:00-19:00 (restaurant until late)"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="40 TL + boat transfer",
                duration="2-3 hours",
                transportation=["Ferry from Üsküdar", "Ferry from Kabataş"],
                best_time="Sunset for romantic atmosphere",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Üsküdar waterfront", "Çamlıca Hill"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Ancient lighthouse and defense tower, featured in many legends",
                practical_tips=[
                    "Advance booking recommended for restaurant",
                    "Boat transfer included in ticket",
                    "360-degree Bosphorus views",
                    "Perfect for marriage proposals"
                ],
                keywords=["tower", "islet", "bosphorus", "restaurant", "views", "romantic"],
                sentiment_tags=["iconic", "romantic", "scenic", "legendary"],
                coordinates=(41.0211, 29.0044)
            ),
            
            'istanbul_modern': AttractionData(
                id='istanbul_modern',
                name="Istanbul Modern",
                turkish_name="İstanbul Modern",
                district="Beyoğlu",
                category=AttractionCategory.MUSEUM,
                description="Premier contemporary art museum showcasing Turkish and international modern art.",
                opening_hours={"tuesday_sunday": "10:00-18:00 (closed Monday)"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="60 TL",
                duration="2-3 hours",
                transportation=["Ferry to Karaköy", "Metro to Şişhane", "Walk from Galata Bridge"],
                best_time="Weekday mornings for peaceful viewing",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Galata Bridge", "Karaköy neighborhood"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Turkey's first museum dedicated to modern and contemporary art",
                practical_tips=[
                    "Excellent museum shop",
                    "Cafe with Bosphorus views",
                    "Rotating exhibitions",
                    "Audio guides available"
                ],
                keywords=["modern art", "contemporary", "museum", "turkish art", "exhibitions"],
                sentiment_tags=["artistic", "contemporary", "sophisticated", "cultural"],
                coordinates=(41.0256, 28.9744)
            ),
            
            'karakoy_neighborhood': AttractionData(
                id='karakoy_neighborhood',
                name="Karaköy Neighborhood",
                turkish_name="Karaköy Mahallesi",
                district="Beyoğlu",
                category=AttractionCategory.NEIGHBORHOOD,
                description="Trendy arts district with galleries, design shops, cafes, and historic architecture.",
                opening_hours={"daily": "24 hours (businesses vary)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free to explore, dining/shopping varies",
                duration="2-4 hours",
                transportation=["Ferry to Karaköy", "Metro to Şişhane", "Walk from Galata Bridge"],
                best_time="Afternoon and evening",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Galata Tower", "Istanbul Modern"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic port district transformed into creative and cultural hub",
                practical_tips=[
                    "Great for art galleries and design shops",
                    "Excellent coffee culture",
                    "Historic Ottoman and European architecture",
                    "Combine with Galata area visit"
                ],
                keywords=["neighborhood", "arts", "galleries", "trendy", "historic", "creative"],
                sentiment_tags=["trendy", "artistic", "historic", "vibrant"],
                coordinates=(41.0253, 28.9744)
            ),
            
            'kadikoy_moda': AttractionData(
                id='kadikoy_moda',
                name="Kadıköy Moda District",
                turkish_name="Kadıköy Moda",
                district="Kadıköy",
                category=AttractionCategory.NEIGHBORHOOD,
                description="Hip Asian-side neighborhood with coastal promenade, cafes, and alternative culture.",
                opening_hours={"daily": "24 hours (businesses vary)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free to explore, dining varies",
                duration="Half day (3-4 hours)",
                transportation=["Ferry to Kadıköy", "Metro to Kadıköy"],
                best_time="Afternoon and evening",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Bağdat Street", "Fenerbahçe Park"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Bohemian heart of Asian Istanbul with strong local culture",
                practical_tips=[
                    "Great local food scene",
                    "Moda coastal walk",
                    "Alternative shops and bars",
                    "Less touristy than European side"
                ],
                keywords=["neighborhood", "asian side", "bohemian", "coastal", "alternative", "local"],
                sentiment_tags=["hip", "local", "alternative", "coastal"],
                coordinates=(40.9833, 29.0333)
            ),
            
            'bagdat_street': AttractionData(
                id='bagdat_street',
                name="Bağdat Street",
                turkish_name="Bağdat Caddesi",
                district="Kadıköy",
                category=AttractionCategory.SHOPPING_DISTRICT,
                description="Upscale shopping avenue on Asian side with luxury brands, cafes, and restaurants.",
                opening_hours={"daily": "Most shops 10:00-22:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free to walk, shopping varies widely",
                duration="2-4 hours",
                transportation=["Metro to Bostancı or Kozyatağı", "Bus along the street"],
                best_time="Afternoon and evening for shopping and dining",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Fenerbahçe Park", "Kadıköy center"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Asian side's answer to İstiklal Street, popular with locals",
                practical_tips=[
                    "More expensive than European side",
                    "Great for people watching",
                    "Wide sidewalks for comfortable walking",
                    "Mix of international and Turkish brands"
                ],
                keywords=["shopping", "avenue", "luxury", "asian side", "brands", "upscale"],
                sentiment_tags=["upscale", "modern", "shopping", "elegant"],
                coordinates=(40.9500, 29.1000)
            ),
            
            'fenerbahce_park': AttractionData(
                id='fenerbahce_park',
                name="Fenerbahçe Park",
                turkish_name="Fenerbahçe Parkı",
                district="Kadıköy",
                category=AttractionCategory.PARK_GARDEN,
                description="Waterfront park with lighthouse, marina, and panoramic views of Marmara Sea.",
                opening_hours={"daily": "24 hours"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="1-2 hours",
                transportation=["Bus to Fenerbahçe", "Metro to Kadıköy then bus"],
                best_time="Sunset for spectacular views",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Bağdat Street", "Kadıköy center"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Historic lighthouse area and popular local recreation spot",
                practical_tips=[
                    "Famous lighthouse for photos",
                    "Great for jogging and cycling",
                    "Tea gardens and cafes",
                    "Less crowded than European side parks"
                ],
                keywords=["park", "lighthouse", "waterfront", "marmara", "marina", "views"],
                sentiment_tags=["peaceful", "scenic", "local", "refreshing"],
                coordinates=(40.9667, 29.0500)
            ),
            
            'asian_side_coast': AttractionData(
                id='asian_side_coast',
                name="Asian Side Coastal Walk",
                turkish_name="Anadolu Sahili",
                district="Multiple",
                category=AttractionCategory.NATURE_RECREATION,
                description="Scenic coastal walkway from Kadıköy to Bostancı with parks, cafes, and sea views.",
                opening_hours={"daily": "24 hours"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="2-4 hours for full walk",
                transportation=["Metro to various stops", "Bus along coast", "Marmaray train"],
                best_time="Morning or late afternoon",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Fenerbahçe Park", "Bağdat Street"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Popular local recreation area away from tourist crowds",
                practical_tips=[
                    "Great for cycling and jogging",
                    "Many seaside cafes and restaurants",
                    "Less touristy than European side",
                    "Beautiful sunrise views"
                ],
                keywords=["coastal", "walking", "cycling", "asian side", "marmara", "recreation"],
                sentiment_tags=["peaceful", "local", "scenic", "refreshing"],
                coordinates=(40.9667, 29.0833)
            ),
            
            'buyukada_island': AttractionData(
                id='buyukada_island',
                name="Büyükada (Princes' Islands)",
                turkish_name="Büyükada",
                district="Adalar",
                category=AttractionCategory.ISLAND_NATURE,
                description="Largest Princes' Island with Victorian mansions, horse carriages, and pine forests.",
                opening_hours={"daily": "Ferry schedule dependent"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="75 TL ferry + island activities",
                duration="Full day (6-8 hours)",
                transportation=["Ferry from Kabataş", "Ferry from Bostancı"],
                best_time="Spring through autumn, weekdays less crowded",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Other Princes' Islands", "Historic mansions"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic retreat with beautiful 19th century Ottoman and Greek architecture",
                practical_tips=[
                    "No cars allowed - bicycles and horse carriages only",
                    "Climb to Aya Yorgi Church for views",
                    "Swimming areas in summer",
                    "Famous for seafood restaurants"
                ],
                keywords=["island", "ferry", "carriages", "mansions", "pine forests", "car-free"],
                sentiment_tags=["peaceful", "nostalgic", "natural", "romantic"],
                coordinates=(40.8667, 29.1167)
            )
            
        })
        
        # ADDITIONAL EXPANSION - NEW DISTRICTS & HIDDEN GEMS (25 attractions)
        attractions.update({
            # TAKSIM-ŞIŞLI DISTRICT (5 attractions)
            'taksim_square': AttractionData(
                id='taksim_square',
                name="Taksim Square",
                turkish_name="Taksim Meydanı",
                district="Taksim",
                category=AttractionCategory.CULTURAL_CENTER,
                description="Istanbul's main square and cultural heart, gateway to modern Istanbul.",
                opening_hours={"daily": "24/7"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="30 minutes",
                transportation=["Taksim Metro", "Kabataş Funicular", "Various buses"],
                best_time="Evening for vibrant atmosphere",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["İstiklal Street", "Galata Tower", "Dolmabahçe Palace"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Symbol of modern Turkey, site of important historical events",
                practical_tips=[
                    "Starting point for İstiklal Street exploration",
                    "Republic Monument commemorates founding of modern Turkey",
                    "Metro hub connecting to major districts",
                    "Street performances and events common"
                ],
                keywords=["square", "modern", "cultural", "hub", "republic", "taksim"],
                sentiment_tags=["vibrant", "modern", "bustling", "central"],
                coordinates=(41.0369, 28.9850)
            ),
            
            'military_museum_harbiye': AttractionData(
                id='military_museum_harbiye',
                name="Military Museum",
                turkish_name="Askeri Müze",
                district="Şişli",
                category=AttractionCategory.MUSEUM,
                description="Comprehensive military history museum with Ottoman and Turkish military artifacts.",
                opening_hours={"wed_sun": "09:00-17:00", "closed": "mon,tue"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="45 TL",
                duration="2-3 hours",
                transportation=["Osmanbey Metro", "Şişli-Mecidiyeköy Metro"],
                best_time="Weekend mornings",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Taksim Square", "İstiklal Street", "Nişantaşı"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Showcases Ottoman and Turkish military heritage through centuries",
                practical_tips=[
                    "Famous Janissary Band performance on weekends",
                    "Extensive weapon and uniform collections",
                    "Photography restricted in some areas",
                    "Good audio guide available"
                ],
                keywords=["military", "ottoman", "janissary", "history", "museum", "weapons"],
                sentiment_tags=["educational", "historical", "impressive", "patriotic"],
                coordinates=(41.0458, 28.9886)
            ),
            
            'nisantasi_shopping': AttractionData(
                id='nisantasi_shopping',
                name="Nişantaşı Shopping District",
                turkish_name="Nişantaşı",
                district="Şişli",
                category=AttractionCategory.SHOPPING,
                description="Istanbul's most elegant shopping district with luxury boutiques and cafes.",
                opening_hours={"daily": "10:00-22:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Varies (luxury shopping)",
                duration="2-4 hours",
                transportation=["Osmanbey Metro", "Various buses"],
                best_time="Afternoon shopping and evening dining",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Military Museum", "Taksim Square", "Maçka Park"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic upscale neighborhood representing modern Turkish lifestyle",
                practical_tips=[
                    "High-end fashion brands and Turkish designers",
                    "Excellent restaurants and patisseries",
                    "Beautiful Art Nouveau architecture",
                    "Valet parking available in many stores"
                ],
                keywords=["shopping", "luxury", "fashion", "elegant", "boutique", "upscale"],
                sentiment_tags=["elegant", "sophisticated", "trendy", "luxurious"],
                coordinates=(41.0497, 28.9944)
            ),
            
            'macka_park': AttractionData(
                id='macka_park',
                name="Maçka Park",
                turkish_name="Maçka Parkı",
                district="Şişli",
                category=AttractionCategory.PARK_GARDEN,
                description="Large urban park with cable car, perfect for families and relaxation.",
                opening_hours={"daily": "06:00-22:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (cable car extra)",
                duration="1-3 hours",
                transportation=["Taksim Metro + walk", "Various buses"],
                best_time="Afternoon and early evening",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Dolmabahçe Palace", "Nişantaşı", "Taksim Square"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Important green space in dense urban area, recreational heart of modern Istanbul",
                practical_tips=[
                    "Cable car connects to Eyüp with Bosphorus views",
                    "Playgrounds and walking paths",
                    "Weekend markets and events",
                    "Good restaurants around the park"
                ],
                keywords=["park", "cable car", "family", "green space", "recreation", "urban"],
                sentiment_tags=["peaceful", "family-friendly", "scenic", "relaxing"],
                coordinates=(41.0431, 28.9889)
            ),
            
            'swissotel_bosphorus_view': AttractionData(
                id='swissotel_bosphorus_view',
                name="Swissôtel The Bosphorus Viewpoint",
                turkish_name="Swissôtel Boğaz Manzarası",
                district="Şişli",
                category=AttractionCategory.VIEWPOINT,
                description="Premium hotel rooftop with spectacular Bosphorus panoramic views.",
                opening_hours={"daily": "Hotel restaurant hours"},
                entrance_fee=BudgetCategory.EXPENSIVE,
                estimated_cost="Restaurant/bar minimum spend",
                duration="1-2 hours",
                transportation=["Taksim Metro + taxi", "Various buses"],
                best_time="Sunset cocktail hour",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Maçka Park", "Dolmabahçe Palace", "Taksim Square"],
                is_family_friendly=False,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Premium vantage point showcasing Istanbul's European and Asian sides",
                practical_tips=[
                    "Reservation required for restaurant",
                    "Dress code enforced",
                    "Expensive but exceptional views",
                    "Perfect for special occasions"
                ],
                keywords=["hotel", "viewpoint", "bosphorus", "luxury", "rooftop", "panoramic"],
                sentiment_tags=["luxurious", "romantic", "spectacular", "premium"],
                coordinates=(41.0447, 28.9936)
            ),
            
            # KARAKÖY-GALATA EXPANSION (3 attractions)
            'galata_mevlevi_lodge': AttractionData(
                id='galata_mevlevi_lodge',
                name="Galata Mevlevi Lodge Museum",
                turkish_name="Galata Mevlevihanesi Müzesi",
                district="Galata",
                category=AttractionCategory.MUSEUM,
                description="Historical dervish lodge showcasing Sufi culture and whirling dervish tradition.",
                opening_hours={"tue_sun": "09:00-17:00", "closed": "monday"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="42 TL",
                duration="1-2 hours",
                transportation=["Karaköy Metro", "Tünel Funicular", "Galata Bridge walk"],
                best_time="Morning for peaceful experience",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Galata Tower", "İstiklal Street", "Pera Museum"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="Ottoman-era Sufi lodge representing mystical Islamic tradition",
                practical_tips=[
                    "Beautiful traditional architecture",
                    "Dervish ceremony demonstrations on weekends",
                    "Peaceful garden courtyard",
                    "Rich collection of Mevlevi artifacts"
                ],
                keywords=["mevlevi", "dervish", "sufi", "lodge", "mystical", "traditional"],
                sentiment_tags=["spiritual", "peaceful", "mystical", "traditional"],
                coordinates=(41.0299, 28.9736)
            ),
            
            'salt_galata': AttractionData(
                id='salt_galata',
                name="SALT Galata",
                turkish_name="SALT Galata",
                district="Galata",
                category=AttractionCategory.CULTURAL_CENTER,
                description="Contemporary art and culture center in historic Ottoman Bank building.",
                opening_hours={"tue_sat": "10:00-20:00", "sun": "12:00-18:00", "closed": "monday"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="1-3 hours",
                transportation=["Karaköy Metro", "Tünel Funicular"],
                best_time="Afternoon for exhibitions",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Galata Tower", "İstanbul Modern", "Karaköy waterfront"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Leading contemporary art space in restored Ottoman Bank headquarters",
                practical_tips=[
                    "Free entrance to most exhibitions",
                    "Research library and archives",
                    "Excellent bookshop and cafe",
                    "Check website for current exhibitions"
                ],
                keywords=["contemporary", "art", "culture", "ottoman bank", "exhibitions", "modern"],
                sentiment_tags=["contemporary", "intellectual", "creative", "inspiring"],
                coordinates=(41.0289, 28.9739)
            ),
            
            'karakoy_waterfront': AttractionData(
                id='karakoy_waterfront',
                name="Karaköy Waterfront Promenade",
                turkish_name="Karaköy Sahil",
                district="Galata",
                category=AttractionCategory.WATERFRONT,
                description="Modern waterfront promenade with restaurants, cafes, and Bosphorus views.",
                opening_hours={"daily": "24/7"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (dining extra)",
                duration="1-2 hours",
                transportation=["Karaköy Metro", "Eminönü Ferry", "Golden Horn boats"],
                best_time="Evening for dining and views",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Galata Bridge", "İstanbul Modern", "SALT Galata"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Revitalized waterfront representing modern Istanbul's urban renewal",
                practical_tips=[
                    "Many trendy restaurants and bars",
                    "Perfect for Bosphorus sunset views",
                    "Weekend fish market nearby",
                    "Easy ferry access to Asian side"
                ],
                keywords=["waterfront", "bosphorus", "promenade", "restaurants", "modern", "trendy"],
                sentiment_tags=["trendy", "scenic", "vibrant", "modern"],
                coordinates=(41.0264, 28.9742)
            ),
            
            # ASIAN SIDE EXPANSION - ÜSKÜDAR & KADIKÖY (4 attractions)
            'maiden_tower': AttractionData(
                id='maiden_tower',
                name="Maiden's Tower",
                turkish_name="Kız Kulesi",
                district="Üsküdar",
                category=AttractionCategory.HISTORICAL_MONUMENT,
                description="Iconic tower on small islet with restaurant and panoramic Istanbul views.",
                opening_hours={"daily": "09:00-19:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="80 TL + boat transfer",
                duration="2-3 hours",
                transportation=["Üsküdar Ferry + special boat", "Kabataş boat tours"],
                best_time="Lunch or dinner for full experience",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Üsküdar Waterfront", "Beylerbeyi Palace", "Çamlıca Hill"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Ancient lighthouse and customs point, featured in many legends and films",
                practical_tips=[
                    "Restaurant reservation recommended",
                    "Boat transfer included with entrance",
                    "Perfect for marriage proposals",
                    "Stunning 360-degree city views"
                ],
                keywords=["tower", "island", "bosphorus", "romantic", "restaurant", "legend"],
                sentiment_tags=["romantic", "legendary", "iconic", "magical"],
                coordinates=(41.0205, 29.0042)
            ),
            
            'atik_valide_mosque': AttractionData(
                id='atik_valide_mosque',
                name="Atik Valide Mosque",
                turkish_name="Atik Valide Camii",
                district="Üsküdar",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="Magnificent 16th-century mosque by Sinan, peaceful atmosphere away from crowds.",
                opening_hours={"daily": "Sunrise-Sunset (except prayer times)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="30-60 minutes",
                transportation=["Üsküdar Ferry", "Marmaray", "Various buses"],
                best_time="Morning or late afternoon",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Çinili Mosque", "Üsküdar Waterfront", "Maiden's Tower"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="Masterpiece by master architect Sinan, representing classical Ottoman architecture",
                practical_tips=[
                    "Dress modestly and remove shoes",
                    "Beautiful tile work and calligraphy",
                    "Peaceful courtyard for reflection",
                    "Less crowded than Sultanahmet mosques"
                ],
                keywords=["mosque", "sinan", "ottoman", "architecture", "peaceful", "hidden"],
                sentiment_tags=["peaceful", "spiritual", "architectural", "authentic"],
                coordinates=(41.0214, 29.0197)
            ),
            
            'kadikoy_bull_statue': AttractionData(
                id='kadikoy_bull_statue',
                name="Kadıköy Bull Statue",
                turkish_name="Kadıköy Boğa Heykeli",
                district="Kadıköy",
                category=AttractionCategory.CULTURAL_CENTER,
                description="Iconic bronze bull statue and meeting point in Kadıköy's cultural heart.",
                opening_hours={"daily": "24/7"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="15-30 minutes",
                transportation=["Kadıköy Ferry", "Marmaray", "Metro"],
                best_time="Any time, evening for area atmosphere",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Moda Park", "Bagdat Street", "Fenerbahçe Park"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Symbol of Kadıköy district and popular meeting point for locals",
                practical_tips=[
                    "Janissary band performances (check schedule)",
                    "Extensive weapon collections",
                    "Ottoman military artifacts",
                    "Less touristy than other museums"
                ],
                keywords=["military", "museum", "ottoman", "janissary", "weapons", "history"],
                sentiment_tags=["historic", "educational", "impressive", "traditional"],
                coordinates=(41.0439, 28.9850)
            ),
            
            'rahmi_koc_industrial': AttractionData(
                id='rahmi_koc_industrial',
                name="Rahmi M. Koç Industrial Museum",
                turkish_name="Rahmi M. Koç Endüstri Müzesi",
                district="Beyoğlu",
                category=AttractionCategory.MUSEUM,
                description="Expanded reference to the industrial heritage museum with vintage transportation.",
                opening_hours={"tuesday_sunday": "10:00-17:00 (weekends until 18:00)"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="25 TL",
                duration="2-4 hours",
                transportation=["Metro to Haliç", "Ferry to Golden Horn"],
                best_time="Weekend for full experience",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Golden Horn", "Miniaturk"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Turkey's premier industrial heritage museum",
                practical_tips=[
                    "Perfect for families with curious kids",
                    "Interactive vintage car exhibits",
                    "Submarine you can explore",
                    "Cafe with waterfront views"
                ],
                keywords=["industrial", "vintage", "cars", "submarine", "interactive", "heritage"],
                sentiment_tags=["educational", "interactive", "nostalgic", "family-fun"],
                coordinates=(41.0322, 28.9711)
            )
            
        })
        
        # EXPANSION PHASE 2: Additional 25 attractions for total of 75
        # TAKSIM-ŞIŞLI DISTRICT (5 attractions)
        attractions.update({
            'taksim_square': AttractionData(
                id='taksim_square',
                name="Taksim Square",
                turkish_name="Taksim Meydanı",
                district="Taksim",
                category=AttractionCategory.CULTURAL_CENTER,
                description="Istanbul's main square and cultural heart, gateway to modern Istanbul.",
                opening_hours={"daily": "24/7"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="30 minutes",
                transportation=["Taksim Metro", "Kabataş Funicular", "Various buses"],
                best_time="Evening for vibrant atmosphere",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["İstiklal Street", "Galata Tower", "Dolmabahçe Palace"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Symbol of modern Turkey, site of important historical events",
                practical_tips=[
                    "Starting point for İstiklal Street exploration",
                    "Republic Monument commemorates founding of modern Turkey",
                    "Metro hub connecting to major districts",
                    "Street performances and events common"
                ],
                keywords=["square", "modern", "cultural", "hub", "republic", "taksim"],
                sentiment_tags=["vibrant", "modern", "bustling", "central"],
                coordinates=(41.0369, 28.9850)
            ),
            
            'military_museum_harbiye': AttractionData(
                id='military_museum_harbiye',
                name="Military Museum",
                turkish_name="Askeri Müze",
                district="Şişli",
                category=AttractionCategory.MUSEUM,
                description="Comprehensive military history museum with Ottoman and Turkish military artifacts.",
                opening_hours={"wed_sun": "09:00-17:00", "closed": "mon,tue"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="45 TL",
                duration="2-3 hours",
                transportation=["Osmanbey Metro", "Şişli-Mecidiyeköy Metro"],
                best_time="Weekend mornings",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Taksim Square", "İstiklal Street", "Nişantaşı"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Showcases Ottoman and Turkish military heritage through centuries",
                practical_tips=[
                    "Famous Janissary Band performance on weekends",
                    "Extensive weapon and uniform collections",
                    "Photography restricted in some areas",
                    "Good audio guide available"
                ],
                keywords=["military", "ottoman", "janissary", "history", "museum", "weapons"],
                sentiment_tags=["educational", "historical", "impressive", "patriotic"],
                coordinates=(41.0458, 28.9886)
            ),
            
            'nisantasi_shopping': AttractionData(
                id='nisantasi_shopping',
                name="Nişantaşı Shopping District",
                turkish_name="Nişantaşı",
                district="Şişli",
                category=AttractionCategory.SHOPPING,
                description="Istanbul's most elegant shopping district with luxury boutiques and cafes.",
                opening_hours={"daily": "10:00-22:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Varies (luxury shopping)",
                duration="2-4 hours",
                transportation=["Osmanbey Metro", "Various buses"],
                best_time="Afternoon shopping and evening dining",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Military Museum", "Taksim Square", "Maçka Park"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic upscale neighborhood representing modern Turkish lifestyle",
                practical_tips=[
                    "High-end fashion brands and Turkish designers",
                    "Excellent restaurants and patisseries",
                    "Beautiful Art Nouveau architecture",
                    "Valet parking available in many stores"
                ],
                keywords=["shopping", "luxury", "fashion", "elegant", "boutique", "upscale"],
                sentiment_tags=["elegant", "sophisticated", "trendy", "luxurious"],
                coordinates=(41.0497, 28.9944)
            ),
            
            'macka_park': AttractionData(
                id='macka_park',
                name="Maçka Park",
                turkish_name="Maçka Parkı",
                district="Şişli",
                category=AttractionCategory.PARK_GARDEN,
                description="Large urban park with cable car, perfect for families and relaxation.",
                opening_hours={"daily": "06:00-22:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (cable car extra)",
                duration="1-3 hours",
                transportation=["Taksim Metro + walk", "Various buses"],
                best_time="Afternoon and early evening",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Dolmabahçe Palace", "Nişantaşı", "Taksim Square"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Important green space in dense urban area, recreational heart of modern Istanbul",
                practical_tips=[
                    "Cable car connects to Eyüp with Bosphorus views",
                    "Playgrounds and walking paths",
                    "Weekend markets and events",
                    "Good restaurants around the park"
                ],
                keywords=["park", "cable car", "family", "green space", "recreation", "urban"],
                sentiment_tags=["peaceful", "family-friendly", "scenic", "relaxing"],
                coordinates=(41.0431, 28.9889)
            ),
            
            'swissotel_bosphorus_view': AttractionData(
                id='swissotel_bosphorus_view',
                name="Swissôtel The Bosphorus Viewpoint",
                turkish_name="Swissôtel Boğaz Manzarası",
                district="Şişli",
                category=AttractionCategory.VIEWPOINT,
                description="Premium hotel rooftop with spectacular Bosphorus panoramic views.",
                opening_hours={"daily": "Hotel restaurant hours"},
                entrance_fee=BudgetCategory.EXPENSIVE,
                estimated_cost="Restaurant/bar minimum spend",
                duration="1-2 hours",
                transportation=["Taksim Metro + taxi", "Various buses"],
                best_time="Sunset cocktail hour",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Maçka Park", "Dolmabahçe Palace", "Taksim Square"],
                is_family_friendly=False,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Premium vantage point showcasing Istanbul's European and Asian sides",
                practical_tips=[
                    "Reservation required for restaurant",
                    "Dress code enforced",
                    "Expensive but exceptional views",
                    "Perfect for special occasions"
                ],
                keywords=["hotel", "viewpoint", "bosphorus", "luxury", "rooftop", "panoramic"],
                sentiment_tags=["luxurious", "romantic", "spectacular", "premium"],
                coordinates=(41.0447, 28.9936)
            )
        })
        
        # KARAKÖY-GALATA EXPANSION (3 attractions) 
        attractions.update({
            'galata_mevlevi_lodge': AttractionData(
                id='galata_mevlevi_lodge',
                name="Galata Mevlevi Lodge Museum",
                turkish_name="Galata Mevlevihanesi Müzesi",
                district="Galata",
                category=AttractionCategory.MUSEUM,
                description="Historical dervish lodge showcasing Sufi culture and whirling dervish tradition.",
                opening_hours={"tue_sun": "09:00-17:00", "closed": "monday"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="42 TL",
                duration="1-2 hours",
                transportation=["Karaköy Metro", "Tünel Funicular", "Galata Bridge walk"],
                best_time="Morning for peaceful experience",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Galata Tower", "İstiklal Street", "Pera Museum"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="Ottoman-era Sufi lodge representing mystical Islamic tradition",
                practical_tips=[
                    "Beautiful traditional architecture",
                    "Dervish ceremony demonstrations on weekends",
                    "Peaceful garden courtyard",
                    "Rich collection of Mevlevi artifacts"
                ],
                keywords=["mevlevi", "dervish", "sufi", "lodge", "mystical", "traditional"],
                sentiment_tags=["spiritual", "peaceful", "mystical", "traditional"],
                coordinates=(41.0299, 28.9736)
            ),
            
            'salt_galata': AttractionData(
                id='salt_galata',
                name="SALT Galata",
                turkish_name="SALT Galata",
                district="Galata",
                category=AttractionCategory.CULTURAL_CENTER,
                description="Contemporary art and culture center in historic Ottoman Bank building.",
                opening_hours={"tue_sat": "10:00-20:00", "sun": "12:00-18:00", "closed": "monday"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="1-3 hours",
                transportation=["Karaköy Metro", "Tünel Funicular"],
                best_time="Afternoon for exhibitions",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Galata Tower", "İstanbul Modern", "Karaköy waterfront"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Leading contemporary art space in restored Ottoman Bank headquarters",
                practical_tips=[
                    "Free entrance to most exhibitions",
                    "Research library and archives",
                    "Excellent bookshop and cafe",
                    "Check website for current exhibitions"
                ],
                keywords=["contemporary", "art", "culture", "ottoman bank", "exhibitions", "modern"],
                sentiment_tags=["contemporary", "intellectual", "creative", "inspiring"],
                coordinates=(41.0289, 28.9739)
            ),
            
            'karakoy_waterfront': AttractionData(
                id='karakoy_waterfront',
                name="Karaköy Waterfront Promenade",
                turkish_name="Karaköy Sahil",
                district="Galata",
                category=AttractionCategory.WATERFRONT,
                description="Modern waterfront promenade with restaurants, cafes, and Bosphorus views.",
                opening_hours={"daily": "24/7"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (dining extra)",
                duration="1-2 hours",
                transportation=["Karaköy Metro", "Eminönü Ferry", "Golden Horn boats"],
                best_time="Evening for dining and views",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Galata Bridge", "İstanbul Modern", "SALT Galata"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Revitalized waterfront representing modern Istanbul's urban renewal",
                practical_tips=[
                    "Many trendy restaurants and bars",
                    "Perfect for Bosphorus sunset views",
                    "Weekend fish market nearby",
                    "Easy ferry access to Asian side"
                ],
                keywords=["waterfront", "bosphorus", "promenade", "restaurants", "modern", "trendy"],
                sentiment_tags=["trendy", "scenic", "vibrant", "modern"],
                coordinates=(41.0264, 28.9742)
            )
        })
        
        # ASIAN SIDE EXPANSION - ÜSKÜDAR & KADIKÖY (4 attractions)
        attractions.update({
            'maiden_tower': AttractionData(
                id='maiden_tower',
                name="Maiden's Tower",
                turkish_name="Kız Kulesi",
                district="Üsküdar",
                category=AttractionCategory.HISTORICAL_MONUMENT,
                description="Iconic tower on small islet with restaurant and panoramic Istanbul views.",
                opening_hours={"daily": "09:00-19:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="80 TL + boat transfer",
                duration="2-3 hours",
                transportation=["Üsküdar Ferry + special boat", "Kabataş boat tours"],
                best_time="Lunch or dinner for full experience",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Üsküdar Waterfront", "Beylerbeyi Palace", "Çamlıca Hill"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Ancient lighthouse and customs point, featured in many legends and films",
                practical_tips=[
                    "Restaurant reservation recommended",
                    "Boat transfer included with entrance",
                    "Perfect for marriage proposals",
                    "Stunning 360-degree city views"
                ],
                keywords=["tower", "island", "bosphorus", "romantic", "restaurant", "legend"],
                sentiment_tags=["romantic", "legendary", "iconic", "magical"],
                coordinates=(41.0205, 29.0042)
            ),
            
            'atik_valide_mosque': AttractionData(
                id='atik_valide_mosque',
                name="Atik Valide Mosque",
                turkish_name="Atik Valide Camii",
                district="Üsküdar",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="Magnificent 16th-century mosque by Sinan, peaceful atmosphere away from crowds.",
                opening_hours={"daily": "Sunrise-Sunset (except prayer times)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="30-60 minutes",
                transportation=["Üsküdar Ferry", "Marmaray", "Various buses"],
                best_time="Morning or late afternoon",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Çinili Mosque", "Üsküdar Waterfront", "Maiden's Tower"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="Masterpiece by master architect Sinan, representing classical Ottoman architecture",
                practical_tips=[
                    "Dress modestly and remove shoes",
                    "Beautiful tile work and calligraphy",
                    "Peaceful courtyard for reflection",
                    "Less crowded than Sultanahmet mosques"
                ],
                keywords=["mosque", "sinan", "ottoman", "architecture", "peaceful", "hidden"],
                sentiment_tags=["peaceful", "spiritual", "architectural", "authentic"],
                coordinates=(41.0214, 29.0197)
            ),
            
            'kadikoy_bull_statue': AttractionData(
                id='kadikoy_bull_statue',
                name="Kadıköy Bull Statue",
                turkish_name="Kadıköy Boğa Heykeli",
                district="Kadıköy",
                category=AttractionCategory.CULTURAL_CENTER,
                description="Iconic bronze bull statue and meeting point in Kadıköy's cultural heart.",
                opening_hours={"daily": "24/7"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="15-30 minutes",
                transportation=["Kadıköy Ferry", "Marmaray", "Metro"],
                best_time="Any time, evening for area atmosphere",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Moda Park", "Bagdat Street", "Fenerbahçe Park"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Symbol of Kadıköy district and popular meeting point for locals",
                practical_tips=[
                    "Popular photo spot",
                    "Starting point for Kadıköy exploration",
                    "Surrounded by cafes and shops",
                    "Weekend street performances"
                ],
                keywords=["statue", "bull", "meeting point", "kadikoy", "symbol", "local"],
                sentiment_tags=["local", "iconic", "friendly", "cultural"],
                coordinates=(40.9663, 29.0268)
            ),
            
            'kadikoy_market': AttractionData(
                id='kadikoy_market',
                name="Kadıköy Market",
                turkish_name="Kadıköy Pazarı",
                district="Kadıköy",
                category=AttractionCategory.MARKET_SHOPPING,
                description="Vibrant local market with fresh produce, spices, and authentic Turkish foods.",
                opening_hours={"daily": "08:00-19:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free to browse, budget for shopping",
                duration="1-2 hours",
                transportation=["Kadıköy Ferry", "Marmaray"],
                best_time="Morning for freshest products",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Bull Statue", "Moda neighborhood", "Bagdat Street"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Traditional Turkish market experience representing local daily life",
                practical_tips=[
                    "Bargaining expected and welcomed",
                    "Try local specialties and street food",
                    "Bring cash for purchases",
                    "Perfect for authentic local experience"
                ],
                keywords=["market", "local", "food", "authentic", "traditional", "shopping"],
                sentiment_tags=["authentic", "vibrant", "local", "traditional"],
                coordinates=(40.9654, 29.0275)
            )
        })
        
        # PRINCES' ISLANDS EXPANSION (3 attractions)
        attractions.update({
            'heybeliada': AttractionData(
                id='heybeliada',
                name="Heybeliada Island",
                turkish_name="Heybeliada",
                district="Adalar",
                category=AttractionCategory.ISLAND_NATURE,
                description="Second-largest Princes' Island with historic Naval Academy and pine forests.",
                opening_hours={"daily": "Ferry schedule dependent"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="Ferry fare + island activities",
                duration="Full day",
                transportation=["Ferry from Kabataş or Eminönü"],
                best_time="Spring through autumn",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Büyükada", "Burgazada", "Kınalıada"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic island with Ottoman-era mansions and important naval heritage",
                practical_tips=[
                    "No motor vehicles - bicycles and horses only",
                    "Historic Naval Academy building",
                    "Beautiful nature walks through pine forests",
                    "Less crowded than Büyükada"
                ],
                keywords=["island", "naval academy", "pine forest", "historic", "peaceful", "nature"],
                sentiment_tags=["peaceful", "natural", "historic", "relaxing"],
                coordinates=(40.8833, 29.0833)
            ),
            
            'burgazada': AttractionData(
                id='burgazada',
                name="Burgazada Island",
                turkish_name="Burgazada",
                district="Adalar",
                category=AttractionCategory.ISLAND_NATURE,
                description="Quiet Princes' Island perfect for hiking, swimming, and escaping city life.",
                opening_hours={"daily": "Ferry schedule dependent"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="Ferry fare + dining",
                duration="Half to full day",
                transportation=["Ferry from Kabataş or Eminönü"],
                best_time="Summer for swimming, spring/autumn for hiking",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Heybeliada", "Büyükada", "Kınalıada"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Peaceful retreat representing traditional island life away from urban stress",
                practical_tips=[
                    "Excellent hiking trails with sea views",
                    "Small beaches for swimming",
                    "Limited restaurants - bring snacks",
                    "Perfect for quiet romantic getaway"
                ],
                keywords=["island", "hiking", "swimming", "quiet", "peaceful", "nature"],
                sentiment_tags=["tranquil", "natural", "secluded", "refreshing"],
                coordinates=(40.8767, 29.0650)
            ),
            
            'kinaliada': AttractionData(
                id='kinaliada',
                name="Kınalıada Island",
                turkish_name="Kınalıada",
                district="Adalar",
                category=AttractionCategory.ISLAND_NATURE,
                description="Smallest inhabited Princes' Island, perfect for short visits and photography.",
                opening_hours={"daily": "Ferry schedule dependent"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="Ferry fare only",
                duration="2-4 hours",
                transportation=["Ferry from Kabataş or Eminönü"],
                best_time="Afternoon visits combined with other islands",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Burgazada", "Heybeliada", "Büyükada"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Intimate island experience representing simple island life",
                practical_tips=[
                    "Small and walkable in 1-2 hours",
                    "Beautiful coastal walks",
                    "Few facilities - bring water",
                    "Perfect for photography enthusiasts"
                ],
                keywords=["island", "small", "photography", "coastal", "intimate", "simple"],
                sentiment_tags=["intimate", "charming", "photogenic", "simple"],
                coordinates=(40.9197, 29.0394)
            )
        })
        
        # HISTORIC PENINSULA HIDDEN GEMS (3 attractions)
        attractions.update({
            'sokollu_mehmet_pasha_mosque': AttractionData(
                id='sokollu_mehmet_pasha_mosque',
                name="Sokollu Mehmet Pasha Mosque",
                turkish_name="Sokollu Mehmet Paşa Camii",
                district="Fatih",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="Intimate 16th-century mosque by Sinan with precious Kaaba stone fragments.",
                opening_hours={"daily": "Sunrise-Sunset (except prayer times)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="30-45 minutes",
                transportation=["Eminönü Tram", "Sultanahmet Tram + walk"],
                best_time="Morning for peaceful experience",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Blue Mosque", "Hippodrome", "Grand Bazaar"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=True,
                cultural_significance="Sinan masterpiece featuring sacred stone fragments from Mecca's Kaaba",
                practical_tips=[
                    "Contains precious Kaaba stone fragments",
                    "Beautiful Iznik tile decoration",
                    "Peaceful alternative to crowded mosques",
                    "Dress modestly and remove shoes"
                ],
                keywords=["mosque", "sinan", "kaaba stone", "iznik tiles", "hidden gem", "peaceful"],
                sentiment_tags=["sacred", "peaceful", "hidden", "precious"],
                coordinates=(41.0067, 28.9714)
            ),
            
            'turkish_and_islamic_arts_museum': AttractionData(
                id='turkish_islamic_arts_museum',
                name="Turkish and Islamic Arts Museum",
                turkish_name="Türk ve İslam Eserleri Müzesi",
                district="Sultanahmet",
                category=AttractionCategory.MUSEUM,
                description="World-class collection of Islamic art, carpets, and manuscripts in historic palace setting.",
                opening_hours={"tue_sun": "09:00-17:00", "closed": "monday"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="75 TL",
                duration="2-3 hours",
                transportation=["Sultanahmet Tram", "Eminönü Ferry"],
                best_time="Morning for fewer crowds",
                weather_preference=WeatherPreference.INDOOR,
                nearby_attractions=["Blue Mosque", "Hippodrome", "Hagia Sophia"],
                is_family_friendly=True,
                is_romantic=False,
                is_hidden_gem=False,
                cultural_significance="Premier collection of Turkish and Islamic art spanning 1,400 years",
                practical_tips=[
                    "World's finest carpet collection",
                    "Beautiful illuminated manuscripts",
                    "Historic Ibrahim Pasha Palace setting",
                    "Audio guide highly recommended"
                ],
                keywords=["islamic art", "carpets", "manuscripts", "palace", "museum", "culture"],
                sentiment_tags=["artistic", "cultural", "educational", "magnificent"],
                coordinates=(41.0063, 28.9722)
            ),
            
            'little_hagia_sophia': AttractionData(
                id='little_hagia_sophia',
                name="Little Hagia Sophia",
                turkish_name="Küçük Ayasofya Camii",
                district="Sultanahmet",
                category=AttractionCategory.RELIGIOUS_SITE,
                description="Charming 6th-century church-turned-mosque with intimate Byzantine atmosphere.",
                opening_hours={"daily": "Sunrise-Sunset (except prayer times)"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="30-45 minutes",
                transportation=["Sultanahmet Tram", "Eminönü Ferry + walk"],
                best_time="Late afternoon for golden light",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Sultanahmet Mosque", "Hagia Sophia", "Kennedy Avenue"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=True,
                cultural_significance="Justinian-era church predating Hagia Sophia, rare surviving Byzantine architecture",
                practical_tips=[
                    "Much less crowded than main Hagia Sophia",
                    "Beautiful Byzantine architectural details",
                    "Peaceful garden courtyard",
                    "Perfect for quiet reflection"
                ],
                keywords=["byzantine", "church", "mosque", "justinian", "peaceful", "hidden"],
                sentiment_tags=["peaceful", "intimate", "historic", "charming"],
                coordinates=(41.0044, 28.9711)
            )
        })
        
        # MODERN ISTANBUL - LEVENT & NATURE (4 attractions)
        attractions.update({
            'istanbul_sapphire': AttractionData(
                id='istanbul_sapphire',
                name="İstanbul Sapphire Observation Deck",
                turkish_name="İstanbul Sapphire Gözlem Terası",
                district="Levent",
                category=AttractionCategory.VIEWPOINT,
                description="Turkey's tallest building with 360-degree panoramic city and Bosphorus views.",
                opening_hours={"daily": "10:00-22:00"},
                entrance_fee=BudgetCategory.MODERATE,
                estimated_cost="150 TL",
                duration="1-2 hours",
                transportation=["Levent Metro", "4. Levent Metro"],
                best_time="Sunset for spectacular views",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["Levent Shopping Centers", "Maslak business district"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Symbol of modern Istanbul's skyline and urban development",
                practical_tips=[
                    "Turkey's highest observation deck",
                    "Glass elevators with views during ascent",
                    "Shopping mall and restaurants in building",
                    "Pre-booking recommended for sunset"
                ],
                keywords=["skyscraper", "observation deck", "panoramic", "modern", "istanbul", "tallest"],
                sentiment_tags=["spectacular", "modern", "impressive", "panoramic"],
                coordinates=(41.0775, 29.0136)
            ),
            
            'zorlu_center': AttractionData(
                id='zorlu_center',
                name="Zorlu Center",
                turkish_name="Zorlu Center",
                district="Beşiktaş",
                category=AttractionCategory.CULTURAL_CENTER,
                description="Premier arts and shopping complex with PSM concert hall and luxury retailers.",
                opening_hours={"daily": "10:00-22:00"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free (shopping/events extra)",
                duration="2-4 hours",
                transportation=["Gayrettepe Metro", "Various buses"],
                best_time="Evening for concerts and dining",
                weather_preference=WeatherPreference.ALL_WEATHER,
                nearby_attractions=["İstanbul Sapphire", "Levent business district"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Modern cultural hub representing contemporary Turkish arts and lifestyle",
                practical_tips=[
                    "World-class Porsche Sanemann concert hall",
                    "High-end shopping and dining",
                    "Regular concerts and cultural events",
                    "Modern architecture worth seeing"
                ],
                keywords=["arts", "concert hall", "shopping", "modern", "cultural", "luxury"],
                sentiment_tags=["sophisticated", "cultural", "modern", "elegant"],
                coordinates=(41.0678, 29.0097)
            ),
            
            'belgrade_forest': AttractionData(
                id='belgrade_forest',
                name="Belgrade Forest",
                turkish_name="Belgrad Ormanı",
                district="Sarıyer",
                category=AttractionCategory.NATURE_RECREATION,
                description="Large forest preserve perfect for hiking, picnicking, and escaping city life.",
                opening_hours={"daily": "Dawn to dusk"},
                entrance_fee=BudgetCategory.FREE,
                estimated_cost="Free",
                duration="Half to full day",
                transportation=["Metro + bus", "Private car recommended"],
                best_time="Spring and autumn for comfortable hiking",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Kilyos Beach", "Sariyer waterfront"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Historic Ottoman reservoir system within preserved natural area",
                practical_tips=[
                    "Multiple hiking trails for all levels",
                    "Picnic areas and barbecue spots",
                    "Historic Ottoman aqueducts",
                    "Bring water and snacks"
                ],
                keywords=["forest", "hiking", "nature", "ottoman", "aqueducts", "picnic"],
                sentiment_tags=["natural", "peaceful", "refreshing", "historic"],
                coordinates=(41.1833, 28.9833)
            ),
            
            'kilyos_beach': AttractionData(
                id='kilyos_beach',
                name="Kilyos Beach",
                turkish_name="Kilyos Plajı",
                district="Sarıyer",
                category=AttractionCategory.BEACH_RECREATION,
                description="Istanbul's most popular Black Sea beach with clean sand and beach clubs.",
                opening_hours={"seasonal": "May-October, dawn to dusk"},
                entrance_fee=BudgetCategory.BUDGET,
                estimated_cost="Beach club fees vary",
                duration="Full day",
                transportation=["Metro + bus", "Private car recommended"],
                best_time="Summer months for swimming",
                weather_preference=WeatherPreference.OUTDOOR,
                nearby_attractions=["Belgrade Forest", "Şile", "Sarıyer"],
                is_family_friendly=True,
                is_romantic=True,
                is_hidden_gem=False,
                cultural_significance="Main beach escape for Istanbul residents, representing coastal recreation culture",
                practical_tips=[
                    "Multiple beach clubs with facilities",
                    "Strong currents - swim carefully",
                    "Parking can be challenging in summer",
                    "Sunscreen essential - limited shade"
                ],
                keywords=["beach", "black sea", "swimming", "summer", "beach clubs", "coastal"],
                sentiment_tags=["refreshing", "summery", "recreational", "coastal"],
                coordinates=(41.2333, 29.0500)
            )
        })
        
        logger.info(f"Loaded {len(attractions)} comprehensive attractions")
        logger.info(f"🏛️ Loaded {len(attractions)} attractions across {len(set(attr.district for attr in attractions.values()))} districts")
        return attractions
    
    def _load_district_data(self) -> Dict[str, Dict[str, Any]]:
        """Load district information for contextual recommendations"""
        return {
            'sultanahmet': {
                'name': 'Sultanahmet',
                'description': 'Historic heart of Istanbul with major Byzantine and Ottoman monuments',
                'characteristics': ['historic', 'tourist-friendly', 'walkable', 'cultural'],
                'main_attractions': ['hagia_sophia', 'blue_mosque', 'topkapi_palace', 'basilica_cistern'],
                'best_for': ['first-time visitors', 'history lovers', 'architecture enthusiasts']
            },
            'galata': {
                'name': 'Galata/Karaköy',
                'description': 'Trendy district with historic tower and modern art scene',
                'characteristics': ['trendy', 'artistic', 'viewpoints', 'modern'],
                'main_attractions': ['galata_tower', 'galata_bridge_sunset'],
                'best_for': ['photographers', 'art lovers', 'romantic couples']
            },
            'besiktas': {
                'name': 'Beşiktaş',
                'description': 'Vibrant district with palace, parks, and Bosphorus waterfront',
                'characteristics': ['vibrant', 'waterfront', 'parks', 'modern'],
                'main_attractions': ['dolmabahce_palace', 'yildiz_park', 'ortakoy_bosphorus'],
                'best_for': ['families', 'nature lovers', 'waterfront enthusiasts']
            },
            'beyoglu': {
                'name': 'Beyoğlu',
                'description': 'Cultural and entertainment hub with shopping and nightlife',
                'characteristics': ['cultural', 'nightlife', 'shopping', 'entertainment'],
                'main_attractions': ['istiklal_street', 'miniaturk', 'rahmi_koc_museum'],
                'best_for': ['nightlife seekers', 'families', 'culture enthusiasts']
            },
            'fatih': {
                'name': 'Fatih',
                'description': 'Traditional district with hidden gems and authentic neighborhoods',
                'characteristics': ['traditional', 'authentic', 'hidden gems', 'local'],
                'main_attractions': ['chora_church', 'balat_colorful_houses'],
                'best_for': ['authentic experiences', 'hidden gem seekers', 'photographers']
            }
        }
    
    def get_attractions_by_category(self, category: AttractionCategory) -> List[AttractionData]:
        """Get attractions filtered by category"""
        return [attraction for attraction in self.attractions.values() 
                if attraction.category == category]
    
    def get_attractions_by_district(self, district: str) -> List[AttractionData]:
        """Get attractions filtered by district"""
        district_lower = district.lower()
        return [attraction for attraction in self.attractions.values() 
                if attraction.district.lower() == district_lower]
    
    def get_family_friendly_attractions(self) -> List[AttractionData]:
        """Get family-friendly attractions"""
        return [attraction for attraction in self.attractions.values() 
                if attraction.is_family_friendly]
    
    def get_romantic_attractions(self) -> List[AttractionData]:
        """Get romantic attractions"""
        return [attraction for attraction in self.attractions.values() 
                if attraction.is_romantic]
    
    def get_hidden_gems(self) -> List[AttractionData]:
        """Get hidden gem attractions"""
        return [attraction for attraction in self.attractions.values() 
                if attraction.is_hidden_gem]
    
    def get_attractions_by_budget(self, budget: BudgetCategory) -> List[AttractionData]:
        """Get attractions filtered by budget"""
        return [attraction for attraction in self.attractions.values() 
                if attraction.entrance_fee == budget]
    
    def get_weather_appropriate_attractions(self, weather_preference: WeatherPreference) -> List[AttractionData]:
        """Get attractions appropriate for weather conditions"""
        return [attraction for attraction in self.attractions.values() 
                if attraction.weather_preference == weather_preference or 
                attraction.weather_preference == WeatherPreference.ALL_WEATHER]
    
    def search_attractions(self, query: str) -> List[Tuple[AttractionData, float]]:
        """Search attractions with relevance scoring"""
        query_lower = query.lower()
        results = []
        
        for attraction in self.attractions.values():
            score = 0.0
            
            # Name matching (highest priority)
            if query_lower in attraction.name.lower():
                score += 10.0
            if query_lower in attraction.turkish_name.lower():
                score += 8.0
            
            # Description matching
            if query_lower in attraction.description.lower():
                score += 5.0
            
            # Keywords matching
            for keyword in attraction.keywords:
                if query_lower in keyword.lower() or keyword.lower() in query_lower:
                    score += 3.0
            
            # District matching
            if query_lower in attraction.district.lower():
                score += 4.0
            
            # Category matching
            if query_lower in attraction.category.value.lower():
                score += 6.0
            
            # Sentiment tags matching
            for tag in attraction.sentiment_tags:
                if query_lower in tag.lower() or tag.lower() in query_lower:
                    score += 2.0
            
            if score > 0:
                results.append((attraction, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_attraction_recommendations(self, preferences: Dict[str, Any]) -> List[AttractionData]:
        """Get personalized attraction recommendations based on user preferences"""
        recommendations = []
        
        # Extract preferences
        categories = preferences.get('categories', [])
        districts = preferences.get('districts', [])
        budget = preferences.get('budget', None)
        weather = preferences.get('weather', None)
        is_family = preferences.get('family_friendly', False)
        is_romantic = preferences.get('romantic', False)
        hidden_gems = preferences.get('hidden_gems', False)
        
        candidates = list(self.attractions.values())
        
        # Apply filters
        if categories:
            candidates = [a for a in candidates if a.category in categories]
        
        if districts:
            candidates = [a for a in candidates if a.district.lower() in [d.lower() for d in districts]]
        
        if budget:
            candidates = [a for a in candidates if a.entrance_fee == budget]
        
        if weather:
            candidates = [a for a in candidates if a.weather_preference == weather or 
                         a.weather_preference == WeatherPreference.ALL_WEATHER]
        
        if is_family:
            candidates = [a for a in candidates if a.is_family_friendly]
        
        if is_romantic:
            candidates = [a for a in candidates if a.is_romantic]
        
        if hidden_gems:
            candidates = [a for a in candidates if a.is_hidden_gem]
        
        return candidates[:10]  # Return top 10 recommendations
    
    def get_attraction_by_id(self, attraction_id: str) -> Optional[AttractionData]:
        """Get specific attraction by ID"""
        return self.attractions.get(attraction_id)
    
    def get_nearby_attractions(self, attraction_id: str, limit: int = 5) -> List[AttractionData]:
        """Get attractions near a specific attraction"""
        base_attraction = self.attractions.get(attraction_id)
        if not base_attraction:
            return []
        
        nearby = []
        for nearby_id in base_attraction.nearby_attractions:
            if nearby_id.replace(' ', '_').lower() in self.attractions:
                attraction_id_formatted = nearby_id.replace(' ', '_').lower()
                nearby.append(self.attractions[attraction_id_formatted])
        
        return nearby[:limit]
    
    def get_attraction_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the attractions database"""
        total = len(self.attractions)
        
        # Category breakdown
        category_counts = {}
        for attraction in self.attractions.values():
            category = attraction.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # District breakdown
        district_counts = {}
        for attraction in self.attractions.values():
            district = attraction.district
            district_counts[district] = district_counts.get(district, 0) + 1
        
        # Budget breakdown
        budget_counts = {}
        for attraction in self.attractions.values():
            budget = attraction.entrance_fee.value
            budget_counts[budget] = budget_counts.get(budget, 0) + 1
        
        # Special categories
        family_friendly = sum(1 for a in self.attractions.values() if a.is_family_friendly)
        romantic = sum(1 for a in self.attractions.values() if a.is_romantic)
        hidden_gems = sum(1 for a in self.attractions.values() if a.is_hidden_gem)
        
        return {
            'total_attractions': total,
            'categories': category_counts,
            'districts': district_counts,
            'budget_levels': budget_counts,
            'special_categories': {
                'family_friendly': family_friendly,
                'romantic': romantic,
                'hidden_gems': hidden_gems
            }
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    attractions_system = IstanbulAttractionsSystem()
    
    # Print statistics
    stats = attractions_system.get_attraction_stats()
    print("📊 Istanbul Attractions System Statistics:")
    print(f"Total Attractions: {stats['total_attractions']}")
    print(f"Categories: {list(stats['categories'].keys())}")
    print(f"Districts: {list(stats['districts'].keys())}")
    print(f"Family-Friendly: {stats['special_categories']['family_friendly']}")
    print(f"Romantic Spots: {stats['special_categories']['romantic']}")
    print(f"Hidden Gems: {stats['special_categories']['hidden_gems']}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality:")
    search_results = attractions_system.search_attractions("mosque")
    for attraction, score in search_results[:3]:
        print(f"- {attraction.name} (Score: {score})")
    
    print("\n🎯 System ready for integration with AI chatbot!")
