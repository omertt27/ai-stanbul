#!/usr/bin/env python3
"""
Istanbul Museum Advising System
Advanced ML-powered museum recommendation system for Istanbul with GPS integration,
opening hours, categories, and nearby attractions.
"""

import json
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuseumCategory(Enum):
    """Categories for Istanbul museums"""
    HISTORICAL = "historical"
    ARCHAEOLOGICAL = "archaeological"
    ART_MODERN = "art_modern"
    ART_CLASSICAL = "art_classical"
    RELIGIOUS = "religious"
    MILITARY = "military"
    MARITIME = "maritime"
    SCIENCE_TECHNOLOGY = "science_technology"
    CULTURAL = "cultural"
    PALACE = "palace"
    SPECIALIZED = "specialized"
    ETHNOGRAPHIC = "ethnographic"

class PriceCategory(Enum):
    """Price categories for museums"""
    FREE = "free"
    BUDGET = "budget"          # 0-50 TL
    MODERATE = "moderate"      # 50-150 TL
    EXPENSIVE = "expensive"    # 150+ TL

@dataclass
class OpeningHours:
    """Museum opening hours data structure"""
    monday: Tuple[str, str] = ("closed", "closed")
    tuesday: Tuple[str, str] = ("09:00", "17:00")
    wednesday: Tuple[str, str] = ("09:00", "17:00")
    thursday: Tuple[str, str] = ("09:00", "17:00")
    friday: Tuple[str, str] = ("09:00", "17:00")
    saturday: Tuple[str, str] = ("09:00", "17:00")
    sunday: Tuple[str, str] = ("09:00", "17:00")
    
    def is_open(self, day_of_week: int, current_time: time) -> bool:
        """Check if museum is currently open"""
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_name = days[day_of_week]
        hours = getattr(self, day_name)
        
        if hours[0] == "closed":
            return False
            
        try:
            open_time = datetime.strptime(hours[0], "%H:%M").time()
            close_time = datetime.strptime(hours[1], "%H:%M").time()
            return open_time <= current_time <= close_time
        except:
            return False
    
    def get_today_hours(self, day_of_week: int) -> str:
        """Get today's opening hours as string"""
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_name = days[day_of_week]
        hours = getattr(self, day_name)
        
        if hours[0] == "closed":
            return "Closed"
        return f"{hours[0]} - {hours[1]}"

@dataclass
class MuseumData:
    """Comprehensive museum data structure"""
    id: str
    name: str
    category: MuseumCategory
    district: str
    address: str
    coordinates: Tuple[float, float]  # (lat, lng)
    price_tl: float
    price_category: PriceCategory
    opening_hours: OpeningHours
    description: str
    highlights: List[str]
    nearby_attractions: List[str]
    nearby_restaurants: List[str]
    accessibility: bool
    family_friendly: bool
    photography_allowed: bool
    guided_tours: bool
    audio_guide: bool
    gift_shop: bool
    cafe: bool
    estimated_visit_duration: int  # minutes
    best_time_to_visit: List[str]  # ["morning", "afternoon", "evening"]
    crowd_level: str  # "low", "moderate", "high"
    languages: List[str]  # Available languages for guides/info
    website: str = ""
    phone: str = ""
    rating: float = 4.0
    reviews_count: int = 0
    keywords: List[str] = field(default_factory=list)
    seasonal_info: str = ""
    special_exhibitions: List[str] = field(default_factory=list)

class IstanbulMuseumSystem:
    """Advanced museum recommendation system for Istanbul"""
    
    def __init__(self):
        """Initialize the museum system with comprehensive museum data"""
        self.museums = self._load_museum_data()
        logger.info(f"🏛️ Loaded {len(self.museums)} museums across Istanbul")
        
    def _load_museum_data(self) -> Dict[str, MuseumData]:
        """Load comprehensive museum database"""
        museums = {
            'hagia_sophia': MuseumData(
                id='hagia_sophia',
                name='Hagia Sophia Grand Mosque',
                category=MuseumCategory.HISTORICAL,
                district='sultanahmet',
                address='Sultan Ahmet, Ayasofya Meydanı No:1, 34122 Fatih/İstanbul',
                coordinates=(41.0086, 28.9802),
                price_tl=0.0,
                price_category=PriceCategory.FREE,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("09:00", "19:00"),
                    wednesday=("09:00", "19:00"),
                    thursday=("09:00", "19:00"),
                    friday=("13:00", "19:00"),
                    saturday=("09:00", "19:00"),
                    sunday=("09:00", "19:00")
                ),
                description="UNESCO World Heritage Site, former Byzantine cathedral and Ottoman mosque, architectural masterpiece spanning 1,500 years.",
                highlights=[
                    "Byzantine mosaics and Islamic calligraphy",
                    "Massive dome and architectural innovation",
                    "Historical significance spanning empires",
                    "Free admission as functioning mosque"
                ],
                nearby_attractions=["Topkapi Palace", "Blue Mosque", "Basilica Cistern"],
                nearby_restaurants=["Matbah Restaurant", "Seven Hills Restaurant", "Deraliye"],
                accessibility=True,
                family_friendly=True,
                photography_allowed=True,
                guided_tours=True,
                audio_guide=True,
                gift_shop=True,
                cafe=False,
                estimated_visit_duration=60,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="high",
                languages=["turkish", "english", "arabic"],
                website="https://www.hagiasophia.com",
                phone="+90 212 522 1750",
                rating=4.6,
                reviews_count=145000,
                keywords=["byzantine", "ottoman", "mosque", "unesco", "dome", "mosaics"],
                seasonal_info="Less crowded in winter months, stunning at sunset"
            ),
            
            'topkapi_palace': MuseumData(
                id='topkapi_palace',
                name='Topkapi Palace Museum',
                category=MuseumCategory.PALACE,
                district='sultanahmet',
                address='Cankurtaran, 34122 Fatih/İstanbul',
                coordinates=(41.0115, 28.9833),
                price_tl=200.0,
                price_category=PriceCategory.EXPENSIVE,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("09:00", "18:00"),
                    wednesday=("09:00", "18:00"),
                    thursday=("09:00", "18:00"),
                    friday=("09:00", "18:00"),
                    saturday=("09:00", "18:00"),
                    sunday=("09:00", "18:00")
                ),
                description="Former Ottoman imperial palace, treasury, and seat of power for 400 years. Four courtyards with stunning Bosphorus views.",
                highlights=[
                    "Ottoman Imperial Treasury",
                    "Sacred Relics collection",
                    "Harem quarters (additional ticket)",
                    "Palace kitchens and Chinese porcelain",
                    "Panoramic Bosphorus views"
                ],
                nearby_attractions=["Hagia Sophia", "Archaeological Museum", "Gülhane Park"],
                nearby_restaurants=["Matbah Restaurant", "Balıkçı Sabahattin", "Pandeli"],
                accessibility=False,
                family_friendly=True,
                photography_allowed=False,
                guided_tours=True,
                audio_guide=True,
                gift_shop=True,
                cafe=True,
                estimated_visit_duration=180,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="high",
                languages=["turkish", "english", "german", "french"],
                website="https://www.millisaraylar.gov.tr",
                phone="+90 212 512 0480",
                rating=4.4,
                reviews_count=98500,
                keywords=["ottoman", "palace", "sultan", "treasury", "harem", "imperial"],
                seasonal_info="Book online to skip queues, best in spring/autumn",
                special_exhibitions=["Ottoman Portraits", "Imperial Costumes"]
            ),
            
            'istanbul_archaeology_museum': MuseumData(
                id='istanbul_archaeology_museum',
                name='Istanbul Archaeology Museums',
                category=MuseumCategory.ARCHAEOLOGICAL,
                district='sultanahmet',
                address='Osman Hamdi Bey Yokuşu, 34122 Fatih/İstanbul',
                coordinates=(41.0117, 28.9814),
                price_tl=60.0,
                price_category=PriceCategory.BUDGET,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("09:00", "18:00"),
                    wednesday=("09:00", "18:00"),
                    thursday=("09:00", "18:00"),
                    friday=("09:00", "18:00"),
                    saturday=("09:00", "18:00"),
                    sunday=("09:00", "18:00")
                ),
                description="Turkey's first archaeological museum complex with ancient artifacts from Ottoman territories and beyond.",
                highlights=[
                    "Alexander Sarcophagus",
                    "Ancient Orient Museum",
                    "Tiled Kiosk Museum",
                    "Treaty of Kadesh - world's oldest peace treaty",
                    "Extensive Byzantine and Ottoman collections"
                ],
                nearby_attractions=["Topkapi Palace", "Gülhane Park", "Hagia Sophia"],
                nearby_restaurants=["Hamdi Restaurant", "Deraliye", "Matbah Restaurant"],
                accessibility=True,
                family_friendly=True,
                photography_allowed=True,
                guided_tours=True,
                audio_guide=True,
                gift_shop=True,
                cafe=True,
                estimated_visit_duration=120,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="moderate",
                languages=["turkish", "english"],
                website="https://muze.gov.tr/arkeoloji-muzeleri",
                phone="+90 212 520 7740",
                rating=4.3,
                reviews_count=12400,
                keywords=["archaeology", "ancient", "sarcophagus", "byzantine", "ottoman", "artifacts"],
                seasonal_info="Perfect for rainy days, less crowded than major attractions"
            ),
            
            'pera_museum': MuseumData(
                id='pera_museum',
                name='Pera Museum',
                category=MuseumCategory.ART_CLASSICAL,
                district='beyoğlu',
                address='Meşrutiyet Cd. No:65, 34430 Beyoğlu/İstanbul',
                coordinates=(41.0367, 28.9754),
                price_tl=50.0,
                price_category=PriceCategory.BUDGET,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("10:00", "19:00"),
                    wednesday=("10:00", "19:00"),
                    thursday=("10:00", "22:00"),
                    friday=("10:00", "22:00"),
                    saturday=("10:00", "22:00"),
                    sunday=("12:00", "19:00")
                ),
                description="Premier private museum showcasing Orientalist paintings, Anatolian weights, and contemporary exhibitions.",
                highlights=[
                    "Orientalist Painting Collection",
                    "Kütahya Tiles and Ceramics",
                    "Anatolian Weights and Measures",
                    "Rotating contemporary exhibitions",
                    "Beautiful historic building"
                ],
                nearby_attractions=["Galata Tower", "İstiklal Street", "Galatasaray Museum"],
                nearby_restaurants=["Lokanta Maya", "Mikla", "Pandeli"],
                accessibility=True,
                family_friendly=True,
                photography_allowed=False,
                guided_tours=True,
                audio_guide=True,
                gift_shop=True,
                cafe=True,
                estimated_visit_duration=90,
                best_time_to_visit=["afternoon", "evening"],
                crowd_level="moderate",
                languages=["turkish", "english"],
                website="https://www.peramuzesi.org.tr",
                phone="+90 212 334 9900",
                rating=4.2,
                reviews_count=8900,
                keywords=["orientalist", "painting", "art", "ceramics", "contemporary"],
                seasonal_info="Extended hours on weekends, hosts special exhibitions",
                special_exhibitions=["Contemporary Istanbul Artists", "Ottoman Photography"]
            ),
            
            'dolmabahce_palace': MuseumData(
                id='dolmabahce_palace',
                name='Dolmabahçe Palace',
                category=MuseumCategory.PALACE,
                district='beşiktaş',
                address='Vişnezade, Dolmabahçe Cd., 34357 Beşiktaş/İstanbul',
                coordinates=(41.0391, 29.0066),
                price_tl=120.0,
                price_category=PriceCategory.MODERATE,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("09:00", "16:00"),
                    wednesday=("09:00", "16:00"),
                    thursday=("09:00", "16:00"),
                    friday=("09:00", "16:00"),
                    saturday=("09:00", "16:00"),
                    sunday=("09:00", "16:00")
                ),
                description="19th-century Ottoman palace blending European and Ottoman architecture, Atatürk's final residence.",
                highlights=[
                    "Crystal Staircase and Ballroom",
                    "Ceremonial Hall with massive chandelier",
                    "Harem section (separate ticket)",
                    "Atatürk's death room and memorabilia",
                    "Waterfront palace gardens"
                ],
                nearby_attractions=["Beşiktaş Square", "Yıldız Park", "Naval Museum"],
                nearby_restaurants=["Lacivert", "Sunset Grill & Bar", "Banyan"],
                accessibility=False,
                family_friendly=True,
                photography_allowed=False,
                guided_tours=True,
                audio_guide=True,
                gift_shop=True,
                cafe=True,
                estimated_visit_duration=120,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="high",
                languages=["turkish", "english", "german"],
                website="https://www.millisaraylar.gov.tr",
                phone="+90 212 236 9000",
                rating=4.5,
                reviews_count=87300,
                keywords=["ottoman", "palace", "atatürk", "crystal", "ballroom", "harem"],
                seasonal_info="Book in advance, beautiful Bosphorus views"
            ),
            
            'rahmi_koc_museum': MuseumData(
                id='rahmi_koc_museum',
                name='Rahmi M. Koç Museum',
                category=MuseumCategory.SCIENCE_TECHNOLOGY,
                district='beyoğlu',
                address='Hasköy Cd. No:5, 34445 Beyoğlu/İstanbul',
                coordinates=(41.0472, 28.9472),
                price_tl=40.0,
                price_category=PriceCategory.BUDGET,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("10:00", "17:00"),
                    wednesday=("10:00", "17:00"),
                    thursday=("10:00", "17:00"),
                    friday=("10:00", "17:00"),
                    saturday=("10:00", "19:00"),
                    sunday=("10:00", "19:00")
                ),
                description="Turkey's first major museum dedicated to history of transport, industry and communications.",
                highlights=[
                    "Historic submarines and ships",
                    "Vintage cars and motorcycles",
                    "Aviation and railway exhibits",
                    "Interactive science demonstrations",
                    "Maritime heritage displays"
                ],
                nearby_attractions=["Golden Horn", "Pierre Loti Hill", "Eyüp Sultan Mosque"],
                nearby_restaurants=["Café du Levant", "Pierre Loti Café", "Sur Balık"],
                accessibility=True,
                family_friendly=True,
                photography_allowed=True,
                guided_tours=True,
                audio_guide=False,
                gift_shop=True,
                cafe=True,
                estimated_visit_duration=150,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="low",
                languages=["turkish", "english"],
                website="https://www.rmk-museum.org.tr",
                phone="+90 212 369 6600",
                rating=4.4,
                reviews_count=15600,
                keywords=["technology", "transport", "submarine", "cars", "interactive", "family"],
                seasonal_info="Great for families, hands-on exhibits year-round"
            ),
            
            'islamic_arts_museum': MuseumData(
                id='islamic_arts_museum',
                name='Turkish and Islamic Arts Museum',
                category=MuseumCategory.RELIGIOUS,
                district='sultanahmet',
                address='At Meydanı No:46, 34122 Fatih/İstanbul',
                coordinates=(41.0055, 28.9717),
                price_tl=50.0,
                price_category=PriceCategory.BUDGET,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("09:00", "19:00"),
                    wednesday=("09:00", "19:00"),
                    thursday=("09:00", "19:00"),
                    friday=("09:00", "19:00"),
                    saturday=("09:00", "19:00"),
                    sunday=("09:00", "19:00")
                ),
                description="World's finest collection of Islamic calligraphy, carpets, and Ottoman decorative arts in historic palace.",
                highlights=[
                    "World's best carpet collection",
                    "Islamic calligraphy masterpieces",
                    "Ottoman miniatures and manuscripts",
                    "Traditional Turkish ethnographic exhibits",
                    "Beautiful Seljuk and Ottoman artifacts"
                ],
                nearby_attractions=["Blue Mosque", "Hippodrome", "Grand Bazaar"],
                nearby_restaurants=["Deraliye", "Sarniç Restaurant", "Olive Anatolian Restaurant"],
                accessibility=True,
                family_friendly=True,
                photography_allowed=False,
                guided_tours=True,
                audio_guide=True,
                gift_shop=True,
                cafe=False,
                estimated_visit_duration=90,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="moderate",
                languages=["turkish", "english", "arabic"],
                website="https://muze.gov.tr/turkislam-eserleri-muzesi",
                phone="+90 212 518 1805",
                rating=4.3,
                reviews_count=9800,
                keywords=["islamic", "carpets", "calligraphy", "ottoman", "seljuk", "arts"],
                seasonal_info="Peaceful alternative to crowded attractions nearby"
            ),
            
            'military_museum': MuseumData(
                id='military_museum',
                name='Military Museum',
                category=MuseumCategory.MILITARY,
                district='şişli',
                address='Vali Konağı Cd. No:2, 34367 Şişli/İstanbul',
                coordinates=(41.0485, 28.9872),
                price_tl=25.0,
                price_category=PriceCategory.BUDGET,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("closed", "closed"),
                    wednesday=("09:00", "17:00"),
                    thursday=("09:00", "17:00"),
                    friday=("09:00", "17:00"),
                    saturday=("09:00", "17:00"),
                    sunday=("09:00", "17:00")
                ),
                description="Comprehensive military history from Ottoman Empire to modern Turkey, featuring weapons, uniforms, and campaigns.",
                highlights=[
                    "Ottoman military history",
                    "Mehter (Ottoman military band) performances",
                    "Historic weapons and armor collection",
                    "Gallipoli and WWI exhibits",
                    "Interactive battle simulations"
                ],
                nearby_attractions=["Taksim Square", "İstiklal Street", "Harbiye Congress Valley"],
                nearby_restaurants=["360 Istanbul", "Zuma", "Nicole Restaurant"],
                accessibility=True,
                family_friendly=True,
                photography_allowed=True,
                guided_tours=True,
                audio_guide=False,
                gift_shop=True,
                cafe=False,
                estimated_visit_duration=120,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="low",
                languages=["turkish", "english"],
                website="https://www.tsk.tr",
                phone="+90 212 233 2720",
                rating=4.1,
                reviews_count=7200,
                keywords=["military", "ottoman", "mehter", "weapons", "gallipoli", "history"],
                seasonal_info="Features live Mehter performances on weekends"
            ),
            
            'sadberk_hanim_museum': MuseumData(
                id='sadberk_hanim_museum',
                name='Sadberk Hanım Museum',
                category=MuseumCategory.CULTURAL,
                district='sarıyer',
                address='Piyasa Cd. No:27-29, 34470 Sarıyer/İstanbul',
                coordinates=(41.1089, 29.0475),
                price_tl=35.0,
                price_category=PriceCategory.BUDGET,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("closed", "closed"),
                    wednesday=("10:00", "17:00"),
                    thursday=("10:00", "17:00"),
                    friday=("10:00", "17:00"),
                    saturday=("10:00", "17:00"),
                    sunday=("10:00", "17:00")
                ),
                description="Turkey's first private museum in historic Bosphorus waterfront mansion, showcasing Turkish decorative arts.",
                highlights=[
                    "Ottoman decorative arts",
                    "Traditional Turkish costumes",
                    "Archaeological artifacts",
                    "Beautiful Bosphorus mansion setting",
                    "Intimate museum experience"
                ],
                nearby_attractions=["Emirgan Park", "Büyükdere Grove", "Bosphorus villages"],
                nearby_restaurants=["Kordon Restaurant", "Tugra Restaurant", "Poseidon"],
                accessibility=False,
                family_friendly=True,
                photography_allowed=False,
                guided_tours=True,
                audio_guide=False,
                gift_shop=True,
                cafe=True,
                estimated_visit_duration=75,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="low",
                languages=["turkish", "english"],
                website="https://www.sadberkhanimmuzesi.org.tr",
                phone="+90 212 242 3813",
                rating=4.2,
                reviews_count=3400,
                keywords=["decorative", "ottoman", "costumes", "bosphorus", "private", "mansion"],
                seasonal_info="Beautiful gardens, perfect for spring visits"
            ),
            
            'naval_museum': MuseumData(
                id='naval_museum',
                name='Naval Museum',
                category=MuseumCategory.MARITIME,
                district='beşiktaş',
                address='Sinanpaşa, Beşiktaş Cd. No:15, 34353 Beşiktaş/İstanbul',
                coordinates=(41.0426, 29.0085),
                price_tl=30.0,
                price_category=PriceCategory.BUDGET,
                opening_hours=OpeningHours(
                    monday=("closed", "closed"),
                    tuesday=("closed", "closed"),
                    wednesday=("09:00", "17:00"),
                    thursday=("09:00", "17:00"),
                    friday=("09:00", "17:00"),
                    saturday=("09:00", "17:00"),
                    sunday=("09:00", "17:00")
                ),
                description="Maritime history of Ottoman Empire and Turkish Navy, featuring historic boats and naval artifacts.",
                highlights=[
                    "Historic Ottoman naval vessels",
                    "Imperial boats and kayiks",
                    "Naval battle dioramas",
                    "Maritime navigation instruments",
                    "Admiral uniforms and medals"
                ],
                nearby_attractions=["Dolmabahçe Palace", "Beşiktaş Pier", "Yıldız Park"],
                nearby_restaurants=["Lacivert", "Wolfgang's Steakhouse", "Feriye Palace"],
                accessibility=True,
                family_friendly=True,
                photography_allowed=True,
                guided_tours=True,
                audio_guide=False,
                gift_shop=True,
                cafe=False,
                estimated_visit_duration=90,
                best_time_to_visit=["morning", "afternoon"],
                crowd_level="low",
                languages=["turkish", "english"],
                website="https://www.dzkk.tsk.tr",
                phone="+90 212 327 4345",
                rating=4.0,
                reviews_count=5600,
                keywords=["naval", "maritime", "boats", "ottoman", "navy", "bosphorus"],
                seasonal_info="Less crowded than major museums, great for maritime enthusiasts"
            )
        }
        
        return museums
    
    def get_museums_by_location(self, user_location: Tuple[float, float], radius_km: float = 5.0) -> List[MuseumData]:
        """Get museums near user location"""
        nearby_museums = []
        
        for museum in self.museums.values():
            distance = self._calculate_distance(user_location, museum.coordinates)
            if distance <= radius_km:
                nearby_museums.append((museum, distance))
        
        # Sort by distance
        nearby_museums.sort(key=lambda x: x[1])
        return [museum for museum, _ in nearby_museums]
    
    def get_museums_by_district(self, district: str) -> List[MuseumData]:
        """Get museums in specific district"""
        district_lower = district.lower()
        return [museum for museum in self.museums.values() 
                if museum.district.lower() == district_lower]
    
    def get_museums_by_category(self, category: MuseumCategory) -> List[MuseumData]:
        """Get museums by category"""
        return [museum for museum in self.museums.values() 
                if museum.category == category]
    
    def get_museums_by_price_range(self, max_price: float) -> List[MuseumData]:
        """Get museums within price range"""
        return [museum for museum in self.museums.values() 
                if museum.price_tl <= max_price]
    
    def get_open_museums(self, day_of_week: int, current_time: time) -> List[MuseumData]:
        """Get museums currently open"""
        return [museum for museum in self.museums.values() 
                if museum.opening_hours.is_open(day_of_week, current_time)]
    
    def search_museums(self, query: str, user_location: Optional[Tuple[float, float]] = None) -> List[MuseumData]:
        """Search museums by name, category, or keywords"""
        query_lower = query.lower()
        results = []
        
        for museum in self.museums.values():
            # Check name, description, highlights, keywords
            if (query_lower in museum.name.lower() or
                query_lower in museum.description.lower() or
                any(query_lower in highlight.lower() for highlight in museum.highlights) or
                any(query_lower in keyword.lower() for keyword in museum.keywords)):
                results.append(museum)
        
        # Sort by proximity if location provided
        if user_location and results:
            results = [(museum, self._calculate_distance(user_location, museum.coordinates)) 
                      for museum in results]
            results.sort(key=lambda x: x[1])
            results = [museum for museum, _ in results]
        
        return results[:10]  # Return top 10 results
    
    def get_museum_recommendations(self, user_profile: dict, user_location: Optional[Tuple[float, float]] = None) -> List[MuseumData]:
        """Get ML-powered museum recommendations based on user profile"""
        
        # Get all museums
        all_museums = list(self.museums.values())
        
        # Apply filters based on user preferences
        filtered_museums = all_museums
        
        # Filter by interests
        if user_profile.get('interests'):
            interests = [interest.lower() for interest in user_profile['interests']]
            if 'history' in interests:
                filtered_museums = [m for m in filtered_museums if m.category in [
                    MuseumCategory.HISTORICAL, MuseumCategory.ARCHAEOLOGICAL, MuseumCategory.PALACE]]
            elif 'art' in interests:
                filtered_museums = [m for m in filtered_museums if m.category in [
                    MuseumCategory.ART_MODERN, MuseumCategory.ART_CLASSICAL]]
            elif 'culture' in interests:
                filtered_museums = [m for m in filtered_museums if m.category in [
                    MuseumCategory.CULTURAL, MuseumCategory.RELIGIOUS, MuseumCategory.ETHNOGRAPHIC]]
            elif 'family' in interests:
                filtered_museums = [m for m in filtered_museums if m.family_friendly]
        
        # Filter by budget
        if user_profile.get('budget_range'):
            budget = user_profile['budget_range'].lower()
            if budget == 'budget':
                filtered_museums = [m for m in filtered_museums if m.price_tl <= 50]
            elif budget == 'moderate':
                filtered_museums = [m for m in filtered_museums if m.price_tl <= 150]
        
        # Filter by accessibility needs
        if user_profile.get('accessibility_needs'):
            filtered_museums = [m for m in filtered_museums if m.accessibility]
        
        # Sort by location if provided
        if user_location:
            filtered_museums = [(museum, self._calculate_distance(user_location, museum.coordinates)) 
                               for museum in filtered_museums]
            filtered_museums.sort(key=lambda x: x[1])
            filtered_museums = [museum for museum, _ in filtered_museums]
        else:
            # Sort by rating and reviews
            filtered_museums.sort(key=lambda x: (x.rating, x.reviews_count), reverse=True)
        
        return filtered_museums[:8]  # Return top 8 recommendations
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two GPS coordinates in kilometers"""
        import math
        
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        
        return c * r

def process_museum_query_enhanced(message: str, user_profile: dict, current_time: datetime, 
                                user_location: Optional[Tuple[float, float]] = None) -> str:
    """Enhanced museum query processing with GPS and ML integration"""
    
    try:
        # Initialize museum system
        museum_system = IstanbulMuseumSystem()
        
        # Extract query parameters
        message_lower = message.lower()
        
        # Determine query type and extract parameters
        query_type = "general"
        specific_museum = None
        category_filter = None
        district_filter = None
        budget_filter = None
        
        # Check for specific museum requests
        for museum_id, museum in museum_system.museums.items():
            if any(word in message_lower for word in museum.name.lower().split()):
                specific_museum = museum
                query_type = "specific"
                break
        
        # Check for category filters
        if any(word in message_lower for word in ['history', 'historical', 'ancient']):
            category_filter = [MuseumCategory.HISTORICAL, MuseumCategory.ARCHAEOLOGICAL]
        elif any(word in message_lower for word in ['art', 'painting', 'modern']):
            category_filter = [MuseumCategory.ART_MODERN, MuseumCategory.ART_CLASSICAL]
        elif any(word in message_lower for word in ['palace', 'ottoman', 'sultan']):
            category_filter = [MuseumCategory.PALACE]
        elif any(word in message_lower for word in ['military', 'war', 'battle']):
            category_filter = [MuseumCategory.MILITARY]
        elif any(word in message_lower for word in ['islamic', 'religious', 'mosque']):
            category_filter = [MuseumCategory.RELIGIOUS]
        
        # Check for district filters
        districts = ['sultanahmet', 'beyoğlu', 'beyoglu', 'beşiktaş', 'besiktas', 'kadıköy', 'kadikoy']
        for district in districts:
            if district in message_lower:
                district_filter = district
                break
        
        # Check for budget constraints
        if any(word in message_lower for word in ['free', 'cheap', 'budget']):
            budget_filter = 50.0
        elif any(word in message_lower for word in ['expensive', 'premium', 'luxury']):
            budget_filter = None
        
        # Check for time-sensitive queries
        check_open_now = any(phrase in message_lower for phrase in ['open now', 'currently open', 'open today'])
        
        # Get recommendations based on query type
        if specific_museum:
            recommendations = [specific_museum]
        elif query_type == "general":
            # Use ML-powered recommendations
            recommendations = museum_system.get_museum_recommendations(user_profile, user_location)
            
            # Apply additional filters
            if category_filter:
                recommendations = [m for m in recommendations if m.category in category_filter]
            if district_filter:
                recommendations = [m for m in recommendations if m.district.lower() == district_filter]
            if budget_filter:
                recommendations = [m for m in recommendations if m.price_tl <= budget_filter]
            if check_open_now:
                day_of_week = current_time.weekday()
                current_time_obj = current_time.time()
                recommendations = [m for m in recommendations if m.opening_hours.is_open(day_of_week, current_time_obj)]
        else:
            # Search-based recommendations
            recommendations = museum_system.search_museums(message, user_location)
        
        if not recommendations:
            return _generate_no_museums_response(message, user_location)
        
        # Generate response
        return _generate_museum_response(recommendations[:5], user_profile, current_time, user_location, message)
        
    except Exception as e:
        logger.error(f"Error processing museum query: {e}")
        return "🏛️ I'd love to help you discover Istanbul's amazing museums! Could you tell me more about what type of museums interest you - history, art, palaces, or something specific?"

def _generate_no_museums_response(message: str, user_location: Optional[Tuple[float, float]]) -> str:
    """Generate response when no museums match the criteria"""
    
    response = "🏛️ **Museum Search Results**\n\n"
    response += "I couldn't find museums matching your exact criteria, but here are some alternatives:\n\n"
    
    # Initialize system for general recommendations
    museum_system = IstanbulMuseumSystem()
    
    # Get top-rated museums
    all_museums = list(museum_system.museums.values())
    all_museums.sort(key=lambda x: x.rating, reverse=True)
    
    response += "🌟 **Top-Rated Istanbul Museums:**\n"
    for i, museum in enumerate(all_museums[:3], 1):
        price_info = "Free" if museum.price_tl == 0 else f"₺{museum.price_tl}"
        response += f"**{i}. {museum.name}**\n"
        response += f"   📍 {museum.district.title()} • {price_info} • ⭐ {museum.rating}/5\n"
        response += f"   {museum.description[:100]}...\n\n"
    
    response += "💡 **Try asking:**\n"
    response += "• 'Show me art museums in Beyoğlu'\n"
    response += "• 'Free museums near me'\n"
    response += "• 'What museums are open now?'\n"
    response += "• 'Historical museums in Sultanahmet'\n"
    
    return response

def _generate_museum_response(recommendations: List[MuseumData], user_profile: dict, 
                            current_time: datetime, user_location: Optional[Tuple[float, float]], 
                            original_query: str) -> str:
    """Generate comprehensive museum recommendation response"""
    
    # Initialize museum system for distance calculations
    museum_system = IstanbulMuseumSystem()
    
    response = "🏛️ **Istanbul Museum Recommendations**\n\n"
    
    # Add personalization context
    if user_location:
        response += "📍 **Based on your location and preferences:**\n\n"
    else:
        response += "🎯 **Personalized museum recommendations:**\n\n"
    
    # Current time context
    day_of_week = current_time.weekday()
    current_time_obj = current_time.time()
    
    for i, museum in enumerate(recommendations, 1):
        # Calculate distance if location available
        distance_info = ""
        if user_location:
            distance = museum_system._calculate_distance(user_location, museum.coordinates)
            walking_time = int(distance * 12)  # Rough walking time estimate
            distance_info = f" • 🚶 {walking_time} min walk"
        
        # Check if open now
        is_open = museum.opening_hours.is_open(day_of_week, current_time_obj)
        today_hours = museum.opening_hours.get_today_hours(day_of_week)
        
        open_status = "🟢 Open now" if is_open else f"🔴 {today_hours}"
        
        # Price display
        price_display = "🆓 Free" if museum.price_tl == 0 else f"💰 ₺{museum.price_tl}"
        
        response += f"**{i}. {museum.name}**\n"
        response += f"📍 {museum.district.title()}{distance_info} • {price_display}\n"
        response += f"⏰ {open_status} • ⭐ {museum.rating}/5 ({museum.reviews_count:,} reviews)\n"
        response += f"🏛️ {museum.category.value.replace('_', ' ').title()}\n\n"
        
        response += f"**About:** {museum.description[:120]}...\n\n"
        
        # Highlights
        response += "✨ **Highlights:**\n"
        for highlight in museum.highlights[:3]:
            response += f"• {highlight}\n"
        
        # Practical info
        response += f"\n⏱️ **Visit Duration:** ~{museum.estimated_visit_duration} minutes\n"
        
        # Accessibility and amenities
        amenities = []
        if museum.accessibility:
            amenities.append("♿ Accessible")
        if museum.family_friendly:
            amenities.append("👨‍👩‍👧‍👦 Family-friendly")
        if museum.photography_allowed:
            amenities.append("📸 Photos allowed")
        if museum.guided_tours:
            amenities.append("🎯 Guided tours")
        if museum.audio_guide:
            amenities.append("🎧 Audio guide")
        if museum.gift_shop:
            amenities.append("🛍️ Gift shop")
        if museum.cafe:
            amenities.append("☕ Café")
        
        if amenities:
            response += f"🎯 **Amenities:** {' • '.join(amenities)}\n"
        
        # Nearby attractions
        if museum.nearby_attractions:
            response += f"🗺️ **Nearby:** {', '.join(museum.nearby_attractions[:3])}\n"
        
        if i < len(recommendations):
            response += "\n" + "─" * 50 + "\n\n"
    
    # Add helpful tips
    response += "\n💡 **Planning Tips:**\n"
    response += "• Book tickets online to skip queues at popular museums\n"
    response += "• Many museums are closed on Mondays\n"
    response += "• Student discounts available with valid ID\n"
    response += "• Museum Pass Istanbul available for multiple visits\n\n"
    
    # Add interactive options
    response += "🎯 **Need More Info?**\n"
    response += "• 'Tell me more about [museum name]' for details\n"
    response += "• 'Museums open on Monday' for day-specific options\n"
    response += "• 'Free museums in Istanbul' for budget options\n"
    response += "• 'Art museums near me' for category-specific search\n"
    
    return response

if __name__ == "__main__":
    # Test the museum system
    museum_system = IstanbulMuseumSystem()
    
    # Test user profile
    test_profile = {
        'interests': ['history', 'art'],
        'budget_range': 'moderate',
        'accessibility_needs': None
    }
    
    # Test location (Sultanahmet area)
    test_location = (41.0082, 28.9784)
    
    # Test query
    test_query = "Show me historical museums near me"
    current_time = datetime.now()
    
    response = process_museum_query_enhanced(test_query, test_profile, current_time, test_location)
    print(response)
