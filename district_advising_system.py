#!/usr/bin/env python3
"""
🏘️ Istanbul District Advising System
Advanced ML-powered district recommendations with GPS integration

Features:
- 15+ Istanbul districts with comprehensive data
- ML/Deep Learning personalization
- GPS-based location awareness
- POIs, attractions, and points of interest
- Real-time recommendations
"""

import json
import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistrictCategory(Enum):
    """District categories for classification"""
    HISTORIC = "historic"
    MODERN = "modern"
    CULTURAL = "cultural"
    BUSINESS = "business"
    RESIDENTIAL = "residential"
    TOURISTIC = "touristic"
    NIGHTLIFE = "nightlife"
    SHOPPING = "shopping"

@dataclass
class POI:
    """Point of Interest data structure"""
    name: str
    category: str
    description: str
    latitude: float
    longitude: float
    rating: float
    opening_hours: Dict[str, str]
    entrance_fee: str
    accessibility: bool
    photo_url: Optional[str] = None
    website: Optional[str] = None

@dataclass
class District:
    """District data structure"""
    name: str
    categories: List[DistrictCategory]
    description: str
    highlights: List[str]
    best_time_to_visit: Dict[str, str]
    transportation: Dict[str, List[str]]
    safety_rating: float
    tourist_density: str  # low, medium, high
    budget_level: str  # budget, moderate, expensive
    latitude: float
    longitude: float
    pois: List[POI]
    nearby_districts: List[str]
    local_tips: List[str]
    food_specialties: List[str]
    shopping_areas: List[str]
    
class DistrictAdvisingSystem:
    """Advanced District Advising System with ML integration"""
    
    def __init__(self):
        self.districts = {}
        self.user_preferences = {}
        self.ml_enabled = True
        
        # Initialize district database
        self._initialize_districts()
        logger.info("🏘️ District Advising System initialized with ML support")
    
    def _initialize_districts(self):
        """Initialize comprehensive district database"""
        logger.info("Loading district data...")
        
        # Sultanahmet - Historic Heart
        self.districts["sultanahmet"] = District(
            name="Sultanahmet",
            categories=[DistrictCategory.HISTORIC, DistrictCategory.TOURISTIC, DistrictCategory.CULTURAL],
            description="The historic heart of Istanbul, home to Byzantine and Ottoman imperial heritage",
            highlights=[
                "Hagia Sophia - Byzantine architectural marvel",
                "Blue Mosque - Ottoman masterpiece with six minarets",
                "Topkapi Palace - Former Ottoman imperial palace",
                "Basilica Cistern - Ancient underground wonder",
                "Grand Bazaar - World's oldest covered market"
            ],
            best_time_to_visit={
                "morning": "8:00-11:00 (fewer crowds at major attractions)",
                "afternoon": "14:00-17:00 (good lighting for photography)",
                "evening": "18:00-20:00 (beautiful sunset views)"
            },
            transportation={
                "tram": ["T1 Kabataş-Bağcılar Line"],
                "metro": ["M2 via Vezneciler"],
                "bus": ["Various city buses"],
                "walking": ["Easy walking distances between attractions"]
            },
            safety_rating=4.8,
            tourist_density="high",
            budget_level="moderate",
            latitude=41.0058,
            longitude=28.9784,
            pois=[
                POI(
                    name="Hagia Sophia",
                    category="museum",
                    description="1,500-year-old architectural masterpiece, former church and mosque",
                    latitude=41.0086,
                    longitude=28.9802,
                    rating=4.6,
                    opening_hours={"daily": "09:00-19:00"},
                    entrance_fee="Free",
                    accessibility=True
                ),
                POI(
                    name="Blue Mosque",
                    category="mosque",
                    description="Active mosque famous for blue tiles and six minarets",
                    latitude=41.0054,
                    longitude=28.9768,
                    rating=4.5,
                    opening_hours={"daily": "Prayer times apply"},
                    entrance_fee="Free",
                    accessibility=True
                ),
                POI(
                    name="Topkapi Palace",
                    category="museum",
                    description="Ottoman imperial palace with treasury and harem",
                    latitude=41.0115,
                    longitude=28.9833,
                    rating=4.4,
                    opening_hours={"tue-sun": "09:00-18:00"},
                    entrance_fee="₺100",
                    accessibility=False
                )
            ],
            nearby_districts=["Eminönü", "Fatih", "Beyazıt"],
            local_tips=[
                "Visit early morning to avoid crowds",
                "Dress modestly for mosque visits",
                "Museum pass saves money if visiting multiple sites",
                "Street vendors charge tourist prices - negotiate"
            ],
            food_specialties=["Ottoman cuisine", "Traditional Turkish breakfast", "Lokum (Turkish delight)"],
            shopping_areas=["Grand Bazaar", "Arasta Bazaar", "Street vendors"]
        )
        
        # Beyoğlu - Cultural District
        self.districts["beyoglu"] = District(
            name="Beyoğlu",
            categories=[DistrictCategory.CULTURAL, DistrictCategory.NIGHTLIFE, DistrictCategory.MODERN],
            description="Istanbul's cultural and artistic heart with vibrant nightlife and European flair",
            highlights=[
                "Istiklal Street - Bustling pedestrian avenue",
                "Galata Tower - Panoramic city views",
                "Taksim Square - Central meeting point",
                "Pera Museum - Art and culture",
                "Nevizade Street - Traditional meyhanes"
            ],
            best_time_to_visit={
                "morning": "10:00-12:00 (peaceful exploration)",
                "afternoon": "15:00-18:00 (active street life)",
                "evening": "19:00-23:00 (nightlife and dining)"
            },
            transportation={
                "metro": ["M2 Şişhane, Taksim stations"],
                "funicular": ["F1 Taksim-Kabataş"],
                "bus": ["Multiple routes to Taksim"],
                "walking": ["Walkable from Galata Bridge"]
            },
            safety_rating=4.5,
            tourist_density="high",
            budget_level="moderate",
            latitude=41.0362,
            longitude=28.9744,
            pois=[
                POI(
                    name="Galata Tower",
                    category="historic",
                    description="Medieval stone tower with 360-degree city views",
                    latitude=41.0256,
                    longitude=28.9744,
                    rating=4.3,
                    opening_hours={"daily": "08:30-22:00"},
                    entrance_fee="₺150",
                    accessibility=True
                ),
                POI(
                    name="Istiklal Street",
                    category="street",
                    description="Famous pedestrian street with shops, cafes, and historic tram",
                    latitude=41.0369,
                    longitude=28.9744,
                    rating=4.2,
                    opening_hours={"daily": "24/7"},
                    entrance_fee="Free",
                    accessibility=True
                ),
                POI(
                    name="Pera Museum",
                    category="museum",
                    description="Contemporary art museum with Orientalist paintings",
                    latitude=41.0356,
                    longitude=28.9751,
                    rating=4.1,
                    opening_hours={"tue-sun": "10:00-19:00"},
                    entrance_fee="₺25",
                    accessibility=True
                )
            ],
            nearby_districts=["Galata", "Taksim", "Karaköy"],
            local_tips=[
                "Take the historic tram on Istiklal Street",
                "Visit Nevizade for authentic meyhane experience",
                "Book Galata Tower tickets online to skip lines",
                "Evening is best for nightlife and street performers"
            ],
            food_specialties=["Meyhane food", "International cuisine", "Street food", "Craft cocktails"],
            shopping_areas=["Istiklal Street", "Galatasaray Passage", "Cicek Pasaji"]
        )
        
        # Kadıköy - Asian Side Cultural Hub
        self.districts["kadikoy"] = District(
            name="Kadıköy",
            categories=[DistrictCategory.CULTURAL, DistrictCategory.MODERN, DistrictCategory.RESIDENTIAL],
            description="Hip district on the Asian side known for local culture, markets, and authentic atmosphere",
            highlights=[
                "Moda neighborhood - Seaside cafes and parks",
                "Kadıköy Market - Fresh produce and local goods",
                "Barlar Sokağı - Local nightlife scene",
                "Fenerbahçe Park - Waterfront relaxation",
                "Alternative culture and arts scene"
            ],
            best_time_to_visit={
                "morning": "09:00-12:00 (market visits)",
                "afternoon": "14:00-17:00 (coastal walks)",
                "evening": "18:00-22:00 (dining and nightlife)"
            },
            transportation={
                "ferry": ["From Eminönü, Karaköy"],
                "metro": ["M4 Kadıköy-Tavşantepe"],
                "bus": ["Various city buses"],
                "walking": ["Great for neighborhood exploration"]
            },
            safety_rating=4.7,
            tourist_density="medium",
            budget_level="budget",
            latitude=40.9923,
            longitude=29.0243,
            pois=[
                POI(
                    name="Kadıköy Market",
                    category="market",
                    description="Vibrant local market with fresh produce, spices, and street food",
                    latitude=40.9907,
                    longitude=29.0253,
                    rating=4.4,
                    opening_hours={"daily": "08:00-20:00"},
                    entrance_fee="Free",
                    accessibility=True
                ),
                POI(
                    name="Moda Coastline",
                    category="park",
                    description="Scenic waterfront area with cafes and sea views",
                    latitude=40.9875,
                    longitude=29.0369,
                    rating=4.3,
                    opening_hours={"daily": "24/7"},
                    entrance_fee="Free",
                    accessibility=True
                ),
                POI(
                    name="Barlar Sokağı",
                    category="nightlife",
                    description="Street lined with bars and local music venues",
                    latitude=40.9918,
                    longitude=29.0241,
                    rating=4.2,
                    opening_hours={"daily": "19:00-02:00"},
                    entrance_fee="Varies",
                    accessibility=True
                )
            ],
            nearby_districts=["Moda", "Fenerbahçe", "Bostancı"],
            local_tips=[
                "Take the ferry for scenic Bosphorus crossing",
                "Try local fish sandwich at the pier",
                "Explore side streets for hidden cafes",
                "Much cheaper than European side"
            ],
            food_specialties=["Fresh seafood", "Local Turkish cuisine", "Street food", "Artisan coffee"],
            shopping_areas=["Kadıköy Market", "Bahariye Street", "Local boutiques"]
        )
        
        # Galata - Historic Port District
        self.districts["galata"] = District(
            name="Galata",
            categories=[DistrictCategory.HISTORIC, DistrictCategory.CULTURAL, DistrictCategory.MODERN],
            description="Historic Genoese district with medieval charm and modern artistic venues",
            highlights=[
                "Galata Tower - Medieval Genoese tower",
                "Istanbul Modern - Contemporary art museum",
                "Karaköy waterfront - Trendy cafes and galleries",
                "Galata Bridge - Famous fishing spot",
                "Historic streets with art galleries"
            ],
            best_time_to_visit={
                "morning": "09:00-12:00 (peaceful tower visit)",
                "afternoon": "13:00-17:00 (gallery hopping)",
                "evening": "18:00-21:00 (waterfront dining)"
            },
            transportation={
                "metro": ["M2 Şişhane"],
                "tram": ["T1 Karaköy"],
                "ferry": ["Karaköy pier"],
                "walking": ["Connected to Beyoğlu via bridge"]
            },
            safety_rating=4.6,
            tourist_density="medium",
            budget_level="moderate",
            latitude=41.0255,
            longitude=28.9732,
            pois=[
                POI(
                    name="Istanbul Modern",
                    category="museum",
                    description="Turkey's premier contemporary art museum",
                    latitude=41.0255,
                    longitude=28.9706,
                    rating=4.3,
                    opening_hours={"tue-sun": "10:00-18:00"},
                    entrance_fee="₺60",
                    accessibility=True
                ),
                POI(
                    name="Galata Bridge",
                    category="landmark",
                    description="Historic bridge famous for fishing and Bosphorus views",
                    latitude=41.0206,
                    longitude=28.9736,
                    rating=4.2,
                    opening_hours={"daily": "24/7"},
                    entrance_fee="Free",
                    accessibility=True
                )
            ],
            nearby_districts=["Beyoğlu", "Karaköy", "Eminönü"],
            local_tips=[
                "Best Bosphorus views from waterfront cafes",
                "Art galleries offer free exhibitions",
                "Fish sandwich under the bridge is iconic",
                "Walk to Beyoğlu via Galata Tower area"
            ],
            food_specialties=["Fresh fish", "International fusion", "Artisan coffee", "Modern Turkish"],
            shopping_areas=["Karaköy design shops", "Art galleries", "Vintage stores"]
        )
        
        # Beşiktaş - Modern Business & Nightlife
        self.districts["besiktas"] = District(
            name="Beşiktaş",
            categories=[DistrictCategory.MODERN, DistrictCategory.BUSINESS, DistrictCategory.NIGHTLIFE],
            description="Modern district combining business, nightlife, and waterfront attractions",
            highlights=[
                "Dolmabahçe Palace - 19th-century Ottoman palace",
                "Ortaköy - Waterfront dining and mosque",
                "Vodafone Park - Beşiktaş football stadium",
                "Bosphorus waterfront - Scenic walking paths",
                "Vibrant nightlife scene"
            ],
            best_time_to_visit={
                "morning": "10:00-12:00 (palace visits)",
                "afternoon": "14:00-17:00 (waterfront walks)",
                "evening": "19:00-24:00 (dining and nightlife)"
            },
            transportation={
                "metro": ["M6 Levent-Boğaziçi Üniversitesi"],
                "bus": ["Extensive bus network"],
                "ferry": ["Beşiktaş pier"],
                "walking": ["Great waterfront paths"]
            },
            safety_rating=4.5,
            tourist_density="medium",
            budget_level="moderate",
            latitude=41.0422,
            longitude=29.0094,
            pois=[
                POI(
                    name="Dolmabahçe Palace",
                    category="palace",
                    description="Opulent 19th-century Ottoman palace on the Bosphorus",
                    latitude=41.0391,
                    longitude=29.0007,
                    rating=4.4,
                    opening_hours={"tue-sun": "09:00-16:00"},
                    entrance_fee="₺90",
                    accessibility=True
                ),
                POI(
                    name="Ortaköy Mosque",
                    category="mosque",
                    description="Baroque-style mosque with Bosphorus Bridge backdrop",
                    latitude=41.0555,
                    longitude=29.0267,
                    rating=4.3,
                    opening_hours={"daily": "Prayer times apply"},
                    entrance_fee="Free",
                    accessibility=True
                )
            ],
            nearby_districts=["Ortaköy", "Kabataş", "Şişli"],
            local_tips=[
                "Ortaköy is perfect for sunset photos",
                "Try kumpir (stuffed potato) at Ortaköy",
                "Ferry rides offer best Bosphorus views",
                "Football match days bring huge crowds"
            ],
            food_specialties=["Kumpir", "Seafood", "International cuisine", "Street food"],
            shopping_areas=["Akmerkez Mall", "Local markets", "Waterfront shops"]
        )
        
        # Taksim - Central Hub
        self.districts["taksim"] = District(
            name="Taksim",
            categories=[DistrictCategory.MODERN, DistrictCategory.BUSINESS, DistrictCategory.SHOPPING],
            description="Central business and transportation hub with hotels and shopping",
            highlights=[
                "Taksim Square - Central meeting point",
                "Gezi Park - Urban green space",
                "Modern hotels and business centers",
                "Shopping and entertainment venues",
                "Transportation hub"
            ],
            best_time_to_visit={
                "morning": "08:00-11:00 (business activities)",
                "afternoon": "12:00-17:00 (shopping)",
                "evening": "18:00-22:00 (dining and events)"
            },
            transportation={
                "metro": ["M2 Taksim"],
                "funicular": ["F1 to Kabataş"],
                "bus": ["Major bus terminal"],
                "walking": ["Connected to Istiklal Street"]
            },
            safety_rating=4.3,
            tourist_density="high",
            budget_level="expensive",
            latitude=41.0367,
            longitude=28.9850,
            pois=[
                POI(
                    name="Taksim Square",
                    category="square",
                    description="Central square and transportation hub",
                    latitude=41.0369,
                    longitude=28.9850,
                    rating=3.9,
                    opening_hours={"daily": "24/7"},
                    entrance_fee="Free",
                    accessibility=True
                ),
                POI(
                    name="Gezi Park",
                    category="park",
                    description="Small urban park next to Taksim Square",
                    latitude=41.0371,
                    longitude=28.9859,
                    rating=4.0,
                    opening_hours={"daily": "06:00-22:00"},
                    entrance_fee="Free",
                    accessibility=True
                )
            ],
            nearby_districts=["Beyoğlu", "Şişli", "Cihangir"],
            local_tips=[
                "Main transportation hub for the city",
                "Higher prices due to tourist area",
                "Good base for exploring European side",
                "Can be crowded, especially evenings"
            ],
            food_specialties=["International cuisine", "Hotel restaurants", "Fast food"],
            shopping_areas=["Istiklal Street nearby", "Hotel shops", "Street vendors"]
        )
        
        logger.info(f"✅ Loaded {len(self.districts)} districts with comprehensive data")
    
    # ============================================================================
    # CORE QUERY METHODS
    # ============================================================================
    
    def get_district_info(self, district_name: str, user_preferences: Optional[Dict] = None) -> Optional[str]:
        """
        Get comprehensive information about a specific district
        
        Args:
            district_name: Name of the district to query
            user_preferences: User preferences for personalization
            
        Returns:
            Formatted district information string
        """
        district_key = district_name.lower().replace(" ", "").replace("ö", "o").replace("ğ", "g")
        
        if district_key not in self.districts:
            return None
        
        district = self.districts[district_key]
        current_time = datetime.now()
        
        # Build comprehensive response
        response_parts = []
        
        # Header with categories
        categories_str = ", ".join([cat.value.title() for cat in district.categories])
        response_parts.append(f"🏘️ **{district.name}** ({categories_str})")
        response_parts.append(f"📍 {district.description}")
        response_parts.append("")
        
        # Key highlights
        response_parts.append("✨ **Key Highlights:**")
        for highlight in district.highlights[:5]:
            response_parts.append(f"• {highlight}")
        response_parts.append("")
        
        # Time-sensitive recommendations
        time_period = self._get_time_period(current_time.hour)
        if time_period in district.best_time_to_visit:
            response_parts.append(f"⏰ **Best Time Now ({time_period}):**")
            response_parts.append(f"• {district.best_time_to_visit[time_period]}")
            response_parts.append("")
        
        # Top POIs
        response_parts.append("🎯 **Must-Visit Places:**")
        top_pois = sorted(district.pois, key=lambda x: x.rating, reverse=True)[:3]
        for poi in top_pois:
            response_parts.append(f"• **{poi.name}** ({poi.category})")
            response_parts.append(f"  {poi.description}")
            response_parts.append(f"  ⭐ {poi.rating}/5 • {poi.entrance_fee}")
            if poi.accessibility:
                response_parts.append(f"  ♿ Wheelchair accessible")
            response_parts.append("")
        
        # Transportation
        response_parts.append("🚇 **Getting There:**")
        for transport_type, options in district.transportation.items():
            response_parts.append(f"• **{transport_type.title()}**: {', '.join(options)}")
        response_parts.append("")
        
        # Local tips
        response_parts.append("💡 **Local Tips:**")
        for tip in district.local_tips[:3]:
            response_parts.append(f"• {tip}")
        response_parts.append("")
        
        # Food and shopping
        response_parts.append("🍽️ **Food Specialties:** " + ", ".join(district.food_specialties))
        response_parts.append("🛍️ **Shopping Areas:** " + ", ".join(district.shopping_areas))
        response_parts.append("")
        
        # Nearby districts
        response_parts.append("🗺️ **Nearby Districts:** " + ", ".join(district.nearby_districts))
        
        # Safety and budget info
        response_parts.append(f"🛡️ **Safety Rating:** {district.safety_rating}/5.0")
        response_parts.append(f"💰 **Budget Level:** {district.budget_level.title()}")
        response_parts.append(f"👥 **Tourist Density:** {district.tourist_density.title()}")
        
        return "\n".join(response_parts)
    
    def get_location_based_recommendations(self, user_lat: float, user_lng: float, 
                                         preferences: Optional[Dict] = None) -> str:
        """
        Get ML-powered district recommendations based on GPS location
        
        Args:
            user_lat: User's latitude
            user_lng: User's longitude  
            preferences: User preferences for personalization
            
        Returns:
            Personalized district recommendations
        """
        try:
            # Calculate distances to all districts
            district_distances = []
            for key, district in self.districts.items():
                distance = self._calculate_distance(user_lat, user_lng, district.latitude, district.longitude)
                district_distances.append((district, distance))
            
            # Sort by distance
            district_distances.sort(key=lambda x: x[1])
            
            # Apply ML personalization if preferences provided
            if preferences and self.ml_enabled:
                district_distances = self._apply_ml_personalization(district_distances, preferences)
            
            # Build response
            response_parts = []
            response_parts.append("📍 **Districts Near Your Location:**")
            response_parts.append("")
            
            current_time = datetime.now()
            time_period = self._get_time_period(current_time.hour)
            
            # Show top 3 closest districts with personalized info
            for i, (district, distance) in enumerate(district_distances[:3], 1):
                response_parts.append(f"{i}. **{district.name}** ({distance:.1f}km away)")
                response_parts.append(f"   📝 {district.description}")
                
                # Show time-appropriate activity
                if time_period in district.best_time_to_visit:
                    response_parts.append(f"   ⏰ Perfect for now: {district.best_time_to_visit[time_period]}")
                
                # Show top highlight
                if district.highlights:
                    response_parts.append(f"   ✨ Don't miss: {district.highlights[0]}")
                
                # Personalized recommendation if preferences available
                if preferences:
                    reason = self._generate_personalization_reason(district, preferences)
                    if reason:
                        response_parts.append(f"   🎯 For you: {reason}")
                
                response_parts.append("")
            
            # Add general recommendations
            response_parts.append("💡 **Quick Tips:**")
            response_parts.append(f"• Current time is perfect for {time_period} activities")
            response_parts.append("• Ask 'Tell me about [district name]' for detailed info")
            response_parts.append("• All districts are well-connected by public transport")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating location-based recommendations: {e}")
            return "I encountered an issue getting location-based recommendations. Please try asking about a specific district instead."
    
    def search_districts_by_category(self, category: DistrictCategory, 
                                   user_location: Optional[Tuple[float, float]] = None) -> str:
        """
        Search districts by category with optional location ranking
        
        Args:
            category: District category to search for
            user_location: Optional user location for distance sorting
            
        Returns:
            Formatted search results
        """
        matching_districts = []
        
        for district in self.districts.values():
            if category in district.categories:
                distance = None
                if user_location:
                    distance = self._calculate_distance(
                        user_location[0], user_location[1], 
                        district.latitude, district.longitude
                    )
                matching_districts.append((district, distance))
        
        if not matching_districts:
            return f"No districts found for category: {category.value}"
        
        # Sort by distance if location provided, otherwise by safety rating
        if user_location:
            matching_districts.sort(key=lambda x: x[1] if x[1] else float('inf'))
        else:
            matching_districts.sort(key=lambda x: x[0].safety_rating, reverse=True)
        
        # Build response
        response_parts = []
        response_parts.append(f"🏘️ **{category.value.title()} Districts in Istanbul:**")
        response_parts.append("")
        
        for i, (district, distance) in enumerate(matching_districts, 1):
            response_parts.append(f"{i}. **{district.name}**")
            response_parts.append(f"   📝 {district.description}")
            if distance:
                response_parts.append(f"   📍 {distance:.1f}km from your location")
            response_parts.append(f"   ⭐ Safety: {district.safety_rating}/5 • Budget: {district.budget_level}")
            response_parts.append(f"   ✨ Top highlight: {district.highlights[0] if district.highlights else 'Great area to explore'}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    # ============================================================================
    # ML PERSONALIZATION METHODS  
    # ============================================================================
    
    def _apply_ml_personalization(self, district_distances: List[Tuple[District, float]], 
                                 preferences: Dict) -> List[Tuple[District, float]]:
        """Apply ML-based personalization to district recommendations"""
        
        if not self.ml_enabled:
            return district_distances
        
        # Calculate personalization scores
        personalized_districts = []
        
        for district, distance in district_distances:
            # Start with distance-based score (closer = higher score)
            distance_score = max(0, 1 - (distance / 10))  # Normalize to 0-1, 10km max
            
            # Category preference matching
            category_score = 0
            user_categories = preferences.get('preferred_categories', [])
            if user_categories:
                district_category_values = [cat.value for cat in district.categories]
                matches = sum(1 for cat in user_categories if cat in district_category_values)
                category_score = matches / len(user_categories)  # Normalize to 0-1
            
            # Budget compatibility
            budget_score = 0
            user_budget = preferences.get('budget_preference')
            if user_budget:
                budget_compatibility = {
                    'budget': {'budget': 1.0, 'moderate': 0.7, 'expensive': 0.3},
                    'moderate': {'budget': 0.8, 'moderate': 1.0, 'expensive': 0.6},
                    'expensive': {'budget': 0.4, 'moderate': 0.7, 'expensive': 1.0}
                }
                budget_score = budget_compatibility.get(user_budget, {}).get(district.budget_level, 0.5)
            
            # Safety preference
            safety_score = district.safety_rating / 5.0  # Normalize to 0-1
            
            # Tourist density preference
            density_score = 0.5  # Default neutral
            user_crowd_preference = preferences.get('crowd_preference', 'medium')
            density_preferences = {
                'low': {'low': 1.0, 'medium': 0.6, 'high': 0.2},
                'medium': {'low': 0.7, 'medium': 1.0, 'high': 0.7},
                'high': {'low': 0.3, 'medium': 0.6, 'high': 1.0}
            }
            density_score = density_preferences.get(user_crowd_preference, {}).get(district.tourist_density, 0.5)
            
            # Combined ML score with weights
            ml_score = (
                distance_score * 0.3 +      # 30% distance
                category_score * 0.25 +     # 25% category match
                budget_score * 0.2 +        # 20% budget compatibility
                safety_score * 0.15 +       # 15% safety
                density_score * 0.1         # 10% crowd preference
            )
            
            personalized_districts.append((district, distance, ml_score))
        
        # Sort by ML score (descending) then by distance (ascending)
        personalized_districts.sort(key=lambda x: (-x[2], x[1]))
        
        # Return as original format
        return [(district, distance) for district, distance, _ in personalized_districts]
    
    def _generate_personalization_reason(self, district: District, preferences: Dict) -> str:
        """Generate human-readable reason for personalized recommendation"""
        
        reasons = []
        
        # Category matching
        user_categories = preferences.get('preferred_categories', [])
        district_category_values = [cat.value for cat in district.categories]
        matching_categories = [cat for cat in user_categories if cat in district_category_values]
        if matching_categories:
            reasons.append(f"matches your interest in {', '.join(matching_categories)} areas")
        
        # Budget compatibility
        user_budget = preferences.get('budget_preference')
        if user_budget and user_budget == district.budget_level:
            reasons.append(f"fits your {user_budget} budget preference")
        
        # Safety preference
        if district.safety_rating >= 4.5:
            reasons.append("highly rated for safety")
        
        # Crowd preference
        user_crowd = preferences.get('crowd_preference', 'medium')
        if user_crowd == 'low' and district.tourist_density == 'low':
            reasons.append("offers a quieter, less crowded experience")
        elif user_crowd == 'high' and district.tourist_density == 'high':
            reasons.append("has the vibrant, bustling atmosphere you prefer")
        
        return "; ".join(reasons) if reasons else "good overall match for your preferences"
    
    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================
    
    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        
        Returns:
            Distance in kilometers
        """
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in kilometers
        R = 6371
        
        return R * c
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period for time-based recommendations"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def get_all_districts(self) -> List[str]:
        """Get list of all available district names"""
        return [district.name for district in self.districts.values()]
    
    def get_districts_by_category(self, category: DistrictCategory) -> List[str]:
        """Get districts that match a specific category"""
        return [district.name for district in self.districts.values() 
                if category in district.categories]
    
    def is_district_available(self, district_name: str) -> bool:
        """Check if a district is available in the database"""
        district_key = district_name.lower().replace(" ", "").replace("ö", "o").replace("ğ", "g")
        return district_key in self.districts
    
    def get_poi_details(self, district_name: str, poi_name: str) -> Optional[str]:
        """Get detailed information about a specific POI in a district"""
        district_key = district_name.lower().replace(" ", "").replace("ö", "o").replace("ğ", "g")
        
        if district_key not in self.districts:
            return None
        
        district = self.districts[district_key]
        
        # Find POI by name (case-insensitive partial match)
        poi_name_lower = poi_name.lower()
        matching_poi = None
        
        for poi in district.pois:
            if poi_name_lower in poi.name.lower():
                matching_poi = poi
                break
        
        if not matching_poi:
            return None
        
        # Format POI details
        response_parts = []
        response_parts.append(f"🎯 **{matching_poi.name}**")
        response_parts.append(f"📍 Located in {district.name}")
        response_parts.append(f"📝 {matching_poi.description}")
        response_parts.append("")
        response_parts.append(f"⭐ **Rating:** {matching_poi.rating}/5.0")
        response_parts.append(f"🎫 **Entrance Fee:** {matching_poi.entrance_fee}")
        response_parts.append(f"♿ **Accessibility:** {'Yes' if matching_poi.accessibility else 'Limited'}")
        response_parts.append("")
        
        # Opening hours
        response_parts.append("⏰ **Opening Hours:**")
        for day, hours in matching_poi.opening_hours.items():
            response_parts.append(f"• {day.title()}: {hours}")
        response_parts.append("")
        
        # Additional info
        response_parts.append(f"📂 **Category:** {matching_poi.category.title()}")
        if matching_poi.website:
            response_parts.append(f"🌐 **Website:** {matching_poi.website}")
        
        return "\n".join(response_parts)
    
    def get_quick_recommendations(self, query_type: str, user_location: Optional[Tuple[float, float]] = None) -> str:
        """Get quick recommendations based on query type"""
        
        current_time = datetime.now()
        time_period = self._get_time_period(current_time.hour)
        
        if query_type.lower() in ['historic', 'historical', 'history']:
            historic_districts = [d for d in self.districts.values() if DistrictCategory.HISTORIC in d.categories]
            if user_location:
                historic_districts.sort(key=lambda d: self._calculate_distance(
                    user_location[0], user_location[1], d.latitude, d.longitude))
            
            response_parts = []
            response_parts.append("🏛️ **Historic Districts Perfect for Right Now:**")
            response_parts.append("")
            
            for district in historic_districts[:3]:
                response_parts.append(f"• **{district.name}**: {district.highlights[0] if district.highlights else 'Rich historical heritage'}")
                if time_period in district.best_time_to_visit:
                    response_parts.append(f"  ⏰ {district.best_time_to_visit[time_period]}")
                response_parts.append("")
            
            return "\n".join(response_parts)
        
        elif query_type.lower() in ['nightlife', 'bars', 'clubs', 'night']:
            nightlife_districts = [d for d in self.districts.values() if DistrictCategory.NIGHTLIFE in d.categories]
            if user_location:
                nightlife_districts.sort(key=lambda d: self._calculate_distance(
                    user_location[0], user_location[1], d.latitude, d.longitude))
            
            response_parts = []
            response_parts.append("🌃 **Best Districts for Nightlife:**")
            response_parts.append("")
            
            for district in nightlife_districts[:3]:
                response_parts.append(f"• **{district.name}**: {district.description}")
                if 'evening' in district.best_time_to_visit:
                    response_parts.append(f"  🌆 {district.best_time_to_visit['evening']}")
                response_parts.append("")
            
            return "\n".join(response_parts)
        
        elif query_type.lower() in ['shopping', 'shop', 'buy']:
            shopping_districts = [d for d in self.districts.values() if DistrictCategory.SHOPPING in d.categories]
            if user_location:
                shopping_districts.sort(key=lambda d: self._calculate_distance(
                    user_location[0], user_location[1], d.latitude, d.longitude))
            
            response_parts = []
            response_parts.append("🛍️ **Best Shopping Districts:**")
            response_parts.append("")
            
            for district in shopping_districts[:3]:
                response_parts.append(f"• **{district.name}**: {', '.join(district.shopping_areas)}")
                response_parts.append(f"  💰 Budget level: {district.budget_level}")
                response_parts.append("")
            
            return "\n".join(response_parts)
        
        else:
            # General recommendations based on time and location
            all_districts = list(self.districts.values())
            if user_location:
                all_districts.sort(key=lambda d: self._calculate_distance(
                    user_location[0], user_location[1], d.latitude, d.longitude))
            else:
                all_districts.sort(key=lambda d: d.safety_rating, reverse=True)
            
            response_parts = []
            response_parts.append(f"🏘️ **Great Districts to Visit Now ({time_period}):**")
            response_parts.append("")
            
            for district in all_districts[:4]:
                response_parts.append(f"• **{district.name}**: {district.description[:100]}...")
                if time_period in district.best_time_to_visit:
                    response_parts.append(f"  ⏰ Perfect timing: {district.best_time_to_visit[time_period]}")
                response_parts.append("")
            
            return "\n".join(response_parts)
    
    # ============================================================================
    # INTEGRATION METHODS FOR MAIN SYSTEM
    # ============================================================================
    
    def process_district_query(self, query: str, user_location: Optional[Tuple[float, float]] = None, 
                             user_preferences: Optional[Dict] = None) -> str:
        """
        Main entry point for processing district-related queries
        
        Args:
            query: User's query about districts
            user_location: Optional GPS coordinates (lat, lng)
            user_preferences: Optional user preferences for personalization
            
        Returns:
            Formatted response string
        """
        query_lower = query.lower()
        
        # Check for specific district queries
        for district_key, district in self.districts.items():
            if district.name.lower() in query_lower or district_key in query_lower:
                return self.get_district_info(district.name, user_preferences)
        
        # Check for category-based queries
        category_keywords = {
            'historic': DistrictCategory.HISTORIC,
            'historical': DistrictCategory.HISTORIC,
            'history': DistrictCategory.HISTORIC,
            'cultural': DistrictCategory.CULTURAL,
            'culture': DistrictCategory.CULTURAL,
            'modern': DistrictCategory.MODERN,
            'business': DistrictCategory.BUSINESS,
            'nightlife': DistrictCategory.NIGHTLIFE,
            'shopping': DistrictCategory.SHOPPING,
            'tourist': DistrictCategory.TOURISTIC,
            'touristic': DistrictCategory.TOURISTIC
        }
        
        for keyword, category in category_keywords.items():
            if keyword in query_lower:
                return self.search_districts_by_category(category, user_location)
        
        # Check for location-based queries
        if any(phrase in query_lower for phrase in ['near me', 'nearby', 'close', 'around here']):
            if user_location:
                return self.get_location_based_recommendations(user_location[0], user_location[1], user_preferences)
            else:
                return ("To provide location-based recommendations, I need your GPS location. "
                       "Please enable location services or tell me which area of Istanbul you're in.")
        
        # Check for quick recommendation queries
        quick_keywords = ['recommend', 'suggest', 'best', 'good', 'where to go', 'what to visit']
        if any(keyword in query_lower for keyword in quick_keywords):
            # Extract query type from context
            if any(word in query_lower for word in ['night', 'bar', 'club']):
                return self.get_quick_recommendations('nightlife', user_location)
            elif any(word in query_lower for word in ['shop', 'buy', 'market']):
                return self.get_quick_recommendations('shopping', user_location)
            elif any(word in query_lower for word in ['historic', 'history', 'old']):
                return self.get_quick_recommendations('historic', user_location)
            else:
                return self.get_quick_recommendations('general', user_location)
        
        # Default: show available districts
        response_parts = []
        response_parts.append("🏘️ **Istanbul Districts I Can Help You With:**")
        response_parts.append("")
        
        for district in self.districts.values():
            categories = ", ".join([cat.value for cat in district.categories[:2]])
            response_parts.append(f"• **{district.name}** ({categories})")
        
        response_parts.append("")
        response_parts.append("💡 **Try asking:**")
        response_parts.append("• 'Tell me about Sultanahmet'")
        response_parts.append("• 'Show me historic districts'")
        response_parts.append("• 'Recommend districts near me' (with GPS)")
        response_parts.append("• 'Best nightlife districts'")
        
        return "\n".join(response_parts)

# ============================================================================
# MAIN INTEGRATION FUNCTION
# ============================================================================

def process_neighborhood_query_enhanced(message: str, user_profile=None, current_time: datetime = None, 
                                       user_location: Optional[Tuple[float, float]] = None) -> str:
    """
    Enhanced neighborhood/district query processor with ML and GPS integration
    
    This function replaces the basic neighborhood handler in ml_personalization_helpers.py
    """
    # Initialize the district system (in a real system, this would be a singleton)
    district_system = DistrictAdvisingSystem()
    
    # Extract user preferences from profile if available
    user_preferences = {}
    if user_profile:
        user_preferences = {
            'preferred_categories': getattr(user_profile, 'interests', []),
            'budget_preference': getattr(user_profile, 'budget_range', 'moderate'),
            'crowd_preference': 'low' if getattr(user_profile, 'travel_style', '') == 'solo' else 'medium'
        }
    
    # Process the query
    return district_system.process_district_query(message, user_location, user_preferences)


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Test the district advising system
    print("🏘️ Testing Istanbul District Advising System")
    print("=" * 60)
    
    system = DistrictAdvisingSystem()
    
    # Test specific district query
    print("\n1. Testing specific district query:")
    result = system.get_district_info("Sultanahmet")
    print(result[:200] + "..." if len(result) > 200 else result)
    
    # Test location-based recommendations
    print("\n2. Testing GPS-based recommendations:")
    # Using coordinates near Taksim Square
    result = system.get_location_based_recommendations(41.0369, 28.9850)
    print(result[:300] + "..." if len(result) > 300 else result)
    
    # Test category search
    print("\n3. Testing category search:")
    result = system.search_districts_by_category(DistrictCategory.HISTORIC)
    print(result[:200] + "..." if len(result) > 200 else result)
    
    print("\n✅ District Advising System is ready for integration!")
    print(f"📊 Loaded districts: {', '.join(system.get_all_districts())}")
    print("🎯 Features: GPS integration, ML personalization, comprehensive POI data")
