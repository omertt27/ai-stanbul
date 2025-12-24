"""
Istanbul Trip Planner System
=============================

Generates multi-day Istanbul itineraries with:
- Pre-built trip templates (1, 3, 5 days)
- Daily routes with attractions
- Transit connections between attractions
- Map visualization data

Author: AI Istanbul Team
Date: December 2024
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TripStyle(Enum):
    """Trip styles for different traveler preferences"""
    CLASSIC = "classic"           # Must-see attractions
    CULTURAL = "cultural"         # Museums, history, art
    FOODIE = "foodie"            # Food tours, markets, restaurants
    LOCAL = "local"              # Off-the-beaten-path, local experiences
    FAMILY = "family"            # Kid-friendly attractions
    ROMANTIC = "romantic"        # Couples activities


@dataclass
class Attraction:
    """An attraction/point of interest"""
    name: str
    name_tr: str
    description: str
    lat: float
    lon: float
    district: str
    category: str  # museum, mosque, market, park, restaurant, etc.
    visit_duration: int  # minutes
    nearest_station: str  # Station ID
    opening_hours: str = "09:00-18:00"
    entrance_fee: str = "Free"
    tips: List[str] = field(default_factory=list)


@dataclass
class DayPlan:
    """A single day's itinerary"""
    day_number: int
    title: str
    title_tr: str
    theme: str
    attractions: List[Attraction]
    total_duration: int  # minutes
    walking_distance: float  # km
    transit_lines: List[str]


@dataclass
class TripPlan:
    """Complete trip itinerary"""
    name: str
    name_tr: str
    duration_days: int
    style: TripStyle
    days: List[DayPlan]
    total_attractions: int
    highlights: List[str]


class IstanbulTripPlanner:
    """
    Multi-day Istanbul trip planner with map visualization.
    
    Features:
    - Pre-built itineraries for 1, 3, 5 days
    - Custom trip generation based on preferences
    - Transit route integration
    - Map data export for visualization
    """
    
    def __init__(self):
        """Initialize with attraction database"""
        self.attractions = self._build_attraction_database()
        self.trip_templates = self._build_trip_templates()
        logger.info(f"✅ Trip Planner initialized with {len(self.attractions)} attractions")
    
    def _build_attraction_database(self) -> Dict[str, Attraction]:
        """Build comprehensive Istanbul attraction database"""
        attractions = {}
        
        # ==========================================
        # SULTANAHMET / OLD CITY
        # ==========================================
        attractions["hagia_sophia"] = Attraction(
            name="Hagia Sophia",
            name_tr="Ayasofya",
            description="6th-century Byzantine cathedral, later mosque, now museum. Stunning architecture with massive dome.",
            lat=41.0086, lon=28.9802,
            district="Sultanahmet",
            category="mosque",
            visit_duration=90,
            nearest_station="T1-Sultanahmet",
            opening_hours="09:00-19:00",
            entrance_fee="Free (mosque)",
            tips=["Visit early morning to avoid crowds", "Dress modestly - head covering required for women", "Photography allowed"]
        )
        
        attractions["blue_mosque"] = Attraction(
            name="Blue Mosque",
            name_tr="Sultanahmet Camii",
            description="17th-century Ottoman mosque famous for its blue İznik tiles. Six minarets.",
            lat=41.0054, lon=28.9768,
            district="Sultanahmet",
            category="mosque",
            visit_duration=45,
            nearest_station="T1-Sultanahmet",
            opening_hours="08:30-18:30",
            entrance_fee="Free",
            tips=["Closed during prayer times", "Remove shoes at entrance", "Free but donations appreciated"]
        )
        
        attractions["topkapi_palace"] = Attraction(
            name="Topkapi Palace",
            name_tr="Topkapı Sarayı",
            description="Ottoman sultans' residence for 400 years. Treasury, harem, and stunning Bosphorus views.",
            lat=41.0115, lon=28.9833,
            district="Sultanahmet",
            category="museum",
            visit_duration=180,
            nearest_station="T1-Gülhane",
            opening_hours="09:00-18:00",
            entrance_fee="₺450 (Harem separate)",
            tips=["Buy combo ticket for Harem", "Tuesday closed", "Allow 3+ hours"]
        )
        
        attractions["basilica_cistern"] = Attraction(
            name="Basilica Cistern",
            name_tr="Yerebatan Sarnıcı",
            description="6th-century underground cistern with 336 columns. Atmospheric lighting and Medusa heads.",
            lat=41.0084, lon=28.9779,
            district="Sultanahmet",
            category="museum",
            visit_duration=45,
            nearest_station="T1-Sultanahmet",
            opening_hours="09:00-18:30",
            entrance_fee="₺350",
            tips=["Cool escape from summer heat", "Mystical atmosphere", "Can get crowded"]
        )
        
        attractions["grand_bazaar"] = Attraction(
            name="Grand Bazaar",
            name_tr="Kapalıçarşı",
            description="One of world's oldest covered markets. 4,000+ shops selling carpets, jewelry, ceramics.",
            lat=41.0106, lon=28.9680,
            district="Beyazıt",
            category="market",
            visit_duration=120,
            nearest_station="T1-Beyazıt-Kapalıçarşı",
            opening_hours="09:00-19:00",
            entrance_fee="Free",
            tips=["Closed Sundays", "Bargain expected", "Easy to get lost - enjoy it!"]
        )
        
        attractions["spice_bazaar"] = Attraction(
            name="Spice Bazaar",
            name_tr="Mısır Çarşısı",
            description="17th-century market selling spices, Turkish delight, dried fruits, and souvenirs.",
            lat=41.0165, lon=28.9706,
            district="Eminönü",
            category="market",
            visit_duration=60,
            nearest_station="T1-Eminönü",
            opening_hours="08:00-19:00",
            entrance_fee="Free",
            tips=["Try free samples", "Great for Turkish delight gifts", "Adjacent to Yeni Mosque"]
        )
        
        # ==========================================
        # BEYOĞLU / TAKSIM
        # ==========================================
        attractions["galata_tower"] = Attraction(
            name="Galata Tower",
            name_tr="Galata Kulesi",
            description="14th-century Genoese tower with panoramic views of Istanbul. 67m tall.",
            lat=41.0256, lon=28.9741,
            district="Beyoğlu",
            category="landmark",
            visit_duration=45,
            nearest_station="T1-Karaköy",
            opening_hours="08:30-23:00",
            entrance_fee="₺200",
            tips=["Best at sunset", "Long queues - book online", "Great views of Bosphorus"]
        )
        
        attractions["istiklal_street"] = Attraction(
            name="Istiklal Street",
            name_tr="İstiklal Caddesi",
            description="1.4km pedestrian street with shops, cafes, historic buildings, and nostalgic tram.",
            lat=41.0335, lon=28.9770,
            district="Beyoğlu",
            category="street",
            visit_duration=90,
            nearest_station="M2-Taksim",
            opening_hours="24/7",
            entrance_fee="Free",
            tips=["Walk from Taksim to Tünel", "Try historic Çiçek Pasajı", "Live music at night"]
        )
        
        attractions["taksim_square"] = Attraction(
            name="Taksim Square",
            name_tr="Taksim Meydanı",
            description="Heart of modern Istanbul. Republic Monument and start of Istiklal Street.",
            lat=41.0369, lon=28.9850,
            district="Beyoğlu",
            category="square",
            visit_duration=30,
            nearest_station="M2-Taksim",
            opening_hours="24/7",
            entrance_fee="Free",
            tips=["Major transit hub", "Good starting point for Beyoğlu exploration"]
        )
        
        # ==========================================
        # BOSPHORUS
        # ==========================================
        attractions["bosphorus_cruise"] = Attraction(
            name="Bosphorus Cruise",
            name_tr="Boğaz Turu",
            description="Scenic boat tour between Europe and Asia. See palaces, fortresses, and bridges.",
            lat=41.0197, lon=28.9739,
            district="Eminönü",
            category="tour",
            visit_duration=120,
            nearest_station="FERRY-Eminönü",
            opening_hours="10:00-18:00",
            entrance_fee="₺100-300",
            tips=["Short (2hr) or full-day options", "Sit on right side for European shore", "Departs from Eminönü"]
        )
        
        attractions["dolmabahce_palace"] = Attraction(
            name="Dolmabahce Palace",
            name_tr="Dolmabahçe Sarayı",
            description="19th-century Ottoman palace on Bosphorus. Lavish European-style interiors, 285 rooms.",
            lat=41.0392, lon=29.0006,
            district="Beşiktaş",
            category="museum",
            visit_duration=120,
            nearest_station="T1-Kabataş",
            opening_hours="09:00-16:00",
            entrance_fee="₺450",
            tips=["Monday closed", "Guided tours only", "World's largest crystal chandelier"]
        )
        
        attractions["ortakoy_mosque"] = Attraction(
            name="Ortaköy Mosque",
            name_tr="Ortaköy Camii",
            description="Baroque-style mosque right on Bosphorus shore, under the bridge. Iconic photo spot.",
            lat=41.0472, lon=29.0275,
            district="Ortaköy",
            category="mosque",
            visit_duration=30,
            nearest_station="T4-Ortaköy",
            opening_hours="09:00-18:00",
            entrance_fee="Free",
            tips=["Try famous Ortaköy kumpir (stuffed potato)", "Sunday market", "Beautiful at night"]
        )
        
        # ==========================================
        # ASIAN SIDE
        # ==========================================
        attractions["kadikoy_market"] = Attraction(
            name="Kadıköy Market",
            name_tr="Kadıköy Çarşısı",
            description="Authentic local market district with food stalls, fish market, and cafes.",
            lat=40.9903, lon=29.0275,
            district="Kadıköy",
            category="market",
            visit_duration=90,
            nearest_station="M4-Kadıköy",
            opening_hours="08:00-20:00",
            entrance_fee="Free",
            tips=["Best fish restaurants in city", "Try Çiya Sofrası for traditional food", "Less touristy than European side"]
        )
        
        attractions["moda_seaside"] = Attraction(
            name="Moda Seaside",
            name_tr="Moda Sahili",
            description="Trendy neighborhood with seaside promenade, cafes, and stunning sunset views.",
            lat=40.9833, lon=29.0250,
            district="Kadıköy",
            category="neighborhood",
            visit_duration=60,
            nearest_station="M4-Kadıköy",
            opening_hours="24/7",
            entrance_fee="Free",
            tips=["Walk along the coast", "Hipster cafes and bars", "Sunset watching spot"]
        )
        
        attractions["uskudar_kuzguncuk"] = Attraction(
            name="Üsküdar & Kuzguncuk",
            name_tr="Üsküdar ve Kuzguncuk",
            description="Historic Asian side neighborhoods. Ottoman architecture, colorful houses, local life.",
            lat=41.0256, lon=29.0100,
            district="Üsküdar",
            category="neighborhood",
            visit_duration=90,
            nearest_station="M5-Üsküdar",
            opening_hours="24/7",
            entrance_fee="Free",
            tips=["Maiden's Tower view from shore", "Kuzguncuk is 'notting hill of Istanbul'", "Tea gardens overlooking Bosphorus"]
        )
        
        attractions["camlica_hill"] = Attraction(
            name="Çamlıca Hill",
            name_tr="Çamlıca Tepesi",
            description="Highest point in Istanbul with panoramic city views. New mosque and TV tower.",
            lat=41.0275, lon=29.0692,
            district="Üsküdar",
            category="viewpoint",
            visit_duration=60,
            nearest_station="M5-Kısıklı",
            opening_hours="24/7",
            entrance_fee="Free",
            tips=["Best panoramic views of Istanbul", "New Çamlıca Mosque is Turkey's largest", "Great at sunset"]
        )
        
        # ==========================================
        # FOOD & CULTURE
        # ==========================================
        attractions["turkish_breakfast"] = Attraction(
            name="Turkish Breakfast Experience",
            name_tr="Türk Kahvaltısı",
            description="Traditional spread with cheese, olives, eggs, honey, and unlimited çay.",
            lat=41.0335, lon=28.9770,
            district="Beyoğlu",
            category="food",
            visit_duration=90,
            nearest_station="M2-Taksim",
            opening_hours="08:00-14:00",
            entrance_fee="₺150-300",
            tips=["Try Van Kahvaltısı or Serpme Kahvaltı", "Best in Beşiktaş or Kadıköy", "Plan 2 hours minimum"]
        )
        
        attractions["hammam_experience"] = Attraction(
            name="Turkish Bath Experience",
            name_tr="Hamam Deneyimi",
            description="Traditional Ottoman bath house experience. Historic options: Çemberlitaş or Ayasofya Hürrem.",
            lat=41.0083, lon=28.9711,
            district="Sultanahmet",
            category="experience",
            visit_duration=90,
            nearest_station="T1-Çemberlitaş",
            opening_hours="07:00-22:00",
            entrance_fee="€50-100",
            tips=["Book in advance", "Separate sections for men/women", "Relaxing after walking tours"]
        )
        
        return attractions
    
    def _build_trip_templates(self) -> Dict[str, TripPlan]:
        """Build pre-designed trip templates"""
        templates = {}
        
        # ==========================================
        # 1-DAY ISTANBUL HIGHLIGHTS
        # ==========================================
        templates["1_day_highlights"] = TripPlan(
            name="Istanbul in 1 Day - Highlights",
            name_tr="1 Günde İstanbul - Öne Çıkanlar",
            duration_days=1,
            style=TripStyle.CLASSIC,
            days=[
                DayPlan(
                    day_number=1,
                    title="Best of Istanbul",
                    title_tr="İstanbul'un En İyileri",
                    theme="Must-see highlights",
                    attractions=[
                        self.attractions["hagia_sophia"],
                        self.attractions["blue_mosque"],
                        self.attractions["basilica_cistern"],
                        self.attractions["grand_bazaar"],
                        self.attractions["galata_tower"],
                        self.attractions["istiklal_street"],
                    ],
                    total_duration=420,  # 7 hours
                    walking_distance=5.0,
                    transit_lines=["T1", "F2"]
                )
            ],
            total_attractions=6,
            highlights=["Hagia Sophia", "Blue Mosque", "Grand Bazaar", "Galata Tower"]
        )
        
        # ==========================================
        # 3-DAY CLASSIC ISTANBUL
        # ==========================================
        templates["3_day_classic"] = TripPlan(
            name="3-Day Classic Istanbul",
            name_tr="3 Günlük Klasik İstanbul",
            duration_days=3,
            style=TripStyle.CLASSIC,
            days=[
                # Day 1: Old City
                DayPlan(
                    day_number=1,
                    title="Old City Wonders",
                    title_tr="Tarihi Yarımada",
                    theme="Byzantine & Ottoman treasures",
                    attractions=[
                        self.attractions["hagia_sophia"],
                        self.attractions["blue_mosque"],
                        self.attractions["topkapi_palace"],
                        self.attractions["basilica_cistern"],
                    ],
                    total_duration=360,
                    walking_distance=3.0,
                    transit_lines=["T1"]
                ),
                # Day 2: Markets & Beyoğlu
                DayPlan(
                    day_number=2,
                    title="Markets & Modern Istanbul",
                    title_tr="Çarşılar ve Modern İstanbul",
                    theme="Shopping and Beyoğlu",
                    attractions=[
                        self.attractions["grand_bazaar"],
                        self.attractions["spice_bazaar"],
                        self.attractions["galata_tower"],
                        self.attractions["istiklal_street"],
                        self.attractions["taksim_square"],
                    ],
                    total_duration=390,
                    walking_distance=6.0,
                    transit_lines=["T1", "F2", "M2"]
                ),
                # Day 3: Bosphorus
                DayPlan(
                    day_number=3,
                    title="Bosphorus Experience",
                    title_tr="Boğaz Deneyimi",
                    theme="Waterfront palaces and views",
                    attractions=[
                        self.attractions["dolmabahce_palace"],
                        self.attractions["bosphorus_cruise"],
                        self.attractions["ortakoy_mosque"],
                    ],
                    total_duration=300,
                    walking_distance=4.0,
                    transit_lines=["T1", "FERRY"]
                )
            ],
            total_attractions=12,
            highlights=["Hagia Sophia", "Topkapi Palace", "Grand Bazaar", "Bosphorus Cruise", "Galata Tower"]
        )
        
        # ==========================================
        # 5-DAY COMPLETE ISTANBUL
        # ==========================================
        templates["5_day_complete"] = TripPlan(
            name="5-Day Complete Istanbul Experience",
            name_tr="5 Günlük Tam İstanbul Deneyimi",
            duration_days=5,
            style=TripStyle.CLASSIC,
            days=[
                # Day 1: Old City Essentials
                DayPlan(
                    day_number=1,
                    title="Sultanahmet Treasures",
                    title_tr="Sultanahmet Hazineleri",
                    theme="UNESCO World Heritage sites",
                    attractions=[
                        self.attractions["hagia_sophia"],
                        self.attractions["blue_mosque"],
                        self.attractions["basilica_cistern"],
                        self.attractions["hammam_experience"],
                    ],
                    total_duration=300,
                    walking_distance=2.5,
                    transit_lines=["T1"]
                ),
                # Day 2: Palaces & History
                DayPlan(
                    day_number=2,
                    title="Ottoman Palaces",
                    title_tr="Osmanlı Sarayları",
                    theme="Royal residences",
                    attractions=[
                        self.attractions["topkapi_palace"],
                        self.attractions["dolmabahce_palace"],
                    ],
                    total_duration=360,
                    walking_distance=4.0,
                    transit_lines=["T1"]
                ),
                # Day 3: Markets & Bazaars
                DayPlan(
                    day_number=3,
                    title="Bazaar Day",
                    title_tr="Çarşı Günü",
                    theme="Shopping and local markets",
                    attractions=[
                        self.attractions["grand_bazaar"],
                        self.attractions["spice_bazaar"],
                        self.attractions["galata_tower"],
                        self.attractions["istiklal_street"],
                    ],
                    total_duration=360,
                    walking_distance=6.0,
                    transit_lines=["T1", "F2"]
                ),
                # Day 4: Bosphorus Day
                DayPlan(
                    day_number=4,
                    title="Bosphorus Journey",
                    title_tr="Boğaz Gezisi",
                    theme="Europe meets Asia",
                    attractions=[
                        self.attractions["bosphorus_cruise"],
                        self.attractions["ortakoy_mosque"],
                        self.attractions["uskudar_kuzguncuk"],
                    ],
                    total_duration=300,
                    walking_distance=4.0,
                    transit_lines=["FERRY", "M5"]
                ),
                # Day 5: Asian Side
                DayPlan(
                    day_number=5,
                    title="Asian Istanbul",
                    title_tr="Anadolu Yakası",
                    theme="Local life & hidden gems",
                    attractions=[
                        self.attractions["turkish_breakfast"],
                        self.attractions["kadikoy_market"],
                        self.attractions["moda_seaside"],
                        self.attractions["camlica_hill"],
                    ],
                    total_duration=360,
                    walking_distance=5.0,
                    transit_lines=["MARMARAY", "M4", "M5"]
                )
            ],
            total_attractions=17,
            highlights=[
                "Hagia Sophia", "Topkapi Palace", "Dolmabahce Palace",
                "Grand Bazaar", "Bosphorus Cruise", "Kadıköy Market",
                "Çamlıca Hill"
            ]
        )
        
        return templates
    
    def get_trip_plan(self, duration: int, style: TripStyle = TripStyle.CLASSIC) -> Optional[TripPlan]:
        """
        Get a trip plan for the specified duration.
        
        Args:
            duration: Number of days (1, 3, or 5)
            style: Trip style preference
            
        Returns:
            TripPlan or None if not found
        """
        template_key = f"{duration}_day_{'highlights' if duration == 1 else 'classic' if duration == 3 else 'complete'}"
        return self.trip_templates.get(template_key)
    
    def get_available_trips(self) -> List[Dict[str, Any]]:
        """Get list of available trip templates"""
        trips = []
        for key, plan in self.trip_templates.items():
            trips.append({
                "id": key,
                "name": plan.name,
                "name_tr": plan.name_tr,
                "duration_days": plan.duration_days,
                "style": plan.style.value,
                "total_attractions": plan.total_attractions,
                "highlights": plan.highlights
            })
        return trips
    
    def get_trip_map_data(self, trip_id: str) -> Optional[Dict[str, Any]]:
        """
        Generate map visualization data for a trip plan.
        
        Returns data compatible with MapVisualization component:
        - All attraction markers with colors by day
        - Routes between attractions for each day
        - Day-by-day breakdown
        """
        plan = self.trip_templates.get(trip_id)
        if not plan:
            return None
        
        # Color palette for different days
        day_colors = [
            "#4285F4",  # Blue - Day 1
            "#EA4335",  # Red - Day 2
            "#34A853",  # Green - Day 3
            "#FBBC05",  # Yellow - Day 4
            "#9C27B0",  # Purple - Day 5
        ]
        
        all_markers = []
        all_routes = []
        all_coordinates = []
        days_data = []
        
        for day in plan.days:
            day_idx = day.day_number - 1
            day_color = day_colors[day_idx % len(day_colors)]
            day_markers = []
            day_coords = []
            
            for i, attraction in enumerate(day.attractions):
                marker_type = "origin" if i == 0 else "destination" if i == len(day.attractions) - 1 else "stop"
                
                marker = {
                    "lat": attraction.lat,
                    "lon": attraction.lon,
                    "label": attraction.name,
                    "title": attraction.name,
                    "description": f"Day {day.day_number}: {attraction.description[:100]}...",
                    "type": marker_type,
                    "day": day.day_number,
                    "category": attraction.category,
                    "duration": attraction.visit_duration,
                    "icon": self._get_category_emoji(attraction.category),
                    "color": day_color
                }
                
                all_markers.append(marker)
                day_markers.append(marker)
                day_coords.append({"lat": attraction.lat, "lng": attraction.lon})
                all_coordinates.append([attraction.lat, attraction.lon])
            
            # Create route for this day
            if len(day_coords) >= 2:
                all_routes.append({
                    "coordinates": day_coords,
                    "color": day_color,
                    "weight": 4,
                    "opacity": 0.8,
                    "mode": "walking",
                    "description": f"Day {day.day_number}: {day.title}",
                    "day": day.day_number
                })
            
            # Day summary
            days_data.append({
                "day_number": day.day_number,
                "title": day.title,
                "title_tr": day.title_tr,
                "theme": day.theme,
                "color": day_color,
                "attractions": [
                    {
                        "name": a.name,
                        "name_tr": a.name_tr,
                        "category": a.category,
                        "duration": a.visit_duration,
                        "lat": a.lat,
                        "lon": a.lon
                    }
                    for a in day.attractions
                ],
                "total_duration": day.total_duration,
                "walking_distance": day.walking_distance,
                "transit_lines": day.transit_lines
            })
        
        # Calculate center and bounds
        if all_coordinates:
            lats = [c[0] for c in all_coordinates]
            lons = [c[1] for c in all_coordinates]
            center = {
                "lat": sum(lats) / len(lats),
                "lon": sum(lons) / len(lons)
            }
        else:
            center = {"lat": 41.0082, "lon": 28.9784}  # Istanbul center
        
        return {
            "type": "trip_plan",
            "trip_id": trip_id,
            "name": plan.name,
            "name_tr": plan.name_tr,
            "duration_days": plan.duration_days,
            "style": plan.style.value,
            "markers": all_markers,
            "routes": all_routes,
            "coordinates": all_coordinates,
            "center": center,
            "zoom": 12,
            "bounds": {"autoFit": True},
            "days": days_data,
            "metadata": {
                "total_attractions": plan.total_attractions,
                "highlights": plan.highlights,
                "total_markers": len(all_markers),
                "total_routes": len(all_routes)
            }
        }
    
    def get_day_map_data(self, trip_id: str, day_number: int) -> Optional[Dict[str, Any]]:
        """
        Generate map data for a specific day of a trip.
        """
        plan = self.trip_templates.get(trip_id)
        if not plan or day_number < 1 or day_number > len(plan.days):
            return None
        
        day = plan.days[day_number - 1]
        day_color = "#4285F4"
        
        markers = []
        coords = []
        coordinates = []
        
        for i, attraction in enumerate(day.attractions):
            marker_type = "origin" if i == 0 else "destination" if i == len(day.attractions) - 1 else "stop"
            
            markers.append({
                "lat": attraction.lat,
                "lon": attraction.lon,
                "label": f"{i+1}. {attraction.name}",
                "title": attraction.name,
                "description": attraction.description,
                "type": marker_type,
                "category": attraction.category,
                "duration": attraction.visit_duration,
                "icon": self._get_category_emoji(attraction.category)
            })
            
            coords.append({"lat": attraction.lat, "lng": attraction.lon})
            coordinates.append([attraction.lat, attraction.lon])
        
        routes = []
        if len(coords) >= 2:
            routes.append({
                "coordinates": coords,
                "color": day_color,
                "weight": 4,
                "opacity": 0.8,
                "mode": "walking",
                "description": day.title
            })
        
        # Calculate center
        if coordinates:
            lats = [c[0] for c in coordinates]
            lons = [c[1] for c in coordinates]
            center = {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}
        else:
            center = {"lat": 41.0082, "lon": 28.9784}
        
        return {
            "type": "day_plan",
            "trip_id": trip_id,
            "day_number": day_number,
            "title": day.title,
            "markers": markers,
            "routes": routes,
            "coordinates": coordinates,
            "center": center,
            "zoom": 13,
            "bounds": {"autoFit": True}
        }
    
    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for attraction category"""
        emoji_map = {
            "mosque": "🕌",
            "museum": "🏛️",
            "market": "🛍️",
            "landmark": "🏰",
            "palace": "👑",
            "viewpoint": "🌅",
            "neighborhood": "🏘️",
            "street": "🚶",
            "square": "⭐",
            "tour": "🚢",
            "food": "🍽️",
            "experience": "💆",
            "park": "🌳"
        }
        return emoji_map.get(category, "📍")
    
    def get_trip_rag_context(self, trip_id: str, language: str = "en") -> str:
        """
        Generate comprehensive RAG context for LLM about a trip plan.
        
        Args:
            trip_id: Trip template ID
            language: Language for context
            
        Returns:
            Formatted string context for LLM prompt with all practical details
        """
        plan = self.trip_templates.get(trip_id)
        if not plan:
            return ""
        
        lines = []
        
        if language == "tr":
            lines.append(f"# 🗓️ {plan.name_tr}")
            lines.append(f"**Süre:** {plan.duration_days} gün | **Toplam:** {plan.total_attractions} mekan")
            lines.append("")
            
            for day in plan.days:
                lines.append(f"## 🌅 Gün {day.day_number}: {day.title_tr}")
                lines.append(f"*Tema: {day.theme}*")
                lines.append(f"⏱️ Tahmini süre: ~{day.total_duration // 60} saat | 🚶 Yürüme: {day.walking_distance:.1f} km")
                if day.transit_lines:
                    lines.append(f"🚇 Ulaşım: {', '.join(day.transit_lines)}")
                lines.append("")
                
                for i, attr in enumerate(day.attractions, 1):
                    emoji = self._get_category_emoji(attr.category)
                    lines.append(f"### {i}. {emoji} {attr.name_tr}")
                    lines.append(f"- ⏱️ Süre: {attr.visit_duration} dakika")
                    lines.append(f"- 💰 Giriş: {attr.entrance_fee}")
                    lines.append(f"- 🕐 Saatler: {attr.opening_hours}")
                    lines.append(f"- 📍 {attr.description}")
                    if attr.tips:
                        lines.append(f"- 💡 İpucu: {attr.tips[0]}")
                    lines.append("")
                lines.append("---")
                lines.append("")
            
            if plan.highlights:
                lines.append(f"## ✨ Öne Çıkanlar")
                lines.append(', '.join(plan.highlights))
        else:
            lines.append(f"# 🗓️ {plan.name}")
            lines.append(f"**Duration:** {plan.duration_days} days | **Total:** {plan.total_attractions} attractions")
            lines.append("")
            
            for day in plan.days:
                lines.append(f"## 🌅 Day {day.day_number}: {day.title}")
                lines.append(f"*Theme: {day.theme}*")
                lines.append(f"⏱️ Estimated time: ~{day.total_duration // 60} hours | 🚶 Walking: {day.walking_distance:.1f} km")
                if day.transit_lines:
                    lines.append(f"🚇 Transit: {', '.join(day.transit_lines)}")
                lines.append("")
                
                for i, attr in enumerate(day.attractions, 1):
                    emoji = self._get_category_emoji(attr.category)
                    lines.append(f"### {i}. {emoji} {attr.name}")
                    lines.append(f"- ⏱️ Duration: {attr.visit_duration} minutes")
                    lines.append(f"- 💰 Entrance: {attr.entrance_fee}")
                    lines.append(f"- 🕐 Hours: {attr.opening_hours}")
                    lines.append(f"- 📍 {attr.description}")
                    if attr.tips:
                        lines.append(f"- 💡 Tip: {attr.tips[0]}")
                    lines.append("")
                lines.append("---")
                lines.append("")
            
            if plan.highlights:
                lines.append(f"## ✨ Trip Highlights")
                lines.append(', '.join(plan.highlights))
        
        return "\n".join(lines)
    
    def parse_user_preferences(self, query: str) -> Dict[str, Any]:
        """
        Parse user query to extract trip preferences.
        
        Args:
            query: User's trip planning query
            
        Returns:
            Dictionary with extracted preferences
        """
        query_lower = query.lower()
        prefs = {
            "duration": 3,  # Default
            "style": TripStyle.CLASSIC,
            "interests": []
        }
        
        # Extract duration
        import re
        day_patterns = [
            (r'(\d+)\s*(?:day|gün)', lambda m: int(m.group(1))),
            (r'(one|bir)\s*day', lambda m: 1),
            (r'(two|iki)\s*day', lambda m: 2),
            (r'(three|üç)\s*day', lambda m: 3),
            (r'(four|dört)\s*day', lambda m: 4),
            (r'(five|beş)\s*day', lambda m: 5),
        ]
        
        for pattern, extractor in day_patterns:
            match = re.search(pattern, query_lower)
            if match:
                prefs["duration"] = extractor(match)
                break
        
        # Extract style preferences
        style_keywords = {
            TripStyle.CULTURAL: ["museum", "history", "cultural", "art", "müze", "tarih", "kültür"],
            TripStyle.FOODIE: ["food", "eat", "restaurant", "cuisine", "yemek", "restoran", "mutfak"],
            TripStyle.LOCAL: ["local", "authentic", "off the beaten", "yerel", "otantik"],
            TripStyle.FAMILY: ["family", "kids", "children", "aile", "çocuk"],
            TripStyle.ROMANTIC: ["romantic", "couple", "honeymoon", "romantik", "çift", "balayı"]
        }
        
        for style, keywords in style_keywords.items():
            if any(kw in query_lower for kw in keywords):
                prefs["style"] = style
                break
        
        # Extract specific interests
        interest_keywords = {
            "mosques": ["mosque", "cami", "islamic"],
            "palaces": ["palace", "saray", "ottoman"],
            "markets": ["market", "bazaar", "shopping", "çarşı", "alışveriş"],
            "food": ["food", "restaurant", "cuisine", "yemek"],
            "views": ["view", "panoramic", "sunset", "manzara"],
            "bosphorus": ["bosphorus", "boğaz", "cruise", "ferry"]
        }
        
        for interest, keywords in interest_keywords.items():
            if any(kw in query_lower for kw in keywords):
                prefs["interests"].append(interest)
        
        return prefs


# Global singleton
_trip_planner: Optional[IstanbulTripPlanner] = None


def get_trip_planner() -> IstanbulTripPlanner:
    """Get or create trip planner singleton"""
    global _trip_planner
    if _trip_planner is None:
        _trip_planner = IstanbulTripPlanner()
    return _trip_planner
