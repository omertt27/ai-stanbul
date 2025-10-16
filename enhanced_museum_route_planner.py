#!/usr/bin/env python3
"""
Enhanced Museum Route Planner with Local Tips
==============================================
Specialized route planning focused on Istanbul's museums and cultural sites
with insider local knowledge and tips
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MuseumCategory(Enum):
    """Museum categories in Istanbul"""
    ARCHAEOLOGICAL = "archaeological"
    ART_CLASSICAL = "art_classical"
    ART_CONTEMPORARY = "art_contemporary"
    HISTORY = "history"
    RELIGIOUS = "religious"
    PALACE = "palace"
    MARITIME = "maritime"
    SCIENCE = "science"
    CULTURAL_HOUSE = "cultural_house"
    PHOTOGRAPHY = "photography"

class LocalTipCategory(Enum):
    """Types of local tips"""
    TIMING = "timing"
    PHOTOGRAPHY = "photography"
    FOOD_NEARBY = "food_nearby"
    SHOPPING = "shopping"
    TRANSPORT = "transport"
    INSIDER_SECRET = "insider_secret"
    CULTURAL_ETIQUETTE = "cultural_etiquette"
    SEASONAL = "seasonal"

@dataclass
class LocalTip:
    """Local insider tip"""
    category: LocalTipCategory
    title: str
    description: str
    importance: int  # 1-5, 5 being most important
    location_specific: bool = True
    season_specific: Optional[str] = None

@dataclass
class Museum:
    """Enhanced museum information with local insights"""
    id: str
    name: str
    category: MuseumCategory
    coordinates: Tuple[float, float]
    district: str
    neighborhood: str
    
    # Detailed Information
    description: str
    highlights: List[str]
    visit_duration_minutes: int
    entry_fee_tl: float
    
    # Timing Information
    opening_hours: Dict[str, str]
    best_visit_times: List[str]
    peak_hours: List[int]
    closed_days: List[str]
    
    # Experience Details
    photography_allowed: bool
    audio_guide_available: bool
    wheelchair_accessible: bool
    languages: List[str]
    
    # Local Tips
    local_tips: List[LocalTip]
    nearby_recommendations: List[str]
    
    # Ratings
    cultural_significance: float  # 0-1
    tourist_rating: float  # 0-5
    local_rating: float   # 0-5
    crowd_level: float    # 0-1, 1 being very crowded

class EnhancedMuseumRoutePlanner:
    """Museum-focused route planner with local expertise"""
    
    def __init__(self):
        self.museums = self._load_istanbul_museums()
        self.local_knowledge = self._load_local_knowledge()
        
    def _load_istanbul_museums(self) -> Dict[str, Museum]:
        """Load comprehensive museum database with local insights"""
        museums = {}
        
        # Major Archaeological Museums
        museums["archaeological_museum"] = Museum(
            id="archaeological_museum",
            name="Istanbul Archaeological Museums",
            category=MuseumCategory.ARCHAEOLOGICAL,
            coordinates=(41.0115, 28.9833),
            district="Fatih",
            neighborhood="Sultanahmet",
            description="Turkey's first museum with over 1 million artifacts spanning 5,000 years",
            highlights=[
                "Alexander Sarcophagus (4th century BC)",
                "Treaty of Kadesh - world's first peace treaty",
                "Babylon's Ishtar Gate glazed bricks",
                "Ancient Orient Museum collection",
                "Tiled Kiosk (Çinili Köşk) ceramics"
            ],
            visit_duration_minutes=120,
            entry_fee_tl=100,
            opening_hours={
                "tuesday": "09:00-19:00",
                "wednesday": "09:00-19:00", 
                "thursday": "09:00-19:00",
                "friday": "09:00-19:00",
                "saturday": "09:00-19:00",
                "sunday": "09:00-19:00",
                "monday": "CLOSED"
            },
            best_visit_times=["09:00-11:00", "16:00-18:00"],
            peak_hours=[11, 12, 13, 14, 15],
            closed_days=["monday"],
            photography_allowed=False,
            audio_guide_available=True,
            wheelchair_accessible=True,
            languages=["Turkish", "English", "German", "French"],
            cultural_significance=0.95,
            tourist_rating=4.3,
            local_rating=4.7,
            crowd_level=0.6,
            local_tips=[
                LocalTip(
                    category=LocalTipCategory.TIMING,
                    title="Early Morning Magic",
                    description="Visit right at 9 AM opening - you'll have the Alexander Sarcophagus almost to yourself for 30 minutes",
                    importance=5
                ),
                LocalTip(
                    category=LocalTipCategory.INSIDER_SECRET,
                    title="Hidden Garden Courtyard",
                    description="Don't miss the peaceful inner courtyard between buildings - perfect for a quiet break with Ottoman tombstones",
                    importance=4
                ),
                LocalTip(
                    category=LocalTipCategory.PHOTOGRAPHY,
                    title="Photography Restrictions",
                    description="No photos inside, but the museum courtyard and exterior are stunning for photos, especially the Tiled Kiosk",
                    importance=3
                ),
                LocalTip(
                    category=LocalTipCategory.FOOD_NEARBY,
                    title="Local Lunch Spot",
                    description="Skip tourist restaurants - walk 5 minutes to Hamdi Restaurant for amazing Ottoman cuisine locals love",
                    importance=4
                )
            ],
            nearby_recommendations=[
                "Topkapi Palace (5 min walk)",
                "Hagia Sophia (10 min walk)", 
                "Gülhane Park (adjacent)",
                "Sirkeci Train Station (historical)"
            ]
        )
        
        # Istanbul Modern
        museums["istanbul_modern"] = Museum(
            id="istanbul_modern",
            name="Istanbul Modern",
            category=MuseumCategory.ART_CONTEMPORARY,
            coordinates=(41.0256, 28.9744),
            district="Beyoğlu",
            neighborhood="Karaköy",
            description="Turkey's first modern and contemporary art museum with stunning Bosphorus views",
            highlights=[
                "Turkish contemporary art collection",
                "International rotating exhibitions",
                "Bosphorus panoramic views from café",
                "Photography and new media sections",
                "Educational workshops and talks"
            ],
            visit_duration_minutes=90,
            entry_fee_tl=120,
            opening_hours={
                "tuesday": "10:00-18:00",
                "wednesday": "10:00-18:00",
                "thursday": "10:00-20:00",
                "friday": "10:00-20:00", 
                "saturday": "10:00-18:00",
                "sunday": "10:00-18:00",
                "monday": "CLOSED"
            },
            best_visit_times=["10:00-12:00", "17:00-19:00"],
            peak_hours=[13, 14, 15, 16],
            closed_days=["monday"],
            photography_allowed=True,
            audio_guide_available=True,
            wheelchair_accessible=True,
            languages=["Turkish", "English"],
            cultural_significance=0.85,
            tourist_rating=4.2,
            local_rating=4.5,
            crowd_level=0.5,
            local_tips=[
                LocalTip(
                    category=LocalTipCategory.PHOTOGRAPHY,
                    title="Golden Hour Bosphorus Views",
                    description="The museum café terrace offers the best Bosphorus sunset photos in the city - come around 6 PM",
                    importance=5
                ),
                LocalTip(
                    category=LocalTipCategory.TIMING,
                    title="Thursday Late Night",
                    description="Thursday nights until 8 PM are perfect - fewer crowds and special evening lighting",
                    importance=4
                ),
                LocalTip(
                    category=LocalTipCategory.FOOD_NEARBY,
                    title="Karaköy Food Scene",
                    description="After the museum, explore Karaköy's trendy food scene - try Karaköy Lokantası for modern Turkish cuisine",
                    importance=4
                ),
                LocalTip(
                    category=LocalTipCategory.TRANSPORT,
                    title="Ferry Connection",
                    description="Take the ferry to/from Karaköy pier - it's a 3-minute walk and gives you beautiful Bosphorus views",
                    importance=3
                )
            ],
            nearby_recommendations=[
                "Galata Tower (10 min walk)",
                "Karaköy Pier (3 min walk)",
                "Salt Galata (cultural center)",
                "Galata Bridge (walking distance)"
            ]
        )
        
        # Topkapi Palace Museum
        museums["topkapi_palace"] = Museum(
            id="topkapi_palace",
            name="Topkapi Palace Museum",
            category=MuseumCategory.PALACE,
            coordinates=(41.0115, 28.9833),
            district="Fatih", 
            neighborhood="Sultanahmet",
            description="Ottoman sultans' primary residence for 400 years, now UNESCO World Heritage site",
            highlights=[
                "Sultan's private chambers and harem",
                "Sacred relics including Prophet Muhammad's cloak",
                "Imperial Treasury with famous diamonds",
                "Chinese and Japanese porcelain collection",
                "Panoramic Bosphorus and Golden Horn views"
            ],
            visit_duration_minutes=180,
            entry_fee_tl=200,  # Palace + Harem
            opening_hours={
                "tuesday": "09:00-18:00",
                "wednesday": "09:00-18:00",
                "thursday": "09:00-18:00", 
                "friday": "09:00-18:00",
                "saturday": "09:00-18:00",
                "sunday": "09:00-18:00",
                "monday": "CLOSED"
            },
            best_visit_times=["09:00-10:30", "16:00-17:30"],
            peak_hours=[10, 11, 12, 13, 14, 15],
            closed_days=["monday"],
            photography_allowed=True,  # Courtyards only
            audio_guide_available=True,
            wheelchair_accessible=False,  # Limited accessibility
            languages=["Turkish", "English", "German", "French", "Arabic"],
            cultural_significance=1.0,
            tourist_rating=4.4,
            local_rating=4.6,
            crowd_level=0.8,
            local_tips=[
                LocalTip(
                    category=LocalTipCategory.TIMING,
                    title="Beat the Crowds Strategy",
                    description="Enter at 9 AM sharp, go straight to Harem first (most crowded), then Treasury, then courtyards",
                    importance=5
                ),
                LocalTip(
                    category=LocalTipCategory.INSIDER_SECRET,
                    title="Fourth Courtyard Secret",
                    description="The Tulip Garden in the 4th courtyard is where sultans had their private moments - stunning views and usually empty",
                    importance=5
                ),
                LocalTip(
                    category=LocalTipCategory.PHOTOGRAPHY,
                    title="Best Photo Spots",
                    description="Second courtyard fountain, Bosphorus view from Baghdad Kiosk, and sunset from 4th courtyard terrace",
                    importance=4
                ),
                LocalTip(
                    category=LocalTipCategory.CULTURAL_ETIQUETTE,
                    title="Harem Respect",
                    description="The Harem was private women's quarters - maintain respectful quiet behavior and dress modestly",
                    importance=4
                )
            ],
            nearby_recommendations=[
                "Hagia Sophia (adjacent)",
                "Blue Mosque (5 min walk)",
                "Gülhane Park (adjacent)",
                "Archaeological Museum (same complex)"
            ]
        )
        
        # Pera Museum
        museums["pera_museum"] = Museum(
            id="pera_museum",
            name="Pera Museum",
            category=MuseumCategory.ART_CLASSICAL,
            coordinates=(41.0342, 28.9784),
            district="Beyoğlu",
            neighborhood="Tepebaşı",
            description="Private museum in historic building showcasing Orientalist paintings and Anatolian weights",
            highlights=[
                "Orientalist Painting Collection",
                "Anatolian Weights and Measures",
                "Kütahya Tiles and Ceramics",
                "Temporary international exhibitions",
                "Historic Hotel Bristol building architecture"
            ],
            visit_duration_minutes=75,
            entry_fee_tl=80,
            opening_hours={
                "tuesday": "10:00-19:00",
                "wednesday": "10:00-19:00",
                "thursday": "10:00-19:00",
                "friday": "10:00-22:00",
                "saturday": "10:00-19:00", 
                "sunday": "12:00-18:00",
                "monday": "CLOSED"
            },
            best_visit_times=["10:00-12:00", "17:00-19:00"],
            peak_hours=[13, 14, 15, 16],
            closed_days=["monday"],
            photography_allowed=False,
            audio_guide_available=True,
            wheelchair_accessible=True,
            languages=["Turkish", "English"],
            cultural_significance=0.8,
            tourist_rating=4.1,
            local_rating=4.4,
            crowd_level=0.4,
            local_tips=[
                LocalTip(
                    category=LocalTipCategory.TIMING,
                    title="Friday Evening Culture",
                    description="Friday evenings until 10 PM often have special events, talks, or music - check their program",
                    importance=4
                ),
                LocalTip(
                    category=LocalTipCategory.INSIDER_SECRET,
                    title="Museum Café Terrace",
                    description="The top floor café terrace has incredible views of Golden Horn - perfect for afternoon tea",
                    importance=4
                ),
                LocalTip(
                    category=LocalTipCategory.SHOPPING,
                    title="Unique Museum Shop",
                    description="Best museum shop in Istanbul for art books, jewelry inspired by collections, and unique Turkish design items",
                    importance=3
                ),
                LocalTip(
                    category=LocalTipCategory.FOOD_NEARBY,
                    title="Historic Istiklal Dining",
                    description="Walk to nearby Nevizade Street (5 min) for authentic Turkish meyhane (tavern) experience locals love",
                    importance=4
                )
            ],
            nearby_recommendations=[
                "Istiklal Street (3 min walk)",
                "Galatasaray High School (historic)",
                "Flower Passage (Çiçek Pasajı)",
                "French Cultural Center"
            ]
        )
        
        return museums
    
    def _load_local_knowledge(self) -> Dict[str, Any]:
        """Load local insider knowledge and tips"""
        return {
            "transportation_tips": {
                "sultanahmet_area": [
                    "Take tram T1 to Sultanahmet station - most convenient",
                    "Avoid taxis during peak hours (11 AM - 4 PM) in old city",
                    "Walking between Sultanahmet museums is pleasant and faster than transport"
                ],
                "beyoglu_area": [
                    "Metro M2 to Şişhane, then walk through Galata district", 
                    "Ferry to Karaköy pier is scenic and avoids traffic",
                    "Füniküler from Karaköy to Tünel connects to historic tram"
                ]
            },
            "seasonal_advice": {
                "summer": [
                    "Start museum visits early (9 AM) to avoid afternoon heat",
                    "Many museums have air conditioning - perfect midday retreat",
                    "Outdoor courtyards are beautiful but hot after 2 PM"
                ],
                "winter": [
                    "Museums are cozy refuges from cold and rain",
                    "Shorter daylight means indoor cultural activities are perfect",
                    "Less crowded - better experience for detailed viewing"
                ]
            },
            "cultural_etiquette": [
                "Dress modestly when visiting religious museums or palace harems",
                "Remove shoes when entering mosque sections of museums", 
                "Maintain quiet, respectful behavior in sacred spaces",
                "Tip: Learn a few Turkish phrases - locals appreciate the effort"
            ],
            "photography_guide": {
                "allowed_locations": [
                    "Museum courtyards and gardens",
                    "Exterior architecture", 
                    "Designated photography areas",
                    "Café and terrace areas"
                ],
                "restricted_areas": [
                    "Inside most galleries (flash damages artifacts)",
                    "Religious artifact displays",
                    "Private chambers in palaces",
                    "Areas with specific no-photo signs"
                ]
            }
        }
    
    async def create_museum_route(
        self, 
        preferences: Dict[str, Any],
        start_location: Tuple[float, float] = None,
        duration_hours: float = 6.0
    ) -> Dict[str, Any]:
        """Create optimized museum route with local tips"""
        
        logger.info(f"🏛️ Creating museum route for {duration_hours} hours")
        
        # Filter museums based on preferences
        selected_museums = self._select_museums_by_preferences(preferences)
        
        # Optimize route based on location and timing
        optimized_route = self._optimize_museum_route(
            selected_museums, 
            start_location, 
            duration_hours
        )
        
        # Add local tips and recommendations
        enhanced_route = self._enhance_route_with_local_tips(optimized_route, preferences)
        
        return enhanced_route
    
    def _select_museums_by_preferences(self, preferences: Dict[str, Any]) -> List[Museum]:
        """Select museums based on user preferences"""
        selected = []
        
        interests = preferences.get("interests", [])
        max_museums = preferences.get("max_museums", 4)
        budget_tl = preferences.get("budget_tl", 1000)
        accessibility_needed = preferences.get("accessibility_needed", False)
        
        # Interest-based filtering
        interest_mapping = {
            "history": [MuseumCategory.ARCHAEOLOGICAL, MuseumCategory.HISTORY, MuseumCategory.PALACE],
            "art": [MuseumCategory.ART_CLASSICAL, MuseumCategory.ART_CONTEMPORARY],
            "culture": [MuseumCategory.CULTURAL_HOUSE, MuseumCategory.RELIGIOUS],
            "photography": [MuseumCategory.PHOTOGRAPHY, MuseumCategory.ART_CONTEMPORARY]
        }
        
        target_categories = []
        for interest in interests:
            target_categories.extend(interest_mapping.get(interest, []))
        
        # Filter and score museums
        candidates = []
        for museum in self.museums.values():
            score = 0
            
            # Interest match
            if museum.category in target_categories:
                score += 3
            
            # Budget consideration
            if museum.entry_fee_tl <= budget_tl * 0.3:  # Max 30% of budget per museum
                score += 2
            
            # Accessibility
            if accessibility_needed and not museum.wheelchair_accessible:
                score -= 5
            elif accessibility_needed and museum.wheelchair_accessible:
                score += 2
            
            # Cultural significance
            score += museum.cultural_significance * 2
            
            # Local rating
            score += museum.local_rating * 0.5
            
            candidates.append((museum, score))
        
        # Sort by score and select top museums
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [museum for museum, score in candidates[:max_museums]]
        
        logger.info(f"📍 Selected {len(selected)} museums based on preferences")
        return selected
    
    def _optimize_museum_route(
        self, 
        museums: List[Museum], 
        start_location: Tuple[float, float],
        duration_hours: float
    ) -> Dict[str, Any]:
        """Optimize museum visiting order and timing"""
        
        if not museums:
            return {"museums": [], "total_duration": 0, "warnings": ["No museums selected"]}
        
        # Calculate distances and create route
        route_museums = []
        current_location = start_location or museums[0].coordinates
        remaining_museums = museums.copy()
        total_time_minutes = 0
        
        while remaining_museums and total_time_minutes < (duration_hours * 60):
            # Find nearest museum
            nearest_museum = min(
                remaining_museums,
                key=lambda m: self._calculate_distance(current_location, m.coordinates)
            )
            
            # Check if we have enough time
            if total_time_minutes + nearest_museum.visit_duration_minutes > (duration_hours * 60):
                break
            
            # Add travel time
            travel_time = self._estimate_travel_time(current_location, nearest_museum.coordinates)
            
            route_museums.append({
                "museum": nearest_museum,
                "arrival_time": self._format_time_from_minutes(total_time_minutes),
                "visit_duration": nearest_museum.visit_duration_minutes,
                "travel_time_minutes": travel_time
            })
            
            total_time_minutes += nearest_museum.visit_duration_minutes + travel_time
            current_location = nearest_museum.coordinates
            remaining_museums.remove(nearest_museum)
        
        return {
            "museums": route_museums,
            "total_duration_hours": total_time_minutes / 60,
            "total_cost_tl": sum(m["museum"].entry_fee_tl for m in route_museums),
            "skipped_museums": remaining_museums,
            "optimization_score": len(route_museums) / len(museums)
        }
    
    def _enhance_route_with_local_tips(
        self, 
        route: Dict[str, Any], 
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add local tips and recommendations to the route"""
        
        enhanced_route = route.copy()
        enhanced_route["local_recommendations"] = []
        enhanced_route["cultural_tips"] = []
        enhanced_route["photography_guide"] = []
        enhanced_route["food_recommendations"] = []
        
        # Collect tips from each museum
        for museum_info in route["museums"]:
            museum = museum_info["museum"]
            
            # Add museum-specific tips
            museum_info["local_tips"] = [
                {
                    "category": tip.category.value,
                    "title": tip.title,
                    "description": tip.description,
                    "importance": tip.importance
                }
                for tip in museum.local_tips
            ]
            
            # Collect high-importance tips for route summary
            for tip in museum.local_tips:
                if tip.importance >= 4:
                    if tip.category == LocalTipCategory.FOOD_NEARBY:
                        enhanced_route["food_recommendations"].append({
                            "location": museum.name,
                            "tip": tip.description
                        })
                    elif tip.category == LocalTipCategory.PHOTOGRAPHY:
                        enhanced_route["photography_guide"].append({
                            "location": museum.name,
                            "tip": tip.description
                        })
                    elif tip.category == LocalTipCategory.CULTURAL_ETIQUETTE:
                        enhanced_route["cultural_tips"].append(tip.description)
        
        # Add general local knowledge
        enhanced_route["transportation_tips"] = self.local_knowledge["transportation_tips"]
        enhanced_route["cultural_etiquette"] = self.local_knowledge["cultural_etiquette"]
        
        # Add seasonal advice if available
        current_season = self._get_current_season()
        if current_season in self.local_knowledge["seasonal_advice"]:
            enhanced_route["seasonal_tips"] = self.local_knowledge["seasonal_advice"][current_season]
        
        return enhanced_route
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates in kilometers"""
        import math
        
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return c * 6371  # Earth's radius
    
    def _estimate_travel_time(self, from_coord: Tuple[float, float], to_coord: Tuple[float, float]) -> int:
        """Estimate travel time in minutes between coordinates"""
        distance_km = self._calculate_distance(from_coord, to_coord)
        
        # Istanbul-specific travel time estimates
        if distance_km < 0.5:  # Walking distance
            return int(distance_km * 15)  # 15 min per km walking
        elif distance_km < 2:  # Tram/Metro
            return int(distance_km * 8) + 10  # 8 min per km + waiting
        else:  # Ferry or longer transport
            return int(distance_km * 6) + 15  # 6 min per km + waiting
    
    def _format_time_from_minutes(self, minutes: int) -> str:
        """Format minutes as HH:MM time string"""
        base_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        target_time = base_time + timedelta(minutes=minutes)
        return target_time.strftime("%H:%M")
    
    def _get_current_season(self) -> str:
        """Get current season for seasonal tips"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [3, 4, 5]:
            return "spring"
        else:
            return "autumn"

# Usage example
async def main():
    """Test the enhanced museum route planner"""
    planner = EnhancedMuseumRoutePlanner()
    
    preferences = {
        "interests": ["history", "art"],
        "max_museums": 3,
        "budget_tl": 500,
        "accessibility_needed": False,
        "duration_hours": 5
    }
    
    route = await planner.create_museum_route(preferences)
    
    print("🏛️ Enhanced Museum Route Plan")
    print("=" * 50)
    print(f"Duration: {route['total_duration_hours']:.1f} hours")
    print(f"Total Cost: {route['total_cost_tl']:.0f} TL")
    print(f"Museums: {len(route['museums'])}")
    
    for i, museum_info in enumerate(route['museums'], 1):
        museum = museum_info['museum']
        print(f"\n{i}. {museum.name}")
        print(f"   📍 {museum.district} - {museum.neighborhood}")
        print(f"   🕒 Arrival: {museum_info['arrival_time']}")
        print(f"   ⏱️ Visit: {museum.visit_duration_minutes} minutes")
        print(f"   💰 Entry: {museum.entry_fee_tl} TL")
        
        # Show top local tips
        top_tips = [tip for tip in museum.local_tips if tip.importance >= 4]
        if top_tips:
            print("   💡 Local Tips:")
            for tip in top_tips[:2]:  # Show top 2 tips
                print(f"      • {tip.title}: {tip.description}")

if __name__ == "__main__":
    asyncio.run(main())
