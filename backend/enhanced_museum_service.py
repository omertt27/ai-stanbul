#!/usr/bin/env python3
"""
Enhanced Museum Service for AI Istanbul
=====================================

Comprehensive museum information with rich content including:
- Detailed historical information
- Visitor practical information
- Cultural context and significance
- Interactive features and recommendations
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MuseumInfo:
    """Comprehensive museum information structure"""
    name: str
    name_turkish: str
    category: str
    location: str
    address: str
    coordinates: tuple
    description: str
    historical_significance: str
    opening_hours: Dict[str, str]
    admission_fees: Dict[str, float]
    estimated_visit_duration: str
    highlights: List[str]
    practical_tips: List[str]
    accessibility: Dict[str, bool]
    facilities: List[str]
    nearby_attractions: List[str]
    best_visiting_times: List[str]
    photography_rules: str
    cultural_notes: List[str]
    guided_tours: Dict[str, Any]
    official_website: Optional[str] = None
    phone: Optional[str] = None

class EnhancedMuseumService:
    """Enhanced museum service with comprehensive information"""
    
    def __init__(self):
        self.museums = self._load_enhanced_museum_data()
        self.cultural_sites = self._load_cultural_sites()
        self.special_exhibitions = self._load_special_exhibitions()
    
    def _load_enhanced_museum_data(self) -> Dict[str, MuseumInfo]:
        """Load comprehensive museum information"""
        museums = {}
        
        # Hagia Sophia
        museums["hagia_sophia"] = MuseumInfo(
            name="Hagia Sophia",
            name_turkish="Ayasofya Camii",
            category="Religious Monument/Museum",
            location="Sultanahmet",
            address="Sultan Ahmet, Ayasofya Meydanı No:1, 34122 Fatih/İstanbul",
            coordinates=(41.0086, 28.9802),
            description="A masterpiece of Byzantine architecture, Hagia Sophia served as a cathedral, mosque, museum, and now functions as a mosque again. Built in 537 AD, it was the world's largest cathedral for nearly 1000 years.",
            historical_significance="Built by Emperor Justinian I as the cathedral of Constantinople, Hagia Sophia represents the pinnacle of Byzantine architecture. Its massive dome was an engineering marvel of its time and influenced countless buildings across the Byzantine and Islamic worlds.",
            opening_hours={
                "daily": "09:00-19:00 (Prayer times excluded for non-Muslim visitors)",
                "friday": "Prayer times: 12:30-14:30 (closed to tourists)",
                "special_note": "5 daily prayer times affect visiting hours"
            },
            admission_fees={
                "general": 0.0,  # Free entry as it's now a functioning mosque
                "note": "Free entry, donations welcome"
            },
            estimated_visit_duration="1-2 hours",
            highlights=[
                "Massive central dome (31m diameter)",
                "Byzantine mosaics and Islamic calligraphy",
                "Emperor's Gate and Marble Door",
                "Weeping Column (Column of St. Gregory)",
                "Sultanate Lodge (Hünkar Mahfili)",
                "Mihrab and Minbar",
                "Imperial Gate mosaics"
            ],
            practical_tips=[
                "Dress modestly (covered shoulders, long pants/skirts)",
                "Remove shoes before entering",
                "Free plastic bags provided for shoes",
                "Avoid prayer times for interior visits",
                "Best lighting for photography: 10:00-15:00",
                "Audio guides available in multiple languages"
            ],
            accessibility={
                "wheelchair_accessible": True,
                "elevator": True,
                "ramps": True,
                "accessible_restrooms": True
            },
            facilities=[
                "Tourist information desk",
                "Gift shop",
                "Restrooms",
                "Security checkpoint",
                "Prayer facilities",
                "Ablution areas"
            ],
            nearby_attractions=[
                "Blue Mosque (2-minute walk)",
                "Topkapi Palace (5-minute walk)",
                "Basilica Cistern (3-minute walk)",
                "Sultanahmet Square",
                "German Fountain"
            ],
            best_visiting_times=[
                "Early morning (09:00-10:00) - fewer crowds",
                "Late afternoon (16:00-17:00) - golden light",
                "Avoid Friday midday prayers",
                "Spring and fall for comfortable weather"
            ],
            photography_rules="Photography allowed in designated areas, no flash during prayers",
            cultural_notes=[
                "Show respect during prayer times",
                "Understand both Christian and Islamic heritage",
                "Appreciate the architectural fusion of cultures",
                "Learn about Byzantine and Ottoman history"
            ],
            guided_tours={
                "available": True,
                "languages": ["Turkish", "English", "Arabic", "Russian", "German"],
                "duration": "45 minutes",
                "cost": "150 TL per group",
                "booking_required": True
            },
            official_website="https://ayasofyacamii.gov.tr",
            phone="+90 212 522 17 50"
        )
        
        # Topkapi Palace
        museums["topkapi_palace"] = MuseumInfo(
            name="Topkapi Palace",
            name_turkish="Topkapı Sarayı",
            category="Palace Museum",
            location="Sultanahmet",
            address="Cankurtaran, 34122 Fatih/İstanbul",
            coordinates=(41.0115, 28.9833),
            description="The primary residence of Ottoman sultans for approximately 400 years, Topkapi Palace is now a museum showcasing imperial collections, sacred relics, and Ottoman court life.",
            historical_significance="Built by Mehmed the Conqueror in 1459, Topkapi Palace served as the administrative center of the Ottoman Empire and home to sultans until the 19th century. It houses one of the world's most important collections of Islamic artifacts.",
            opening_hours={
                "summer": "09:00-18:45 (April-October)",
                "winter": "09:00-16:45 (November-March)",
                "closed": "Tuesdays",
                "last_entry": "1 hour before closing"
            },
            admission_fees={
                "palace_general": 200.0,
                "harem_additional": 120.0,
                "combined_ticket": 300.0,
                "student_discount": 50.0,
                "child_under_8": 0.0
            },
            estimated_visit_duration="3-4 hours (full palace), 2 hours (main sections)",
            highlights=[
                "Sacred Safekeeping Rooms (Holy Relics)",
                "Imperial Treasury (Topkapi Dagger, Spoonmaker's Diamond)",
                "Harem Quarters",
                "Imperial Council Chamber (Divan)",
                "Baghdad Kiosk",
                "Audience Hall",
                "Palace Kitchens (Chinese porcelain collection)",
                "View of Bosphorus from palace grounds"
            ],
            practical_tips=[
                "Buy tickets online to skip queues",
                "Visit Harem separately (additional ticket required)",
                "Start early to avoid crowds",
                "Wear comfortable walking shoes",
                "Bring water and snacks (long visit)",
                "Photography forbidden in some rooms",
                "Audio guide highly recommended"
            ],
            accessibility={
                "wheelchair_accessible": False,
                "limited_access": True,
                "stairs_many": True,
                "accessible_restrooms": True
            },
            facilities=[
                "Museum shop",
                "Restaurant with Bosphorus view",
                "Café",
                "Restrooms",
                "Audio guide rental",
                "Tourist information",
                "Security lockers"
            ],
            nearby_attractions=[
                "Hagia Sophia (5-minute walk)",
                "Blue Mosque (7-minute walk)",
                "Sultanahmet Archaeological Park",
                "Gülhane Park",
                "Istanbul Archaeology Museums"
            ],
            best_visiting_times=[
                "Wednesday-Friday mornings (less crowded)",
                "Late afternoon in summer (cooler)",
                "Spring and fall (ideal weather)",
                "Avoid weekends and holidays"
            ],
            photography_rules="Photography allowed in courtyards and some rooms, forbidden in Sacred Relics and some Treasury rooms",
            cultural_notes=[
                "Understand Ottoman palace hierarchy",
                "Respect the sacred nature of Holy Relics",
                "Appreciate Ottoman artistic traditions",
                "Learn about harem life and women's quarters"
            ],
            guided_tours={
                "available": True,
                "languages": ["Turkish", "English", "German", "French", "Spanish"],
                "duration": "2.5 hours",
                "cost": "300 TL per group",
                "booking_required": True,
                "specialty_tours": ["Harem tour", "Treasury focus", "Architecture tour"]
            },
            official_website="https://topkapisarayi.gov.tr",
            phone="+90 212 512 04 80"
        )
        
        # Blue Mosque
        museums["blue_mosque"] = MuseumInfo(
            name="Blue Mosque",
            name_turkish="Sultan Ahmet Camii",
            category="Active Mosque/Historic Monument",
            location="Sultanahmet",
            address="Sultan Ahmet, Atmeydanı Cd. No:7, 34122 Fatih/İstanbul",
            coordinates=(41.0054, 28.9766),
            description="Built between 1609-1616 during the reign of Ahmed I, the Blue Mosque is famous for its six minarets and stunning blue Iznik tiles that give it its popular name.",
            historical_significance="Designed by Sedefkâr Mehmed Ağa, the Blue Mosque represents the culmination of classical Ottoman architecture. Its six minarets caused controversy as only Mecca's mosque had six minarets at the time.",
            opening_hours={
                "daily": "08:30-11:30, 13:00-14:30, 15:30-16:30, 17:30-18:30 (outside prayer times)",
                "friday": "14:30-16:30, 17:30-18:30 (limited visiting hours)",
                "note": "Closed during 5 daily prayer times"
            },
            admission_fees={
                "general": 0.0,
                "note": "Free entry, donations welcome"
            },
            estimated_visit_duration="30-45 minutes",
            highlights=[
                "Six minarets (unique feature)",
                "Over 20,000 handmade blue Iznik tiles",
                "Massive central dome and semi-domes",
                "260 stained glass windows",
                "Intricate Islamic calligraphy",
                "Ornate mihrab and minbar",
                "Beautiful carpet covering the floor"
            ],
            practical_tips=[
                "Dress very modestly (long pants/skirts, covered shoulders)",
                "Remove shoes before entering (bags provided)",
                "Avoid prayer times completely",
                "Enter through tourist entrance (not main entrance)",
                "Be quiet and respectful inside",
                "No pointing feet toward Mecca direction",
                "Best photos from Sultanahmet Square at sunset"
            ],
            accessibility={
                "wheelchair_accessible": True,
                "ramps": True,
                "accessible_entrance": True,
                "accessible_restrooms": True
            },
            facilities=[
                "Shoe storage area",
                "Tourist information point",
                "Ablution facilities",
                "Restrooms",
                "Small gift area",
                "Prayer facilities"
            ],
            nearby_attractions=[
                "Hagia Sophia (2-minute walk)",
                "Sultanahmet Square",
                "Hippodrome of Constantinople",
                "German Fountain",
                "Museum of Turkish and Islamic Arts"
            ],
            best_visiting_times=[
                "Early morning (08:30-09:30)",
                "Between afternoon prayers (15:30-16:30)",
                "Spring and fall for comfortable weather",
                "Avoid Friday afternoons"
            ],
            photography_rules="Photography allowed but be respectful, no flash, avoid photographing people praying",
            cultural_notes=[
                "Active place of worship - show utmost respect",
                "Understand Islamic prayer practices",
                "Appreciate Ottoman architectural achievement",
                "Learn about mosque etiquette and customs"
            ],
            guided_tours={
                "available": True,
                "languages": ["Turkish", "English", "Arabic"],
                "duration": "30 minutes",
                "cost": "100 TL per group",
                "booking_required": False,
                "note": "Tours must respect prayer times"
            },
            official_website="https://sultanahmetcamii.org/",
            phone="+90 212 458 44 68"
        )
        
        return museums
    
    def _load_cultural_sites(self) -> Dict[str, Any]:
        """Load additional cultural sites information"""
        return {
            "basilica_cistern": {
                "name": "Basilica Cistern",
                "highlights": ["Medusa column bases", "336 marble columns", "Atmospheric lighting"],
                "historical_period": "Byzantine (532 AD)",
                "unique_features": ["Underground water reservoir", "Acoustic properties", "Column forest"]
            },
            "galata_tower": {
                "name": "Galata Tower",
                "highlights": ["360-degree city panorama", "Genoese architecture", "Restaurant and café"],
                "historical_period": "Genoese (1348)",
                "unique_features": ["67m height", "Panoramic elevator", "Historic observation deck"]
            },
            "dolmabahce_palace": {
                "name": "Dolmabahçe Palace",
                "highlights": ["European-style Ottoman palace", "Crystal staircase", "World's largest Bohemian crystal chandelier"],
                "historical_period": "Ottoman 19th century",
                "unique_features": ["Atatürk's residence", "Baroque and Neoclassical mix", "Bosphorus location"]
            }
        }
    
    def _load_special_exhibitions(self) -> Dict[str, Any]:
        """Load current and upcoming special exhibitions"""
        return {
            "current_exhibitions": [
                {
                    "title": "Byzantine Mosaics: Hidden Treasures",
                    "location": "Hagia Sophia",
                    "duration": "2024-2025",
                    "description": "Restoration insights and newly discovered mosaics"
                },
                {
                    "title": "Ottoman Court Life",
                    "location": "Topkapi Palace",
                    "duration": "Permanent enhanced display",
                    "description": "Interactive exhibits about daily palace life"
                }
            ],
            "seasonal_events": [
                {
                    "title": "Museum Night",
                    "frequency": "Annually in May",
                    "description": "Free or discounted entry to multiple museums"
                },
                {
                    "title": "Cultural Heritage Days",
                    "frequency": "September",
                    "description": "Special tours and events across historic sites"
                }
            ]
        }
    
    def get_museum_info(self, museum_name: str) -> Optional[MuseumInfo]:
        """Get detailed information about a specific museum"""
        museum_key = museum_name.lower().replace(" ", "_").replace("'", "")
        return self.museums.get(museum_key)
    
    def get_museum_recommendations(self, interests: List[str], time_available: str, location: str = None) -> List[Dict[str, Any]]:
        """Get personalized museum recommendations"""
        recommendations = []
        
        # Filter museums based on interests and location
        for key, museum in self.museums.items():
            if location and location.lower() not in museum.location.lower():
                continue
            
            score = 0
            reasons = []
            
            # Score based on interests
            if "history" in interests:
                score += 3
                reasons.append("Rich historical significance")
            
            if "architecture" in interests:
                score += 2
                reasons.append("Architectural masterpiece")
            
            if "religion" in interests and "mosque" in museum.category.lower():
                score += 2
                reasons.append("Religious and cultural importance")
            
            if "art" in interests and any(word in museum.description.lower() for word in ["art", "mosaic", "painting"]):
                score += 2
                reasons.append("Exceptional artistic collections")
            
            # Consider time availability
            duration_match = False
            if time_available == "short" and "30" in museum.estimated_visit_duration:
                duration_match = True
            elif time_available == "medium" and any(x in museum.estimated_visit_duration for x in ["1-2", "2"]):
                duration_match = True
            elif time_available == "long" and any(x in museum.estimated_visit_duration for x in ["3-4", "4"]):
                duration_match = True
            
            if duration_match:
                score += 1
                reasons.append(f"Perfect for {time_available} visits")
            
            if score > 0:
                recommendations.append({
                    "museum": museum,
                    "score": score,
                    "reasons": reasons,
                    "practical_info": {
                        "duration": museum.estimated_visit_duration,
                        "cost": museum.admission_fees.get("general", 0),
                        "best_time": museum.best_visiting_times[0] if museum.best_visiting_times else "Morning"
                    }
                })
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:5]
    
    def get_visiting_plan(self, museums: List[str], preferred_day: str = None) -> Dict[str, Any]:
        """Create an optimized visiting plan for multiple museums"""
        plan = {
            "total_duration": "0 hours",
            "total_cost": 0.0,
            "recommended_route": [],
            "timing_suggestions": [],
            "important_notes": []
        }
        
        total_duration_hours = 0
        total_cost = 0.0
        
        for museum_name in museums:
            museum = self.get_museum_info(museum_name)
            if museum:
                # Extract duration in hours
                duration_str = museum.estimated_visit_duration
                if "1-2" in duration_str:
                    duration_hours = 1.5
                elif "3-4" in duration_str:
                    duration_hours = 3.5
                elif "30" in duration_str:
                    duration_hours = 0.5
                else:
                    duration_hours = 2.0
                
                total_duration_hours += duration_hours
                total_cost += museum.admission_fees.get("general", 0)
                
                plan["recommended_route"].append({
                    "museum": museum.name,
                    "duration": museum.estimated_visit_duration,
                    "cost": museum.admission_fees.get("general", 0),
                    "key_highlights": museum.highlights[:3],
                    "practical_tips": museum.practical_tips[:2]
                })
        
        plan["total_duration"] = f"{total_duration_hours:.1f} hours"
        plan["total_cost"] = total_cost
        
        # Add timing suggestions
        if total_duration_hours > 6:
            plan["timing_suggestions"].append("Consider splitting across 2 days")
        if total_duration_hours > 8:
            plan["timing_suggestions"].append("Definitely plan for 2-3 days")
        
        plan["timing_suggestions"].extend([
            "Start early (09:00) to avoid crowds",
            "Book tickets online when possible",
            "Check prayer times for mosque visits"
        ])
        
        # Add important notes
        plan["important_notes"] = [
            "Dress modestly for religious sites",
            "Bring comfortable walking shoes",
            "Stay hydrated and take breaks",
            "Respect photography rules",
            "Check opening hours and days"
        ]
        
        return plan
    
    def get_museum_info(self, query: str, location: str = None) -> Dict[str, Any]:
        """Wrapper method for main.py integration - get museum information based on query"""
        try:
            museums = []
            cultural_insights = ""
            
            # Extract interests from query (simplified NLP approach)
            interests = []
            if any(word in query.lower() for word in ['history', 'historical']):
                interests.append('history')
            if any(word in query.lower() for word in ['art', 'painting', 'sculpture']):
                interests.append('art')
            if any(word in query.lower() for word in ['architecture', 'building']):
                interests.append('architecture')
            if any(word in query.lower() for word in ['islamic', 'mosque', 'religion']):
                interests.append('religion')
            
            # Get recommendations based on interests
            recommendations = self.get_museum_recommendations(interests or ['history'], "half_day", location)
            
            # Convert MuseumInfo objects to dictionaries for JSON serialization
            for rec in recommendations[:4]:  # Top 4 recommendations
                museum_key = rec.get('museum_name', '').lower().replace(' ', '_').replace("'", "")
                museum_info = self.museums.get(museum_key)
                
                if museum_info:
                    museums.append({
                        'name': museum_info.name,
                        'location': museum_info.location,
                        'highlights': museum_info.description[:200] + "..." if len(museum_info.description) > 200 else museum_info.description,
                        'practical_info': f"Hours: {museum_info.opening_hours} | Entry: {museum_info.entrance_fee}",
                        'cultural_context': museum_info.cultural_significance,
                        'category': museum_info.category,
                        'rating': rec.get('score', 3.5),
                        'recommendation_reason': ', '.join(rec.get('reasons', []))
                    })
            
            # Add cultural insights
            if location and location.lower() in ['sultanahmet', 'historic peninsula']:
                cultural_insights = "Sultanahmet is the heart of Byzantine and Ottoman Istanbul, where you'll find the most significant historical monuments. Plan to spend a full day here and wear comfortable walking shoes."
            elif interests and 'art' in interests:
                cultural_insights = "Istanbul's art scene beautifully blends traditional Islamic art with contemporary works. Many museums offer both historical artifacts and modern interpretations of Turkish culture."
            else:
                cultural_insights = "Istanbul's museums tell the story of a city that has been the capital of three empires. Each site offers unique insights into the layers of history that make Istanbul special."
            
            return {
                "success": True,
                "museums": museums,
                "cultural_insights": cultural_insights,
                "query_processed": query,
                "location_context": location,
                "interests_identified": interests
            }
            
        except Exception as e:
            logger.error(f"Error in get_museum_info wrapper: {e}")
            return {
                "success": False,
                "error": str(e),
                "museums": [],
                "cultural_insights": ""
            }
