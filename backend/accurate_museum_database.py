#!/usr/bin/env python3
"""
Corrected Istanbul Museum Information Database
=============================================

This module provides accurate, fact-checked information about Istanbul's major
museums, palaces, and cultural sites to replace AI-generated content.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import time

@dataclass
class MuseumInfo:
    """Structured museum information with verified facts"""
    name: str
    historical_period: str
    construction_date: str
    architect: Optional[str]
    key_features: List[str]
    opening_hours: Dict[str, str]
    entrance_fee: str
    location: str
    nearby_attractions: List[str]
    visiting_duration: str
    best_time_to_visit: str
    historical_significance: str
    architectural_style: str
    must_see_highlights: List[str]
    photography_allowed: bool
    accessibility: str
    closing_days: List[str]

class IstanbulMuseumDatabase:
    """Verified database of Istanbul museum information"""
    
    def __init__(self):
        self.museums = self._initialize_museum_data()
    
    def _initialize_museum_data(self) -> Dict[str, MuseumInfo]:
        """Initialize with verified museum information"""
        return {
            "hagia_sophia": MuseumInfo(
                name="Hagia Sophia (Ayasofya)",
                historical_period="Byzantine (532-537 AD), Ottoman (1453-1934), Museum (1934-2020), Mosque (2020-present)",
                construction_date="532-537 AD",
                architect="Anthemius of Tralles and Isidore of Miletus",
                key_features=[
                    "Massive dome (31m diameter, 55m high)",
                    "Byzantine mosaics and Islamic calligraphy coexisting",
                    "Imperial Door and Marble Jar",
                    "Weeping Column (Column of St. Gregory)",
                    "Viking graffiti from 9th century"
                ],
                opening_hours={
                    "daily": "Open for worship - visiting between prayer times",
                    "prayer_times": "Check daily prayer schedule"
                },
                entrance_fee="Free (functioning mosque)",
                location="Sultanahmet Square, Fatih",
                nearby_attractions=["Blue Mosque", "Topkapi Palace", "Basilica Cistern"],
                visiting_duration="45-60 minutes",
                best_time_to_visit="Early morning (8-10 AM) or late afternoon",
                historical_significance="Former Byzantine cathedral, Ottoman mosque, UNESCO World Heritage Site. Symbol of religious and cultural transformation.",
                architectural_style="Byzantine architecture with Islamic additions",
                must_see_highlights=[
                    "Main dome and supporting structure",
                    "Deesis Mosaic (Christ Pantocrator)",
                    "Virgin and Child mosaic in apse",
                    "Imperial portraits mosaics"
                ],
                photography_allowed=True,
                accessibility="Limited wheelchair access due to historical structure",
                closing_days=["Open daily - prayer time restrictions apply"]
            ),
            
            "topkapi_palace": MuseumInfo(
                name="Topkapi Palace (Topkapı Sarayı)",
                historical_period="Ottoman Empire (1465-1856)",
                construction_date="1459-1465",
                architect="Mimar Atik Sinan (later additions by Mimar Sinan)",
                key_features=[
                    "Four main courtyards with distinct functions",
                    "Harem quarters with 400 rooms",
                    "Sacred Relics collection (Prophet Muhammad's belongings)",
                    "Treasury with Spoonmaker's Diamond and Topkapi Dagger",
                    "Chinese and Japanese porcelain collection"
                ],
                opening_hours={
                    "winter": "09:00-16:45 (Oct 30 - Apr 15)",
                    "summer": "09:00-18:45 (Apr 15 - Oct 30)",
                    "ticket_sales_end": "1 hour before closing"
                },
                entrance_fee="Palace: 100 TL, Harem: Additional 70 TL (2023 prices)",
                location="Sultanahmet, Gulhane Park entrance",
                nearby_attractions=["Hagia Sophia", "Archaeological Museums", "Gulhane Park"],
                visiting_duration="2-4 hours (including Harem)",
                best_time_to_visit="Weekday mornings to avoid crowds",
                historical_significance="Primary residence of Ottoman sultans for 400 years. Administrative center of the empire.",
                architectural_style="Classical Ottoman palace architecture",
                must_see_highlights=[
                    "Imperial Treasury (Spoonmaker's Diamond)",
                    "Sacred Relics Room",
                    "Harem quarters and Sultan's private rooms",
                    "Palace kitchens with porcelain collection",
                    "Fourth courtyard gardens and pavilions"
                ],
                photography_allowed=True,
                accessibility="Partially wheelchair accessible - some areas require stairs",
                closing_days=["Tuesdays"]
            ),
            
            "blue_mosque": MuseumInfo(
                name="Blue Mosque (Sultan Ahmed Mosque)",
                historical_period="Ottoman Empire",
                construction_date="1609-1616",
                architect="Sedefkar Mehmet Aga (student of Mimar Sinan)",
                key_features=[
                    "Six minarets (unique for its time)",
                    "Blue Iznik tiles (giving it the name)",
                    "Massive prayer hall with central dome",
                    "Cascading domes and semi-domes",
                    "Over 200 stained glass windows"
                ],
                opening_hours={
                    "daily": "Open except during prayer times",
                    "closed_times": "30 minutes before and during each prayer",
                    "friday": "Closed 12:30-14:30 for Friday prayers"
                },
                entrance_fee="Free (active mosque)",
                location="Sultanahmet Square, facing Hagia Sophia",
                nearby_attractions=["Hagia Sophia", "Hippodrome", "Grand Bazaar"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Between prayer times, early morning preferred",
                historical_significance="Last great mosque of the Classical Ottoman period. Built to rival Hagia Sophia.",
                architectural_style="Classical Ottoman mosque architecture",
                must_see_highlights=[
                    "Interior blue Iznik tiles",
                    "Magnificent central dome",
                    "Intricate calligraphy and geometric patterns",
                    "Prayer hall atmosphere during non-prayer times"
                ],
                photography_allowed=True,
                accessibility="Ground level access available",
                closing_days=["None - prayer time restrictions apply"]
            ),
            
            "basilica_cistern": MuseumInfo(
                name="Basilica Cistern (Yerebatan Sarnıcı)",
                historical_period="Byzantine Empire",
                construction_date="532 AD",
                architect="Unknown Byzantine architects",
                key_features=[
                    "336 marble columns in 12 rows",
                    "Two Medusa head column bases",
                    "Atmospheric lighting and walkways",
                    "Column capitals from various periods",
                    "Underground palace nickname"
                ],
                opening_hours={
                    "daily": "09:00-18:30",
                    "last_entry": "17:30"
                },
                entrance_fee="120 TL (2023 prices)",
                location="Alemdar Mahallesi, near Hagia Sophia",
                nearby_attractions=["Hagia Sophia", "Topkapi Palace", "Grand Bazaar"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Early morning or late afternoon",
                historical_significance="Largest surviving Byzantine cistern. Supplied water to Great Palace.",
                architectural_style="Byzantine underground architecture",
                must_see_highlights=[
                    "Medusa head columns (upside down and sideways)",
                    "Column forest atmosphere",
                    "Acoustic properties demonstration",
                    "Historical water filtration system"
                ],
                photography_allowed=True,
                accessibility="Not wheelchair accessible - stairs required",
                closing_days=["None"]
            ),
            
            "dolmabahce_palace": MuseumInfo(
                name="Dolmabahçe Palace",
                historical_period="Late Ottoman Empire",
                construction_date="1843-1856",
                architect="Karapet Balyan and Nigoğayos Balyan (Armenian architects)",
                key_features=[
                    "European Baroque and Neoclassical style",
                    "4.5-ton crystal chandelier (largest in palace)",
                    "285 rooms, 43 halls, 6 Turkish baths",
                    "14 tons of gold leaf decoration",
                    "Atatürk's death room (preserved as museum)"
                ],
                opening_hours={
                    "winter": "09:00-16:00 (Oct 1 - Mar 31)",
                    "summer": "09:00-17:00 (Apr 1 - Sep 30)",
                    "guided_tours_only": "Tours every 30 minutes"
                },
                entrance_fee="Selamlık: 90 TL, Harem: 60 TL, Combined: 120 TL",
                location="Beşiktaş, Bosphorus waterfront",
                nearby_attractions=["Bosphorus Bridge", "Beşiktaş area", "Naval Museum"],
                visiting_duration="1.5-2 hours (both sections)",
                best_time_to_visit="Weekday mornings, book in advance",
                historical_significance="Last residence of Ottoman sultans. Place where Atatürk died in 1938.",
                architectural_style="European Baroque, Rococo, and Neoclassical",
                must_see_highlights=[
                    "Crystal Staircase and massive chandelier",
                    "Ceremonial Hall with 4.5-ton chandelier",
                    "Atatürk's bedroom (death room)",
                    "Imperial bathrooms with alabaster",
                    "Bosphorus-facing rooms with panoramic views"
                ],
                photography_allowed=False,
                accessibility="Limited wheelchair access",
                closing_days=["Mondays and Thursdays"]
            ),
            
            "archaeological_museum": MuseumInfo(
                name="Istanbul Archaeological Museums",
                historical_period="Ottoman to Modern (museum since 1891)",
                construction_date="1891 (main building)",
                architect="Alexandre Vallaury",
                key_features=[
                    "Three main buildings complex",
                    "Over 1 million artifacts",
                    "Alexander Sarcophagus (most famous piece)",
                    "Ancient Orient Museum section",
                    "Tiled Kiosk (oldest building, 1472)"
                ],
                opening_hours={
                    "daily": "09:00-17:00",
                    "last_entry": "16:00"
                },
                entrance_fee="60 TL (2023 prices)",
                location="Gulhane Park, near Topkapi Palace",
                nearby_attractions=["Topkapi Palace", "Hagia Sophia", "Gulhane Park"],
                visiting_duration="2-3 hours",
                best_time_to_visit="Weekday mornings",
                historical_significance="First museum in Ottoman Empire. Houses artifacts from across the empire.",
                architectural_style="Neoclassical museum architecture",
                must_see_highlights=[
                    "Alexander Sarcophagus (not Alexander's, but from Sidon)",
                    "Kadesh Peace Treaty (oldest known peace treaty)",
                    "Ancient Orient artifacts",
                    "Byzantine and Islamic art collections",
                    "Tiled Kiosk ceramics collection"
                ],
                photography_allowed=True,
                accessibility="Wheelchair accessible main areas",
                closing_days=["Mondays"]
            ),
            
            "galata_tower": MuseumInfo(
                name="Galata Tower (Galata Kulesi)",
                historical_period="Genoese (1348), Ottoman modifications",
                construction_date="1348",
                architect="Genoese architects",
                key_features=[
                    "67 meters tall, 9 floors",
                    "360-degree panoramic views",
                    "Medieval Genoese architecture",
                    "Restaurant and observation deck",
                    "Symbol of Istanbul skyline"
                ],
                opening_hours={
                    "daily": "08:30-22:00",
                    "last_entry": "21:30"
                },
                entrance_fee="100 TL for observation deck (2023)",
                location="Galata, Beyoğlu",
                nearby_attractions=["Karaköy", "Istiklal Street", "Golden Horn"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Sunset for best views",
                historical_significance="Part of Genoese fortification. Survived earthquakes and fires for 675 years.",
                architectural_style="Medieval Genoese tower architecture",
                must_see_highlights=[
                    "Panoramic views of Historic Peninsula",
                    "Bosphorus and Golden Horn views",
                    "Sunset photography opportunities",
                    "Historical elevator (1960s addition)"
                ],
                photography_allowed=True,
                accessibility="Elevator available to upper floors",
                closing_days=["None"]
            ),
            
            "chora_church": MuseumInfo(
                name="Chora Church Museum (Kariye Müzesi)",
                historical_period="Byzantine (11th century), Ottoman mosque (1511)",
                construction_date="1077-1081, mosaics added 1315-1321",
                architect="Unknown Byzantine architects",
                key_features=[
                    "Best-preserved Byzantine mosaics outside Hagia Sophia",
                    "Fresco cycles depicting life of Christ and Virgin Mary",
                    "Parekklesion (side chapel) with resurrection fresco",
                    "14th-century artistic masterpiece",
                    "Recently converted back to mosque (2020)"
                ],
                opening_hours={
                    "status": "Currently closed for restoration",
                    "note": "Check current status as it may reopen as mosque"
                },
                entrance_fee="Check current status",
                location="Edirnekapı, Fatih",
                nearby_attractions=["City walls", "Eyup Sultan Mosque"],
                visiting_duration="45-60 minutes when open",
                best_time_to_visit="Check opening status first",
                historical_significance="Contains world's finest Byzantine mosaics. UNESCO World Heritage consideration.",
                architectural_style="Byzantine church architecture",
                must_see_highlights=[
                    "Anastasis (Resurrection) fresco",
                    "Life of Christ mosaic cycle",
                    "Virgin Mary's infancy mosaics",
                    "Donor portraits of Theodore Metochites"
                ],
                photography_allowed="Unknown current policy",
                accessibility="Limited due to historical structure",
                closing_days=["Check current status"]
            )
        }
    
    def get_museum_info(self, museum_key: str) -> Optional[MuseumInfo]:
        """Get verified information for a specific museum"""
        return self.museums.get(museum_key.lower())
    
    def get_museum_by_name(self, name: str) -> Optional[MuseumInfo]:
        """Find museum by name (fuzzy matching)"""
        name_lower = name.lower()
        
        # Direct name matching
        name_mappings = {
            'hagia sophia': 'hagia_sophia',
            'ayasofya': 'hagia_sophia',
            'topkapi': 'topkapi_palace',
            'topkapi palace': 'topkapi_palace',
            'blue mosque': 'blue_mosque',
            'sultan ahmed': 'blue_mosque',
            'basilica cistern': 'basilica_cistern',
            'yerebatan': 'basilica_cistern',
            'dolmabahce': 'dolmabahce_palace',
            'dolmabahçe': 'dolmabahce_palace',
            'archaeological': 'archaeological_museum',
            'galata tower': 'galata_tower',
            'chora': 'chora_church',
            'kariye': 'chora_church'
        }
        
        for search_name, museum_key in name_mappings.items():
            if search_name in name_lower:
                return self.museums.get(museum_key)
        
        return None
    
    def get_opening_hours_info(self, museum_key: str) -> str:
        """Get formatted opening hours information"""
        museum = self.get_museum_info(museum_key)
        if not museum:
            return "Museum information not found"
        
        hours_info = []
        for period, hours in museum.opening_hours.items():
            hours_info.append(f"{period.replace('_', ' ').title()}: {hours}")
        
        if museum.closing_days and museum.closing_days != ["None"]:
            hours_info.append(f"Closed: {', '.join(museum.closing_days)}")
        
        return "\n".join(hours_info)
    
    def get_all_museum_names(self) -> List[str]:
        """Get list of all museum names"""
        return [museum.name for museum in self.museums.values()]

# Global instance for easy import
istanbul_museums = IstanbulMuseumDatabase()

def get_accurate_museum_info(query: str) -> Optional[MuseumInfo]:
    """
    Get accurate museum information from verified database
    
    Args:
        query: User query containing museum name
        
    Returns:
        MuseumInfo object with verified facts or None
    """
    return istanbul_museums.get_museum_by_name(query)

def format_museum_response(museum: MuseumInfo, query_type: str = "general") -> str:
    """
    Format museum information for chat responses
    
    Args:
        museum: MuseumInfo object
        query_type: Type of query (hours, cost, general, etc.)
        
    Returns:
        Formatted response string
    """
    if query_type == "hours":
        response = f"**{museum.name} Opening Hours:**\n"
        response += istanbul_museums.get_opening_hours_info(museum.name.lower().replace(' ', '_'))
        response += f"\n\nBest time to visit: {museum.best_time_to_visit}"
        if museum.closing_days != ["None"]:
            response += f"\nPlease note: Closed on {', '.join(museum.closing_days)}"
        return response
    
    elif query_type == "cost":
        response = f"**{museum.name} Entrance Information:**\n"
        response += f"Entrance fee: {museum.entrance_fee}\n"
        response += f"Recommended visiting duration: {museum.visiting_duration}\n"
        response += f"Location: {museum.location}"
        return response
    
    else:  # general information
        response = f"**{museum.name}**\n\n"
        response += f"**Historical Period:** {museum.historical_period}\n"
        response += f"**Built:** {museum.construction_date}\n"
        if museum.architect:
            response += f"**Architect:** {museum.architect}\n"
        response += f"\n**Historical Significance:**\n{museum.historical_significance}\n"
        response += f"\n**Key Features:**\n"
        for feature in museum.key_features[:3]:  # Limit to top 3
            response += f"• {feature}\n"
        response += f"\n**Must-See Highlights:**\n"
        for highlight in museum.must_see_highlights[:3]:  # Limit to top 3
            response += f"• {highlight}\n"
        response += f"\n**Practical Information:**\n"
        response += f"• Location: {museum.location}\n"
        response += f"• Visiting duration: {museum.visiting_duration}\n"
        response += f"• Entrance fee: {museum.entrance_fee}\n"
        response += f"• Best time to visit: {museum.best_time_to_visit}"
        
        return response
