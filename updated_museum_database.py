#!/usr/bin/env python3
"""
Updated Istanbul Museum Information Database
==========================================

Updated structure without fee prices and with MuseumPass information.
Opening hours to be verified from Google Maps.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import time

@dataclass
class MuseumInfo:
    """Updated museum information structure without fees"""
    name: str
    name_turkish: str
    historical_period: str
    construction_date: str
    architect: Optional[str]
    key_features: List[str]
    opening_hours: Dict[str, str]  # To be updated from Google Maps
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
    museum_pass_valid: bool
    museum_pass_notes: str
    audio_guide_available: bool
    virtual_museum_available: bool
    virtual_museum_url: Optional[str]
    administration: str
    funding_type: str
    museum_type: str
    icomos_cards_valid: bool
    press_cards_valid: bool

class MuseumPassInfo:
    """MuseumPass Istanbul E-Card Information"""
    
    @staticmethod
    def get_museum_pass_info() -> str:
        return """
🎫 **MuseumPass İstanbul E-Card**

**What is MuseumPass İstanbul?**
With MuseumPass Istanbul, you can visit 13 museums that belong to Türkiye Ministry of Culture and Tourism and enjoy your journey through history.

**Key Details:**
• **Validity**: 5 days starting from your first museum visit
• **Usage**: You can enter each museum once with the card
• **Time Restrictions**: 
  - Galata Tower: Entry no later than 18:14
  - Istanbul Archaeological Museums: Entry no later than 18:45
  - Museum of Turkish and Islamic Art: Entry no later than 18:45
• **Not Valid**: Night museums after 19:00

**Museums Included:**
1. Istanbul Archaeological Museums
2. Topkapi Palace Museum
3. Museum of Turkish and Islamic Arts
4. Galata Tower Museum
5. Galata Mevlevi House Museum
6. Rumeli Fortress Museum
7. Maiden's Tower Museum
8. Hagia Irene Museum
9. Great Palace Mosaics Museum
10. Museum of the History of Science and Technology in Islam
11. Fethiye (Pammakaristos) Museum
12. Chora Museum
13. And other participating museums

**Important Notes:**
• Check museum status (restoration/open/closed) before visiting
• Not valid for Night Museology programs
• Purchase at official MuseumPass sales points
"""

class UpdatedIstanbulMuseumDatabase:
    """Updated database without fee pricing"""
    
    def __init__(self):
        self.museums = self._initialize_updated_museum_data()
        self.museum_pass_info = MuseumPassInfo.get_museum_pass_info()
    
    def _initialize_updated_museum_data(self) -> Dict[str, MuseumInfo]:
        """Initialize with updated museum information"""
        return {
            "istanbul_archaeological_museums": MuseumInfo(
                name="Istanbul Archaeological Museums",
                name_turkish="İstanbul Arkeoloji Müzeleri",
                historical_period="Founded 1891",
                construction_date="1891",
                architect="Alexander Vallaury",
                key_features=[
                    "Three main buildings: Archaeological Museum, Ancient Orient Museum, Tiled Kiosk",
                    "Alexander Sarcophagus (4th century BC)",
                    "Treaty of Kadesh (oldest known peace treaty)",
                    "Extensive collection of ancient artifacts"
                ],
                opening_hours={
                    "winter": "09:00-17:00 (Oct 30 - Mar 31)",
                    "summer": "09:00-19:00 (Apr 1 - Oct 30)",
                    "note": "Hours to be verified from Google Maps"
                },
                location="Osman Hamdi Bey Yokuşu, Sultanahmet",
                nearby_attractions=["Topkapi Palace", "Gulhane Park", "Hagia Sophia"],
                visiting_duration="2-3 hours",
                best_time_to_visit="Weekday mornings",
                historical_significance="Turkey's first archaeological museum, houses priceless ancient artifacts",
                architectural_style="Neoclassical",
                must_see_highlights=[
                    "Alexander Sarcophagus",
                    "Sarcophagus of the Crying Women",
                    "Treaty of Kadesh",
                    "Ancient Orient artifacts"
                ],
                photography_allowed=True,
                accessibility="Partially accessible",
                closing_days=["Mondays"],
                museum_pass_valid=True,
                museum_pass_notes="Entry no later than 18:45 with MuseumPass",
                audio_guide_available=True,
                virtual_museum_available=True,
                virtual_museum_url="sanalmuze.gov.tr",
                administration="Provincial Culture and Tourism Directorate",
                funding_type="State Funded",
                museum_type="Archaeology Museum",
                icomos_cards_valid=True,
                press_cards_valid=True
            ),
            
            "topkapi_palace": MuseumInfo(
                name="Topkapi Palace Museum",
                name_turkish="Topkapı Sarayı Müzesi",
                historical_period="Ottoman Empire (1465-1856)",
                construction_date="1459-1465",
                architect="Mimar Atik Sinan (later additions by Mimar Sinan)",
                key_features=[
                    "Four main courtyards with distinct functions",
                    "Harem quarters with 400 rooms",
                    "Sacred Relics collection",
                    "Treasury with precious artifacts",
                    "Chinese and Japanese porcelain collection"
                ],
                opening_hours={
                    "winter": "09:00-16:45 (Oct 30 - Apr 15)",
                    "summer": "09:00-18:45 (Apr 15 - Oct 30)",
                    "note": "Hours to be verified from Google Maps"
                },
                location="Sultanahmet, Gulhane Park entrance",
                nearby_attractions=["Hagia Sophia", "Archaeological Museums", "Gulhane Park"],
                visiting_duration="2-4 hours (including Harem)",
                best_time_to_visit="Weekday mornings",
                historical_significance="Primary residence of Ottoman sultans for 400 years",
                architectural_style="Ottoman palace architecture",
                must_see_highlights=[
                    "Sacred Relics Chamber",
                    "Treasury collection",
                    "Harem quarters",
                    "Imperial Kitchens"
                ],
                photography_allowed=True,
                accessibility="Limited due to historical structure",
                closing_days=["Tuesdays"],
                museum_pass_valid=True,
                museum_pass_notes="Combined ticket includes Palace & Harem & Hagia Irene",
                audio_guide_available=True,
                virtual_museum_available=False,
                virtual_museum_url=None,
                administration="National Palace Museums",
                funding_type="State Funded",
                museum_type="Palace Museum",
                icomos_cards_valid=True,
                press_cards_valid=False
            ),
            
            "museum_turkish_islamic_arts": MuseumInfo(
                name="Museum of Turkish and Islamic Arts",
                name_turkish="Türk ve İslam Eserleri Müzesi",
                historical_period="Founded 1914",
                construction_date="Building: 1524 (Ibrahim Pasha Palace)",
                architect="Unknown (Ottoman period)",
                key_features=[
                    "Housed in Ibrahim Pasha Palace",
                    "Extensive carpet collection",
                    "Islamic manuscripts and calligraphy",
                    "Ethnographic displays"
                ],
                opening_hours={
                    "daily": "09:00-17:00",
                    "note": "Hours to be verified from Google Maps"
                },
                location="Sultanahmet Square, Hippodrome",
                nearby_attractions=["Blue Mosque", "Hagia Sophia", "Hippodrome"],
                visiting_duration="1-2 hours",
                best_time_to_visit="Any time",
                historical_significance="Premier collection of Turkish and Islamic art",
                architectural_style="Ottoman palace architecture",
                must_see_highlights=[
                    "Carpet collection",
                    "Islamic manuscripts",
                    "Traditional Turkish crafts",
                    "Palace architecture"
                ],
                photography_allowed=True,
                accessibility="Partially accessible",
                closing_days=["Mondays"],
                museum_pass_valid=True,
                museum_pass_notes="Entry no later than 18:45 with MuseumPass",
                audio_guide_available=True,
                virtual_museum_available=True,
                virtual_museum_url="sanalmuze.gov.tr",
                administration="Provincial Culture and Tourism Directorate",
                funding_type="State Funded",
                museum_type="Arts & Crafts Museum",
                icomos_cards_valid=True,
                press_cards_valid=True
            ),
            
            "galata_tower": MuseumInfo(
                name="Galata Tower Museum",
                name_turkish="Galata Kulesi Müzesi",
                historical_period="Genoese period (1348)",
                construction_date="1348",
                architect="Unknown (Genoese)",
                key_features=[
                    "360-degree panoramic views",
                    "Medieval tower architecture",
                    "Historical exhibitions",
                    "Observation deck"
                ],
                opening_hours={
                    "daily": "08:30-22:00",
                    "note": "Hours to be verified from Google Maps"
                },
                location="Galata, Beyoğlu",
                nearby_attractions=["Galata Bridge", "Karaköy", "Tünel"],
                visiting_duration="30-45 minutes",
                best_time_to_visit="Sunset for views",
                historical_significance="Medieval Genoese tower, symbol of Istanbul",
                architectural_style="Medieval Genoese",
                must_see_highlights=[
                    "Panoramic city views",
                    "Historical tower structure",
                    "Sunset views",
                    "Interactive displays"
                ],
                photography_allowed=True,
                accessibility="Limited (elevator available)",
                closing_days=["None"],
                museum_pass_valid=True,
                museum_pass_notes="Entry no later than 18:14 with MuseumPass",
                audio_guide_available=True,
                virtual_museum_available=False,
                virtual_museum_url=None,
                administration="Provincial Culture and Tourism Directorate",
                funding_type="State Funded",
                museum_type="Historical Tower Museum",
                icomos_cards_valid=True,
                press_cards_valid=True
            )
            
            # Additional museums will be added following the same pattern
        }
    
    def get_museum_info(self, museum_key: str) -> Optional[MuseumInfo]:
        """Get information for a specific museum"""
        return self.museums.get(museum_key)
    
    def get_all_museums(self) -> Dict[str, MuseumInfo]:
        """Get all museum information"""
        return self.museums
    
    def get_museums_by_type(self, museum_type: str) -> List[MuseumInfo]:
        """Get museums by type"""
        return [museum for museum in self.museums.values() 
                if museum.museum_type.lower() == museum_type.lower()]
    
    def get_museum_pass_museums(self) -> List[MuseumInfo]:
        """Get museums that accept MuseumPass"""
        return [museum for museum in self.museums.values() 
                if museum.museum_pass_valid]
    
    def format_museum_response(self, museum: MuseumInfo) -> str:
        """Format museum information for AI responses"""
        response = f"""
🏛️ **{museum.name}** ({museum.name_turkish})

📍 **Location**: {museum.location}
🏗️ **Built**: {museum.construction_date}
⏰ **Hours**: {museum.opening_hours.get('daily', 'Check current hours')}
🎫 **MuseumPass**: {'✅ Valid' if museum.museum_pass_valid else '❌ Not valid'}
🎧 **Audio Guide**: {'✅ Available' if museum.audio_guide_available else '❌ Not available'}
💻 **Virtual Tour**: {'✅ Available' if museum.virtual_museum_available else '❌ Not available'}

**Key Features:**
{chr(10).join(f'• {feature}' for feature in museum.key_features)}

**Must-See Highlights:**
{chr(10).join(f'• {highlight}' for highlight in museum.must_see_highlights)}

**Practical Info:**
• **Visit Duration**: {museum.visiting_duration}
• **Best Time**: {museum.best_time_to_visit}
• **Photography**: {'Allowed' if museum.photography_allowed else 'Restricted'}
• **Accessibility**: {museum.accessibility}

{self.museum_pass_info if museum.museum_pass_valid else ''}
"""
        return response
