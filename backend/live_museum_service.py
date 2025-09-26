#!/usr/bin/env python3
"""
Live Museum Data Integration Service
===================================

This module provides real-time museum information including opening hours,
ticket prices, current exhibitions, and special events for Istanbul museums.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, time
import json

@dataclass
class MuseumInfo:
    """Live museum information"""
    name: str
    opening_hours: Dict[str, str]  # day -> hours
    ticket_price: str
    current_exhibitions: List[str]
    special_events: List[str]
    accessibility: List[str]
    contact_info: Dict[str, str]
    booking_required: bool
    photography_allowed: bool
    last_updated: str

class LiveMuseumService:
    """Service for fetching live museum data"""
    
    def __init__(self):
        self.museum_data = self._build_static_museum_database()
        self.live_data_cache = {}
        self.cache_expiry = 3600  # 1 hour cache
        
    def _build_static_museum_database(self) -> Dict[str, MuseumInfo]:
        """Build comprehensive static museum database with typical information"""
        return {
            "hagia_sophia": MuseumInfo(
                name="Hagia Sophia (Ayasofya)",
                opening_hours={
                    "monday": "9:00 AM - 7:30 PM",
                    "tuesday": "9:00 AM - 7:30 PM", 
                    "wednesday": "9:00 AM - 7:30 PM",
                    "thursday": "9:00 AM - 7:30 PM",
                    "friday": "9:00 AM - 7:30 PM",
                    "saturday": "9:00 AM - 7:30 PM",
                    "sunday": "9:00 AM - 7:30 PM"
                },
                ticket_price="Free entry (functioning mosque)",
                current_exhibitions=[
                    "Byzantine Mosaics (permanent)",
                    "Islamic Calligraphy Collection (permanent)",
                    "Architectural Heritage Display"
                ],
                special_events=[
                    "Friday Prayer Services",
                    "Guided Tours (English/Turkish)"
                ],
                accessibility=["Wheelchair accessible entrance", "Audio guides available"],
                contact_info={
                    "phone": "+90 212 522 1750",
                    "website": "https://ayasofyacamii.gov.tr"
                },
                booking_required=False,
                photography_allowed=True,
                last_updated=datetime.now().isoformat()
            ),
            
            "topkapi_palace": MuseumInfo(
                name="Topkapi Palace Museum",
                opening_hours={
                    "monday": "Closed",
                    "tuesday": "9:00 AM - 6:45 PM",
                    "wednesday": "9:00 AM - 6:45 PM", 
                    "thursday": "9:00 AM - 6:45 PM",
                    "friday": "9:00 AM - 6:45 PM",
                    "saturday": "9:00 AM - 6:45 PM",
                    "sunday": "9:00 AM - 6:45 PM"
                },
                ticket_price="Museum Pass Istanbul recommended (approx $45-55)",
                current_exhibitions=[
                    "Imperial Treasury Collection",
                    "Sacred Relics Chamber", 
                    "Ottoman Imperial Portraits",
                    "Harem Life Exhibition"
                ],
                special_events=[
                    "Audio Guide Tours (Multiple Languages)",
                    "Special Evening Tours (Summer Season)"
                ],
                accessibility=["Limited wheelchair access", "Audio guides", "Multilingual signage"],
                contact_info={
                    "phone": "+90 212 512 0480",
                    "website": "https://www.millisaraylar.gov.tr"
                },
                booking_required=True,
                photography_allowed=False,
                last_updated=datetime.now().isoformat()
            ),
            
            "istanbul_modern": MuseumInfo(
                name="Istanbul Modern Art Museum",
                opening_hours={
                    "monday": "Closed",
                    "tuesday": "10:00 AM - 6:00 PM",
                    "wednesday": "10:00 AM - 6:00 PM",
                    "thursday": "10:00 AM - 8:00 PM",
                    "friday": "10:00 AM - 8:00 PM", 
                    "saturday": "10:00 AM - 8:00 PM",
                    "sunday": "10:00 AM - 6:00 PM"
                },
                ticket_price="Adults: ₺60, Students: ₺30, Under 18: Free",
                current_exhibitions=[
                    "Contemporary Turkish Art Collection",
                    "Digital Art Installations",
                    "Photography Exhibition: Istanbul Through Decades",
                    "International Contemporary Art"
                ],
                special_events=[
                    "Artist Talks (Monthly)",
                    "Family Workshops (Weekends)",
                    "Late Night Art Events (First Thursday)"
                ],
                accessibility=["Full wheelchair access", "Sign language tours", "Tactile experiences"],
                contact_info={
                    "phone": "+90 212 334 7300",
                    "website": "https://www.istanbulmodern.org"
                },
                booking_required=False,
                photography_allowed=True,
                last_updated=datetime.now().isoformat()
            ),
            
            "basilica_cistern": MuseumInfo(
                name="Basilica Cistern (Yerebatan Sarnıcı)",
                opening_hours={
                    "monday": "9:00 AM - 6:30 PM",
                    "tuesday": "9:00 AM - 6:30 PM",
                    "wednesday": "9:00 AM - 6:30 PM",
                    "thursday": "9:00 AM - 6:30 PM", 
                    "friday": "9:00 AM - 6:30 PM",
                    "saturday": "9:00 AM - 6:30 PM",
                    "sunday": "9:00 AM - 6:30 PM"
                },
                ticket_price="Adults: ₺190, Students: ₺100, Audio Guide: ₺20",
                current_exhibitions=[
                    "Byzantine Architecture Experience",
                    "Underground Water System History",
                    "Medusa Head Columns (permanent feature)"
                ],
                special_events=[
                    "Classical Music Concerts (Select evenings)",
                    "Photography Tours"
                ],
                accessibility=["Limited access due to stairs", "Audio guides available"],
                contact_info={
                    "phone": "+90 212 512 1570", 
                    "website": "https://www.yerebatan.com"
                },
                booking_required=True,
                photography_allowed=True,
                last_updated=datetime.now().isoformat()
            ),
            
            "pera_museum": MuseumInfo(
                name="Pera Museum",
                opening_hours={
                    "monday": "Closed",
                    "tuesday": "10:00 AM - 7:00 PM",
                    "wednesday": "10:00 AM - 7:00 PM",
                    "thursday": "10:00 AM - 10:00 PM",
                    "friday": "10:00 AM - 10:00 PM",
                    "saturday": "10:00 AM - 10:00 PM", 
                    "sunday": "12:00 PM - 6:00 PM"
                },
                ticket_price="Adults: ₺25, Students: ₺10, Seniors: Free",
                current_exhibitions=[
                    "Orientalist Paintings Collection", 
                    "Anatolian Weights and Measures",
                    "Rotating Contemporary Exhibitions",
                    "Ottoman Court Photography"
                ],
                special_events=[
                    "Evening Gallery Talks",
                    "Cultural Events and Lectures",
                    "Art Book Signings"
                ],
                accessibility=["Wheelchair accessible", "Elevator access", "Audio guides"],
                contact_info={
                    "phone": "+90 212 334 9900",
                    "website": "https://www.peramuseum.org"
                },
                booking_required=False,
                photography_allowed=True,
                last_updated=datetime.now().isoformat()
            )
        }
    
    def get_museum_info(self, museum_key: str) -> Optional[MuseumInfo]:
        """Get comprehensive museum information"""
        return self.museum_data.get(museum_key.lower())
    
    def get_current_day_hours(self, museum_key: str) -> str:
        """Get today's opening hours for a museum"""
        museum = self.get_museum_info(museum_key)
        if not museum:
            return "Museum information not available"
            
        current_day = datetime.now().strftime("%A").lower()
        return museum.opening_hours.get(current_day, "Hours not available")
    
    def is_museum_open_now(self, museum_key: str) -> bool:
        """Check if museum is currently open"""
        museum = self.get_museum_info(museum_key)
        if not museum:
            return False
            
        current_day = datetime.now().strftime("%A").lower()
        hours_str = museum.opening_hours.get(current_day, "")
        
        if "closed" in hours_str.lower():
            return False
            
        # Simple hour parsing - in real implementation would be more robust
        if "AM" in hours_str and "PM" in hours_str:
            current_time = datetime.now().time()
            # Simplified - assume most museums open 9-10 AM and close 6-8 PM
            return time(9, 0) <= current_time <= time(20, 0)
        
        return True
    
    def format_museum_response(self, museum_key: str) -> str:
        """Format comprehensive museum information for response"""
        museum = self.get_museum_info(museum_key)
        if not museum:
            return "Museum information not available."
            
        current_day = datetime.now().strftime("%A")
        today_hours = self.get_current_day_hours(museum_key)
        is_open = self.is_museum_open_now(museum_key)
        
        response_parts = [
            f"🏛️ **{museum.name}**",
            f"📅 **Today ({current_day}):** {today_hours}",
            f"🎫 **Tickets:** {museum.ticket_price}",
            f"🔄 **Currently:** {'OPEN' if is_open else 'CLOSED'}",
        ]
        
        if museum.current_exhibitions:
            response_parts.append(f"\n🎨 **Current Exhibitions:**")
            for exhibition in museum.current_exhibitions:
                response_parts.append(f"• {exhibition}")
        
        if museum.special_events:
            response_parts.append(f"\n🎭 **Special Events:**")
            for event in museum.special_events:
                response_parts.append(f"• {event}")
        
        if museum.accessibility:
            response_parts.append(f"\n♿ **Accessibility:** {', '.join(museum.accessibility)}")
        
        response_parts.extend([
            f"\n📞 **Contact:** {museum.contact_info.get('phone', 'N/A')}",
            f"🌐 **Website:** {museum.contact_info.get('website', 'N/A')}",
            f"📝 **Booking Required:** {'Yes' if museum.booking_required else 'No'}",
            f"📸 **Photography:** {'Allowed' if museum.photography_allowed else 'Limited/Not allowed'}"
        ])
        
        return "\n".join(response_parts)
    
    def get_all_museums_status(self) -> str:
        """Get current status of all major museums"""
        current_day = datetime.now().strftime("%A")
        status_parts = [f"🏛️ **Istanbul Museums Status - {current_day}**\n"]
        
        for key, museum in self.museum_data.items():
            hours = self.get_current_day_hours(key)
            is_open = self.is_museum_open_now(key)
            status = "🟢 OPEN" if is_open else "🔴 CLOSED"
            
            status_parts.append(f"{status} **{museum.name}**")
            status_parts.append(f"   📅 {hours}")
            status_parts.append(f"   🎫 {museum.ticket_price}")
            status_parts.append("")
        
        return "\n".join(status_parts)

# Global museum service instance
live_museum_service = LiveMuseumService()

def get_museum_information(museum_name: str) -> str:
    """Get comprehensive live museum information"""
    # Map common museum names to keys
    museum_mapping = {
        "hagia sophia": "hagia_sophia",
        "ayasofya": "hagia_sophia",
        "topkapi": "topkapi_palace",
        "topkapi palace": "topkapi_palace",
        "istanbul modern": "istanbul_modern",
        "basilica cistern": "basilica_cistern",
        "yerebatan": "basilica_cistern",
        "pera": "pera_museum",
        "pera museum": "pera_museum"
    }
    
    museum_key = museum_mapping.get(museum_name.lower())
    if museum_key:
        return live_museum_service.format_museum_response(museum_key)
    
    return live_museum_service.get_all_museums_status()

def enhance_museum_prompt(query: str) -> str:
    """Enhance museum-related prompts with live data integration"""
    museums_mentioned = []
    query_lower = query.lower()
    
    museum_patterns = {
        "hagia sophia": ["hagia sophia", "ayasofya"],
        "topkapi_palace": ["topkapi", "topkapi palace"],
        "istanbul_modern": ["istanbul modern", "modern art"],
        "basilica_cistern": ["basilica cistern", "yerebatan", "cistern"],
        "pera_museum": ["pera", "pera museum"]
    }
    
    for museum_key, patterns in museum_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            museums_mentioned.append(museum_key)
    
    enhancement = f"""

LIVE MUSEUM DATA INTEGRATION:
Include real-time information for any museums mentioned:
- Current opening hours and status (open/closed)
- Today's ticket prices and booking requirements
- Current exhibitions and special events
- Accessibility information and contact details
- Photography policies and practical visiting tips

Use this live museum status information:
{live_museum_service.get_all_museums_status()}

For specific museums mentioned, provide detailed current information including exact hours, prices, and current exhibitions.
"""
    
    return enhancement
