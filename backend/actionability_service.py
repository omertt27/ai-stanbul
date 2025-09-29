#!/usr/bin/env python3
"""
Actionability Enhancement Service
================================

Ensures all responses include specific addresses, timing, directions,
and actionable next steps for visitors.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, time
import logging

@dataclass
class ActionableInfo:
    """Structured actionable information"""
    exact_addresses: List[str]
    walking_directions: List[str]
    transportation_details: List[str]
    timing_information: List[str]
    contact_details: List[str]
    next_steps: List[str]
    missing_elements: List[str]
    actionability_score: float

@dataclass
class LocationInfo:
    """Enhanced location information for actionability"""
    name: str
    exact_address: str
    nearest_metro_station: str
    walking_distance_from_metro: str
    walking_directions: str
    landmark_references: List[str]
    gps_coordinates: Optional[str] = None

class ActionabilityEnhancementService:
    """Service to enhance response actionability"""
    
    def __init__(self):
        self.istanbul_landmarks = self._initialize_landmarks()
        self.metro_stations = self._initialize_metro_stations()
        self.common_locations = self._initialize_common_locations()
        
    def _initialize_landmarks(self) -> Dict[str, Dict[str, str]]:
        """Initialize major Istanbul landmarks for reference"""
        return {
            "sultanahmet_square": {
                "name": "Sultanahmet Square",
                "address": "Sultanahmet Meydanı, Fatih/İstanbul",
                "metro": "Vezneciler (M2)",
                "walking_from_metro": "5 minutes northeast",
                "coordinates": "41.0055, 28.9769"
            },
            "taksim_square": {
                "name": "Taksim Square", 
                "address": "Taksim Meydanı, Beyoğlu/İstanbul",
                "metro": "Taksim (M2)",
                "walking_from_metro": "Direct access",
                "coordinates": "41.0370, 28.9858"
            },
            "galata_tower": {
                "name": "Galata Tower",
                "address": "Bereketzade Mahallesi, Galata Kulesi Sk., Beyoğlu/İstanbul",
                "metro": "Şişhane (M2)",
                "walking_from_metro": "8 minutes uphill via Galip Dede Caddesi",
                "coordinates": "41.0256, 28.9740"
            },
            "grand_bazaar": {
                "name": "Grand Bazaar",
                "address": "Beyazıt, Kalpakçılar Cd., Fatih/İstanbul",
                "metro": "Beyazıt-Kapalıçarşı (M1A)",
                "walking_from_metro": "2 minutes east",
                "coordinates": "41.0108, 28.9680"
            },
            "galata_bridge": {
                "name": "Galata Bridge",
                "address": "Galata Köprüsü, Eminönü-Karaköy",
                "metro": "Karaköy (M2) or Eminönü (T1)",
                "walking_from_metro": "Direct access",
                "coordinates": "41.0195, 28.9744"
            }
        }
    
    def _initialize_metro_stations(self) -> Dict[str, Dict[str, str]]:
        """Initialize metro stations with location details"""
        return {
            "taksim": {
                "line": "M2 Green Line",
                "address": "Taksim Meydanı, Beyoğlu",
                "exits": "A (Taksim Square), B (Istiklal Street), C (Hotel area)"
            },
            "sishhane": {
                "line": "M2 Green Line", 
                "address": "Şişhane Meydanı, Beyoğlu",
                "exits": "A (Galata Tower direction), B (Tunnel Square)"
            },
            "vezneciler": {
                "line": "M2 Green Line",
                "address": "Vezneciler Caddesi, Fatih",
                "exits": "A (University), B (Sultanahmet direction)"
            },
            "beyazit_kapalicarsi": {
                "line": "M1A Light Blue Line",
                "address": "Beyazıt Meydanı, Fatih", 
                "exits": "A (Grand Bazaar), B (Istanbul University)"
            },
            "kadikoy": {
                "line": "M4 Pink Line", 
                "address": "Kadıköy İskelesi, Kadıköy",
                "exits": "A (Ferry terminal), B (Market area), C (Bahariye Street)"
            }
        }
    
    def _initialize_common_locations(self) -> Dict[str, LocationInfo]:
        """Initialize common tourist locations with detailed info"""
        return {
            "hagia_sophia": LocationInfo(
                name="Hagia Sophia",
                exact_address="Ayasofya Meydanı No:1, Sultanahmet, Fatih/İstanbul",
                nearest_metro_station="Vezneciler (M2 Green Line)",
                walking_distance_from_metro="5 minutes (450 meters)",
                walking_directions="Exit Vezneciler station via Exit B, walk northeast on Ordu Caddesi, turn right on Divan Yolu, continue to Sultanahmet Square",
                landmark_references=["Across from Blue Mosque", "Next to Sultanahmet Square", "Near Topkapi Palace entrance"],
                gps_coordinates="41.0086, 28.9802"
            ),
            "topkapi_palace": LocationInfo(
                name="Topkapi Palace",
                exact_address="Cankurtaran Mahallesi, 34122 Fatih/İstanbul",
                nearest_metro_station="Gülhane (M1A Light Blue Line)",
                walking_distance_from_metro="3 minutes (250 meters)",
                walking_directions="Exit Gülhane station, walk north through Gülhane Park entrance, palace entrance on your right",
                landmark_references=["Behind Hagia Sophia", "Entrance through Gülhane Park", "Overlooking Bosphorus"],
                gps_coordinates="41.0115, 28.9833"
            ),
            "blue_mosque": LocationInfo(
                name="Blue Mosque (Sultan Ahmed Mosque)",
                exact_address="Sultanahmet Mahallesi, Atmeydanı Cd. No:7, Fatih/İstanbul",
                nearest_metro_station="Vezneciler (M2 Green Line)",
                walking_distance_from_metro="6 minutes (500 meters)",
                walking_directions="Exit Vezneciler station via Exit B, walk northeast on Ordu Caddesi, turn right on Divan Yolu, continue to Sultanahmet Square, mosque on the south side",
                landmark_references=["Across from Hagia Sophia", "On Sultanahmet Square", "Near Hippodrome"],
                gps_coordinates="41.0054, 28.9768"
            ),
            "galata_tower": LocationInfo(
                name="Galata Tower",
                exact_address="Bereketzade Mahallesi, Galata Kulesi Sk. No:8, Beyoğlu/İstanbul",
                nearest_metro_station="Şişhane (M2 Green Line)",
                walking_distance_from_metro="8 minutes uphill (600 meters)",
                walking_directions="Exit Şişhane station via Exit A, walk north on Galip Dede Caddesi uphill, tower visible ahead",
                landmark_references=["Near Tunnel Square", "Above Karaköy", "Visible from Galata Bridge"],
                gps_coordinates="41.0256, 28.9740"
            )
        }
    
    def enhance_response_actionability(self, response_text: str, query: str, category: str) -> Dict[str, any]:
        """
        Enhance response with actionable information
        """
        try:
            # Analyze current actionability
            current_actionability = self._analyze_current_actionability(response_text)
            
            # Extract mentioned locations
            mentioned_locations = self._extract_mentioned_locations(response_text)
            
            # Generate enhanced actionable information
            actionable_enhancements = self._generate_actionable_enhancements(
                mentioned_locations, category, query
            )
            
            # Create actionability score
            actionability_score = self._calculate_actionability_score(
                current_actionability, actionable_enhancements
            )
            
            return {
                "success": True,
                "original_actionability": current_actionability,
                "enhancements": actionable_enhancements,
                "actionability_score": actionability_score,
                "enhanced_response": self._create_enhanced_response(
                    response_text, actionable_enhancements
                ),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error enhancing actionability: {e}")
            return {
                "success": False,
                "error": str(e),
                "actionability_score": 0.0
            }
    
    def _analyze_current_actionability(self, response_text: str) -> ActionableInfo:
        """Analyze current level of actionability in response"""
        
        # Extract existing actionable elements
        exact_addresses = self._extract_addresses(response_text)
        walking_directions = self._extract_walking_directions(response_text)
        transportation_details = self._extract_transportation_details(response_text)
        timing_information = self._extract_timing_information(response_text)
        contact_details = self._extract_contact_details(response_text)
        next_steps = self._extract_next_steps(response_text)
        
        # Identify missing elements
        missing_elements = []
        if not exact_addresses:
            missing_elements.append("exact_addresses")
        if not walking_directions:
            missing_elements.append("walking_directions")
        if not transportation_details:
            missing_elements.append("transportation_details")
        if not timing_information:
            missing_elements.append("timing_information")
        if not next_steps:
            missing_elements.append("actionable_next_steps")
        
        # Calculate actionability score
        total_elements = 6
        present_elements = total_elements - len(missing_elements)
        actionability_score = present_elements / total_elements
        
        return ActionableInfo(
            exact_addresses=exact_addresses,
            walking_directions=walking_directions,
            transportation_details=transportation_details,
            timing_information=timing_information,
            contact_details=contact_details,
            next_steps=next_steps,
            missing_elements=missing_elements,
            actionability_score=actionability_score
        )
    
    def _extract_addresses(self, text: str) -> List[str]:
        """Extract exact addresses from text"""
        address_patterns = [
            r'[A-Z][a-zA-ZğüşıöçĞÜŞIÖÇ\s]+ (?:Caddesi|Cd\.?|Sokak|Sk\.?|Mahallesi),?\s*[A-Z][a-zA-ZğüşıöçĞÜŞIÖÇ\s]*/?İstanbul',
            r'[A-Z][a-zA-ZğüşıöçĞÜŞIÖÇ\s]+ No:\d+,?\s*[A-Z][a-zA-ZğüşıöçĞÜŞIÖÇ\s]*/?İstanbul',
            r'\d{5}\s+[A-Z][a-zA-ZğüşıöçĞÜŞIÖÇ\s]*/?İstanbul'
        ]
        
        addresses = []
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(matches)
        
        return list(set(addresses))
    
    def _extract_walking_directions(self, text: str) -> List[str]:
        """Extract walking directions from text"""
        direction_patterns = [
            r'walk \d+(?:\.\d+)? (?:minutes?|mins?) [a-zA-Z\s,]+',
            r'(?:from|exit) [A-Z][a-zA-Z\s]+ (?:station|metro|tram),? walk [a-zA-Z\s,]+',
            r'head [a-zA-Z]+ (?:on|via|along) [A-Z][a-zA-Z\s]+ (?:street|caddesi|street)',
            r'turn (?:left|right) (?:on|at) [A-Z][a-zA-Z\s]+'
        ]
        
        directions = []
        for pattern in direction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            directions.extend(matches)
        
        return directions
    
    def _extract_transportation_details(self, text: str) -> List[str]:
        """Extract transportation details from text"""
        transport_patterns = [
            r'(?:M\d+[A-Z]?|T\d+) (?:Metro|Tram|Line)',
            r'(?:Metro|Tram) (?:M\d+[A-Z]?|T\d+)',
            r'(?:from|to) [A-Z][a-zA-Z\s]+ (?:station|metro|tram)',
            r'(?:every|frequency) \d+(?:-\d+)? minutes?',
            r'operating hours? \d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?'
        ]
        
        transport_details = []
        for pattern in transport_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            transport_details.extend(matches)
        
        return transport_details
    
    def _extract_timing_information(self, text: str) -> List[str]:
        """Extract timing and scheduling information"""
        timing_patterns = [
            r'open (?:from )?\d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?',
            r'opening hours? \d{1,2}:\d{2}(?:\s*-\s*\d{1,2}:\d{2})?',
            r'(?:best time|visit) (?:between|from) \d{1,2}(?::\d{2})? ?(?:AM|PM|am|pm)?',
            r'(?:avoid|busy) (?:between|from) \d{1,2}(?::\d{2})? ?(?:AM|PM|am|pm)?',
            r'closed (?:on )?(?:mondays?|tuesdays?|wednesdays?|thursdays?|fridays?|saturdays?|sundays?)'
        ]
        
        timing_info = []
        for pattern in timing_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            timing_info.extend(matches)
        
        return timing_info
    
    def _extract_contact_details(self, text: str) -> List[str]:
        """Extract contact information"""
        contact_patterns = [
            r'\+90\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}',
            r'0\d{3}\s*\d{3}\s*\d{2}\s*\d{2}',
            r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
        ]
        
        contacts = []
        for pattern in contact_patterns:
            matches = re.findall(pattern, text)
            contacts.extend(matches)
        
        return contacts
    
    def _extract_next_steps(self, text: str) -> List[str]:
        """Extract actionable next steps"""
        next_step_patterns = [
            r'(?:first|next|then),? [a-zA-Z\s,]+',
            r'you (?:should|can|need to) [a-zA-Z\s,]+',
            r'(?:book|reserve|call|visit|check) [a-zA-Z\s,]+',
            r'(?:download|use|try) [a-zA-Z\s,]+ app'
        ]
        
        steps = []
        for pattern in next_step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            steps.extend(matches)
        
        return steps
    
    def _extract_mentioned_locations(self, text: str) -> List[str]:
        """Extract location names mentioned in the response"""
        locations = []
        
        # Check for common Istanbul locations
        location_names = list(self.common_locations.keys()) + list(self.istanbul_landmarks.keys())
        
        for location_key in location_names:
            # Get location names properly from dataclass objects and dictionaries
            common_location_name = ""
            if location_key in self.common_locations:
                common_location_name = self.common_locations[location_key].name
            
            landmark_name = ""
            if location_key in self.istanbul_landmarks:
                landmark_name = self.istanbul_landmarks[location_key].get("name", "")
            
            location_variants = [
                location_key.replace("_", " "),
                common_location_name,
                landmark_name
            ]
            
            for variant in location_variants:
                if variant and variant.lower() in text.lower():
                    locations.append(location_key)
                    break
        
        return list(set(locations))
    
    def _generate_actionable_enhancements(self, locations: List[str], category: str, query: str) -> Dict[str, List[str]]:
        """Generate actionable enhancements for mentioned locations"""
        
        enhancements = {
            "exact_addresses": [],
            "walking_directions": [], 
            "transportation_details": [],
            "timing_recommendations": [],
            "contact_information": [],
            "next_steps": []
        }
        
        # Add location-specific enhancements
        for location_key in locations:
            if location_key in self.common_locations:
                location_info = self.common_locations[location_key]
                
                enhancements["exact_addresses"].append(
                    f"{location_info.name}: {location_info.exact_address}"
                )
                
                enhancements["walking_directions"].append(
                    f"To {location_info.name}: {location_info.walking_directions}"
                )
                
                enhancements["transportation_details"].append(
                    f"Nearest metro: {location_info.nearest_metro_station} ({location_info.walking_distance_from_metro})"
                )
        
        # Category-specific enhancements
        if category == "museum":
            enhancements["next_steps"].extend([
                "Check official museum website for current hours and special exhibitions",
                "Consider purchasing tickets online to avoid queues",
                "Bring valid ID for potential student/senior discounts"
            ])
        elif category == "restaurant":
            enhancements["next_steps"].extend([
                "Call ahead for reservations, especially for dinner",
                "Check Google Maps for current opening hours",
                "Ask for English menu if needed"
            ])
        elif category == "transportation":
            enhancements["next_steps"].extend([
                "Purchase Istanbulkart at any metro station",
                "Download Citymapper or Moovit app for real-time updates",
                "Keep backup transportation options in mind"
            ])
        
        # General actionable steps
        enhancements["next_steps"].extend([
            "Save location on Google Maps for offline navigation",
            "Take note of nearby landmarks for easy identification",
            "Check weather conditions before visiting"
        ])
        
        return enhancements
    
    def _calculate_actionability_score(self, current: ActionableInfo, enhancements: Dict[str, List[str]]) -> float:
        """Calculate overall actionability score"""
        
        # Base score from current actionability
        base_score = current.actionability_score
        
        # Enhancement bonus
        enhancement_elements = sum(1 for values in enhancements.values() if values)
        enhancement_bonus = min(enhancement_elements / 6, 0.3)  # Max 0.3 bonus
        
        # Penalty for missing critical elements
        critical_missing = [elem for elem in current.missing_elements 
                          if elem in ["exact_addresses", "walking_directions"]]
        critical_penalty = len(critical_missing) * 0.15
        
        final_score = min(base_score + enhancement_bonus - critical_penalty, 1.0)
        return max(final_score, 0.0)
    
    def _create_enhanced_response(self, original_response: str, enhancements: Dict[str, List[str]]) -> str:
        """Create enhanced response with actionable information"""
        
        enhanced_response = original_response
        
        # Add actionable enhancements at the end
        if any(enhancements.values()):
            enhanced_response += "\n\n🎯 ACTIONABLE INFORMATION:\n"
            
            if enhancements["exact_addresses"]:
                enhanced_response += "\n📍 EXACT ADDRESSES:\n"
                for address in enhancements["exact_addresses"]:
                    enhanced_response += f"• {address}\n"
            
            if enhancements["walking_directions"]:
                enhanced_response += "\n🚶 WALKING DIRECTIONS:\n"
                for direction in enhancements["walking_directions"][:3]:
                    enhanced_response += f"• {direction}\n"
            
            if enhancements["transportation_details"]:
                enhanced_response += "\n🚇 TRANSPORTATION:\n"
                for transport in enhancements["transportation_details"][:3]:
                    enhanced_response += f"• {transport}\n"
            
            if enhancements["next_steps"]:
                enhanced_response += "\n✅ NEXT STEPS:\n"
                for step in enhancements["next_steps"][:4]:
                    enhanced_response += f"• {step}\n"
        
        return enhanced_response
    
    def get_location_actionable_info(self, location_name: str) -> Optional[LocationInfo]:
        """Get detailed actionable information for a specific location"""
        
        # Check common locations
        location_key = location_name.lower().replace(" ", "_")
        if location_key in self.common_locations:
            return self.common_locations[location_key]
        
        # Check landmarks
        if location_key in self.istanbul_landmarks:
            landmark = self.istanbul_landmarks[location_key]
            return LocationInfo(
                name=landmark["name"],
                exact_address=landmark["address"],
                nearest_metro_station=landmark["metro"],
                walking_distance_from_metro=landmark["walking_from_metro"],
                walking_directions=f"From {landmark['metro']}: {landmark['walking_from_metro']}",
                landmark_references=[],
                gps_coordinates=landmark["coordinates"]
            )
        
        return None

# Global instance
actionability_enhancer = ActionabilityEnhancementService()
