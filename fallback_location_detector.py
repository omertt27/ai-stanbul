#!/usr/bin/env python3
"""
Fallback Location Detection System for AI Istanbul
Provides location detection when GPS is not available or permission is denied
"""

import json
import logging
import re
import math
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

# Import existing location detection services
try:
    from backend.services.intelligent_location_detector import (
        IntelligentLocationDetector, DetectedLocation, LocationConfidence
    )
    INTELLIGENT_DETECTOR_AVAILABLE = True
except ImportError:
    INTELLIGENT_DETECTOR_AVAILABLE = False
    logging.warning("Intelligent location detector not available")

try:
    from ml_enhanced_transportation_system import GPSLocation, TransportMode
    ML_TRANSPORT_AVAILABLE = True
except ImportError:
    ML_TRANSPORT_AVAILABLE = False
    logging.warning("ML enhanced transportation system not available")
    # Define GPSLocation as a fallback type when not available
    @dataclass
    class GPSLocation:
        latitude: float
        longitude: float
        district: str = ""
        accuracy: Optional[float] = None
        address: Optional[str] = None
        timestamp: Optional[datetime] = None

logger = logging.getLogger(__name__)

class LocationDetectionMethod(Enum):
    """Methods for detecting user location"""
    GPS = "gps"
    USER_INPUT = "user_input"
    ML_INFERENCE = "ml_inference"
    IP_GEOLOCATION = "ip_geolocation"
    BROWSER_CONTEXT = "browser_context"
    PREVIOUS_SESSION = "previous_session"
    LANDMARK_RECOGNITION = "landmark_recognition"

class LocationInputType(Enum):
    """Types of location input from users"""
    DISTRICT_NAME = "district_name"
    NEIGHBORHOOD = "neighborhood"
    LANDMARK = "landmark"
    ADDRESS = "address"
    TRANSPORT_HUB = "transport_hub"
    DESCRIPTIVE = "descriptive"
    COORDINATES = "coordinates"

@dataclass
class LocationFallbackOption:
    """Represents a fallback location option"""
    method: LocationDetectionMethod
    confidence: float  # 0.0 to 1.0
    location: Optional[GPSLocation] = None
    description: str = ""
    user_friendly_name: str = ""
    district: str = ""
    neighborhood: str = ""
    source_data: Dict[str, Any] = None
    requires_user_confirmation: bool = False
    
    def __post_init__(self):
        if self.source_data is None:
            self.source_data = {}

@dataclass
class UserLocationPrompt:
    """Prompt for user location input"""
    message: str
    input_type: LocationInputType
    examples: List[str]
    validation_pattern: Optional[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

class FallbackLocationDetector:
    """
    Comprehensive fallback location detection system
    """
    
    def __init__(self):
        self.intelligent_detector = None
        if INTELLIGENT_DETECTOR_AVAILABLE:
            self.intelligent_detector = IntelligentLocationDetector()
        
        # Istanbul districts and neighborhoods for validation
        self.istanbul_districts = {
            "adalar": ["büyükada", "heybeliada", "burgazada", "kınalıada"],
            "arnavutköy": ["arnavutköy merkez", "hadımköy", "bolluca"],
            "ataşehir": ["ataşehir merkez", "küçükbakkalköy", "ferhatpaşa"],
            "avcilar": ["avcılar merkez", "ambarlı", "gümüşpala"],
            "bağcılar": ["bağcılar merkez", "kemalpaşa", "yıldıztepe"],
            "bahçelievler": ["bahçelievler merkez", "kocasinan", "şirinevler"],
            "bakırköy": ["bakırköy merkez", "yeşilköy", "ataköy"],
            "başakşehir": ["başakşehir merkez", "bahçeşehir", "kayabaşı"],
            "bayrampaşa": ["bayrampaşa merkez", "muratpaşa", "altıntepsi"],
            "beşiktaş": ["beşiktaş merkez", "ortaköy", "bebek", "etiler", "levent"],
            "beylikdüzü": ["beylikdüzü merkez", "esenyurt", "büyükçekmece"],
            "beyoğlu": ["galata", "karaköy", "taksim", "cihangir", "galatasaray"],
            "büyükçekmece": ["büyükçekmece merkez", "mimarsinan", "alkent"],
            "çatalca": ["çatalca merkez", "yalıköy", "karacaköy"],
            "çekmeköy": ["çekmeköy merkez", "taşdelen", "alemdağ"],
            "esenler": ["esenler merkez", "havaalanı", "menderes"],
            "esenyurt": ["esenyurt merkez", "yenikent", "süleymaniye"],
            "eyüpsultan": ["eyüp", "alibeyköy", "kemerburgaz", "rami"],
            "fatih": ["sultanahmet", "beyazıt", "balat", "fener", "aksaray"],
            "gaziosmanpaşa": ["gaziosmanpaşa merkez", "sarıyer", "talatpaşa"],
            "güngören": ["güngören merkez", "merter", "akıncılar"],
            "kadıköy": ["kadıköy merkez", "moda", "fenerbahçe", "göztepe"],
            "kağıthane": ["kağıthane merkez", "çeliktepe", "gültepe"],
            "kartal": ["kartal merkez", "maltepe", "dragos"],
            "küçükçekmece": ["küçükçekmece merkez", "halkalı", "florya"],
            "maltepe": ["maltepe merkez", "bağlarbaşı", "başıbüyük"],
            "pendik": ["pendik merkez", "kaynarca", "sapanca"],
            "sancaktepe": ["sancaktepe merkez", "samandıra", "eski"],
            "sarıyer": ["sarıyer merkez", "kilyos", "tarabya", "yeniköy"],
            "silivri": ["silivri merkez", "selimpaşa", "çanta"],
            "sultanbeyli": ["sultanbeyli merkez", "abdurrahmangazi"],
            "sultangazi": ["sultangazi merkez", "gazi", "esentepe"],
            "şile": ["şile merkez", "ağva", "kumbaba"],
            "şişli": ["şişli merkez", "mecidiyeköy", "bomonti", "nişantaşı"],
            "tuzla": ["tuzla merkez", "orhanlı", "tepeören"],
            "ümraniye": ["ümraniye merkez", "ataşehir", "dudullu"],
            "üsküdar": ["üsküdar merkez", "çengelköy", "beylerbeyi", "kuzguncuk"],
            "zeytinburnu": ["zeytinburnu merkez", "maltepe", "seyitnizam"]
        }
        
        # Common landmarks and their locations
        self.landmark_locations = {
            "sultanahmet": GPSLocation(41.0082, 28.9784, district="fatih"),
            "taksim": GPSLocation(41.0369, 28.9857, district="beyoğlu"),
            "galata kulesi": GPSLocation(41.0256, 28.9744, district="beyoğlu"),
            "topkapı sarayı": GPSLocation(41.0115, 28.9833, district="fatih"),
            "ayasofya": GPSLocation(41.0086, 28.9802, district="fatih"),
            "kapalıçarşı": GPSLocation(41.0107, 28.9683, district="fatih"),
            "galata köprüsü": GPSLocation(41.0197, 28.9737, district="beyoğlu"),
            "dolmabahçe sarayı": GPSLocation(41.0391, 29.0000, district="beşiktaş"),
            "ortaköy": GPSLocation(41.0473, 29.0264, district="beşiktaş"),
            "bebek": GPSLocation(41.0839, 29.0434, district="beşiktaş"),
            "kadıköy": GPSLocation(40.9780, 29.0375, district="kadıköy"),
            "üsküdar": GPSLocation(41.0276, 29.0097, district="üsküdar"),
            "levent": GPSLocation(41.0814, 29.0056, district="beşiktaş"),
            "maslak": GPSLocation(41.1086, 29.0253, district="sarıyer"),
            "atatürk havalimanı": GPSLocation(40.9769, 28.8169, district="bakırköy"),
            "sabiha gökçen havalimanı": GPSLocation(40.8986, 29.3092, district="pendik"),
            "boğaziçi köprüsü": GPSLocation(41.0418, 29.0208, district="beşiktaş"),
            "fatih sultan mehmet köprüsü": GPSLocation(41.0972, 29.0464, district="sarıyer"),
            "istinye park": GPSLocation(41.1086, 29.0253, district="sarıyer"),
            "cevahir": GPSLocation(41.0587, 28.9811, district="şişli")
        }
        
        # Transport hubs and their locations
        self.transport_hubs = {
            "atatürk havalimanı": GPSLocation(40.9769, 28.8169, district="bakırköy"),
            "sabiha gökçen havalimanı": GPSLocation(40.8986, 29.3092, district="pendik"),
            "haydarpaşa garı": GPSLocation(40.9667, 29.0167, district="kadıköy"),
            "sirkeci garı": GPSLocation(41.0172, 28.9764, district="fatih"),
            "eminönü": GPSLocation(41.0178, 28.9714, district="fatih"),
            "kabataş": GPSLocation(41.0297, 28.9953, district="beyoğlu"),
            "beşiktaş": GPSLocation(41.0422, 29.0061, district="beşiktaş"),
            "üsküdar": GPSLocation(41.0276, 29.0097, district="üsküdar"),
            "kadıköy": GPSLocation(40.9780, 29.0375, district="kadıköy"),
            "bostancı": GPSLocation(40.9600, 29.0933, district="kadıköy")
        }
    
    async def detect_fallback_location(self, 
                                     user_input: Optional[str] = None, 
                                     user_context: Optional[Dict] = None,
                                     ip_address: Optional[str] = None,
                                     session_data: Optional[Dict] = None) -> List[LocationFallbackOption]:
        """
        Detect user location using various fallback methods
        
        Args:
            user_input: User's location description or preference
            user_context: Previous conversation context
            ip_address: User's IP address for geolocation
            session_data: Previous session data
            
        Returns:
            List of location fallback options, sorted by confidence
        """
        options = []
        
        # Method 1: Parse direct user input
        if user_input:
            user_options = await self._detect_from_user_input(user_input)
            options.extend(user_options)
        
        # Method 2: Use ML inference from context
        if user_context and self.intelligent_detector:
            ml_options = await self._detect_from_ml_inference(user_context)
            options.extend(ml_options)
        
        # Method 3: IP-based geolocation
        if ip_address:
            ip_options = await self._detect_from_ip(ip_address)
            options.extend(ip_options)
        
        # Method 4: Session-based detection
        if session_data:
            session_options = self._detect_from_session(session_data)
            options.extend(session_options)
        
        # Method 5: Landmark recognition from conversation
        if user_context:
            landmark_options = self._detect_from_landmarks(user_context)
            options.extend(landmark_options)
        
        # Sort by confidence and remove duplicates
        options = self._deduplicate_options(options)
        options.sort(key=lambda x: x.confidence, reverse=True)
        
        return options[:5]  # Return top 5 options
    
    async def _detect_from_user_input(self, user_input: str) -> List[LocationFallbackOption]:
        """Detect location from direct user input"""
        options = []
        user_input_lower = user_input.lower().strip()
        
        # Check for coordinates
        coord_match = re.search(r'(\d+\.\d+),\s*(\d+\.\d+)', user_input)
        if coord_match:
            lat, lng = float(coord_match.group(1)), float(coord_match.group(2))
            if 40.5 <= lat <= 41.5 and 28.0 <= lng <= 30.0:  # Istanbul bounds
                location = GPSLocation(lat, lng)
                location = await self._enrich_location_with_district(location)
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.USER_INPUT,
                    confidence=0.95,
                    location=location,
                    description=f"GPS coordinates: {lat}, {lng}",
                    user_friendly_name=f"Location at {lat:.4f}, {lng:.4f}",
                    district=location.district
                ))
        
        # Check for exact landmark matches
        for landmark, location in self.landmark_locations.items():
            if landmark in user_input_lower:
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.USER_INPUT,
                    confidence=0.9,
                    location=location,
                    description=f"Landmark: {landmark.title()}",
                    user_friendly_name=landmark.title(),
                    district=location.district
                ))
        
        # Check for district names
        for district, neighborhoods in self.istanbul_districts.items():
            if district in user_input_lower:
                # Use approximate center coordinates for districts
                center_location = await self._get_district_center(district)
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.USER_INPUT,
                    confidence=0.8,
                    location=center_location,
                    description=f"District: {district.title()}",
                    user_friendly_name=f"{district.title()} District",
                    district=district
                ))
            
            # Check for neighborhood names
            for neighborhood in neighborhoods:
                if neighborhood in user_input_lower:
                    neighborhood_location = await self._get_neighborhood_location(district, neighborhood)
                    options.append(LocationFallbackOption(
                        method=LocationDetectionMethod.USER_INPUT,
                        confidence=0.85,
                        location=neighborhood_location,
                        description=f"Neighborhood: {neighborhood.title()}, {district.title()}",
                        user_friendly_name=f"{neighborhood.title()}, {district.title()}",
                        district=district,
                        neighborhood=neighborhood
                    ))
        
        # Check for transport hubs
        for hub, location in self.transport_hubs.items():
            if hub in user_input_lower or any(word in user_input_lower for word in hub.split()):
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.USER_INPUT,
                    confidence=0.9,
                    location=location,
                    description=f"Transport Hub: {hub.title()}",
                    user_friendly_name=hub.title(),
                    district=location.district
                ))
        
        return options
    
    async def _detect_from_ml_inference(self, user_context: Dict) -> List[LocationFallbackOption]:
        """Use ML inference to detect location from conversation context"""
        options = []
        
        if not self.intelligent_detector:
            return options
        
        try:
            # Extract text from context
            context_text = ""
            if "messages" in user_context:
                context_text = " ".join([msg.get("content", "") for msg in user_context["messages"]])
            elif "conversation" in user_context:
                context_text = str(user_context["conversation"])
            else:
                context_text = str(user_context)
            
            # Use intelligent location detector
            detected = await self.intelligent_detector.detect_location_from_text(context_text, user_context)
            
            if detected.latitude and detected.longitude:
                location = GPSLocation(detected.latitude, detected.longitude, district=detected.district or "")
                confidence_map = {
                    LocationConfidence.VERY_HIGH: 0.9,
                    LocationConfidence.HIGH: 0.8,
                    LocationConfidence.MEDIUM: 0.6,
                    LocationConfidence.LOW: 0.4,
                    LocationConfidence.UNKNOWN: 0.2
                }
                
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.ML_INFERENCE,
                    confidence=confidence_map.get(detected.confidence, 0.5),
                    location=location,
                    description=f"ML Inference: {detected.name or 'Detected location'}",
                    user_friendly_name=detected.name or f"Detected location in {detected.district or 'Istanbul'}",
                    district=detected.district or "",
                    neighborhood=detected.neighborhood or "",
                    source_data={"detection_source": detected.source}
                ))
        
        except Exception as e:
            logger.warning(f"ML inference failed: {e}")
        
        return options
    
    async def _detect_from_ip(self, ip_address: str) -> List[LocationFallbackOption]:
        """Detect location from IP address"""
        options = []
        
        if not self.intelligent_detector:
            return options
        
        try:
            detected = await self.intelligent_detector.get_location_from_ip(ip_address)
            
            if detected.latitude and detected.longitude:
                location = GPSLocation(detected.latitude, detected.longitude, district=detected.district or "")
                
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.IP_GEOLOCATION,
                    confidence=0.6,  # IP geolocation is less precise
                    location=location,
                    description=f"IP Geolocation: {detected.name or 'Detected from IP'}",
                    user_friendly_name=detected.name or "Your approximate location",
                    district=detected.district or "",
                    source_data={"ip_address": ip_address},
                    requires_user_confirmation=True
                ))
        
        except Exception as e:
            logger.warning(f"IP geolocation failed: {e}")
        
        return options
    
    def _detect_from_session(self, session_data: Dict) -> List[LocationFallbackOption]:
        """Detect location from previous session data"""
        options = []
        
        if "last_location" in session_data:
            last_loc = session_data["last_location"]
            if isinstance(last_loc, dict) and "latitude" in last_loc and "longitude" in last_loc:
                location = GPSLocation(
                    last_loc["latitude"], 
                    last_loc["longitude"],
                    district=last_loc.get("district", "")
                )
                
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.PREVIOUS_SESSION,
                    confidence=0.7,
                    location=location,
                    description="Previous session location",
                    user_friendly_name=f"Your previous location in {last_loc.get('district', 'Istanbul')}",
                    district=last_loc.get("district", ""),
                    requires_user_confirmation=True
                ))
        
        return options
    
    def _detect_from_landmarks(self, user_context: Dict) -> List[LocationFallbackOption]:
        """Detect location from landmark mentions in conversation"""
        options = []
        
        # Extract text from context
        context_text = ""
        if "messages" in user_context:
            context_text = " ".join([msg.get("content", "") for msg in user_context["messages"]])
        elif "conversation" in user_context:
            context_text = str(user_context["conversation"])
        else:
            context_text = str(user_context)
        
        context_lower = context_text.lower()
        
        # Look for landmark mentions
        for landmark, location in self.landmark_locations.items():
            if landmark in context_lower:
                options.append(LocationFallbackOption(
                    method=LocationDetectionMethod.LANDMARK_RECOGNITION,
                    confidence=0.75,
                    location=location,
                    description=f"Mentioned landmark: {landmark.title()}",
                    user_friendly_name=f"Near {landmark.title()}",
                    district=location.district,
                    source_data={"mentioned_landmark": landmark}
                ))
        
        return options
    
    def _deduplicate_options(self, options: List[LocationFallbackOption]) -> List[LocationFallbackOption]:
        """Remove duplicate location options"""
        seen_locations = set()
        deduplicated = []
        
        for option in options:
            if option.location:
                location_key = (
                    round(option.location.latitude, 4), 
                    round(option.location.longitude, 4)
                )
                if location_key not in seen_locations:
                    seen_locations.add(location_key)
                    deduplicated.append(option)
            else:
                deduplicated.append(option)
        
        return deduplicated
    
    async def _get_district_center(self, district: str) -> GPSLocation:
        """Get approximate center coordinates for a district"""
        # District center coordinates (approximate)
        district_centers = {
            "fatih": GPSLocation(41.0120, 28.9675, district="fatih"),
            "beyoğlu": GPSLocation(41.0370, 28.9857, district="beyoğlu"),
            "beşiktaş": GPSLocation(41.0422, 29.0061, district="beşiktaş"),
            "kadıköy": GPSLocation(40.9780, 29.0375, district="kadıköy"),
            "üsküdar": GPSLocation(41.0276, 29.0097, district="üsküdar"),
            "şişli": GPSLocation(41.0587, 28.9811, district="şişli"),
            "bakırköy": GPSLocation(40.9769, 28.8169, district="bakırköy"),
            "ataşehir": GPSLocation(40.9833, 29.1167, district="ataşehir"),
            "pendik": GPSLocation(40.8986, 29.3092, district="pendik"),
            "sarıyer": GPSLocation(41.1086, 29.0253, district="sarıyer")
        }
        
        return district_centers.get(district, GPSLocation(41.0082, 28.9784, district=district))
    
    async def _get_neighborhood_location(self, district: str, neighborhood: str) -> GPSLocation:
        """Get approximate coordinates for a neighborhood"""
        # For simplicity, offset neighborhood locations from district centers
        district_center = await self._get_district_center(district)
        
        # Add small random offset for neighborhoods (this would be more precise with real data)
        import random
        lat_offset = random.uniform(-0.005, 0.005)
        lng_offset = random.uniform(-0.005, 0.005)
        
        return GPSLocation(
            district_center.latitude + lat_offset,
            district_center.longitude + lng_offset,
            district=district
        )
    
    async def _enrich_location_with_district(self, location: GPSLocation) -> GPSLocation:
        """Enrich a GPS location with district information"""
        # Simple reverse geocoding using district boundaries
        # In a real implementation, this would use a proper reverse geocoding service
        
        for district, center_location in {
            "fatih": GPSLocation(41.0120, 28.9675),
            "beyoğlu": GPSLocation(41.0370, 28.9857),
            "beşiktaş": GPSLocation(41.0422, 29.0061),
            "kadıköy": GPSLocation(40.9780, 29.0375),
            "üsküdar": GPSLocation(41.0276, 29.0097),
        }.items():
            distance = self._calculate_distance(location, center_location)
            if distance < 5.0:  # Within 5km of district center
                location.district = district
                break
        
        return location
    
    def _calculate_distance(self, loc1: GPSLocation, loc2: GPSLocation) -> float:
        """Calculate distance between two GPS locations in kilometers"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(loc1.latitude)
        lat2_rad = math.radians(loc2.latitude)
        delta_lat = math.radians(loc2.latitude - loc1.latitude)
        delta_lng = math.radians(loc2.longitude - loc1.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def generate_location_prompts(self) -> List[UserLocationPrompt]:
        """Generate user-friendly prompts for location input"""
        return [
            UserLocationPrompt(
                message="Could you tell me your current location or which district you're in?",
                input_type=LocationInputType.DISTRICT_NAME,
                examples=[
                    "I'm in Sultanahmet",
                    "I'm near Taksim Square",
                    "I'm in Kadıköy district",
                    "I'm at Galata Tower"
                ],
                suggestions=list(self.istanbul_districts.keys())
            ),
            UserLocationPrompt(
                message="Which landmark or neighborhood are you closest to?",
                input_type=LocationInputType.LANDMARK,
                examples=[
                    "Topkapi Palace",
                    "Galata Bridge",
                    "Ortaköy",
                    "Levent"
                ],
                suggestions=list(self.landmark_locations.keys())
            ),
            UserLocationPrompt(
                message="What's the nearest metro station or transport hub?",
                input_type=LocationInputType.TRANSPORT_HUB,
                examples=[
                    "Kabataş",
                    "Eminönü",
                    "Sirkeci Station",
                    "Atatürk Airport"
                ],
                suggestions=list(self.transport_hubs.keys())
            ),
            UserLocationPrompt(
                message="Can you describe your location? (e.g., 'near the Bosphorus', 'in the old city')",
                input_type=LocationInputType.DESCRIPTIVE,
                examples=[
                    "Near the Bosphorus Bridge",
                    "In the historical peninsula",
                    "Close to the Grand Bazaar",
                    "In the business district"
                ]
            )
        ]

# Convenience function for easy import
async def detect_user_location_fallback(user_input: str = None, 
                                       user_context: Dict = None,
                                       ip_address: str = None,
                                       session_data: Dict = None) -> List[LocationFallbackOption]:
    """
    Convenience function to detect user location using fallback methods
    """
    detector = FallbackLocationDetector()
    return await detector.detect_fallback_location(user_input, user_context, ip_address, session_data)

def get_location_prompts() -> List[UserLocationPrompt]:
    """
    Convenience function to get location input prompts
    """
    detector = FallbackLocationDetector()
    return detector.generate_location_prompts()

# JSON serialization helpers
def location_fallback_option_to_dict(option: LocationFallbackOption) -> Dict:
    """Convert LocationFallbackOption to dictionary for JSON serialization"""
    result = asdict(option)
    if option.location:
        result["location"] = {
            "latitude": option.location.latitude,
            "longitude": option.location.longitude,
            "accuracy": getattr(option.location, 'accuracy', None),
            "address": getattr(option.location, 'address', None),
            "district": getattr(option.location, 'district', ''),
            "timestamp": option.location.timestamp.isoformat() if hasattr(option.location, 'timestamp') and option.location.timestamp else None
        }
    result["method"] = option.method.value
    return result

def user_location_prompt_to_dict(prompt: UserLocationPrompt) -> Dict:
    """Convert UserLocationPrompt to dictionary for JSON serialization"""
    result = asdict(prompt)
    result["input_type"] = prompt.input_type.value
    return result

if __name__ == "__main__":
    # Test the fallback location detector
    async def test_detector():
        detector = FallbackLocationDetector()
        
        # Test with user input
        print("Testing with user input: 'I'm in Sultanahmet'")
        options = await detector.detect_fallback_location(user_input="I'm in Sultanahmet")
        for option in options:
            print(f"  {option.method.value}: {option.user_friendly_name} (confidence: {option.confidence:.2f})")
        
        print("\nTesting with coordinates: '41.0082, 28.9784'")
        options = await detector.detect_fallback_location(user_input="41.0082, 28.9784")
        for option in options:
            print(f"  {option.method.value}: {option.user_friendly_name} (confidence: {option.confidence:.2f})")
        
        print("\nLocation prompts:")
        prompts = detector.generate_location_prompts()
        for prompt in prompts:
            print(f"  {prompt.input_type.value}: {prompt.message}")
    
    asyncio.run(test_detector())
