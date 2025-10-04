"""
Information Retrieval Service
Replaces GPT for basic attraction, restaurant, and practical information queries.
Uses SQL database and comprehensive Istanbul knowledge database.
"""

import json
import re
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Add parent directory to path for knowledge database import
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from istanbul_knowledge_database import IstanbulKnowledgeDatabase
    KNOWLEDGE_DB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Istanbul Knowledge Database not available: {e}")
    KNOWLEDGE_DB_AVAILABLE = False

# Import SQL models
try:
    from database import SessionLocal
    from models import Place, Museum
    SQL_DB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SQL Database not available: {e}")
    SQL_DB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class InfoResult:
    """Structured result for information queries"""
    content: str
    source: str
    confidence: float
    data_type: str
    
class InfoRetrievalService:
    """
    Replace GPT usage for basic information retrieval with structured database responses
    Handles: attraction info, restaurant info, practical information
    Uses both SQL database and comprehensive knowledge database
    """
    
    def __init__(self, database_path: str = "data/"):
        self.database_path = database_path
        
        # Initialize knowledge database
        if KNOWLEDGE_DB_AVAILABLE:
            self.knowledge_db = IstanbulKnowledgeDatabase()
            print("✅ Istanbul Knowledge Database loaded")
        else:
            self.knowledge_db = None
            
        # Load fallback databases
        self.attractions_db = self._load_attractions_database()
        self.restaurants_db = self._load_restaurants_database()
        self.practical_db = self._load_practical_database()
        
    def _load_attractions_database(self) -> Dict:
        """Load structured attractions database"""
        try:
            # This will be built from existing istanbul_knowledge_database.py
            return {
                "hagia_sophia": {
                    "name": "Hagia Sophia",
                    "type": "Museum/Mosque",
                    "district": "Sultanahmet",
                    "hours": "9:00-19:00 (closed Mondays)",
                    "description": "Byzantine cathedral turned mosque, now museum",
                    "highlights": ["Byzantine mosaics", "Islamic calligraphy", "Historic architecture"],
                    "visit_duration": "1-2 hours",
                    "crowd_level": "Very High",
                    "best_time": "Early morning or late afternoon",
                    "nearby": ["Blue Mosque", "Topkapi Palace", "Basilica Cistern"]
                },
                "blue_mosque": {
                    "name": "Blue Mosque (Sultan Ahmed Mosque)",
                    "type": "Mosque",
                    "district": "Sultanahmet", 
                    "hours": "Prayer times vary, tourist visits between prayers",
                    "description": "Famous Ottoman mosque with blue tiles",
                    "highlights": ["Blue Iznik tiles", "Six minarets", "Prayer hall"],
                    "visit_duration": "30-45 minutes",
                    "crowd_level": "High",
                    "dress_code": "Conservative dress required",
                    "nearby": ["Hagia Sophia", "Hippodrome", "Grand Bazaar"]
                },
                # More attractions will be added...
            }
        except Exception as e:
            logger.error(f"Error loading attractions database: {e}")
            return {}
    
    def _load_restaurants_database(self) -> Dict:
        """Load structured restaurants database"""
        return {
            "pandeli": {
                "name": "Pandeli",
                "type": "Ottoman Cuisine",
                "district": "Eminönü",
                "price_range": "Upscale",
                "specialties": ["Ottoman dishes", "Historical ambiance"],
                "hours": "12:00-15:00, 19:00-24:00",
                "reservation": "Recommended",
                "atmosphere": "Historic, elegant"
            },
            # More restaurants...
        }
    
    def _load_practical_database(self) -> Dict:
        """Load practical information database"""
        return {
            "visa_requirements": {
                "content": "Most tourists can get e-visa online or visa on arrival",
                "details": ["90-day tourist visa available", "E-visa costs $50", "Some nationalities visa-free"]
            },
            "currency": {
                "content": "Turkish Lira (TRY) is the official currency",
                "details": ["Credit cards widely accepted", "ATMs available everywhere", "USD/EUR accepted in tourist areas"]
            },
            "transportation": {
                "content": "Comprehensive public transport system",
                "details": ["Istanbul Card for all transport", "Metro, bus, ferry, tram available", "Taxi and ride-sharing available"]
            }
        }
    
    def get_attraction_info(self, query: str) -> InfoResult:
        """Get attraction information using SQL database and knowledge base"""
        query_lower = query.lower()
        
        # First try SQL database for structured place data
        if SQL_DB_AVAILABLE:
            try:
                db = SessionLocal()
                
                # Search for attractions in SQL database
                places = db.query(Place).filter(
                    Place.category.in_(['Historical Site', 'Museum', 'Palace', 'Tower', 'Mosque'])
                ).all()
                
                for place in places:
                    name_lower = place.name.lower()
                    
                    # Create aliases for common attractions
                    aliases = self._get_attraction_aliases(place.name)
                    all_names = [name_lower] + [alias.lower() for alias in aliases]
                    
                    # Check if query matches any name or alias
                    match_found = False
                    for name in all_names:
                        if (name in query_lower or 
                            any(word in query_lower for word in name.split() if len(word) > 2) or
                            any(keyword in name for keyword in query_lower.split() if len(keyword) > 3)):
                            match_found = True
                            break
                    
                    if match_found:
                        
                        # Try to get detailed info from knowledge database
                        detailed_info = None
                        if KNOWLEDGE_DB_AVAILABLE:
                            try:
                                kb = IstanbulKnowledgeDatabase()
                                # Try to match with knowledge database
                                for kb_place in kb.places:
                                    if (kb_place['name'].lower() in name_lower or 
                                        name_lower in kb_place['name'].lower()):
                                        detailed_info = kb_place
                                        break
                            except Exception as e:
                                logger.error(f"Knowledge database error: {e}")
                        
                        # Format response using SQL data and knowledge base if available
                        response = self._format_sql_attraction_response(place, detailed_info)
                        db.close()
                        
                        return InfoResult(
                            content=response,
                            source="sql_database" + ("_enhanced" if detailed_info else ""),
                            confidence=0.9 if detailed_info else 0.8,
                            data_type="attraction"
                        )
                
                db.close()
            except Exception as e:
                logger.error(f"SQL database error: {e}")
        
        # Fallback to hardcoded data
        for attraction_id, attraction_data in self.attractions_db.items():
            name_lower = attraction_data["name"].lower()
            if name_lower in query_lower or any(word in query_lower for word in name_lower.split()):
                response = self._format_attraction_response(attraction_data)
                return InfoResult(
                    content=response,
                    source="database",
                    confidence=0.9,
                    data_type="attraction"
                )
        
        return InfoResult(
            content="I couldn't find specific information about that attraction. Could you be more specific?",
            source="fallback",
            confidence=0.3,
            data_type="attraction"
        )
    
    def get_restaurant_info(self, query: str) -> InfoResult:
        """Get restaurant information using database lookup"""
        query_lower = query.lower()
        
        for restaurant_id, restaurant_data in self.restaurants_db.items():
            name_lower = restaurant_data["name"].lower()
            if name_lower in query_lower or restaurant_data["type"].lower() in query_lower:
                response = self._format_restaurant_response(restaurant_data)
                return InfoResult(
                    content=response,
                    source="database",
                    confidence=0.9,
                    data_type="restaurant"
                )
        
        # If no specific restaurant found, provide general restaurant guidance
        if any(word in query_lower for word in ["restaurant", "eat", "food", "dining"]):
            return InfoResult(
                content="Istanbul offers amazing cuisine! Popular areas include Sultanahmet for traditional Ottoman food, Beyoğlu for modern dining, and Kadıköy for local favorites. Would you like recommendations for a specific type of cuisine?",
                source="general_guidance",
                confidence=0.7,
                data_type="restaurant"
            )
        
        return InfoResult(
            content="I couldn't find information about that restaurant. Could you provide more details?",
            source="fallback",
            confidence=0.3,
            data_type="restaurant"
        )
    
    def get_practical_info(self, query: str) -> InfoResult:
        """Get practical information using database lookup"""
        query_lower = query.lower()
        
        # Match practical info keywords
        practical_keywords = {
            "visa": "visa_requirements",
            "money": "currency",
            "currency": "currency", 
            "transport": "transportation",
            "metro": "transportation",
            "bus": "transportation"
        }
        
        for keyword, info_key in practical_keywords.items():
            if keyword in query_lower:
                info = self.practical_db.get(info_key, {})
                response = self._format_practical_response(info)
                return InfoResult(
                    content=response,
                    source="database",
                    confidence=0.95,
                    data_type="practical"
                )
        
        return InfoResult(
            content="I can help with practical information about visas, currency, transportation, and more. What specific information do you need?",
            source="fallback",
            confidence=0.5,
            data_type="practical"
        )
    
    def _format_attraction_response(self, attraction: Dict) -> str:
        """Format attraction data into natural response"""
        response_parts = [
            f"📍 **{attraction['name']}**",
            f"📍 Located in {attraction['district']}",
            f"🕒 Hours: {attraction['hours']}",
            f"📝 {attraction['description']}"
        ]
        
        if "highlights" in attraction:
            highlights = ", ".join(attraction["highlights"])
            response_parts.append(f"✨ Highlights: {highlights}")
        
        if "visit_duration" in attraction:
            response_parts.append(f"⏱️ Suggested visit time: {attraction['visit_duration']}")
        
        if "best_time" in attraction:
            response_parts.append(f"🕒 Best time to visit: {attraction['best_time']}")
        
        if "nearby" in attraction:
            nearby = ", ".join(attraction["nearby"])
            response_parts.append(f"🗺️ Nearby attractions: {nearby}")
        
        return "\n".join(response_parts)
    
    def _format_restaurant_response(self, restaurant: Dict) -> str:
        """Format restaurant data into natural response"""
        response_parts = [
            f"🍽️ **{restaurant['name']}**",
            f"📍 Located in {restaurant['district']}",
            f"🍴 Cuisine: {restaurant['type']}",
            f"💰 Price range: {restaurant['price_range']}"
        ]
        
        if "hours" in restaurant:
            response_parts.append(f"🕒 Hours: {restaurant['hours']}")
        
        if "specialties" in restaurant:
            specialties = ", ".join(restaurant["specialties"])
            response_parts.append(f"⭐ Specialties: {specialties}")
        
        if "reservation" in restaurant:
            response_parts.append(f"📞 Reservations: {restaurant['reservation']}")
        
        return "\n".join(response_parts)
    
    def _format_practical_response(self, info: Dict) -> str:
        """Format practical information into natural response"""
        if not info:
            return "Information not available at the moment."
        
        response_parts = [f"ℹ️ {info['content']}"]
        
        if "details" in info:
            response_parts.append("\n📋 Details:")
            for detail in info["details"]:
                response_parts.append(f"• {detail}")
        
        return "\n".join(response_parts)
    
    def _format_sql_attraction_response(self, place, detailed_info=None) -> str:
        """Format SQL place data with optional knowledge base enhancement"""
        response_parts = [
            f"📍 **{place.name}**",
            f"🏛️ Category: {place.category}",
            f"📍 District: {place.district or 'Istanbul'}"
        ]
        
        # Add detailed information if available from knowledge base
        if detailed_info:
            if 'description' in detailed_info:
                response_parts.append(f"📝 {detailed_info['description']}")
            
            if 'historical_significance' in detailed_info:
                response_parts.append(f"🏛️ Historical significance: {detailed_info['historical_significance']}")
            
            if 'architecture' in detailed_info:
                response_parts.append(f"🏗️ Architecture: {detailed_info['architecture']}")
            
            if 'visitor_tips' in detailed_info:
                if isinstance(detailed_info['visitor_tips'], list):
                    tips = '\n'.join([f"   • {tip}" for tip in detailed_info['visitor_tips']])
                    response_parts.append(f"💡 Visitor tips:\n{tips}")
                else:
                    response_parts.append(f"💡 Visitor tips: {detailed_info['visitor_tips']}")
            
            if 'opening_hours' in detailed_info:
                response_parts.append(f"🕒 Hours: {detailed_info['opening_hours']}")
            
            if 'nearby_attractions' in detailed_info and detailed_info['nearby_attractions']:
                nearby = ', '.join(detailed_info['nearby_attractions'][:3])  # Show top 3
                response_parts.append(f"🗺️ Nearby: {nearby}")
        else:
            # Basic information when no detailed data available
            category_descriptions = {
                'Historical Site': 'A significant historical location with cultural importance.',
                'Museum': 'A cultural institution housing artifacts and exhibitions.',
                'Palace': 'A former royal residence showcasing Ottoman architecture.',
                'Tower': 'An iconic structure offering panoramic city views.',
                'Mosque': 'A beautiful Islamic place of worship with architectural significance.'
            }
            
            if place.category in category_descriptions:
                response_parts.append(f"📝 {category_descriptions[place.category]}")
            
            response_parts.append("💡 Visit during off-peak hours for the best experience.")
        
        return "\n".join(response_parts)

    def _get_attraction_aliases(self, attraction_name: str) -> List[str]:
        """Get common aliases and alternative names for attractions"""
        aliases_map = {
            "Hagia Sophia Grand Mosque": ["hagia sophia", "ayasofya", "aya sofya", "santa sofia"],
            "The Blue Mosque": ["blue mosque", "sultanahmet mosque", "sultan ahmed mosque", "sultanahmet camii"],
            "Topkapi Palace Museum": ["topkapi palace", "topkapi", "topkapı sarayı", "topkapi sarayi"],
            "Basilica Cistern": ["basilica cistern", "yerebatan cistern", "yerebatan sarnıcı", "underground cistern"],
            "Grand Bazaar": ["grand bazaar", "kapali carsi", "kapalı çarşı", "covered bazaar"],
            "Galata Tower": ["galata tower", "galata kulesi", "tower of galata"],
            "Dolmabahçe Palace": ["dolmabahce palace", "dolmabahçe sarayı", "dolmabahce sarayi"],
            "Maiden's Tower": ["maidens tower", "kiz kulesi", "kız kulesi", "leanders tower"],
            "Istiklal Street": ["istiklal street", "istiklal caddesi", "independence street"],
            "Taksim Square": ["taksim square", "taksim meydani", "taksim meydanı"],
            "Pera Museum": ["pera museum", "pera müzesi", "pera muzesi"],
            "Naval Museum": ["naval museum", "deniz müzesi", "deniz muzesi", "maritime museum"]
        }
        
        return aliases_map.get(attraction_name, [])

    def search_info(self, query: str) -> InfoResult:
        """Main search function that tries all information types"""
        query_lower = query.lower()
        
        # Determine query type and route accordingly
        if any(word in query_lower for word in ["museum", "mosque", "palace", "attraction", "visit", "see"]):
            return self.get_attraction_info(query)
        elif any(word in query_lower for word in ["restaurant", "eat", "food", "dining", "cuisine"]):
            return self.get_restaurant_info(query)
        elif any(word in query_lower for word in ["visa", "money", "currency", "transport", "practical", "how", "where"]):
            return self.get_practical_info(query)
        else:
            # Try all types and return best match
            results = [
                self.get_attraction_info(query),
                self.get_restaurant_info(query),
                self.get_practical_info(query)
            ]
            
            # Return result with highest confidence
            best_result = max(results, key=lambda r: r.confidence)
            return best_result

# Example usage and testing
if __name__ == "__main__":
    service = InfoRetrievalService()
    
    # Test queries
    test_queries = [
        "Tell me about Hagia Sophia",
        "What are the hours for Blue Mosque?",
        "Good restaurants in Istanbul",
        "Do I need a visa for Turkey?",
        "How to use public transport in Istanbul"
    ]
    
    for query in test_queries:
        result = service.search_info(query)
        print(f"\nQuery: {query}")
        print(f"Response: {result.content}")
        print(f"Source: {result.source}, Confidence: {result.confidence}")
