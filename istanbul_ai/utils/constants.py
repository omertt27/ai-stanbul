"""
Constants and Enums for Istanbul AI System
"""

from enum import Enum
from typing import Dict, List

class ConversationTone(Enum):
    """Different conversation tones available"""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional" 
    CASUAL = "casual"
    HELPFUL = "helpful"
    ENTHUSIASTIC = "enthusiastic"
    CULTURAL = "cultural"

class UserType(Enum):
    """Different types of users"""
    FIRST_TIME_VISITOR = "first_time_visitor"
    RETURNING_VISITOR = "returning_visitor"
    LOCAL_RESIDENT = "local_resident"
    BUSINESS_TRAVELER = "business_traveler"
    CULTURAL_ENTHUSIAST = "cultural_enthusiast"
    FOODIE = "foodie"
    BUDGET_TRAVELER = "budget_traveler"
    LUXURY_TRAVELER = "luxury_traveler"

# Istanbul districts
ISTANBUL_DISTRICTS = [
    "Sultanahmet", "Beyoğlu", "Galata", "Karaköy", "Taksim", "Şişli", 
    "Beşiktaş", "Kadıköy", "Üsküdar", "Fatih", "Eminönü", "Bakırköy",
    "Sarıyer", "Ortaköy", "Arnavutköy", "Bebek", "Etiler", "Levent",
    "Maslak", "Mecidiyeköy", "Nişantaşı", "Cihangir", "Balat", "Fener"
]

# Cuisine types
CUISINE_TYPES = [
    "Turkish", "Ottoman", "Seafood", "Kebab", "Meze", "Street Food",
    "Breakfast", "Dessert", "International", "Vegetarian", "Vegan"
]

# Transportation modes
TRANSPORT_MODES = [
    "Metro", "Bus", "Ferry", "Tram", "Metrobus", "Taxi", "Walking"
]

# Default responses
DEFAULT_RESPONSES = {
    "greeting": "Merhaba! Welcome to Istanbul! How can I help you explore this beautiful city?",
    "fallback": "I'd be happy to help you with information about Istanbul attractions, restaurants, transportation, or events!",
    "error": "I apologize, but I'm having trouble processing your request. Please try asking about Istanbul attractions, restaurants, or events."
}
