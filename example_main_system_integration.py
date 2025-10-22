#!/usr/bin/env python3
"""
Example: Integrating Intent Classifier into Istanbul AI Main System
Shows how to use the production classifier in a real chat endpoint
"""

from production_intent_classifier import get_production_classifier
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize classifier once at startup (singleton pattern)
intent_classifier = get_production_classifier()


def handle_emergency(query: str, confidence: float) -> Dict:
    """Handle emergency queries"""
    return {
        "type": "emergency",
        "message": "🚨 **Emergency Response**\n\n"
                  "Here are emergency contacts:\n"
                  "- Police: 155\n"
                  "- Ambulance: 112\n"
                  "- Fire: 110\n"
                  "- Tourist Police: +90 212 527 4503\n\n"
                  "Stay calm. Help is on the way.",
        "urgent": True,
        "contacts": {
            "police": "155",
            "ambulance": "112",
            "fire": "110",
            "tourist_police": "+90 212 527 4503"
        }
    }


def handle_attraction(query: str, confidence: float) -> Dict:
    """Handle attraction queries"""
    # In real system, integrate with attraction database
    return {
        "type": "attraction",
        "message": "🏛️ **Top Istanbul Attractions:**\n\n"
                  "1. **Hagia Sophia** - Historic mosque/museum\n"
                  "2. **Topkapi Palace** - Ottoman imperial palace\n"
                  "3. **Blue Mosque** - Iconic 6-minaret mosque\n"
                  "4. **Grand Bazaar** - World's oldest covered market\n"
                  "5. **Bosphorus Cruise** - Scenic strait tour\n\n"
                  "Would you like details on any of these?",
        "suggestions": [
            "Tell me about Hagia Sophia",
            "How to get to Topkapi Palace",
            "Bosphorus cruise prices"
        ]
    }


def handle_restaurant(query: str, confidence: float) -> Dict:
    """Handle restaurant queries"""
    return {
        "type": "restaurant",
        "message": "🍽️ **Restaurant Recommendations:**\n\n"
                  "**Seafood:**\n"
                  "- Balıkçı Sabahattin (Sultanahmet)\n"
                  "- Tarihi Karaköy Balıkçısı (Karaköy)\n\n"
                  "**Traditional Turkish:**\n"
                  "- Hamdi Restaurant (Eminönü)\n"
                  "- Pandeli (Spice Bazaar)\n\n"
                  "**Kebab:**\n"
                  "- Zübeyir Ocakbaşı (Beyoğlu)\n"
                  "- Hamdi Et Lokantası (Eminönü)\n\n"
                  "Need more specific recommendations?",
        "suggestions": [
            "Vegetarian restaurants",
            "Budget-friendly dining",
            "Bosphorus view restaurants"
        ]
    }


def handle_transportation(query: str, confidence: float) -> Dict:
    """Handle transportation queries"""
    return {
        "type": "transportation",
        "message": "🚇 **Istanbul Transportation Guide:**\n\n"
                  "**Metro:** Fast, modern system\n"
                  "- M1: Airport to Yenikapı\n"
                  "- M2: Hacıosman to Yenikapı\n\n"
                  "**Tram:** T1 line covers major sites\n"
                  "- Sultanahmet, Eminönü, Karaköy\n\n"
                  "**İstanbulkart:** Rechargeable card for all transport\n"
                  "- Available at kiosks and stations\n\n"
                  "Need specific route info?",
        "suggestions": [
            "How to buy İstanbulkart",
            "Metro to Taksim",
            "Airport transfer options"
        ]
    }


def handle_weather(query: str, confidence: float) -> Dict:
    """Handle weather queries"""
    # In real system, integrate with weather API
    return {
        "type": "weather",
        "message": "🌤️ **Istanbul Weather:**\n\n"
                  "**Today:** Partly cloudy, 18°C\n"
                  "**Tomorrow:** Sunny, 20°C\n"
                  "**This Week:** Mild temperatures, occasional rain\n\n"
                  "Best time to visit: April-May, September-October",
        "suggestions": [
            "What to wear today",
            "Best season to visit"
        ]
    }


def handle_accommodation(query: str, confidence: float) -> Dict:
    """Handle accommodation queries"""
    return {
        "type": "accommodation",
        "message": "🏨 **Accommodation Options:**\n\n"
                  "**Budget:**\n"
                  "- Hostelworld hostels (€15-30/night)\n"
                  "- Budget hotels in Sultanahmet\n\n"
                  "**Mid-Range:**\n"
                  "- Boutique hotels in Beyoğlu (€60-100/night)\n"
                  "- Chain hotels near Taksim\n\n"
                  "**Luxury:**\n"
                  "- Ciragan Palace (Bosphorus view)\n"
                  "- Four Seasons Sultanahmet\n\n"
                  "What's your budget?",
        "suggestions": [
            "Hotels in Sultanahmet",
            "Boutique hotels with character",
            "Airbnb recommendations"
        ]
    }


def handle_museum(query: str, confidence: float) -> Dict:
    """Handle museum queries"""
    return {
        "type": "museum",
        "message": "🎨 **Istanbul Museums:**\n\n"
                  "**Top Museums:**\n"
                  "1. Hagia Sophia (€25)\n"
                  "2. Topkapi Palace Museum (€20)\n"
                  "3. Istanbul Archaeology Museums (€10)\n"
                  "4. Istanbul Modern (€12)\n"
                  "5. Pera Museum (€8)\n\n"
                  "**Museum Pass:** €85 (5 days, 12 museums)\n\n"
                  "Need hours or directions?",
        "suggestions": [
            "Museum pass worth it?",
            "Free museum days",
            "Art museums"
        ]
    }


def handle_shopping(query: str, confidence: float) -> Dict:
    """Handle shopping queries"""
    return {
        "type": "shopping",
        "message": "🛍️ **Shopping in Istanbul:**\n\n"
                  "**Markets:**\n"
                  "- Grand Bazaar (historic, souvenirs)\n"
                  "- Spice Bazaar (spices, sweets)\n"
                  "- Arasta Bazaar (carpets, ceramics)\n\n"
                  "**Malls:**\n"
                  "- İstinye Park (luxury brands)\n"
                  "- Zorlu Center (high-end shopping)\n\n"
                  "**Streets:**\n"
                  "- İstiklal Avenue (mixed brands)\n\n"
                  "What are you looking for?",
        "suggestions": [
            "Turkish carpet shopping",
            "Authentic souvenirs",
            "Local markets"
        ]
    }


def handle_family_activities(query: str, confidence: float) -> Dict:
    """Handle family activities queries"""
    return {
        "type": "family_activities",
        "message": "👨‍👩‍👧‍👦 **Family-Friendly Activities:**\n\n"
                  "**Attractions:**\n"
                  "- Miniaturk (miniature park)\n"
                  "- Istanbul Aquarium\n"
                  "- Vialand (theme park)\n\n"
                  "**Museums:**\n"
                  "- Rahmi M. Koç Museum (interactive)\n"
                  "- Istanbul Toy Museum\n\n"
                  "**Outdoor:**\n"
                  "- Emirgan Park\n"
                  "- Princes' Islands (bike rides)\n\n"
                  "Kids' age range?",
        "suggestions": [
            "Indoor activities for rainy days",
            "Educational activities",
            "Playgrounds and parks"
        ]
    }


def handle_general_info(query: str, confidence: float) -> Dict:
    """Handle general information queries"""
    return {
        "type": "general_info",
        "message": f"ℹ️ **How can I help you?**\n\n"
                  f"I can provide information about:\n"
                  f"- Attractions and sightseeing\n"
                  f"- Restaurants and dining\n"
                  f"- Transportation and getting around\n"
                  f"- Hotels and accommodation\n"
                  f"- Museums and culture\n"
                  f"- Shopping and markets\n"
                  f"- Family activities\n"
                  f"- Weather and best times to visit\n\n"
                  f"What would you like to know?",
        "suggestions": [
            "Show me top attractions",
            "Where to eat",
            "How to get around",
            "What's the weather like"
        ]
    }


# Intent handler mapping
INTENT_HANDLERS = {
    "emergency": handle_emergency,
    "attraction": handle_attraction,
    "restaurant": handle_restaurant,
    "transportation": handle_transportation,
    "weather": handle_weather,
    "accommodation": handle_accommodation,
    "museum": handle_museum,
    "shopping": handle_shopping,
    "family_activities": handle_family_activities,
    # Add remaining intents...
}


def process_user_query(query: str, user_context: Dict = None) -> Dict:
    """
    Main query processing function
    
    Args:
        query: User's question/request
        user_context: Optional context (location, preferences, etc.)
    
    Returns:
        Response dictionary with message and suggestions
    """
    # Step 1: Classify intent
    intent, confidence = intent_classifier.classify(query)
    
    # Step 2: Log for analytics
    logger.info(f"Query: '{query[:50]}...' → Intent: {intent} (conf: {confidence:.1%})")
    
    # Step 3: Get handler (fallback to general_info)
    handler = INTENT_HANDLERS.get(intent, handle_general_info)
    
    # Step 4: Generate response
    response = handler(query, confidence)
    
    # Step 5: Add metadata
    response['intent'] = intent
    response['confidence'] = confidence
    response['query'] = query
    
    return response


def demo():
    """Demo the system with various queries"""
    print("="*60)
    print("ISTANBUL AI CHAT SYSTEM - DEMO")
    print("="*60)
    print()
    
    test_queries = [
        "Acil kayboldum yardım edin!",
        "Where can I visit Hagia Sophia?",
        "Balık yemek için nerede gidebilirim?",
        "Metro nasıl kullanılır?",
        "Yarın hava nasıl olacak?",
        "Çocuklarla nereye gidebilirim?",
        "Looking for cheap hotel",
        "Which museums should I visit?",
        "Where is Grand Bazaar?",
        "Merhaba!",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"👤 USER: {query}")
        print(f"{'='*60}")
        
        # Process query
        response = process_user_query(query)
        
        # Display response
        print(f"\n🤖 ASSISTANT:")
        print(f"   Intent: {response['intent']} ({response['confidence']:.1%})")
        print()
        print(response['message'])
        
        if 'suggestions' in response:
            print("\n💡 Suggestions:")
            for suggestion in response['suggestions']:
                print(f"   • {suggestion}")


if __name__ == "__main__":
    demo()
