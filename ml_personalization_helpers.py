from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# =============================
# ML PERSONALIZATION HELPER METHODS
# =============================

def handle_preference_update(user_profile, message: str, user_id: str) -> str:
    """Handle user preference updates through natural language"""
    message_lower = message.lower()
    updates = []
    
    # Extract dietary preferences
    if 'vegetarian' in message_lower:
        user_profile.dietary_restrictions = list(set(user_profile.dietary_restrictions + ['vegetarian']))
        updates.append("dietary preferences (vegetarian)")
    
    if 'vegan' in message_lower:
        user_profile.dietary_restrictions = list(set(user_profile.dietary_restrictions + ['vegan']))
        updates.append("dietary preferences (vegan)")
        
    if 'halal' in message_lower:
        user_profile.dietary_restrictions = list(set(user_profile.dietary_restrictions + ['halal']))
        updates.append("dietary preferences (halal)")
    
    # Extract travel preferences
    if 'family' in message_lower or 'kids' in message_lower or 'children' in message_lower:
        user_profile.interests = list(set(user_profile.interests + ['family-friendly']))
        updates.append("travel style (family-friendly)")
        
    if 'solo' in message_lower or 'alone' in message_lower:
        user_profile.interests = list(set(user_profile.interests + ['solo-travel']))
        updates.append("travel style (solo travel)")
        
    if 'couple' in message_lower or 'romantic' in message_lower:
        user_profile.interests = list(set(user_profile.interests + ['romantic']))
        updates.append("travel style (romantic)")
    
    # Extract interests
    interests_keywords = {
        'history': ['history', 'historical', 'museum', 'ancient'],
        'food': ['food', 'cuisine', 'restaurant', 'eating', 'taste'],
        'culture': ['culture', 'cultural', 'traditional', 'local'],
        'shopping': ['shopping', 'bazaar', 'market', 'souvenir'],
        'nightlife': ['nightlife', 'bar', 'club', 'evening'],
        'art': ['art', 'gallery', 'artist', 'creative'],
        'architecture': ['architecture', 'building', 'mosque', 'palace']
    }
    
    for interest, keywords in interests_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            if interest not in user_profile.interests:
                user_profile.interests.append(interest)
                updates.append(f"interests ({interest})")
    
    # Extract budget preferences
    if any(word in message_lower for word in ['budget', 'cheap', 'affordable', 'economical']):
        user_profile.budget_range = 'budget'
        updates.append("budget preferences (budget-friendly)")
    elif any(word in message_lower for word in ['luxury', 'expensive', 'premium', 'high-end']):
        user_profile.budget_range = 'luxury'
        updates.append("budget preferences (luxury)")
    elif any(word in message_lower for word in ['mid-range', 'moderate', 'medium']):
        user_profile.budget_range = 'mid-range'
        updates.append("budget preferences (mid-range)")
    
    if updates:
        response = f"✅ I've updated your {', '.join(updates)}! Your personalized recommendations will now better match your preferences.\n\n"
        response += "🎯 Your current preferences:\n"
        response += f"• Interests: {', '.join(user_profile.interests) if user_profile.interests else 'None specified'}\n"
        if user_profile.dietary_restrictions:
            response += f"• Dietary: {', '.join(user_profile.dietary_restrictions)}\n"
        if user_profile.budget_range:
            response += f"• Budget: {user_profile.budget_range}\n"
        response += "\nFeel free to ask for recommendations - they'll be tailored just for you! 🌟"
    else:
        response = "I'd love to learn more about your preferences! You can tell me about:\n"
        response += "• Your interests (history, food, culture, shopping, etc.)\n"
        response += "• Dietary restrictions (vegetarian, vegan, halal, etc.)\n"
        response += "• Travel style (family-friendly, romantic, solo travel)\n"
        response += "• Budget preferences (budget, mid-range, luxury)\n\n"
        response += "Just tell me naturally, like 'I'm vegetarian and love history' or 'I prefer budget-friendly family activities'!"
    
    return response

def handle_recommendation_feedback(user_profile, message: str, user_id: str) -> str:
    """Handle user feedback on recommendations"""
    message_lower = message.lower()
    
    # Determine sentiment
    positive_words = ['loved', 'amazing', 'great', 'excellent', 'wonderful', 'fantastic', 'perfect']
    negative_words = ['didn\'t like', 'bad', 'terrible', 'awful', 'disappointing', 'not good']
    
    is_positive = any(word in message_lower for word in positive_words)
    is_negative = any(word in message_lower for word in negative_words)
    
    if is_positive:
        # Extract what they liked to reinforce similar recommendations
        if 'restaurant' in message_lower or 'food' in message_lower:
            if 'food' not in user_profile.interests:
                user_profile.interests.append('food')
        elif 'museum' in message_lower or 'history' in message_lower:
            if 'history' not in user_profile.interests:
                user_profile.interests.append('history')
        elif 'shopping' in message_lower or 'bazaar' in message_lower:
            if 'shopping' not in user_profile.interests:
                user_profile.interests.append('shopping')
        
        response = "🌟 Thank you for the positive feedback! I'm learning from your preferences to make even better recommendations.\n\n"
        response += "Your feedback helps me understand what you enjoy most about Istanbul. Would you like more similar recommendations?"
        
    elif is_negative:
        response = "😔 I'm sorry that didn't meet your expectations. Your feedback is valuable for improving future recommendations.\n\n"
        response += "Could you tell me what specifically you'd prefer? For example:\n"
        response += "• Different type of cuisine or atmosphere\n"  
        response += "• Different price range or location\n"
        response += "• Different activity style or timing\n\n"
        response += "This will help me suggest better options for you!"
        
    else:
        # General rating request
        response = "📝 I'd love to hear your thoughts! Please let me know:\n"
        response += "• What did you think of the recommendation?\n"
        response += "• What worked well or what could be improved?\n"
        response += "• Any specific preferences for future suggestions?\n\n"
        response += "Your feedback helps me personalize recommendations just for you! 🎯"
    
    return response

def get_personalization_insights(user_profile, user_id: str) -> str:
    """Provide insights about the user's personalization data"""
    response = f"📊 **Your Personalization Profile**\n\n"
    response += f"🆔 **User ID**: {user_profile.user_id}\n"
    response += f"� **User Type**: {user_profile.user_type.value if hasattr(user_profile.user_type, 'value') else str(user_profile.user_type)}\n"
    response += f"💬 **Interaction History**: {len(getattr(user_profile, 'interaction_history', []))} conversations\n\n"
    
    # Interests
    if user_profile.interests:
        response += f"🎯 **Your Interests**: {', '.join(user_profile.interests)}\n"
    else:
        response += f"🎯 **Your Interests**: None specified yet\n"
    
    # Dietary restrictions
    if user_profile.dietary_restrictions:
        response += f"🍽️ **Dietary Preferences**: {', '.join(user_profile.dietary_restrictions)}\n"
    else:
        response += f"🍽️ **Dietary Preferences**: None specified\n"
    
    # Budget preferences
    if user_profile.budget_range:
        response += f"💰 **Budget Range**: {user_profile.budget_range}\n"
    else:
        response += f"💰 **Budget Range**: Not specified\n"
        
    # Most visited areas
    if hasattr(user_profile, 'visit_frequency') and user_profile.visit_frequency:
        top_areas = list(user_profile.visit_frequency.keys())[:3]
        response += f"📍 **Favorite Areas**: {', '.join(top_areas)}\n"
    else:
        response += f"📍 **Favorite Areas**: None visited yet\n"
    
    # Personalization level
    total_data_points = len(user_profile.interests) + len(user_profile.dietary_restrictions) + \
                       (1 if user_profile.budget_range else 0) + len(getattr(user_profile, 'visit_frequency', {}))
    
    if total_data_points >= 5:
        personalization_level = "High"
    elif total_data_points >= 2:
        personalization_level = "Medium"
    else:
        personalization_level = "Low"
        
    response += f"\n🎭 **Personalization Level**: {personalization_level}\n"
    
    if personalization_level == "Low":
        response += "\n💡 **Tip**: Tell me more about your preferences to get better personalized recommendations!"
    
    response += f"\n🔒 **Privacy**: Your data is stored locally and used only to improve your experience."
    response += f"\nType 'show privacy settings' to manage your data or 'clear my data' to reset."
    
    return response

# =============================
# ML RECOMMENDATION ADAPTATION HELPERS
# =============================

def calculate_personalization_score(user_profile) -> float:
    """Calculate how complete and useful the user profile is for personalization"""
    score_components = {
        'basic_info': 0.2 if user_profile.travel_style else 0.0,
        'interests': min(len(user_profile.interests) * 0.1, 0.3),
        'preferences': min(len(user_profile.cuisine_preferences) * 0.05, 0.2),
        'behavioral_data': min(len(user_profile.interaction_history) * 0.02, 0.2),
        'feedback_data': min(len(user_profile.recommendation_feedback) * 0.03, 0.1)
    }
    
    total_score = sum(score_components.values())
    return min(max(total_score, 0.3), 1.0)  # Ensure minimum 0.3, maximum 1.0

def calculate_recommendation_compatibility(recommendation: Dict, user_profile, context) -> float:
    """Calculate how compatible a recommendation is with user preferences"""
    compatibility_score = 0.5  # Base score
    
    # Interest alignment
    if user_profile.interests:
        rec_category = recommendation.get('category', '').lower()
        interest_match = any(interest.lower() in rec_category or rec_category in interest.lower() 
                           for interest in user_profile.interests)
        if interest_match:
            compatibility_score += 0.2
    
    # Budget alignment
    rec_price_level = recommendation.get('price_level', 'moderate').lower()
    user_budget = (user_profile.budget_range or 'moderate').lower()
    if rec_price_level == user_budget:
        compatibility_score += 0.15
    
    # Travel style alignment
    if user_profile.travel_style:
        style_bonus = get_travel_style_bonus(recommendation, user_profile.travel_style)
        compatibility_score += style_bonus
    
    # Accessibility considerations
    if user_profile.accessibility_needs:
        accessibility_score = check_accessibility_compatibility(recommendation, user_profile)
        compatibility_score += accessibility_score
    
    return min(compatibility_score, 1.0)

# =============================
# INTENT CLASSIFICATION HELPERS
# =============================

def enhance_intent_classification(message: str) -> str:
    """Enhanced intent classification with attraction support"""
    message_lower = message.lower()
    
    # Transportation keywords
    transport_keywords = ['metro', 'bus', 'ferry', 'tram', 'taxi', 'transport', 'get to', 'how to reach']
    if any(keyword in message_lower for keyword in transport_keywords):
        return 'transportation_query'
    
    # Attraction keywords
    attraction_keywords = ['visit', 'see', 'attraction', 'museum', 'palace', 'mosque', 'tower', 'monument']
    if any(keyword in message_lower for keyword in attraction_keywords):
        return 'attraction_query'
    
    # Cultural keywords
    cultural_keywords = ['culture', 'cultural', 'traditional', 'heritage', 'historic']
    if any(keyword in message_lower for keyword in cultural_keywords):
        return 'cultural_query'
    
    return 'general_conversation'

def is_transportation_query(message: str) -> bool:
    """Check if message is about transportation"""
    transport_keywords = [
        'metro', 'bus', 'ferry', 'tram', 'taxi', 'transport', 'transportation',
        'get to', 'how to reach', 'how to get', 'travel to', 'go to',
        'metro station', 'bus stop', 'ferry terminal', 'airport',
        'dolmuş', 'metrobüs', 'vapur', 'otobüs'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in transport_keywords)

def is_restaurant_query(message: str) -> bool:
    """Check if message is about restaurants"""
    restaurant_keywords = [
        'restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal',
        'breakfast', 'lunch', 'dinner', 'café', 'coffee',
        'lokanta', 'restoran', 'yemek', 'kahvaltı'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in restaurant_keywords)

def is_neighborhood_query(message: str) -> bool:
    """Check if message is about neighborhoods"""
    neighborhood_keywords = [
        'neighborhood', 'district', 'area', 'quarter',
        'sultanahmet', 'beyoğlu', 'galata', 'taksim', 'kadıköy',
        'beşiktaş', 'şişli', 'fatih', 'üsküdar', 'ortaköy'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in neighborhood_keywords)

# =============================
# RESPONSE GENERATION HELPERS
# =============================

def generate_fallback_response(user_profile, interests: list = None) -> str:
    """Generate fallback response when other methods fail"""
    responses = [
        "I'm here to help you explore Istanbul! What would you like to know about the city?",
        "Let me know what you're interested in - restaurants, attractions, transportation, or anything else about Istanbul!",
        "I can help you with restaurant recommendations, finding attractions, getting around the city, and much more. What interests you?",
        "What would you like to discover about Istanbul today? I'm here to help with personalized recommendations!"
    ]
    
    # Use user profile to personalize the fallback
    if interests and 'food' in interests:
        return "I notice you're interested in food! Would you like restaurant recommendations, or are you curious about something else in Istanbul?"
    elif interests and 'history' in interests:
        return "Given your interest in history, I can recommend historical sites, museums, or help with other Istanbul questions. What would you like to know?"
    
    # Return a random friendly response
    import random
    return random.choice(responses)

def enhance_multi_intent_response(multi_intent_response: str, entities: dict, user_profile, current_hour: int) -> str:
    """Enhance multi-intent response with Istanbul-specific context and personalization"""
    enhanced_response = multi_intent_response
    
    # Add time-based context
    if current_hour < 11:
        time_context = "Since it's morning, consider places that serve good breakfast!"
    elif current_hour < 16:
        time_context = "Perfect timing for lunch recommendations!"
    else:
        time_context = "Great time for dinner suggestions!"
    
    # Add personalized context based on user profile
    personal_context = ""
    if user_profile.interests:
        if 'food' in user_profile.interests:
            personal_context = "Based on your love for food, I've focused on culinary experiences."
        elif 'family-friendly' in user_profile.interests:
            personal_context = "I've made sure these are family-friendly options."
    
    # Add dietary considerations
    dietary_context = ""
    if user_profile.dietary_restrictions:
        dietary_context = f"I've considered your {', '.join(user_profile.dietary_restrictions)} preferences."
    
    # Combine contexts
    context_additions = []
    if personal_context:
        context_additions.append(personal_context)
    if dietary_context:
        context_additions.append(dietary_context)
    if time_context:
        context_additions.append(time_context)
    
    if context_additions:
        enhanced_response += f"\n\n💡 {' '.join(context_additions)}"
    
    return enhanced_response

def generate_location_response(entities: dict, traffic_info: dict) -> str:
    """Generate location-based response"""
    if entities.get('neighborhoods'):
        neighborhood = entities['neighborhoods'][0]
        response = f"📍 You're asking about {neighborhood.title()}! "
        
        if traffic_info.get('status') == 'heavy':
            response += "Traffic is heavy right now, so consider walking or using metro if possible. "
        
        response += f"This area is known for its unique character and local attractions. What specifically would you like to know?"
        return response
    
    return "I'd be happy to help with location information! Could you tell me which area of Istanbul you're interested in?"

def generate_time_response(entities: dict, current_hour: int) -> str:
    """Generate time-based response"""
    if current_hour < 10:
        return "It's still early! Most attractions open around 9-10 AM. Perfect time for a traditional Turkish breakfast though!"
    elif current_hour > 22:
        return "It's getting late! Most museums and attractions are closed, but Istanbul's nightlife is just getting started. Looking for evening entertainment?"
    else:
        return "Good timing! Most places are open now. What would you like to visit or do?"

def generate_conversational_response(message: str, user_profile) -> str:
    """Generate conversational response for general chat"""
    responses = [
        "That's interesting! Istanbul has so much to offer. What aspect of the city interests you most?",
        "I love talking about Istanbul! There's always something new to discover. What would you like to explore?",
        "Istanbul is such a fascinating city where East meets West. What brings you here or what are you curious about?"
    ]
    
    # Personalize based on user profile
    if user_profile.interests:
        if 'history' in user_profile.interests:
            return "Given your interest in history, you'll love Istanbul's rich Byzantine and Ottoman heritage! What historical aspect interests you most?"
        elif 'food' in user_profile.interests:
            return "As a food lover, you're in for a treat in Istanbul! The culinary scene here is incredible. What type of cuisine are you in the mood for?"
    
    import random
    return random.choice(responses)

# =============================
# LOCATION AND GPS HELPERS
# =============================

def get_or_request_gps_location(user_profile, context) -> dict:
    """Get GPS location or request it from user"""
    if user_profile.gps_location:
        return user_profile.gps_location
    
    # Mock GPS location for testing (in real app, this would request actual GPS)
    return {'lat': 41.0082, 'lng': 28.9784, 'accuracy': 10}  # Istanbul center

def request_location_for_restaurant(message: str, user_profile) -> str:
    """Request location information for restaurant recommendations"""
    return """📍 **Location needed for personalized recommendations!**

To give you the best restaurant suggestions, I need to know your location. You can:

🎯 **Tell me the neighborhood**: "I'm in Sultanahmet" or "Near Taksim Square"
📱 **Share your GPS location**: Enable location sharing for precise recommendations
🗺️ **Describe nearby landmarks**: "I'm near the Blue Mosque" or "Close to Galata Tower"

Which area of Istanbul are you in or planning to visit?"""

def extract_or_request_location(message: str, user_profile, context, gps_location: dict) -> dict:
    """Extract location information from various sources"""
    if gps_location:
        # Convert GPS to neighborhood (simplified)
        lat, lng = gps_location['lat'], gps_location['lng']
        
        # Simple neighborhood mapping based on coordinates
        if 41.000 <= lat <= 41.015 and 28.975 <= lng <= 28.985:
            return {'neighborhood': 'sultanahmet', 'source': 'gps'}
        elif 41.025 <= lat <= 41.035 and 28.970 <= lng <= 28.985:
            return {'neighborhood': 'beyoğlu', 'source': 'gps'}
        elif 41.015 <= lat <= 41.030 and 28.985 <= lng <= 29.000:
            return {'neighborhood': 'galata', 'source': 'gps'}
        else:
            return {'neighborhood': 'istanbul_center', 'source': 'gps'}
    
    # Fallback to user's known location
    if user_profile.current_location:
        return {'neighborhood': user_profile.current_location, 'source': 'profile'}
    
    # Default fallback
    return {'neighborhood': 'sultanahmet', 'source': 'default'}

# =============================
# TRANSPORTATION HELPERS
# =============================

def process_transportation_query(message: str, user_profile, current_time, context=None) -> str:
    """Process transportation-related queries"""
    message_lower = message.lower()
    
    response = "🚇 **Istanbul Transportation Help**\n\n"
    
    if 'metro' in message_lower:
        response += "**Metro System:**\n"
        response += "• M1: Airport ↔ Yenikapı\n"
        response += "• M2: Veliefendi ↔ Hacıosman\n"
        response += "• M3: Kirazlı ↔ Başakşehir\n"
        response += "• Operating hours: 06:00 - 00:30\n\n"
    
    if 'bus' in message_lower:
        response += "**Bus System:**\n"
        response += "• Extensive network covering all districts\n"
        response += "• Use IstanbulKart for payment\n"
        response += "• Metrobüs: High-speed bus line\n\n"
    
    if 'ferry' in message_lower:
        response += "**Ferry System:**\n"
        response += "• Scenic way to cross the Bosphorus\n"
        response += "• Main routes: Eminönü ↔ Kadıköy ↔ Üsküdar\n"
        response += "• Great for sightseeing!\n\n"
    
    response += "💡 **Tips:**\n"
    response += "• Get an IstanbulKart for all public transport\n"
    response += "• Download BiTaksi or Uber for taxis\n"
    response += "• Traffic is heaviest 8-10 AM and 5-7 PM\n"
    
    return response

def process_neighborhood_query(message: str, user_profile, current_time) -> str:
    """Process neighborhood-related queries"""
    message_lower = message.lower()
    
    neighborhoods = {
        'sultanahmet': {
            'description': 'Historic heart of Istanbul with major attractions',
            'highlights': ['Hagia Sophia', 'Blue Mosque', 'Topkapi Palace'],
            'food': 'Traditional Ottoman cuisine',
            'best_time': 'Early morning or late afternoon'
        },
        'beyoğlu': {
            'description': 'Modern cultural district with vibrant nightlife',
            'highlights': ['Galata Tower', 'Istiklal Street', 'Pera Museum'],
            'food': 'International cuisine and trendy restaurants',
            'best_time': 'Evening for nightlife, afternoon for shopping'
        },
        'kadıköy': {
            'description': 'Hip Asian side district with local atmosphere',
            'highlights': ['Moda coastline', 'Bagdat Street', 'Local markets'],
            'food': 'Authentic Turkish street food',
            'best_time': 'Afternoon and evening'
        }
    }
    
    for neighborhood, info in neighborhoods.items():
        if neighborhood in message_lower:
            response = f"🏘️ **{neighborhood.title()} Overview**\n\n"
            response += f"📝 {info['description']}\n\n"
            response += f"🎯 **Must-see**: {', '.join(info['highlights'])}\n"
            response += f"🍽️ **Food**: {info['food']}\n"
            response += f"⏰ **Best time**: {info['best_time']}\n\n"
            response += "What specific information would you like about this area?"
            return response
    
    return "I'd love to help you explore Istanbul's neighborhoods! Which area interests you - Sultanahmet, Beyoğlu, Kadıköy, or somewhere else?"

# =============================
# ML SCORING HELPERS
# =============================

def get_travel_style_bonus(recommendation: dict, travel_style: str) -> float:
    """Calculate bonus based on travel style alignment"""
    if travel_style == 'family' and recommendation.get('family_friendly', False):
        return 0.15
    elif travel_style == 'couple' and recommendation.get('romantic', False):
        return 0.15
    elif travel_style == 'solo' and recommendation.get('solo_friendly', True):
        return 0.10
    return 0.0

def check_accessibility_compatibility(recommendation: dict, user_profile) -> float:
    """Check accessibility compatibility"""
    if user_profile.accessibility_needs and recommendation.get('accessible', False):
        return 0.10
    return 0.0

def get_group_type_bonus(recommendation: dict, user_profile) -> float:
    """Calculate group type bonus"""
    if user_profile.group_type == 'family' and recommendation.get('family_friendly', False):
        return 0.10
    elif user_profile.group_type == 'couple' and recommendation.get('romantic', False):
        return 0.10
    return 0.05

def get_time_preference_bonus(recommendation: dict, user_profile, current_hour: int) -> float:
    """Calculate time preference bonus"""
    suitable_times = recommendation.get('suitable_times', [])
    
    if current_hour < 11 and 'breakfast' in suitable_times:
        return 0.10
    elif 11 <= current_hour < 16 and 'lunch' in suitable_times:
        return 0.10
    elif current_hour >= 19 and 'dinner' in suitable_times:
        return 0.10
    
    return 0.05

def find_similar_recommendations(recommendation: dict, user_profile) -> dict:
    """Find similar recommendations in feedback history"""
    similar_recs = {}
    rec_category = recommendation.get('category', '').lower()
    rec_location = recommendation.get('location', '').lower()
    
    for rec_id, rating in user_profile.recommendation_feedback.items():
        # Simple similarity based on category and location matching
        if rec_category in rec_id.lower() or rec_location in rec_id.lower():
            similar_recs[rec_id] = rating
    
    return similar_recs

def get_time_period(hour: int) -> str:
    """Get time period from hour"""
    if hour < 11:
        return 'morning'
    elif hour < 16:
        return 'afternoon'
    elif hour < 20:
        return 'evening'
    else:
        return 'night'

def calculate_interest_match_score(recommendation: dict, user_profile) -> float:
    """Calculate how well recommendation matches user interests"""
    if not user_profile.interests:
        return 0.5
    
    rec_category = recommendation.get('category', '').lower()
    matches = sum(1 for interest in user_profile.interests 
                 if interest.lower() in rec_category or rec_category in interest.lower())
    
    return min(matches / len(user_profile.interests), 1.0)

def calculate_travel_style_score(recommendation: dict, user_profile) -> float:
    """Calculate travel style compatibility score"""
    if not user_profile.travel_style:
        return 0.5
    
    style_bonuses = {
        'family': recommendation.get('family_friendly', False),
        'couple': recommendation.get('romantic', False),
        'solo': recommendation.get('solo_friendly', True),
        'group': recommendation.get('group_friendly', True)
    }
    
    return 0.9 if style_bonuses.get(user_profile.travel_style, False) else 0.4

def calculate_budget_score(recommendation: dict, user_profile) -> float:
    """Calculate budget compatibility score"""
    user_budget = (user_profile.budget_range or 'moderate').lower()
    rec_price = recommendation.get('price_level', 'moderate').lower()
    
    if user_budget == rec_price:
        return 1.0
    elif abs(['budget', 'moderate', 'expensive'].index(user_budget) - 
             ['budget', 'moderate', 'expensive'].index(rec_price)) == 1:
        return 0.6
    else:
        return 0.3

def calculate_accessibility_score(recommendation: dict, user_profile) -> float:
    """Calculate accessibility compatibility score"""
    if not user_profile.accessibility_needs:
        return 0.7  # Neutral score
    
    return 1.0 if recommendation.get('accessible', False) else 0.2

def calculate_time_suitability_score(recommendation: dict, current_time) -> float:
    """Calculate time suitability score"""
    from datetime import datetime
    
    current_hour = current_time.hour if hasattr(current_time, 'hour') else datetime.now().hour
    suitable_times = recommendation.get('suitable_times', ['morning', 'afternoon', 'evening'])
    time_period = get_time_period(current_hour)
    
    return 0.9 if time_period in suitable_times else 0.5

def calculate_feedback_score(recommendation: dict, user_profile) -> float:
    """Calculate score based on past feedback"""
    if not hasattr(user_profile, 'recommendation_feedback') or not user_profile.recommendation_feedback:
        return 0.5
    
    similar_recs = find_similar_recommendations(recommendation, user_profile.recommendation_feedback)
    if not similar_recs:
        return 0.5
    
    avg_rating = sum(similar_recs.values()) / len(similar_recs)
    return avg_rating / 5.0  # Normalize to 0-1

def calculate_interaction_history_score(recommendation: dict, user_profile) -> float:
    """Calculate score based on interaction history"""
    interaction_history = getattr(user_profile, 'interaction_history', [])
    if not interaction_history:
        return 0.5
    
    # Simple scoring based on interaction frequency
    total_interactions = len(interaction_history)
    return min(total_interactions * 0.1, 1.0)

def apply_behavioral_patterns(recommendation: dict, user_profile) -> float:
    """Apply learned behavioral patterns to recommendation scoring"""
    from datetime import datetime
    
    pattern_score = 0.5  # Base score
    
    # Analyze past feedback
    if user_profile.recommendation_feedback:
        similar_recs = find_similar_recommendations(recommendation, user_profile.recommendation_feedback)
        if similar_recs:
            avg_feedback = sum(similar_recs.values()) / len(similar_recs)
            pattern_score += (avg_feedback - 0.5) * 0.3  # Adjust based on past feedback
    
    # Frequency patterns
    rec_location = recommendation.get('location', '').lower()
    if hasattr(user_profile, 'visit_frequency') and rec_location in user_profile.visit_frequency:
        # Boost score for frequently visited areas, but add some variety
        visit_count = user_profile.visit_frequency[rec_location]
        frequency_bonus = min(visit_count * 0.05, 0.2) - (visit_count * 0.01)  # Diminishing returns
        pattern_score += frequency_bonus
    
    # Temporal patterns
    current_hour = datetime.now().hour
    if hasattr(user_profile, 'preferred_times') and user_profile.preferred_times:
        time_period = get_time_period(current_hour)
        if time_period in user_profile.preferred_times:
            pattern_score += 0.1
    
    return min(max(pattern_score, 0.0), 1.0)

# =============================
# PROFILE MANAGEMENT HELPERS
# =============================

def update_user_profile(user_profile, message: str, intent: str, entities: dict):
    """Update user profile based on interaction"""
    from datetime import datetime
    
    # Initialize and update interaction count
    if not hasattr(user_profile, 'interaction_count'):
        user_profile.interaction_count = 0
    user_profile.interaction_count += 1
    
    # Update last interaction time
    user_profile.last_interaction = datetime.now()
    
    # Add to interaction history if available
    if hasattr(user_profile, 'interaction_history'):
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'intent': intent,
            'entities': entities
        }
        user_profile.interaction_history.append(interaction)
        
        # Keep only last 50 interactions
        if len(user_profile.interaction_history) > 50:
            user_profile.interaction_history = user_profile.interaction_history[-50:]
    
    # Update visited locations based on entities
    if entities.get('neighborhoods'):
        for neighborhood in entities['neighborhoods']:
            if hasattr(user_profile, 'visit_frequency'):
                user_profile.visit_frequency[neighborhood] = user_profile.visit_frequency.get(neighborhood, 0) + 1
            else:
                user_profile.visit_frequency = {neighborhood: 1}
    
    # Update interests based on intent
    if intent == 'restaurant_query' and 'food' not in user_profile.interests:
        user_profile.interests.append('food')
    elif intent == 'transportation_query' and 'transportation' not in user_profile.interests:
        user_profile.interests.append('transportation')
    elif intent == 'attraction_query' and 'sightseeing' not in user_profile.interests:
        user_profile.interests.append('sightseeing')

def update_context_memory(context, message: str, entities: dict, intent: str):
    """Update conversation context memory"""
    from datetime import datetime
    
    # Update current topic based on intent
    if intent in ['restaurant_query', 'restaurant_recommendation']:
        context.current_topic = 'restaurant_search'
    elif intent == 'transportation_query':
        context.current_topic = 'transportation'
    elif intent in ['attraction_query', 'cultural_query']:
        context.current_topic = 'attractions'
    else:
        context.current_topic = 'general'
    
    # Store recent entities for context
    if not hasattr(context, 'recent_entities'):
        context.recent_entities = []
    
    context.recent_entities.append({
        'timestamp': datetime.now().isoformat(),
        'entities': entities,
        'intent': intent
    })
    
    # Keep only last 5 entity sets for context
    if len(context.recent_entities) > 5:
        context.recent_entities = context.recent_entities[-5:]

# =============================
# REAL-TIME DATA HELPERS
# =============================

def get_transport_status() -> dict:
    """Get real-time transport status (mock implementation)"""
    from datetime import datetime
    try:
        # Mock transport status - in real implementation, this would connect to IBB API
        return {
            'metro': {
                'status': 'operational',
                'delays': [],
                'message': 'All metro lines running normally'
            },
            'bus': {
                'status': 'operational', 
                'delays': ['Line 28: 5 min delay due to traffic'],
                'message': 'Minor delays on some bus routes'
            },
            'ferry': {
                'status': 'operational',
                'delays': [],
                'message': 'Ferry services running on schedule'
            },
            'traffic_density': 'moderate',
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unavailable',
            'message': 'Transport information temporarily unavailable'
        }

def get_traffic_status() -> dict:
    """Get real-time traffic status (mock implementation)"""
    from datetime import datetime
    try:
        # Mock traffic status - in real implementation, this would connect to traffic APIs
        return {
            'overall_status': 'moderate',
            'congestion_level': 65,
            'problem_areas': [
                'Bosphorus Bridge - Heavy traffic',
                'Fatih Sultan Mehmet Bridge - Moderate congestion',
                'E-5 Highway (European side) - Slow moving'
            ],
            'estimated_travel_times': {
                'Sultanahmet to Taksim': '25-35 minutes',
                'Kadıköy to Beşiktaş': '30-40 minutes',
                'Airport to Sultanahmet': '45-60 minutes'
            },
            'recommendations': [
                'Use metro when possible for cross-city travel',
                'Consider ferry for Bosphorus crossings',
                'Avoid E-5 highway during peak hours'
            ],
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unavailable',
            'message': 'Traffic information temporarily unavailable'
        }

def get_local_events() -> dict:
    """Get local events information (curated local events)"""
    from datetime import datetime
    try:
        # Curated local events - this could be expanded with real event data
        current_month = datetime.now().month
        
        # Sample events based on season/month
        events = []
        
        if current_month in [6, 7, 8]:  # Summer
            events = [
                {
                    'name': 'Istanbul Music Festival',
                    'location': 'Various venues',
                    'type': 'music',
                    'description': 'Classical music performances across the city'
                },
                {
                    'name': 'Bosphorus Sunset Concerts',
                    'location': 'Ortaköy',
                    'type': 'music',
                    'description': 'Evening concerts with Bosphorus views'
                }
            ]
        elif current_month in [9, 10, 11]:  # Autumn
            events = [
                {
                    'name': 'Istanbul Biennial',
                    'location': 'Various galleries',
                    'type': 'art',
                    'description': 'Contemporary art exhibitions'
                },
                {
                    'name': 'Autumn Food Festival',
                    'location': 'Galata',
                    'type': 'food',
                    'description': 'Seasonal Turkish cuisine showcase'
                }
            ]
        else:  # Winter/Spring
            events = [
                {
                    'name': 'Tulip Festival',
                    'location': 'Emirgan Park',
                    'type': 'nature',
                    'description': 'Beautiful tulip displays across the city'
                }
            ]
        
        return {
            'current_events': events,
            'event_count': len(events),
            'categories': list(set([event['type'] for event in events])),
            'last_updated': datetime.now().isoformat(),
            'note': 'Curated local events - check official sources for exact dates and times'
        }
        
    except Exception as e:
        return {
            'current_events': [],
            'status': 'unavailable',
            'message': 'Event information temporarily unavailable'
        }

# =============================
# ADDITIONAL SCORING HELPERS
# =============================

def find_recommendation_by_id(recommendation_id: str, user_profile) -> dict:
    """Find a recommendation by ID in user's interaction history"""
    
    # Search in recent interactions
    for interaction in user_profile.interaction_history:
        if hasattr(interaction, 'recommendations'):
            for rec in interaction.recommendations:
                if rec.get('id') == recommendation_id:
                    return rec
    
    # Search in stored recommendations
    if hasattr(user_profile, 'stored_recommendations'):
        for rec in user_profile.stored_recommendations:
            if rec.get('id') == recommendation_id:
                return rec
    
    return None

def calculate_scoring_factors(recommendation: dict, user_profile) -> dict:
    """Calculate detailed scoring factors for a recommendation"""
    factors = {
        'interest_match': calculate_interest_match_score(recommendation, user_profile),
        'travel_style_fit': calculate_travel_style_score(recommendation, user_profile),
        'budget_alignment': calculate_budget_score(recommendation, user_profile),
        'accessibility_score': calculate_accessibility_score(recommendation, user_profile),
        'time_suitability': calculate_time_suitability_score(recommendation, datetime.now()),
        'feedback_score': calculate_feedback_score(recommendation, user_profile),
        'interaction_history_score': calculate_interaction_history_score(recommendation, user_profile)
    }
    
    return factors

def get_recommendation_explanation(user_profiles: dict, recommendation_id: str, user_id: str) -> dict:
    """Generate detailed explanation for why a specific recommendation was made"""
    
    if user_id not in user_profiles:
        return {'error': 'User profile not found'}
    
    user_profile = user_profiles[user_id]
    
    # Find the recommendation in recent interactions
    recommendation_data = find_recommendation_by_id(recommendation_id, user_profile)
    
    if not recommendation_data:
        return {'error': 'Recommendation not found'}
    
    explanation = {
        'recommendation_id': recommendation_id,
        'recommendation_name': recommendation_data.get('name', 'Unknown'),
        'explanation_summary': generate_personalization_reason(recommendation_data, user_profile),
        'detailed_factors': generate_detailed_explanation_factors(recommendation_data, user_profile),
        'transparency_info': generate_transparency_info(user_profile),
        'data_usage': explain_data_usage(user_profile),
        'confidence_breakdown': explain_confidence_score(recommendation_data, user_profile),
        'alternatives_considered': explain_alternatives(recommendation_data, user_profile),
        'privacy_context': get_privacy_context(user_profile)
    }
    
    return explanation

def get_meal_context(hour: int) -> str:
    """Determine meal context based on hour"""
    if hour < 11:
        return "breakfast"
    elif hour < 16:
        return "lunch"
    else:
        return "dinner"

def generate_personalization_reason(recommendation: dict, user_profile) -> str:
    """Generate human-readable reason for why this recommendation was personalized"""
    
    reasons = []
    
    # Interest-based reasons
    if user_profile.interests:
        rec_category = recommendation.get('category', '').lower()
        matching_interests = [interest for interest in user_profile.interests 
                            if interest.lower() in rec_category or rec_category in interest.lower()]
        if matching_interests:
            reasons.append(f"matches your interest in {', '.join(matching_interests)}")
    
    # Travel style reasons
    if user_profile.travel_style == 'family' and recommendation.get('family_friendly', False):
        reasons.append("perfect for families")
    elif user_profile.travel_style == 'solo' and recommendation.get('solo_friendly', True):
        reasons.append("great for solo travelers")
    elif user_profile.travel_style == 'couple' and recommendation.get('romantic', False):
        reasons.append("romantic atmosphere")
    
    # Budget reasons
    user_budget = user_profile.budget_range or 'moderate'
    if recommendation.get('price_level', '').lower() == user_budget.lower():
        reasons.append(f"fits your {user_budget} budget")
    
    # Accessibility reasons
    if user_profile.accessibility_needs and recommendation.get('accessible', False):
        reasons.append("accessible for your needs")
    
    # Past behavior reasons
    if hasattr(user_profile, 'favorite_neighborhoods') and user_profile.favorite_neighborhoods:
        rec_location = recommendation.get('location', '').lower()
        matching_neighborhoods = [n for n in user_profile.favorite_neighborhoods if n.lower() in rec_location]
        if matching_neighborhoods:
            reasons.append(f"in your favorite area ({matching_neighborhoods[0]})")
    
    if not reasons:
        return "recommended based on your profile"
    
    return "Recommended because it " + " and ".join(reasons)

def calculate_confidence_level(ml_score: float, user_profile) -> str:
    """Calculate confidence level for the recommendation"""
    
    profile_completeness = getattr(user_profile, 'profile_completeness', 0.5)
    
    if ml_score >= 0.8 and profile_completeness >= 0.7:
        return "very_high"
    elif ml_score >= 0.7 and profile_completeness >= 0.5:
        return "high"
    elif ml_score >= 0.6 and profile_completeness >= 0.3:
        return "medium"
    else:
        return "low"

def generate_explanation_summary(recommendation: dict, user_profile) -> str:
    """Generate summary explanation for recommendation"""
    reasons = []
    
    # Interest match
    if user_profile.interests:
        rec_category = recommendation.get('category', '').lower()
        matching_interests = [interest for interest in user_profile.interests 
                            if interest.lower() in rec_category or rec_category in interest.lower()]
        if matching_interests:
            reasons.append(f"Matches your {', '.join(matching_interests)} interests")
    
    # Budget alignment
    user_budget = user_profile.budget_range or 'moderate'
    if recommendation.get('price_level', '').lower() == user_budget.lower():
        reasons.append(f"Fits your {user_budget} budget")
    
    # Travel style
    if user_profile.travel_style:
        if user_profile.travel_style == 'family' and recommendation.get('family_friendly', False):
            reasons.append("Family-friendly")
        elif user_profile.travel_style == 'couple' and recommendation.get('romantic', False):
            reasons.append("Perfect for couples")
    
    if not reasons:
        return "Based on your general preferences"
    
    return ". ".join(reasons)

def show_privacy_settings(user_profile, user_id: str) -> str:
    """Show current privacy settings and available controls"""
    return f"""🔒 Privacy Settings for User {user_id}

Current Settings:
• Data Collection: Enabled (for personalized recommendations)
• Location Sharing: Ask each time
• Recommendation History: Stored locally
• Profile Analytics: Basic level

Available Controls:
• 'disable location sharing' - Stop location-based suggestions
• 'clear my data' - Remove all stored information
• 'show my data' - View all stored information
• 'explain data usage' - Learn how your data is used

Your privacy matters! You can control your data at any time."""

def show_user_data(user_profile, user_id: str) -> str:
    """Show all data stored about the user"""
    interaction_count = len(getattr(user_profile, 'interaction_history', []))
    feedback_count = len(getattr(user_profile, 'recommendation_feedback', {}))
    
    return f"""📊 Your Data Summary for User {user_id}

Stored Information:
• Profile creation date
• Preference categories: {len(user_profile.interests)} interests
• Interaction history: {interaction_count} interactions
• Recommendation feedback: {feedback_count} ratings
• Travel style: {user_profile.travel_style or 'Not specified'}
• Budget preference: {user_profile.budget_range or 'Not specified'}

Data Usage:
• Used only for personalizing your experience
• Never shared with third parties
• Stored locally on secure servers
• You can delete anytime with 'clear my data'

Profile completeness: {getattr(user_profile, 'profile_completeness', 0.3):.1%}"""

def clear_user_data(user_profiles: dict, active_conversations: dict, user_id: str) -> str:
    """Clear all user data"""
    if user_id in user_profiles:
        del user_profiles[user_id]
    
    # Clear active conversations for this user
    sessions_to_remove = [session_id for session_id, context in active_conversations.items() 
                         if hasattr(context, 'user_profile') and context.user_profile.user_id == user_id]
    
    for session_id in sessions_to_remove:
        del active_conversations[session_id]
    
    return f"""✅ Data Cleared Successfully

All your data has been permanently deleted:
• User profile and preferences
• Interaction history
• Recommendation feedback
• Conversation context
• Learning patterns

You can start fresh anytime! Your privacy is protected.
Feel free to interact normally - I'll only remember what you tell me in this session."""

def collect_recommendation_feedback(user_profiles: dict, user_id: str, recommendation_id: str, rating: float, feedback_text: str = None) -> bool:
    """Collect user feedback on recommendations for ML improvement"""
    from datetime import datetime
    
    if user_id not in user_profiles:
        return False
    
    user_profile = user_profiles[user_id]
    
    # Initialize recommendation_feedback if not present
    if not hasattr(user_profile, 'recommendation_feedback'):
        user_profile.recommendation_feedback = {}
    
    # Store feedback
    user_profile.recommendation_feedback[recommendation_id] = rating
    
    # Update success rate
    all_ratings = list(user_profile.recommendation_feedback.values())
    user_profile.recommendation_success_rate = sum(r >= 3.0 for r in all_ratings) / len(all_ratings)
    
    # Update satisfaction score (weighted average)
    user_profile.satisfaction_score = (getattr(user_profile, 'satisfaction_score', 0.8) * 0.8 + (rating / 5.0) * 0.2)
    
    # Store detailed feedback if provided
    if feedback_text:
        feedback_entry = {
            'recommendation_id': recommendation_id,
            'rating': rating,
            'text': feedback_text,
            'timestamp': datetime.now().isoformat()
        }
        
        if not hasattr(user_profile, 'learning_patterns'):
            user_profile.learning_patterns = {}
        
        if 'detailed_feedback' not in user_profile.learning_patterns:
            user_profile.learning_patterns['detailed_feedback'] = []
        
        user_profile.learning_patterns['detailed_feedback'].append(feedback_entry)
        
        # Keep only last 50 feedback entries
        if len(user_profile.learning_patterns['detailed_feedback']) > 50:
            user_profile.learning_patterns['detailed_feedback'] = user_profile.learning_patterns['detailed_feedback'][-50:]
    
    # Update profile completeness
    recalculate_profile_completeness(user_profile)
    
    return True

def update_user_interests(user_profiles: dict, user_id: str, interests: list, travel_style: str = None, accessibility_needs: str = None) -> bool:
    """Update user interests and preferences for better personalization"""
    from datetime import datetime
    
    if user_id not in user_profiles:
        return False
    
    user_profile = user_profiles[user_id]
    
    # Update interests
    if interests:
        current_interests = getattr(user_profile, 'interests', [])
        user_profile.interests = list(set(current_interests + interests))  # Avoid duplicates
    
    # Update travel style if provided
    if travel_style:
        user_profile.travel_style = travel_style
    
    # Update accessibility needs if provided
    if accessibility_needs:
        user_profile.accessibility_needs = accessibility_needs
    
    # Update last interaction time
    user_profile.last_interaction = datetime.now()
    
    # Recalculate profile completeness
    recalculate_profile_completeness(user_profile)
    
    return True

def recalculate_profile_completeness(user_profile):
    """Recalculate the completeness score of user profile"""
    
    completeness_factors = {
        'basic_travel_style': 0.2 if getattr(user_profile, 'travel_style', None) else 0.0,
        'interests': min(len(getattr(user_profile, 'interests', [])) * 0.1, 0.3),
        'cuisine_preferences': min(len(getattr(user_profile, 'cuisine_preferences', [])) * 0.05, 0.2),
        'budget_range': 0.1 if getattr(user_profile, 'budget_range', None) else 0.0,
        'accessibility_needs': 0.1 if getattr(user_profile, 'accessibility_needs', None) else 0.0,
        'interaction_history': min(len(getattr(user_profile, 'interaction_history', [])) * 0.02, 0.2)
    }
    
    user_profile.profile_completeness = sum(completeness_factors.values())
