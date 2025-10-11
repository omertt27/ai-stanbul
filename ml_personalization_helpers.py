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
        if user_profile.travel_style == 'family' and recommendation.get('family_friendly', False):
            compatibility_score += 0.15
        elif user_profile.travel_style == 'couple' and recommendation.get('romantic', False):
            compatibility_score += 0.15
        elif user_profile.travel_style == 'solo' and recommendation.get('solo_friendly', True):
            compatibility_score += 0.10
    
    # Accessibility considerations
    if user_profile.accessibility_needs:
        if recommendation.get('accessible', False):
            compatibility_score += 0.10
    
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

def generate_conversational_response_enhanced(message: str, context, user_profile) -> str:
    """Generate enhanced conversational response with ML personalization"""
    try:
        # Time-based greeting
        current_hour = context.current_time.hour if hasattr(context, 'current_time') and hasattr(context.current_time, 'hour') else 12
        time_greeting = get_time_greeting(current_hour)
        
        # Analyze message for context
        message_lower = message.lower()
        
        # Handle common conversational patterns
        if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return f"{time_greeting} How can I help you explore Istanbul today? I can recommend restaurants, attractions, neighborhoods, or help with transportation!"
        
        elif any(word in message_lower for word in ['thank you', 'thanks', 'appreciate']):
            responses = [
                "You're very welcome! I'm here whenever you need help exploring Istanbul! 😊",
                "My pleasure! Feel free to ask about anything else in Istanbul! 🌟",
                "Happy to help! Let me know if you need more Istanbul recommendations! 🏛️"
            ]
            import random
            return random.choice(responses)
        
        elif any(word in message_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
            return "Safe travels and enjoy your time in Istanbul! Feel free to come back anytime for more recommendations! 🌅✨"
        
        elif any(word in message_lower for word in ['help', 'assist', 'support']):
            return """I'm here to help you with everything Istanbul! I can assist with:
            
🍽️ **Restaurant recommendations** - Traditional Turkish cuisine, international food, budget or luxury options
🏛️ **Attractions & museums** - Historical sites, cultural experiences, hidden gems
🌆 **Neighborhoods** - Best areas for shopping, nightlife, local culture
🚇 **Transportation** - Metro, bus, ferry connections and travel tips
🏨 **Accommodation** - Area recommendations based on your interests
📅 **Events & activities** - Current happenings and seasonal recommendations

What would you like to explore first?"""
        
        elif any(word in message_lower for word in ['weather', 'temperature', 'climate']):
            return """Istanbul has a transitional climate between Mediterranean and humid subtropical:

🌤️ **Current season tips:**
• **Spring (Mar-May)**: Perfect for walking, mild temperatures
• **Summer (Jun-Aug)**: Hot and humid, great for Bosphorus tours
• **Autumn (Sep-Nov)**: Ideal weather, fewer crowds
• **Winter (Dec-Feb)**: Mild but rainy, perfect for museums and indoor attractions

Would you like specific recommendations based on the current weather?"""
        
        elif any(word in message_lower for word in ['language', 'turkish', 'english', 'speak']):
            return """🗣️ **Language in Istanbul:**
• **Turkish** is the official language
• **English** is widely spoken in tourist areas, hotels, and restaurants
• **Arabic, German, Russian** are also common in many areas
• Most signs in tourist areas have English translations
• Restaurant menus often have English versions

Don't worry about the language barrier - Istanbul is very tourist-friendly! Would you like some basic Turkish phrases?"""
        
        # Personalized response based on user profile
        interests = getattr(user_profile, 'interests', [])
        travel_style = getattr(user_profile, 'travel_style', 'general')
        
        if interests:
            interest_text = f"Based on your interests in {', '.join(interests[:3])}, "
        else:
            interest_text = ""
        
        if travel_style == 'luxury':
            style_suggestion = "I can recommend premium experiences and upscale venues"
        elif travel_style == 'budget':
            style_suggestion = "I know great budget-friendly options and free activities"
        elif travel_style == 'family':
            style_suggestion = "I can suggest family-friendly activities and kid-safe areas"
        else:
            style_suggestion = "I can help you discover the best of Istanbul"
        
        # Default conversational response
        default_responses = [
            f"{time_greeting} {interest_text}{style_suggestion}. What aspect of Istanbul interests you most?",
            f"I'd love to help you explore Istanbul! {interest_text}I can recommend restaurants, attractions, or neighborhoods. What sounds interesting?",
            f"{time_greeting} Istanbul has so much to offer! {interest_text}Would you like suggestions for food, sightseeing, or getting around the city?",
            f"Welcome to your Istanbul guide! {style_suggestion}. What would you like to discover today?"
        ]
        
        import random
        return random.choice(default_responses)
        
    except Exception as e:
        return "I'm here to help you explore Istanbul! What would you like to know about - restaurants, attractions, neighborhoods, or transportation?"

def process_transportation_query_enhanced(message: str, user_profile, current_time, context=None) -> str:
    """Enhanced transportation query processing with ML personalization"""
    try:
        # Get current hour for time-based recommendations
        hour = current_time.hour if hasattr(current_time, 'hour') else 12
        if hour < 11:
            time_period = 'morning'
        elif hour < 16:
            time_period = 'afternoon'
        elif hour < 20:
            time_period = 'evening'
        else:
            time_period = 'night'
        
        message_lower = message.lower()
        
        # Analyze transportation type
        if any(word in message_lower for word in ['metro', 'subway', 'underground']):
            transport_type = "metro"
        elif any(word in message_lower for word in ['bus', 'buses']):
            transport_type = "bus"
        elif any(word in message_lower for word in ['ferry', 'boat', 'bosphorus']):
            transport_type = "ferry"
        elif any(word in message_lower for word in ['taxi', 'uber', 'car']):
            transport_type = "taxi"
        elif any(word in message_lower for word in ['tram', 'tramway']):
            transport_type = "tram"
        else:
            transport_type = "general"
        
        # Get user preferences
        budget_conscious = hasattr(user_profile, 'travel_style') and user_profile.travel_style == 'budget'
        
        # Build response based on query type and user profile
        if transport_type == "metro":
            response = f"""🚇 **Istanbul Metro System** ({time_period}):

**Main Lines:**
• **M1 (Red)**: Airport ↔ Kirazlı (connects to M3)
• **M2 (Green)**: Veliefendi ↔ Hacıosman (covers Taksim, Şişli)
• **M3 (Blue)**: Kirazlı ↔ Başakşehir 
• **M4 (Pink)**: Kadıköy ↔ Sabiha Gökçen Airport
• **M7 (Purple)**: Kabataş ↔ Mecidiyeköy

**💡 Tips:**
• Use **Istanbulkart** for all public transport
• Metro runs 6:00-24:00 (extended on weekends)
• Clean, safe, and air-conditioned"""
            
            if budget_conscious:
                response += "\n• Very economical - much cheaper than taxis!"
                
        elif transport_type == "ferry":
            response = f"""⛴️ **Bosphorus Ferries** ({time_period}):

**Popular Routes:**
• **Eminönü ↔ Üsküdar**: Historic peninsula to Asian side
• **Kabataş ↔ Kadıköy**: European to Asian side
• **Bosphorus Tour**: Full strait tour with amazing views

**💡 Perfect for:**
• Scenic transportation between continents
• Sunset views (especially {time_period})
• Avoiding traffic congestion"""

        elif transport_type == "taxi":
            response = f"""🚖 **Taxis & Ride-sharing** ({time_period}):

**Options:**
• **Yellow Taxis**: Everywhere, use meter
• **Uber**: Available in most areas
• **BiTaksi**: Local ride-sharing app

**💡 Tips:**
• Always insist on using the meter
• Traffic can be heavy during {time_period}
• BiTaksi often cheaper than international apps"""
            
            if budget_conscious:
                response += "\n• Consider metro/ferry for longer distances to save money!"
                
        else:
            response = f"""🚌 **Istanbul Public Transport** ({time_period}):

**Best Options:**
• **Metro**: Fast, reliable, air-conditioned
• **Bus**: Extensive network, frequent services  
• **Ferry**: Scenic, avoids traffic
• **Tram**: Connects historic areas
• **Minibus (Dolmuş)**: Local transport, authentic experience

**💳 Payment:**
• **Istanbulkart** works for all public transport
• Buy at metro stations, kiosks, or online
• Transfers between different transport types get discounts"""
            
            if budget_conscious:
                response += "\n• Public transport is very budget-friendly!"
        
        # Add personalized recommendations based on user interests
        if hasattr(user_profile, 'interests'):
            if 'history' in user_profile.interests:
                response += "\n\n🏛️ **Historic Route Tip**: Take the tram from Kabataş to Sultanahmet for easy access to historical sites!"
            if 'culture' in user_profile.interests:
                response += "\n\n🎭 **Cultural Tip**: Ferry rides offer great views of both European and Asian cultures!"
        
        response += f"\n\nNeed specific route planning? Tell me where you're going!"
        
        return response
        
    except Exception as e:
        return "I can help you navigate Istanbul's transport system! Metro, bus, ferry, or taxi - what do you need to know?"
def format_attraction_response_text(attraction_response: dict, user_profile, current_time) -> str:
    """Format attraction system response with personalization"""
    try:
        attractions = attraction_response.get('attractions', [])
        if not attractions:
            return "I couldn't find specific attractions matching your request, but I'd be happy to suggest some popular places based on your interests!"
        
        # Build personalized response
        response_parts = []
        
        # Add greeting based on time
        hour = current_time.hour if hasattr(current_time, 'hour') else 12
        time_greeting = get_time_greeting(hour)
        response_parts.append(f"{time_greeting} Here are some wonderful attractions I think you'll love:")
        
        # Add attractions with personalization
        for i, attraction in enumerate(attractions[:3], 1):  # Limit to top 3
            name = attraction.get('name', 'Unknown')
            district = attraction.get('district', 'Istanbul')
            description = attraction.get('description', 'A wonderful place to visit')
            
            # Truncate description if too long
            if len(description) > 150:
                description = description[:147] + "..."
            
            response_parts.append(f"\n{i}. **{name}** ({district})")
            response_parts.append(f"   {description}")
            
            # Add personalized context if available
            if hasattr(user_profile, 'interests') and user_profile.interests:
                matching_interests = [interest for interest in user_profile.interests 
                                    if interest.lower() in description.lower()]
                if matching_interests:
                    response_parts.append(f"   💡 Perfect for your interest in {', '.join(matching_interests[:2])}")
        
        # Add helpful context
        response_parts.append(f"\n📍 These attractions are perfect for exploring Istanbul's rich history and culture!")
        
        # Add travel tip based on user profile
        if hasattr(user_profile, 'travel_style'):
            if user_profile.travel_style == 'luxury':
                response_parts.append("🌟 Pro tip: Consider booking guided tours for the premium experience!")
            elif user_profile.travel_style == 'budget':
                response_parts.append("💰 Pro tip: Many of these attractions offer student or group discounts!")
            elif user_profile.travel_style == 'family':
                response_parts.append("👨‍👩‍👧‍👦 Pro tip: These are all family-friendly with facilities for children!")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"I found some great attractions for you! While I process the details, would you like me to suggest some popular spots in the area you're interested in?"

def get_time_greeting(hour: int) -> str:
    """Get appropriate greeting based on time of day"""
    if 5 <= hour < 12:
        return "Good morning! 🌅"
    elif 12 <= hour < 17:
        return "Good afternoon! ☀️"
    elif 17 <= hour < 21:
        return "Good evening! 🌆"
    else:
        return "Hello there! 🌙"

# =============================
# MISSING ML HELPER FUNCTIONS
# =============================

def generate_personalization_reason(recommendation: dict, user_profile) -> str:
    """Generate human-readable reason for why this recommendation was personalized"""
    
    reasons = []
    
    # Interest-based reasons
    if hasattr(user_profile, 'interests') and user_profile.interests:
        rec_category = recommendation.get('category', '').lower()
        matching_interests = [interest for interest in user_profile.interests 
                            if interest.lower() in rec_category or rec_category in interest.lower()]
        if matching_interests:
            reasons.append(f"matches your interest in {', '.join(matching_interests)}")
    
    # Travel style reasons
    if hasattr(user_profile, 'travel_style') and user_profile.travel_style:
        if user_profile.travel_style == 'family' and recommendation.get('family_friendly', False):
            reasons.append("perfect for families")
        elif user_profile.travel_style == 'solo' and recommendation.get('solo_friendly', True):
            reasons.append("great for solo travelers")
        elif user_profile.travel_style == 'couple' and recommendation.get('romantic', False):
            reasons.append("romantic atmosphere")
    
    # Budget reasons
    if hasattr(user_profile, 'budget_range') and user_profile.budget_range:
        if recommendation.get('price_level', '').lower() == user_profile.budget_range.lower():
            reasons.append(f"fits your {user_profile.budget_range} budget")
    
    # Accessibility reasons
    if hasattr(user_profile, 'accessibility_needs') and user_profile.accessibility_needs:
        if recommendation.get('accessible', False):
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
    if hasattr(user_profile, 'interests') and user_profile.interests:
        rec_category = recommendation.get('category', '').lower()
        matching_interests = [interest for interest in user_profile.interests 
                            if interest.lower() in rec_category or rec_category in interest.lower()]
        if matching_interests:
            reasons.append(f"Matches your {', '.join(matching_interests)} interests")
    
    # Budget alignment
    if hasattr(user_profile, 'budget_range') and user_profile.budget_range:
        if recommendation.get('price_level', '').lower() == user_profile.budget_range.lower():
            reasons.append(f"Fits your {user_profile.budget_range} budget")
    
    # Travel style
    if hasattr(user_profile, 'travel_style') and user_profile.travel_style:
        if user_profile.travel_style == 'family' and recommendation.get('family_friendly', False):
            reasons.append("Family-friendly")
        elif user_profile.travel_style == 'couple' and recommendation.get('romantic', False):
            reasons.append("Perfect for couples")
    
    if not reasons:
        return "Based on your general preferences"
    
    return ". ".join(reasons)

def apply_behavioral_patterns(recommendation: dict, user_profile) -> float:
    """Apply learned behavioral patterns to recommendation scoring"""
    from datetime import datetime
    
    pattern_score = 0.5  # Base score
    
    # Analyze past feedback
    if hasattr(user_profile, 'recommendation_feedback') and user_profile.recommendation_feedback:
        similar_recs = find_similar_recommendations(recommendation, user_profile.recommendation_feedback)
        if similar_recs:
            avg_feedback = sum(similar_recs.values()) / len(similar_recs)
            pattern_score += (avg_feedback - 0.5) * 0.3  # Adjust based on past feedback
    
    # Frequency patterns
    rec_location = recommendation.get('location', '').lower()
    if hasattr(user_profile, 'visit_frequency') and user_profile.visit_frequency and rec_location in user_profile.visit_frequency:
        visit_count = user_profile.visit_frequency[rec_location]
        frequency_bonus = min(visit_count * 0.05, 0.2) - (visit_count * 0.01)  # Diminishing returns
        pattern_score += frequency_bonus
    
    # Temporal patterns
    current_hour = datetime.now().hour
    if hasattr(user_profile, 'preferred_times') and user_profile.preferred_times:
        time_period = get_time_period_simple(current_hour)
        if time_period in user_profile.preferred_times:
            pattern_score += 0.1
    
    return min(max(pattern_score, 0.0), 1.0)

def get_time_period_simple(hour: int) -> str:
    """Get time period from hour"""
    if hour < 11:
        return 'morning'
    elif hour < 16:
        return 'afternoon'
    elif hour < 20:
        return 'evening'
    else:
        return 'night'

def get_meal_context(hour: int) -> str:
    """Determine meal context based on hour"""
    if hour < 11:
        return "breakfast"
    elif hour < 16:
        return "lunch"
    else:
        return "dinner"

def update_learning_patterns(user_profile, recommendations: list):
    """Update ML learning patterns based on generated recommendations"""
    from datetime import datetime
    
    # Initialize learning patterns if not present
    if not hasattr(user_profile, 'learning_patterns'):
        user_profile.learning_patterns = {}
    
    # Update adaptation weights based on recommendation success
    current_patterns = user_profile.learning_patterns.get('recommendation_patterns', {})
    
    # Track recommendation types generated
    rec_types = [rec.get('category', 'general') for rec in recommendations]
    for rec_type in rec_types:
        current_patterns[rec_type] = current_patterns.get(rec_type, 0) + 1
    
    # Update learning patterns
    user_profile.learning_patterns['recommendation_patterns'] = current_patterns
    user_profile.learning_patterns['last_update'] = datetime.now().isoformat()
    user_profile.learning_patterns['total_recommendations'] = user_profile.learning_patterns.get('total_recommendations', 0) + len(recommendations)

def get_adaptation_factors(recommendation: dict, user_profile, context) -> dict:
    """Get detailed breakdown of adaptation factors"""
    
    factors = {
        'interest_match': 0.0,
        'budget_alignment': 0.0,
        'travel_style_fit': 0.0,
        'accessibility_score': 0.0,
        'behavioral_pattern': 0.0,
        'temporal_relevance': 0.0,
        'location_preference': 0.0
    }
    
    # Calculate each factor
    if hasattr(user_profile, 'interests') and user_profile.interests:
        rec_category = recommendation.get('category', '').lower()
        factors['interest_match'] = 0.8 if any(interest.lower() in rec_category 
                                             for interest in user_profile.interests) else 0.2
    
    if hasattr(user_profile, 'budget_range') and user_profile.budget_range:
        if recommendation.get('price_level', '').lower() == user_profile.budget_range.lower():
            factors['budget_alignment'] = 0.9
    
    # Travel style fit
    if hasattr(user_profile, 'travel_style') and user_profile.travel_style:
        if user_profile.travel_style == 'family' and recommendation.get('family_friendly', False):
            factors['travel_style_fit'] = 0.9
        elif user_profile.travel_style == 'couple' and recommendation.get('romantic', False):
            factors['travel_style_fit'] = 0.9
        else:
            factors['travel_style_fit'] = 0.5
    
    return factors

def apply_diversity_filter(recommendations: list, user_profile) -> list:
    """Apply diversity filtering to avoid monotonous recommendations"""
    
    if len(recommendations) <= 3:
        return recommendations
    
    diverse_recommendations = []
    used_categories = set()
    used_locations = set()
    
    # First pass: Select diverse recommendations
    for rec in recommendations:
        category = rec.get('category', 'general')
        location = rec.get('location', 'unknown')
        
        # Add if category and location are not overrepresented
        if (len([r for r in diverse_recommendations if r.get('category') == category]) < 2 and
            len([r for r in diverse_recommendations if r.get('location') == location]) < 3):
            diverse_recommendations.append(rec)
            used_categories.add(category)
            used_locations.add(location)
    
    # Second pass: Fill remaining slots with highest scoring items
    for rec in recommendations:
        if len(diverse_recommendations) >= 8:
            break
        if rec not in diverse_recommendations:
            diverse_recommendations.append(rec)
    
    return diverse_recommendations

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
    
    return True

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
    if not hasattr(user_profile, 'interaction_history'):
        user_profile.interaction_history = []
    
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
        if not hasattr(user_profile, 'visit_frequency'):
            user_profile.visit_frequency = {}
        for neighborhood in entities['neighborhoods']:
            user_profile.visit_frequency[neighborhood] = user_profile.visit_frequency.get(neighborhood, 0) + 1
    
    # Update interests based on intent
    if not hasattr(user_profile, 'interests'):
        user_profile.interests = []
    
    if intent == 'restaurant_query' and 'food' not in user_profile.interests:
        user_profile.interests.append('food')
    elif intent == 'transportation_query' and 'transportation' not in user_profile.interests:
        user_profile.interests.append('transportation')
    elif intent == 'attraction_query' and 'sightseeing' not in user_profile.interests:
        user_profile.interests.append('sightseeing')
    
    # Update dietary restrictions if mentioned
    message_lower = message.lower()
    if 'vegetarian' in message_lower and 'vegetarian' not in user_profile.dietary_restrictions:
        user_profile.dietary_restrictions.append('vegetarian')
    if 'vegan' in message_lower and 'vegan' not in user_profile.dietary_restrictions:
        user_profile.dietary_restrictions.append('vegan')
    if 'halal' in message_lower and 'halal' not in user_profile.dietary_restrictions:
        user_profile.dietary_restrictions.append('halal')
    
    # Update budget range if mentioned
    budget_keywords = {
        'budget': 'budget',
        'luxury': 'luxury',
        'mid-range': 'mid-range',
        'affordable': 'budget',
        'expensive': 'luxury',
        'premium': 'luxury',
        'moderate': 'mid-range',
        'economical': 'budget'
    }
    
    for word, budget in budget_keywords.items():
        if word in message_lower:
            user_profile.budget_range = budget
            break

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

def get_recommendation_explanation(user_profiles: dict, recommendation_id: str, user_id: str) -> dict:
    """Generate detailed explanation for why a specific recommendation was made"""
    
    if user_id not in user_profiles:
        return {'error': 'User profile not found'}
    
    user_profile = user_profiles[user_id]
    
    # Mock explanation for now
    explanation = {
        'recommendation_id': recommendation_id,
        'explanation_summary': "This recommendation was made based on your preferences",
        'user_profile_factors': {
            'interests_match': 0.8,
            'travel_style_alignment': 0.7,
            'budget_compatibility': 0.9
        },
        'confidence_level': 'high'
    }
    
    return explanation

def find_similar_recommendations(recommendation: dict, recommendation_feedback: dict) -> dict:
    """Find similar recommendations in feedback history"""
    similar_recs = {}
    rec_category = recommendation.get('category', '').lower()
    rec_location = recommendation.get('location', '').lower()
    
    for rec_id, rating in recommendation_feedback.items():
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

def show_privacy_settings(user_profile, user_id: str) -> str:
    """Show current privacy settings and available controls"""
    return f"""🔒 **Your Privacy Settings**

**Current Status:**
• Location sharing: ❌ Disabled
• Profile personalization: ✅ Enabled
• Recommendation history: ✅ Stored locally
• Learning patterns: ✅ Active

**Available Controls:**
• 'disable personalization' - Turn off profile learning
• 'clear my data' - Remove all stored information
• 'show my data' - View all stored information
• 'explain data usage' - Learn how your data is used

**Data Policy:**
• Your data stays on secure servers
• Never shared with third parties
• You control what's collected
• Delete anytime with 'clear my data'

Your privacy matters! 🔒"""

def show_user_data(user_profile, user_id: str) -> str:
    """Show all data stored about the user"""
    interaction_count = len(getattr(user_profile, 'interaction_history', []))
    feedback_count = len(getattr(user_profile, 'recommendation_feedback', {}))
    
    return f"""📊 **Your Data Summary**

**Profile Information:**
• User ID: {user_id}
• Interests: {', '.join(getattr(user_profile, 'interests', [])) or 'None specified'}
• Travel style: {getattr(user_profile, 'travel_style', 'Not specified')}
• Budget preference: {getattr(user_profile, 'budget_range', 'Not specified')}

**Activity Data:**
• Total interactions: {interaction_count}
• Recommendation ratings: {feedback_count}
• Profile completeness: {getattr(user_profile, 'profile_completeness', 0.3):.1%}

**Data Usage:**
• Used only for personalizing your experience
• Never shared with third parties
• Stored locally on secure servers
• You can delete anytime with 'clear my data'

Need to update anything? Just tell me your preferences!"""

def clear_user_data(user_profiles: dict, active_conversations: dict, user_id: str) -> str:
    """Clear all user data"""
    if user_id in user_profiles:
        del user_profiles[user_id]
    
    # Clear active conversations for this user
    sessions_to_remove = [session_id for session_id, context in active_conversations.items() 
                         if hasattr(context, 'user_profile') and context.user_profile.user_id == user_id]
    
    for session_id in sessions_to_remove:
        del active_conversations[session_id]
    
    return f"""✅ **Data Cleared Successfully**

All your data has been permanently deleted:
• User profile and preferences ❌
• Interaction history ❌
• Recommendation feedback ❌
• Conversation context ❌
• Learning patterns ❌

**Fresh Start Ready!**
You can start using the system normally - I'll only remember what you tell me in new conversations.

Your privacy is fully protected! 🔒"""
