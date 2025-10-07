"""
Enhanced Response Templates for AI Istanbul Travel Assistant
Ultra-specialized local guide style responses that compete with GPT
"""

import re
import random
from datetime import datetime
from typing import Dict, List, Optional, Any

class IstanbulResponseTemplates:
    """Enhanced response templates for natural, local guide-style interactions"""
    
    def __init__(self):
        self.weather_contexts = {
            "sunny": ["Since it's sunny today", "Perfect weather for", "Great day to explore"],
            "rainy": ["With the rain today", "Indoor spots are perfect when", "Cozy weather for"],
            "cloudy": ["Nice overcast day for", "Perfect cloudy weather for", "Great atmospheric day for"]
        }
        
        self.time_contexts = {
            "morning": ["This morning", "Early hours are perfect for", "Start your day at"],
            "afternoon": ["This afternoon", "Perfect time to visit", "Great afternoon spot"],
            "evening": ["As evening approaches", "Perfect sunset timing for", "Evening magic at"]
        }
        
        self.district_personalities = {
            "sultanahmet": {
                "character": "the heart of historic Istanbul",
                "vibe": "ancient stones meeting modern life",
                "best_time": "early morning before crowds arrive",
                "transport": "Sultanahmet tram station or a short walk from Eminönü ferry"
            },
            "beyoglu": {
                "character": "Istanbul's vibrant cultural hub",
                "vibe": "art galleries, rooftop bars, and endless energy",
                "best_time": "late afternoon into the night",
                "transport": "metro to Şişhane or historic tünel funicular"
            },
            "kadikoy": {
                "character": "the creative soul of the Asian side",
                "vibe": "bohemian cafés, street art, and local authenticity",
                "best_time": "weekend mornings for the fish market",
                "transport": "ferry from Eminönü - the scenic route locals love"
            },
            "besiktas": {
                "character": "where Ottoman elegance meets Bosphorus beauty",
                "vibe": "waterfront palaces and passionate football culture",
                "best_time": "sunset for the best Bosphorus views",
                "transport": "ferry or metro - both offer great approaches"
            }
        }

    def format_restaurant_response(self, restaurants: List[Dict], location: str, query_context: str = "") -> str:
        """Format restaurant recommendations in conversational style"""
        if not restaurants:
            return self._get_no_restaurants_fallback(location, query_context)
        
        # Limit to 5 restaurants maximum
        restaurants = restaurants[:5]
        count = len(restaurants)
        
        # Context-aware opening
        opening = self._get_restaurant_opening(location, count, query_context)
        
        # Format each restaurant
        formatted_restaurants = []
        for i, restaurant in enumerate(restaurants):
            formatted = self._format_single_restaurant(restaurant, i == 0)
            formatted_restaurants.append(formatted)
        
        # Join with natural connectors
        restaurant_text = self._join_restaurants_naturally(formatted_restaurants)
        
        # Add contextual closing
        closing = self._get_restaurant_closing(location, query_context)
        
        return f"{opening}\n\n{restaurant_text}\n\n{closing}"

    def _get_restaurant_opening(self, location: str, count: int, query_context: str) -> str:
        """Generate context-aware opening for restaurant recommendations"""
        location_clean = location.replace("District", "").replace("district", "").strip()
        
        # Context-specific openings
        if "expensive" in query_context.lower() or "luxury" in query_context.lower():
            return f"Here are {count} premium dining spots in {location_clean}:"
        elif "budget" in query_context.lower() or "cheap" in query_context.lower():
            return f"Here are {count} budget-friendly restaurants in {location_clean}:"
        elif "coffee" in query_context.lower():
            return f"Here are {count} great coffee spots in {location_clean}:"
        elif "vegan" in query_context.lower() or "vegetarian" in query_context.lower():
            return f"Here are {count} restaurants with excellent vegan options in {location_clean}:"
        elif any(cuisine in query_context.lower() for cuisine in ["turkish", "seafood", "italian", "french"]):
            cuisine = next(c for c in ["turkish", "seafood", "italian", "french"] if c in query_context.lower())
            return f"Here are {count} excellent {cuisine} restaurants in {location_clean}:"
        else:
            return f"Here are {count} restaurants in {location_clean}:"

    def _format_single_restaurant(self, restaurant: Dict, is_first: bool) -> str:
        """Format a single restaurant with natural language"""
        name = restaurant.get('name', 'Unknown Restaurant')
        cuisine = restaurant.get('cuisine', 'International')
        budget = restaurant.get('budget', 'mid-range')
        rating = restaurant.get('rating', 4.0)
        description = restaurant.get('description', 'Great local dining spot')
        
        # Price context
        price_context = {
            'budget': 'wallet-friendly',
            'mid-range': 'reasonably priced',
            'premium': 'upscale',
            'luxury': 'high-end'
        }.get(budget, 'good value')
        
        # Rating context
        rating_text = ""
        if rating >= 4.3:
            rating_text = " (locals absolutely love this place)"
        elif rating >= 4.0:
            rating_text = " (consistently good reviews)"
        
        # Natural connector
        connector = "**" if is_first else "• **"
        
        return f"{connector}{name}** — {description} {price_context} {cuisine.lower()} spot{rating_text}."

    def _join_restaurants_naturally(self, restaurants: List[str]) -> str:
        """Join restaurant descriptions with natural flow"""
        if len(restaurants) == 1:
            return restaurants[0]
        
        # Add variety in connectors
        connectors = [
            "\n\n", "\n\nAlso worth trying: ", "\n\nDon't miss: ", 
            "\n\nLocals recommend: ", "\n\nAnother great choice: "
        ]
        
        result = restaurants[0]
        for i, restaurant in enumerate(restaurants[1:], 1):
            if i < len(connectors):
                result += connectors[i] + restaurant[2:]  # Remove bullet point
            else:
                result += "\n\n" + restaurant
        
        return result

    def _get_restaurant_closing(self, location: str, query_context: str) -> str:
        """Generate contextual closing for restaurant recommendations"""
        closings = [
            "Want specific directions to any of these spots?",
            "Need more details about opening hours or reservations?",
            "Looking for something else in the area?",
            f"There are also some great cafés nearby if you want to explore more of {location}."
        ]
        return random.choice(closings)

    def _get_no_restaurants_fallback(self, location: str, query_context: str) -> str:
        """Fallback when no restaurants found"""
        alternatives = {
            "sultanahmet": "Beyoğlu has amazing dining scenes",
            "beyoglu": "Kadıköy's food scene is incredible", 
            "kadikoy": "Beşiktaş has some hidden gems",
            "besiktas": "Sultanahmet has traditional options"
        }
        
        alt_location = alternatives.get(location.lower(), "nearby districts have great options")
        return f"Hmm, I don't have specific recommendations for that area right now, but {alt_location}. Want me to check those instead?"

    def format_attraction_response(self, attractions: List[Dict], location: str) -> str:
        """Format attraction recommendations in local guide style"""
        if not attractions:
            return f"That area's still being mapped, but nearby districts have amazing spots — want to explore those instead?"
        
        main_attraction = attractions[0]
        name = main_attraction.get('name', 'Unknown Attraction')
        description = main_attraction.get('description', 'Great place to visit')
        
        # Get current time context
        current_hour = datetime.now().hour
        time_context = "morning" if current_hour < 12 else "afternoon" if current_hour < 18 else "evening"
        
        response = f"You shouldn't miss **{name}** — {description}. "
        
        # Add time/weather context
        if time_context == "morning":
            response += f"Perfect for a {time_context} visit, and usually takes about an hour to explore. "
        else:
            response += f"Great for {time_context} exploration, giving you about an hour of discovery. "
        
        # Add nearby suggestion if available
        if len(attractions) > 1:
            nearby = attractions[1]
            response += f"If you have time, nearby you can also stop by **{nearby.get('name', 'another great spot')}**."
        
        return response

    def format_district_guide(self, district: str, attractions: List[Dict] = None) -> str:
        """Format comprehensive district guide"""
        district_lower = district.lower().replace("district", "").strip()
        
        if district_lower in self.district_personalities:
            info = self.district_personalities[district_lower]
            
            response = f"**{district.title()}** is {info['character']} — think {info['vibe']}. "
            response += f"It's best to visit around {info['best_time']}. "
            
            if attractions and len(attractions) >= 2:
                highlight = attractions[0].get('name', 'amazing spots')
                hidden = attractions[1].get('name', 'local favorites')
                response += f"You'll find {highlight}, plus hidden gems like {hidden}. "
            else:
                response += "You'll find incredible historic sites, plus plenty of hidden local gems. "
            
            response += f"It's easy to reach via {info['transport']}. "
            response += "Want me to build you a short walking route?"
            
            return response
        
        # Generic fallback
        return f"**{district}** has its own unique character worth exploring. The best way to discover it is wandering the streets and chatting with locals. Want specific attraction recommendations for the area?"

    def format_transportation_response(self, from_location: str, to_location: str, route_info: Dict = None) -> str:
        """Format transportation guidance in confident, helpful style"""
        if not route_info:
            # Generic helpful response
            return f"The easiest route from **{from_location}** to **{to_location}** is typically by metro or ferry, depending on which side of the city you're crossing. Want me to check current transport options and timing for you?"
        
        transport_mode = route_info.get('mode', 'metro')
        travel_time = route_info.get('time', '20-30 minutes')
        
        response = f"The easiest route from **{from_location}** to **{to_location}** is by **{transport_mode}**. "
        
        if 'line' in route_info:
            response += f"Take {route_info['line']} towards {route_info.get('direction', 'your destination')}, "
            if 'transfer' in route_info:
                response += f"then {route_info['transfer']}. "
            else:
                response += "direct route. "
        
        response += f"It usually takes around {travel_time}. "
        
        # Add alternative if available
        if 'alternative' in route_info:
            response += f"If you'd rather avoid crowds, you can also try {route_info['alternative']}."
        
        return response

    def format_daily_plan(self, plan_data: Dict, user_preferences: Dict = None) -> str:
        """Format daily itinerary in structured but friendly style"""
        response = "Here's a relaxed plan for today:\n\n"
        
        # Morning
        morning = plan_data.get('morning', {})
        if morning:
            response += f"• **Morning** — {morning.get('place', 'Start exploring')}: {morning.get('note', 'Great way to begin the day')}\n\n"
        
        # Lunch
        lunch = plan_data.get('lunch', {})
        if lunch:
            response += f"• **Lunch** — {lunch.get('restaurant', 'Local favorite')}: {lunch.get('cuisine_note', 'Authentic local cuisine')}\n\n"
        
        # Afternoon
        afternoon = plan_data.get('afternoon', {})
        if afternoon:
            response += f"• **Afternoon** — {afternoon.get('place', 'Continue exploring')}: {afternoon.get('activity_note', 'Perfect afternoon activity')}\n\n"
        
        # Evening
        evening = plan_data.get('evening', {})
        if evening:
            response += f"• **Evening** — {evening.get('place', 'End the day')}: {evening.get('view_or_dining_note', 'Beautiful way to end the day')}\n"
        
        # Contextual follow-up
        preferences = []
        if user_preferences:
            if user_preferences.get('budget'): preferences.append(f"budget of {user_preferences['budget']}")
            if user_preferences.get('weather'): preferences.append(f"today's {user_preferences['weather']} weather")
            if user_preferences.get('group_type'): preferences.append(f"{user_preferences['group_type']} travel")
        
        if preferences:
            response += f"\nWould you like me to adjust it for {' and '.join(preferences)}?"
        else:
            response += f"\nWould you like me to adjust the timing or add specific neighborhoods you're interested in?"
        
        return response

    def add_context_memory(self, user_query: str, response: str, context_data: Dict) -> str:
        """Add contextual memory references to responses"""
        # Extract previous interests from context
        previous_districts = context_data.get('visited_districts', [])
        previous_cuisines = context_data.get('liked_cuisines', [])
        previous_attractions = context_data.get('visited_attractions', [])
        
        # Add natural references if relevant
        memory_additions = []
        
        if previous_districts and len(previous_districts) > 0:
            last_district = previous_districts[-1]
            if last_district.lower() not in user_query.lower():
                memory_additions.append(f"Since you enjoyed exploring {last_district}")
        
        if previous_cuisines and len(previous_cuisines) > 0:
            if any(cuisine in user_query.lower() for cuisine in previous_cuisines):
                memory_additions.append("Building on your taste preferences")
        
        # Add memory context naturally
        if memory_additions and not any(phrase in response for phrase in ["Since you", "Building on"]):
            memory_intro = random.choice(memory_additions)
            # Insert naturally into response
            if "Here are" in response:
                response = response.replace("Here are", f"{memory_intro}, here are", 1)
        
        return response

    def get_fallback_response(self, query_type: str, location: str = "") -> str:
        """Generate appropriate fallback responses"""
        fallbacks = {
            'restaurants': f"Looks like the dining scene in {location} is still being updated. Would you like me to suggest some amazing spots in nearby districts instead?",
            'attractions': f"Most outdoor spots might be crowded today. Would you like indoor attractions instead?",
            'transportation': f"Public transport's a bit limited between those areas right now — want me to suggest the fastest taxi or ferry option instead?",
            'general': "I'm still learning about that area, but I'd love to help you discover something else nearby!"
        }
        
        return fallbacks.get(query_type, fallbacks['general'])

# Global instance for easy import
istanbul_templates = IstanbulResponseTemplates()

def apply_enhanced_formatting(response: str, query: str, context: Dict = None) -> str:
    """Apply enhanced formatting to any response"""
    if not response or not query:
        return response
    
    # Check if this is a restaurant query
    restaurant_keywords = ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'meal', 'lunch', 'dinner', 'breakfast']
    if any(keyword in query.lower() for keyword in restaurant_keywords):
        # Ensure proper restaurant response format
        if not response.startswith("Here are") and "restaurant" in response.lower():
            # Try to extract location and count
            import re
            location_match = re.search(r'in (\w+)', query, re.IGNORECASE)
            location = location_match.group(1) if location_match else "the area"
            
            # Count restaurants mentioned
            restaurant_count = response.lower().count('restaurant') + response.count('**')
            restaurant_count = min(restaurant_count, 5)  # Cap at 5
            
            if restaurant_count > 0:
                response = f"Here are {restaurant_count} restaurants in {location}:\n\n{response}"
    
    # Add natural transitions and local guide personality
    response = response.replace(" - ", " — ")  # Use em dashes
    response = response.replace("Rating:", "Locals rate this:")
    response = response.replace("Price level:", "Budget-wise:")
    
    # Add conversational elements if missing
    if context and context.get('session_turns', 0) > 1:
        conversation_starters = ["Also,", "By the way,", "Another thing —"]
        if not any(starter in response for starter in conversation_starters):
            response = "Also, " + response if random.random() < 0.3 else response
    
    return response
