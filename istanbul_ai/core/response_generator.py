"""
Response Generator
Enhanced response generation with comprehensive recommendations and contextual awareness.
"""

import random
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Add backend services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))
try:
    from services.location_coordinates import get_location_coordinates_service
    LOCATION_COORDS_AVAILABLE = True
except ImportError:
    LOCATION_COORDS_AVAILABLE = False
    print("⚠️ Location coordinates service not available")

# Import transportation directions service
try:
    from services.transportation_directions_service import get_transportation_service
    TRANSPORTATION_SERVICE_AVAILABLE = True
except ImportError:
    TRANSPORTATION_SERVICE_AVAILABLE = False
    print("⚠️ Transportation directions service not available")

from ..core.models import UserProfile, ConversationContext


class ResponseGenerator:
    """Enhanced response generator with comprehensive Istanbul recommendations"""
    
    def __init__(self):
        self.initialize_response_templates()
        # Initialize location coordinates service
        if LOCATION_COORDS_AVAILABLE:
            self.location_service = get_location_coordinates_service()
        else:
            self.location_service = None
        
        # Initialize transportation service
        if TRANSPORTATION_SERVICE_AVAILABLE:
            self.transport_service = get_transportation_service()
        else:
            self.transport_service = None
    
    def initialize_response_templates(self):
        """Initialize comprehensive response templates"""
        
        self.response_templates = {
            'greeting': [
                "🌟 Merhaba! Welcome to Istanbul, the city where two continents meet! I'm your local AI guide, ready to help you discover the magic of this incredible city. What brings you to Istanbul today?",
                "🎯 Hello! I'm your Istanbul AI assistant, here to unlock the secrets of this magnificent city. Whether you're seeking historic wonders, culinary delights, or hidden gems, I'm here to guide you. What would you like to explore?",
                "🏰 Welcome to Istanbul! As your personal AI guide, I'm excited to help you experience this city's rich tapestry of culture, history, and flavors. Tell me, what's calling to your adventurous spirit today?"
            ],
            
            'restaurant_intro': [
                "🍽️ Istanbul's culinary scene is absolutely incredible! From traditional Ottoman cuisine to modern fusion, the city offers flavors that will transform your understanding of Turkish food.",
                "🥘 Let me guide you through Istanbul's amazing food culture! This city is where East meets West on your plate, creating unforgettable dining experiences.",
                "🍴 Food is the heart of Istanbul culture! I'm excited to recommend places that will give you authentic tastes and memorable experiences."
            ],
            
            'location_intro': [
                "📍 Istanbul is a city of incredible diversity! Each neighborhood has its own character, from the historic peninsula to the trendy European side.",
                "🗺️ This magnificent city spans two continents and offers countless experiences! Let me help you navigate its wonders.",
                "🌉 Istanbul's location is truly unique - where Europe meets Asia, history meets modernity, and tradition meets innovation."
            ]
        }
        
        # Enhanced attraction templates with comprehensive information
        self.attraction_templates = {
            'historic': {
                'intro': "🏛️ Istanbul's history spans over 2,500 years, with layers of Byzantine, Roman, and Ottoman heritage:",
                'details': [
                    "Hagia Sophia - Marvel at this architectural wonder that served as a church, mosque, and now museum. The golden mosaics and soaring dome represent 1,500 years of history.",
                    "Topkapi Palace - Explore the opulent former residence of Ottoman sultans, with stunning views over the Bosphorus and priceless imperial collections.",
                    "Blue Mosque - Admire the six minarets and blue Iznik tiles of this active place of worship, a masterpiece of Ottoman architecture.",
                    "Basilica Cistern - Descend into this mystical underground marvel with 336 columns, featured in countless films and legends."
                ]
            },
            'cultural': {
                'intro': "🎭 Istanbul's cultural richness reflects its position as a bridge between worlds:",
                'details': [
                    "Grand Bazaar - Navigate through 4,000 shops in this historic covered market, perfect for authentic Turkish carpets, ceramics, and spices.",
                    "Galata Tower - Enjoy panoramic 360° views of the city from this medieval Genoese tower, especially magical at sunset.",
                    "Turkish and Islamic Arts Museum - Discover world-class collections of calligraphy, ceramics, and the famous carpet collection.",
                    "Dolmabahce Palace - Experience 19th-century Ottoman luxury in this European-style palace on the Bosphorus."
                ]
            }
        }
    
    def generate_comprehensive_recommendation(self, recommendation_type: str, entities: Dict, 
                                           user_profile: UserProfile, context: ConversationContext,
                                           return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate comprehensive 150-300 word recommendations with practical information
        
        Args:
            recommendation_type: Type of recommendation (restaurant, attraction, neighborhood, transportation)
            entities: Extracted entities from user query
            user_profile: User profile information
            context: Conversation context
            return_structured: If True, return dict with 'response' and 'map_data'; if False, return string
            
        Returns:
            If return_structured=False: String response (backward compatible)
            If return_structured=True: Dict with 'response', 'map_data', 'recommendation_type'
        """
        
        current_time = datetime.now()
        hour = current_time.hour
        
        # Get weather-appropriate suggestions
        weather_context = self._get_weather_context(current_time)
        
        if recommendation_type == 'restaurant':
            return self._generate_enhanced_restaurant_recommendation(entities, user_profile, context, current_time, return_structured)
        elif recommendation_type == 'attraction':
            return self._generate_enhanced_attraction_recommendation(entities, user_profile, context, current_time, return_structured)
        elif recommendation_type == 'neighborhood':
            return self._generate_enhanced_neighborhood_recommendation(entities, user_profile, context, current_time, return_structured)
        elif recommendation_type == 'transportation' or recommendation_type == 'route_planning':
            return self._generate_transportation_directions(entities, user_profile, context, return_structured)
        else:
            result = self._generate_fallback_response(context, user_profile)
            return {'response': result, 'map_data': None, 'recommendation_type': 'general'} if return_structured else result
    
    def _generate_enhanced_restaurant_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                                   context: ConversationContext, current_time: datetime,
                                                   return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate comprehensive restaurant recommendations with practical details
        
        Args:
            return_structured: If True, return dict with response and map_data
        """
        
        hour = current_time.hour
        meal_context = self._get_meal_context(hour)
        
        # Base recommendations
        recommendations = []
        
        # Traditional Turkish restaurants
        if 'turkish_traditional' in entities.get('cuisines', []) or not entities.get('cuisines'):
            recommendations.extend([
                {
                    'name': 'Pandeli',
                    'type': 'Traditional Ottoman',
                    'location': 'Eminönü (above Spice Bazaar)',
                    'specialty': 'Ottoman palace cuisine',
                    'price_range': 'Mid-range (150-300 TL per person)',
                    'hours': '12:00-17:00 (closed Sundays)',
                    'transport': 'Eminönü metro/tram station (5 min walk)',
                    'highlights': 'Historic 1901 building, ceramic tiles, traditional recipes',
                    'best_for': 'Cultural dining experience'
                },
                {
                    'name': 'Hünkar',
                    'type': 'Traditional Turkish',
                    'location': 'Nişantaşı, Fatih',
                    'specialty': 'Home-style Turkish cooking (ev yemeği)',
                    'price_range': 'Moderate (100-200 TL per person)',
                    'hours': '11:30-22:00 daily',
                    'transport': 'Osmanbey metro (10 min walk to Nişantaşı branch)',
                    'highlights': 'Family recipes since 1950, lamb dishes, traditional desserts',
                    'best_for': 'Authentic home cooking experience'
                }
            ])
        
        # Street food options
        if 'street_food' in entities.get('cuisines', []) or hour < 12:
            recommendations.extend([
                {
                    'name': 'Tarihi Eminönü Balık Ekmek',
                    'type': 'Street Food',
                    'location': 'Eminönü waterfront',
                    'specialty': 'Fresh fish sandwiches from boats',
                    'price_range': 'Budget (15-25 TL per sandwich)',
                    'hours': '09:00-23:00 daily',
                    'transport': 'Eminönü ferry terminal (directly at waterfront)',
                    'highlights': 'Caught daily, grilled on boats, Istanbul institution',
                    'best_for': 'Authentic local experience'
                }
            ])
        
        # Seafood restaurants
        if 'seafood' in entities.get('cuisines', []) or 'bosphorus' in entities.get('landmarks', []):
            recommendations.extend([
                {
                    'name': 'Balıkçı Sabahattin',
                    'type': 'Seafood',
                    'location': 'Sultanahmet',
                    'specialty': 'Fresh Bosphorus fish, meze',
                    'price_range': 'High-end (300-500 TL per person)',
                    'hours': '12:00-24:00 daily',
                    'transport': 'Sultanahmet tram station (8 min walk)',
                    'highlights': 'Historic Ottoman house, extensive wine list, live music',
                    'best_for': 'Special occasions, romantic dinners'
                }
            ])
        
        # Format comprehensive response
        response_parts = []
        
        # Personalized greeting
        if user_profile.travel_style == 'couple':
            response_parts.append(f"🍽️ Perfect dining spots for couples in Istanbul! Given the {meal_context.lower()}, here are my top recommendations:")
        elif user_profile.has_children:
            response_parts.append(f"👨‍👩‍👧‍👦 Family-friendly restaurants in Istanbul! Here are great spots for {meal_context.lower()} that welcome children:")
        else:
            response_parts.append(f"🥘 Incredible {meal_context.lower()} spots in Istanbul! Here's where locals and food lovers gather:")
        
        # Add recommendations with full details including coordinates
        for i, rec in enumerate(recommendations[:3], 1):
            # Get coordinates for the restaurant
            coords_data = None
            if self.location_service:
                coords_data = self.location_service.get_coordinates(rec['name'], 'restaurant')
            
            location_text = f"📍 Location: {rec['location']}"
            if coords_data:
                location_text += f" [GPS: {coords_data['lat']:.4f}, {coords_data['lng']:.4f}]"
            
            response_parts.append(f"""
{i}. {rec['name']} ({rec['type']})
{location_text}
🍴 Specialty: {rec['specialty']}
💰 Price range: {rec['price_range']}
🕐 Hours: {rec['hours']}
🚇 Transport: {rec['transport']}
✨ Why visit: {rec['highlights']}
👥 Best for: {rec['best_for']}""")
        
        # Add practical tips
        response_parts.append(f"""
💡 Practical Tips:
• Make reservations for dinner, especially weekends
• Try Turkish tea (çay) or coffee after meals
• Tipping: 10-15% is standard for good service
• Many restaurants offer English menus in tourist areas
• Ask for 'hesap' (heh-sahp) when you want the bill""")
        
        # Add weather-appropriate suggestion
        weather_context = self._get_weather_context(current_time)
        if weather_context:
            response_parts.append(f"🌤️ Weather note: {weather_context}")
        
        # Build final response text
        response_text = '\n'.join(response_parts)
        
        # Return structured response with map data if requested
        if return_structured:
            map_data = self._extract_location_data(recommendations, 'restaurant')
            return {
                'response': response_text,
                'map_data': map_data,
                'recommendation_type': 'restaurant'
            }
        else:
            return response_text
    
    def _generate_enhanced_attraction_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                                   context: ConversationContext, current_time: datetime,
                                                   return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate comprehensive attraction recommendations with practical details
        
        Args:
            return_structured: If True, return dict with response and map_data
        """
        
        hour = current_time.hour
        
        # Determine best attractions based on time and user preferences
        morning_attractions = ['hagia_sophia', 'topkapi_palace', 'blue_mosque']
        afternoon_attractions = ['grand_bazaar', 'galata_tower', 'basilica_cistern']
        evening_attractions = ['bosphorus_cruise', 'galata_tower', 'rooftop_bars']
        
        if hour < 12:
            recommended_attractions = morning_attractions
            time_context = "morning exploration"
        elif hour < 17:
            recommended_attractions = afternoon_attractions
            time_context = "afternoon discovery"
        else:
            recommended_attractions = evening_attractions
            time_context = "evening experience"
        
        # Build comprehensive attraction list
        attractions = [
            {
                'name': 'Hagia Sophia',
                'type': 'Historic Monument',
                'location': 'Sultanahmet',
                'visit_duration': '1-2 hours',
                'access': 'Free entry (respectful behavior required)',
                'hours': 'Open daily (prayer times may affect access)',
                'transport': 'Sultanahmet tram station (2 min walk)',
                'highlights': '1,500-year history, stunning dome, Byzantine mosaics',
                'best_time': 'Early morning (9-11 AM) or late afternoon',
                'photography': 'Allowed, but respectful of worshippers',
                'accessibility': 'Main floor accessible, upper galleries have stairs'
            },
            {
                'name': 'Topkapi Palace',
                'type': 'Palace Museum',
                'location': 'Sultanahmet',
                'visit_duration': '2-3 hours',
                'access': 'Museum entry required (check current rates)',
                'hours': '09:00-18:00 (closed Tuesdays in winter)',
                'transport': 'Sultanahmet tram station (5 min walk)',
                'highlights': 'Ottoman imperial treasures, Bosphorus views, sacred relics',
                'best_time': 'Morning (9-11 AM) to avoid crowds',
                'photography': 'Limited in some sections',
                'accessibility': 'Some areas have stairs and uneven surfaces'
            },
            {
                'name': 'Grand Bazaar',
                'type': 'Historic Market',
                'location': 'Beyazıt/Eminönü',
                'visit_duration': '1-3 hours',
                'access': 'Free entry',
                'hours': '09:00-19:00 (closed Sundays)',
                'transport': 'Beyazıt-Kapalıçarşı tram station (1 min walk)',
                'highlights': '4,000 shops, authentic Turkish crafts, historic architecture',
                'best_time': 'Morning for better prices, afternoon for atmosphere',
                'photography': 'Allowed in corridors, ask permission in shops',
                'accessibility': 'Ground level, but can be crowded'
            }
        ]
        
        # Format comprehensive response
        response_parts = []
        
        # Personalized intro
        if user_profile.interests:
            interests_text = ', '.join(user_profile.interests[:3])
            response_parts.append(f"🏛️ Perfect {time_context} for someone interested in {interests_text}! Here are Istanbul's must-see attractions:")
        else:
            response_parts.append(f"🌟 Incredible attractions for your {time_context} in Istanbul! Here's what I recommend:")
        
        # Add detailed attraction information
        for i, attraction in enumerate(attractions[:3], 1):
            response_parts.append(f"""
{i}. {attraction['name']} ({attraction['type']})
📍 Location: {attraction['location']}
⏰ Visit time: {attraction['visit_duration']}
🎫 Access: {attraction['access']}
🕐 Hours: {attraction['hours']}
🚇 Transport: {attraction['transport']}
✨ Highlights: {attraction['highlights']}
📸 Photography: {attraction['photography']}
♿ Accessibility: {attraction['accessibility']}
🌟 Best time: {attraction['best_time']}""")
        
        # Add practical visiting tips
        response_parts.append(f"""
💡 Essential Visiting Tips:
• Museum Pass: Consider Istanbul Museum Pass for multiple attractions (check current pricing)
• Dress code: Modest clothing for mosques (covering shoulders/knees)
• Prayer times: Some mosques close 30 min before prayers
• Crowds: Visit major attractions early morning or late afternoon
• Guided tours: Available in multiple languages at most sites
• Audio guides: Often available for self-guided exploration""")
        
        # Add transportation and route suggestions
        response_parts.append(f"""
🚇 Getting Around:
• Sultanahmet area: Most historic attractions within walking distance
• Istanbulkart: Essential transport card (check current rates)
• Tram T1: Connects Sultanahmet to Galata Bridge and beyond
• Metro/Tram combos: Efficient for crossing between districts""")
        
        # Build final response text
        response_text = '\n'.join(response_parts)
        
        # Return structured response with map data if requested
        if return_structured:
            map_data = self._extract_location_data(attractions, 'attraction')
            return {
                'response': response_text,
                'map_data': map_data,
                'recommendation_type': 'attraction'
            }
        else:
            return response_text
    
    def _generate_enhanced_neighborhood_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                                     context: ConversationContext, current_time: datetime,
                                                     return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate comprehensive neighborhood recommendations
        
        Args:
            return_structured: If True, return dict with response and map_data
        """
        
        neighborhoods = [
            {
                'name': 'Sultanahmet (Historic Peninsula)',
                'character': 'Historic heart of Byzantine and Ottoman Istanbul',
                'best_for': 'First-time visitors, history lovers, cultural exploration',
                'highlights': [
                    'Hagia Sophia, Blue Mosque, Topkapi Palace',
                    'Traditional restaurants and Ottoman cuisine',
                    'Historic hammams (Turkish baths)',
                    'Carpet and souvenir shops'
                ],
                'atmosphere': 'Tourist-friendly but authentic, mix of locals and visitors',
                'transport': 'Sultanahmet tram station, central to everything',
                'budget': 'Mid-range to high-end restaurants, free historic sites',
                'best_time': 'Early morning or late afternoon to avoid crowds'
            },
            {
                'name': 'Beyoğlu (European Side Modern)',
                'character': 'Trendy, artistic, nightlife hub with European flair',
                'best_for': 'Nightlife, shopping, contemporary culture, young travelers',
                'highlights': [
                    'Galata Tower and panoramic views',
                    'Istiklal Street pedestrian avenue',
                    'Trendy cafes, bars, and rooftop restaurants',
                    'Art galleries and vintage shops'
                ],
                'atmosphere': 'Cosmopolitan, energetic, mix of tourists and locals',
                'transport': 'Karaköy metro, historic tram on Istiklal',
                'budget': 'Wide range from budget street food to upscale dining',
                'best_time': 'Afternoon and evening for full experience'
            },
            {
                'name': 'Kadıköy (Asian Side Local)',
                'character': 'Authentic local life, hipster culture, foodie paradise',
                'best_for': 'Local experiences, food tours, avoiding tourist crowds',
                'highlights': [
                    'Bustling food market and street food',
                    'Moda neighborhood seaside walks',
                    'Local bars and live music venues',
                    'Vintage shopping and local boutiques'
                ],
                'atmosphere': 'Genuine local vibe, younger crowd, artistic community',
                'transport': 'Ferry from Eminönü (scenic 20-min ride)',
                'budget': 'Very affordable, authentic local prices',
                'best_time': 'Any time, but evenings are especially lively'
            }
        ]
        
        response_parts = []
        
        # Personalized intro based on user profile
        if user_profile.user_type.value == 'first_time_visitor':
            response_parts.append("🏘️ Perfect neighborhoods for first-time visitors! Each offers a unique slice of Istanbul's character:")
        elif user_profile.user_type.value == 'repeat_visitor':
            response_parts.append("🗺️ Time to explore beyond the obvious! Here are neighborhoods that reveal Istanbul's authentic personality:")
        else:
            response_parts.append("🌆 Istanbul's diverse neighborhoods each tell a different story. Here's where to experience the city's many faces:")
        
        # Add detailed neighborhood information
        for i, neighborhood in enumerate(neighborhoods, 1):
            response_parts.append(f"""
{i}. {neighborhood['name']}
🏘️ Character: {neighborhood['character']}
👥 Best for: {neighborhood['best_for']}
✨ Highlights:
{chr(10).join(f'   • {highlight}' for highlight in neighborhood['highlights'])}
🎭 Atmosphere: {neighborhood['atmosphere']}
🚇 Transport: {neighborhood['transport']}
💰 Budget: {neighborhood['budget']}
⏰ Best time: {neighborhood['best_time']}""")
        
        # Add comprehensive area guide
        response_parts.append(f"""
🗺️ Navigation Tips:
• Cross-Continental: Take ferries between European and Asian sides
• Historic walking: Sultanahmet to Galata Bridge is a beautiful walk
• Local transport: Each neighborhood has distinct transport connections
• District hopping: Plan 2-3 hours minimum per neighborhood

🎯 Choosing Your Base:
• History focus: Stay in Sultanahmet
• Nightlife/modern: Choose Beyoğlu/Galata
• Local experience: Consider Asian side (Kadıköy/Üsküdar)
• Luxury/views: Bosphorus-facing areas in Beşiktaş""")
        
        # Build final response text
        response_text = '\n'.join(response_parts)
        
        # Return structured response with map data if requested
        if return_structured:
            map_data = self._extract_location_data(neighborhoods, 'neighborhood')
            return {
                'response': response_text,
                'map_data': map_data,
                'recommendation_type': 'neighborhood'
            }
        else:
            return response_text
    
    def _generate_transportation_directions(
        self,
        entities: Dict,
        user_profile: UserProfile,
        context: ConversationContext,
        return_structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate detailed transportation directions (Google Maps style)
        
        Args:
            entities: Extracted entities (origin, destination, locations)
            user_profile: User profile
            context: Conversation context
            return_structured: If True, return dict with response and map_data
            
        Returns:
            Detailed transportation directions with route visualization
        """
        
        # Extract origin and destination from entities
        origin = entities.get('origin')
        destination = entities.get('destination')
        landmarks = entities.get('landmarks', [])
        
        # Try to resolve location names to coordinates
        origin_coords = None
        destination_coords = None
        origin_name = "your location"
        destination_name = "destination"
        
        if origin and self.location_service:
            coords_data = self.location_service.get_coordinates(origin, 'auto')
            if coords_data:
                origin_coords = (coords_data['lat'], coords_data['lng'])
                origin_name = origin
        
        if destination and self.location_service:
            coords_data = self.location_service.get_coordinates(destination, 'auto')
            if coords_data:
                destination_coords = (coords_data['lat'], coords_data['lng'])
                destination_name = destination
        
        # If no specific origin/destination but landmarks mentioned, find route between first two
        if not origin_coords and not destination_coords and len(landmarks) >= 2:
            if self.location_service:
                first_coords = self.location_service.get_coordinates(landmarks[0], 'auto')
                second_coords = self.location_service.get_coordinates(landmarks[1], 'auto')
                if first_coords and second_coords:
                    origin_coords = (first_coords['lat'], first_coords['lng'])
                    destination_coords = (second_coords['lat'], second_coords['lng'])
                    origin_name = landmarks[0]
                    destination_name = landmarks[1]
        
        # Default to popular routes if nothing specified
        if not origin_coords or not destination_coords:
            # Default example: Sultanahmet to Taksim
            origin_coords = (41.0054, 28.9768)  # Sultanahmet
            destination_coords = (41.0370, 28.9850)  # Taksim
            origin_name = "Sultanahmet"
            destination_name = "Taksim Square"
        
        # Get detailed directions
        route = None
        if self.transport_service:
            try:
                route = self.transport_service.get_directions(
                    start=origin_coords,
                    end=destination_coords,
                    start_name=origin_name,
                    end_name=destination_name
                )
            except Exception as e:
                print(f"⚠️ Transportation service error: {e}")
        
        # Build response
        response_parts = []
        
        # Header
        response_parts.append(f"🗺️ **Directions from {origin_name} to {destination_name}**\n")
        
        if route:
            # Format the detailed route
            response_parts.append(f"⏱️ **Total Time:** {route.total_duration} minutes")
            response_parts.append(f"📏 **Total Distance:** {route.total_distance/1000:.1f} km")
            
            modes_icons = {
                'walk': '🚶 Walking',
                'metro': '🚇 Metro',
                'tram': '🚊 Tram',
                'bus': '🚌 Bus',
                'ferry': '⛴️ Ferry'
            }
            modes_str = ', '.join([modes_icons.get(m, m) for m in set(route.modes_used)])
            response_parts.append(f"🚉 **Transport Modes:** {modes_str}\n")
            
            # Detailed steps
            response_parts.append("**Step-by-Step Directions:**\n")
            for i, step in enumerate(route.steps, 1):
                icon = {
                    'walk': '🚶',
                    'metro': '🚇',
                    'tram': '🚊',
                    'bus': '🚌',
                    'ferry': '⛴️'
                }.get(step.mode, '➡️')
                
                response_parts.append(f"**{i}. {icon} {step.instruction}**")
                response_parts.append(f"   • Distance: {step.distance/1000:.1f} km")
                response_parts.append(f"   • Duration: {step.duration} minutes")
                
                if step.line_name:
                    response_parts.append(f"   • Line: {step.line_name}")
                if step.stops_count:
                    response_parts.append(f"   • Stops: {step.stops_count}")
                response_parts.append("")
        else:
            # Fallback general directions
            response_parts.append(f"Here's how to travel between {origin_name} and {destination_name}:\n")
            response_parts.append("🚇 **Metro/Tram Option:**")
            response_parts.append("1. Walk to the nearest metro or tram station")
            response_parts.append("2. Take the appropriate line towards your destination")
            response_parts.append("3. Transfer if needed at major hubs")
            response_parts.append("4. Walk to your final destination\n")
            
            response_parts.append("💡 **Tips:**")
            response_parts.append("• Get an Istanbulkart for easy payment on all public transport")
            response_parts.append("• Metro frequency: Every 5-10 minutes during peak hours")
            response_parts.append("• Trams run frequently on major routes")
            response_parts.append("• Download the official IETT app for real-time schedules")
        
        # Add general Istanbul transportation info
        response_parts.append("\n📱 **Transportation Resources:**")
        response_parts.append("• **Istanbulkart:** Essential rechargeable transport card")
        response_parts.append("• **IETT App:** Real-time bus tracking and schedules")
        response_parts.append("• **Metro hours:** Approximately 6:00 AM - 12:00 AM")
        response_parts.append("• **Ferry schedules:** Check İDO or Şehir Hatları websites")
        response_parts.append("• **Taxi apps:** BiTaksi, iTaksi for reliable rides")
        
        response_text = '\n'.join(response_parts)
        
        # Build map data with route visualization
        if return_structured:
            map_data = self._build_route_map_data(route, origin_coords, destination_coords, origin_name, destination_name)
            return {
                'response': response_text,
                'map_data': map_data,
                'recommendation_type': 'transportation',
                'route_data': self._serialize_route(route) if route else None
            }
        else:
            return response_text
    
    def _build_route_map_data(
        self,
        route,
        origin_coords: Tuple[float, float],
        destination_coords: Tuple[float, float],
        origin_name: str,
        destination_name: str
    ) -> Dict[str, Any]:
        """Build map data for route visualization"""
        
        locations = []
        route_polyline = []
        
        # Add origin and destination markers
        locations.append({
            'lat': origin_coords[0],
            'lon': origin_coords[1],
            'name': origin_name,
            'type': 'origin',
            'metadata': {
                'description': 'Starting point',
                'icon': 'start'
            }
        })
        
        locations.append({
            'lat': destination_coords[0],
            'lon': destination_coords[1],
            'name': destination_name,
            'type': 'destination',
            'metadata': {
                'description': 'Destination',
                'icon': 'end'
            }
        })
        
        # Build route polyline from steps
        if route and route.steps:
            for step in route.steps:
                # Add waypoints if available
                if step.waypoints:
                    for wp in step.waypoints:
                        route_polyline.append({'lat': wp[0], 'lng': wp[1]})
                else:
                    # Add start and end of step
                    route_polyline.append({'lat': step.start_location[0], 'lng': step.start_location[1]})
                    route_polyline.append({'lat': step.end_location[0], 'lng': step.end_location[1]})
        
        # If no detailed route, add simple line
        if not route_polyline:
            route_polyline = [
                {'lat': origin_coords[0], 'lng': origin_coords[1]},
                {'lat': destination_coords[0], 'lng': destination_coords[1]}
            ]
        
        # Calculate bounds
        all_lats = [loc['lat'] for loc in locations] + [p['lat'] for p in route_polyline]
        all_lons = [loc['lon'] for loc in locations] + [p['lng'] for p in route_polyline]
        
        return {
            'locations': locations,
            'route_polyline': route_polyline,
            'center': {
                'lat': sum(all_lats) / len(all_lats),
                'lon': sum(all_lons) / len(all_lons)
            },
            'bounds': {
                'north': max(all_lats),
                'south': min(all_lats),
                'east': max(all_lons),
                'west': min(all_lons)
            },
            'zoom': 13
        }
    
    def _serialize_route(self, route) -> Optional[Dict[str, Any]]:
        """Serialize route object to JSON-compatible dict"""
        if not route:
            return None
        
        return {
            'total_distance': route.total_distance,
            'total_duration': route.total_duration,
            'summary': route.summary,
            'modes_used': route.modes_used,
            'steps': [
                {
                    'mode': step.mode,
                    'instruction': step.instruction,
                    'distance': step.distance,
                    'duration': step.duration,
                    'start_location': step.start_location,
                    'end_location': step.end_location,
                    'line_name': step.line_name,
                    'stops_count': step.stops_count
                }
                for step in route.steps
            ]
        }
    
    def _get_meal_context(self, current_time) -> str:
        """Get appropriate meal context based on time"""
        # Handle both datetime objects and integers
        if isinstance(current_time, datetime):
            hour = current_time.hour
        else:
            hour = current_time
            
        if hour < 11:
            return "Turkish breakfast"
        elif hour < 15:
            return "lunch"
        elif hour < 18:
            return "afternoon tea"
        else:
            return "dinner"
    
    def _get_weather_context(self, current_time: datetime) -> str:
        """Get weather-appropriate suggestions (mock implementation)"""
        month = current_time.month
        
        if month in [12, 1, 2]:  # Winter
            return "Consider indoor attractions like museums and covered markets during winter months."
        elif month in [6, 7, 8]:  # Summer
            return "Perfect weather for Bosphorus activities and outdoor dining!"
        elif month in [3, 4, 5]:  # Spring
            return "Beautiful spring weather - ideal for walking tours and outdoor cafes."
        else:  # Fall
            return "Lovely autumn weather - great time for photography and outdoor exploration."
    
    def extract_location_data_for_map(self, recommendations: List[Dict], location_type: str = 'auto') -> List[Dict]:
        """
        Extract structured location data with coordinates for map visualization
        
        Args:
            recommendations: List of recommendation dictionaries with 'name' keys
            location_type: Type of locations ('restaurant', 'attraction', 'neighborhood', 'auto')
        
        Returns:
            List of location objects ready for map display
        """
        if not self.location_service:
            return []
        
        map_locations = []
        for rec in recommendations:
            location_name = rec.get('name', '')
            if not location_name:
                continue
            
            coords = self.location_service.get_coordinates(location_name, location_type)
            if coords:
                map_locations.append({
                    'name': location_name,
                    'lat': coords['lat'],
                    'lng': coords['lng'],
                    'address': coords['address'],
                    'type': rec.get('type', location_type),
                    'description': rec.get('specialty', rec.get('description', '')),
                    'details': {
                        'price_range': rec.get('price_range'),
                        'hours': rec.get('hours'),
                        'highlights': rec.get('highlights'),
                        'best_for': rec.get('best_for'),
                        'visit_duration': rec.get('visit_duration'),
                        'best_time': rec.get('best_time'),
                    }
                })
        
        return map_locations
    
    def get_response_with_map_data(self, recommendation_type: str, entities: Dict, 
                                  user_profile: UserProfile, context: ConversationContext) -> Dict:
        """
        Generate response with both text and structured map data
        
        Args:
            recommendation_type: Type of recommendation
            entities: Extracted entities
            user_profile: User profile
            context: Conversation context
        
        Returns:
            Dictionary with 'text' and 'map_data' keys
        """
        # Generate text response
        text_response = self.generate_comprehensive_recommendation(
            recommendation_type, entities, user_profile, context
        )
        
        # Get structured location data for map
        map_data = []
        
        if recommendation_type == 'restaurant':
            # Extract restaurant recommendations
            recs = self._get_restaurant_recommendations(entities, user_profile, context)
            map_data = self.extract_location_data_for_map(recs, 'restaurant')
        elif recommendation_type == 'attraction':
            # Extract attraction recommendations
            recs = self._get_attraction_recommendations(entities, user_profile, context)
            map_data = self.extract_location_data_for_map(recs, 'attraction')
        elif recommendation_type == 'neighborhood':
            # Extract neighborhood recommendations
            recs = self._get_neighborhood_recommendations(entities, user_profile, context)
            map_data = self.extract_location_data_for_map(recs, 'neighborhood')
        
        return {
            'text': text_response,
            'map_data': map_data,
            'has_locations': len(map_data) > 0
        }
    
    def _get_restaurant_recommendations(self, entities: Dict, user_profile: UserProfile, 
                                       context: ConversationContext) -> List[Dict]:
        """Extract restaurant recommendations as structured data"""
        recommendations = []
        
        # Traditional Turkish restaurants
        if 'turkish_traditional' in entities.get('cuisines', []) or not entities.get('cuisines'):
            recommendations.extend([
                {
                    'name': 'Pandeli',
                    'type': 'Traditional Ottoman',
                    'specialty': 'Ottoman palace cuisine',
                    'price_range': 'Mid-range (150-300 TL per person)',
                    'hours': '12:00-17:00 (closed Sundays)',
                    'highlights': 'Historic 1901 building, ceramic tiles, traditional recipes',
                    'best_for': 'Cultural dining experience'
                },
                {
                    'name': 'Hünkar',
                    'type': 'Traditional Turkish',
                    'specialty': 'Home-style Turkish cooking (ev yemeği)',
                    'price_range': 'Moderate (100-200 TL per person)',
                    'hours': '11:30-22:00 daily',
                    'highlights': 'Family recipes since 1950, lamb dishes, traditional desserts',
                    'best_for': 'Authentic home cooking experience'
                }
            ])
        
        # Street food
        recommendations.extend([
            {
                'name': 'Tarihi Eminönü Balık Ekmek',
                'type': 'Street Food',
                'specialty': 'Fresh fish sandwiches from boats',
                'price_range': 'Budget (15-25 TL per sandwich)',
                'hours': '09:00-23:00 daily',
                'highlights': 'Caught daily, grilled on boats, Istanbul institution',
                'best_for': 'Authentic local experience'
            }
        ])
        
        return recommendations[:3]
    
    def _get_attraction_recommendations(self, entities: Dict, user_profile: UserProfile, 
                                       context: ConversationContext) -> List[Dict]:
        """Extract attraction recommendations as structured data"""
        return [
            {
                'name': 'Hagia Sophia',
                'type': 'Historic Monument',
                'description': '1,500-year history, stunning dome, Byzantine mosaics',
                'visit_duration': '1-2 hours',
                'hours': 'Open daily (prayer times may affect access)',
                'highlights': 'Byzantine mosaics, massive dome, Islamic calligraphy',
                'best_time': 'Early morning (9-11 AM) or late afternoon'
            },
            {
                'name': 'Topkapi Palace',
                'type': 'Palace Museum',
                'description': 'Ottoman imperial treasures, Bosphorus views',
                'visit_duration': '2-3 hours',
                'hours': '09:00-18:00 (closed Tuesdays in winter)',
                'highlights': 'Imperial treasury, harem, sacred relics',
                'best_time': 'Morning (9-11 AM) to avoid crowds'
            },
            {
                'name': 'Grand Bazaar',
                'type': 'Historic Market',
                'description': '4,000 shops, authentic Turkish crafts',
                'visit_duration': '1-3 hours',
                'hours': '09:00-19:00 (closed Sundays)',
                'highlights': 'Turkish carpets, ceramics, spices, historic architecture',
                'best_time': 'Morning for better prices, afternoon for atmosphere'
            }
        ]
    
    def _get_neighborhood_recommendations(self, entities: Dict, user_profile: UserProfile, 
                                         context: ConversationContext) -> List[Dict]:
        """Extract neighborhood recommendations as structured data"""
        return [
            {
                'name': 'Sultanahmet',
                'type': 'Historic District',
                'description': 'Historic peninsula with major monuments',
                'highlights': 'Hagia Sophia, Blue Mosque, Topkapi Palace',
                'best_for': 'History lovers, first-time visitors'
            },
            {
                'name': 'Beyoğlu',
                'type': 'Modern District',
                'description': 'Trendy area with nightlife and culture',
                'highlights': 'Galata Tower, Istiklal Street, rooftop bars',
                'best_for': 'Nightlife, shopping, contemporary culture'
            },
            {
                'name': 'Kadıköy',
                'type': 'Asian Side',
                'description': 'Authentic local life on the Asian side',
                'highlights': 'Street food, local markets, young vibe',
                'best_for': 'Authentic local experience, food exploration'
            }
        ]
    
    def _extract_location_data(self, recommendations: List[Dict], rec_type: str = 'poi') -> Optional[Dict]:
        """Extract location data for map visualization
        
        Args:
            recommendations: List of recommendation dicts with coordinates
            rec_type: Type of recommendation (restaurant, attraction, neighborhood)
            
        Returns:
            Dict with locations, center, bounds or None if no coordinates
        """
        locations = []
        
        for rec in recommendations:
            # Check for coordinates in various formats
            coords = None
            if 'coordinates' in rec and rec['coordinates']:
                coords = rec['coordinates']
            elif 'coords' in rec and rec['coords']:
                coords = rec['coords']
            elif 'lat' in rec and 'lon' in rec:
                coords = (rec['lat'], rec['lon'])
            elif 'lat' in rec and 'lng' in rec:
                coords = (rec['lat'], rec['lng'])
            
            # Also try to get from location service
            if not coords and self.location_service and 'name' in rec:
                try:
                    coords_data = self.location_service.get_coordinates(rec['name'], rec_type)
                    if coords_data:
                        coords = (coords_data['lat'], coords_data['lng'])
                except Exception as e:
                    # Silently skip if coordinate lookup fails
                    pass
            
            if coords:
                try:
                    lat, lon = coords
                    locations.append({
                        'lat': float(lat),
                        'lon': float(lon),
                        'name': rec.get('name', 'Unknown'),
                        'type': rec_type,
                        'metadata': {
                            'description': rec.get('description', rec.get('specialty', '')),
                            'address': rec.get('location', ''),
                            'rating': rec.get('rating'),
                            'price': rec.get('price_range'),
                            'hours': rec.get('hours'),
                            'highlights': rec.get('highlights'),
                            'best_for': rec.get('best_for')
                        }
                    })
                except (ValueError, TypeError):
                    # Skip invalid coordinates
                    continue
        
        if not locations:
            return None
        
        # Calculate center and bounds
        lats = [loc['lat'] for loc in locations]
        lons = [loc['lon'] for loc in locations]
        
        return {
            'locations': locations,
            'center': {
                'lat': sum(lats) / len(lats),
                'lon': sum(lons) / len(lons)
            },
            'bounds': {
                'north': max(lats),
                'south': min(lats),
                'east': max(lons),
                'west': min(lons)
            },
            'zoom': 13 if len(locations) > 1 else 15
        }

    def _generate_transportation_directions(
        self,
        entities: Dict,
        user_profile: UserProfile,
        context: ConversationContext,
        return_structured: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate detailed transportation directions (Google Maps style)
        
        Args:
            entities: Extracted entities (origin, destination, locations)
            user_profile: User profile
            context: Conversation context
            return_structured: If True, return dict with response and map_data
            
        Returns:
            Detailed transportation directions with route visualization
        """
        
        # Extract origin and destination from entities
        origin = entities.get('origin')
        destination = entities.get('destination')
        landmarks = entities.get('landmarks', [])
        
        # Try to resolve location names to coordinates
        origin_coords = None
        destination_coords = None
        origin_name = "your location"
        destination_name = "destination"
        
        if origin and self.location_service:
            coords_data = self.location_service.get_coordinates(origin, 'auto')
            if coords_data:
                origin_coords = (coords_data['lat'], coords_data['lng'])
                origin_name = origin
        
        if destination and self.location_service:
            coords_data = self.location_service.get_coordinates(destination, 'auto')
            if coords_data:
                destination_coords = (coords_data['lat'], coords_data['lng'])
                destination_name = destination
        
        # If no specific origin/destination but landmarks mentioned, find route between first two
        if not origin_coords and not destination_coords and len(landmarks) >= 2:
            if self.location_service:
                first_coords = self.location_service.get_coordinates(landmarks[0], 'auto')
                second_coords = self.location_service.get_coordinates(landmarks[1], 'auto')
                if first_coords and second_coords:
                    origin_coords = (first_coords['lat'], first_coords['lng'])
                    destination_coords = (second_coords['lat'], second_coords['lng'])
                    origin_name = landmarks[0]
                    destination_name = landmarks[1]
        
        # Default to popular routes if nothing specified
        if not origin_coords or not destination_coords:
            # Default example: Sultanahmet to Taksim
            origin_coords = (41.0054, 28.9768)  # Sultanahmet
            destination_coords = (41.0370, 28.9850)  # Taksim
            origin_name = "Sultanahmet"
            destination_name = "Taksim Square"
        
        # Get detailed directions
        route = None
        if self.transport_service:
            try:
                route = self.transport_service.get_directions(
                    start=origin_coords,
                    end=destination_coords,
                    start_name=origin_name,
                    end_name=destination_name
                )
            except Exception as e:
                print(f"⚠️ Transportation service error: {e}")
        
        # Build response
        response_parts = []
        
        # Header
        response_parts.append(f"🗺️ **Directions from {origin_name} to {destination_name}**\n")
        
        if route:
            # Format the detailed route
            response_parts.append(f"⏱️ **Total Time:** {route.total_duration} minutes")
            response_parts.append(f"📏 **Total Distance:** {route.total_distance/1000:.1f} km")
            
            modes_icons = {
                'walk': '🚶 Walking',
                'metro': '🚇 Metro',
                'tram': '🚊 Tram',
                'bus': '🚌 Bus',
                'ferry': '⛴️ Ferry'
            }
            modes_str = ', '.join([modes_icons.get(m, m) for m in set(route.modes_used)])
            response_parts.append(f"🚉 **Transport Modes:** {modes_str}\n")
            
            # Detailed steps
            response_parts.append("**Step-by-Step Directions:**\n")
            for i, step in enumerate(route.steps, 1):
                icon = {
                    'walk': '🚶',
                    'metro': '🚇',
                    'tram': '🚊',
                    'bus': '🚌',
                    'ferry': '⛴️'
                }.get(step.mode, '➡️')
                
                response_parts.append(f"**{i}. {icon} {step.instruction}**")
                response_parts.append(f"   • Distance: {step.distance/1000:.1f} km")
                response_parts.append(f"   • Duration: {step.duration} minutes")
                
                if step.line_name:
                    response_parts.append(f"   • Line: {step.line_name}")
                if step.stops_count:
                    response_parts.append(f"   • Stops: {step.stops_count}")
                response_parts.append("")
        else:
            # Fallback general directions
            response_parts.append(f"Here's how to travel between {origin_name} and {destination_name}:\n")
            response_parts.append("🚇 **Metro/Tram Option:**")
            response_parts.append("1. Walk to the nearest metro or tram station")
            response_parts.append("2. Take the appropriate line towards your destination")
            response_parts.append("3. Transfer if needed at major hubs")
            response_parts.append("4. Walk to your final destination\n")
            
            response_parts.append("💡 **Tips:**")
            response_parts.append("• Get an Istanbulkart for easy payment on all public transport")
            response_parts.append("• Metro frequency: Every 5-10 minutes during peak hours")
            response_parts.append("• Trams run frequently on major routes")
            response_parts.append("• Download the official IETT app for real-time schedules")
        
        # Add general Istanbul transportation info
        response_parts.append("\n📱 **Transportation Resources:**")
        response_parts.append("• **Istanbulkart:** Essential rechargeable transport card")
        response_parts.append("• **IETT App:** Real-time bus tracking and schedules")
        response_parts.append("• **Metro hours:** Approximately 6:00 AM - 12:00 AM")
        response_parts.append("• **Ferry schedules:** Check İDO or Şehir Hatları websites")
        response_parts.append("• **Taxi apps:** BiTaksi, iTaksi for reliable rides")
        
        response_text = '\n'.join(response_parts)
        
        # Build map data with route visualization
        if return_structured:
            map_data = self._build_route_map_data(route, origin_coords, destination_coords, origin_name, destination_name)
            return {
                'response': response_text,
                'map_data': map_data,
                'recommendation_type': 'transportation',
                'route_data': self._serialize_route(route) if route else None
            }
        else:
            return response_text
    
    def _build_route_map_data(
        self,
        route,
        origin_coords: Tuple[float, float],
        destination_coords: Tuple[float, float],
        origin_name: str,
        destination_name: str
    ) -> Dict[str, Any]:
        """Build map data for route visualization"""
        
        locations = []
        route_polyline = []
        
        # Add origin and destination markers
        locations.append({
            'lat': origin_coords[0],
            'lon': origin_coords[1],
            'name': origin_name,
            'type': 'origin',
            'metadata': {
                'description': 'Starting point',
                'icon': 'start'
            }
        })
        
        locations.append({
            'lat': destination_coords[0],
            'lon': destination_coords[1],
            'name': destination_name,
            'type': 'destination',
            'metadata': {
                'description': 'Destination',
                'icon': 'end'
            }
        })
        
        # Build route polyline from steps
        if route and route.steps:
            for step in route.steps:
                # Add waypoints if available
                if step.waypoints:
                    for wp in step.waypoints:
                        route_polyline.append({'lat': wp[0], 'lng': wp[1]})
                else:
                    # Add start and end of step
                    route_polyline.append({'lat': step.start_location[0], 'lng': step.start_location[1]})
                    route_polyline.append({'lat': step.end_location[0], 'lng': step.end_location[1]})
        
        # If no detailed route, add simple line
        if not route_polyline:
            route_polyline = [
                {'lat': origin_coords[0], 'lng': origin_coords[1]},
                {'lat': destination_coords[0], 'lng': destination_coords[1]}
            ]
        
        # Calculate bounds
        all_lats = [loc['lat'] for loc in locations] + [p['lat'] for p in route_polyline]
        all_lons = [loc['lon'] for loc in locations] + [p['lng'] for p in route_polyline]
        
        return {
            'locations': locations,
            'route_polyline': route_polyline,
            'center': {
                'lat': sum(all_lats) / len(all_lats),
                'lon': sum(all_lons) / len(all_lons)
            },
            'bounds': {
                'north': max(all_lats),
                'south': min(all_lats),
                'east': max(all_lons),
                'west': min(all_lons)
            },
            'zoom': 13
        }
    
    def _serialize_route(self, route) -> Optional[Dict[str, Any]]:
        """Serialize route object to JSON-compatible dict"""
        if not route:
            return None
        
        return {
            'total_distance': route.total_distance,
            'total_duration': route.total_duration,
            'summary': route.summary,
            'modes_used': route.modes_used,
            'steps': [
                {
                    'mode': step.mode,
                    'instruction': step.instruction,
                    'distance': step.distance,
                    'duration': step.duration,
                    'start_location': step.start_location,
                    'end_location': step.end_location,
                    'line_name': step.line_name,
                    'stops_count': step.stops_count
                }
                for step in route.steps
            ]
        }
