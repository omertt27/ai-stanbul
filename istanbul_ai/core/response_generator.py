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

# Import hidden gems service
try:
    from services.hidden_gems_service import HiddenGemsService
    HIDDEN_GEMS_AVAILABLE = True
except ImportError:
    HIDDEN_GEMS_AVAILABLE = False
    print("⚠️ Hidden gems service not available")

# Import airport transport service
try:
    from services.airport_transport_service import get_airport_transport_service
    AIRPORT_TRANSPORT_AVAILABLE = True
except ImportError:
    AIRPORT_TRANSPORT_AVAILABLE = False
    print("⚠️ Airport transport service not available")

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
        
        # Initialize hidden gems service
        if HIDDEN_GEMS_AVAILABLE:
            self.hidden_gems_service = HiddenGemsService()
        else:
            self.hidden_gems_service = None
        
        # Initialize airport transport service
        if AIRPORT_TRANSPORT_AVAILABLE:
            self.airport_transport_service = get_airport_transport_service()
        else:
            self.airport_transport_service = None
    
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
        elif recommendation_type == 'hidden_gems':
            return self._generate_hidden_gems_recommendation(entities, user_profile, context, return_structured)
        elif recommendation_type == 'airport_transport':
            return self._generate_airport_transport_recommendation(entities, user_profile, context, return_structured)
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
            },
            {
                'name': 'Beşiktaş (Bosphorus Waterfront)',
                'character': 'Modern district with Bosphorus views, shopping, and football culture',
                'best_for': 'Shopping, waterfront walks, modern Istanbul experience, sports fans',
                'highlights': [
                    'Dolmabahçe Palace (Ottoman grandeur)',
                    'Beşiktaş Maritime Museum and naval history',
                    'Modern shopping centers (Akmerkez, Zorlu Center)',
                    'Bosphorus waterfront parks and promenades',
                    'BJK İnönü Stadium (football culture)'
                ],
                'atmosphere': 'Mix of business district and leisure, family-friendly, upscale',
                'transport': 'Metro M6, ferries, buses from Taksim (15 min)',
                'budget': 'Mid-range to upscale, international chains and local favorites',
                'best_time': 'Weekday mornings for palace visits, evenings for waterfront'
            },
            {
                'name': 'Üsküdar (Asian Side Historic)',
                'character': 'Traditional Islamic culture, stunning city views, peaceful atmosphere',
                'best_for': 'Cultural immersion, panoramic views, religious sites, escaping crowds',
                'highlights': [
                    'Maiden\'s Tower (Kız Kulesi) - iconic Bosphorus landmark',
                    'Mihrimah Sultan Mosque (stunning sunset views)',
                    'Traditional tea gardens with Golden Horn views',
                    'Historic Ottoman wooden houses',
                    'Çamlıca Hill - highest point with panoramic Istanbul views'
                ],
                'atmosphere': 'Conservative, traditional, family-oriented, peaceful',
                'transport': 'Ferry from Eminönü/Karaköy (15-20 min), Marmaray train',
                'budget': 'Very affordable, traditional Turkish pricing',
                'best_time': 'Late afternoon for best light and sunset views'
            },
            {
                'name': 'Şişli (Business & Shopping Hub)',
                'character': 'Modern commercial center, upscale shopping, international atmosphere',
                'best_for': 'Shopping, business meetings, modern dining, luxury experiences',
                'highlights': [
                    'Nişantaşı luxury shopping district',
                    'City\'s (Şişli) modern shopping mall complex',
                    'High-end restaurants and international cuisine',
                    'Business district with modern architecture',
                    'Cevahir Shopping Center (one of Europe\'s largest)'
                ],
                'atmosphere': 'Cosmopolitan, business-oriented, upscale, international',
                'transport': 'Metro M2 (Şişli-Mecidiyeköy stations), extensive bus network',
                'budget': 'Higher-end dining and shopping, business expense level',
                'best_time': 'Weekdays for business, weekends for shopping and dining'
            },
            {
                'name': 'Sarıyer (Bosphorus Village Life)',
                'character': 'Historic fishing villages, forest walks, weekend retreat atmosphere',
                'best_for': 'Nature lovers, historic sites, weekend escapes, seafood dining',
                'highlights': [
                    'Rumeli Fortress (medieval castle with Bosphorus views)',
                    'Belgrade Forest (hiking, picnics, fresh air)',
                    'Traditional seafood restaurants along Bosphorus',
                    'Historic Sadberk Hanım Museum',
                    'Emirgan Park (famous for tulips in spring)',
                    'Traditional Turkish village atmosphere'
                ],
                'atmosphere': 'Relaxed, village-like, nature-focused, weekend destination',
                'transport': 'Bus 25E from Kabataş, dolmuş (shared taxi, cash only) from Beşiktaş',
                'budget': 'Mid-range, especially known for seafood restaurants',
                'best_time': 'Spring for tulips, weekends for full village experience'
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
            response_parts.append("• Get an Istanbulkart for easy payment on official public transport (metro, bus, tram, ferry)")
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
    
    def _generate_hidden_gems_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                           context: ConversationContext, return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate hidden gems recommendations using the dedicated service"""
        
        if not self.hidden_gems_service:
            fallback = "I'd love to share hidden gems with you, but my secret knowledge base isn't available right now. Try asking about specific districts or types of places you're interested in!"
            return {'response': fallback, 'map_data': None, 'recommendation_type': 'hidden_gems'} if return_structured else fallback
        
        # Extract the original query from context to pass to the service
        query = context.last_query if hasattr(context, 'last_query') else ""
        
        # If no specific query, construct one from entities
        if not query:
            query_parts = []
            if entities.get('districts'):
                query_parts.append(f"in {entities['districts'][0]}")
            if entities.get('categories'):
                query_parts.append(entities['categories'][0])
            query_parts.append("hidden gems")
            query = " ".join(query_parts)
        
        try:
            # Get hidden gems recommendations
            gems_response = self.hidden_gems_service.search_hidden_gems(query)
            
            # Add contextual introduction based on user profile and time
            current_time = datetime.now()
            intro_context = []
            
            if current_time.hour < 12:
                intro_context.append("Good morning! Perfect timing to discover some of Istanbul's hidden treasures.")
            elif current_time.hour < 17:
                intro_context.append("Great afternoon to explore some secret spots!")
            else:
                intro_context.append("Evening is a magical time to discover hidden gems in Istanbul.")
            
            # Add user-specific context
            if user_profile and hasattr(user_profile, 'interests'):
                if 'photography' in str(user_profile.interests).lower():
                    intro_context.append("As a photography enthusiast, you'll love these spots for unique shots.")
                elif 'history' in str(user_profile.interests).lower():
                    intro_context.append("These historical secrets will fascinate any history lover.")
            
            # Combine introduction with gems response
            if intro_context:
                response = "\n".join(intro_context) + "\n\n" + gems_response
            else:
                response = gems_response
            
            # Add practical closing advice
            response += "\n\n💡 **Pro tip**: These hidden gems are best visited during weekdays when they're less crowded. Don't forget to respect local customs and photography restrictions!"
            
            if return_structured:
                # Build map data for hidden gems if coordinates are available
                map_data = self._build_gems_map_data()
                return {
                    'response': response,
                    'map_data': map_data,
                    'recommendation_type': 'hidden_gems'
                }
            else:
                return response
                
        except Exception as e:
            print(f"❌ Error generating hidden gems recommendation: {e}")
            fallback = "I'm having trouble accessing my collection of hidden gems right now. Try asking about specific neighborhoods or types of places you'd like to discover!"
            return {'response': fallback, 'map_data': None, 'recommendation_type': 'hidden_gems'} if return_structured else fallback
    
    def _build_gems_map_data(self) -> Dict[str, Any]:
        """Build map data for hidden gems visualization"""
        if not self.hidden_gems_service or not self.hidden_gems_service.gems:
            return None
        
        map_data = {
            'type': 'hidden_gems',
            'center': [28.9784, 41.0082],  # Istanbul center
            'zoom': 11,
            'markers': []
        }
        
        # Add markers for gems with coordinates
        for gem in self.hidden_gems_service.gems[:10]:  # Limit to first 10 for performance
            location = gem.get('location', {})
            coords = location.get('coordinates')
            if coords and len(coords) == 2:
                marker = {
                    'coordinates': [coords[0], coords[1]],  # [lng, lat]
                    'title': gem.get('name', 'Hidden Gem'),
                    'description': gem.get('description', '')[:100] + '...',
                    'category': gem.get('category', 'attraction'),
                    'icon': self._get_gem_icon(gem.get('category', 'attraction'))
                }
                map_data['markers'].append(marker)
        
        return map_data
    
    def _get_gem_icon(self, category: str) -> str:
        """Get appropriate icon for gem category"""
        icon_map = {
            'historical': '🏛️',
            'cultural': '🎭',
            'culinary': '🍽️',
            'nature': '🌿',
            'shopping': '🛍️',
            'nightlife': '🌙'
        }
        return icon_map.get(category, '💎')

    def _generate_fallback_response(self, context, user_profile) -> str:
        """Generate a fallback response when no specific intent is detected"""
        return """👋 I'm here to help you explore Istanbul! I can assist with:

🍽️ **Restaurants**: Find dining by cuisine, district, or dietary needs
🏛️ **Attractions**: Discover museums, landmarks, and hidden gems  
🏘️ **Neighborhoods**: Get detailed guides for different districts
🚇 **Transportation**: Directions, metro routes, and travel tips
🎭 **Events**: Current cultural events and activities

What would you like to explore in Istanbul?"""
    
    def _get_weather_context(self, current_time: datetime) -> Dict[str, Any]:
        """Get weather context for recommendations"""
        # This is a simple fallback weather context
        # In a full implementation, this would call a weather service
        season = self._get_season(current_time)
        return {
            'season': season,
            'outdoor_friendly': season in ['spring', 'summer', 'early_fall'],
            'indoor_recommended': season in ['winter', 'late_fall']
        }
    
    def _get_season(self, current_time: datetime) -> str:
        """Determine current season"""
        month = current_time.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring' 
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _generate_airport_transport_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                                 context: ConversationContext, return_structured: bool = False) -> Union[str, Dict[str, Any]]:
        """Generate airport transport recommendations using the dedicated service"""
        
        if not self.airport_transport_service:
            fallback = "I'd love to help with airport transport, but my airport information service isn't available right now. Try asking about specific airports (IST, SAW) or transport options."
            return {'response': fallback, 'map_data': None, 'recommendation_type': 'airport_transport'} if return_structured else fallback
        
        try:
            # Extract airport code and destination from entities or context
            airport_code = None
            destination = None
            
            # Check for airport mentions in the query
            query = context.last_query if hasattr(context, 'last_query') else ""
            query_lower = query.lower()
            
            # Detect airport codes
            if any(code in query_lower for code in ['ist', 'istanbul airport', 'new airport']):
                airport_code = 'IST'
            elif any(code in query_lower for code in ['saw', 'sabiha gökçen', 'sabiha gokcen', 'asian side airport']):
                airport_code = 'SAW'
            elif any(code in query_lower for code in ['atl', 'atatürk', 'ataturk', 'old airport']):
                airport_code = 'ATL'  # Will trigger closed airport message
            
            # Detect destinations from entities
            if entities.get('districts'):
                destination = entities['districts'][0]
            elif entities.get('neighborhoods'):
                destination = entities['neighborhoods'][0]
            
            # If no specific airport detected, show comparison
            if not airport_code:
                response = self.airport_transport_service.get_airport_comparison()
                response = "🛩️ **Istanbul Airport Transport Guide**\n\n" + response + "\n\n💡 Ask me about specific airports (IST or SAW) for detailed transport options!"
            else:
                response = self.airport_transport_service.get_route_recommendations(airport_code, destination)
            
            # Add contextual tips based on time and user profile
            current_time = datetime.now()
            hour = current_time.hour
            
            if hour < 6 or hour > 23:
                response += "\n\n🌙 **Late night/Early morning**: Check if metro is running - buses may be your only option during off-peak hours."
            elif 7 <= hour <= 9 or 17 <= hour <= 19:
                response += "\n\n⏰ **Rush hour alert**: Allow extra 30-45 minutes for your journey due to heavy traffic."
            
            # Add user-specific recommendations
            if user_profile and hasattr(user_profile, 'user_type'):
                if 'budget' in str(user_profile.user_type).lower():
                    response += "\n\n💰 **Budget tip**: Metro lines (M11/M4) are the most cost-effective options at 13.5₺."
                elif 'business' in str(user_profile.user_type).lower():
                    response += "\n\n💼 **Business traveler**: Consider taxi/ride-share for door-to-door service, especially with luggage."
            
            if return_structured:
                # Build map data for airport transport
                map_data = self._build_airport_map_data(airport_code)
                return {
                    'response': response,
                    'map_data': map_data,
                    'recommendation_type': 'airport_transport'
                }
            else:
                return response
                
        except Exception as e:
            print(f"❌ Error generating airport transport recommendation: {e}")
            fallback = "I'm having trouble accessing airport transport information right now. Try asking about IST (Istanbul Airport) or SAW (Sabiha Gökçen Airport) specifically."
            return {'response': fallback, 'map_data': None, 'recommendation_type': 'airport_transport'} if return_structured else fallback
    
    def _build_airport_map_data(self, airport_code: str) -> Dict[str, Any]:
        """Build map data for airport transport visualization"""
        if not self.airport_transport_service or not airport_code:
            return None
        
        airport = self.airport_transport_service.get_airport_info(airport_code)
        if not airport or airport.status != 'active':
            return None
        
        map_data = {
            'type': 'airport_transport',
            'center': list(airport.coordinates),  # [lat, lng]
            'zoom': 10,
            'markers': []
        }
        
        # Add airport marker
        airport_marker = {
            'coordinates': list(airport.coordinates),
            'title': f"{airport.name} ({airport.code})",
            'description': f"Airport located in {airport.location}",
            'icon': '✈️',
            'type': 'airport'
        }
        map_data['markers'].append(airport_marker)
        
        # Add transport route endpoints
        routes = self.airport_transport_service.get_transport_options(airport_code)
        for route in routes[:5]:  # Limit to first 5 routes for performance
            # This would need destination coordinates in a real implementation
            # For now, just add the airport marker
            pass
        
        return map_data
