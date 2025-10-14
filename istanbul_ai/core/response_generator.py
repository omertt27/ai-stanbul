"""
Response Generator
Enhanced response generation with comprehensive recommendations and contextual awareness.
"""

import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from ..core.models import UserProfile, ConversationContext


class ResponseGenerator:
    """Enhanced response generator with comprehensive Istanbul recommendations"""
    
    def __init__(self):
        self.initialize_response_templates()
    
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
                    "**Hagia Sophia** - Marvel at this architectural wonder that served as a church, mosque, and now museum. The golden mosaics and soaring dome represent 1,500 years of history.",
                    "**Topkapi Palace** - Explore the opulent former residence of Ottoman sultans, with stunning views over the Bosphorus and priceless imperial collections.",
                    "**Blue Mosque** - Admire the six minarets and blue Iznik tiles of this active place of worship, a masterpiece of Ottoman architecture.",
                    "**Basilica Cistern** - Descend into this mystical underground marvel with 336 columns, featured in countless films and legends."
                ]
            },
            'cultural': {
                'intro': "🎭 Istanbul's cultural richness reflects its position as a bridge between worlds:",
                'details': [
                    "**Grand Bazaar** - Navigate through 4,000 shops in this historic covered market, perfect for authentic Turkish carpets, ceramics, and spices.",
                    "**Galata Tower** - Enjoy panoramic 360° views of the city from this medieval Genoese tower, especially magical at sunset.",
                    "**Turkish and Islamic Arts Museum** - Discover world-class collections of calligraphy, ceramics, and the famous carpet collection.",
                    "**Dolmabahce Palace** - Experience 19th-century Ottoman luxury in this European-style palace on the Bosphorus."
                ]
            }
        }
    
    def generate_comprehensive_recommendation(self, recommendation_type: str, entities: Dict, 
                                           user_profile: UserProfile, context: ConversationContext) -> str:
        """Generate comprehensive 150-300 word recommendations with practical information"""
        
        current_time = datetime.now()
        hour = current_time.hour
        
        # Get weather-appropriate suggestions
        weather_context = self._get_weather_context(current_time)
        
        if recommendation_type == 'restaurant':
            return self._generate_enhanced_restaurant_recommendation(entities, user_profile, context, current_time)
        elif recommendation_type == 'attraction':
            return self._generate_enhanced_attraction_recommendation(entities, user_profile, context, current_time)
        elif recommendation_type == 'neighborhood':
            return self._generate_enhanced_neighborhood_recommendation(entities, user_profile, context, current_time)
        else:
            return self._generate_fallback_response(context, user_profile)
    
    def _generate_enhanced_restaurant_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                                   context: ConversationContext, current_time: datetime) -> str:
        """Generate comprehensive restaurant recommendations with practical details"""
        
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
        
        # Add recommendations with full details
        for i, rec in enumerate(recommendations[:3], 1):
            response_parts.append(f"""
**{i}. {rec['name']}** ({rec['type']})
📍 **Location**: {rec['location']}
🍴 **Specialty**: {rec['specialty']}
💰 **Price**: {rec['price_range']}
🕐 **Hours**: {rec['hours']}
🚇 **Transport**: {rec['transport']}
✨ **Why visit**: {rec['highlights']}
👥 **Best for**: {rec['best_for']}""")
        
        # Add practical tips
        response_parts.append(f"""
**💡 Practical Tips:**
• Make reservations for dinner, especially weekends
• Try Turkish tea (çay) or coffee after meals
• Tipping: 10-15% is standard for good service
• Many restaurants offer English menus in tourist areas
• Ask for 'hesap' (heh-sahp) when you want the bill""")
        
        # Add weather-appropriate suggestion
        weather_context = self._get_weather_context(current_time)
        if weather_context:
            response_parts.append(f"🌤️ **Weather note**: {weather_context}")
        
        return '\n'.join(response_parts)
    
    def _generate_enhanced_attraction_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                                   context: ConversationContext, current_time: datetime) -> str:
        """Generate comprehensive attraction recommendations with practical details"""
        
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
                'entry_fee': 'Free (donations welcome)',
                'hours': '24/7 (prayer times may restrict access)',
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
                'entry_fee': '100 TL (Harem additional 70 TL)',
                'hours': '09:00-18:00 (closed Tuesdays)',
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
                'entry_fee': 'Free',
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
**{i}. {attraction['name']}** ({attraction['type']})
📍 **Location**: {attraction['location']}
⏰ **Visit time**: {attraction['visit_duration']}
💰 **Entry fee**: {attraction['entry_fee']}
🕐 **Hours**: {attraction['hours']}
🚇 **Transport**: {attraction['transport']}
✨ **Highlights**: {attraction['highlights']}
📸 **Photography**: {attraction['photography']}
♿ **Accessibility**: {attraction['accessibility']}
🌟 **Best time**: {attraction['best_time']}""")
        
        # Add practical visiting tips
        response_parts.append(f"""
**💡 Essential Visiting Tips:**
• **Museum Pass**: Consider Istanbul Museum Pass (325 TL) for multiple attractions
• **Dress code**: Modest clothing for mosques (covering shoulders/knees)
• **Prayer times**: Some mosques close 30 min before prayers
• **Crowds**: Visit major attractions early morning or late afternoon
• **Guided tours**: Available in multiple languages at most sites
• **Audio guides**: Often available for self-guided exploration""")
        
        # Add transportation and route suggestions
        response_parts.append(f"""
**🚇 Getting Around:**
• **Sultanahmet area**: Most historic attractions within walking distance
• **Istanbulkart**: Essential transport card (13 TL + credit)
• **Tram T1**: Connects Sultanahmet to Galata Bridge and beyond
• **Metro/Tram combos**: Efficient for crossing between districts""")
        
        return '\n'.join(response_parts)
    
    def _generate_enhanced_neighborhood_recommendation(self, entities: Dict, user_profile: UserProfile, 
                                                     context: ConversationContext, current_time: datetime) -> str:
        """Generate comprehensive neighborhood recommendations"""
        
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
**{i}. {neighborhood['name']}**
🏘️ **Character**: {neighborhood['character']}
👥 **Best for**: {neighborhood['best_for']}
✨ **Highlights**:
{chr(10).join(f'   • {highlight}' for highlight in neighborhood['highlights'])}
🎭 **Atmosphere**: {neighborhood['atmosphere']}
🚇 **Transport**: {neighborhood['transport']}
💰 **Budget**: {neighborhood['budget']}
⏰ **Best time**: {neighborhood['best_time']}""")
        
        # Add comprehensive area guide
        response_parts.append(f"""
**🗺️ Navigation Tips:**
• **Cross-Continental**: Take ferries between European and Asian sides
• **Historic walking**: Sultanahmet to Galata Bridge is a beautiful walk
• **Local transport**: Each neighborhood has distinct transport connections
• **District hopping**: Plan 2-3 hours minimum per neighborhood

**🎯 Choosing Your Base:**
• **History focus**: Stay in Sultanahmet
• **Nightlife/modern**: Choose Beyoğlu/Galata
• **Local experience**: Consider Asian side (Kadıköy/Üsküdar)
• **Luxury/views**: Bosphorus-facing areas in Beşiktaş""")
        
        return '\n'.join(response_parts)
    
    def _generate_fallback_response(self, context: ConversationContext, user_profile: UserProfile) -> str:
        """Generate comprehensive fallback response when specific recommendations aren't available"""
        
        current_time = datetime.now()
        hour = current_time.hour
        
        # Time-based suggestions
        if hour < 10:
            time_suggestion = "🌅 **Perfect morning activities**: Visit Hagia Sophia before crowds, enjoy traditional Turkish breakfast in Sultanahmet, or take an early Bosphorus walk."
        elif hour < 14:
            time_suggestion = "☀️ **Great midday options**: Explore Grand Bazaar, enjoy lunch in a traditional lokanta, visit Topkapi Palace with its shaded courtyards."
        elif hour < 18:
            time_suggestion = "🌤️ **Wonderful afternoon choices**: Climb Galata Tower for sunset views, stroll through trendy Beyoğlu, discover local cafes in Karaköy."
        else:
            time_suggestion = "🌆 **Evening magic awaits**: Experience Istanbul's vibrant nightlife, enjoy dinner with Bosphorus views, or explore illuminated historic monuments."
        
        fallback_response = f"""
🎯 I'd love to give you more specific recommendations! While I gather more details about what you're looking for, here are some wonderful Istanbul experiences:

{time_suggestion}

**🏛️ Must-See Attractions (any time):**
• **Hagia Sophia** - Architectural marvel spanning 1,500 years
• **Blue Mosque** - Stunning Ottoman architecture with six minarets  
• **Grand Bazaar** - 4,000 shops in historic covered market
• **Bosphorus** - The strait that divides Europe and Asia

**🍽️ Culinary Experiences:**
• **Traditional breakfast** - Try serpme kahvaltı (spread breakfast)
• **Street food** - Döner, simit, and fresh fish sandwiches
• **Ottoman cuisine** - Historic recipes in traditional restaurants
• **Turkish coffee & baklava** - Perfect afternoon treats

**🏘️ Neighborhood Character:**
• **Sultanahmet** - Historic heart with major monuments
• **Beyoğlu** - Modern, trendy area with nightlife
• **Kadıköy** - Authentic local life on Asian side
• **Beşiktaş** - Upscale with beautiful Bosphorus views

**💡 Pro Tips:**
• Get an Istanbulkart for easy public transport
• Dress modestly when visiting mosques
• Learn basic Turkish greetings - locals appreciate it!
• Always negotiate prices in markets
• Try Turkish tea (çay) - it's offered everywhere!

What specifically interests you most? I can provide detailed recommendations based on your preferences, budget, time available, or any particular experiences you're seeking!
"""
        
        return fallback_response
    
    def _enhance_multi_intent_response(self, base_response: str, intents: List[str], 
                                     entities: Dict, user_profile: UserProfile) -> str:
        """Enhance responses when multiple intents are detected"""
        
        if len(intents) <= 1:
            return base_response
        
        enhancement_parts = [base_response]
        
        # Add connections between different intents
        if 'restaurant' in intents and 'attraction' in intents:
            enhancement_parts.append("""
**🍽️➡️🏛️ Perfect Combinations:**
• Visit Hagia Sophia, then lunch at nearby Pandeli restaurant
• Explore Grand Bazaar, then traditional Ottoman dinner in Sultanahmet
• Morning at Topkapi Palace, afternoon tea in historic Soğukçeşme Street""")
        
        if 'transportation' in intents and ('restaurant' in intents or 'attraction' in intents):
            enhancement_parts.append("""
**🚇 Easy Transport Connections:**
• Sultanahmet tram connects all major historic sites
• Ferry rides offer scenic routes between districts
• Metro system efficiently connects modern areas
• Walking between nearby attractions saves time and money""")
        
        if 'neighborhood' in intents and 'restaurant' in intents:
            enhancement_parts.append("""
**🏘️🍴 Neighborhood Food Specialties:**
• **Sultanahmet**: Traditional Ottoman cuisine and tourist-friendly restaurants
• **Beyoğlu**: Trendy cafes, international cuisine, and rooftop dining
• **Kadıköy**: Authentic local eateries and incredible street food markets
• **Beşiktaş**: Upscale dining with Bosphorus views""")
        
        # Add user-specific enhancements
        if user_profile.travel_style == 'family':
            enhancement_parts.append("""
**👨‍👩‍👧‍👦 Family-Friendly Tips:**
• Many restaurants welcome children and offer high chairs
• Historic sites have facilities and shorter visit options
• Parks and waterfront areas great for kids to play
• Traditional Turkish breakfast is perfect for families""")
        
        return '\n'.join(enhancement_parts)
    
    def _get_meal_context(self, hour: int) -> str:
        """Get appropriate meal context based on time"""
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
