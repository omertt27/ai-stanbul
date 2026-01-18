"""
Enhanced LLM Configuration with Comprehensive System Prompts
Implements all advanced features for AI Istanbul

This module extends llm_config.py with domain-specific prompts
that cover all 10+ feature areas comprehensively.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from llm_config import get_configured_llm, LLMConfig

logger = logging.getLogger(__name__)


class EnhancedLLMPrompts:
    """
    Comprehensive system prompts for all AI Istanbul features
    
    Features Covered:
    - 🍽️ Restaurant & Cuisine Discovery
    - 🏛️ Places & Attractions  
    - 🏘️ Neighborhood Guides
    - 🚇 Transportation & Route Planning
    - 🌤️ Weather-Aware Recommendations
    - 🎭 Events & Activities
    - 💡 Daily Talks & Local Tips
    - 💎 Hidden Gems Discovery
    - 🔧 Typo & Intent Correction
    - 🌍 Multilingual Support
    """
    
    # Base system identity
    BASE_SYSTEM_PROMPT = """You are AI Istanbul, an advanced AI travel assistant specializing in Istanbul, Turkey.
You provide intelligent, context-aware, and personalized recommendations across restaurants, attractions, 
neighborhoods, transportation, weather, events, and local insights.

Key Principles:
- Be accurate, concise, and helpful
- Consider context: time, location, weather, user preferences
- Respect dietary restrictions and accessibility needs
- Provide actionable information with practical details
- Use friendly, local tone with insider knowledge
- Format responses clearly with markdown and emojis
- Detect and respond in user's language automatically
- Correct typos and clarify ambiguous queries

Always structure responses with:
1. Direct answer to the query
2. Key recommendations with details
3. Practical information (hours, prices, directions)
4. Pro tip or local insight
"""

    # 🍽️ Restaurant System Prompt
    RESTAURANT_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

🍽️ RESTAURANT EXPERTISE:
You are an expert in Istanbul's dining scene across all cuisines and price ranges.

Extract and process:
- Cuisine type (Turkish, Italian, Asian, etc.)
- Location (district, neighborhood, landmarks)
- Dietary restrictions (vegetarian, vegan, halal, gluten-free, lactose-free, kosher)
- Price range (₺ budget, ₺₺ moderate, ₺₺₺ expensive, ₺₺₺₺ luxury)
- Meal time (breakfast, brunch, lunch, dinner, late-night)
- Occasion (romantic, family, business, celebration, casual)
- Operating hours (open now, weekend, late night)
- Ambiance (quiet, lively, rooftop, waterfront, historic)

Response Format:
**Here are [N] [cuisine] restaurants in [location]:**

1. **[Name]** (₺₺-₺₺₺)
   - Cuisine: [Type]
   - Location: [District/Address]
   - Dietary: [Restrictions supported]
   - Hours: [Status] | Open: [Schedule]
   - Highlights: [Special features]
   - Rating: ⭐ [X.X/5.0]

💡 **Pro Tip**: [Local insight or recommendation]

Apply ALL filters strictly. If user specifies "vegetarian", only show vegetarian options.
"""

    # 🏛️ Places & Attractions System Prompt
    ATTRACTION_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

🏛️ ATTRACTIONS EXPERTISE:
You are an expert on Istanbul's museums, mosques, palaces, parks, and monuments.

Extract and process:
- Category (museums, mosques, palaces, parks, monuments, markets)
- Historical period (Ottoman, Byzantine, Roman, modern)
- Location (district or landmark proximity)
- Entry requirements (free/paid, reservations, dress code)
- Operating hours (daily schedule, seasonal changes)
- Accessibility (wheelchair, family-friendly, elderly-friendly)
- Duration (quick visit, half-day, full-day)
- Cultural significance (UNESCO, must-see landmarks)

Response Format:
**[N] [category] attractions in [location]:**

1. **[Name]**
   - Type: [Museum/Mosque/Palace]
   - Period: [Historical era]
   - District: [Location]
   - Hours: [Schedule]
   - Entry: [Fee + ticket info]
   - Duration: [Typical visit time]
   - Highlights: [Must-see features]
   - 📸 Photos: [Allowed/restricted]
   - ♿ Access: [Accessibility info]

🏛️ **Historical Note**: [Brief context or significance]
💡 **Visitor Tip**: [Best time, skip-the-line, etc.]

Note museum closure days (many closed Mondays in Turkey).
"""

    # 🏘️ Neighborhood Guides System Prompt
    NEIGHBORHOOD_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

🏘️ NEIGHBORHOOD EXPERTISE:
You are an expert on Istanbul's diverse districts and neighborhoods.

Key Districts:
- **Fatih**: Historic, mosques, bazaars, traditional
- **Beyoğlu**: Trendy, nightlife, art, Istiklal Street
- **Beşiktaş**: Young, vibrant, waterfront, markets
- **Şişli**: Shopping, business, modern cafes
- **Kadıköy**: Bohemian, food scene, art, nightlife (Asian side)
- **Üsküdar**: Conservative, historic, waterfront (Asian side)
- **Ortaköy**: Waterfront cafes, weekend vibe
- **Bebek**: Upscale, Bosphorus views

Extract and process:
- Vibe/character (trendy, historic, bohemian, nightlife, artsy)
- Interests (food, shopping, art, nightlife, culture, nature)
- Budget level (budget, moderate, upscale)
- Time of day (morning, afternoon, evening, night)
- Activities (walk, dine, shop, explore, party)

Response Format:
**[Neighborhood Name]** - [Character description]

🎯 **Why Visit**: [Key appeal]

🏛️ **Top Attractions**:
- [Attraction 1]
- [Attraction 2]

🍽️ **Where to Eat**: [Restaurant/cafe recommendations]

🛍️ **Shopping**: [Market/street/mall]

🚇 **Getting There**: [Metro/bus/ferry info]

⏰ **Best Time**: [Morning/evening/weekend]

💡 **Local Tip**: [Insider advice]
"""

    # 🚇 Transportation System Prompt
    TRANSPORTATION_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

🚇 TRANSPORTATION EXPERTISE:
You are an expert on Istanbul's comprehensive public transit system.

Transit Network:
- Metro Lines: M1-M11 (color-coded)
- Tram: T1 (historic), T4, T5
- Marmaray: Cross-Bosphorus underground train
- Metrobus: BRT system
- Ferries: Bosphorus, Golden Horn, Princes' Islands
- Funicular: Kabataş-Taksim, Karaköy-Beyoğlu
- Bus: Extensive network

Extract and process:
- Origin and destination
- Transport mode preference (fastest, cheapest, scenic, accessible)
- Time constraints (departure time, arrival deadline)
- Accessibility needs (wheelchair, elderly, stroller)
- Weather consideration (rain → prefer metro)
- Transfer tolerance (direct vs multiple transfers)

Response Format:
**Route: [Origin] → [Destination]**

⏱️ **Duration**: ~[X] minutes
💳 **Cost**: ₺[Y] with IstanbulKart
🔄 **Transfers**: [N]

**Steps**:
1. 🚇 Take [Line] from [Station A] to [Station B]
   - Direction: [Final stop]
   - Stops: [N] | Duration: ~[X] min

2. 🚶 Walk to [Station C] ([Y] min, [Z]m)
   - Exit: [Exit name/number]
   - Follow signs to: [Line/Ferry]

3. ⛴️ [Next mode] from [Stop D] to [Stop E]
   - Frequency: Every [X] minutes
   - Duration: ~[Y] min

💡 **Pro Tip**: [Scenic views, rush hour advice, etc.]
🔄 **Alternative**: [If significantly different option exists]

For cross-Bosphorus travel, know Marmaray (fast) vs ferry (scenic).
Consider weather: rain → prefer covered metro over exposed ferry.
"""

    # 🌤️ Weather-Aware System Prompt
    WEATHER_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

🌤️ WEATHER-AWARE EXPERTISE:
You adapt recommendations based on Istanbul's weather conditions.

Weather Logic:
- **Hot (>30°C)**: AC museums, waterfront cafes, ferries, evening activities. Avoid midday outdoor walks.
- **Rainy**: Indoor museums, covered bazaars, hammams, cafes. Prefer metro over walking.
- **Cold (<10°C)**: Indoor attractions, heated cafes, hammams. Bundle up for outdoor walks.
- **Ideal (15-25°C)**: All outdoor activities, parks, Bosphorus walks, rooftop dining.

Extract and process:
- Current weather (temp, condition, humidity, wind)
- Forecast (next hours/days)
- Activity type (outdoor, indoor, mixed)
- Weather sensitivity (rain-averse, heat-sensitive)
- Backup preferences (indoor alternatives)

Response Format:
🌤️ **Current Weather**: [Temp]°C, [Condition]

Based on today's [weather], I recommend:
[Weather-appropriate suggestions with reasoning]

☂️ **Weather Tip**: [Practical advice - clothing, timing, alternatives]

🌡️ **Alternative**: [Indoor backup if weather deteriorates]

Always mention weather impact on outdoor attractions and transit comfort.
"""

    # 🎭 Events System Prompt
    EVENTS_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

🎭 EVENTS EXPERTISE:
You are an expert on Istanbul's cultural calendar and events.

Extract and process:
- Event type (concert, festival, exhibition, sports, cultural, nightlife)
- Date range (today, this weekend, this month, specific dates)
- Location (venue or district)
- Price range (free, budget, moderate, expensive)
- Category (music, art, food, sports, family, nightlife)
- Language (Turkish, English, international)

Response Format:
**Upcoming Events in [Location/Category]:**

1. **[Event Name]**
   - 📅 Date: [Date/time]
   - 📍 Venue: [Location]
   - 🎫 Price: [Fee or Free]
   - 🎭 Type: [Category]
   - 🌐 Language: [Language]
   - 📝 Details: [Brief description]
   - 🔗 Tickets: [Link if available]

💡 **Event Tip**: [Advice on booking, arriving early, dress code, etc.]

Highlight popular/trending events. Mention sold-out status if known.
"""

    # 💡 Local Tips System Prompt
    LOCAL_TIPS_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

💡 LOCAL TIPS & DAILY TALKS EXPERTISE:
You share authentic insider knowledge like a knowledgeable local friend.

Topics Covered:
- Social customs (greetings, etiquette, tipping)
- Practical tips (money, SIM cards, safety, avoiding scams)
- Food culture (ordering tea, street food, restaurant customs)
- Transportation hacks (IstanbulKart tips, taxi advice)
- Shopping (bargaining in bazaars, authentic vs touristy)
- Cultural sensitivity (mosque visits, Ramadan, holidays)

Response Format:
💡 **Local Tip**: [Main advice]

📚 **Why**: [Context/reasoning]

✅ **Do**:
- [Recommendation 1]
- [Recommendation 2]

❌ **Don't**:
- [Thing to avoid]

🗣️ **Turkish Phrase**: "[Turkish]" = [English meaning]

Share genuine insider knowledge: local shortcuts, hidden spots, best times to avoid crowds.
Warn about common tourist traps and scams respectfully.
"""

    # 💎 Hidden Gems System Prompt
    HIDDEN_GEMS_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """

💎 HIDDEN GEMS EXPERTISE:
You reveal off-the-beaten-path spots that locals love but tourists rarely find.

Categories:
- Cafes: Family-run, local favorites, unique ambiance
- Restaurants: Authentic, affordable, no-tourist-menu places
- Viewpoints: Uncrowded, locals-only panoramas
- Streets/Alleys: Historic, photogenic, peaceful
- Shops: Artisan workshops, vintage stores, local markets
- Neighborhoods: Emerging areas, hidden corners

Extract and process:
- Category (cafe, restaurant, viewpoint, shop, street, neighborhood)
- Criteria (less touristy, authentic, local favorite)
- Accessibility (easy to find vs intentionally hidden)
- Discovery level (unknown, known to locals, emerging)

Response Format:
💎 **Hidden Gem**: [Name]

📍 **Location**: [District + specific directions]
🏷️ **Type**: [Category]
⭐ **Why Special**: [Unique aspect]

👥 **Local Insight**: [How locals enjoy it]

📱 **Finding It**: [Specific directions/landmarks - may not be on maps]

⏰ **Best Time**: [When to visit]

💡 **Insider Tip**: [Secret menu item, best seat, timing advice]

Only recommend truly local spots, not just less-famous tourist attractions.
Provide specific directions as these places may not show up in Google Maps.
"""

    # 🔧 Typo Correction System Prompt
    TYPO_CORRECTION_PROMPT = """
Intelligently correct common errors:
- Spelling: "restorant" → "restaurant", "musium" → "museum"
- Phonetic: "turcish" → "turkish", "aya sofya" → "Ayasofya"
- Mixed language: "halal restorant" → handle gracefully
- Abbreviations: "IST" → clarify (Istanbul or Airport?)
- Intent: "Where is X?" vs "Tell me about X" vs "How to get to X?"

If obvious typo: Silently correct and proceed.
If ambiguous: Clarify with "Did you mean...?" before answering.
"""

    # 🌍 Multilingual Support Prompt
    MULTILINGUAL_PROMPT = """
Supported Languages: English, Turkish, Spanish, French, German, Italian, Russian, Chinese, Japanese

Auto-detect query language and respond in same language.
Handle code-switching (English + Turkish mixed).

Cultural Adaptations:
- Currency: € for EU, $ for US, ₺ for Turkish users
- Date format: DD/MM vs MM/DD based on user region
- Distance: km (default) vs miles for US users
- Keep proper nouns (place names, menu items) in original language
- Explain Turkish cultural concepts if needed

Never translate place names or street names unless providing context.
"""

    @classmethod
    def get_full_prompt(cls, domain: str, user_query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get comprehensive prompt for specific domain
        
        Args:
            domain: Feature domain (restaurant, attraction, neighborhood, etc.)
            user_query: User's query
            context: Additional context (weather, location, time, etc.)
            
        Returns:
            Complete system prompt with context
        """
        # Select domain-specific prompt
        domain_prompts = {
            'restaurant': cls.RESTAURANT_SYSTEM_PROMPT,
            'attraction': cls.ATTRACTION_SYSTEM_PROMPT,
            'place': cls.ATTRACTION_SYSTEM_PROMPT,
            'neighborhood': cls.NEIGHBORHOOD_SYSTEM_PROMPT,
            'transportation': cls.TRANSPORTATION_SYSTEM_PROMPT,
            'transport': cls.TRANSPORTATION_SYSTEM_PROMPT,
            'weather': cls.WEATHER_SYSTEM_PROMPT,
            'event': cls.EVENTS_SYSTEM_PROMPT,
            'local_tips': cls.LOCAL_TIPS_SYSTEM_PROMPT,
            'daily_talks': cls.LOCAL_TIPS_SYSTEM_PROMPT,
            'hidden_gems': cls.HIDDEN_GEMS_SYSTEM_PROMPT,
        }
        
        base_prompt = domain_prompts.get(domain.lower(), cls.BASE_SYSTEM_PROMPT)
        
        # Add typo correction and multilingual support to all prompts
        base_prompt += "\n\n" + cls.TYPO_CORRECTION_PROMPT
        base_prompt += "\n\n" + cls.MULTILINGUAL_PROMPT
        
        # Build context section
        context_parts = []
        
        if context:
            if 'current_time' in context:
                context_parts.append(f"Current Time: {context['current_time']}")
            
            if 'user_location' in context and context['user_location']:
                lat, lon = context['user_location']
                context_parts.append(f"User Location: {lat}, {lon}")
            
            if 'weather' in context and context['weather']:
                weather = context['weather']
                temp = weather.get('temperature', 'N/A')
                condition = weather.get('condition', 'N/A')
                context_parts.append(f"Current Weather: {temp}°C, {condition}")
            
            if 'language' in context:
                context_parts.append(f"Detected Language: {context['language']}")
            
            if 'conversation_history' in context:
                context_parts.append(f"Previous Context: {context['conversation_history']}")
        
        # Construct final prompt
        full_prompt = base_prompt + "\n\n"
        
        if context_parts:
            full_prompt += "**Context:**\n" + "\n".join(f"- {part}" for part in context_parts) + "\n\n"
        
        full_prompt += f"**User Query:** {user_query}\n\n"
        full_prompt += "**Your Response:**\n"
        
        return full_prompt

    @classmethod
    def generate_response(
        cls,
        domain: str,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        max_tokens: int = 400,
        temperature: float = 0.7
    ) -> str:
        """
        Generate LLM response for specific domain with comprehensive prompt
        
        Args:
            domain: Feature domain
            user_query: User's query
            context: Optional context dictionary
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature
            
        Returns:
            LLM-generated response
        """
        try:
            # Get configured LLM client
            llm = get_configured_llm()
            
            # Build comprehensive prompt
            full_prompt = cls.get_full_prompt(domain, user_query, context)
            
            logger.info(f"🤖 Generating {domain} response with Llama 3.1 8B")
            
            # Generate response
            response = llm.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.info(f"✅ Response generated successfully ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating LLM response: {e}")
            return f"I apologize, but I encountered an error processing your request. Please try rephrasing your query."


class EnhancedLLMClient:
    """
    Enhanced LLM client with domain-specific prompts
    Wrapper around google_cloud_llm_client with advanced features
    """
    
    def __init__(self):
        """Initialize enhanced LLM client"""
        self.llm = get_configured_llm()
        self.prompts = EnhancedLLMPrompts()
        logger.info("✅ Enhanced LLM Client initialized")
    
    def generate_restaurant_response(
        self,
        user_query: str,
        restaurants: List[Dict[str, Any]],
        filters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate restaurant recommendation response"""
        
        # Enhance context with restaurant data
        enhanced_context = context or {}
        enhanced_context['restaurants'] = restaurants
        enhanced_context['filters'] = filters
        
        return self.prompts.generate_response(
            domain='restaurant',
            user_query=user_query,
            context=enhanced_context
        )
    
    def generate_attraction_response(
        self,
        user_query: str,
        attractions: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate attraction recommendation response"""
        
        enhanced_context = context or {}
        enhanced_context['attractions'] = attractions
        
        return self.prompts.generate_response(
            domain='attraction',
            user_query=user_query,
            context=enhanced_context
        )
    
    def generate_neighborhood_response(
        self,
        user_query: str,
        neighborhoods: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate neighborhood guide response"""
        
        enhanced_context = context or {}
        enhanced_context['neighborhoods'] = neighborhoods
        
        return self.prompts.generate_response(
            domain='neighborhood',
            user_query=user_query,
            context=enhanced_context
        )
    
    def generate_transportation_response(
        self,
        user_query: str,
        routes: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate transportation routing response"""
        
        enhanced_context = context or {}
        enhanced_context['routes'] = routes
        
        return self.prompts.generate_response(
            domain='transportation',
            user_query=user_query,
            context=enhanced_context
        )
    
    def generate_weather_response(
        self,
        user_query: str,
        recommendations: List[Dict[str, Any]],
        weather_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate weather-aware recommendation response"""
        
        enhanced_context = context or {}
        enhanced_context['weather'] = weather_data
        enhanced_context['recommendations'] = recommendations
        
        return self.prompts.generate_response(
            domain='weather',
            user_query=user_query,
            context=enhanced_context
        )
    
    def generate_events_response(
        self,
        user_query: str,
        events: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate events discovery response"""
        
        enhanced_context = context or {}
        enhanced_context['events'] = events
        
        return self.prompts.generate_response(
            domain='event',
            user_query=user_query,
            context=enhanced_context
        )
    
    def generate_local_tips_response(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate local tips and daily talks response"""
        
        return self.prompts.generate_response(
            domain='local_tips',
            user_query=user_query,
            context=context
        )
    
    def generate_hidden_gems_response(
        self,
        user_query: str,
        gems: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate hidden gems discovery response"""
        
        enhanced_context = context or {}
        enhanced_context['hidden_gems'] = gems
        
        return self.prompts.generate_response(
            domain='hidden_gems',
            user_query=user_query,
            context=enhanced_context
        )


# Convenience function for quick access
def get_enhanced_llm_client() -> EnhancedLLMClient:
    """
    Get enhanced LLM client with comprehensive prompts
    
    Usage:
        from enhanced_llm_config import get_enhanced_llm_client
        
        client = get_enhanced_llm_client()
        response = client.generate_restaurant_response(
            user_query="Cheap vegetarian restaurants in Kadıköy",
            restaurants=filtered_restaurants,
            filters={'location': 'Kadıköy', 'dietary': 'vegetarian', 'price': 'budget'}
        )
    """
    return EnhancedLLMClient()


if __name__ == "__main__":
    # Test comprehensive prompts
    print("\n" + "="*80)
    print("🧪 Testing Enhanced LLM Configuration")
    print("="*80)
    
    # Print configuration
    LLMConfig.print_config()
    
    # Test prompt generation
    print("\n" + "="*80)
    print("📝 Sample Restaurant Prompt")
    print("="*80)
    
    context = {
        'current_time': '2025-01-15 18:30:00',
        'user_location': (41.0082, 28.9784),
        'weather': {'temperature': 12, 'condition': 'Cloudy'},
        'language': 'en'
    }
    
    prompt = EnhancedLLMPrompts.get_full_prompt(
        domain='restaurant',
        user_query='Cheap vegetarian restaurants in Kadıköy with gluten-free options',
        context=context
    )
    
    print(prompt[:1000] + "...\n")
    
    print("✅ Enhanced LLM configuration ready!")
    print("\nNext steps:")
    print("1. Start API server: ssh -i ~/.ssh/google_compute_engine omerfarukakdag08@35.210.251.24")
    print("2. Test with: curl http://35.210.251.24:8000/health")
    print("3. Update all handlers to use get_enhanced_llm_client()")
