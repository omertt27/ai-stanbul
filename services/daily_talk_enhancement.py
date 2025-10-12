#!/usr/bin/env python3
"""
Daily Talk Enhancement System for A/ISTANBUL
Provides personalized, weather-aware daily conversations
"""

import os
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Try to import weather services
try:
    from services.weather_cache_service import get_weather_for_ai, weather_cache
    from services.route_cache import get_weather_aware_route_recommendations, get_transportation_advice_for_weather
    WEATHER_SERVICES_AVAILABLE = True
except ImportError:
    WEATHER_SERVICES_AVAILABLE = False

# Try to import advanced daily talk AI
try:
    from services.advanced_daily_talk_ai import process_advanced_daily_talk, advanced_daily_talk_ai
    ADVANCED_AI_AVAILABLE = True
except ImportError:
    ADVANCED_AI_AVAILABLE = False

class MoodType(Enum):
    ENERGETIC = "energetic"
    RELAXED = "relaxed"
    CURIOUS = "curious"
    TIRED = "tired"
    EXCITED = "excited"
    CONTEMPLATIVE = "contemplative"

class TimeOfDay(Enum):
    EARLY_MORNING = "early_morning"  # 6-9 AM
    MORNING = "morning"              # 9-12 PM
    AFTERNOON = "afternoon"          # 12-17 PM
    EVENING = "evening"             # 17-21 PM
    NIGHT = "night"                 # 21-6 AM

@dataclass
class DailyContext:
    """Context for daily conversations"""
    time_of_day: TimeOfDay
    weather_condition: str
    temperature: float
    user_location: str
    user_mood: MoodType
    is_weekday: bool
    special_events: List[str]
    last_conversation_topic: Optional[str]

class DailyTalkPersonality:
    """Local Istanbul friend personality for daily conversations"""
    
    def __init__(self):
        self.personality_traits = {
            "friendly": 0.9,
            "local": 0.95,
            "humorous": 0.7,
            "caring": 0.85,
            "knowledgeable": 0.9
        }
        
        # Istanbul-specific expressions and slang
        self.istanbul_expressions = {
            "positive_weather": [
                "Hava bomba bugün! 🌟 (Weather's amazing today!)",
                "Perfect day for a Bosphorus walk! 🌊",
                "The kind of weather that makes you love Istanbul even more! ☀️"
            ],
            "rainy_weather": [
                "Classic Istanbul rain 🌧️ - perfect for cozy cafe time!",
                "Rain means fewer crowds at the museums! 🏛️",
                "Time for some covered bazaar exploration! ☂️"
            ],
            "hot_weather": [
                "İstanbul yazı! (Istanbul summer!) Stay cool, my friend! 🌡️",
                "Ferry breeze weather - Bosphorus is calling! ⛴️",
                "Perfect for those shaded Çamlıca tea gardens! 🍃"
            ],
            "cold_weather": [
                "Jacket weather in the city! Perfect for Turkish tea! ☕",
                "Cozy hamam weather - time to warm up traditionally! 🛁",
                "Great day for museum hopping and hot soup! 🏛️"
            ]
        }
        
        # Daily conversation starters based on time and weather
        self.conversation_starters = {
            TimeOfDay.EARLY_MORNING: {
                "sunny": [
                    "Günaydın! ☀️ Early bird catching the best of Istanbul today? The morning light on the Bosphorus is magical right now!",
                    "Perfect morning for a walk along Galata Bridge! Coffee and sunrise - what more do you need? ☕🌅",
                    "Up early like a true İstanbullu! The city's waking up beautifully - want some morning adventure ideas?"
                ],
                "rainy": [
                    "Good morning! 🌧️ Cozy rainy start - perfect excuse for a warm breakfast at a neighborhood pastane!",
                    "Rain's creating that special Istanbul mood... Time for covered Grand Bazaar exploration? ☂️",
                    "Rainy mornings = best tea time! Know any good spots for a proper Turkish breakfast?"
                ],
                "cloudy": [
                    "Morning! 🌤️ Cloudy but cozy - great photography light for the city today!",
                    "Perfect morning for wandering without the harsh sun. The old city looks mysterious in this light!",
                    "Soft morning light = perfect for those Instagram shots of Sultanahmet! 📸"
                ]
            },
            TimeOfDay.MORNING: {
                "sunny": [
                    "What a gorgeous İstanbul morning! ☀️ The kind of day that makes you want to explore every neighborhood!",
                    "Sun's shining on the Golden Horn - perfect for a ferry ride or waterfront walk! ⛴️",
                    "Beautiful morning! Feeling the urge to discover some hidden gems today? 💎"
                ],
                "rainy": [
                    "Rain's painting the city in soft colors today 🎨 Perfect museum or covered market weather!",
                    "İstanbul rain has its own charm! Great day for discovering cozy indoor treasures ☔",
                    "Rainy morning = fewer tourists, more authentic experiences! Want some insider tips?"
                ]
            },
            TimeOfDay.AFTERNOON: {
                "sunny": [
                    "Perfect afternoon for a Bosphorus view lunch! 🍽️ Sun's hitting all the right spots today!",
                    "Sunshine's calling for rooftop terraces and sea views! Know what you're craving? 🌞",
                    "Golden hour approaching - time to position yourself for those epic sunset shots! 📸"
                ],
                "hot": [
                    "Afternoon heat = perfect excuse for ferry air conditioning! 🚢 Plus those water views... 💙",
                    "Too hot for walking? Let's find you some shaded courtyards and cool museums! 🏛️",
                    "İstanbul summer afternoon! Time for waterfront dining with a breeze? 🌊"
                ]
            },
            TimeOfDay.EVENING: {
                "clear": [
                    "İyi akşamlar! 🌆 Perfect evening light hitting the city - golden hour magic time!",
                    "Evening's here and Istanbul's getting ready to sparkle! ✨ What's your vibe tonight?",
                    "The city's putting on its evening show - ready to join the magic? 🎭"
                ],
                "rainy": [
                    "Cozy rainy evening perfect for traditional meyhane atmosphere! 🍷",
                    "Rain creates the most romantic city mood... Perfect for covered dining! 💕",
                    "Evening rain = authentic İstanbul vibes! Great night for warm interiors and local conversation 🏮"
                ]
            },
            TimeOfDay.NIGHT: {
                "clear": [
                    "İstanbul nights are legendary! ✨ The city lights reflecting on the Bosphorus... magical!",
                    "Night owls unite! The city's energy is just getting started! 🌙",
                    "Perfect night for waterfront strolls and late-night discoveries! 🌃"
                ],
                "any": [
                    "İstanbul gecesi! (Istanbul night!) The city has a different soul after dark... 🌙",
                    "Night time = local time! Ready for some authentic after-dark experiences? 🌃"
                ]
            }
        }
    
    def get_daily_greeting(self, context: DailyContext) -> str:
        """Generate personalized daily greeting based on context"""
        
        # Determine weather category for conversation starter
        weather_category = self._categorize_weather(context.weather_condition, context.temperature)
        
        # Get base greeting from time and weather
        time_greetings = self.conversation_starters.get(context.time_of_day, {})
        
        if weather_category in time_greetings:
            greetings = time_greetings[weather_category]
        elif "any" in time_greetings:
            greetings = time_greetings["any"]
        else:
            greetings = ["Hey there! Ready for another İstanbul adventure? 🏙️"]
        
        base_greeting = random.choice(greetings)
        
        # Add weather-specific enhancement
        weather_enhancement = self._get_weather_enhancement(context)
        if weather_enhancement:
            base_greeting += f"\n\n{weather_enhancement}"
        
        return base_greeting
    
    def _categorize_weather(self, condition: str, temp: float) -> str:
        """Categorize weather for conversation selection"""
        condition_lower = condition.lower()
        
        if 'rain' in condition_lower or 'drizzle' in condition_lower:
            return "rainy"
        elif temp > 28:
            return "hot"
        elif temp < 10:
            return "cold"
        elif 'cloud' in condition_lower:
            return "cloudy"
        else:
            return "sunny"
    
    def _get_weather_enhancement(self, context: DailyContext) -> Optional[str]:
        """Get weather-specific conversation enhancement"""
        weather_category = self._categorize_weather(context.weather_condition, context.temperature)
        
        if weather_category in self.istanbul_expressions:
            return random.choice(self.istanbul_expressions[weather_category])
        
        return None

class DailyTalkEnhancementSystem:
    """Enhanced daily talk system with weather awareness and local personality"""
    
    def __init__(self):
        self.personality = DailyTalkPersonality()
        self.user_daily_states = {}  # Track daily mood and context per user
    
    def generate_daily_conversation(self, user_id: str, user_location: str = "Istanbul") -> Dict[str, Any]:
        """Generate daily conversation with weather-aware context"""
        
        # Get current context
        context = self._build_daily_context(user_location)
        
        # Generate personalized greeting
        greeting = self.personality.get_daily_greeting(context)
        
        # Get weather-aware recommendations
        recommendations = self._get_daily_recommendations(context)
        
        # Generate conversation flow
        conversation_flow = self._build_conversation_flow(context, recommendations)
        
        return {
            "greeting": greeting,
            "context": context,
            "recommendations": recommendations,
            "conversation_flow": conversation_flow,
            "mood_suggestions": self._get_mood_suggestions(context),
            "local_tips": self._get_daily_local_tips(context)
        }
    
    def _build_daily_context(self, location: str) -> DailyContext:
        """Build daily context with weather awareness"""
        
        # Determine time of day
        current_hour = datetime.now().hour
        if 6 <= current_hour < 9:
            time_of_day = TimeOfDay.EARLY_MORNING
        elif 9 <= current_hour < 12:
            time_of_day = TimeOfDay.MORNING
        elif 12 <= current_hour < 17:
            time_of_day = TimeOfDay.AFTERNOON
        elif 17 <= current_hour < 21:
            time_of_day = TimeOfDay.EVENING
        else:
            time_of_day = TimeOfDay.NIGHT
        
        # Get weather info
        weather_condition = "Clear"
        temperature = 20.0
        
        if WEATHER_SERVICES_AVAILABLE:
            try:
                weather_data = get_weather_for_ai()
                if weather_data and 'condition' in weather_data:
                    weather_condition = weather_data['condition']
                    temperature = weather_data.get('temperature', 20.0)
            except Exception:
                pass  # Use defaults
        
        # Determine user mood based on time and weather
        user_mood = self._infer_mood_from_context(time_of_day, weather_condition, temperature)
        
        return DailyContext(
            time_of_day=time_of_day,
            weather_condition=weather_condition,
            temperature=temperature,
            user_location=location,
            user_mood=user_mood,
            is_weekday=datetime.now().weekday() < 5,
            special_events=[],  # Could integrate with events API
            last_conversation_topic=None
        )
    
    def _infer_mood_from_context(self, time_of_day: TimeOfDay, weather: str, temp: float) -> MoodType:
        """Infer likely user mood from environmental context"""
        
        if time_of_day == TimeOfDay.EARLY_MORNING:
            return MoodType.ENERGETIC
        elif 'rain' in weather.lower():
            return MoodType.CONTEMPLATIVE
        elif temp > 25:
            return MoodType.RELAXED
        elif temp < 10:
            return MoodType.TIRED
        else:
            return MoodType.CURIOUS
    
    def _get_daily_recommendations(self, context: DailyContext) -> List[Dict[str, Any]]:
        """Get weather-aware daily recommendations"""
        recommendations = []
        
        if WEATHER_SERVICES_AVAILABLE:
            try:
                # Get weather-aware route recommendations
                route_recs = get_weather_aware_route_recommendations(
                    context.user_location.lower(), 'balanced'
                )
                
                if route_recs and route_recs.get('weather_recommendations'):
                    recommendations.extend([
                        {
                            "type": "weather_activity",
                            "title": "Perfect for today's weather",
                            "description": rec,
                            "priority": "high"
                        } for rec in route_recs['weather_recommendations'][:3]
                    ])
                
                # Get transportation advice
                transport_advice = get_transportation_advice_for_weather()
                if isinstance(transport_advice, dict) and 'recommended_routes' in transport_advice:
                    recommendations.append({
                        "type": "transportation",
                        "title": "Best way to get around today",
                        "description": transport_advice['weather_impact'],
                        "details": transport_advice['recommended_routes'][0] if transport_advice['recommended_routes'] else None,
                        "priority": "medium"
                    })
                
            except Exception as e:
                pass  # Fallback to generic recommendations
        
        # Add time-specific recommendations
        if context.time_of_day == TimeOfDay.MORNING:
            recommendations.append({
                "type": "activity",
                "title": "Perfect morning activity",
                "description": "Start your day with a walk across Galata Bridge - watch the city wake up!",
                "priority": "medium"
            })
        elif context.time_of_day == TimeOfDay.EVENING:
            recommendations.append({
                "type": "activity", 
                "title": "Evening magic",
                "description": "Sunset from Çamlıca Hill or a waterfront dinner in Ortaköy",
                "priority": "high"
            })
        
        return recommendations
    
    def _build_conversation_flow(self, context: DailyContext, recommendations: List[Dict]) -> List[str]:
        """Build natural conversation flow options"""
        
        flows = []
        
        # Weather-based conversation starters
        if 'rain' in context.weather_condition.lower():
            flows.append("Perfect day to discover İstanbul's covered treasures - Grand Bazaar or cozy cafes?")
        elif context.temperature > 25:
            flows.append("Hot day calls for waterfront breezes - ferry ride or seaside dining?")
        else:
            flows.append("Great day for exploring - feeling more like culture, food, or hidden gems?")
        
        # Time-based suggestions
        if context.time_of_day == TimeOfDay.MORNING:
            flows.append("Morning energy! Want breakfast recommendations or activity ideas?")
        elif context.time_of_day == TimeOfDay.EVENING:
            flows.append("Evening vibes - dinner spots, cultural events, or nightlife?")
        
        # Mood-based options
        if context.user_mood == MoodType.CURIOUS:
            flows.append("Feeling curious? I know some amazing hidden spots locals love!")
        elif context.user_mood == MoodType.RELAXED:
            flows.append("Relaxed mood = perfect for peaceful parks or waterfront lounging")
        
        return flows
    
    def _get_mood_suggestions(self, context: DailyContext) -> List[str]:
        """Get mood-appropriate activity suggestions"""
        
        mood_activities = {
            MoodType.ENERGETIC: [
                "Hill climbing in Üsküdar for panoramic views",
                "Walking the entire length of İstiklal Street",
                "Ferry hopping between Asian and European sides"
            ],
            MoodType.RELAXED: [
                "Tea garden session with Bosphorus views",
                "Peaceful stroll through Gülhane Park",
                "Slow-paced exploration of Balat's colorful streets"
            ],
            MoodType.CURIOUS: [
                "Hidden courtyards in the Grand Bazaar",
                "Secret rooftop terraces with city views",
                "Local artisan workshops in Karaköy"
            ],
            MoodType.CONTEMPLATIVE: [
                "Quiet moments at Süleymaniye Mosque",
                "Reflective walk along the Golden Horn",
                "Museum browsing at less crowded hours"
            ]
        }
        
        return mood_activities.get(context.user_mood, ["Great day for any İstanbul adventure!"])
    
    def _get_daily_local_tips(self, context: DailyContext) -> List[str]:
        """Get daily local tips and insights"""
        
        tips = []
        
        # Weather-specific tips
        if 'rain' in context.weather_condition.lower():
            tips.extend([
                "Locals know: Grand Bazaar stays dry and has amazing energy on rainy days",
                "Rainy day secret: Underground Basilica Cistern is always the perfect temperature",
                "Pro tip: Covered Spice Bazaar is bustling and aromatic when it rains"
            ])
        elif context.temperature > 28:
            tips.extend([
                "Local wisdom: Ferries have AC and amazing views - best way to stay cool",
                "İstanbullu secret: Çamlıca Hill gets a great breeze even on hot days",
                "Insider tip: Shaded courtyards of old mosques offer cool refuge"
            ])
        
        # Time-specific tips
        if context.time_of_day == TimeOfDay.EARLY_MORNING:
            tips.append("Early morning secret: Galata Bridge fishermen are most active now - grab tea and watch!")
        elif context.time_of_day == TimeOfDay.EVENING:
            tips.append("Evening magic: Traditional Turkish music often starts around 8 PM in Beyoğlu")
        
        # Day-specific tips
        if context.is_weekday:
            tips.append("Weekday advantage: Museums and attractions are less crowded!")
        else:
            tips.append("Weekend vibes: Perfect time for longer explorations and leisurely meals")
        
        return tips[:3]  # Return top 3 tips

# Global instance
daily_talk_system = DailyTalkEnhancementSystem()

def get_advanced_daily_conversation(user_input: str, user_id: str, location: str = "Istanbul") -> Dict[str, Any]:
    """Get advanced AI-powered daily conversation"""
    if ADVANCED_AI_AVAILABLE:
        # Get weather context for advanced AI
        weather_context = None
        if WEATHER_SERVICES_AVAILABLE:
            try:
                weather_data = get_weather_for_ai()
                if weather_data and 'condition' in weather_data:
                    weather_context = {
                        "condition": weather_data['condition'],
                        "temperature": weather_data.get('temperature', 20.0),
                        "description": weather_data.get('description', '')
                    }
            except Exception as e:
                pass  # Use without weather context
        
        # Process with advanced AI
        result = process_advanced_daily_talk(user_input, user_id, weather_context)
        
        # Enhance with traditional daily talk elements
        traditional_conversation = get_daily_conversation(user_id, location)
        
        # Combine advanced AI response with traditional elements
        enhanced_result = {
            "response": result["response"],
            "analysis": result["analysis"],
            "conversation_state": result["conversation_state"],
            "personalization_level": result["personalization_level"],
            "emotional_state": result["emotional_state"],
            "suggestions": result["suggestions"],
            "local_tips": traditional_conversation["local_tips"],
            "weather_aware": WEATHER_SERVICES_AVAILABLE,
            "ai_level": "advanced"
        }
        
        return enhanced_result
    else:
        # Fallback to traditional system
        traditional_result = get_daily_conversation(user_id, location)
        return {
            "response": traditional_result["greeting"],
            "recommendations": traditional_result["recommendations"],
            "conversation_flow": traditional_result["conversation_flow"],
            "local_tips": traditional_result["local_tips"],
            "ai_level": "traditional"
        }

def get_daily_conversation(user_id: str, location: str = "Istanbul") -> Dict[str, Any]:
    """Get daily conversation for user"""
    return daily_talk_system.generate_daily_conversation(user_id, location)

def get_weather_aware_daily_greeting(user_id: str, location: str = "Istanbul") -> str:
    """Get weather-aware daily greeting"""
    conversation = get_daily_conversation(user_id, location)
    return conversation["greeting"]

if __name__ == "__main__":
    # Demo the daily talk system
    print("🗣️ A/ISTANBUL Daily Talk System Demo")
    print("=" * 50)
    
    conversation = get_daily_conversation("demo_user")
    
    print("🌅 DAILY GREETING:")
    print(conversation["greeting"])
    print()
    
    print("🎯 PERSONALIZED RECOMMENDATIONS:")
    for rec in conversation["recommendations"][:3]:
        print(f"• {rec['title']}: {rec['description']}")
    print()
    
    print("💬 CONVERSATION FLOW OPTIONS:")
    for flow in conversation["conversation_flow"]:
        print(f"• {flow}")
    print()
    
    print("💡 LOCAL TIPS:")
    for tip in conversation["local_tips"]:
        print(f"• {tip}")
    print()
    
    print("🎉 Daily Talk System Ready!")
