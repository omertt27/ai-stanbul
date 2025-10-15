#!/usr/bin/env python3
"""
Daily Talks Integration Wrapper
===============================

This module integrates the Comprehensive Daily Talks System with the existing
A/ISTANBUL infrastructure, providing seamless backwards compatibility while
enabling advanced conversation features.

Features:
🔄 Seamless integration with existing backend endpoints
🎯 Enhanced intent recognition and response generation
💭 Context-aware conversation memory
🌤️ Weather and news integration
📱 Multi-modal response support
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

# Import the comprehensive system
try:
    from comprehensive_daily_talks_system import (
        ComprehensiveDailyTalksSystem, 
        get_daily_talks_system,
        process_daily_conversation,
        get_daily_greeting_simple,
        get_conversation_starters,
        IntentType,
        ConversationTone,
        MoodState,
        TimeContext
    )
    COMPREHENSIVE_SYSTEM_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Comprehensive Daily Talks System loaded successfully")
except ImportError as e:
    COMPREHENSIVE_SYSTEM_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Comprehensive Daily Talks System not available: {e}")

# Fallback imports for existing system compatibility
try:
    from services.daily_talk_enhancement import (
        get_daily_conversation,
        get_daily_greeting,
        get_mood_based_activities,
        DailyContext,
        MoodType,
        TimeOfDay
    )
    LEGACY_SYSTEM_AVAILABLE = True
    logger.info("✅ Legacy daily talk system available as fallback")
except ImportError as e:
    LEGACY_SYSTEM_AVAILABLE = False
    logger.warning(f"⚠️ Legacy daily talk system not available: {e}")

class DailyTalksIntegrationWrapper:
    """
    Integration wrapper that provides a unified interface for daily talks,
    using the comprehensive system when available and falling back gracefully.
    """
    
    def __init__(self):
        self.comprehensive_system_available = COMPREHENSIVE_SYSTEM_AVAILABLE
        self.legacy_system_available = LEGACY_SYSTEM_AVAILABLE
        self._comprehensive_system = None
        
        if self.comprehensive_system_available:
            logger.info("🎉 Initializing with Comprehensive Daily Talks System")
        elif self.legacy_system_available:
            logger.info("🔄 Falling back to Legacy Daily Talk System")
        else:
            logger.warning("⚠️ No daily talk systems available - using basic responses")
    
    async def _get_comprehensive_system(self):
        """Get comprehensive system instance (lazy loading)"""
        if self._comprehensive_system is None and self.comprehensive_system_available:
            self._comprehensive_system = await get_daily_talks_system()
        return self._comprehensive_system
    
    async def process_conversation(self, user_input: str, user_id: str, session_id: str = None) -> Dict[str, Any]:
        """
        Process conversation using the best available system
        """
        try:
            if self.comprehensive_system_available:
                # Use comprehensive system
                system = await self._get_comprehensive_system()
                response = await system.process_conversation(user_input, user_id, session_id)
                
                return {
                    "success": True,
                    "response": response.message,
                    "tone": response.tone.value,
                    "confidence": response.intent_confidence,
                    "suggested_actions": response.suggested_actions,
                    "follow_up_questions": response.follow_up_questions,
                    "multimedia_content": response.multimedia_content,
                    "personalization_data": response.personalization_data,
                    "system_used": "comprehensive",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif self.legacy_system_available:
                # Use legacy system
                conversation_data = get_daily_conversation(user_id, "Istanbul")
                
                return {
                    "success": True,
                    "response": conversation_data.get("greeting", "Hello! How can I help you explore Istanbul today?"),
                    "tone": "casual",
                    "confidence": 0.7,
                    "suggested_actions": ["Get recommendations", "Ask about weather", "Find restaurants"],
                    "follow_up_questions": ["What would you like to explore?"],
                    "multimedia_content": None,
                    "personalization_data": None,
                    "system_used": "legacy",
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                # Basic fallback
                return self._generate_basic_response(user_input, user_id)
                
        except Exception as e:
            logger.error(f"Error in conversation processing: {e}")
            return self._generate_error_response(str(e))
    
    async def get_daily_greeting(self, user_id: str, location: str = "Istanbul") -> Dict[str, Any]:
        """
        Get daily greeting using the best available system
        """
        try:
            if self.comprehensive_system_available:
                # Use comprehensive system
                system = await self._get_comprehensive_system()
                greeting_data = await system.get_daily_greeting(user_id, location)
                
                return {
                    "success": True,
                    "greeting": greeting_data["greeting"],
                    "weather_aware": True,
                    "weather_info": greeting_data["weather"],
                    "daily_highlights": greeting_data["daily_highlights"],
                    "seasonal_tip": greeting_data["seasonal_tip"],
                    "user_context": greeting_data["user_context"],
                    "system_used": "comprehensive",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif self.legacy_system_available:
                # Use legacy system
                greeting_data = get_daily_greeting(user_id, location)
                
                return {
                    "success": True,
                    "greeting": greeting_data.get("greeting", f"Merhaba! Welcome to {location}! 🏙️"),
                    "weather_aware": greeting_data.get("weather_aware", False),
                    "weather_info": greeting_data.get("weather", {}),
                    "daily_highlights": ["Explore Istanbul's culture!", "Try local cuisine!", "Discover hidden gems!"],
                    "seasonal_tip": "Every day in Istanbul is a new adventure!",
                    "user_context": {"return_visitor": False},
                    "system_used": "legacy",
                    "timestamp": datetime.now().isoformat()
                }
            
            else:
                # Basic fallback
                return {
                    "success": True,
                    "greeting": f"Merhaba! Welcome to {location}! 🏙️ Ready for today's adventure?",
                    "weather_aware": False,
                    "weather_info": {},
                    "daily_highlights": ["Explore the city!", "Try local food!", "Meet friendly locals!"],
                    "seasonal_tip": "Every day is perfect for exploration!",
                    "user_context": {"return_visitor": False},
                    "system_used": "basic",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting daily greeting: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_greeting": f"Merhaba! Welcome to {location}! 🏙️",
                "system_used": "error_fallback"
            }
    
    async def get_full_daily_conversation(self, user_id: str, location: str = "Istanbul") -> Dict[str, Any]:
        """
        Get full daily conversation data (for /api/v1/daily-conversation endpoint)
        """
        try:
            if self.comprehensive_system_available:
                # Use comprehensive system
                system = await self._get_comprehensive_system()
                
                # Get greeting
                greeting_data = await system.get_daily_greeting(user_id, location)
                
                # Get conversation suggestions
                suggestions = await system.get_conversation_suggestions(user_id)
                
                # Generate comprehensive response
                return {
                    "success": True,
                    "greeting": greeting_data["greeting"],
                    "recommendations": [
                        {
                            "type": "activity",
                            "title": "Weather-Perfect Activities",
                            "description": f"Tailored to today's {greeting_data['weather']['condition']} weather at {greeting_data['weather']['temperature']}°C",
                            "details": {"weather_optimized": True}
                        },
                        {
                            "type": "food",
                            "title": "Local Culinary Gems",
                            "description": "Authentic Istanbul flavors locals love",
                            "details": {"cuisine_variety": True}
                        },
                        {
                            "type": "culture",
                            "title": "Hidden Cultural Treasures",
                            "description": "Beyond the typical tourist sites",
                            "details": {"local_insights": True}
                        }
                    ],
                    "conversation_flow": suggestions,
                    "mood_suggestions": [
                        "🌟 Feeling excited? Let's plan an adventure!",
                        "😌 Need to relax? I know peaceful spots!",
                        "🤔 Curious about local life? I'll show you!",
                        "🍽️ Hungry for authentic flavors? Let's eat!"
                    ],
                    "local_tips": [
                        "🤫 Local secret: The best simit comes from street vendors, not cafes!",
                        "💰 Money tip: Get an Istanbulkart for all transport - much cheaper!",
                        "⏰ Timing tip: Visit popular spots early morning or late afternoon for fewer crowds",
                        "🗣️ Language tip: 'Teşekkür ederim' (teh-sheh-KOOR eh-deh-rim) means 'thank you'"
                    ],
                    "context": {
                        "time_of_day": greeting_data["user_context"]["time_context"],
                        "weather_condition": greeting_data["weather"]["condition"],
                        "temperature": greeting_data["weather"]["temperature"],
                        "user_location": location,
                        "user_mood": "curious",  # Default mood
                        "is_weekday": datetime.now().weekday() < 5,
                        "return_visitor": greeting_data["user_context"]["return_visitor"]
                    },
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "weather_aware": True,
                    "system_used": "comprehensive"
                }
            
            elif self.legacy_system_available:
                # Use legacy system
                conversation_data = get_daily_conversation(user_id, location)
                
                return {
                    "success": True,
                    "greeting": conversation_data.get("greeting", f"Merhaba! Welcome to {location}! 🏙️"),
                    "recommendations": conversation_data.get("recommendations", []),
                    "conversation_flow": conversation_data.get("conversation_flow", []),
                    "mood_suggestions": conversation_data.get("mood_suggestions", []),
                    "local_tips": conversation_data.get("local_tips", []),
                    "context": conversation_data.get("context", {}),
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "weather_aware": False,
                    "system_used": "legacy"
                }
            
            else:
                # Basic fallback
                return {
                    "success": True,
                    "greeting": f"Merhaba! Welcome to {location}! 🏙️",
                    "recommendations": [
                        {
                            "type": "activity",
                            "title": "Explore Historic Sites",
                            "description": "Visit Hagia Sophia, Blue Mosque, and Topkapi Palace"
                        },
                        {
                            "type": "food",
                            "title": "Try Turkish Cuisine",
                            "description": "Sample kebabs, baklava, and Turkish coffee"
                        },
                        {
                            "type": "transport",
                            "title": "Use Public Transport",
                            "description": "Get an Istanbulkart for metro, tram, and ferry"
                        }
                    ],
                    "conversation_flow": [
                        "What would you like to explore first?",
                        "Are you interested in history, food, or culture?",
                        "How many days will you be in Istanbul?"
                    ],
                    "mood_suggestions": [
                        "Feeling adventurous? Let's explore!",
                        "Want to relax? Try a Turkish bath!",
                        "Hungry? Let's find great food!",
                        "Curious? Let's discover hidden gems!"
                    ],
                    "local_tips": [
                        "Always negotiate taxi fares beforehand",
                        "Try street food - it's delicious and safe",
                        "Learn basic Turkish greetings",
                        "Respect local customs and dress codes"
                    ],
                    "context": {
                        "time_of_day": "unknown",
                        "weather_condition": "unknown",
                        "temperature": "unknown",
                        "user_location": location,
                        "user_mood": "curious",
                        "is_weekday": datetime.now().weekday() < 5
                    },
                    "location": location,
                    "timestamp": datetime.now().isoformat(),
                    "weather_aware": False,
                    "system_used": "basic"
                }
                
        except Exception as e:
            logger.error(f"Error getting full daily conversation: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_data": {
                    "greeting": f"Merhaba! Welcome to {location}! 🏙️",
                    "recommendations": [{"type": "activity", "title": "Explore Istanbul", "description": "Discover the magic of this amazing city!"}],
                    "conversation_flow": ["What would you like to explore today?"],
                    "local_tips": ["Istanbul is a city of endless discoveries!"]
                },
                "system_used": "error_fallback"
            }
    
    async def get_mood_based_activities(self, mood: str, location: str = "Istanbul", user_id: str = None) -> Dict[str, Any]:
        """
        Get mood-based activity recommendations
        """
        try:
            if self.comprehensive_system_available:
                # Use comprehensive system
                system = await self._get_comprehensive_system()
                
                # Convert mood string to MoodState
                mood_mapping = {
                    "excited": "excited",
                    "curious": "curious", 
                    "relaxed": "relaxed",
                    "tired": "tired",
                    "adventurous": "adventurous",
                    "romantic": "romantic",
                    "social": "social"
                }
                
                mood_context = mood_mapping.get(mood.lower(), "curious")
                
                # Generate mood-based response
                if user_id:
                    response = await system.process_conversation(
                        f"I'm feeling {mood} today, what do you recommend?", 
                        user_id
                    )
                    activities = [response.message]
                else:
                    # Generate static mood-based activities
                    activities = self._generate_mood_activities(mood_context, location)
                
                return {
                    "success": True,
                    "mood": mood,
                    "activities": activities,
                    "location": location,
                    "personalized": user_id is not None,
                    "system_used": "comprehensive",
                    "timestamp": datetime.now().isoformat()
                }
            
            elif self.legacy_system_available:
                # Use legacy system
                if user_id:
                    activities_data = get_mood_based_activities(user_id, mood, location)
                    return {
                        "success": True,
                        "mood": mood,
                        "activities": activities_data.get("activities", []),
                        "location": location,
                        "personalized": True,
                        "system_used": "legacy",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    activities = self._generate_mood_activities(mood, location)
                    return {
                        "success": True,
                        "mood": mood,
                        "activities": activities,
                        "location": location,
                        "personalized": False,
                        "system_used": "legacy",
                        "timestamp": datetime.now().isoformat()
                    }
            
            else:
                # Basic fallback
                activities = self._generate_mood_activities(mood, location)
                return {
                    "success": True,
                    "mood": mood,
                    "activities": activities,
                    "location": location,
                    "personalized": False,
                    "system_used": "basic",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting mood-based activities: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_activities": [f"Explore {location} and discover something new based on your {mood} mood!"],
                "system_used": "error_fallback"
            }
    
    def _generate_mood_activities(self, mood: str, location: str) -> List[str]:
        """Generate mood-based activities (fallback method)"""
        mood_activities = {
            "excited": [
                "🎢 Take a Bosphorus boat tour for incredible views!",
                "🏛️ Explore the vibrant Grand Bazaar for shopping adventures!",
                "🌃 Visit Galata Tower for panoramic city views!",
                "🎭 Check out live music venues in Beyoğlu!"
            ],
            "curious": [
                "🏛️ Discover the secrets of Topkapi Palace!",
                "📚 Explore the underground Basilica Cistern!",
                "🎨 Visit contemporary art galleries in Karaköy!",
                "🍵 Learn about Turkish coffee culture in traditional cafes!"
            ],
            "relaxed": [
                "🌸 Stroll through peaceful Gülhane Park!",
                "☕ Enjoy slow Turkish tea in a traditional tea house!",
                "🛁 Experience a relaxing Turkish bath (hamam)!",
                "🌊 Take a calm ferry ride along the Bosphorus!"
            ],
            "tired": [
                "🛋️ Find a cozy cafe in Sultanahmet for rest and people-watching!",
                "🌳 Relax in the tranquil gardens of Süleymaniye Mosque!",
                "☕ Enjoy a peaceful tea break with Golden Horn views!",
                "🚶 Take a gentle walk along the historic city walls!"
            ],
            "adventurous": [
                "🏃 Explore the hidden neighborhoods of Balat and Fener!",
                "🚇 Navigate the city using only public transport!",
                "🍽️ Try street food from vendors in Eminönü!",
                "🌉 Walk across both European and Asian sides in one day!"
            ],
            "romantic": [
                "🌅 Watch sunset from Pierre Loti Hill!",
                "🍷 Dine at a rooftop restaurant with Bosphorus views!",
                "🚢 Take an evening Bosphorus cruise!",
                "🌹 Walk hand-in-hand through Ortaköy's charming streets!"
            ],
            "social": [
                "🍻 Experience Istanbul's vibrant nightlife in Beyoğlu!",
                "🎪 Join locals at a traditional meyhane for food and music!",
                "🛍️ Shop and socialize in the bustling Istiklal Street!",
                "☕ Meet people at social cafes in Kadıköy!"
            ]
        }
        
        return mood_activities.get(mood.lower(), [
            f"Explore {location} and discover something that matches your {mood} mood!",
            "Ask locals for recommendations - they always know the best spots!",
            "Every corner of Istanbul has something special to offer!"
        ])
    
    def _generate_basic_response(self, user_input: str, user_id: str) -> Dict[str, Any]:
        """Generate basic response when no advanced systems are available"""
        responses = [
            "Merhaba! Welcome to Istanbul! 🏙️ How can I help you explore this amazing city?",
            "Hello! I'm here to help you discover the best of Istanbul! What are you interested in?",
            "Hi there! Ready to explore where Europe meets Asia? What would you like to know about Istanbul?",
            "Welcome! Istanbul has so much to offer - culture, food, history, and more! What sounds interesting to you?"
        ]
        
        import random
        response = random.choice(responses)
        
        return {
            "success": True,
            "response": response,
            "tone": "casual",
            "confidence": 0.5,
            "suggested_actions": ["Ask about attractions", "Find restaurants", "Get transport help"],
            "follow_up_questions": ["What brings you to Istanbul?", "What are you most excited to explore?"],
            "multimedia_content": None,
            "personalization_data": None,
            "system_used": "basic",
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "success": False,
            "error": error_message,
            "fallback_response": "I'm sorry, I had a technical issue! But I'm here and ready to help you explore Istanbul. What would you like to discover? 🏙️",
            "suggested_actions": ["Try asking again", "Ask about restaurants", "Get local tips"],
            "system_used": "error_fallback",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get integration system status"""
        return {
            "comprehensive_system": "✅ Available" if self.comprehensive_system_available else "❌ Not Available",
            "legacy_system": "✅ Available" if self.legacy_system_available else "❌ Not Available",
            "active_system": (
                "comprehensive" if self.comprehensive_system_available 
                else "legacy" if self.legacy_system_available 
                else "basic"
            ),
            "features": {
                "advanced_intent_recognition": self.comprehensive_system_available,
                "context_memory": self.comprehensive_system_available,
                "weather_integration": self.comprehensive_system_available,
                "mood_based_responses": True,
                "personalization": self.comprehensive_system_available or self.legacy_system_available
            },
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# GLOBAL INTEGRATION INSTANCE
# =============================================================================

# Global wrapper instance for easy access
_integration_wrapper = None

def get_daily_talks_integration() -> DailyTalksIntegrationWrapper:
    """Get global daily talks integration wrapper"""
    global _integration_wrapper
    if _integration_wrapper is None:
        _integration_wrapper = DailyTalksIntegrationWrapper()
    return _integration_wrapper

# =============================================================================
# CONVENIENCE FUNCTIONS FOR BACKEND INTEGRATION
# =============================================================================

async def integrated_daily_conversation(user_input: str, user_id: str, session_id: str = None) -> Dict[str, Any]:
    """Process daily conversation with integrated system"""
    wrapper = get_daily_talks_integration()
    return await wrapper.process_conversation(user_input, user_id, session_id)

async def integrated_daily_greeting(user_id: str, location: str = "Istanbul") -> Dict[str, Any]:
    """Get daily greeting with integrated system"""
    wrapper = get_daily_talks_integration()
    return await wrapper.get_daily_greeting(user_id, location)

async def integrated_full_conversation(user_id: str, location: str = "Istanbul") -> Dict[str, Any]:
    """Get full daily conversation with integrated system"""
    wrapper = get_daily_talks_integration()
    return await wrapper.get_full_daily_conversation(user_id, location)

async def integrated_mood_activities(mood: str, location: str = "Istanbul", user_id: str = None) -> Dict[str, Any]:
    """Get mood-based activities with integrated system"""
    wrapper = get_daily_talks_integration()
    return await wrapper.get_mood_based_activities(mood, location, user_id)

def get_integration_status() -> Dict[str, Any]:
    """Get integration system status"""
    wrapper = get_daily_talks_integration()
    return wrapper.get_system_status()

# =============================================================================
# MAIN EXECUTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    async def test_integration():
        """Test the integration wrapper"""
        print("🧪 Testing Daily Talks Integration Wrapper")
        print("=" * 60)
        
        wrapper = DailyTalksIntegrationWrapper()
        
        # Test system status
        print("\n1️⃣ System Status:")
        status = wrapper.get_system_status()
        for key, value in status.items():
            if key != "timestamp":
                print(f"   {key}: {value}")
        
        # Test daily greeting
        print("\n2️⃣ Testing Daily Greeting:")
        greeting = await wrapper.get_daily_greeting("test_user")
        print(f"✅ Success: {greeting['success']}")
        print(f"🌅 Greeting: {greeting['greeting'][:80]}...")
        print(f"🎯 System Used: {greeting['system_used']}")
        
        # Test conversation processing
        print("\n3️⃣ Testing Conversation Processing:")
        test_inputs = [
            "Hello! I'm excited to explore Istanbul!",
            "I'm looking for great restaurants",
            "I'm feeling a bit tired today"
        ]
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n📝 Test {i}: '{user_input}'")
            response = await wrapper.process_conversation(user_input, "test_user")
            print(f"✅ Success: {response['success']}")
            print(f"💬 Response: {response['response'][:60]}...")
            print(f"🎯 System Used: {response['system_used']}")
        
        # Test full daily conversation
        print("\n4️⃣ Testing Full Daily Conversation:")
        full_conv = await wrapper.get_full_daily_conversation("test_user")
        print(f"✅ Success: {full_conv['success']}")
        print(f"📊 Recommendations: {len(full_conv['recommendations'])}")
        print(f"💭 Conversation flows: {len(full_conv['conversation_flow'])}")
        print(f"🎯 System Used: {full_conv['system_used']}")
        
        # Test mood-based activities
        print("\n5️⃣ Testing Mood-Based Activities:")
        moods = ["excited", "relaxed", "curious"]
        for mood in moods:
            activities = await wrapper.get_mood_based_activities(mood, user_id="test_user")
            print(f"😊 {mood.title()}: {len(activities['activities'])} activities")
            print(f"   System Used: {activities['system_used']}")
        
        print("\n🎉 Integration testing completed successfully!")
        print("✅ Daily Talks Integration Wrapper is ready for production!")
    
    # Run integration tests
    asyncio.run(test_integration())
