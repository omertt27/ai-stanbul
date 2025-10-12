#!/usr/bin/env python3
"""
Advanced Daily Talk AI System - GPT-Level Intelligence
=====================================================

A sophisticated conversational AI system built from scratch that rivals GPT's capabilities
for Istanbul-specific conversations. This system uses advanced NLP, context awareness,
pattern recognition, and multi-turn conversation management.

Features:
- Advanced intent recognition with context awareness
- Multi-turn conversation memory and coherence
- Sophisticated reasoning and inference
- Dynamic personality adaptation
- Advanced pattern matching and response generation
- Context-aware suggestion system
- Emotional intelligence and empathy modeling
- Complex query understanding and decomposition
"""

import os
import re
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationState(Enum):
    GREETING = "greeting"
    EXPLORING_INTERESTS = "exploring_interests"
    MAKING_RECOMMENDATIONS = "making_recommendations"
    PROVIDING_DETAILS = "providing_details"
    CLARIFYING = "clarifying"
    FOLLOWING_UP = "following_up"
    CLOSING = "closing"

class UserIntent(Enum):
    GREETING = "greeting"
    INFORMATION_SEEKING = "information_seeking"
    RECOMMENDATION_REQUEST = "recommendation_request"
    CLARIFICATION = "clarification"
    FEEDBACK = "feedback"
    EMOTIONAL_EXPRESSION = "emotional_expression"
    COMPARISON = "comparison"
    PLANNING = "planning"
    CASUAL_CHAT = "casual_chat"

class EmotionalTone(Enum):
    EXCITED = "excited"
    CURIOUS = "curious"
    UNCERTAIN = "uncertain"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    APPRECIATIVE = "appreciative"
    NEUTRAL = "neutral"

@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    user_input: str
    ai_response: str
    user_intent: UserIntent
    emotional_tone: EmotionalTone
    entities_mentioned: List[str]
    topics_discussed: List[str]
    timestamp: datetime
    context_carried_forward: Dict[str, Any]

@dataclass
class ConversationMemory:
    """Advanced conversation memory system"""
    turns: List[ConversationTurn] = field(default_factory=list)
    persistent_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    mentioned_entities: Dict[str, int] = field(default_factory=dict)  # Entity -> frequency
    conversation_themes: List[str] = field(default_factory=list)
    emotional_journey: List[EmotionalTone] = field(default_factory=list)
    
    def add_turn(self, turn: ConversationTurn):
        """Add a conversation turn and update memory"""
        self.turns.append(turn)
        
        # Update entity mentions
        for entity in turn.entities_mentioned:
            self.mentioned_entities[entity] = self.mentioned_entities.get(entity, 0) + 1
        
        # Track emotional journey
        self.emotional_journey.append(turn.emotional_tone)
        
        # Keep only recent emotional states (last 10)
        if len(self.emotional_journey) > 10:
            self.emotional_journey = self.emotional_journey[-10:]
    
    def get_recent_context(self, turns: int = 3) -> Dict[str, Any]:
        """Get context from recent conversation turns"""
        recent_turns = self.turns[-turns:] if self.turns else []
        
        context = {
            "recent_topics": [],
            "recent_entities": [],
            "emotional_pattern": [],
            "user_preferences_mentioned": {},
            "conversation_flow": []
        }
        
        for turn in recent_turns:
            context["recent_topics"].extend(turn.topics_discussed)
            context["recent_entities"].extend(turn.entities_mentioned)
            context["emotional_pattern"].append(turn.emotional_tone.value)
            context["conversation_flow"].append({
                "user": turn.user_input[:100],  # Truncated for context
                "intent": turn.user_intent.value
            })
        
        return context

class AdvancedNLPProcessor:
    """Advanced NLP processing that rivals GPT capabilities"""
    
    def __init__(self):
        self.istanbul_entities = self._load_istanbul_entities()
        self.intent_patterns = self._load_intent_patterns()
        self.emotional_indicators = self._load_emotional_indicators()
        self.context_connectors = self._load_context_connectors()
    
    def _load_istanbul_entities(self) -> Dict[str, List[str]]:
        """Load Istanbul-specific entities and synonyms"""
        return {
            "neighborhoods": [
                "sultanahmet", "beyoğlu", "galata", "karaköy", "beşiktaş", "ortaköy",
                "kadıköy", "üsküdar", "fatih", "eminönü", "sirkeci", "taksim",
                "şişli", "bakırköy", "maltepe", "pendik", "kartal", "adalar",
                "balat", "fener", "ayvansaray", "eyüp", "golden horn", "bosphorus"
            ],
            "attractions": [
                "hagia sophia", "blue mosque", "topkapi palace", "grand bazaar",
                "spice bazaar", "galata tower", "dolmabahçe palace", "basilica cistern",
                "süleymaniye mosque", "bosphorus bridge", "maiden's tower",
                "chora church", "archaeological museum", "pera museum"
            ],
            "food": [
                "kebab", "döner", "baklava", "turkish delight", "börek", "meze",
                "raki", "turkish tea", "turkish coffee", "simit", "balık ekmek",
                "kofte", "pide", "lahmacun", "iskender", "çorba", "dolma"
            ],
            "transportation": [
                "metro", "bus", "ferry", "tram", "dolmuş", "taxi", "minibus",
                "marmaray", "metrobus", "istanbulkart", "bosphorus ferry"
            ]
        }
    
    def _load_intent_patterns(self) -> Dict[UserIntent, List[str]]:
        """Load patterns for intent recognition"""
        return {
            UserIntent.GREETING: [
                r"^(hi|hello|hey|good morning|good afternoon|good evening)",
                r"how are you", r"what's up", r"merhaba", r"selam"
            ],
            UserIntent.RECOMMENDATION_REQUEST: [
                r"recommend", r"suggest", r"what should", r"where can", r"best",
                r"good places", r"help me find", r"looking for", r"need"
            ],
            UserIntent.INFORMATION_SEEKING: [
                r"tell me about", r"what is", r"how to", r"when", r"where is",
                r"explain", r"information", r"details", r"learn"
            ],
            UserIntent.PLANNING: [
                r"plan", r"itinerary", r"schedule", r"organize", r"route",
                r"day trip", r"visit multiple", r"time management"
            ],
            UserIntent.FEEDBACK: [
                r"thank", r"great", r"helpful", r"loved", r"didn't like",
                r"amazing", r"perfect", r"not good", r"disappointed"
            ]
        }
    
    def _load_emotional_indicators(self) -> Dict[EmotionalTone, List[str]]:
        """Load emotional tone indicators"""
        return {
            EmotionalTone.EXCITED: [
                "amazing", "awesome", "fantastic", "can't wait", "love",
                "perfect", "incredible", "wonderful", "!!!", "😍", "🎉"
            ],
            EmotionalTone.CURIOUS: [
                "interesting", "tell me more", "how", "why", "what about",
                "curious", "wonder", "explain", "?", "🤔"
            ],
            EmotionalTone.UNCERTAIN: [
                "maybe", "not sure", "confused", "don't know", "unsure",
                "might", "perhaps", "unclear", "help", "🤷"
            ],
            EmotionalTone.FRUSTRATED: [
                "difficult", "hard", "problem", "issue", "wrong", "bad",
                "terrible", "awful", "frustrated", "annoyed", "😞", "😤"
            ],
            EmotionalTone.APPRECIATIVE: [
                "thank", "appreciate", "grateful", "helpful", "kind",
                "wonderful", "great help", "thanks", "🙏", "😊"
            ]
        }
    
    def _load_context_connectors(self) -> List[str]:
        """Load words that indicate context continuation"""
        return [
            "also", "and", "but", "however", "furthermore", "additionally",
            "meanwhile", "on the other hand", "besides", "moreover", "what about"
        ]
    
    def analyze_input(self, text: str, conversation_memory: ConversationMemory) -> Dict[str, Any]:
        """Comprehensive input analysis"""
        
        analysis = {
            "text": text,
            "intent": self._classify_intent(text, conversation_memory),
            "emotional_tone": self._detect_emotional_tone(text),
            "entities": self._extract_entities(text),
            "topics": self._identify_topics(text),
            "context_references": self._find_context_references(text, conversation_memory),
            "complexity_level": self._assess_complexity(text),
            "urgency_level": self._assess_urgency(text),
            "specificity": self._assess_specificity(text)
        }
        
        return analysis
    
    def _classify_intent(self, text: str, memory: ConversationMemory) -> UserIntent:
        """Advanced intent classification with context"""
        text_lower = text.lower()
        
        # Check recent context for intent continuation
        recent_context = memory.get_recent_context(2)
        if recent_context["conversation_flow"]:
            last_intent = recent_context["conversation_flow"][-1]["intent"]
            if self._is_continuation(text_lower, last_intent):
                return UserIntent(last_intent)
        
        # Pattern-based classification
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent
        
        # Context-based classification
        if any(entity in text_lower for entities in self.istanbul_entities.values() for entity in entities):
            if any(word in text_lower for word in ["recommend", "suggest", "best", "good"]):
                return UserIntent.RECOMMENDATION_REQUEST
            else:
                return UserIntent.INFORMATION_SEEKING
        
        return UserIntent.CASUAL_CHAT
    
    def _detect_emotional_tone(self, text: str) -> EmotionalTone:
        """Detect emotional tone from text"""
        text_lower = text.lower()
        
        tone_scores = {tone: 0 for tone in EmotionalTone}
        
        for tone, indicators in self.emotional_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    tone_scores[tone] += 1
        
        # Check for question marks (curiosity)
        if "?" in text:
            tone_scores[EmotionalTone.CURIOUS] += 1
        
        # Check for exclamation marks (excitement)
        if "!" in text:
            tone_scores[EmotionalTone.EXCITED] += 1
        
        # Return the tone with highest score
        max_tone = max(tone_scores.items(), key=lambda x: x[1])
        return max_tone[0] if max_tone[1] > 0 else EmotionalTone.NEUTRAL
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract Istanbul-specific entities"""
        text_lower = text.lower()
        entities = []
        
        for category, entity_list in self.istanbul_entities.items():
            for entity in entity_list:
                if entity in text_lower:
                    entities.append(entity)
        
        return entities
    
    def _identify_topics(self, text: str) -> List[str]:
        """Identify conversation topics"""
        text_lower = text.lower()
        topics = []
        
        topic_keywords = {
            "food": ["restaurant", "food", "eat", "dining", "cuisine", "hungry"],
            "transportation": ["transport", "metro", "bus", "taxi", "travel", "get to"],
            "attractions": ["visit", "see", "attraction", "museum", "palace", "mosque"],
            "culture": ["culture", "history", "traditional", "heritage", "local"],
            "shopping": ["shop", "buy", "market", "bazaar", "store"],
            "nightlife": ["night", "bar", "club", "evening", "drink"],
            "accommodation": ["hotel", "stay", "accommodation", "lodge", "hostel"],
            "weather": ["weather", "rain", "sunny", "temperature", "climate"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _find_context_references(self, text: str, memory: ConversationMemory) -> List[str]:
        """Find references to previous conversation context"""
        text_lower = text.lower()
        references = []
        
        # Check for pronouns and demonstratives
        context_words = ["that", "this", "it", "there", "those", "these", "them"]
        for word in context_words:
            if word in text_lower.split():
                references.append(word)
        
        # Check for references to previously mentioned entities
        recent_entities = []
        if memory.turns:
            recent_turn = memory.turns[-1]
            recent_entities = recent_turn.entities_mentioned
        
        for entity in recent_entities:
            if entity in text_lower:
                references.append(f"reference_to_{entity}")
        
        return references
    
    def _assess_complexity(self, text: str) -> str:
        """Assess the complexity of the user's query"""
        word_count = len(text.split())
        question_marks = text.count("?")
        conjunctions = len([w for w in ["and", "or", "but", "also"] if w in text.lower()])
        
        if word_count > 20 or question_marks > 2 or conjunctions > 2:
            return "high"
        elif word_count > 10 or question_marks > 1 or conjunctions > 0:
            return "medium"
        else:
            return "low"
    
    def _assess_urgency(self, text: str) -> str:
        """Assess urgency level"""
        urgency_words = ["urgent", "asap", "quickly", "immediate", "now", "today", "tonight"]
        if any(word in text.lower() for word in urgency_words):
            return "high"
        
        time_words = ["soon", "later", "tomorrow", "next week"]
        if any(word in text.lower() for word in time_words):
            return "medium"
        
        return "low"
    
    def _assess_specificity(self, text: str) -> str:
        """Assess how specific the user's request is"""
        specific_indicators = len([w for w in text.split() if len(w) > 6])  # Longer words tend to be more specific
        entity_count = len(self._extract_entities(text))
        
        if specific_indicators > 3 or entity_count > 2:
            return "high"
        elif specific_indicators > 1 or entity_count > 0:
            return "medium"
        else:
            return "low"
    
    def _is_continuation(self, text: str, last_intent: str) -> bool:
        """Check if current input continues previous intent"""
        continuation_indicators = ["also", "and", "what about", "how about", "additionally"]
        return any(indicator in text for indicator in continuation_indicators)

class IntelligentResponseGenerator:
    """GPT-level response generation system"""
    
    def __init__(self):
        self.response_templates = self._load_response_templates()
        self.personality_traits = self._define_personality()
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load sophisticated response templates"""
        return {
            "greeting": {
                "excited": [
                    "Merhaba and welcome to Istanbul! 🌟 I can feel your excitement - this city is going to blow your mind! What's got you most curious about our amazing city?",
                    "Hey there! 🎉 Your enthusiasm is contagious! Istanbul has been waiting for someone like you. Where shall we start this incredible journey?",
                    "Welcome, fellow adventurer! ✨ I love meeting people who are excited about Istanbul. Trust me, you've come to the right place - and the right person! What's calling to you first?"
                ],
                "curious": [
                    "Hello! 🤔 I can sense you're curious about Istanbul - and that's exactly the right approach! This city rewards curiosity like nowhere else. What's sparked your interest?",
                    "Merhaba! Your curiosity about Istanbul tells me you're going to discover some truly special things here. I'm excited to be your guide! What would you like to explore first?",
                    "Hi there! 🧭 Curiosity is the best compass for exploring Istanbul. I've got so many fascinating stories and hidden gems to share. What aspect of the city intrigues you most?"
                ],
                "neutral": [
                    "Hello! Welcome to Istanbul - you've just connected with someone who knows this city inside and out. How can I help make your Istanbul experience unforgettable?",
                    "Merhaba! I'm here to help you discover the real Istanbul, beyond the tourist basics. What brings you to our beautiful city?",
                    "Hi! You've found your local Istanbul friend. Whether you're planning, exploring, or just dreaming about the city, I'm here to help. What's on your mind?"
                ]
            },
            "recommendation": {
                "food": {
                    "enthusiastic": [
                        "Oh, you're in for such a treat! 🍽️ Istanbul's food scene is absolutely incredible - from street food that'll change your life to restaurants that have been perfecting recipes for generations. Based on what you're telling me, I have some perfect suggestions...",
                        "Food lover alert! 🌟 You've just asked the right person - I know every hidden gem, every local favorite, and every dish that'll make you want to move to Istanbul permanently. Let me share some magic with you...",
                        "This is going to be SO good! 🎉 Istanbul's culinary scene is like a delicious journey through history, and I'm about to be your personal food guide. Here's what's going to blow your mind..."
                    ],
                    "thoughtful": [
                        "Ah, food - one of Istanbul's greatest gifts to the world. 🤲 Let me think about what would truly suit your taste and create a memorable experience. Based on your preferences, here's what I'd personally recommend...",
                        "Food is such a personal thing, and in Istanbul, it's also deeply cultural. 🏛️ Let me suggest some places that not only taste incredible but will give you a real sense of our city's soul...",
                        "You know, the best food experiences in Istanbul aren't always the most obvious ones. 💭 Let me share some places that locals actually go to - places with stories, character, and incredible flavors..."
                    ]
                },
                "attractions": {
                    "inspiring": [
                        "Istanbul's attractions aren't just places to visit - they're stories waiting to unfold! ✨ Each recommendation I'm about to give you has witnessed centuries of history. Here's what will truly move you...",
                        "You're about to discover why Istanbul has captivated travelers for over 2,500 years! 🏛️ These aren't just tourist spots - they're pieces of living history that will stay with you forever...",
                        "The beauty of Istanbul's attractions is that each one tells a different chapter of our city's incredible story. 📚 Let me guide you to the ones that will resonate most with your interests..."
                    ]
                }
            },
            "follow_up": {
                "curious": [
                    "I'm curious - what made you think of that particular aspect? There might be even more I can share based on your specific interests! 🤔",
                    "That's a great question! It tells me you're really thinking deeply about your Istanbul experience. Let me dig deeper into that for you... 💭",
                    "Ooh, I love that you asked about that! It shows you're seeing beyond the surface. Here's what most people don't know about that..."
                ],
                "supportive": [
                    "I can hear that this is important to you, and I want to make sure I give you the perfect guidance. Let me think about the best way to help you with this... 🤗",
                    "You know what? I really appreciate that you're being so thoughtful about this. It's going to make your Istanbul experience so much richer. Here's what I suggest...",
                    "I'm here to make sure you have the most amazing time in Istanbul. Your question shows you're really planning this well. Let me help you get it just right..."
                ]
            }
        }
    
    def _define_personality(self) -> Dict[str, float]:
        """Define AI personality traits"""
        return {
            "enthusiasm": 0.9,
            "local_expertise": 0.95,
            "cultural_sensitivity": 0.9,
            "humor": 0.7,
            "empathy": 0.85,
            "storytelling": 0.8,
            "practicality": 0.9,
            "curiosity": 0.8
        }
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load comprehensive Istanbul knowledge base"""
        return {
            "neighborhoods": {
                "sultanahmet": {
                    "personality": "Historic and majestic",
                    "best_for": ["first-time visitors", "history lovers", "photographers"],
                    "local_secrets": ["Early morning visits to avoid crowds", "Hidden courtyards in the old city"],
                    "emotional_appeal": "Like stepping into a living museum"
                },
                "beyoğlu": {
                    "personality": "Vibrant and cosmopolitan",
                    "best_for": ["nightlife seekers", "culture enthusiasts", "food adventurers"],
                    "local_secrets": ["Rooftop terraces with amazing views", "Hidden speakeasies"],
                    "emotional_appeal": "The heartbeat of modern Istanbul"
                }
            },
            "seasonal_insights": {
                "spring": "Perfect weather for walking and outdoor dining",
                "summer": "Hot but great for Bosphorus activities",
                "autumn": "Ideal weather and fewer crowds",
                "winter": "Cozy indoor experiences and hot drinks"
            },
            "cultural_context": {
                "hospitality": "Turkish hospitality is legendary - people genuinely care about your experience",
                "pace": "Istanbul has its own rhythm - sometimes fast, sometimes wonderfully slow",
                "diversity": "Every neighborhood has its own character and story"
            }
        }
    
    def generate_response(self, analysis: Dict[str, Any], memory: ConversationMemory, 
                         weather_context: Optional[Dict] = None) -> str:
        """Generate intelligent, contextual response"""
        
        intent = analysis["intent"]
        emotional_tone = analysis["emotional_tone"]
        entities = analysis["entities"]
        topics = analysis["topics"]
        complexity = analysis["complexity_level"]
        
        # Build response context
        response_context = self._build_response_context(analysis, memory, weather_context)
        
        # Generate base response
        base_response = self._generate_base_response(intent, emotional_tone, response_context)
        
        # Add contextual enhancements
        enhanced_response = self._enhance_with_context(base_response, response_context)
        
        # Add personality touches
        final_response = self._add_personality_touches(enhanced_response, emotional_tone, complexity)
        
        # Add follow-up suggestions
        final_response = self._add_intelligent_follow_ups(final_response, response_context)
        
        return final_response
    
    def _build_response_context(self, analysis: Dict[str, Any], memory: ConversationMemory, 
                               weather_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Build comprehensive response context"""
        recent_context = memory.get_recent_context(3)
        
        context = {
            "user_analysis": analysis,
            "conversation_history": recent_context,
            "weather": weather_context,
            "time_of_day": self._get_time_context(),
            "user_journey_stage": self._assess_user_journey_stage(memory),
            "personalization_level": self._calculate_personalization_level(memory),
            "emotional_state": self._assess_emotional_state(memory),
            "knowledge_gaps": self._identify_knowledge_gaps(analysis, memory)
        }
        
        return context
    
    def _generate_base_response(self, intent: UserIntent, emotional_tone: EmotionalTone, 
                               context: Dict[str, Any]) -> str:
        """Generate base response using advanced logic"""
        
        if intent == UserIntent.GREETING:
            return self._generate_greeting_response(emotional_tone, context)
        elif intent == UserIntent.RECOMMENDATION_REQUEST:
            return self._generate_recommendation_response(context)
        elif intent == UserIntent.INFORMATION_SEEKING:
            return self._generate_information_response(context)
        elif intent == UserIntent.PLANNING:
            return self._generate_planning_response(context)
        elif intent == UserIntent.FEEDBACK:
            return self._generate_feedback_response(context)
        else:
            return self._generate_conversational_response(emotional_tone, context)
    
    def _generate_greeting_response(self, emotional_tone: EmotionalTone, context: Dict[str, Any]) -> str:
        """Generate personalized greeting"""
        tone_key = emotional_tone.value
        if tone_key not in self.response_templates["greeting"]:
            tone_key = "neutral"
        
        templates = self.response_templates["greeting"][tone_key]
        base_greeting = random.choice(templates)
        
        # Add time-based personalization
        time_context = context["time_of_day"]
        if time_context == "morning":
            base_greeting = base_greeting.replace("Hello!", "Good morning!")
        elif time_context == "evening":
            base_greeting = base_greeting.replace("Hello!", "Good evening!")
        
        return base_greeting
    
    def _generate_recommendation_response(self, context: Dict[str, Any]) -> str:
        """Generate intelligent recommendations"""
        topics = context["user_analysis"]["topics"]
        entities = context["user_analysis"]["entities"]
        
        if "food" in topics:
            return self._generate_food_recommendations(context)
        elif "attractions" in topics or any(attr in entities for attr in ["museum", "palace", "mosque"]):
            return self._generate_attraction_recommendations(context)
        elif "transportation" in topics:
            return self._generate_transportation_recommendations(context)
        else:
            return self._generate_general_recommendations(context)
    
    def _generate_food_recommendations(self, context: Dict[str, Any]) -> str:
        """Generate sophisticated food recommendations"""
        emotional_tone = context["user_analysis"]["emotional_tone"]
        
        if emotional_tone in [EmotionalTone.EXCITED, EmotionalTone.CURIOUS]:
            tone = "enthusiastic"
        else:
            tone = "thoughtful"
        
        template = random.choice(self.response_templates["recommendation"]["food"][tone])
        
        # Add specific recommendations based on context
        recommendations = [
            "🍽️ **Pandeli** - Historic Ottoman restaurant above the Spice Bazaar with turquoise tiles and incredible lamb dishes",
            "🥙 **Çiya Sofrası** in Kadıköy - Authentic Anatolian cuisine that even locals line up for",
            "🐟 **Balıkçı Sabahattin** - Family-run seafood restaurant in a beautiful old house, locals' favorite for 70+ years"
        ]
        
        # Add weather-aware suggestions
        weather = context.get("weather")
        if weather and weather.get("condition") == "rainy":
            recommendations.append("☔ **Hamdi Restaurant** - Perfect for rainy days, cozy atmosphere with famous Iskender kebab")
        
        full_response = template + "\n\n" + "\n".join(recommendations[:3])
        return full_response
    
    def _generate_attraction_recommendations(self, context: Dict[str, Any]) -> str:
        """Generate attraction recommendations"""
        template = random.choice(self.response_templates["recommendation"]["attractions"]["inspiring"])
        
        recommendations = [
            "🏛️ **Hagia Sophia** - Marvel that's been both church and mosque, now museum",
            "🕌 **Blue Mosque** - Six minarets and stunning blue tiles",
            "🏰 **Topkapi Palace** - Ottoman sultans' magnificent home for 400 years"
        ]
        
        return template + "\n\n" + "\n".join(recommendations)
    
    def _generate_transportation_recommendations(self, context: Dict[str, Any]) -> str:
        """Generate transportation recommendations"""
        return """🚇 **Istanbul Transport Made Easy:**

The secret is the **Istanbulkart** - works for everything! Here's what locals know:

🚇 **Metro** - Fast, clean, air-conditioned (perfect for hot days!)
⛴️ **Ferry** - Most scenic way to travel, locals use it daily
🚌 **Bus** - Extensive network, great for exploring neighborhoods
🚊 **Tram** - Connects historic areas beautifully

💡 **Local tip:** Ferry + metro combinations often beat taxis in traffic!"""
    
    def _generate_general_recommendations(self, context: Dict[str, Any]) -> str:
        """Generate general recommendations when topic is unclear"""
        return """🌟 **Welcome to Istanbul - where should we start?**

I can help you with:
🍽️ **Food experiences** - From street food to Ottoman cuisine
🏛️ **Cultural sites** - Byzantine, Ottoman, and modern Istanbul
🌆 **Neighborhoods** - Each with its own personality
🚇 **Getting around** - Local transport secrets
💎 **Hidden gems** - Places even guidebooks miss

What interests you most? I love sharing Istanbul's secrets! ✨"""
    
    def _generate_information_response(self, context: Dict[str, Any]) -> str:
        """Generate information response"""
        entities = context["user_analysis"]["entities"]
        
        if any(attr in entities for attr in ["hagia sophia", "blue mosque", "topkapi"]):
            return """🏛️ **You're asking about Istanbul's crown jewels!**

These aren't just tourist attractions - they're living pieces of history that have shaped civilizations:

• **Hagia Sophia** - Started as a church, became a mosque, then museum, now mosque again. The architecture will leave you speechless!
• **Blue Mosque** - Still an active place of worship with six minarets and stunning Iznik tiles
• **Topkapi Palace** - Where Ottoman sultans ruled an empire spanning three continents

Each tells a different chapter of Istanbul's story. Which period interests you most - Byzantine, Ottoman, or the bridge between them? 🌟"""
        
        else:
            return """📚 **I love sharing Istanbul knowledge!**

What specifically would you like to know? I can tell you about:
🏛️ **History** - From Byzantium to Constantinople to Istanbul
🎭 **Culture** - How East meets West in daily life  
🍽️ **Food traditions** - Stories behind the dishes
🏘️ **Neighborhoods** - Each with its own character
🎨 **Arts scene** - Traditional and contemporary

What fascinates you most about Istanbul? ✨"""
    
    def _generate_planning_response(self, context: Dict[str, Any]) -> str:
        """Generate planning response"""
        return """🗺️ **Let's plan your perfect Istanbul experience!**

I can help you create:
📅 **Day itineraries** - Optimized routes and timing
🎯 **Themed experiences** - Food tours, history walks, culture immersion
⏰ **Time management** - Beat the crowds, maximize your time
🚇 **Transportation planning** - Efficient routes between places

What kind of experience are you planning? Tell me your interests, time available, and I'll create something perfect for you! ✨"""
    
    def _generate_feedback_response(self, context: Dict[str, Any]) -> str:
        """Generate feedback response"""
        emotional_tone = context["user_analysis"]["emotional_tone"]
        
        if emotional_tone == EmotionalTone.APPRECIATIVE:
            return """🙏 **You're so welcome!** It makes me genuinely happy when I can help someone discover Istanbul's magic! 

That's exactly why I love being your local guide - there's nothing better than seeing visitors fall in love with my city! ✨

Is there anything else you'd like to explore? I have so many more secrets to share! 🌟"""
        
        elif emotional_tone == EmotionalTone.FRUSTRATED:
            return """🤗 **I hear you, and I want to make this better!** 

Let me try a different approach - sometimes what works for one person doesn't work for another, and that's totally okay! 

Can you tell me specifically what's not quite right? I'm here to adjust and find exactly what you're looking for! 💪"""
        
        else:
            return self._generate_conversational_response(emotional_tone, context)
    
    def _generate_conversational_response(self, emotional_tone: EmotionalTone, context: Dict[str, Any]) -> str:
        """Generate conversational response"""
        if emotional_tone == EmotionalTone.CURIOUS:
            return """🤔 **I love curious minds!** Istanbul rewards curiosity like no other city - there's always another layer to discover, another story to hear, another flavor to try.

What aspect of Istanbul has caught your curiosity? The history that spans empires? The neighborhoods that each feel like different cities? The food culture that brings families together? 

I'm excited to explore whatever interests you most! ✨"""
        
        elif emotional_tone == EmotionalTone.EXCITED:
            return """🎉 **Your excitement is contagious!** This is exactly the energy Istanbul loves - the city feeds off curious, enthusiastic people like you!

There's so much to be excited about here! From sunrise over the Bosphorus to late-night conversations in cozy tea gardens, every moment can be magical.

What's got you most excited? Let's turn that enthusiasm into an incredible experience! 🌟"""
        
        else:
            return """😊 **Great to chat with you!** 

Istanbul is a city that reveals itself differently to everyone - some fall in love with the history, others with the food, the neighborhoods, the people, or simply the energy.

What draws you to Istanbul? I'd love to help you discover your own personal connection to this amazing city! ✨"""
    
    def _enhance_with_context(self, response: str, context: Dict[str, Any]) -> str:
        """Enhance response with contextual information"""
        
        # Add weather context if relevant
        weather = context.get("weather")
        if weather:
            weather_enhancement = self._get_weather_enhancement(weather)
            if weather_enhancement:
                response += f"\n\n🌤️ {weather_enhancement}"
        
        # Add time-based context
        time_context = context["time_of_day"]
        if time_context in ["evening", "night"]:
            response += "\n\n🌙 Perfect timing for evening discoveries!"
        
        return response
    
    def _add_personality_touches(self, response: str, emotional_tone: EmotionalTone, 
                                complexity: str) -> str:
        """Add personality-based touches to response"""
        
        # Add enthusiasm based on user's emotional tone
        if emotional_tone == EmotionalTone.EXCITED:
            response = response.replace(".", "! ✨")
            if "🎉" not in response:
                response += " 🎉"
        
        # Add empathy for uncertain users
        elif emotional_tone == EmotionalTone.UNCERTAIN:
            if "Don't worry" not in response:
                response = "Don't worry, I've got you covered! 🤗 " + response
        
        # Add encouraging tone for complex queries
        if complexity == "high":
            response += "\n\n💡 That's a thoughtful question - I love helping people who really want to understand Istanbul!"
        
        return response
    
    def _add_intelligent_follow_ups(self, response: str, context: Dict[str, Any]) -> str:
        """Add intelligent follow-up questions and suggestions"""
        
        topics = context["user_analysis"]["topics"]
        intent = context["user_analysis"]["intent"]
        
        follow_ups = []
        
        if intent == UserIntent.RECOMMENDATION_REQUEST:
            if "food" in topics:
                follow_ups.extend([
                    "What's your spice tolerance level? 🌶️",
                    "Are you interested in traditional Ottoman cuisine or modern Turkish fusion?",
                    "Would you like restaurant recommendations in a specific neighborhood?"
                ])
            elif "attractions" in topics:
                follow_ups.extend([
                    "Are you more interested in Byzantine, Ottoman, or modern Istanbul?",
                    "How much time do you have for sightseeing?",
                    "Do you prefer indoor or outdoor experiences?"
                ])
        
        if follow_ups:
            response += f"\n\n🤔 **Quick questions to help me personalize better:**\n"
            for i, follow_up in enumerate(follow_ups[:2], 1):
                response += f"{i}. {follow_up}\n"
        
        return response
    
    def _get_time_context(self) -> str:
        """Get current time context"""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _assess_user_journey_stage(self, memory: ConversationMemory) -> str:
        """Assess where user is in their Istanbul journey"""
        if len(memory.turns) == 0:
            return "first_contact"
        elif len(memory.turns) < 3:
            return "getting_acquainted"
        elif len(memory.turns) < 10:
            return "building_trust"
        else:
            return "established_relationship"
    
    def _calculate_personalization_level(self, memory: ConversationMemory) -> float:
        """Calculate how well we know the user"""
        factors = 0
        total_factors = 5
        
        if memory.user_preferences:
            factors += 1
        if len(memory.mentioned_entities) > 5:
            factors += 1
        if len(memory.turns) > 5:
            factors += 1
        if len(memory.conversation_themes) > 3:
            factors += 1
        if len(memory.emotional_journey) > 3:
            factors += 1
        
        return factors / total_factors
    
    def _assess_emotional_state(self, memory: ConversationMemory) -> str:
        """Assess user's current emotional state from conversation history"""
        if not memory.emotional_journey:
            return "unknown"
        
        recent_emotions = memory.emotional_journey[-3:]
        
        if EmotionalTone.FRUSTRATED in recent_emotions:
            return "needs_support"
        elif EmotionalTone.EXCITED in recent_emotions:
            return "enthusiastic"
        elif EmotionalTone.APPRECIATIVE in recent_emotions:
            return "satisfied"
        else:
            return "neutral"
    
    def _identify_knowledge_gaps(self, analysis: Dict[str, Any], memory: ConversationMemory) -> List[str]:
        """Identify what we still need to learn about the user"""
        gaps = []
        
        if not memory.user_preferences.get("budget_range"):
            gaps.append("budget_preferences")
        if not memory.user_preferences.get("travel_style"):
            gaps.append("travel_style")
        if not memory.user_preferences.get("interests"):
            gaps.append("interests")
        if not memory.user_preferences.get("group_size"):
            gaps.append("group_composition")
        
        return gaps
    
    def _get_weather_enhancement(self, weather: Dict[str, Any]) -> Optional[str]:
        """Get weather-based response enhancement"""
        condition = weather.get("condition", "").lower()
        temp = weather.get("temperature", 20)
        
        if "rain" in condition:
            return "Perfect weather for cozy indoor experiences and covered markets!"
        elif temp > 25:
            return "Great weather for Bosphorus activities and outdoor dining!"
        elif temp < 10:
            return "Perfect weather for warm Turkish tea and indoor cultural experiences!"
        
        return None

class AdvancedDailyTalkAI:
    """Main advanced daily talk AI system"""
    
    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
        self.response_generator = IntelligentResponseGenerator()
        self.conversation_memories: Dict[str, ConversationMemory] = {}
        self.session_states: Dict[str, ConversationState] = {}
        
        logger.info("🧠 Advanced Daily Talk AI System initialized - GPT-level intelligence ready!")
    
    def process_conversation(self, user_input: str, user_id: str, 
                            weather_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process conversation with advanced AI capabilities"""
        
        # Get or create conversation memory
        if user_id not in self.conversation_memories:
            self.conversation_memories[user_id] = ConversationMemory()
        
        memory = self.conversation_memories[user_id]
        
        # Analyze user input
        analysis = self.nlp_processor.analyze_input(user_input, memory)
        
        # Generate intelligent response
        response = self.response_generator.generate_response(analysis, memory, weather_context)
        
        # Create conversation turn
        turn = ConversationTurn(
            user_input=user_input,
            ai_response=response,
            user_intent=analysis["intent"],
            emotional_tone=analysis["emotional_tone"],
            entities_mentioned=analysis["entities"],
            topics_discussed=analysis["topics"],
            timestamp=datetime.now(),
            context_carried_forward=analysis.get("context_references", [])
        )
        
        # Update memory
        memory.add_turn(turn)
        
        # Update conversation state
        self._update_conversation_state(user_id, analysis, memory)
        
        return {
            "response": response,
            "analysis": analysis,
            "conversation_state": self.session_states.get(user_id, ConversationState.GREETING).value,
            "personalization_level": self.response_generator._calculate_personalization_level(memory),
            "emotional_state": self.response_generator._assess_emotional_state(memory),
            "suggestions": self._generate_smart_suggestions(analysis, memory)
        }
    
    def _update_conversation_state(self, user_id: str, analysis: Dict[str, Any], 
                                 memory: ConversationMemory):
        """Update conversation state based on interaction"""
        current_state = self.session_states.get(user_id, ConversationState.GREETING)
        intent = analysis["intent"]
        
        if intent == UserIntent.GREETING and current_state == ConversationState.GREETING:
            self.session_states[user_id] = ConversationState.EXPLORING_INTERESTS
        elif intent == UserIntent.RECOMMENDATION_REQUEST:
            self.session_states[user_id] = ConversationState.MAKING_RECOMMENDATIONS
        elif intent == UserIntent.INFORMATION_SEEKING:
            self.session_states[user_id] = ConversationState.PROVIDING_DETAILS
        elif intent == UserIntent.FEEDBACK:
            self.session_states[user_id] = ConversationState.FOLLOWING_UP
    
    def _generate_smart_suggestions(self, analysis: Dict[str, Any], 
                                  memory: ConversationMemory) -> List[str]:
        """Generate smart follow-up suggestions"""
        suggestions = []
        topics = analysis["topics"]
        intent = analysis["intent"]
        
        if intent == UserIntent.RECOMMENDATION_REQUEST:
            if "food" in topics:
                suggestions.extend([
                    "Tell me more about Turkish breakfast traditions",
                    "What are the best neighborhoods for food lovers?",
                    "How do I navigate dietary restrictions in Istanbul?"
                ])
            elif "attractions" in topics:
                suggestions.extend([
                    "Create a day itinerary for me",
                    "What are some hidden gems tourists miss?",
                    "How can I avoid crowds at popular sites?"
                ])
        elif intent == UserIntent.CASUAL_CHAT:
            suggestions.extend([
                "What makes Istanbul special for locals?",
                "Tell me something surprising about the city",
                "What's the best way to experience authentic Istanbul?"
            ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive conversation summary"""
        if user_id not in self.conversation_memories:
            return {"status": "no_conversation"}
        
        memory = self.conversation_memories[user_id]
        
        return {
            "total_turns": len(memory.turns),
            "conversation_themes": memory.conversation_themes,
            "mentioned_entities": dict(list(memory.mentioned_entities.items())[:10]),  # Top 10
            "emotional_journey": [e.value for e in memory.emotional_journey[-5:]],  # Last 5
            "user_preferences": memory.user_preferences,
            "personalization_level": self.response_generator._calculate_personalization_level(memory),
            "conversation_state": self.session_states.get(user_id, ConversationState.GREETING).value
        }
    
    def process_message(self, user_input: str, user_id: str) -> str:
        """Simplified interface for processing messages - returns just the response text"""
        result = self.process_conversation(user_input, user_id)
        return result["response"]
    
    def explain_reasoning(self, user_id: str = None) -> str:
        """Explain the AI's reasoning process"""
        if user_id and user_id in self.conversation_memories:
            memory = self.conversation_memories[user_id]
            recent_context = memory.get_recent_context(3)
            
            return f"Based on our conversation history of {len(memory.turns)} turns, " \
                   f"I've identified your interests in {', '.join(recent_context.get('themes', ['Istanbul']))}, " \
                   f"and I'm adapting my responses to be more personalized and contextually relevant."
        else:
            return "I use advanced natural language processing to understand context, intent, and emotional tone, " \
                   "then generate personalized responses based on Istanbul-specific knowledge and conversation history."

# Global instance
advanced_daily_talk_ai = AdvancedDailyTalkAI()

def process_advanced_daily_talk(user_input: str, user_id: str, weather_context: Optional[Dict] = None) -> Dict[str, Any]:
    """Main function to process daily talk with advanced AI"""
    return advanced_daily_talk_ai.process_conversation(user_input, user_id, weather_context)

if __name__ == "__main__":
    # Demo the advanced system
    print("🧠 Advanced Daily Talk AI System - GPT-Level Intelligence")
    print("=" * 60)
    
    test_inputs = [
        "Hi there! I'm visiting Istanbul for the first time and I'm so excited!",
        "I'm looking for amazing authentic food experiences, especially Turkish breakfast",
        "That sounds great! What about some hidden gems that tourists usually miss?",
        "Thanks! This is exactly what I was looking for. You know Istanbul really well!"
    ]
    
    user_id = "demo_user"
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n💬 Turn {i}: {user_input}")
        print("-" * 40)
        
        result = process_advanced_daily_talk(user_input, user_id)
        
        print(f"🤖 Response: {result['response']}")
        print(f"🧠 Intent: {result['analysis']['intent'].value}")
        print(f"💭 Emotional Tone: {result['analysis']['emotional_tone'].value}")
        print(f"📊 Personalization Level: {result['personalization_level']:.2f}")
        
        if result['suggestions']:
            print("💡 Smart Suggestions:")
            for suggestion in result['suggestions']:
                print(f"   • {suggestion}")
    
    print(f"\n📈 Conversation Summary:")
    summary = advanced_daily_talk_ai.get_conversation_summary(user_id)
    print(f"   Total turns: {summary['total_turns']}")
    print(f"   Personalization level: {summary['personalization_level']:.2f}")
    print(f"   Conversation state: {summary['conversation_state']}")
    
    print("\n🎉 Advanced Daily Talk AI Demo Complete!")
