#!/usr/bin/env python3
"""
Advanced AI Orchestrator for Istanbul Chatbot
Combines multiple AI models for superior performance
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import os
from datetime import datetime, timedelta

# Optional import for HTTP requests
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

class AIProvider(Enum):
    OPENAI_GPT4 = "gpt-4-turbo-preview"
    OPENAI_GPT35 = "gpt-3.5-turbo"
    ANTHROPIC_CLAUDE = "claude-3-sonnet-20240229"
    GOOGLE_GEMINI = "gemini-pro"
    LOCAL_LLAMA = "llama-2-70b"

@dataclass
class AIResponse:
    content: str
    provider: AIProvider
    confidence: float
    response_time: float
    tokens_used: int
    cost_estimate: float

@dataclass
class QueryContext:
    user_query: str
    session_id: str
    location: Optional[str] = None
    user_preferences: Optional[Dict] = None
    conversation_history: Optional[List] = None
    real_time_data: Optional[Dict] = None
    urgency: str = "normal"  # low, normal, high, urgent

# Simple fallback implementation for missing OpenAI async client
try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    AsyncOpenAI = _AsyncOpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    
    class ChatCompletions:  # type: ignore
        async def create(self, **kwargs):
            raise Exception("OpenAI library not available")
    
    class Chat:  # type: ignore
        def __init__(self):
            self.completions = ChatCompletions()
    
    class AsyncOpenAI:  # type: ignore
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = Chat()

class OpenAIProvider:
    """OpenAI provider implementation"""
    
    def __init__(self, model: str, max_tokens: int, temperature: float, priority: int):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.priority = priority
        
        try:
            if OPENAI_AVAILABLE:
                self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            else:
                self.client = None
        except Exception:
            logger.error("OpenAI client initialization failed")
            self.client = None
    
    async def generate(self, prompt: str, context: QueryContext) -> str:
        """Generate response using OpenAI"""
        
        if not self.client or not OPENAI_AVAILABLE:
            # Return a helpful fallback response
            return f"I understand you're asking about: {context.user_query}. Let me help you with Istanbul travel information based on my knowledge!"
        
        try:
            # Use the correct async OpenAI client method
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are KAM, an expert Istanbul travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return response.choices[0].message.content or "No response generated"
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            # Return a fallback response instead of raising
            return f"I understand you're asking about: {context.user_query}. Let me help you with Istanbul travel information!"

class AnthropicProvider:
    """Anthropic Claude provider implementation"""
    
    def __init__(self, model: str, max_tokens: int, priority: int):
        self.model = model
        self.max_tokens = max_tokens
        self.priority = priority
        
        # Initialize Anthropic client (placeholder)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
    
    async def generate(self, prompt: str, context: QueryContext) -> str:
        """Generate response using Anthropic Claude"""
        
        # Placeholder implementation
        # TODO: Implement actual Anthropic API calls
        return f"Claude response for: {context.user_query}"

class GeminiProvider:
    """Google Gemini provider implementation"""
    
    def __init__(self, model: str, priority: int):
        self.model = model
        self.priority = priority
        self.api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    async def generate(self, prompt: str, context: QueryContext) -> str:
        """Generate response using Google Gemini"""
        
        # Placeholder implementation
        # TODO: Implement actual Gemini API calls
        return f"Gemini response for: {context.user_query}"

class AILoadBalancer:
    """Load balancer for AI providers"""
    
    def __init__(self):
        self.provider_usage = {}
        self.provider_performance = {}
    
    def get_best_provider(self, providers: List[AIProvider]) -> AIProvider:
        """Get the best available provider based on performance and load"""
        
        if not providers:
            return AIProvider.OPENAI_GPT35
        
        # Simple round-robin for now
        return providers[0]
    
    def update_performance(self, provider: AIProvider, response_time: float, success: bool):
        """Update provider performance metrics"""
        
        if provider not in self.provider_performance:
            self.provider_performance[provider] = {
                "total_requests": 0,
                "successful_requests": 0,
                "avg_response_time": 0.0
            }
        
        metrics = self.provider_performance[provider]
        metrics["total_requests"] += 1
        
        if success:
            metrics["successful_requests"] += 1
        
        # Update average response time
        current_avg = metrics["avg_response_time"]
        total_requests = metrics["total_requests"]
        metrics["avg_response_time"] = ((current_avg * (total_requests - 1)) + response_time) / total_requests

class ResponseQualityScorer:
    """Scores and ranks AI responses"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    async def score_response(self, response: AIResponse, context: QueryContext) -> float:
        """Score response quality based on multiple factors"""
        
        score = 0.0
        
        # Length appropriateness (20%)
        length_score = self._score_length(response.content, context.user_query)
        score += length_score * 0.2
        
        # Relevance (30%)
        relevance_score = self._score_relevance(response.content, context.user_query)
        score += relevance_score * 0.3
        
        # Provider confidence (20%)
        score += response.confidence * 0.2
        
        # Response time (15%)
        time_score = min(1.0, 10.0 / max(response.response_time, 0.1))  # Faster is better
        score += time_score * 0.15
        
        # Istanbul-specific knowledge (15%)
        istanbul_score = self._score_istanbul_knowledge(response.content)
        score += istanbul_score * 0.15
        
        return min(1.0, score)
    
    def _score_length(self, content: str, query: str) -> float:
        """Score response length appropriateness"""
        
        content_words = len(content.split())
        query_words = len(query.split())
        
        # Ideal response is 10-20 times query length
        ideal_min = query_words * 10
        ideal_max = query_words * 20
        
        if ideal_min <= content_words <= ideal_max:
            return 1.0
        elif content_words < ideal_min:
            return content_words / ideal_min
        else:
            return ideal_max / content_words
    
    def _score_relevance(self, content: str, query: str) -> float:
        """Score response relevance to query"""
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(query_words.intersection(content_words))
        return min(1.0, overlap / len(query_words))
    
    def _score_istanbul_knowledge(self, content: str) -> float:
        """Score Istanbul-specific knowledge in response"""
        
        istanbul_terms = [
            'istanbul', 'beyoglu', 'sultanahmet', 'galata', 'kadikoy', 'besiktas',
            'bosphorus', 'hagia sophia', 'blue mosque', 'grand bazaar', 'spice bazaar',
            'topkapi', 'dolmabahce', 'turkish', 'ottoman', 'byzantine'
        ]
        
        content_lower = content.lower()
        found_terms = sum(1 for term in istanbul_terms if term in content_lower)
        
        return min(1.0, found_terms / 5)  # Max score if 5+ Istanbul terms

class RealTimeDataManager:
    """Manages real-time data sources for Istanbul"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    async def get_weather_data(self) -> Dict:
        """Get current weather data for Istanbul"""
        
        cache_key = "weather"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            # Placeholder for real weather API integration
            weather_data = {
                "temperature": "22°C",
                "condition": "Partly cloudy",
                "humidity": "65%",
                "wind": "10 km/h NE",
                "recommendation": "Perfect weather for walking around Istanbul!"
            }
            
            self._update_cache(cache_key, weather_data)
            return weather_data
            
        except Exception as e:
            logger.error(f"Weather data fetch failed: {e}")
            return {"error": "Weather data unavailable"}
    
    async def get_traffic_data(self) -> Dict:
        """Get current traffic and transport data"""
        
        cache_key = "traffic"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            traffic_data = {
                "metro_status": "All lines operational",
                "ferry_status": "Regular schedule",
                "traffic_level": "Moderate",
                "estimated_travel_time": {
                    "sultanahmet_to_taksim": "25 minutes by metro",
                    "kadikoy_to_besiktas": "30 minutes by ferry"
                }
            }
            
            self._update_cache(cache_key, traffic_data)
            return traffic_data
            
        except Exception as e:
            logger.error(f"Traffic data fetch failed: {e}")
            return {"error": "Traffic data unavailable"}
    
    async def get_events_data(self) -> Dict:
        """Get current events and happenings"""
        
        cache_key = "events"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            events_data = {
                "today": [
                    "Istanbul Modern Art Exhibition - Contemporary Turkish Art",
                    "Whirling Dervish Ceremony - Galata Mevlevihanesi (19:00)",
                    "Bosphorus Sunset Cruise - Departing from Eminönü (18:30)"
                ],
                "this_weekend": [
                    "Grand Bazaar Special Events",
                    "Traditional Music Concert at Hagia Sophia"
                ]
            }
            
            self._update_cache(cache_key, events_data)
            return events_data
            
        except Exception as e:
            logger.error(f"Events data fetch failed: {e}")
            return {"error": "Events data unavailable"}
    
    async def get_restaurant_availability(self) -> Dict:
        """Get restaurant availability and booking info"""
        
        cache_key = "restaurants"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            availability_data = {
                "high_demand_areas": ["Beyoglu", "Sultanahmet", "Galata"],
                "booking_recommendation": "Recommended to book 2-3 hours in advance for popular restaurants",
                "current_wait_times": {
                    "beyoglu_average": "15-30 minutes",
                    "sultanahmet_average": "20-40 minutes"
                }
            }
            
            self._update_cache(cache_key, availability_data)
            return availability_data
            
        except Exception as e:
            logger.error(f"Restaurant availability fetch failed: {e}")
            return {"error": "Restaurant availability data unavailable"}
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]["timestamp"]
        return (time.time() - cache_time) < self.cache_timeout
    
    def _update_cache(self, key: str, data: Dict):
        """Update cache with new data"""
        
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

class AdvancedAIOrchestrator:
    """
    Advanced AI orchestrator that combines multiple AI models
    for the best possible responses about Istanbul
    """
    
    def __init__(self):
        self.providers = {}
        self.response_cache = {}
        self.performance_metrics = {}
        self.load_balancer = AILoadBalancer()
        self.quality_scorer = ResponseQualityScorer()
        
        # Initialize AI providers
        self._initialize_providers()
        
        # Real-time data sources
        self.data_sources = RealTimeDataManager()
        
    def _initialize_providers(self):
        """Initialize all available AI providers"""
        
        # OpenAI GPT-4 (Premium)
        if os.getenv("OPENAI_API_KEY"):
            self.providers[AIProvider.OPENAI_GPT4] = OpenAIProvider(
                model="gpt-4-turbo-preview",
                max_tokens=4000,
                temperature=0.7,
                priority=1  # Highest priority
            )
            
        # OpenAI GPT-3.5 (Fast)
        if os.getenv("OPENAI_API_KEY"):
            self.providers[AIProvider.OPENAI_GPT35] = OpenAIProvider(
                model="gpt-3.5-turbo",
                max_tokens=2000,
                temperature=0.7,
                priority=2
            )
            
        # Anthropic Claude (Reasoning)
        if os.getenv("ANTHROPIC_API_KEY"):
            self.providers[AIProvider.ANTHROPIC_CLAUDE] = AnthropicProvider(
                model="claude-3-sonnet-20240229",
                max_tokens=3000,
                priority=1
            )
            
        # Google Gemini (Multimodal)
        if os.getenv("GOOGLE_AI_API_KEY"):
            self.providers[AIProvider.GOOGLE_GEMINI] = GeminiProvider(
                model="gemini-pro",
                priority=2
            )
    
    async def generate_response(self, context: QueryContext) -> str:
        """
        Generate the best possible response using multiple AI models
        """
        start_time = time.time()
        
        try:
            # 1. Analyze query and determine best strategy
            strategy = await self._analyze_query_strategy(context)
            
            # 2. Fetch real-time data if needed
            real_time_data = await self._fetch_real_time_data(context)
            context.real_time_data = real_time_data
            
            # 3. Generate responses from multiple providers
            responses = await self._generate_parallel_responses(context, strategy)
            
            # 4. Score and rank responses
            best_response = await self._select_best_response(responses, context)
            
            # 5. Apply post-processing enhancements
            enhanced_response = await self._enhance_response(best_response, context)
            
            # 6. Update performance metrics
            total_time = time.time() - start_time
            await self._update_metrics(strategy, best_response, total_time)
            
            logger.info(f"🚀 Generated response in {total_time:.2f}s using {best_response.provider.value}")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"AI Orchestrator error: {e}")
            # Fallback to simple response
            return await self._generate_fallback_response(context)
    
    async def _analyze_query_strategy(self, context: QueryContext) -> Dict:
        """Analyze query to determine the best AI strategy"""
        
        query_lower = context.user_query.lower()
        
        strategy = {
            "complexity": "medium",
            "preferred_providers": [AIProvider.OPENAI_GPT4],
            "parallel_count": 1,
            "use_real_time": False,
            "response_style": "informative"
        }
        
        # Complex queries need premium models
        if any(word in query_lower for word in ['compare', 'analyze', 'recommend', 'plan', 'itinerary']):
            strategy["complexity"] = "high"
            strategy["preferred_providers"] = [AIProvider.OPENAI_GPT4, AIProvider.ANTHROPIC_CLAUDE]
            strategy["parallel_count"] = 2
            
        # Real-time queries need current data
        if any(word in query_lower for word in ['now', 'today', 'current', 'weather', 'traffic', 'events']):
            strategy["use_real_time"] = True
            
        # Simple queries can use faster models
        if len(context.user_query.split()) < 5:
            strategy["complexity"] = "low"
            strategy["preferred_providers"] = [AIProvider.OPENAI_GPT35]
            
        # Urgent queries need fastest response
        if context.urgency == "urgent":
            strategy["preferred_providers"] = [AIProvider.OPENAI_GPT35]
            strategy["parallel_count"] = 1
            
        return strategy
    
    async def _fetch_real_time_data(self, context: QueryContext) -> Dict:
        """Fetch real-time data relevant to the query"""
        
        data = {}
        
        try:
            # Weather data
            if any(word in context.user_query.lower() for word in ['weather', 'rain', 'sunny', 'temperature']):
                data['weather'] = await self.data_sources.get_weather_data()
                
            # Traffic data
            if any(word in context.user_query.lower() for word in ['traffic', 'transport', 'metro', 'bus']):
                data['traffic'] = await self.data_sources.get_traffic_data()
                
            # Events data
            if any(word in context.user_query.lower() for word in ['event', 'concert', 'festival', 'happening']):
                data['events'] = await self.data_sources.get_events_data()
                
            # Restaurant availability
            if any(word in context.user_query.lower() for word in ['restaurant', 'book', 'reservation']):
                data['restaurant_availability'] = await self.data_sources.get_restaurant_availability()
                
        except Exception as e:
            logger.warning(f"Failed to fetch real-time data: {e}")
            
        return data
    
    async def _generate_parallel_responses(self, context: QueryContext, strategy: Dict) -> List[AIResponse]:
        """Generate responses from multiple AI providers in parallel"""
        
        providers = strategy["preferred_providers"][:strategy["parallel_count"]]
        
        tasks = []
        for provider in providers:
            if provider in self.providers:
                task = asyncio.create_task(
                    self._generate_single_response(provider, context)
                )
                tasks.append(task)
        
        # Wait for all responses with timeout
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout
            )
            
            # Filter out exceptions
            valid_responses = [r for r in responses if isinstance(r, AIResponse)]
            
            return valid_responses
            
        except asyncio.TimeoutError:
            logger.warning("AI response timeout, using available responses")
            # Return any completed responses
            completed = [task.result() for task in tasks if task.done() and not task.exception()]
            return completed
    
    async def _generate_single_response(self, provider: AIProvider, context: QueryContext) -> AIResponse:
        """Generate response from a single AI provider"""
        
        start_time = time.time()
        
        try:
            provider_instance = self.providers[provider]
            
            # Build enhanced prompt
            prompt = await self._build_enhanced_prompt(context)
            
            # Generate response
            content = await provider_instance.generate(prompt, context)
            
            response_time = time.time() - start_time
            
            # Calculate confidence score
            confidence = await self._calculate_confidence(content, context)
            
            return AIResponse(
                content=content,
                provider=provider,
                confidence=confidence,
                response_time=response_time,
                tokens_used=len(content.split()),  # Approximate
                cost_estimate=self._estimate_cost(provider, len(content))
            )
            
        except Exception as e:
            logger.error(f"Provider {provider.value} failed: {e}")
            raise
    
    async def _build_enhanced_prompt(self, context: QueryContext) -> str:
        """Build an enhanced prompt with context and real-time data"""
        
        prompt = f"""You are KAM, the world's most knowledgeable and helpful Istanbul travel assistant. You have access to real-time data and deep local knowledge.

User Query: {context.user_query}

Context:
- Session ID: {context.session_id}
- User Location: {context.location or 'Unknown'}
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        # Add conversation history
        if context.conversation_history:
            prompt += "\nConversation History:\n"
            for msg in context.conversation_history[-3:]:  # Last 3 messages
                prompt += f"- {msg}\n"
        
        # Add real-time data
        if context.real_time_data:
            prompt += "\nReal-time Data:\n"
            for key, value in context.real_time_data.items():
                prompt += f"- {key.title()}: {value}\n"
        
        # Add user preferences
        if context.user_preferences:
            prompt += "\nUser Preferences:\n"
            for key, value in context.user_preferences.items():
                prompt += f"- {key}: {value}\n"
        
        prompt += """
Guidelines:
1. Provide accurate, up-to-date information about Istanbul
2. Include practical details (prices, hours, directions)
3. Personalize based on user context and preferences
4. Use real-time data when relevant
5. Be enthusiastic but professional
6. Include actionable recommendations
7. Anticipate follow-up questions

Response Style: Conversational, informative, and locally authentic
"""
        
        return prompt
    
    async def _select_best_response(self, responses: List[AIResponse], context: QueryContext) -> AIResponse:
        """Select the best response from multiple options"""
        
        if not responses:
            raise Exception("No valid responses generated")
        
        if len(responses) == 1:
            return responses[0]
        
        # Score each response
        scored_responses = []
        for response in responses:
            score = await self.quality_scorer.score_response(response, context)
            scored_responses.append((score, response))
        
        # Sort by score (highest first)
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        
        best_response = scored_responses[0][1]
        
        logger.info(f"Selected best response from {best_response.provider.value} (score: {scored_responses[0][0]:.2f})")
        
        return best_response
    
    async def _enhance_response(self, response: AIResponse, context: QueryContext) -> str:
        """Apply post-processing enhancements to the response"""
        
        content = response.content
        
        # Add real-time context if relevant
        if context.real_time_data:
            content = await self._inject_real_time_context(content, context.real_time_data)
        
        # Add personalization
        if context.user_preferences:
            content = await self._personalize_response(content, context.user_preferences)
        
        # Format for better readability
        content = await self._format_response(content)
        
        # Add call-to-action if appropriate
        content = await self._add_call_to_action(content, context)
        
        return content
    
    async def _calculate_confidence(self, content: str, context: QueryContext) -> float:
        """Calculate confidence score for a response"""
        
        confidence = 0.5  # Base confidence
        
        # Length factor
        if 50 <= len(content) <= 1000:
            confidence += 0.2
        
        # Specificity factor
        if any(word in content.lower() for word in ['address', 'price', 'hours', 'phone']):
            confidence += 0.1
        
        # Istanbul relevance
        istanbul_terms = ['istanbul', 'turkish', 'bosphorus', 'galata', 'sultanahmet']
        relevance_count = sum(1 for term in istanbul_terms if term in content.lower())
        confidence += min(relevance_count * 0.05, 0.2)
        
        # Real-time data usage
        if context.real_time_data and any(str(value) in content for value in context.real_time_data.values()):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _estimate_cost(self, provider: AIProvider, content_length: int) -> float:
        """Estimate cost for AI provider usage"""
        
        # Rough cost estimates (in USD)
        costs = {
            AIProvider.OPENAI_GPT4: 0.03 * (content_length / 1000),
            AIProvider.OPENAI_GPT35: 0.002 * (content_length / 1000),
            AIProvider.ANTHROPIC_CLAUDE: 0.008 * (content_length / 1000),
            AIProvider.GOOGLE_GEMINI: 0.001 * (content_length / 1000),
        }
        
        return costs.get(provider, 0.0)
    
    async def _generate_fallback_response(self, context: QueryContext) -> str:
        """Generate fallback response when all AI providers fail"""
        
        return f"""I apologize, but I'm experiencing technical difficulties right now. 
However, I'd still love to help you with your Istanbul question: "{context.user_query}"

In the meantime, here are some general Istanbul resources:
• Visit Istanbul Official Website: istanbul.com
• Istanbul Transportation: iett.istanbul
• Emergency: 112

Please try asking your question again in a moment!"""

    async def _update_metrics(self, strategy: Dict, response: AIResponse, total_time: float):
        """Update performance metrics"""
        provider_name = response.provider.value
        if provider_name not in self.performance_metrics:
            self.performance_metrics[provider_name] = {
                "total_requests": 0,
                "total_time": 0,
                "average_confidence": 0,
                "success_rate": 0
            }
        
        metrics = self.performance_metrics[provider_name]
        metrics["total_requests"] += 1
        metrics["total_time"] += total_time
        metrics["average_confidence"] = (metrics["average_confidence"] + response.confidence) / 2
    
    async def _inject_real_time_context(self, content: str, real_time_data: Dict) -> str:
        """Inject real-time context into the response"""
        if not real_time_data:
            return content
        
        # Add weather info if relevant
        if "weather" in real_time_data and any(word in content.lower() for word in ["weather", "rain", "sunny"]):
            weather = real_time_data["weather"]
            content += f"\n\n🌤️ **Current Weather:** {weather.get('temperature', 'N/A')}°C, {weather.get('condition', 'N/A')}"
        
        # Add traffic info if relevant
        if "traffic" in real_time_data and any(word in content.lower() for word in ["transport", "metro", "traffic"]):
            traffic = real_time_data["traffic"]
            content += f"\n\n🚇 **Transport Status:** {traffic.get('metro_status', 'N/A')}"
        
        return content
    
    async def _personalize_response(self, content: str, preferences: Dict) -> str:
        """Personalize response based on user preferences"""
        if not preferences:
            return content
        
        # Add personalization based on budget
        if "budget" in preferences:
            budget = preferences["budget"]
            if budget == "low":
                content += "\n\n💰 *Budget-friendly tip: Look for local markets and street food for authentic experiences at great prices!*"
            elif budget == "high":
                content += "\n\n✨ *Luxury option: Consider premium experiences like private tours or high-end restaurants for exceptional service.*"
        
        return content
    
    async def _format_response(self, content: str) -> str:
        """Format response for better readability"""
        # Add emojis and structure
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append(line)
                continue
            
            # Add icons for different types of content
            if any(word in line.lower() for word in ["restaurant", "eat", "food"]):
                if not line.startswith('🍽️'):
                    line = "🍽️ " + line
            elif any(word in line.lower() for word in ["museum", "gallery", "art"]):
                if not line.startswith('🏛️'):
                    line = "🏛️ " + line
            elif any(word in line.lower() for word in ["metro", "bus", "transport"]):
                if not line.startswith('🚇'):
                    line = "🚇 " + line
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    async def _add_call_to_action(self, content: str, context: QueryContext) -> str:
        """Add appropriate call-to-action to the response"""
        
        query_lower = context.user_query.lower()
        
        # Restaurant queries
        if any(word in query_lower for word in ["restaurant", "eat", "food"]):
            content += "\n\n💡 *Would you like me to help you find specific restaurants in any particular district or cuisine type?*"
        
        # Transport queries
        elif any(word in query_lower for word in ["transport", "metro", "get to"]):
            content += "\n\n🗺️ *Need directions from your current location? Just let me know where you're starting from!*"
        
        # General exploration
        else:
            content += "\n\n🌟 *Is there anything specific about Istanbul you'd like to explore further?*"
        
        return content
