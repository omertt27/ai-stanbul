"""
LLM Response Enhancer
=====================

Phase 3 of LLM Enhancement: Intelligent response enhancement with contextual insights.

Enhances ALL responses with:
- Weather-aware suggestions
- Time-aware tips (rush hour, closing times)
- Cultural insights and local knowledge
- Personalized recommendations based on history
- Context-aware follow-up suggestions

Example enhancement:
- Base: "Route from Sultanahmet to Galata Tower: 2.3km, 28 minutes walking"
- Enhanced: "Route from Sultanahmet to Galata Tower: 2.3km, 28 minutes walking. 
            ☀️ Beautiful weather today - perfect for walking! 
            💡 Tip: Stop by the Galata Bridge for stunning Bosphorus views.
            ⏰ Galata Tower closes at 8:30 PM - you'll arrive by 3:30 PM."

Author: Istanbul AI Team
Date: December 2025
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from .models import EnhancedResponse

logger = logging.getLogger(__name__)


class LLMResponseEnhancer:
    """
    LLM-powered response enhancer that adds contextual intelligence.
    
    This service runs AFTER the primary handler (route, info, restaurant, etc.)
    and enhances the response with:
    
    - Weather-aware suggestions
    - Time-sensitive tips
    - Cultural insights
    - Personalized recommendations
    - Smart follow-up suggestions
    
    Architecture:
    ```
    Handler Response
           ↓
    [Response Enhancer] ← Weather Service
           ↓            ← Time Context
    Enhanced Response   ← User History
           ↓            ← POI Database
    Return to User
    ```
    """
    
    def __init__(
        self,
        llm_client=None,
        weather_service=None,
        poi_database=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLM Response Enhancer.
        
        Args:
            llm_client: LLM API client for response enhancement
            weather_service: Weather service for context
            poi_database: POI database for recommendations
            config: Configuration overrides
        """
        self.llm_client = llm_client
        self.weather_service = weather_service
        self.poi_database = poi_database
        
        # Configuration
        self.config = {
            'enable_weather_tips': True,
            'enable_time_tips': True,
            'enable_cultural_insights': True,
            'enable_personalization': True,
            'enable_suggestions': True,
            'max_enhancement_length': 500,
            'timeout_seconds': 3,
            'fallback_on_error': True,
            **(config or {})
        }
        
        # Statistics
        self.stats = {
            'total_enhancements': 0,
            'llm_enhancements': 0,
            'fallback_enhancements': 0,
            'average_latency_ms': 0.0
        }
        
        logger.info("✅ LLM Response Enhancer initialized")
    
    async def enhance_response(
        self,
        base_response: str,
        original_query: str,
        user_context: Optional[Dict[str, Any]] = None,
        route_data: Optional[Dict[str, Any]] = None,
        response_type: str = "general"
    ) -> EnhancedResponse:
        """
        Enhance response with LLM-generated contextual insights.
        
        Args:
            base_response: Original response from handler
            original_query: User's original query
            user_context: User context (GPS, history, preferences)
            route_data: Optional route data for route-specific tips
            response_type: Type of response (route, info, restaurant, etc.)
            
        Returns:
            EnhancedResponse with original + enhancements
        """
        start_time = datetime.now()
        
        try:
            self.stats['total_enhancements'] += 1
            
            # Gather context
            context = await self._gather_context(user_context, route_data)
            
            # Build enhancement prompt
            prompt = self._build_enhancement_prompt(
                base_response=base_response,
                original_query=original_query,
                context=context,
                response_type=response_type
            )
            
            # Call LLM for enhancement
            if self.llm_client and self.config.get('enable_llm', True):
                enhancement = await self._llm_enhance(prompt)
                self.stats['llm_enhancements'] += 1
            else:
                # Fallback to rule-based enhancement
                enhancement = self._rule_based_enhance(
                    base_response, context, response_type
                )
                self.stats['fallback_enhancements'] += 1
            
            # Create enhanced response
            result = EnhancedResponse(
                original_response=base_response,
                enhanced_response=self._merge_responses(base_response, enhancement),
                enhancements=enhancement,
                context_used=context,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
            # Update stats
            self._update_stats(result.processing_time_ms)
            
            logger.info(
                f"✅ Enhanced response (type={response_type}, "
                f"time={result.processing_time_ms:.0f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error enhancing response: {e}", exc_info=True)
            
            # Return unenhanced response on error
            if self.config['fallback_on_error']:
                return EnhancedResponse(
                    original_response=base_response,
                    enhanced_response=base_response,
                    enhancements={},
                    context_used={},
                    processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            else:
                raise
    
    async def _gather_context(
        self,
        user_context: Optional[Dict[str, Any]],
        route_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Gather all relevant context for enhancement.
        
        Returns:
            Dict with weather, time, location, and other context
        """
        context = {
            'timestamp': datetime.now().isoformat(),
            'time_of_day': self._get_time_of_day(),
            'day_of_week': datetime.now().strftime('%A')
        }
        
        # Add weather context
        if self.config['enable_weather_tips'] and self.weather_service:
            try:
                weather = await self.weather_service.get_current_weather()
                context['weather'] = {
                    'condition': weather.get('condition'),
                    'temperature': weather.get('temperature'),
                    'is_raining': weather.get('is_raining', False)
                }
            except Exception as e:
                logger.warning(f"Could not fetch weather: {e}")
        
        # Add user location context
        if user_context and 'gps' in user_context:
            context['user_location'] = user_context['gps']
        
        # Add route-specific context
        if route_data:
            context['route'] = {
                'distance_km': route_data.get('distance', 0) / 1000,
                'duration_min': route_data.get('duration', 0) / 60,
                'mode': route_data.get('mode', 'walking'),
                'start': route_data.get('start'),
                'end': route_data.get('end')
            }
        
        # Add time-sensitive info
        if self.config['enable_time_tips']:
            context['time_context'] = self._get_time_context()
        
        return context
    
    def _build_enhancement_prompt(
        self,
        base_response: str,
        original_query: str,
        context: Dict[str, Any],
        response_type: str
    ) -> str:
        """
        Build LLM prompt for response enhancement.
        
        Args:
            base_response: Base response to enhance
            original_query: User's query
            context: Gathered context
            response_type: Type of response
            
        Returns:
            LLM prompt string
        """
        prompt = f"""You are an expert Istanbul travel assistant enhancing a travel response.

**Original User Query:** "{original_query}"

**Base Response:**
{base_response}

**Context:**
- Time: {context.get('time_of_day')} on {context.get('day_of_week')}
- Weather: {context.get('weather', {}).get('condition', 'unknown')}
- Temperature: {context.get('weather', {}).get('temperature', 'unknown')}°C
"""
        
        if 'route' in context:
            route = context['route']
            prompt += f"""- Route: {route['distance_km']:.1f}km, {route['duration_min']:.0f} min by {route['mode']}
"""
        
        prompt += """
**Your Task:**
Enhance the base response with helpful, contextual insights. Add:

1. **Weather-Aware Tips** (if relevant):
   - Suggestions based on current weather
   - What to bring (umbrella, sunscreen, etc.)

2. **Time-Aware Tips** (if relevant):
   - Rush hour warnings
   - Opening/closing times
   - Crowd levels at this time

3. **Cultural Insights** (brief):
   - Interesting facts about locations
   - Local customs or etiquette
   - Historical context (1-2 sentences max)

4. **Practical Recommendations**:
   - Nearby attractions to visit
   - Good photo spots
   - Where to get food/drinks

5. **Smart Suggestions** (optional):
   - What to do next
   - Alternative routes or activities

**Important:**
- Keep enhancements concise and actionable
- Use friendly, conversational tone
- Add emojis for visual appeal (☀️ 🌧️ ⏰ 💡 📍 🍽️)
- Maximum 3-4 additional sentences
- Don't repeat information from base response

Return ONLY the enhancement text (do not include the base response).
"""
        
        return prompt
    
    async def _llm_enhance(self, prompt: str) -> Dict[str, Any]:
        """
        Use LLM to generate enhancement.
        
        Args:
            prompt: Enhancement prompt
            
        Returns:
            Dict with enhancement sections
        """
        try:
            from openai import OpenAI
            
            client = OpenAI() if not self.llm_client else self.llm_client
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful Istanbul travel assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300,
                timeout=self.config['timeout_seconds']
            )
            
            enhancement_text = response.choices[0].message.content.strip()
            
            return {
                'text': enhancement_text,
                'method': 'llm'
            }
            
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            raise
    
    def _rule_based_enhance(
        self,
        base_response: str,
        context: Dict[str, Any],
        response_type: str
    ) -> Dict[str, Any]:
        """
        Fallback rule-based enhancement when LLM unavailable.
        
        Args:
            base_response: Base response
            context: Context dict
            response_type: Response type
            
        Returns:
            Dict with enhancement text
        """
        enhancements = []
        
        # Only add transportation-specific enhancements for relevant query types
        is_transport_query = response_type in ['route', 'transportation', 'directions', 'gps_routing']
        
        # Weather tip - relevant for outdoor activities and transportation
        if 'weather' in context:
            weather_condition = context['weather'].get('condition', '').lower()
            if context['weather'].get('is_raining'):
                enhancements.append("🌧️ It's raining - consider taking metro/tram instead of walking.")
            elif weather_condition in ['sunny', 'clear'] and is_transport_query:
                enhancements.append("☀️ Beautiful weather today - perfect for walking!")
        
        # Time tip - ONLY for transportation queries
        if is_transport_query:
            time_context = context.get('time_context', {})
            hour = datetime.now().hour
            is_weekend = time_context.get('is_weekend', False)
            
            if time_context.get('is_rush_hour'):
                if is_weekend:
                    # Weekends have less traffic, skip rush hour warning
                    pass
                elif 7 <= hour <= 9:
                    enhancements.append("⏰ Morning rush hour (7-9 AM) - metro and buses will be crowded. Allow extra time.")
                elif 17 <= hour <= 19:
                    enhancements.append("⏰ Evening rush hour (5-7 PM) - public transport is very busy. Consider leaving earlier or later if possible.")
        
        # Route-specific tips
        if 'route' in context:
            route = context['route']
            if route['distance_km'] > 3:
                enhancements.append(
                    f"💡 That's a {route['distance_km']:.1f}km walk - "
                    "you might want to consider metro or tram."
                )
        
        enhancement_text = " ".join(enhancements) if enhancements else ""
        
        return {
            'text': enhancement_text,
            'method': 'rule-based'
        }
    
    def _merge_responses(
        self,
        base_response: str,
        enhancement: Dict[str, Any]
    ) -> str:
        """
        Merge base response with enhancements.
        
        Args:
            base_response: Original response
            enhancement: Enhancement dict
            
        Returns:
            Merged response string
        """
        if not enhancement.get('text'):
            return base_response
        
        # Add enhancements after base response
        return f"{base_response}\n\n{enhancement['text']}"
    
    def _get_time_of_day(self) -> str:
        """Get time of day description."""
        hour = datetime.now().hour
        
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _get_time_context(self) -> Dict[str, Any]:
        """Get time-sensitive context."""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()  # 0 = Monday, 6 = Sunday
        
        return {
            'is_rush_hour': (7 <= hour <= 9) or (17 <= hour <= 19),
            'is_weekend': day_of_week >= 5,
            'is_evening': 17 <= hour < 21,
            'is_night': hour >= 21 or hour < 6
        }
    
    def _update_stats(self, latency_ms: float):
        """Update running statistics."""
        total = self.stats['total_enhancements']
        if total > 1:
            self.stats['average_latency_ms'] = (
                (self.stats['average_latency_ms'] * (total - 1) + latency_ms) / total
            )
        else:
            self.stats['average_latency_ms'] = latency_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """Get enhancement statistics."""
        return self.stats.copy()


# Global enhancer instance
_response_enhancer: Optional[LLMResponseEnhancer] = None


def get_response_enhancer(
    llm_client=None,
    weather_service=None,
    poi_database=None,
    config: Optional[Dict[str, Any]] = None
) -> LLMResponseEnhancer:
    """
    Get or create global response enhancer instance.
    
    Args:
        llm_client: LLM client
        weather_service: Weather service
        poi_database: POI database
        config: Configuration
        
    Returns:
        LLMResponseEnhancer singleton
    """
    global _response_enhancer
    
    if _response_enhancer is None:
        _response_enhancer = LLMResponseEnhancer(
            llm_client=llm_client,
            weather_service=weather_service,
            poi_database=poi_database,
            config=config
        )
    
    return _response_enhancer


async def enhance_response(
    base_response: str,
    original_query: str,
    **kwargs
) -> EnhancedResponse:
    """
    Convenience function for response enhancement.
    
    Args:
        base_response: Base response to enhance
        original_query: User's original query
        **kwargs: Additional arguments for enhancer
        
    Returns:
        EnhancedResponse
    """
    enhancer = get_response_enhancer()
    return await enhancer.enhance_response(
        base_response=base_response,
        original_query=original_query,
        **kwargs
    )
