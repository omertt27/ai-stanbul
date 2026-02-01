"""
LLM-based Intent Detection Service for AI Istanbul Chatbot

This service replaces keyword-based transportation detection with 
proper LLM intent classification to handle natural language queries
in multiple languages robustly.

Uses RunPod LLM instead of OpenAI for intent detection.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import os

logger = logging.getLogger(__name__)

class IntentType(Enum):
    TRANSPORTATION = "transportation"
    TOURISM = "tourism"
    RESTAURANTS = "restaurants"
    GENERAL_INFORMATION = "general_information"
    WEATHER = "weather"
    CULTURE_EVENTS = "culture_events"
    SHOPPING = "shopping"
    OTHER = "other"

@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    reasoning: str
    is_transportation: bool
    requires_location: bool
    language_detected: Optional[str] = None

class LLMIntentDetector:
    """
    LLM-powered intent detection service that can understand natural language
    queries in multiple languages (Turkish, English, Arabic, French, German, Russian)
    and classify them into appropriate intents.
    
    Uses RunPod LLM for classification.
    """
    
    def __init__(self):
        # Import and use singleton RunPod LLM client
        try:
            from .runpod_llm_client import get_llm_client
            self.llm_client = get_llm_client()
            self.llm_available = self.llm_client is not None and self.llm_client.enabled
            if self.llm_available:
                logger.info("🤖 LLM Intent Detector initialized with RunPod LLM (singleton)")
            else:
                logger.warning("⚠️ RunPod LLM not available - will use keyword fallback")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RunPod LLM client: {str(e)}")
            self.llm_client = None
            self.llm_available = False
        
    async def detect_intent(self, query: str, user_location: Optional[Dict] = None, binary_mode: bool = False) -> IntentResult:
        """
        Detect the intent of a user query using LLM classification.
        
        Args:
            query: The user's query text
            user_location: Optional user location context
            binary_mode: If True, use simplified binary classification (Transportation vs Others)
            
        Returns:
            IntentResult with classification details
        """
        # If RunPod LLM is not available, fallback to keyword detection immediately
        if not self.llm_available:
            logger.warning("🔄 RunPod LLM not available - using keyword fallback")
            return self._fallback_keyword_detection(query)
        
        try:
            # Construct the prompt for intent detection
            if binary_mode:
                prompt = self._build_binary_detection_prompt(query, user_location)
            else:
                prompt = self._build_intent_detection_prompt(query, user_location)
            
            # Call RunPod LLM API
            response = await self._call_runpod_llm(prompt, max_tokens=100 if binary_mode else 200)
            
            if not response:
                raise Exception("Empty response from RunPod LLM")
            
            # Parse the response
            if binary_mode:
                intent_data = self._parse_binary_response(response)
            else:
                intent_data = self._parse_intent_response(response)
            
            # Create IntentResult
            intent_type = IntentType(intent_data.get('intent', 'other'))
            confidence = float(intent_data.get('confidence', 0.5))
            reasoning = intent_data.get('reasoning', 'LLM classification')
            language = intent_data.get('language_detected')
            
            # Determine if this is transportation-related
            is_transportation = intent_type == IntentType.TRANSPORTATION
            requires_location = intent_data.get('requires_location', False)
            
            result = IntentResult(
                intent=intent_type,
                confidence=confidence,
                reasoning=reasoning,
                is_transportation=is_transportation,
                requires_location=requires_location,
                language_detected=language
            )
            
            mode_text = "BINARY" if binary_mode else "FULL"
            logger.info(f"🎯 RunPod LLM Intent Detection ({mode_text}): {intent_type.value} (confidence: {confidence:.2f}) - {reasoning}")
            return result
            
        except Exception as e:
            logger.error(f"❌ RunPod LLM Intent Detection failed: {str(e)}")
            # Fallback to keyword-based detection
            return self._fallback_keyword_detection(query)
    
    def _build_binary_detection_prompt(self, query: str, user_location: Optional[Dict] = None) -> str:
        """Build a simplified prompt for binary classification (Transportation vs Others)."""
        location_context = ""
        if user_location:
            location_context = f"User is in Istanbul (lat: {user_location.get('latitude', 'unknown')}, lon: {user_location.get('longitude', 'unknown')}). "
        
        prompt = f"""
You are a transportation intent classifier for Istanbul travel assistant. 
Your job is SIMPLE: determine if the user wants transportation/route help or something else.

{location_context}

USER QUERY: "{query}"

CLASSIFICATION RULES:
- "transportation" = User wants directions, routes, how to get somewhere, travel between places, metro/bus/taxi info
- "other" = Everything else (restaurants, attractions, weather, general info, etc.)

Examples:
"How do I get to Galata Tower?" → transportation
"Best restaurants in Sultanahmet?" → other  
"Taksim'den Kadıköy'e nasıl giderim?" → transportation
"What's the weather today?" → other

Respond with JSON:
{{"intent": "transportation" or "other", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
        return prompt
    
    def _parse_binary_response(self, response: str) -> Dict[str, Any]:
        """Parse the binary classification response from RunPod LLM."""
        try:
            # Extract JSON from response
            response = response.strip()
            
            # Handle code blocks
            if '```' in response:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    response = response[start_idx:end_idx]
            
            # Find JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Normalize intent value
                intent_value = data.get('intent', 'other').lower().strip()
                if intent_value == 'transportation':
                    data['intent'] = 'transportation'
                else:
                    data['intent'] = 'other'
                
                return data
            else:
                raise Exception("No JSON object found in response")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse binary classification JSON: {e}")
            logger.error(f"   Response was: {response}")
            raise Exception(f"Invalid JSON in binary classification: {e}")
        except Exception as e:
            logger.error(f"❌ Binary response parsing error: {e}")
            raise
    
    def _build_intent_detection_prompt(self, query: str, user_location: Optional[Dict] = None) -> str:
        """Build the prompt for LLM intent detection."""
        location_context = ""
        if user_location:
            location_context = f"User is located in Istanbul (lat: {user_location.get('latitude', 'unknown')}, lon: {user_location.get('longitude', 'unknown')}). "
        
        prompt = f"""
You are an expert intent classifier for an AI Istanbul travel assistant. Your task is to analyze user queries and classify them into the correct intent category.

{location_context}

SUPPORTED INTENTS:
1. "transportation" - Questions about how to get somewhere, routes, directions, public transport, travel between locations
2. "tourism" - Questions about tourist attractions, historical sites, museums, landmarks to visit
3. "restaurants" - Questions about food, dining, restaurant recommendations, Turkish cuisine
4. "general_information" - General questions about Istanbul, city info, population, history, etc.
5. "weather" - Weather-related questions
6. "culture_events" - Cultural events, festivals, shows, concerts
7. "shopping" - Shopping recommendations, markets, malls, souvenirs
8. "other" - Any other type of query

USER QUERY: "{query}"

Analyze this query and respond with a JSON object containing:
- "intent": one of the supported intent values
- "confidence": confidence score between 0.0 and 1.0
- "reasoning": brief explanation of why this intent was chosen
- "language_detected": detected language (turkish, english, arabic, french, german, russian, or other)
- "requires_location": true if the query would benefit from location context

IMPORTANT:
- Focus on the user's actual intention, not just keywords
- Transportation intent includes: directions, routes, how to get somewhere, travel between places
- Be confident in your classification (confidence 0.8+ for clear cases)
- Consider the context of Istanbul as a tourist destination

Respond with valid JSON only:
"""
        return prompt
    
    async def _call_runpod_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """Call RunPod LLM API for intent detection."""
        try:
            # Use RunPod LLM client to generate response
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,  # Configurable token limit
                temperature=0.1  # Low temperature for consistent classification
            )
            
            if not response or not response.get('generated_text'):
                raise Exception("No generated text in RunPod LLM response")
            
            generated_text = response['generated_text'].strip()
            logger.debug(f"🤖 RunPod LLM generated: {generated_text[:200]}...")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ RunPod LLM API call failed: {str(e)}")
            raise
    
    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from RunPod LLM."""
        try:
            # Try to extract JSON from the response
            response = response.strip()
            
            # Handle different response formats from LLM
            if response.startswith('```json'):
                # Extract JSON from code block
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    response = response[start_idx:end_idx]
            elif response.startswith('```'):
                response = response[3:]
            
            if response.endswith('```'):
                response = response[:-3]
            
            # Find JSON object in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                return data
            else:
                # Try parsing the entire response as JSON
                data = json.loads(response)
                return data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse intent JSON: {str(e)}")
            logger.error(f"Raw response: {response}")
            
            # Try to extract intent information using simple parsing as fallback
            response_lower = response.lower()
            if "transportation" in response_lower:
                return {
                    "intent": "transportation",
                    "confidence": 0.8,
                    "reasoning": "Parsed from LLM response text",
                    "requires_location": True,
                    "language_detected": "unknown"
                }
            elif "tourism" in response_lower:
                return {
                    "intent": "tourism", 
                    "confidence": 0.7,
                    "reasoning": "Parsed from LLM response text",
                    "requires_location": True,
                    "language_detected": "unknown"
                }
            else:
                return {
                    "intent": "other",
                    "confidence": 0.5,
                    "reasoning": "Fallback parsing",
                    "requires_location": False,
                    "language_detected": "unknown"
                }
    
    def _fallback_keyword_detection(self, query: str) -> IntentResult:
        """Fallback to keyword-based detection if LLM fails."""
        logger.warning("🔄 Using fallback keyword detection")
        
        query_lower = query.lower()
        
        # Transportation keywords (multilingual)
        transportation_keywords = [
            # English
            'how do i get', 'how can i get', 'how to get', 'route to', 'way to', 
            'from', 'directions to', 'navigate to', 'take me to', 'go to',
            'metro', 'bus', 'tram', 'ferry', 'taxi',
            # Turkish
            'nasıl giderim', 'nasıl gidebilirim', 'nasıl ulaşırım', 'nasıl varabilirim',
            'yolunu tarif', 'güzergah', 'rota', 'ulaşım', 'gidiş', 'vakyolu',
            'metro', 'otobüs', 'tramvay', 'vapur', 'dolmuş', 'taksi',
            # Common location names that often indicate transportation
            'taksim', 'kadıköy', 'sultanahmet', 'galata', 'beşiktaş'
        ]
        
        is_transportation = any(keyword in query_lower for keyword in transportation_keywords)
        
        if is_transportation:
            return IntentResult(
                intent=IntentType.TRANSPORTATION,
                confidence=0.7,  # Lower confidence for keyword detection
                reasoning="Keyword-based transportation detection (fallback)",
                is_transportation=True,
                requires_location=True,
                language_detected=None
            )
        
        # Default to general information
        return IntentResult(
            intent=IntentType.GENERAL_INFORMATION,
            confidence=0.5,
            reasoning="Fallback classification",
            is_transportation=False,
            requires_location=False,
            language_detected=None
        )
    
    async def extract_locations(self, query: str, user_has_gps: bool = False) -> Optional[Dict[str, Any]]:
        """
        Use LLM to extract origin and destination from a transportation query.
        
        This is much more accurate than regex patterns because the LLM understands:
        - Natural language variations ("how can i go to X from Y", "from Y to X", "get to X")
        - Multiple languages (Turkish, English, etc.)
        - Typos and informal language
        - Context (when no origin is specified, it might mean "from current location")
        
        Args:
            query: The user's transportation query
            user_has_gps: Whether the user has GPS location available
            
        Returns:
            Dict with 'origin', 'destination', 'use_gps_for_origin' keys, or None if extraction fails
        """
        if not self.llm_available:
            logger.warning("🔄 LLM not available for location extraction - using fallback")
            return self._fallback_location_extraction(query, user_has_gps)
        
        try:
            prompt = f"""Extract the origin and destination from this Istanbul transportation query.

Query: "{query}"

User has GPS location: {user_has_gps}

Instructions:
1. Identify the ORIGIN (starting point) and DESTINATION (ending point)
2. If no origin is specified but user has GPS, set use_gps_for_origin to true
3. If query says "from my location", "from here", "from current location", set use_gps_for_origin to true
4. Clean up location names (e.g., "atakoy" → "Ataköy", "taksim" → "Taksim")
5. Handle reversed order (e.g., "to Taksim from Ataköy" means origin=Ataköy, destination=Taksim)

Respond with JSON only:
{{
    "origin": "location name or null",
    "destination": "location name or null", 
    "use_gps_for_origin": true/false,
    "confidence": 0.0-1.0
}}

Examples:
- "how can i go to taksim from atakoy" → {{"origin": "Ataköy", "destination": "Taksim", "use_gps_for_origin": false, "confidence": 0.95}}
- "how do I get to Sultanahmet" → {{"origin": null, "destination": "Sultanahmet", "use_gps_for_origin": true, "confidence": 0.9}}
- "from kadikoy to besiktas" → {{"origin": "Kadıköy", "destination": "Beşiktaş", "use_gps_for_origin": false, "confidence": 0.95}}
- "taksimden kadıköye nasıl giderim" → {{"origin": "Taksim", "destination": "Kadıköy", "use_gps_for_origin": false, "confidence": 0.95}}

JSON response:"""

            response = await self._call_runpod_llm(prompt)
            
            if not response:
                raise Exception("Empty response from LLM")
            
            # Parse JSON from response
            result = self._parse_location_response(response)
            
            if result:
                logger.info(f"🎯 LLM Location Extraction: {result.get('origin')} → {result.get('destination')} (GPS: {result.get('use_gps_for_origin')})")
                return result
            else:
                raise Exception("Failed to parse location response")
                
        except Exception as e:
            logger.error(f"❌ LLM location extraction failed: {str(e)}")
            return self._fallback_location_extraction(query, user_has_gps)
    
    def _parse_location_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON response from location extraction."""
        try:
            response = response.strip()
            
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Validate required fields
                if 'destination' in data:
                    return {
                        'origin': data.get('origin'),
                        'destination': data.get('destination'),
                        'use_gps_for_origin': data.get('use_gps_for_origin', False),
                        'confidence': data.get('confidence', 0.8)
                    }
            
            return None
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse location JSON: {str(e)}")
            return None
    
    def _fallback_location_extraction(self, query: str, user_has_gps: bool) -> Optional[Dict[str, Any]]:
        """Simple fallback location extraction using basic patterns."""
        import re
        
        query_lower = query.lower()
        
        # Istanbul locations to recognize
        locations = [
            'taksim', 'kadıköy', 'kadikoy', 'sultanahmet', 'beşiktaş', 'besiktas',
            'ataköy', 'atakoy', 'eminönü', 'eminonu', 'karaköy', 'karakoy',
            'galata', 'üsküdar', 'uskudar', 'bakırköy', 'bakirkoy', 'mecidiyeköy',
            'levent', 'maslak', 'şişli', 'sisli', 'beykoz', 'sarıyer', 'sariyer',
            'fatih', 'beyoğlu', 'beyoglu', 'ortaköy', 'ortakoy', 'bebek',
            'airport', 'havalimanı', 'havalimani', 'ist airport', 'sabiha',
            'grand bazaar', 'kapalıçarşı', 'kapalicarsi', 'spice bazaar',
            'hagia sophia', 'ayasofya', 'blue mosque', 'topkapı', 'topkapi'
        ]
        
        # Normalize location names
        location_map = {
            'kadikoy': 'Kadıköy', 'kadıköy': 'Kadıköy',
            'taksim': 'Taksim',
            'sultanahmet': 'Sultanahmet',
            'besiktas': 'Beşiktaş', 'beşiktaş': 'Beşiktaş',
            'atakoy': 'Ataköy', 'ataköy': 'Ataköy',
            'eminonu': 'Eminönü', 'eminönü': 'Eminönü',
            'karakoy': 'Karaköy', 'karaköy': 'Karaköy',
            'uskudar': 'Üsküdar', 'üsküdar': 'Üsküdar',
            'bakirkoy': 'Bakırköy', 'bakırköy': 'Bakırköy',
            'sisli': 'Şişli', 'şişli': 'Şişli',
            'beyoglu': 'Beyoğlu', 'beyoğlu': 'Beyoğlu',
            'ortakoy': 'Ortaköy', 'ortaköy': 'Ortaköy',
            'galata': 'Galata', 'levent': 'Levent', 'maslak': 'Maslak',
            'bebek': 'Bebek', 'fatih': 'Fatih', 'beykoz': 'Beykoz',
            'sariyer': 'Sarıyer', 'sarıyer': 'Sarıyer',
            'mecidiyeköy': 'Mecidiyeköy', 'mecidiyekoy': 'Mecidiyeköy',
        }
        
        found_locations = []
        for loc in locations:
            if loc in query_lower:
                normalized = location_map.get(loc, loc.title())
                if normalized not in [l[1] for l in found_locations]:
                    found_locations.append((query_lower.find(loc), normalized))
        
        # Sort by position in query
        found_locations.sort(key=lambda x: x[0])
        
        if len(found_locations) >= 2:
            # Check for "to X from Y" pattern (reversed)
            if ' from ' in query_lower:
                from_idx = query_lower.find(' from ')
                to_idx = query_lower.find(' to ')
                if to_idx != -1 and to_idx < from_idx:
                    # "to X from Y" pattern - second location is origin
                    return {
                        'origin': found_locations[1][1],
                        'destination': found_locations[0][1],
                        'use_gps_for_origin': False,
                        'confidence': 0.7
                    }
            
            # Default: first is origin, second is destination
            return {
                'origin': found_locations[0][1],
                'destination': found_locations[1][1],
                'use_gps_for_origin': False,
                'confidence': 0.7
            }
        elif len(found_locations) == 1:
            # Only destination found, use GPS for origin if available
            return {
                'origin': None,
                'destination': found_locations[0][1],
                'use_gps_for_origin': user_has_gps,
                'confidence': 0.6
            }
        
        return None


# Global instance
_intent_detector = None

def get_intent_detector() -> LLMIntentDetector:
    """Get global intent detector instance."""
    global _intent_detector
    if _intent_detector is None:
        _intent_detector = LLMIntentDetector()
    return _intent_detector

async def detect_query_intent(query: str, user_location: Optional[Dict] = None, binary_mode: bool = False) -> IntentResult:
    """
    Convenience function to detect query intent.
    
    Args:
        query: User's query text
        user_location: Optional user location context
        binary_mode: If True, use simplified binary classification (Transportation vs Others)
        
    Returns:
        IntentResult with classification
    """
    detector = get_intent_detector()
    return await detector.detect_intent(query, user_location, binary_mode)

async def extract_query_locations(query: str, user_has_gps: bool = False) -> Optional[Dict[str, Any]]:
    """
    Convenience function to extract locations from a transportation query.
    
    Args:
        query: User's query text
        user_has_gps: Whether user has GPS location available
        
    Returns:
        Dict with origin, destination, use_gps_for_origin, confidence
        or None if extraction fails
    """
    detector = get_intent_detector()
    return await detector.extract_locations(query, user_has_gps)
