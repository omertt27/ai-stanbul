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
        
    async def detect_intent(self, query: str, user_location: Optional[Dict] = None) -> IntentResult:
        """
        Detect the intent of a user query using LLM classification.
        
        Args:
            query: The user's query text
            user_location: Optional user location context
            
        Returns:
            IntentResult with classification details
        """
        # If RunPod LLM is not available, fallback to keyword detection immediately
        if not self.llm_available:
            logger.warning("🔄 RunPod LLM not available - using keyword fallback")
            return self._fallback_keyword_detection(query)
        
        try:
            # Construct the prompt for intent detection
            prompt = self._build_intent_detection_prompt(query, user_location)
            
            # Call RunPod LLM API
            response = await self._call_runpod_llm(prompt)
            
            if not response:
                raise Exception("Empty response from RunPod LLM")
            
            # Parse the response
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
            
            logger.info(f"🎯 RunPod LLM Intent Detection: {intent_type.value} (confidence: {confidence:.2f}) - {reasoning}")
            return result
            
        except Exception as e:
            logger.error(f"❌ RunPod LLM Intent Detection failed: {str(e)}")
            # Fallback to keyword-based detection
            return self._fallback_keyword_detection(query)
    
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
    
    async def _call_runpod_llm(self, prompt: str) -> str:
        """Call RunPod LLM API for intent detection."""
        try:
            # Use RunPod LLM client to generate response
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=200,  # Short response for classification
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

# Global instance
_intent_detector = None

def get_intent_detector() -> LLMIntentDetector:
    """Get global intent detector instance."""
    global _intent_detector
    if _intent_detector is None:
        _intent_detector = LLMIntentDetector()
    return _intent_detector

async def detect_query_intent(query: str, user_location: Optional[Dict] = None) -> IntentResult:
    """
    Convenience function to detect query intent.
    
    Args:
        query: User's query text
        user_location: Optional user location context
        
    Returns:
        IntentResult with classification
    """
    detector = get_intent_detector()
    return await detector.detect_intent(query, user_location)
