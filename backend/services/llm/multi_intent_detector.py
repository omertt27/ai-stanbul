"""
Phase 4.3: Multi-Intent Detector

LLM-powered detection of multiple intents in a single query.
This module gives the LLM 95% control over understanding complex queries with multiple goals.

Examples:
- "Show me route to Hagia Sophia and find restaurants near there"
  → Intent 1: get_directions, Intent 2: find_restaurants (dependent)
  
- "What's the weather and show me nearby attractions?"
  → Intent 1: get_weather, Intent 2: find_attractions (parallel)
  
- "If it's not raining, plan a walking tour, otherwise show indoor museums"
  → Intent 1: get_weather, Intent 2: conditional (plan_tour OR find_museums)

Author: AI Istanbul Team
Date: December 2, 2025
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from .models import (
    MultiIntentDetection,
    DetectedIntent,
    IntentRelationship
)

logger = logging.getLogger(__name__)


class MultiIntentDetector:
    """
    LLM-powered multi-intent detector.
    
    Detects and analyzes queries with multiple intents, extracting
    parameters and relationships for orchestrated execution.
    
    LLM Responsibility: 95%
    - Intent detection and counting
    - Parameter extraction per intent
    - Relationship identification
    - Execution strategy recommendation
    
    Fallback: 5%
    - Simple "and" splitting for basic multi-intent
    - Sequential execution default
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-intent detector.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'timeout_seconds': 5,
            'max_retries': 2,
            'fallback_enabled': True,
            **(config or {})
        }
        
        logger.info(f"MultiIntentDetector initialized")
    
    async def detect_intents(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MultiIntentDetection:
        """
        Detect multiple intents in a query using LLM.
        
        Args:
            query: User query to analyze
            context: Optional conversation context
            
        Returns:
            MultiIntentDetection with all detected intents and relationships
        """
        start_time = time.time()
        
        try:
            # Prepare context information
            context_info = ""
            if context:
                if context.get("session_history"):
                    context_info += f"\nRecent conversation: {context['session_history'][-3:]}"
                if context.get("user_location"):
                    context_info += f"\nUser location: {context['user_location']}"
            
            # Build prompt
            prompt = f"""Analyze this query for multiple intents:

Query: "{query}"
{context_info}

Detect all intents, extract parameters, and identify relationships.

{self._get_detection_instructions()}

Return ONLY a valid JSON object with the structure specified above."""
            
            logger.info(f"Detecting intents for query: {query[:100]}...")
            
            # Call LLM via existing client
            llm_output = await self._call_llm(prompt)
            
            # Parse JSON response
            data = json.loads(llm_output)

            
            # Parse JSON response
            data = json.loads(llm_output)
            
            # Build MultiIntentDetection
            intents = [
                DetectedIntent(
                    intent_type=intent_data["intent_type"],
                    parameters=intent_data.get("parameters", {}),
                    priority=intent_data.get("priority", i + 1),
                    confidence=intent_data.get("confidence", 0.8),
                    requires_location=intent_data.get("requires_location", False),
                    depends_on=intent_data.get("depends_on"),
                    condition=intent_data.get("condition")
                )
                for i, intent_data in enumerate(data["intents"])
            ]
            
            relationships = [
                IntentRelationship(
                    relationship_type=rel_data["relationship_type"],
                    intent_indices=rel_data["intent_indices"],
                    description=rel_data["description"]
                )
                for rel_data in data.get("relationships", [])
            ]
            
            processing_time = (time.time() - start_time) * 1000
            
            result = MultiIntentDetection(
                original_query=query,
                intent_count=data["intent_count"],
                intents=intents,
                relationships=relationships,
                execution_strategy=data["execution_strategy"],
                is_multi_intent=data["is_multi_intent"],
                confidence=data.get("confidence", 0.85),
                detection_method="llm",
                processing_time_ms=processing_time
            )
            
            logger.info(
                f"Detected {result.intent_count} intent(s), "
                f"multi_intent={result.is_multi_intent}, "
                f"strategy={result.execution_strategy} "
                f"({processing_time:.0f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM multi-intent detection failed: {e}, using fallback")
            return self._fallback_detection(query, context)
    
    def _get_detection_instructions(self) -> str:
        """Get multi-intent detection instructions for LLM."""
        return """You are an expert at analyzing travel queries to detect multiple intents.

Your task is to:
1. Identify ALL intents in the query (primary and secondary)
2. Extract parameters for each intent
3. Detect relationships between intents (sequential, parallel, dependent, conditional)
4. Recommend execution strategy

INTENT TYPES:
- route: Navigation/directions
- restaurant: Restaurant recommendations
- information: POI information
- hidden_gems: Off-beaten-path recommendations
- event: Event information
- weather: Weather information
- museum: Museum information
- transport: Transportation info
- attraction: Tourist attractions
- shopping: Shopping recommendations
- nightlife: Nightlife recommendations
- hotel: Hotel recommendations
- general: General chat

RELATIONSHIPS:
- sequential: Execute one after another (e.g., "route then restaurants")
- parallel: Execute simultaneously (e.g., "weather and attractions")
- dependent: Second intent depends on first result (e.g., "route to X and find restaurants NEAR THAT location")
- conditional: Execute based on condition (e.g., "if sunny, outdoor tour; if rainy, museums")

RESPOND IN JSON:
{
  "intent_count": <number>,
  "is_multi_intent": <boolean>,
  "intents": [
    {
      "intent_type": "<type>",
      "parameters": {
        "origin": "...",
        "destination": "...",
        "category": "...",
        "time": "...",
        "preferences": "..."
      },
      "priority": 1,
      "confidence": 0.95,
      "requires_location": true/false,
      "depends_on": [0],
      "condition": null
    }
  ],
  "relationships": [
    {
      "relationship_type": "sequential|parallel|conditional|dependent",
      "intent_indices": [0, 1],
      "description": "Intent 2 needs location from Intent 1"
    }
  ],
  "execution_strategy": "sequential|parallel|conditional|mixed",
  "confidence": 0.9
}

IMPORTANT:
- If only ONE clear intent: is_multi_intent = false, intent_count = 1
- Mark depends_on when second intent uses first intent's result
- Use "dependent" relationship when location/data flows between intents
- Use "conditional" for if/else logic
- Be precise with parameters extraction"""
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM via existing client.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            LLM response text
        """
        try:
            # Check if client has OpenAI-style interface
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                # OpenAI-style client
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert travel query analyzer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    timeout=self.config['timeout_seconds']
                )
                return response.choices[0].message.content.strip()
            else:
                # Generic LLM client with generate method
                return await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.3
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _fallback_detection(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MultiIntentDetection:
        """
        Fallback multi-intent detection using simple heuristics.
        
        Uses basic "and" splitting and keyword detection.
        This is used when LLM fails (5% of cases).
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            MultiIntentDetection with basic intent splitting
        """
        query_lower = query.lower()
        
        # Check for "and" connectors
        and_variants = [" and ", " then ", " also ", " plus "]
        has_connector = any(variant in query_lower for variant in and_variants)
        
        if not has_connector:
            # Single intent
            intent_type = self._guess_intent_type(query_lower)
            return MultiIntentDetection(
                original_query=query,
                intent_count=1,
                intents=[
                    DetectedIntent(
                        intent_type=intent_type,
                        parameters={"query": query},
                        priority=1,
                        confidence=0.6,
                        requires_location=intent_type in ["route", "restaurant", "attraction"]
                    )
                ],
                relationships=[],
                execution_strategy="sequential",
                is_multi_intent=False,
                confidence=0.6,
                detection_method="fallback"
            )
        
        # Split on "and" and detect each part
        parts = [part.strip() for part in query_lower.split(" and ")]
        intents = []
        
        for i, part in enumerate(parts[:3]):  # Max 3 intents in fallback
            intent_type = self._guess_intent_type(part)
            intents.append(
                DetectedIntent(
                    intent_type=intent_type,
                    parameters={"query": part},
                    priority=i + 1,
                    confidence=0.5,
                    requires_location=intent_type in ["route", "restaurant", "attraction"],
                    depends_on=[i - 1] if i > 0 else None  # Simple sequential dependency
                )
            )
        
        return MultiIntentDetection(
            original_query=query,
            intent_count=len(intents),
            intents=intents,
            relationships=[
                IntentRelationship(
                    relationship_type="sequential",
                    intent_indices=list(range(len(intents))),
                    description="Simple sequential execution (fallback)"
                )
            ],
            execution_strategy="sequential",
            is_multi_intent=len(intents) > 1,
            confidence=0.5,
            detection_method="fallback"
        )
    
    def _guess_intent_type(self, query: str) -> str:
        """
        Guess intent type from query keywords (fallback only).
        
        Args:
            query: Query text (lowercase)
            
        Returns:
            Guessed intent type
        """
        # Route keywords
        if any(kw in query for kw in ["route", "direction", "way to", "how to get", "navigate"]):
            return "route"
        
        # Restaurant keywords
        if any(kw in query for kw in ["restaurant", "eat", "food", "dining", "lunch", "dinner"]):
            return "restaurant"
        
        # Weather keywords
        if any(kw in query for kw in ["weather", "temperature", "rain", "sunny", "forecast"]):
            return "weather"
        
        # Attraction keywords
        if any(kw in query for kw in ["attraction", "sight", "tourist", "visit", "see"]):
            return "attraction"
        
        # Museum keywords
        if any(kw in query for kw in ["museum", "gallery", "exhibition"]):
            return "museum"
        
        # Hidden gems keywords
        if any(kw in query for kw in ["hidden", "secret", "local", "off the beaten"]):
            return "hidden_gems"
        
        # Event keywords
        if any(kw in query for kw in ["event", "concert", "festival", "show", "performance"]):
            return "event"
        
        # Transport keywords
        if any(kw in query for kw in ["metro", "bus", "tram", "ferry", "transport", "ticket"]):
            return "transport"
        
        # Default to general
        return "general"


# Singleton instance
_detector_instance: Optional[MultiIntentDetector] = None


def get_multi_intent_detector(llm_client, config: Optional[Dict[str, Any]] = None) -> MultiIntentDetector:
    """
    Get or create the singleton MultiIntentDetector instance.
    
    Args:
        llm_client: LLM API client
        config: Optional configuration
        
    Returns:
        MultiIntentDetector instance
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = MultiIntentDetector(llm_client=llm_client, config=config)
    return _detector_instance


async def detect_multi_intent(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    llm_client=None
) -> MultiIntentDetection:
    """
    Convenience function to detect multiple intents.
    
    Args:
        query: User query
        context: Optional conversation context
        llm_client: LLM client (uses singleton if provided once)
        
    Returns:
        MultiIntentDetection result
    """
    if llm_client:
        detector = get_multi_intent_detector(llm_client)
    else:
        detector = _detector_instance
        if detector is None:
            raise ValueError("MultiIntentDetector not initialized. Provide llm_client.")
    
    return await detector.detect_intents(query, context)
