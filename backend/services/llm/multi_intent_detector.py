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
import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .llm_response_parser import parse_llm_json_response

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
            'timeout_seconds': 5,  # Fast timeout for multi-intent detection
            'max_retries': 1,  # Reduced retries for speed
            'fallback_enabled': True,
            'enable_fast_path': True,  # NEW: Skip LLM for simple queries
            **(config or {})
        }
        
        logger.info(f"MultiIntentDetector initialized with fast-path optimization")
    
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
            # FAST PATH: Check if query is obviously single-intent
            if self.config.get('enable_fast_path', True):
                if self._is_simple_query(query):
                    logger.info(f"Fast-path: Query appears single-intent, skipping LLM detection")
                    return self._single_intent_response(query, context)
            
            # Prepare context information (minimal for speed)
            context_info = ""
            if context and context.get("user_location"):
                context_info = f"\nUser location: {context['user_location']}"
            
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
            
            # Parse JSON response (handles both dict and string responses)
            data = parse_llm_json_response(llm_output)
            
            if data is None:
                # Fallback to single intent
                logger.error("LLM multi-intent detection failed: unable to parse response, using fallback")
                return MultiIntentDetection(
                    intents=[DetectedIntent(
                        intent_type="unknown",
                        parameters={},
                        priority=1,
                        confidence=0.6,
                        requires_location=False
                    )],
                    is_multi_intent=False,
                    intent_relationships=[],
                    execution_mode="sequential",
                    confidence=0.6
                )
            
            # Validate and sanitize data before creating MultiIntentDetection
            # Ensure intent_count is at least 1
            intent_count = data.get("intent_count", 1)
            if intent_count < 1:
                logger.warning(f"Invalid intent_count ({intent_count}), setting to 1")
                intent_count = 1
            
            # Ensure execution_strategy is valid
            valid_strategies = ["sequential", "parallel", "conditional", "mixed"]
            execution_strategy = data.get("execution_strategy", "sequential")
            if execution_strategy not in valid_strategies:
                logger.warning(f"Invalid execution_strategy ('{execution_strategy}'), defaulting to 'sequential'")
                execution_strategy = "sequential"
            
            # Ensure we have at least one intent
            intents_data = data.get("intents", [])
            if not intents_data:
                logger.warning("No intents in LLM response, using fallback")
                return self._fallback_detection(query, context)
            
            # Build DetectedIntent objects
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
                for i, intent_data in enumerate(intents_data)
            ]
            
            # Build IntentRelationship objects
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
                intent_count=intent_count,
                intents=intents,
                relationships=relationships,
                execution_strategy=execution_strategy,
                is_multi_intent=data.get("is_multi_intent", len(intents) > 1),
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
        """Get multi-intent detection instructions for LLM (OPTIMIZED FOR SPEED)."""
        return """Analyze query for multiple intents. Return JSON ONLY.

INTENT TYPES: route, restaurant, attraction, museum, event, weather, transport, shopping, nightlife, hotel, hidden_gems, general

JSON FORMAT:
{
  "intent_count": <number>,
  "is_multi_intent": <boolean>,
  "intents": [{"intent_type": "<type>", "parameters": {}, "priority": 1, "confidence": 0.9, "requires_location": false}],
  "relationships": [{"relationship_type": "sequential|parallel|dependent|conditional", "intent_indices": [0,1], "description": "..."}],
  "execution_strategy": "sequential|parallel|conditional|mixed",
  "confidence": 0.9
}

Examples:
- Single: "Show me restaurants" → intent_count=1, is_multi_intent=false
- Multi: "Route to museum and nearby restaurants" → intent_count=2, is_multi_intent=true, execution_strategy="sequential"

Be fast and accurate."""
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM via existing client with timeout.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            LLM response text
        """
        timeout = self.config['timeout_seconds']
        
        try:
            # Wrap in asyncio.wait_for for timeout enforcement
            async def _make_call():
                # Check if client has OpenAI-style interface
                if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                    # OpenAI-style client
                    response = await self.llm_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a fast travel query analyzer."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,  # Lower for speed
                        max_tokens=500  # Reduced for speed
                    )
                    return response.choices[0].message.content.strip()
                else:
                    # Generic LLM client with generate method
                    result = await self.llm_client.generate(
                        prompt=prompt,
                        max_tokens=500,  # Reduced for speed
                        temperature=0.2  # Lower for speed
                    )
                    # Handle both dict and string responses
                    if isinstance(result, dict):
                        return result.get('generated_text', '')
                    return str(result)
            
            # Apply timeout
            return await asyncio.wait_for(_make_call(), timeout=timeout)
            
        except asyncio.TimeoutError:
            logger.warning(f"Multi-intent LLM call timed out after {timeout}s")
            raise Exception(f"LLM timeout after {timeout}s")
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
    
    def _is_simple_query(self, query: str) -> bool:
        """
        Check if query is obviously single-intent (fast-path optimization).
        
        Looks for multi-intent markers like "and", "then", "also", multiple question marks, etc.
        
        Args:
            query: User query
            
        Returns:
            True if query appears to be single-intent
        """
        query_lower = query.lower()
        
        # Multi-intent markers
        multi_intent_markers = [
            " and ", " then ", " also ", " plus ",
            " after that", " first ", " second ",
            " both ", " or ", " either "
        ]
        
        # Count potential intents
        if any(marker in query_lower for marker in multi_intent_markers):
            return False
        
        # Multiple questions
        if query.count("?") > 1:
            return False
        
        # Very long queries are more likely to be multi-intent
        if len(query.split()) > 20:
            return False
        
        # Otherwise, assume single intent
        return True
    
    def _single_intent_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> MultiIntentDetection:
        """
        Create a single-intent response (fast-path).
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            MultiIntentDetection with single intent
        """
        intent_type = self._guess_intent_type(query.lower())
        
        return MultiIntentDetection(
            original_query=query,
            intent_count=1,
            intents=[DetectedIntent(
                intent_type=intent_type,
                parameters={},
                priority=1,
                confidence=0.8,
                requires_location=False
            )],
            relationships=[],
            execution_strategy="sequential",
            is_multi_intent=False,
            confidence=0.8,
            detection_method="fast_path",
            processing_time_ms=0.0
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
