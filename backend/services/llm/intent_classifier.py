"""
LLM Intent Classifier - Phase 1 Implementation

This module implements LLM-powered intent classification that runs BEFORE
any handler routing. It uses the LLM to understand user intent, extract
locations, detect preferences, and provide confidence scores.

This is the foundation of the LLM-First architecture, giving the LLM a
central role in understanding user queries.

Author: AI Istanbul Team
Date: 2024
"""

import asyncio
import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import (
    IntentClassification,
    QueryAnalysis,
    LLMPromptTemplate,
    CachedIntentResult,
    LocationMatch,
    LocationResolution
)

logger = logging.getLogger(__name__)


class LLMIntentClassifier:
    """
    LLM-powered intent classification service.
    
    This service uses a lightweight LLM call to classify user intent,
    extract entities, and provide routing recommendations BEFORE any
    specialized handler is invoked.
    
    Features:
    - Intelligent intent classification (route, info, restaurant, etc.)
    - Location extraction with fuzzy matching
    - User preference detection
    - Multi-intent handling
    - Confidence scoring
    - Semantic caching for performance
    - Graceful fallback to regex/keyword matching
    
    Architecture:
    ```
    User Query + Context
           ↓
    [Cache Check] → Hit? → Return cached result
           ↓ Miss
    [LLM Intent Classification]
           ↓
    [Validation & Confidence Scoring]
           ↓
    [Cache Result]
           ↓
    Return IntentClassification
    ```
    """
    
    # Intent classification prompt template
    INTENT_CLASSIFICATION_PROMPT = LLMPromptTemplate(
        name="intent_classification",
        template="""You are an expert travel assistant AI for Istanbul tourism.

Analyze this user query and extract structured information in JSON format.

**User Query**: "{query}"

**User Context**:
- GPS Available: {has_gps}
- GPS Location: {gps_location}
- Recent Queries: {recent_queries}
- User Preferences: {user_preferences}

**Your Task**:
Extract and classify the following in valid JSON format:

1. **primary_intent**: Choose ONE from:
   - "route" (wants directions/navigation)
   - "restaurant" (wants restaurant recommendations)
   - "information" (wants info about POIs, attractions, history)
   - "hidden_gems" (wants off-the-beaten-path, local spots)
   - "event" (wants event information)
   - "weather" (wants weather information)
   - "museum" (wants museum information)
   - "transport" (wants transportation info)
   - "general" (general chat/greeting/unclear)
   - "multi_intent" (multiple intents detected)

2. **secondary_intents**: Array of additional intents (if multi_intent)

3. **origin**: Origin location name (or null if not specified or using GPS)

4. **destination**: Destination location name (or null if not specified)

5. **entities**: Object with extracted entities:
   - time: time constraints (e.g., "morning", "evening", "2 hours")
   - date: date information
   - count: numbers (e.g., "3 restaurants", "top 5")
   - category: category/type (e.g., "seafood", "Ottoman", "contemporary")

6. **user_preferences**: Object with detected preferences:
   - budget: "cheap", "moderate", "expensive", or null
   - accessibility: true if wheelchair/accessibility mentioned
   - family_friendly: true if family/kids mentioned
   - vegetarian: true if vegetarian/vegan mentioned
   - popular: true if wants popular/famous places
   - quiet: true if wants quiet/peaceful places
   - cultural: true if interested in culture/history
   - outdoor: true if wants outdoor activities

7. **confidence**: Float 0-1 (how confident you are in classification)

8. **ambiguities**: Array of unclear/ambiguous parts

**Response Format** (valid JSON only, no markdown):
{{
  "primary_intent": "route",
  "secondary_intents": [],
  "origin": "Sultanahmet",
  "destination": "Taksim Square",
  "entities": {{"time": "evening", "count": null, "category": null}},
  "user_preferences": {{"budget": "moderate", "popular": true}},
  "confidence": 0.95,
  "ambiguities": []
}}

**Important**:
- Return ONLY valid JSON, no markdown, no explanation
- If location unclear, set to null
- If GPS is available and no origin specified, user likely wants route from current location
- Turkish locations: normalize to English names (Ayasofya → Hagia Sophia)
- Confidence < 0.6 means ambiguous query""",
        variables=["query", "has_gps", "gps_location", "recent_queries", "user_preferences"],
        system_message="You are a precise JSON-generating assistant. Always return valid JSON.",
        temperature=0.3,  # Low temperature for consistent classification
        max_tokens=500
    )
    
    def __init__(
        self,
        llm_client,
        db_connection=None,
        cache_manager=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LLM Intent Classifier.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            db_connection: Database connection for logging
            cache_manager: Cache manager for semantic caching
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        self.db = db_connection
        self.cache = cache_manager
        
        # Configuration
        self.config = {
            'enable_caching': True,
            'cache_ttl_seconds': 3600,  # 1 hour cache
            'confidence_threshold': 0.6,  # Below this = ambiguous
            'max_retries': 2,
            'timeout_seconds': 5,
            'fallback_to_regex': True,
            'log_classifications': True,
            **(config or {})
        }
        
        # Statistics
        self.stats = {
            'total_classifications': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'fallback_used': 0,
            'average_latency_ms': 0.0
        }
        
        logger.info("✅ LLM Intent Classifier initialized")
    
    async def classify_intent(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> IntentClassification:
        """
        Classify user intent using LLM.
        
        This is the main entry point for intent classification. It checks
        the cache first, then calls the LLM, validates the response, and
        caches the result.
        
        Args:
            query: User query string
            user_context: Optional user context (GPS, history, preferences)
            use_cache: Whether to use caching (default True)
            
        Returns:
            IntentClassification: Structured intent classification
            
        Example:
            >>> classifier = LLMIntentClassifier(llm_client)
            >>> intent = await classifier.classify_intent(
            ...     "How do I get to Hagia Sophia from Taksim?",
            ...     user_context={'gps': {'lat': 41.0082, 'lon': 28.9784}}
            ... )
            >>> print(intent.primary_intent)  # "route"
            >>> print(intent.confidence)  # 0.95
        """
        start_time = time.time()
        
        try:
            # Normalize context
            context = user_context or {}
            has_gps = 'gps' in context or 'user_location' in context or 'location' in context
            
            # Check cache first
            if use_cache and self.config['enable_caching'] and self.cache:
                cached = await self._check_cache(query, context)
                if cached:
                    self.stats['cache_hits'] += 1
                    logger.info(f"✅ Cache hit for query: '{query[:50]}...'")
                    return cached
                self.stats['cache_misses'] += 1
            
            # Call LLM for classification
            logger.info(f"🤖 Calling LLM for intent classification: '{query[:50]}...'")
            llm_result = await self._call_llm_for_intent(query, context)
            
            # Parse and validate LLM response
            intent = self._parse_llm_response(llm_result, query, has_gps)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            intent.processing_time_ms = processing_time
            
            # Update statistics
            self.stats['total_classifications'] += 1
            self.stats['llm_calls'] += 1
            self._update_average_latency(processing_time)
            
            # Cache the result
            if use_cache and self.config['enable_caching'] and self.cache:
                await self._cache_result(query, context, intent)
            
            # Log if enabled
            if self.config['log_classifications']:
                self._log_classification(query, intent, context)
            
            logger.info(
                f"✅ Intent classified: {intent.primary_intent} "
                f"(confidence: {intent.confidence:.2f}, "
                f"time: {processing_time:.0f}ms)"
            )
            
            return intent
            
        except Exception as e:
            logger.error(f"❌ Intent classification failed: {e}")
            
            # Fallback to regex-based classification
            if self.config['fallback_to_regex']:
                logger.info("🔄 Falling back to regex-based classification")
                self.stats['fallback_used'] += 1
                return self._fallback_classification(query, context)
            
            raise
    
    async def _call_llm_for_intent(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Call LLM for intent classification.
        
        Args:
            query: User query
            context: User context
            
        Returns:
            str: LLM response (JSON string)
        """
        # Prepare prompt variables
        gps_location = "Not available"
        if 'gps' in context:
            gps = context['gps']
            gps_location = f"{{lat: {gps.get('lat')}, lon: {gps.get('lon')}}}"
        elif 'user_location' in context:
            gps = context['user_location']
            gps_location = f"{{lat: {gps.get('lat')}, lon: {gps.get('lon')}}}"
        elif 'location' in context:
            gps = context['location']
            gps_location = f"{{lat: {gps.get('lat')}, lon: {gps.get('lon')}}}"
        
        recent_queries = context.get('recent_queries', [])
        recent_queries_str = ", ".join(recent_queries[-3:]) if recent_queries else "None"
        
        user_prefs = context.get('preferences', {})
        user_prefs_str = json.dumps(user_prefs) if user_prefs else "{}"
        
        # Render prompt
        prompt = self.INTENT_CLASSIFICATION_PROMPT.render(
            query=query,
            has_gps="Yes" if ('gps' in context or 'user_location' in context or 'location' in context) else "No",
            gps_location=gps_location,
            recent_queries=recent_queries_str,
            user_preferences=user_prefs_str
        )
        
        # Call LLM with timeout
        try:
            # Use existing LLM client (RunPod or OpenAI)
            if hasattr(self.llm_client, 'generate'):
                # RunPod client
                response = await asyncio.wait_for(
                    self.llm_client.generate(
                        prompt,
                        temperature=self.INTENT_CLASSIFICATION_PROMPT.temperature,
                        max_tokens=self.INTENT_CLASSIFICATION_PROMPT.max_tokens
                    ),
                    timeout=self.config['timeout_seconds']
                )
            elif hasattr(self.llm_client, 'chat'):
                # OpenAI-style client
                response = await asyncio.wait_for(
                    self.llm_client.chat(
                        messages=[
                            {"role": "system", "content": self.INTENT_CLASSIFICATION_PROMPT.system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.INTENT_CLASSIFICATION_PROMPT.temperature,
                        max_tokens=self.INTENT_CLASSIFICATION_PROMPT.max_tokens
                    ),
                    timeout=self.config['timeout_seconds']
                )
            else:
                raise ValueError("Unsupported LLM client")
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"⏱️ LLM call timed out after {self.config['timeout_seconds']}s")
            raise
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            raise
    
    def _parse_llm_response(
        self,
        llm_response: str,
        original_query: str,
        has_gps: bool
    ) -> IntentClassification:
        """
        Parse and validate LLM response into IntentClassification.
        
        Args:
            llm_response: Raw LLM response (should be JSON)
            original_query: Original user query
            has_gps: Whether GPS is available
            
        Returns:
            IntentClassification: Validated intent classification
        """
        try:
            # Clean response (remove markdown code blocks if present)
            response = llm_response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Create IntentClassification
            intent = IntentClassification(
                primary_intent=data.get('primary_intent', 'general'),
                secondary_intents=data.get('secondary_intents', []),
                origin=data.get('origin'),
                destination=data.get('destination'),
                entities=data.get('entities', {}),
                user_preferences=data.get('user_preferences', {}),
                confidence=data.get('confidence', 0.5),
                ambiguities=data.get('ambiguities', []),
                original_query=original_query,
                has_gps=has_gps,
                classification_method="llm"
            )
            
            return intent
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {llm_response[:200]}")
            raise ValueError(f"Invalid JSON from LLM: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to create IntentClassification: {e}")
            raise
    
    def _fallback_classification(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> IntentClassification:
        """
        Fallback to regex/keyword-based classification when LLM fails.
        
        This provides graceful degradation when the LLM is unavailable.
        
        Args:
            query: User query
            context: User context
            
        Returns:
            IntentClassification: Basic intent classification
        """
        q_lower = query.lower()
        has_gps = 'gps' in context or 'user_location' in context or 'location' in context
        
        # Detect intent using keywords
        intent = "general"
        confidence = 0.5
        
        # Route/navigation keywords
        if any(kw in q_lower for kw in ['from', ' to ', 'route', 'directions', 'how to get', 'navigate', 'way to']):
            intent = "route"
            confidence = 0.7
        
        # Restaurant keywords
        elif any(kw in q_lower for kw in ['restaurant', 'eat', 'food', 'dinner', 'lunch', 'breakfast', 'cafe']):
            intent = "restaurant"
            confidence = 0.7
        
        # Information keywords
        elif any(kw in q_lower for kw in ['what are', 'tell me about', 'information', 'about', 'history']):
            intent = "information"
            confidence = 0.6
        
        # Hidden gems keywords
        elif any(kw in q_lower for kw in ['hidden', 'local', 'secret', 'off the beaten', 'authentic']):
            intent = "hidden_gems"
            confidence = 0.7
        
        # Museum keywords
        elif any(kw in q_lower for kw in ['museum', 'gallery', 'exhibition']):
            intent = "museum"
            confidence = 0.7
        
        # Weather keywords
        elif any(kw in q_lower for kw in ['weather', 'temperature', 'rain', 'sunny']):
            intent = "weather"
            confidence = 0.8
        
        logger.warning(f"⚠️ Using fallback classification: {intent} (confidence: {confidence})")
        
        return IntentClassification(
            primary_intent=intent,
            secondary_intents=[],
            origin=None,
            destination=None,
            entities={},
            user_preferences={},
            confidence=confidence,
            ambiguities=["Fallback classification used - LLM unavailable"],
            original_query=query,
            has_gps=has_gps,
            classification_method="fallback"
        )
    
    async def _check_cache(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Optional[IntentClassification]:
        """Check cache for existing classification."""
        if not self.cache:
            return None
        
        try:
            cache_key = self._generate_cache_key(query, context)
            cached = await self.cache.get(cache_key)
            
            if cached and isinstance(cached, CachedIntentResult):
                if not cached.is_expired():
                    cached.record_hit()
                    return cached.intent
                else:
                    # Expired, remove from cache
                    await self.cache.delete(cache_key)
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
            return None
    
    async def _cache_result(
        self,
        query: str,
        context: Dict[str, Any],
        intent: IntentClassification
    ):
        """Cache classification result."""
        if not self.cache:
            return
        
        try:
            cache_key = self._generate_cache_key(query, context)
            cached_result = CachedIntentResult(
                query_hash=cache_key,
                intent=intent,
                ttl_seconds=self.config['cache_ttl_seconds']
            )
            
            await self.cache.set(
                cache_key,
                cached_result,
                ttl=self.config['cache_ttl_seconds']
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key from query and context."""
        # Create hash from query + GPS availability
        has_gps = 'gps' in context or 'user_location' in context or 'location' in context
        cache_input = f"{query.lower().strip()}|gps:{has_gps}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _log_classification(
        self,
        query: str,
        intent: IntentClassification,
        context: Dict[str, Any]
    ):
        """Log classification to database for analytics."""
        try:
            if self.db:
                # Log to database (implement based on your schema)
                logger.debug(f"Logging classification: {intent.primary_intent}")
                # TODO: Implement database logging
        except Exception as e:
            logger.warning(f"Failed to log classification: {e}")
    
    def _update_average_latency(self, latency_ms: float):
        """Update average latency statistic."""
        n = self.stats['llm_calls']
        if n == 1:
            self.stats['average_latency_ms'] = latency_ms
        else:
            # Running average
            current_avg = self.stats['average_latency_ms']
            self.stats['average_latency_ms'] = (current_avg * (n - 1) + latency_ms) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return {
            **self.stats,
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
                else 0.0
            )
        }


# Singleton instance
_classifier_instance: Optional[LLMIntentClassifier] = None


def get_intent_classifier(
    llm_client=None,
    db_connection=None,
    cache_manager=None,
    config: Optional[Dict[str, Any]] = None
) -> LLMIntentClassifier:
    """
    Get or create LLM Intent Classifier singleton.
    
    Args:
        llm_client: LLM client (required on first call)
        db_connection: Database connection
        cache_manager: Cache manager
        config: Configuration overrides
        
    Returns:
        LLMIntentClassifier: Singleton instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        if llm_client is None:
            raise ValueError("llm_client is required on first initialization")
        
        _classifier_instance = LLMIntentClassifier(
            llm_client=llm_client,
            db_connection=db_connection,
            cache_manager=cache_manager,
            config=config
        )
    
    return _classifier_instance
