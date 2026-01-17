"""
Unified LLM Service
Single entry point for ALL LLM operations in Istanbul AI

Integrates:
- RunPod LLM Client (existing)
- Prompt Builder (existing)  
- Shared Cache (new)
- Metrics (new)

Author: Istanbul AI Team
Date: January 17, 2026
"""

import os
import sys
import logging
import time
import hashlib
import json
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import existing components
try:
    from services.runpod_llm_client import RunPodLLMClient, get_llm_client
    from services.llm.prompts import PromptBuilder
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  Could not import backend components: {e}")
    COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class IntentClassificationPrompts:
    """
    Centralized intent classification prompts
    Extracted from istanbul_ai/routing/llm_intent_classifier.py
    """
    
    # Supported intents (from llm_intent_classifier.py)
    SUPPORTED_INTENTS = [
        'restaurant', 'attraction', 'transportation', 'weather', 'events',
        'neighborhood', 'shopping', 'hidden_gems', 'airport_transport',
        'route_planning', 'museum_route_planning', 'gps_route_planning',
        'nearby_locations', 'greeting', 'general'
    ]
    
    @staticmethod
    def get_llama3_intent_prompt(message: str, entities: Optional[Dict] = None, language: str = 'en') -> str:
        """
        Llama 3.1 8B optimized intent classification prompt
        
        Args:
            message: User message to classify
            entities: Optional extracted entities
            language: User language
            
        Returns:
            Formatted prompt for Llama 3.1
        """
        # Build entity context if provided
        entity_context = ""
        if entities:
            entity_items = [f"- {k}: {v}" for k, v in entities.items() if v]
            if entity_items:
                entity_context = "Entities found:\\n" + "\\n".join(entity_items) + "\\n\\n"
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an intent classification system for Istanbul travel queries. Analyze user messages and return ONLY valid JSON.

Output format: {{"primary_intent":"intent_name","confidence":0.95,"all_intents":["intent_name"]}}

Supported intents: {', '.join(IntentClassificationPrompts.SUPPORTED_INTENTS)}

Classification rules:
- greeting: Hello, hi, thanks, goodbye, how are you, good morning/evening, greetings in any language
- restaurant: Food, dining, eat, cuisine queries
- attraction: Sightseeing, landmarks, places to visit
- transportation: How to get, metro, bus, taxi, ferry
- weather: Weather, temperature, forecast
- general: Everything else

Examples:
"Hello!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Merhaba!" -> {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Find restaurant" -> {{"primary_intent":"restaurant","confidence":0.90,"all_intents":["restaurant"]}}
"Weather today?" -> {{"primary_intent":"weather","confidence":0.90,"all_intents":["weather"]}}
"How to Taksim?" -> {{"primary_intent":"transportation","confidence":0.90,"all_intents":["transportation"]}}
<|eot_id|><|start_header_id|>user<|end_header_id|>

{entity_context}Classify: "{message}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    @staticmethod
    def get_tinyllama_intent_prompt(message: str) -> str:
        """
        TinyLlama optimized prompt (compact format)
        
        Args:
            message: User message
            
        Returns:
            Compact prompt for TinyLlama
        """
        prompt = f"""Classify intent. Output only JSON: {{"primary_intent":"X","confidence":0.9,"all_intents":["X"]}}

"Hello!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Merhaba!" = {{"primary_intent":"greeting","confidence":0.95,"all_intents":["greeting"]}}
"Weather?" = {{"primary_intent":"weather","confidence":0.90,"all_intents":["weather"]}}
"Find restaurant" = {{"primary_intent":"restaurant","confidence":0.90,"all_intents":["restaurant"]}}

"{message}" = """
        return prompt


class UnifiedLLMService:
    """
    Unified LLM Service - Single source of truth for LLM operations
    
    Features:
    - Wraps RunPod LLM client
    - Uses centralized prompts
    - Shared caching across all components
    - Unified metrics tracking
    - Feature flag support
    
    Usage:
        from unified_system import get_unified_llm
        
        llm = get_unified_llm()
        result = llm.classify_intent("recommend a restaurant", language="en")
    """
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Feature flags
        self.enabled = os.getenv('USE_UNIFIED_LLM_SERVICE', 'false').lower() == 'true'
        self.cache_enabled = os.getenv('UNIFIED_CACHE_ENABLED', 'true').lower() == 'true'
        self.cache_ttl = int(os.getenv('UNIFIED_CACHE_TTL', '3600'))  # 1 hour
        
        # Core components
        if COMPONENTS_AVAILABLE:
            self.llm_client = get_llm_client()  # Reuse existing RunPod client
            self.prompt_builder = PromptBuilder()  # Reuse existing prompt system
            logger.info("✅ Unified LLM Service - components loaded")
        else:
            self.llm_client = None
            self.prompt_builder = None
            logger.warning("⚠️  Unified LLM Service - components not available")
        
        # New components
        self.cache = {}  # Simple dict cache for now (TODO: upgrade to Redis)
        self.cache_timestamps = {}
        self.metrics = self._init_metrics()
        
        self._initialized = True
        
        if self.enabled:
            logger.info("✅ Unified LLM Service initialized (ENABLED)")
        else:
            logger.info("ℹ️  Unified LLM Service initialized (DISABLED - set USE_UNIFIED_LLM_SERVICE=true to enable)")
    
    def _init_metrics(self) -> Dict[str, Any]:
        """Initialize metrics collection"""
        return {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'llm_calls': 0,
            'total_latency_ms': 0.0,
            'by_intent': {},
            'by_component': {},
            'by_operation': {}
        }
    
    async def classify_intent(
        self,
        message: str,
        entities: Optional[Dict] = None,
        language: str = 'en',
        component: str = 'unknown',
        model_type: str = 'llama3'
    ) -> Dict[str, Any]:
        """
        Classify intent using RunPod LLM
        
        Args:
            message: User message
            entities: Extracted entities (optional)
            language: User language
            component: Calling component (for metrics)
            model_type: 'llama3' or 'tinyllama'
            
        Returns:
            Intent classification result with keys:
            - primary_intent: str
            - confidence: float
            - all_intents: List[str]
        """
        if not self.llm_client:
            logger.warning("LLM client not available")
            return {'primary_intent': 'general', 'confidence': 0.5, 'all_intents': ['general']}
        
        # Check cache first
        if self.cache_enabled:
            cache_key = self._cache_key('intent', message, language)
            cached = self._get_from_cache(cache_key)
            if cached:
                self.metrics['cache_hits'] += 1
                self.metrics['total_calls'] += 1
                logger.debug(f"💾 Cache HIT: {cache_key[:16]}...")
                return cached
            self.metrics['cache_misses'] += 1
        
        # Get prompt based on model type
        if model_type == 'llama3':
            prompt = IntentClassificationPrompts.get_llama3_intent_prompt(message, entities, language)
        else:
            prompt = IntentClassificationPrompts.get_tinyllama_intent_prompt(message)
        
        # Call RunPod LLM (async)
        start = time.time()
        try:
            llm_result = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.2
            )
            
            # Extract text from LLM result
            if llm_result is None:
                logger.error("❌ LLM returned None")
                return {'primary_intent': 'general', 'confidence': 0.5, 'all_intents': ['general']}
            
            raw_response = llm_result.get('generated_text')
            if raw_response is None:
                logger.error(f"❌ No generated_text in LLM result: {llm_result.keys()}")
                return {'primary_intent': 'general', 'confidence': 0.5, 'all_intents': ['general']}
            
            if raw_response == '':
                logger.error(f"❌ LLM returned empty text (model may not be generating properly)")
                logger.error(f"   Raw result: {llm_result}")
                return {'primary_intent': 'general', 'confidence': 0.5, 'all_intents': ['general']}
                
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return {'primary_intent': 'general', 'confidence': 0.5, 'all_intents': ['general']}
        
        latency_ms = (time.time() - start) * 1000
        
        # Parse result
        parsed = self._parse_intent_response(raw_response)
        
        # Update metrics
        self.metrics['total_calls'] += 1
        self.metrics['llm_calls'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        self.metrics['by_component'][component] = self.metrics['by_component'].get(component, 0) + 1
        self.metrics['by_operation']['classify_intent'] = self.metrics['by_operation'].get('classify_intent', 0) + 1
        
        intent = parsed.get('primary_intent', 'general')
        self.metrics['by_intent'][intent] = self.metrics['by_intent'].get(intent, 0) + 1
        
        # Cache result
        if self.cache_enabled:
            self._set_cache(cache_key, parsed)
        
        logger.info(f"🤖 LLM intent: {intent} (conf:{parsed.get('confidence', 0):.2f}, {latency_ms:.0f}ms)")
        
        return parsed
    
    async def generate_response(
        self,
        query: str,
        intent: str,
        context: Dict[str, Any],
        language: str = 'en',
        component: str = 'unknown'
    ) -> str:
        """
        Generate response using RunPod LLM
        
        Args:
            query: User query
            intent: Detected intent
            context: Context data (database results, RAG, etc.)
            language: Response language
            component: Calling component
            
        Returns:
            Generated response text
        """
        if not self.llm_client or not self.prompt_builder:
            logger.warning("LLM components not available")
            return "I'm here to help! What would you like to know about Istanbul?"
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._cache_key('response', query, intent, language)
            cached = self._get_from_cache(cache_key)
            if cached:
                self.metrics['cache_hits'] += 1
                self.metrics['total_calls'] += 1
                return cached
            self.metrics['cache_misses'] += 1
        
        # Build prompt using existing PromptBuilder
        prompt = self.prompt_builder.build_prompt(
            query=query,
            signals={'detected_intent': intent},
            context=context,
            language=language
        )
        
        # Call RunPod LLM (async)
        start = time.time()
        try:
            llm_result = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.7
            )
            
            # Extract text from LLM result
            if not llm_result:
                logger.error("❌ LLM returned None")
                return "I'm having trouble generating a response. Please try again."
            
            result = llm_result.get('generated_text', '')
            if not result:
                logger.error(f"❌ No generated_text in LLM result: {llm_result.keys()}")
                return "I'm having trouble generating a response. Please try again."
                
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return "I'm having trouble generating a response. Please try again."
        
        latency_ms = (time.time() - start) * 1000
        
        # Update metrics
        self.metrics['total_calls'] += 1
        self.metrics['llm_calls'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        self.metrics['by_intent'][intent] = self.metrics['by_intent'].get(intent, 0) + 1
        self.metrics['by_component'][component] = self.metrics['by_component'].get(component, 0) + 1
        self.metrics['by_operation']['generate_response'] = self.metrics['by_operation'].get('generate_response', 0) + 1
        
        # Cache result
        if self.cache_enabled:
            self._set_cache(cache_key, result)
        
        logger.info(f"🤖 LLM response: {len(result)} chars ({latency_ms:.0f}ms, intent:{intent})")
        
        return result
    
    def _cache_key(self, operation: str, *args) -> str:
        """Generate cache key from operation and arguments"""
        key_str = f"{operation}:" + ":".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self.cache:
            return None
        
        # Check TTL
        timestamp = self.cache_timestamps.get(key, 0)
        if time.time() - timestamp > self.cache_ttl:
            # Expired
            del self.cache[key]
            del self.cache_timestamps[key]
            return None
        
        return self.cache[key]
    
    def _set_cache(self, key: str, value: Any):
        """Set value in cache with timestamp"""
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()
    
    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM intent classification response
        
        Expects JSON: {"primary_intent": "X", "confidence": 0.9, "all_intents": ["X"]}
        """
        try:
            # Try to parse as JSON
            result = json.loads(response.strip())
            
            # Validate required fields
            if 'primary_intent' not in result:
                result['primary_intent'] = 'general'
            if 'confidence' not in result:
                result['confidence'] = 0.5
            if 'all_intents' not in result:
                result['all_intents'] = [result['primary_intent']]
            
            return result
        except (json.JSONDecodeError, AttributeError):
            # Fallback: try to extract intent from text
            logger.warning(f"⚠️  Failed to parse LLM response as JSON: {response[:100]}")
            return {
                'primary_intent': 'general',
                'confidence': 0.5,
                'all_intents': ['general']
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        total = max(self.metrics['total_calls'], 1)  # Avoid division by zero
        llm_calls = self.metrics['llm_calls']
        
        return {
            **self.metrics,
            'cache_hit_rate': (self.metrics['cache_hits'] / total) * 100,
            'cache_size': len(self.cache),
            'avg_latency_ms': self.metrics['total_latency_ms'] / llm_calls if llm_calls > 0 else 0,
            'llm_call_rate': (llm_calls / total) * 100
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = self._init_metrics()
        logger.info("📊 Metrics reset")
    
    def clear_cache(self):
        """Clear all cached data"""
        size = len(self.cache)
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info(f"🗑️  Cache cleared ({size} items removed)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'hits': self.metrics['cache_hits'],
            'misses': self.metrics['cache_misses'],
            'hit_rate': (self.metrics['cache_hits'] / max(self.metrics['total_calls'], 1)) * 100,
            'ttl': self.cache_ttl,
            'enabled': self.cache_enabled
        }


# Global instance (singleton)
_unified_llm = None

def get_unified_llm() -> UnifiedLLMService:
    """
    Get or create unified LLM service singleton
    
    Returns:
        UnifiedLLMService instance
    """
    global _unified_llm
    if _unified_llm is None:
        _unified_llm = UnifiedLLMService()
    return _unified_llm
