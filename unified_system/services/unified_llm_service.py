"""
Unified LLM Service
Single entry point for ALL LLM operations in Istanbul AI

Integrates:
- RunPod LLM Client (existing)
- Prompt Builder (existing)  
- Shared Cache (new)
- Metrics (new)
- Circuit Breaker (new)
- Streaming Support (new)

Author: Istanbul AI Team
Date: January 17, 2026
Last Updated: January 18, 2026 - Week 2 Integration
"""

import os
import sys
import logging
import time
import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import existing components
try:
    # Try both import patterns
    try:
        from services.runpod_llm_client import RunPodLLMClient, get_llm_client
        from services.llm.prompts import PromptBuilder
    except ImportError:
        # Alternative import path (when called from backend)
        import sys
        sys.path.insert(0, str(backend_path))
        from runpod_llm_client import RunPodLLMClient, get_llm_client
        from llm.prompts import PromptBuilder
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️  Could not import backend components: {e}")
    COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes to close circuit from half-open
    timeout_seconds: int = 60  # Time to wait before trying half-open
    

class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for LLM calls
    
    Prevents cascading failures by:
    - Tracking consecutive failures
    - Opening circuit after threshold failures
    - Allowing periodic retry attempts
    - Closing circuit after successful recoveries
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_changes = []
        
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
                    return True
            return False
        
        # HALF_OPEN: allow one request through
        return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                self.success_count = 0
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif self.failure_count >= self.config.failure_threshold:
            self._transition_to(CircuitState.OPEN)
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self.state
        self.state = new_state
        self.state_changes.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': datetime.now().isoformat(),
            'failure_count': self.failure_count
        })
        logger.warning(f"🔌 Circuit breaker: {old_state.value} → {new_state.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'recent_transitions': self.state_changes[-5:]  # Last 5 transitions
        }


# ============================================================================
# Enhanced Metrics
# ============================================================================

@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call"""
    timestamp: datetime
    operation: str  # classify_intent, generate_response, complete, stream
    component: str  # caller component
    intent: Optional[str] = None
    cache_hit: bool = False
    latency_ms: float = 0.0
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


class MetricsCollector:
    """
    Centralized metrics collection for UnifiedLLMService
    
    Tracks:
    - Call counts by operation, component, intent
    - Cache hit rates
    - Latency percentiles
    - Error rates
    - Token usage
    """
    
    def __init__(self):
        self.calls: List[LLMCallMetrics] = []
        self.max_history = 10000  # Keep last 10k calls
        
    def record_call(self, metrics: LLMCallMetrics):
        """Record a call with metrics"""
        self.calls.append(metrics)
        
        # Trim history if too large
        if len(self.calls) > self.max_history:
            self.calls = self.calls[-self.max_history:]
    
    def get_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_calls = [c for c in self.calls if c.timestamp >= cutoff]
        
        if not recent_calls:
            return {
                'period_minutes': minutes,
                'total_calls': 0,
                'cache_hit_rate': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'error_rate': 0,
                'errors': 0,
                'avg_latency_ms': 0,
                'latency_p50_ms': 0,
                'latency_p95_ms': 0,
                'latency_p99_ms': 0,
                'total_tokens': 0,
                'by_operation': {},
                'by_component': {}
            }
        
        total = len(recent_calls)
        cache_hits = sum(1 for c in recent_calls if c.cache_hit)
        errors = sum(1 for c in recent_calls if not c.success)
        total_latency = sum(c.latency_ms for c in recent_calls)
        total_tokens = sum(c.tokens_used for c in recent_calls)
        
        # By operation
        by_operation = {}
        for call in recent_calls:
            by_operation[call.operation] = by_operation.get(call.operation, 0) + 1
        
        # By component
        by_component = {}
        for call in recent_calls:
            by_component[call.component] = by_component.get(call.component, 0) + 1
        
        # Latency percentiles
        latencies = sorted([c.latency_ms for c in recent_calls if c.latency_ms > 0])
        p50 = latencies[len(latencies)//2] if latencies else 0
        p95 = latencies[int(len(latencies)*0.95)] if latencies else 0
        p99 = latencies[int(len(latencies)*0.99)] if latencies else 0
        
        return {
            'period_minutes': minutes,
            'total_calls': total,
            'cache_hit_rate': (cache_hits / total * 100) if total > 0 else 0,
            'cache_hits': cache_hits,
            'cache_misses': total - cache_hits,
            'error_rate': (errors / total * 100) if total > 0 else 0,
            'errors': errors,
            'avg_latency_ms': total_latency / total if total > 0 else 0,
            'latency_p50_ms': p50,
            'latency_p95_ms': p95,
            'latency_p99_ms': p99,
            'total_tokens': total_tokens,
            'by_operation': by_operation,
            'by_component': by_component
        }


# ============================================================================
# Intent Classification Prompts
# ============================================================================

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
        self.enabled = os.getenv('USE_UNIFIED_LLM_SERVICE', 'true').lower() == 'true'  # Changed to true by default
        self.cache_enabled = os.getenv('UNIFIED_CACHE_ENABLED', 'true').lower() == 'true'
        self.cache_ttl = int(os.getenv('UNIFIED_CACHE_TTL', '3600'))  # 1 hour
        self.circuit_breaker_enabled = os.getenv('CIRCUIT_BREAKER_ENABLED', 'true').lower() == 'true'
        
        # Core components
        if COMPONENTS_AVAILABLE:
            self.llm_client = get_llm_client()  # Reuse existing RunPod client
            self.prompt_builder = PromptBuilder()  # Reuse existing prompt system
            logger.info("✅ Unified LLM Service - components loaded")
        else:
            self.llm_client = None
            self.prompt_builder = None
            logger.warning("⚠️  Unified LLM Service - components not available")
        
        # Enhanced components
        self.cache = {}  # Simple dict cache for now (TODO: upgrade to Redis)
        self.cache_timestamps = {}
        self.metrics_collector = MetricsCollector()
        self.circuit_breaker = CircuitBreaker() if self.circuit_breaker_enabled else None
        self.context_builder = None  # Optional context builder (not used in current implementation)
        
        # Legacy metrics (for backward compatibility)
        self.metrics = self._init_legacy_metrics()
        
        self._initialized = True
        
        if self.enabled:
            logger.info("✅ Unified LLM Service initialized (ENABLED)")
            logger.info(f"   Cache: {'✅' if self.cache_enabled else '❌'}")
            logger.info(f"   Circuit Breaker: {'✅' if self.circuit_breaker_enabled else '❌'}")
        else:
            logger.info("ℹ️  Unified LLM Service initialized (DISABLED)")
    
    def _init_legacy_metrics(self) -> Dict[str, Any]:
        """Initialize legacy metrics (for backward compatibility)"""
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
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            logger.warning("🔌 Circuit breaker open - request denied")
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
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        component: str = 'unknown'
    ) -> Dict[str, Any]:
        """
        Core LLM completion method - Main interface for all services
        
        This is the primary method that all other services should use.
        It provides:
        - Caching (semantic deduplication)
        - Circuit breaker protection
        - Metrics tracking
        - Consistent interface
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            component: Name of calling component (for metrics)
            
        Returns:
            {
                "text": "generated text",
                "finish_reason": "stop|length|error",
                "usage": {"prompt_tokens": int, "completion_tokens": int},
                "cached": bool,
                "latency_ms": float
            }
        """
        if not self.llm_client:
            logger.warning("LLM client not available")
            return {
                "text": "LLM service unavailable",
                "finish_reason": "error",
                "usage": {},
                "cached": False,
                "latency_ms": 0
            }
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            logger.warning(f"🔌 Circuit breaker OPEN - rejecting request from {component}")
            return {
                "text": "LLM service temporarily unavailable (circuit breaker open)",
                "finish_reason": "circuit_breaker",
                "usage": {},
                "cached": False,
                "latency_ms": 0
            }
        
        # Check cache
        cached = False
        if self.cache_enabled:
            cache_key = self._cache_key('complete', prompt, max_tokens, temperature)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                # Cache hit
                self.metrics['cache_hits'] += 1
                self.metrics['total_calls'] += 1
                
                self.metrics_collector.record_call(LLMCallMetrics(
                    timestamp=datetime.now(),
                    operation='complete',
                    component=component,
                    cache_hit=True,
                    latency_ms=0,
                    success=True
                ))
                
                logger.debug(f"💾 Cache HIT: {component} ({cache_key[:16]}...)")
                cached_result['cached'] = True
                return cached_result
            
            self.metrics['cache_misses'] += 1
        
        # Call LLM via RunPodLLMClient
        start = time.time()
        success = False
        result_text = ""
        finish_reason = "error"
        usage = {}
        error_msg = None
        
        try:
            llm_result = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if not llm_result:
                error_msg = "LLM returned None"
                logger.error(f"❌ {error_msg}")
            else:
                result_text = llm_result.get('generated_text', '')
                finish_reason = llm_result.get('finish_reason', 'stop')
                usage = llm_result.get('usage', {})
                success = True
                
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ LLM generation failed ({component}): {e}")
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
        
        latency_ms = (time.time() - start) * 1000
        
        # Build result
        result = {
            "text": result_text,
            "finish_reason": finish_reason,
            "usage": usage,
            "cached": False,
            "latency_ms": latency_ms
        }
        
        # Update metrics
        self.metrics['total_calls'] += 1
        self.metrics['llm_calls'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        self.metrics['by_component'][component] = self.metrics['by_component'].get(component, 0) + 1
        self.metrics['by_operation']['complete'] = self.metrics['by_operation'].get('complete', 0) + 1
        
        self.metrics_collector.record_call(LLMCallMetrics(
            timestamp=datetime.now(),
            operation='complete',
            component=component,
            cache_hit=False,
            latency_ms=latency_ms,
            tokens_used=usage.get('completion_tokens', 0),
            success=success,
            error=error_msg
        ))
        
        # Cache successful result
        if success and self.cache_enabled:
            self._set_cache(cache_key, result)
        
        logger.info(f"🤖 LLM complete: {len(result_text)} chars ({latency_ms:.0f}ms, {component})")
        
        return result
    
    async def complete_text(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        component: str = 'unknown'
    ) -> str:
        """
        Complete a prompt and return just the text (for backwards compatibility)
        
        This is a convenience wrapper around complete() for classifiers and other
        services that just need the text response.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            component: Name of calling component
            
        Returns:
            Generated text string
        """
        result = await self.complete(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            component=component
        )
        return result.get('text', '')
    
    def complete_text_sync(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        component: str = 'unknown',
        **kwargs
    ) -> str:
        """
        Synchronous wrapper for complete_text() for handler compatibility.
        
        This allows synchronous handlers to use UnifiedLLMService without
        dealing with async/await.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            component: Name of calling component
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Generated text string
        """
        # Create new event loop if needed
        try:
            loop = asyncio.get_running_loop()
            # Already in async context - this shouldn't happen in sync handlers
            logger.warning(f"⚠️ complete_text_sync called from async context ({component})")
            # Use asyncio.create_task instead
            raise RuntimeError("Cannot use sync method in async context")
        except RuntimeError:
            # No running loop - create one (this is the normal path for handlers)
            return asyncio.run(
                self.complete_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    component=component
                )
            )
    
    async def stream_completion(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        component: str = 'unknown'
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM completion (yields chunks as they're generated)
        
        For real-time user-facing applications that need progressive response.
        Note: Streaming responses are NOT cached.
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            component: Name of calling component
            
        Yields:
            Text chunks as they arrive
        """
        if not self.llm_client:
            logger.warning("LLM client not available")
            yield "LLM service unavailable"
            return
        
        # Check circuit breaker
        if self.circuit_breaker and not self.circuit_breaker.can_execute():
            logger.warning(f"🔌 Circuit breaker OPEN - rejecting stream request from {component}")
            yield "LLM service temporarily unavailable"
            return
        
        # Check if client supports streaming
        if not hasattr(self.llm_client, 'stream_generate'):
            logger.warning(f"⚠️  LLM client doesn't support streaming, falling back to complete()")
            result = await self.complete(prompt, max_tokens, temperature, component)
            yield result['text']
            return
        
        # Stream from LLM
        start = time.time()
        success = False
        total_chunks = 0
        error_msg = None
        
        try:
            async for chunk in self.llm_client.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ):
                total_chunks += 1
                yield chunk
            
            success = True
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ LLM stream failed ({component}): {e}")
            
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            yield f"\n\n[Error: {error_msg}]"
        
        latency_ms = (time.time() - start) * 1000
        
        # Update metrics
        self.metrics['total_calls'] += 1
        self.metrics['llm_calls'] += 1
        self.metrics['total_latency_ms'] += latency_ms
        self.metrics['by_component'][component] = self.metrics['by_component'].get(component, 0) + 1
        self.metrics['by_operation']['stream'] = self.metrics['by_operation'].get('stream', 0) + 1
        
        self.metrics_collector.record_call(LLMCallMetrics(
            timestamp=datetime.now(),
            operation='stream',
            component=component,
            cache_hit=False,
            latency_ms=latency_ms,
            success=success,
            error=error_msg
        ))
        
        logger.info(f"🌊 LLM stream: {total_chunks} chunks ({latency_ms:.0f}ms, {component})")
    
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
    
    def _init_legacy_metrics(self) -> Dict[str, Any]:
        """Initialize legacy metrics (for backward compatibility)"""
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
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics (enhanced with MetricsCollector data)"""
        # Legacy metrics
        total = max(self.metrics['total_calls'], 1)
        llm_calls = self.metrics['llm_calls']
        
        legacy = {
            **self.metrics,
            'cache_hit_rate': (self.metrics['cache_hits'] / total) * 100,
            'cache_size': len(self.cache),
            'avg_latency_ms': self.metrics['total_latency_ms'] / llm_calls if llm_calls > 0 else 0,
            'llm_call_rate': (llm_calls / total) * 100
        }
        
        # Enhanced metrics from collector
        enhanced = self.metrics_collector.get_summary(minutes=60)
        
        # Circuit breaker status
        circuit_status = self.circuit_breaker.get_status() if self.circuit_breaker else None
        
        return {
            'legacy': legacy,  # Backward compatible
            'last_hour': enhanced,
            'circuit_breaker': circuit_status,
            'cache': self.get_cache_stats()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check service health status
        
        Returns:
            Health status dict with component availability
        """
        health = {
            'status': 'healthy' if self.enabled else 'disabled',
            'enabled': self.enabled,
            'cache_enabled': self.cache_enabled,
            'circuit_breaker_state': self.circuit_breaker.state if self.circuit_breaker else 'N/A',
            'llm_client_available': self.llm_client is not None,
            'prompt_builder_available': self.prompt_builder is not None,
            'context_builder_available': self.context_builder is not None,
        }
        
        return health
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary
        
        Returns:
            Dict with all metrics including cache stats, component breakdown, etc.
        """
        summary = {
            'totals': {
                'total_calls': self.metrics['total_calls'],
                'llm_calls': self.metrics['llm_calls'],
                'cache_hits': self.metrics['cache_hits'],
                'cache_misses': self.metrics['cache_misses'],
                'cache_hit_rate': (
                    self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
                    if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0
                    else 0.0
                ),
            },
            'latency': {
                'total_ms': self.metrics['total_latency_ms'],
                'average_ms': (
                    self.metrics['total_latency_ms'] / self.metrics['llm_calls']
                    if self.metrics['llm_calls'] > 0
                    else 0.0
                ),
            },
            'by_component': dict(self.metrics['by_component']),
            'by_operation': dict(self.metrics['by_operation']),
            'by_intent': dict(self.metrics['by_intent']),
            'circuit_breaker': {
                'state': self.circuit_breaker.state.value if self.circuit_breaker else 'N/A',
                'failure_count': self.circuit_breaker.failure_count if self.circuit_breaker else 0,
            } if self.circuit_breaker else None,
        }
        
        return summary
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dict with cache size, hit rate, and TTL info
        """
        total_requests = self.metrics['cache_hits'] + self.metrics['cache_misses']
        hit_rate = (self.metrics['cache_hits'] / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'hits': self.metrics['cache_hits'],
            'misses': self.metrics['cache_misses'],
            'hit_rate': hit_rate,
            'ttl_seconds': self.cache_ttl,
            'enabled': self.cache_enabled
        }


# ============================================================================
# Module-level convenience functions
# ============================================================================

def get_unified_llm() -> UnifiedLLMService:
    """
    Get singleton instance of UnifiedLLMService
    
    Returns:
        UnifiedLLMService instance
    """
    return UnifiedLLMService()
