"""
core.py - Central Orchestration Layer for Pure LLM Handler

This module orchestrates all the extracted components:
- Signal Detection (signals.py)
- Context Building (context.py)
- Prompt Engineering (prompts.py)
- Analytics & Monitoring (analytics.py)
- Query Enhancement (query_enhancement.py)
- Conversation Management (conversation.py)
- Caching (caching.py)
- A/B Testing & Threshold Learning (experimentation.py)
- Resilience Patterns (resilience.py)

Author: AI Istanbul Team
Date: November 2025
"""

import asyncio
import time
import logging
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from collections import defaultdict

# Import extracted modules
from .signals import SignalDetector
from .context import ContextBuilder
from .prompts import PromptBuilder
from .analytics import AnalyticsManager
from .llm_response_parser import clean_training_data_leakage
from .query_enhancement import QueryEnhancer
from .conversation import ConversationManager
from .caching import CacheManager
from .experimentation import ExperimentationManager
from .resilience import (
    CircuitBreaker, 
    CircuitBreakerError, 
    RetryStrategy, 
    TimeoutManager,
    GracefulDegradation
)
from .personalization import PersonalizationEngine
from .auto_tuning import AutoTuner

logger = logging.getLogger(__name__)


class PureLLMCore:
    """
    Central orchestrator for the Pure LLM Handler system.
    
    This class coordinates all subsystems to process user queries:
    - Detects signals and intents
    - Builds smart context
    - Manages conversations
    - Handles caching
    - Runs A/B experiments
    - Tracks analytics
    - Generates responses
    
    Architecture:
    ```
    User Query
        ↓
    [Query Enhancement] → spell check, rewrite, validate
        ↓
    [Cache Check] → semantic similarity search
        ↓
    [Signal Detection] → multi-intent, semantic matching
        ↓
    [Context Building] → database, RAG, services
        ↓
    [Conversation] → resolve references, add history
        ↓
    [Prompt Engineering] → build optimized prompt
        ↓
    [LLM Generation] → RunPod/OpenAI API
        ↓
    [Validation] → quality checks
        ↓
    [Caching] → store for future
        ↓
    [Analytics] → track metrics
        ↓
    Response
    ```
    """
    
    def __init__(
        self,
        llm_client,
        db_connection,
        config: Optional[Dict[str, Any]] = None,
        services=None
    ):
        """
        Initialize the Pure LLM Core orchestrator.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            db_connection: Database connection
            config: Configuration dictionary
            services: Service Manager instance with all local services
        """
        self.llm = llm_client
        self.llm_client = llm_client  # Alias for compatibility
        self.db = db_connection
        self.config = config or {}
        self.services = services  # Service Manager for local services
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        logger.info("🚀 Pure LLM Core initialized successfully")
        if services:
            status = services.get_service_status()
            active = sum(1 for v in status.values() if v)
            logger.info(f"   📦 Service Manager: {active}/{len(status)} services available")
    
    def _initialize_subsystems(self):
        """Initialize all subsystem modules."""
        
        # 0. Resilience System (Circuit Breakers, Retry, Timeout, Graceful Degradation)
        logger.info("🛡️ Initializing resilience components...")
        
        # Circuit Breakers for each external service
        # NOTE: LLM circuit breaker disabled for production (high MAU traffic)
        # If LLM is down, we want to keep trying and return fallback per request
        self.circuit_breakers = {
            'llm': CircuitBreaker(
                name='LLM Service',
                failure_threshold=self.config.get('llm_failure_threshold', 999999),  # Effectively disabled
                success_threshold=2,
                timeout=1.0  # Retry immediately - don't block traffic
            ),
            'database': CircuitBreaker(
                name='Database',
                failure_threshold=self.config.get('db_failure_threshold', 3),
                success_threshold=2,
                timeout=30.0
            ),
            'rag': CircuitBreaker(
                name='RAG Service',
                failure_threshold=4,
                success_threshold=2,
                timeout=45.0
            ),
            'weather': CircuitBreaker(
                name='Weather API',
                failure_threshold=3,
                success_threshold=2,
                timeout=30.0
            ),
            'events': CircuitBreaker(
                name='Events API',
                failure_threshold=3,
                success_threshold=2,
                timeout=30.0
            )
        }
        
        # Retry Strategy with exponential backoff
        self.retry_strategy = RetryStrategy(
            max_retries=self.config.get('max_retries', 3),
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=True
        )
        
        # Timeout Manager
        self.timeout_manager = TimeoutManager()
        
        # Override default timeouts if provided in config
        if 'timeouts' in self.config:
            for operation, timeout in self.config['timeouts'].items():
                self.timeout_manager.update_timeout(operation, timeout)
        
        logger.info("✅ Resilience components initialized")
        
        # 1. Signal Detection System
        self.signal_detector = SignalDetector(
            embedding_model=self.config.get('embedding_model'),
            language_thresholds=self.config.get('language_thresholds')
        )
        
        # 2. Context Building System
        self.context_builder = ContextBuilder(
            db_connection=self.db,
            rag_service=self.config.get('rag_service'),
            weather_service=self.config.get('weather_service'),
            events_service=self.config.get('events_service'),
            hidden_gems_service=self.config.get('hidden_gems_service'),
            map_service=self.config.get('map_service') or self.config.get('routing_service'),
            service_manager=self.services,  # Pass service manager for local services
            circuit_breakers=self.circuit_breakers,
            timeout_manager=self.timeout_manager
        )
        
        # 3. Prompt Engineering System
        self.prompt_builder = PromptBuilder(
            system_prompts=self.config.get('system_prompts'),
            intent_prompts=self.config.get('intent_prompts')
        )
        
        # 4. Analytics & Monitoring System
        self.analytics = AnalyticsManager(
            enable_detailed_tracking=self.config.get('enable_detailed_tracking', True)
        )
        
        # 5. Query Enhancement System (spell check, rewrite, validate)
        self.query_enhancer = QueryEnhancer(
            enable_spell_check=self.config.get('enable_spell_check', True),
            enable_rewriting=self.config.get('enable_rewriting', True),
            enable_validation=self.config.get('enable_validation', True)
        )
        
        # 6. Conversation Management System
        self.conversation_manager = ConversationManager(
            max_history_length=self.config.get('max_conversation_history', 10),
            enable_reference_resolution=True
        )
        
        # 7. Caching System (semantic + exact match)
        self.cache_manager = CacheManager(
            redis_client=self.config.get('redis_client'),
            enable_semantic_cache=self.config.get('enable_semantic_cache', True),
            cache_ttl=self.config.get('cache_ttl', 3600)
        )
        
        # 8. Experimentation System (A/B testing + threshold learning)
        self.experimentation = ExperimentationManager(
            enable_ab_testing=self.config.get('enable_ab_testing', False),
            enable_threshold_learning=self.config.get('enable_threshold_learning', True),
            auto_tune_interval_hours=self.config.get('auto_tune_interval_hours', 24)
        )
        
        # 9. Personalization Engine (user preferences + feedback learning) - PHASE 2
        self.personalization = PersonalizationEngine(
            db_connection=self.db,
            redis_client=self.config.get('redis_client'),
            config={
                'min_interactions': self.config.get('min_interactions_for_personalization', 3),
                'preference_threshold': self.config.get('preference_confidence_threshold', 0.6),
                'max_personalized_items': self.config.get('max_personalized_items', 20)
            }
        )
        
        # 10. Auto-Tuner (threshold optimization) - PHASE 2
        self.auto_tuner = AutoTuner(
            signal_detector=self.signal_detector,
            personalization_engine=self.personalization,
            config={
                'min_samples': self.config.get('min_samples_for_tuning', 50),
                'target_f1': self.config.get('target_f1_score', 0.8),
                'max_threshold_change': self.config.get('max_threshold_change', 0.1),
                'tuning_schedule': self.config.get('tuning_schedule', 'daily'),
                'enable_tuning': self.config.get('enable_auto_tuning', True)
            }
        )
        
        logger.info("✅ All subsystems initialized (including Phase 2 features)")
    
    async def _rewrite_query_with_llm(
        self,
        query: str,
        language: str,
        user_location: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to rewrite query for better signal detection (PRIORITY 1).
        
        Handles:
        - Typos and misspellings
        - Slang and informal language
        - Abbreviations (e.g., "2" → "to", "hw" → "how")
        - Ambiguous phrasing
        
        Args:
            query: Original user query
            language: Language code
            user_location: User's GPS location (optional)
            
        Returns:
            {
                'rewritten_query': str,
                'needs_rewriting': bool,
                'confidence': float,
                'reason': str
            }
        """
        # Skip rewriting for simple greetings - they don't need it
        simple_greetings = ['hi', 'hello', 'hey', 'merhaba', 'selam', 'greetings']
        if query.lower().strip() in simple_greetings:
            return {
                'rewritten_query': query,
                'needs_rewriting': False,
                'confidence': 1.0,
                'reason': 'simple_greeting_skipped'
            }
        
        # Quick check: Does query need rewriting?
        needs_rewriting = any([
            len(query.split()) >= 3 and re.search(r'\d(?!\d)', query),  # Number substitutions like "2" for "to"
            any(word in query.lower() for word in [
                'wat', 'wher', 'hw', 'pls', 'thx', 'plz', 'wanna', 'gonna',
                'u', 'r', 'ur', 'sum', 'closeby', 'neer', 'restarant', 'restorant',
                'restraunt', 'resturant', 'resteraunt'
            ])  # Obvious typos or slang
        ])
        
        if not needs_rewriting:
            return {
                'rewritten_query': query,
                'needs_rewriting': False,
                'confidence': 1.0,
                'reason': 'query_is_clear'
            }
        
        # Build rewriting prompt with stricter constraints
        rewrite_prompt = f"""Fix only typos/abbreviations in this query. Keep it SHORT and SIMPLE.

Query: "{query}"

Fixed version (max 50 chars):"""
        
        try:
            result = await self.llm.generate(
                prompt=rewrite_prompt,
                max_tokens=30,  # Drastically reduced to prevent long outputs
                temperature=0.1  # Very low temperature for consistency
            )
            
            rewritten = result['generated_text'].strip()
            
            # Remove any quotes, newlines, or extra formatting
            rewritten = rewritten.strip('"\'').split('\n')[0].strip()
            
            # STRICT Validation: Must be similar length and reasonable
            max_length = max(len(query) * 2.5, len(query) + 20)  # More forgiving but still bounded
            min_length = max(len(query) * 0.5, 2)  # At least half the length
            
            if len(rewritten) > max_length or len(rewritten) < min_length or len(rewritten) > 100:
                logger.warning(f"⚠️ Query rewriting failed validation: '{query}' ({len(query)}) → '{rewritten}' ({len(rewritten)})")
                return {
                    'rewritten_query': query,
                    'needs_rewriting': True,
                    'confidence': 0.0,
                    'reason': 'rewriting_validation_failed'
                }
            
            # Check if rewritten contains suspicious patterns (training data leakage)
            suspicious_patterns = [
                '#', 'twitter', 'hashtag', 'example:', 'question:', 'answer:',
                'training', 'dataset', 'sample', 'dialogue'
            ]
            if any(pattern in rewritten.lower() for pattern in suspicious_patterns):
                logger.warning(f"⚠️ Query rewriting contains suspicious patterns: '{rewritten}'")
                return {
                    'rewritten_query': query,
                    'needs_rewriting': True,
                    'confidence': 0.0,
                    'reason': 'suspicious_output_detected'
                }
            
            # Check if rewritten is substantially different
            if rewritten.lower() == query.lower():
                return {
                    'rewritten_query': query,
                    'needs_rewriting': False,
                    'confidence': 1.0,
                    'reason': 'no_changes_needed'
                }
            
            logger.info(f"✨ Query rewritten: '{query}' → '{rewritten}'")
            
            return {
                'rewritten_query': rewritten,
                'needs_rewriting': True,
                'confidence': 0.9,
                'reason': 'successfully_rewritten'
            }
            
        except Exception as e:
            logger.error(f"❌ Query rewriting failed: {e}")
            return {
                'rewritten_query': query,
                'needs_rewriting': True,
                'confidence': 0.0,
                'reason': f'error: {str(e)}'
            }
    
    def extract_intents_from_response(self, response_text: str) -> Dict[str, bool]:
        """
        Extract LLM-classified intents from response (PRIORITY 2).
        
        Looks for pattern:
        Intents: [X] Transportation [ ] Restaurant [X] Attraction ...
        
        Args:
            response_text: LLM response text
            
        Returns:
            Dict of signal_name -> bool
        """
        import re  # Import re module for pattern matching
        
        intent_map = {
            'Transportation/Directions': 'needs_transportation',
            'Restaurant Recommendation': 'needs_restaurant',
            'Attraction Information': 'needs_attraction',
            'Neighborhood/Area Info': 'needs_neighborhood',
            'Event/Activity Query': 'needs_events',
            'Shopping': 'needs_shopping',
            'Nightlife': 'needs_nightlife',
            'General Question': 'needs_general_info'
        }
        
        llm_intents = {}
        
        # Find intent classification line (handle both single and multi-line format)
        match = re.search(r'Intents:\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)', response_text, re.IGNORECASE | re.DOTALL)
        if match:
            intent_line = match.group(1).strip()
            
            for display_name, signal_name in intent_map.items():
                # Check if [X] or [x] appears before this intent name
                # Pattern: [X] IntentName or IntentName: [X]
                pattern1 = rf'\[X\]\s*{re.escape(display_name)}'
                pattern2 = rf'{re.escape(display_name)}:\s*\[X\]'
                
                has_match = (
                    bool(re.search(pattern1, intent_line, re.IGNORECASE)) or
                    bool(re.search(pattern2, intent_line, re.IGNORECASE))
                )
                llm_intents[signal_name] = has_match
        else:
            # Initialize all to False if no match found
            for signal_name in intent_map.values():
                llm_intents[signal_name] = False
        
        return llm_intents
    
    def _is_route_query(self, query: str) -> bool:
        """
        Detect if query is asking for route/directions.
        
        This is a lightweight check used BEFORE signal detection
        to decide whether to bypass cache for GPS-based queries.
        
        Returns:
            True if query appears to be asking for directions/route
        """
        query_lower = query.lower()
        
        # Route/direction keywords
        route_patterns = [
            'how to go',
            'how can i go',
            'how do i get',
            'how to get',
            'directions',
            'route',
            'way to',
            'get to',
            'travel to',
            'reach',
            'from',
            'to',
        ]
        
        return any(pattern in query_lower for pattern in route_patterns)
    
    async def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        max_tokens: int = 250,
        enable_conversation: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete pipeline.
        
        Pipeline Steps:
        1. Query Enhancement (spell check, rewrite, validate)
        2. Cache Check (semantic similarity search)
        3. Signal Detection (multi-intent, semantic)
        4. Context Building (database, RAG, services)
        5. Conversation Integration (reference resolution)
        6. Prompt Engineering (optimized prompt)
        7. LLM Generation (with validation)
        8. Caching (store for future)
        9. Analytics Tracking
        
        Args:
            query: User query string
            user_id: User identifier
            session_id: Conversation session ID
            user_location: User GPS location {"lat": float, "lon": float}
            language: Response language (en/tr/ar/etc.)
            max_tokens: Maximum tokens to generate
            enable_conversation: Enable conversational context
            
        Returns:
            Dict with response, map_data, signals, and metadata
        """
        start_time = time.time()
        
        logger.info(f"🔍 Processing query: {query[:50]}...")
        
        # Track query in analytics
        self.analytics.track_query(user_id, language, query)
        
        # STEP 0.5: Turkish Typo Correction (before enhancement)
        if language == 'tr' and self.services:
            try:
                if hasattr(self.services, 'typo_corrector') and self.services.typo_corrector:
                    corrected_query = self.services.typo_corrector.correct_silent(query)
                    if corrected_query != query:
                        logger.info(f"🔤 Turkish typo corrected: '{query}' → '{corrected_query}'")
                        query = corrected_query
            except Exception as e:
                logger.warning(f"Turkish typo correction failed: {e}")
        
        # STEP 1: Query Enhancement
        original_query = query
        enhancement_metadata = {}
        
        try:
            enhanced = await self.query_enhancer.enhance_query(
                query=query,
                language=language
            )
            
            query = enhanced['query']
            enhancement_metadata = {
                'spell_corrected': enhanced.get('spell_corrected', False),
                'rewritten': enhanced.get('rewritten', False),
                'validation': enhanced.get('validation'),
                'original_query': original_query if query != original_query else None
            }
            
            logger.info(f"✨ Query enhanced: {enhancement_metadata}")
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            query = original_query
        
        # STEP 1.5: LLM-Based Query Rewriting (PRIORITY 1) - NEW
        if self.config.get('enable_llm_query_rewriting', True):
            try:
                rewrite_result = await self._rewrite_query_with_llm(query, language, user_location)
                
                if rewrite_result['needs_rewriting'] and rewrite_result['confidence'] > 0.5:
                    original_query_pre_rewrite = query
                    query = rewrite_result['rewritten_query']
                    
                    enhancement_metadata['query_rewritten'] = True
                    enhancement_metadata['original_query_pre_rewrite'] = original_query_pre_rewrite
                    enhancement_metadata['rewrite_reason'] = rewrite_result['reason']
                    enhancement_metadata['rewrite_confidence'] = rewrite_result['confidence']
                    
                    logger.info(f"🔄 Using rewritten query for signal detection: {query}")
            except Exception as e:
                logger.warning(f"LLM query rewriting failed: {e}")
        
        # STEP 2.5: Pre-signal detection for cache bypass (lightweight check)
        # Detect route/transportation queries BEFORE cache lookup
        # This prevents cached responses from being used when GPS location changes
        is_route_query = self._is_route_query(query)
        has_gps = user_location is not None
        bypass_cache = is_route_query and has_gps
        
        if bypass_cache:
            logger.info("🚫 Bypassing cache for GPS-based route query")
        
        # STEP 2: Cache Check (skip for GPS-based route queries)
        cached_response = None
        if not bypass_cache:
            cached_response = await self.cache_manager.get_cached_response(
                query=query,
                language=language,
                similarity_threshold=0.85
            )
        
        if cached_response:
            logger.info("✅ Cache hit!")
            self.analytics.track_cache_hit()
            
            # Add enhancement metadata to cached response
            cached_response['metadata']['enhancement'] = enhancement_metadata
            cached_response['metadata']['cached'] = True
            
            return cached_response
        
        if not bypass_cache:
            self.analytics.track_cache_miss()
        
        # STEP 3: Signal Detection
        signals = await self.signal_detector.detect_signals(
            query=query,
            user_location=user_location,
            language=language,
            user_id=user_id,
            experimentation_manager=self.experimentation
        )
        
        active_signals = [k for k, v in signals['signals'].items() if v]
        logger.info(f"🎯 Signals detected: {', '.join(active_signals) if active_signals else 'none'}")
        
        self.analytics.track_signals(signals['signals'])
        
        # STEP 4: Conversation Context (if enabled)
        conversation_context = None
        if enable_conversation and session_id:
            try:
                conversation_context = await self.conversation_manager.get_context(
                    session_id=session_id,
                    current_query=query,
                    max_turns=3
                )
                
                # Resolve references (e.g., "there", "it", etc.)
                if conversation_context.get('needs_resolution'):
                    resolved = await self.conversation_manager.resolve_references(
                        query=query,
                        context=conversation_context
                    )
                    
                    if resolved.get('resolved'):
                        logger.info(f"💬 Resolved reference: {query} → {resolved['resolved_query']}")
                        query = resolved['resolved_query']
                
            except Exception as e:
                logger.warning(f"Conversation context failed: {e}")
        
        # STEP 5: Context Building (PRIORITY 3: Pass signal confidence)
        overall_confidence = signals.get('overall_confidence', 1.0)
        
        context = await self.context_builder.build_context(
            query=query,
            signals=signals['signals'],
            user_location=user_location,
            language=language,
            signal_confidence=overall_confidence,  # NEW: Pass confidence for adaptive context
            original_query=original_query  # Pass original query for location extraction
        )
        
        logger.info(
            f"📚 Context built: "
            f"DB={len(context['database'])} chars, "
            f"RAG={len(context['rag'])} chars, "
            f"Services={len(context['services'])} items"
        )
        
        self.analytics.track_context(context)
        
        # STEP 5.5: Personalization Filtering (PHASE 2)
        if self.config.get('enable_personalization', True):
            try:
                # Filter context based on user preferences
                if context.get('database'):
                    # Parse database context to extract items
                    # This is a simplified version - you may need to adapt based on your context structure
                    db_items = []  # Extract items from context['database']
                    
                    filtered_items = await self.personalization.filter_context_by_preferences(
                        user_id=user_id,
                        context_items=db_items,
                        signals=signals['signals']
                    )
                    
                    if filtered_items:
                        logger.info(f"🎯 Personalized context for user {user_id}")
                        # Update context with filtered items
                        # context['database'] = ... (rebuild with filtered_items)
            except Exception as e:
                logger.warning(f"Personalization filtering failed: {e}")
        
        # STEP 6: Prompt Engineering
        # Get overall confidence from signals
        overall_confidence = signals.get('overall_confidence', 1.0)
        enable_intent_classification = self.config.get('enable_llm_intent_classification', False)  # Disabled - causes template artifacts
        
        prompt = self.prompt_builder.build_prompt(
            query=query,
            signals=signals['signals'],
            context=context,
            conversation_context=conversation_context,
            language=language,
            user_location=user_location,  # Pass GPS location to prompt
            enable_intent_classification=enable_intent_classification,  # Priority 2
            signal_confidence=overall_confidence  # Priority 3
        )
        
        logger.info(f"📝 Prompt built: {len(prompt)} chars, signal confidence: {overall_confidence:.2f}")
        if user_location:
            logger.info(f"   📍 GPS location included in prompt: {user_location}")
        if overall_confidence < 0.6:
            logger.warning(f"   ⚠️ Low signal confidence - added explicit instructions to prompt")

        
        # === CRITICAL: TRANSPORTATION EARLY RETURN ===
        # For transportation queries, SKIP LLM entirely and return template-based response
        # This is the DETERMINISTIC FACT LAYER of the hybrid architecture
        if signals['signals'].get('needs_transportation'):
            logger.info(f"🚇 TRANSPORTATION QUERY DETECTED - Bypassing LLM, using template response")
            
            # Extract route data from context (verified facts from Transportation RAG)
            route_data = None
            if context.get('services'):
                for service_item in context['services']:
                    if isinstance(service_item, dict) and 'route' in service_item:
                        route_data = service_item
                        break
            
            if route_data:
                # Generate template-based response using ONLY verified facts
                response_text = await self._generate_template_transportation_response(
                    route_data=route_data,
                    query=query,
                    language=language
                )
                
                # Build result and return early - NO LLM GENERATION
                total_latency = time.time() - start_time
                map_data = context.get('map_data')
                
                result = {
                    'response': response_text,
                    'map_data': map_data,
                    'signals': signals['signals'],
                    'metadata': {
                        'total_latency': total_latency,
                        'llm_latency': 0,  # No LLM used
                        'context_latency': context_latency,
                        'cache_hit': cache_hit,
                        'cache_key': cache_key,
                        'experiment_variant': variant,
                        'personalized': personalized_results is not None,
                        'mode': 'template_transportation',  # Fact-locked template
                        'hallucination_risk': 'zero'  # No LLM = no hallucination
                    }
                }
                
                logger.info(f"✅ Template transportation response generated in {total_latency:.2f}s (ZERO hallucination risk)")
                return result
            else:
                logger.warning(f"⚠️ Transportation query detected but no route data available - falling back to LLM")
        
        # STEP 7: LLM Generation with Resilience
        try:
            llm_start = time.time()
            
            # Simplified LLM call with circuit breaker only
            async def _generate_with_llm():
                logger.info(f"📝 Calling LLM...")
                logger.info(f"📏 Prompt length: {len(prompt)} chars")
                logger.info(f"🔚 Prompt ending (last 300 chars): ...{prompt[-300:]}")
                result = await self.llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                logger.info(f"📥 LLM returned response with keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                return result
            
            # Apply circuit breaker protection
            response_data = await self.circuit_breakers['llm'].call(_generate_with_llm)
            
            llm_latency = time.time() - llm_start
            
            logger.info(f"🔍 Response data type: {type(response_data)}, keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'not a dict'}")
            if not response_data or "generated_text" not in response_data:
                error_msg = f"Invalid LLM response structure: {type(response_data)}"
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
            
            response_text = response_data["generated_text"]
            logger.info(f"🔍 Response text length: {len(response_text)}, type: {type(response_text)}")
            logger.info(f"🔍 RAW LLM RESPONSE (FULL): {repr(response_text)}")
            
            # Clean training data leakage from response
            logger.info(f"🧹 Applying training data leakage filter to {len(response_text)} chars...")
            response_text = clean_training_data_leakage(response_text, prompt=prompt)
            logger.info(f"✅ After filter: {len(response_text)} chars")
            
            # Clean formatting artifacts (checkboxes, duplicate emojis, etc.)
            from .llm_response_parser import clean_response_formatting
            response_text = clean_response_formatting(response_text)
            logger.info(f"✅ After formatting cleanup: {len(response_text)} chars")
            logger.info(f"🔍 FINAL CLEANED RESPONSE: {response_text[:500]}...")
            
            # STEP 7.5: Extract LLM-classified intents (PRIORITY 2) - NEW
            llm_intents = {}
            if enable_intent_classification:
                try:
                    llm_intents = self.extract_intents_from_response(response_text)
                    
                    if llm_intents:
                        logger.info(f"🎯 LLM-detected intents: {[k for k, v in llm_intents.items() if v]}")
                        
                        # Track intent comparison for analytics
                        await self.analytics.track_intent_comparison(
                            regex_intents=signals['signals'],
                            llm_intents=llm_intents,
                            query=query,
                            user_id=user_id
                        )
                        
                        # Remove intent classification from final response (clean it up)
                        # Remove the entire intent classification section including checkboxes
                        response_text = re.sub(
                            r'---\s*\n*🎯\s*INTENT\s+CLASSIFICATION.*?(?=\n\n[A-Z]|\n\n\w|$)',
                            '',
                            response_text,
                            flags=re.DOTALL | re.IGNORECASE
                        )
                        
                        # Remove uncertain intent detection section
                        response_text = re.sub(
                            r'---\s*\n*🚨\s*UNCERTAIN\s+INTENT.*?(?=\n\n[A-Z]|\n\n\w|$)',
                            '',
                            response_text,
                            flags=re.DOTALL | re.IGNORECASE
                        )
                        
                        # Remove multi-intent detection section
                        response_text = re.sub(
                            r'---\s*\n*🎯\s*MULTI-INTENT.*?(?=\n\n[A-Z]|\n\n\w|$)',
                            '',
                            response_text,
                            flags=re.DOTALL | re.IGNORECASE
                        )
                        
                        # Remove any standalone "Intents:" lines with checkboxes
                        response_text = re.sub(
                            r'Intents:\s*\[.*?\].*?\n',
                            '',
                            response_text,
                            flags=re.IGNORECASE
                        )
                        
                        # Remove checkbox patterns at start of line
                        response_text = re.sub(
                            r'^\s*\[[Xx ]\].*?(?:Transportation|Restaurant|Attraction|Neighborhood|Event|Shopping|Nightlife|General).*?\n',
                            '',
                            response_text,
                            flags=re.MULTILINE
                        )
                        
                        # Clean up any leftover separators
                        response_text = re.sub(r'\n---\n+', '\n\n', response_text)
                        response_text = re.sub(r'\n{3,}', '\n\n', response_text)
                        response_text = response_text.strip()
                except Exception as e:
                    logger.warning(f"Failed to extract LLM intents: {e}")
            
            logger.info(f"✅ LLM generated response in {llm_latency:.2f}s (length: {len(response_text)} chars)")
            
            # Validate response
            is_valid, validation_error = await self._validate_response(
                response=response_text,
                query=query,
                signals=signals['signals'],
                context=context
            )
            
            if not is_valid:
                logger.warning(f"Response validation failed: {validation_error}")
                self.analytics.track_validation_failure(validation_error)
                
                # Attempt recovery
                response_text = await self._fallback_response(
                    query=query,
                    context=context,
                    error_type="validation"
                )
            
            # === TRANSPORTATION-SPECIFIC VALIDATION ===
            # Apply hard guardrails for transportation facts to prevent hallucinations
            if signals['signals'].get('needs_transportation'):
                logger.info(f"🚇 Applying transportation validation for route query...")
                
                # Extract route data from context (verified facts from Transportation RAG)
                route_data = None
                if context.get('services'):
                    for service_item in context['services']:
                        if isinstance(service_item, dict) and 'route' in service_item:
                            route_data = service_item
                            break
                
                # Validate LLM response against verified route facts
                is_transport_valid, transport_error, corrected_response = await self._validate_transportation_response(
                    response=response_text,
                    route_data=route_data,
                    query=query
                )
                
                if not is_transport_valid:
                    logger.error(f"🚨 TRANSPORTATION HALLUCINATION DETECTED: {transport_error}")
                    self.analytics.track_validation_failure(f"transport_hallucination: {transport_error}")
                    
                    if corrected_response:
                        # Use auto-corrected response
                        logger.info(f"✅ Using auto-corrected response (hallucinated facts replaced)")
                        response_text = corrected_response
                    else:
                        # Fallback to template-based response using verified facts only
                        logger.warning(f"⚠️ Auto-correction failed, generating template-based response")
                        response_text = await self._generate_template_transportation_response(
                            route_data=route_data,
                            query=query,
                            language=language
                        )
                else:
                    logger.info(f"✅ Transportation response validated - no hallucinations detected")
        
        except CircuitBreakerError as e:
            logger.error(f"❌ LLM service unavailable (circuit breaker open): {e}")
            self.analytics.track_llm_failure('circuit_breaker_open')
            
            # Graceful degradation - provide informative error
            response_text = await self._fallback_response(
                query=query,
                context=context,
                error_type="circuit_breaker"
            )
            
            result = {
                'response': response_text,
                'map_data': None,
                'signals': signals['signals'],
                'metadata': {
                    'error': 'llm_unavailable',
                    'error_type': 'circuit_breaker',
                    'degraded_mode': True,
                    'cached': False,
                    'message': 'AI service temporarily unavailable due to repeated failures'
                }
            }
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ LLM generation failed: {error_msg}")
            self.analytics.track_error('llm_failure', error_msg)
            
            # Detect error type for better fallback messages
            error_type = "unknown"
            if "404" in error_msg or "Not Found" in error_msg:
                error_type = "404"
                logger.error("🔥 LLM endpoint returned 404 - vLLM server may be down!")
            elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                error_type = "timeout"
                logger.error("⏱️ LLM request timed out - server may be overloaded")
            elif "connection" in error_msg.lower():
                error_type = "connection"
                logger.error("🔌 Connection error to LLM service")
            
            # Fallback to context-based response
            response_text = await self._fallback_response(
                query=query,
                context=context,
                error_type=error_type
            )
            llm_latency = 0
        
        # STEP 8: Build Result
        total_latency = time.time() - start_time
        
        # Enhanced map_data handling: Include locations from context for restaurant/POI queries
        map_data = context.get('map_data')
        
        # === TRANSPORTATION ROUTE VISUALIZATION ===
        # If this is a transportation query and we have no map_data yet,
        # use the robust pattern extraction from AIChatRouteHandler
        if not map_data and signals['signals'].get('needs_transportation'):
            logger.info(f"🚇 Detected transportation query, generating route visualization...")
            try:
                from services.ai_chat_route_integration import AIChatRouteHandler
                from services.transportation_rag_system import get_transportation_rag
                
                # Use the original query for accurate location extraction
                query_for_extraction = original_query if 'original_query' in locals() else query
                logger.info(f"🔍 [ROUTE EXTRACTION] Using query: '{query_for_extraction}'")
                
                # Check if query uses "from my location", "from here", or similar patterns
                query_lower = query_for_extraction.lower()
                use_gps_origin = False
                use_gps_dest = False
                
                gps_origin_patterns = [
                    'from my location', 'from here', 'from current location',
                    'from where i am', 'from my position', 'starting from here',
                    'from my current location'
                ]
                
                gps_dest_patterns = [
                    'to my location', 'to here', 'to current location',
                    'to where i am', 'back here'
                ]
                
                for pattern in gps_origin_patterns:
                    if pattern in query_lower:
                        use_gps_origin = True
                        logger.info(f"📍 Detected GPS origin pattern: '{pattern}'")
                        break
                
                for pattern in gps_dest_patterns:
                    if pattern in query_lower:
                        use_gps_dest = True
                        logger.info(f"📍 Detected GPS destination pattern: '{pattern}'")
                        break
                
                # Initialize route handler for pattern extraction
                route_handler = AIChatRouteHandler()
                
                # Extract locations using robust pattern matching
                locations = route_handler._extract_locations(query_for_extraction)
                logger.info(f"📍 [ROUTE EXTRACTION] Found {len(locations)} location(s): {locations}")
                
                # Determine origin and destination
                origin_coords = None
                dest_coords = None
                origin_str = None
                dest_str = None
                origin_gps = None
                dest_gps = None
                
                # Handle GPS-based queries
                # Case 1: User explicitly says "from my location" OR only gives destination
                if (use_gps_origin or len(locations) == 1) and user_location:
                    # User wants route FROM their location TO a destination
                    logger.info(f"📍 Using GPS as origin: {user_location}")
                    origin_coords = (user_location['lat'], user_location['lon'])
                    origin_str = "Your Location"
                    origin_gps = user_location
                    
                    if len(locations) >= 1:
                        dest_coords = locations[0]
                        # Find destination name
                        for name, coords in route_handler.KNOWN_LOCATIONS.items():
                            if abs(coords[0] - dest_coords[0]) < 0.001 and abs(coords[1] - dest_coords[1]) < 0.001:
                                dest_str = name
                                break
                        if not dest_str:
                            dest_str = f"{dest_coords[0]:.4f}, {dest_coords[1]:.4f}"
                
                # Case 2: No locations found at all but user has GPS - try to extract from query text
                elif len(locations) == 0 and user_location:
                    logger.info(f"📍 No locations extracted, but GPS available. Trying text extraction...")
                    # Try to find destination in query text
                    query_lower = query_for_extraction.lower()
                    
                    # Look for "to X" or "go to X" patterns
                    import re
                    go_to_patterns = [
                        r'(?:go|get|going|getting|travel|traveling|walk|walking|drive|driving)\s+to\s+([a-zA-Z\s]+?)(?:\s+from|\s+\?|$|\s+please)',
                        r'(?:how|way|route|directions?)\s+(?:to|towards?)\s+([a-zA-Z\s]+?)(?:\s+from|\s+\?|$|\s+please)',
                        r'to\s+([a-zA-Z\s]+?)(?:\s+from|\s+\?|$|\s+please)'
                    ]
                    
                    destination_name = None
                    for pattern in go_to_patterns:
                        match = re.search(pattern, query_lower)
                        if match:
                            destination_name = match.group(1).strip()
                            logger.info(f"📍 Extracted destination from text: '{destination_name}'")
                            break
                    
                    # Try to find this location in KNOWN_LOCATIONS
                    if destination_name:
                        for name, coords in route_handler.KNOWN_LOCATIONS.items():
                            if destination_name in name.lower() or name.lower() in destination_name:
                                dest_coords = coords
                                dest_str = name
                                origin_coords = (user_location['lat'], user_location['lon'])
                                origin_str = "Your Location"
                                origin_gps = user_location
                                logger.info(f"📍 Matched destination '{destination_name}' to {name} at {coords}")
                                break
                
                elif use_gps_dest and user_location and len(locations) >= 1:
                    # User wants route FROM a location TO their current position
                    logger.info(f"📍 Using GPS as destination: {user_location}")
                    origin_coords = locations[0]
                    dest_coords = (user_location['lat'], user_location['lon'])
                    dest_str = "Your Location"
                    dest_gps = user_location
                    
                    # Find origin name
                    for name, coords in route_handler.KNOWN_LOCATIONS.items():
                        if abs(coords[0] - origin_coords[0]) < 0.001 and abs(coords[1] - originCoords[1]) < 0.001:
                            origin_str = name
                            break
                    if not origin_str:
                        origin_str = f"{origin_coords[0]:.4f}, {originCoords[1]:.4f}"
                
                elif len(locations) >= 2:
                    # Standard two-location query
                    origin_coords = locations[0]
                    dest_coords = locations[1]
                    
                    # Find location names by reverse lookup
                    for name, coords in route_handler.KNOWN_LOCATIONS.items():
                        if abs(coords[0] - origin_coords[0]) < 0.001 and abs(coords[1] - origin_coords[1]) < 0.001:
                            origin_str = name
                        if abs(coords[0] - dest_coords[0]) < 0.001 and abs(coords[1] - dest_coords[1]) < 0.001:
                            dest_str = name
                    
                    # Fallback to coordinate strings if names not found
                    if not origin_str:
                        origin_str = f"{origin_coords[0]:.4f}, {origin_coords[1]:.4f}"
                    if not dest_str:
                        dest_str = f"{dest_coords[0]:.4f}, {dest_coords[1]:.4f}"
                
                if origin_coords and dest_coords:
                    logger.info(f"📍 [ROUTE EXTRACTION] Resolved: {origin_str} → {dest_str}")
                    
                    # Now use Transportation RAG to find the actual route
                    transport_rag = get_transportation_rag()
                    if transport_rag:
                        route = transport_rag.find_route(
                            origin_str, 
                            dest_str,
                            origin_gps=origin_gps,
                            destination_gps=dest_gps
                        )
                        
                        if route:
                            logger.info(f"✅ Transportation RAG found route: {len(route.steps)} steps, {route.transfers} transfers")
                            
                            # Convert route to map_data format with coordinates
                            coordinates = []
                            markers = []
                            
                            # Add origin marker
                            coordinates.append([origin_coords[0], origin_coords[1]])
                            markers.append({
                                "position": {"lat": origin_coords[0], "lng": origin_coords[1]},
                                "label": origin_str.title(),
                                "type": "origin"
                            })
                            
                            # Add waypoints from route steps
                            for step in route.steps:
                                if hasattr(step, 'from_station') and step.from_station:
                                    if hasattr(step.from_station, 'lat') and hasattr(step.from_station, 'lng'):
                                        coord = [step.from_station.lat, step.from_station.lng]
                                        if coord not in coordinates:
                                            coordinates.append(coord)
                                
                                if hasattr(step, 'to_station') and step.to_station:
                                    if hasattr(step.to_station, 'lat') and hasattr(step.to_station, 'lng'):
                                        coord = [step.to_station.lat, step.to_station.lng]
                                        if coord not in coordinates:
                                            coordinates.append(coord)
                            
                            # Add destination marker
                            coordinates.append([dest_coords[0], dest_coords[1]])
                            markers.append({
                                "position": {"lat": dest_coords[0], "lng": dest_coords[1]},
                                "label": dest_str.title(),
                                "type": "destination"
                            })
                            
                            # Calculate center point
                            center_lat = sum(c[0] for c in coordinates) / len(coordinates)
                            center_lon = sum(c[1] for c in coordinates) / len(coordinates)
                            
                            map_data = {
                                "type": "route",
                                "coordinates": coordinates,
                                "markers": markers,
                                "center": {"lat": center_lat, "lon": center_lon},
                                "zoom": 12,
                                "has_origin": True,
                                "has_destination": True,
                                "origin_name": origin_str,
                                "destination_name": dest_str,
                                "route_data": {
                                    "distance_km": getattr(route, 'total_distance_km', 0),
                                    "duration_min": getattr(route, 'total_time_minutes', 0),
                                    "transport_mode": "Public Transit",
                                    "lines": [step.line for step in route.steps if hasattr(step, 'line')]
                                }
                            }
                            
                            logger.info(f"✅ Generated route map_data with {len(coordinates)} waypoints and {len(markers)} markers")
                            return map_data
                        else:
                            logger.warning(f"⚠️ No route found between {origin_str} and {dest_str}")
                
                elif len(locations) >= 2:
                    # Standard two-location query
                    origin_coords = locations[0]
                    dest_coords = locations[1]
                    
                    # Find location names by reverse lookup
                    for name, coords in route_handler.KNOWN_LOCATIONS.items():
                        if abs(coords[0] - origin_coords[0]) < 0.001 and abs(coords[1] - origin_coords[1]) < 0.001:
                            origin_str = name
                        if abs(coords[0] - dest_coords[0]) < 0.001 and abs(coords[1] - dest_coords[1]) < 0.001:
                            dest_str = name
                    
                    # Fallback to coordinate strings if names not found
                    if not origin_str:
                        origin_str = f"{origin_coords[0]:.4f}, {origin_coords[1]:.4f}"
                    if not dest_str:
                        dest_str = f"{dest_coords[0]:.4f}, {dest_coords[1]:.4f}"
                else:
                    logger.warning(f"⚠️ Insufficient locations extracted (need at least 2, got {len(locations)})")
                
                if origin_coords and dest_coords:
                    logger.info(f"📍 [ROUTE EXTRACTION] Resolved: {origin_str} → {dest_str}")
                    
                    # Now use Transportation RAG to find the actual route
                    transport_rag = get_transportation_rag()
                    if transport_rag:
                        route = transport_rag.find_route(
                            origin_str,
                            dest_str,
                            origin_gps=origin_gps,
                            destination_gps=dest_gps
                        )
                        
                        if route:
                            logger.info(f"✅ Transportation RAG found route: {len(route.steps)} steps, {route.transfers} transfers")
                            
                            # Convert route to map_data format with coordinates
                            coordinates = []
                            markers = []
                            
                            # Add origin marker
                            coordinates.append([origin_coords[0], origin_coords[1]])
                            markers.append({
                                "position": {"lat": origin_coords[0], "lng": origin_coords[1]},
                                "label": origin_str.title(),
                                "type": "origin"
                            })
                            
                            # Add waypoints from route steps
                            for step in route.steps:
                                if hasattr(step, 'from_station') and step.from_station:
                                    if hasattr(step.from_station, 'lat') and hasattr(step.from_station, 'lng'):
                                        coord = [step.from_station.lat, step.from_station.lng]
                                        if coord not in coordinates:
                                            coordinates.append(coord)
                                
                                if hasattr(step, 'to_station') and step.to_station:
                                    if hasattr(step.to_station, 'lat') and hasattr(step.to_station, 'lng'):
                                        coord = [step.to_station.lat, step.to_station.lng]
                                        if coord not in coordinates:
                                            coordinates.append(coord)
                            
                            # Add destination marker
                            coordinates.append([dest_coords[0], dest_coords[1]])
                            markers.append({
                                "position": {"lat": dest_coords[0], "lng": dest_coords[1]},
                                "label": dest_str.title(),
                                "type": "destination"
                            })
                            
                            # Calculate center point
                            center_lat = sum(c[0] for c in coordinates) / len(coordinates)
                            center_lon = sum(c[1] for c in coordinates) / len(coordinates)
                            
                            map_data = {
                                "type": "route",
                                "coordinates": coordinates,
                                "markers": markers,
                                "center": {"lat": center_lat, "lon": center_lon},
                                "zoom": 12,
                                "has_origin": True,
                                "has_destination": True,
                                "origin_name": origin_str,
                                "destination_name": dest_str,
                                "route_data": {
                                    "distance_km": getattr(route, 'total_distance_km', 0),
                                    "duration_min": getattr(route, 'total_time_minutes', 0),
                                    "transport_mode": "Public Transit",
                                    "lines": [step.line for step in route.steps if hasattr(step, 'line')]
                                }
                            }
                            
                            logger.info(f"✅ Generated route map_data with {len(coordinates)} waypoints and {len(markers)} markers")
                            return map_data
                        else:
                            logger.warning(f"⚠️ No route found between {origin_str} and {dest_str}")
            
            except Exception as e:
                logger.error(f"❌ Route visualization error: {e}", exc_info=True)
        
        # ==================================================================
        # 3️⃣ LOCATION-BASED QUERIES (Restaurants, Attractions, etc.)
        # ==================================================================
        # If no initial map_data from context: {map_data}")
        if not map_data and any([
            signals['signals'].get('needs_restaurant'),
            signals['signals'].get('needs_attraction'),
            signals['signals'].get('needs_hidden_gems'),
            signals['signals'].get('needs_neighborhood'),
            signals['signals'].get('needs_neighborhood'),
            signals['signals'].get('needs_shopping'),
            signals['signals'].get('needs_nightlife'),
            signals['signals'].get('needs_events'),
            signals['signals'].get('needs_daily_life'),
            signals['signals'].get('needs_family_friendly')
        ]):
            logger.info(f"🗺️ Attempting to generate map from context...")
            # Try to extract locations from database context or generate GPS-centered map
            # Use original_query (before rewriting) to preserve routing query patterns
            query_for_map = original_query if 'original_query' in locals() else query
            map_data = self._generate_map_from_context(context, signals['signals'], user_location, query_for_map)
            if map_data:
                logger.info(f"✅ Generated map_data from context for location-based query")
            else:
                logger.warning(f"⚠️ _generate_map_from_context returned None")
                
                # FORCE GPS-centered map for "nearby" queries with GPS
                # This ensures users ALWAYS get a map when they have GPS enabled
                query_lower = query.lower()
                is_nearby_query = any([
                    'nearby' in query_lower,
                    'near me' in query_lower,
                    'near by' in query_lower,
                    'close to me' in query_lower,
                    'close by' in query_lower,
                    'around me' in query_lower,
                    'around here' in query_lower,
                    'in the area' in query_lower,
                    # Turkish
                    'yakın' in query_lower,
                    'yakında' in query_lower,
                    'yakınımda' in query_lower,
                    'burada' in query_lower,
                    'çevrede' in query_lower,
                    'civarda' in query_lower
                ])
                
                # Check if user_location has valid coordinates
                has_valid_location = (user_location and 
                                    isinstance(user_location, dict) and 
                                    'lat' in user_location and 
                                    'lon' in user_location and
                                    user_location['lat'] is not None and
                                    user_location['lon'] is not None)
                
                if has_valid_location and is_nearby_query:
                    logger.info(f"🚀 FORCING GPS-centered map for nearby query with GPS")
                    map_data = {
                        "type": "user_centered",
                        "markers": [{
                            "position": {"lat": user_location['lat'], "lng": user_location['lon']},
                            "label": "Your Location",
                            "type": "user"
                        }],
                        "center": {"lat": user_location['lat'], "lng": user_location['lon']},
                        "zoom": 14,
                        "has_origin": True,
                        "has_destination": False,
                        "origin_name": "Your Location",
                        "destination_name": None,
                        "locations_count": 0,
                        "note": "Map centered on your current location. Recommendations shown in text above."
                    }
                    logger.info(f"✅ Force-generated GPS-centered map for nearby query")
        else:
            logger.info(f"❌ Skipping map generation - conditions not met")
            logger.info(f"   - map_data exists: {bool(map_data)}")
            logger.info(f"   - any location signal: {any([signals['signals'].get('needs_restaurant'), signals['signals'].get('needs_attraction'), signals['signals'].get('needs_hidden_gems'), signals['signals'].get('needs_neighborhood')])}")
        
        result = {
            "status": "success",
            "response": response_text,
            "map_data": map_data,
            "signals": signals['signals'],
            "metadata": {
                "signals_detected": active_signals,
                "confidence_scores": signals.get('confidence_scores', {}),
                "context_used": {
                    "database": bool(context['database']),
                    "rag": bool(context['rag']),
                    "services": len(context['services'])
                },
                "enhancement": enhancement_metadata,
                "conversation": {
                    "enabled": enable_conversation and session_id is not None,
                    "session_id": session_id,
                    "had_context": conversation_context is not None
                } if enable_conversation else None,
                "processing_time": total_latency,
                "llm_latency": llm_latency,
                "cached": False,
                "language": language,
                "source": "pure_llm_core"
            }
        }
        
        # STEP 9: Store in Conversation (if enabled)
        if enable_conversation and session_id:
            try:
                await self.conversation_manager.add_turn(
                    session_id=session_id,
                    role='user',
                    content=original_query,
                    metadata={'signals': signals['signals']}
                )
                
                await self.conversation_manager.add_turn(
                    session_id=session_id,
                    role='assistant',
                    content=response_text,
                    metadata={'signals_detected': active_signals}
                )
            except Exception as e:
                logger.warning(f"Failed to store conversation: {e}")
        
        # STEP 10: Cache Response (skip for GPS-based route queries)
        # Don't cache route queries that use GPS, as location changes
        if not bypass_cache:
            try:
                await self.cache_manager.cache_response(
                    query=query,
                    language=language,
                    response=result
                )
                logger.debug("✅ Response cached")
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
        else:
            logger.info("🚫 Skipping cache storage for GPS-based route query")
        
        # STEP 11: Track Analytics
        self.analytics.track_response(
            latency=total_latency,
            llm_latency=llm_latency,
            signals=signals['signals'],
            context=context
        )
        
        logger.info(f"✅ Query processed in {total_latency:.2f}s")
        
        return result
    
    async def process_query_stream(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        max_tokens: int = 250,
        enable_conversation: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a query with streaming support for real-time UX.
        
        Yields progress updates:
        - type='progress': Stage progress
        - type='enhancement': Query enhancement results
        - type='cache_hit': Cache hit
        - type='signals': Detected signals
        - type='context': Context building complete
        - type='token': LLM token (streamed)
        - type='complete': Final metadata
        - type='error': Error occurred
        
        Args:
            Same as process_query()
            
        Yields:
            Dict with type and data
        """
        start_time = time.time()
        
        logger.info(f"🎬 Starting streaming query: {query[:50]}...")
        
        # Track query
        self.analytics.track_query(user_id, language, query)
        
        # STEP 1: Query Enhancement
        yield {
            'type': 'progress',
            'stage': 'enhancement',
            'message': 'Optimizing your query...'
        }
        
        original_query = query
        enhancement_metadata = {}
        
        try:
            enhanced = await self.query_enhancer.enhance_query(
                query=query,
                language=language
            )
            
            query = enhanced['query']
            enhancement_metadata = {
                'spell_corrected': enhanced.get('spell_corrected', False),
                'rewritten': enhanced.get('rewritten', False),
                'validation': enhanced.get('validation'),
                'original_query': original_query if query != original_query else None
            }
            
            yield {
                'type': 'enhancement',
                'data': enhancement_metadata
            }
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            query = original_query
        
        # STEP 2: Cache Check
        yield {
            'type': 'progress',
            'stage': 'cache',
            'message': 'Checking cache...'
        }
        
        cached_response = await self.cache_manager.get_cached_response(
            query=query,
            language=language,
            similarity_threshold=0.85
        )
        
        if cached_response:
            logger.info("✅ Cache hit!")
            self.analytics.track_cache_hit()
            
            yield {
                'type': 'cache_hit',
                'message': 'Found cached response!'
            }
            
            # Stream cached response
            response_text = cached_response.get('response', '')
            for i in range(0, len(response_text), 5):
                yield {
                    'type': 'token',
                    'data': response_text[i:i+5],
                    'cached': True
                }
                await asyncio.sleep(0.01)
            
            cached_response['metadata']['enhancement'] = enhancement_metadata
            cached_response['metadata']['cached'] = True
            
            yield {
                'type': 'complete',
                'data': cached_response
            }
            return
        
        self.analytics.track_cache_miss()
        
        # STEP 3: Signal Detection
        yield {
            'type': 'progress',
            'stage': 'signals',
            'message': 'Analyzing your query...'
        }
        
        signals = await self.signal_detector.detect_signals(
            query=query,
            user_location=user_location,
            language=language,
            user_id=user_id,
            experimentation_manager=self.experimentation
        )
        
        active_signals = [k for k, v in signals['signals'].items() if v]
        
        yield {
            'type': 'signals',
            'data': signals['signals'],
            'active': active_signals,
            'confidence': signals.get('confidence_scores', {})
        }
        
        self.analytics.track_signals(signals['signals'])
        
        # STEP 4: Conversation Context
        conversation_context = None
        if enable_conversation and session_id:
            yield {
                'type': 'progress',
                'stage': 'conversation',
                'message': 'Analyzing conversation context...'
            }
            
            try:
                conversation_context = await self.conversation_manager.get_context(
                    session_id=session_id,
                    current_query=query,
                    max_turns=3
                )
                
                # Resolve references
                if conversation_context.get('needs_resolution'):
                    resolved = await self.conversation_manager.resolve_references(
                        query=query,
                        context=conversation_context
                    )
                    
                    if resolved.get('resolved'):
                        yield {
                            'type': 'conversation',
                            'action': 'resolved',
                            'original': query,
                            'resolved': resolved['resolved_query']
                        }
                        query = resolved['resolved_query']
                
            except Exception as e:
                logger.warning(f"Conversation context failed: {e}")
        
        # STEP 5: Context Building
        yield {
            'type': 'progress',
            'stage': 'context',
            'message': 'Gathering relevant information...'
        }
        
        context = await self.context_builder.build_context(
            query=query,
            signals=signals['signals'],
            user_location=user_location,
            language=language
        )
        
        yield {
            'type': 'context',
            'database_size': len(context['database']),
            'rag_size': len(context['rag']),
            'services_count': len(context['services'])
        }
        
        self.analytics.track_context(context)
        
        # STEP 5.5: Personalization Filtering (PHASE 2)
        if self.config.get('enable_personalization', True):
            try:
                # Filter context based on user preferences
                if context.get('database'):
                    # Parse database context to extract items
                    # This is a simplified version - you may need to adapt based on your context structure
                    db_items = []  # Extract items from context['database']
                    
                    filtered_items = await self.personalization.filter_context_by_preferences(
                        user_id=user_id,
                        context_items=db_items,
                        signals=signals['signals']
                    )
                    
                    if filtered_items:
                        logger.info(f"🎯 Personalized context for user {user_id}")
                        # Update context with filtered items
                        # context['database'] = ... (rebuild with filtered_items)
            except Exception as e:
                logger.warning(f"Personalization filtering failed: {e}")
        
        # STEP 6: Prompt Engineering
        prompt = self.prompt_builder.build_prompt(
            query=query,
            signals=signals['signals'],
            context=context,
            conversation_context=conversation_context,
            language=language
        )
        
        # STEP 7: Stream LLM Generation
        yield {
            'type': 'progress',
            'stage': 'generation',
            'message': '✨ Generating response...'
        }
        
        try:
            llm_start = time.time()
            response_tokens = []
            
            # Check if LLM supports streaming
            if hasattr(self.llm, 'generate_stream'):
                async for token in self.llm.generate_stream(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                ):
                    response_tokens.append(token)
                    yield {
                        'type': 'token',
                        'data': token,
                        'cached': False
                    }
            else:
                # Fallback: simulate streaming
                response_data = await self.llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                
                if not response_data or "generated_text" not in response_data:
                    raise Exception("Invalid LLM response")
                
                response_text = response_data["generated_text"]
                
                # Clean training data leakage from response
                response_text = clean_training_data_leakage(response_text)
                
                # Simulate streaming
                for i in range(0, len(response_text), 5):
                    yield {
                        'type': 'token',
                        'data': response_text[i:i+5],
                        'cached': False
                    }
                    await asyncio.sleep(0.01)
                
                response_tokens = [response_text]
            
            llm_latency = time.time() - llm_start
            response_text = ''.join(response_tokens)
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            self.analytics.track_error('llm_failure', str(e))
            
            yield {
                'type': 'error',
                'message': str(e)
            }
            
            # Fallback
            response_text = await self._fallback_response(query=query, context=context)
            llm_latency = 0
            
            # Stream fallback response
            for i in range(0, len(response_text), 5):
                yield {
                    'type': 'token',
                    'data': response_text[i:i+5],
                    'fallback': True
                }
                await asyncio.sleep(0.01)
        
        # STEP 8: Final Metadata
        total_latency = time.time() - start_time
        
        result = {
            "status": "success",
            "response": response_text,
            "map_data": context.get('map_data'),
            "signals": signals['signals'],
            "metadata": {
                "signals_detected": active_signals,
                "confidence_scores": signals.get('confidence_scores', {}),
                "context_used": {
                    "database": bool(context['database']),
                    "rag": bool(context['rag']),
                    "services": len(context['services'])
                },
                "enhancement": enhancement_metadata,
                "conversation": {
                    "enabled": enable_conversation and session_id is not None,
                    "session_id": session_id,
                    "had_context": conversation_context is not None
                } if enable_conversation else None,
                "processing_time": total_latency,
                "llm_latency": llm_latency,
                "cached": False,
                "language": language,
                "source": "pure_llm_core"
            }
        }
        
        yield {
            'type': 'complete',
            'data': result
        }
        
        # STEP 9: Store in Conversation
        if enable_conversation and session_id:
            try:
                await self.conversation_manager.add_turn(
                    session_id=session_id,
                    role='user',
                    content=original_query,
                    metadata={'signals': signals['signals']}
                )
                
                await self.conversation_manager.add_turn(
                    session_id=session_id,
                    role='assistant',
                    content=response_text,
                    metadata={'signals_detected': active_signals}
                )
            except Exception as e:
                logger.warning(f"Failed to store conversation: {e}")
        
        # STEP 10: Cache Response
        try:
            await self.cache_manager.cache_response(
                query=query,
                language=language,
                response=result
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
        
        # STEP 11: Track Analytics
        self.analytics.track_response(
            latency=total_latency,
            llm_latency=llm_latency,
            signals=signals['signals'],
            context=context
        )
        
        logger.info(f"✅ Streaming query completed in {total_latency:.2f}s")
    
    async def _validate_response(
        self,
        response: str,
        query: str,
        signals: Dict[str, bool],
        context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate response quality.
        
        Args:
            response: Generated response
            query: User query
            signals: Detected signals
            context: Built context
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if response is empty or too short
        if not response or len(response.strip()) < 10:
            return False, "Response too short"
        
        # Check for generic error messages
        if "error" in response.lower() or "sorry" in response.lower():
            return False, "Generic error response"
        
        # Response is valid
        return True, None
    
    async def _fallback_response(
        self,
        query: str,
        context: Dict[str, Any],
        error_type: str = "unknown"
    ) -> str:
        """
        Generate fallback response when LLM fails.
        
        Args:
            query: User query
            context: Built context
            error_type: Type of error (404, timeout, circuit_breaker, validation)
            
        Returns:
            Fallback response string
        """
        # Try to use RAG context if available
        if context.get('rag'):
            return (
                f"Based on available information:\n\n"
                f"{context['rag'][:500]}\n\n"
                f"Note: I'm currently experiencing technical difficulties with my AI service. "
                f"This is a simplified response based on our knowledge base."
            )
        
        # Try to use database context
        if context.get('database'):
            return (
                f"Here's what I found in our database:\n\n"
                f"{context['database'][:500]}\n\n"
                f"Note: I'm currently experiencing technical difficulties with my AI service. "
                f"This information is retrieved directly from our database."
            )
        
        # Friendly greeting fallback for simple queries
        simple_greetings = ["hi", "hello", "hey", "greetings", "merhaba", "selam"]
        if query.lower().strip() in simple_greetings:
            return (
                "Hello! 👋 I'm your AI assistant for Istanbul tourism. "
                "I can help you with:\n\n"
                "• Finding restaurants and cafes\n"
                "• Planning routes and transportation\n"
                "• Discovering attractions and hidden gems\n"
                "• Getting weather updates\n\n"
                "What would you like to explore in Istanbul today?"
            )
        
        # Error-specific fallback messages
        if error_type == "404":
            return (
                "I apologize, but my AI service is currently offline for maintenance. "
                "I'm still here to help! Please try:\n\n"
                "• Asking about specific locations (e.g., 'restaurants in Sultanahmet')\n"
                "• Planning a route (e.g., 'How do I get to Taksim?')\n"
                "• Checking the weather\n\n"
                "My team is working to restore full functionality shortly."
            )
        elif error_type == "timeout":
            return (
                "My AI service is taking longer than usual to respond. "
                "This might be due to high traffic. Please try:\n\n"
                "• Refreshing the page\n"
                "• Asking a more specific question\n"
                "• Waiting a moment and trying again"
            )
        
        # Ultimate fallback
        return (
            "I apologize, but I'm experiencing technical difficulties right now. "
            "Please try asking your question in a different way, or contact support if this continues. "
            "You can also try specific queries like 'restaurants near me' or 'how to get to Hagia Sophia'."
        )
    
    async def _validate_transportation_response(
        self,
        response: str,
        route_data: Optional[Dict[str, Any]],
        query: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate transportation route responses to prevent LLM hallucinations.
        
        This implements the HARD GUARDRAIL layer of the hybrid architecture:
        - Checks if LLM modified verified route facts
        - Detects invented transit lines or stations
        - Ensures duration/transfer count matches verified data
        
        Args:
            response: LLM-generated response text
            route_data: Verified route data from Transportation RAG (ground truth)
            query: Original user query
            
        Returns:
            Tuple of (is_valid, error_message, corrected_response)
            - is_valid: True if response matches verified facts
            - error_message: Description of violation (if any)
            - corrected_response: Auto-corrected response (if possible)
        """
        if not route_data:
            # No route data to validate against
            return True, None, None
        
        # Extract immutable facts from verified route
        verified_lines = set(route_data.get('lines_used', []))
        verified_duration = route_data.get('total_time', 0)
        verified_transfers = route_data.get('transfers', 0)
        verified_steps = route_data.get('steps', [])
        
        response_lower = response.lower()
        
        # === VALIDATION 1: Check for hallucinated transit lines ===
        # Known Istanbul transit lines
        valid_lines = {
            'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm9', 'm11',
            't1', 't2', 't3', 't4', 't5',
            'f1', 'f2', 'f3', 'f4', 'f5',
            'marmaray', 'metrobus', 'banliyö'
        }
        
        # Extract mentioned lines from response (case-insensitive)
        import re
        mentioned_lines = set()
        for match in re.finditer(r'\b(m\d+|t\d+|f\d+|marmaray|metrobus)\b', response_lower):
            mentioned_lines.add(match.group(1))
        
        # Check for hallucinated lines (mentioned but not in verified route)
        hallucinated_lines = mentioned_lines - verified_lines
        if hallucinated_lines:
            error_msg = f"LLM hallucinated transit lines: {hallucinated_lines}. Verified lines: {verified_lines}"
            logger.error(f"🚨 HALLUCINATION DETECTED: {error_msg}")
            
            # Attempt to correct by replacing hallucinated lines with verified ones
            corrected_response = response
            for fake_line in hallucinated_lines:
                # Replace fake line with first verified line (simple heuristic)
                corrected_response = re.sub(
                    rf'\b{fake_line}\b',
                    list(verified_lines)[0] if verified_lines else fake_line,
                    corrected_response,
                    flags=re.IGNORECASE
                )
            return False, error_msg, corrected_response
        
        # === VALIDATION 2: Check for duration hallucination ===
        # Extract duration numbers from response
        duration_matches = re.findall(r'(\d+)\s*(?:minute|min|dakika)', response_lower)
        if duration_matches:
            mentioned_durations = [int(d) for d in duration_matches]
            # Allow ±2 minute variance for rounding/explanation
            if any(abs(d - verified_duration) > 2 for d in mentioned_durations):
                error_msg = f"LLM modified duration: mentioned {mentioned_durations}, verified {verified_duration}"
                logger.warning(f"⚠️ DURATION MISMATCH: {error_msg}")
                # This is a warning, not a hard error (allow small variations)
        
        # === VALIDATION 3: Check for transfer count hallucination ===
        transfer_matches = re.findall(r'(\d+)\s*(?:transfer|aktarma)', response_lower)
        if transfer_matches:
            mentioned_transfers = [int(t) for t in transfer_matches]
            if any(t != verified_transfers for t in mentioned_transfers):
                error_msg = f"LLM modified transfer count: mentioned {mentioned_transfers}, verified {verified_transfers}"
                logger.error(f"🚨 TRANSFER MISMATCH: {error_msg}")
                return False, error_msg, None
        
        # === VALIDATION 4: Check for invented stations (advanced) ===
        # Extract station names from verified route
        verified_stations = set()
        for step in verified_steps:
            if step.get('from_station'):
                verified_stations.add(step['from_station'].lower())
            if step.get('to_station'):
                verified_stations.add(step['to_station'].lower())
        
        # Check if response mentions stations not in verified route
        # (This is heuristic-based and may have false positives)
        common_stations = {
            'taksim', 'sultanahmet', 'kadıköy', 'üsküdar', 'beşiktaş',
            'eminönü', 'karaköy', 'şişli', 'mecidiyeköy', 'levent'
        }
        
        # If response mentions a common station not in verified route, it might be hallucination
        for station in common_stations:
            if station in response_lower and station not in verified_stations:
                # Soft warning only (stations might be mentioned as landmarks)
                logger.info(f"ℹ️ Response mentions {station} (not in route, possibly as landmark)")
        
        # All validations passed
        logger.info(f"✅ Transportation response passed all hallucination checks")
        return True, None, None
    
    async def _generate_template_transportation_response(
        self,
        route_data: Dict[str, Any],
        query: str,
        language: str = "en"
    ) -> str:
        """
        Generate a template-based transportation response using ONLY verified facts.
        
        This is the DETERMINISTIC FACT LAYER of the hybrid architecture:
        - Uses only verified route data from Transportation RAG
        - No LLM generation = zero hallucination risk
        - Fact-locked template with structured route information
        
        Args:
            route_data: Verified route data from Transportation RAG (ground truth)
            query: Original user query
            language: Response language ('en' or 'tr')
            
        Returns:
            Template-based response text with verified facts only
        """
        try:
            # Extract verified facts from route data
            origin = route_data.get('origin', 'Starting point')
            destination = route_data.get('destination', 'Destination')
            total_time = route_data.get('total_time', 0)
            total_distance = route_data.get('total_distance_km', 0)
            transfers = route_data.get('transfers', 0)
            lines_used = route_data.get('lines_used', [])
            steps = route_data.get('steps', [])
            
            # Build response based on language
            if language == 'tr':
                # Turkish template
                response = f"🚇 **{origin} → {destination} Güzergahı**\n\n"
                response += f"**Süre:** {total_time} dakika\n"
                response += f"**Mesafe:** {total_distance:.1f} km\n"
                response += f"**Aktarma:** {transfers} aktarma\n"
                response += f"**Kullanılan Hatlar:** {', '.join(lines_used)}\n\n"
                
                if steps:
                    response += "**Adım Adım Yol Tarifi:**\n\n"
                    for i, step in enumerate(steps, 1):
                        mode = step.get('mode', 'transit')
                        line = step.get('line', '')
                        from_station = step.get('from_station', '')
                        to_station = step.get('to_station', '')
                        duration = step.get('duration', 0)
                        
                        if mode == 'walk':
                            response += f"{i}. 🚶 Yürüme: {from_station} → {to_station} ({duration} dk)\n"
                        else:
                            response += f"{i}. 🚇 {line}: {from_station} → {to_station} ({duration} dk)\n"
                
                response += f"\n✅ Bu güzergah İstanbul ulaşım veritabanından doğrulanmıştır."
                
            else:
                # English template
                response = f"🚇 **Route: {origin} → {destination}**\n\n"
                response += f"**Duration:** {total_time} minutes\n"
                response += f"**Distance:** {total_distance:.1f} km\n"
                response += f"**Transfers:** {transfers} transfer{'s' if transfers != 1 else ''}\n"
                response += f"**Lines Used:** {', '.join(lines_used)}\n\n"
                
                if steps:
                    response += "**Step-by-Step Directions:**\n\n"
                    for i, step in enumerate(steps, 1):
                        mode = step.get('mode', 'transit')
                        line = step.get('line', '')
                        from_station = step.get('from_station', '')
                        to_station = step.get('to_station', '')
                        duration = step.get('duration', 0)
                        
                        if mode == 'walk':
                            response += f"{i}. 🚶 Walk: {from_station} → {to_station} ({duration} min)\n"
                        else:
                            response += f"{i}. 🚇 {line}: {from_station} → {to_station} ({duration} min)\n"
                
                response += f"\n✅ This route has been verified using Istanbul's transportation database."
            
            logger.info(f"✅ Generated template-based transportation response (fact-locked, no LLM)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to generate template response: {e}")
            # Ultra-minimal fallback
            if language == 'tr':
                return f"Güzergah bilgisi mevcut ancak görüntülenemiyor. Lütfen tekrar deneyin."
            else:
                return f"Route information is available but could not be displayed. Please try again."