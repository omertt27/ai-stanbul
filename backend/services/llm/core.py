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
        
        # STEP 2: Cache Check
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
            signal_confidence=overall_confidence  # NEW: Pass confidence for adaptive context
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
        # call the Transportation RAG system to generate route visualization
        if not map_data and signals['signals'].get('needs_transportation'):
            logger.info(f"🚇 Detected transportation query, generating route visualization...")
            try:
                from services.transportation_rag_system import get_transportation_rag
                import re
                
                transport_rag = get_transportation_rag()
                if transport_rag:
                    # Extract origin and destination from query
                    query_lower = query.lower()
                    
                    # Try pattern: "from X to Y" or "X to Y"
                    from_to_match = re.search(
                        r'(?:from|leaving|starting)\s+([a-zığüşöçıİĞÜŞÖÇ\s]+?)\s+(?:to|going|heading|reach)\s+([a-zığüşöçıİĞÜŞÖÇ\s]+?)(?:\?|$|\s+and|\s+or)',
                        query_lower,
                        re.IGNORECASE
                    )
                    
                    # Try pattern: "how to get to Y from X" or "how can i go to Y from X"
                    if not from_to_match:
                        from_to_match = re.search(
                            r'(?:how|way).*?(?:to|go|get|reach)\s+(?:to\s+)?([a-zığüşöçıİĞÜŞÖÇ\s]+?)\s+from\s+([a-zığüşöçıİĞÜŞÖÇ\s]+?)(?:\?|$)',
                            query_lower,
                            re.IGNORECASE
                        )
                        
                        if from_to_match:
                            # Swap groups (destination comes before origin in this pattern)
                            destination_str = from_to_match.group(1).strip()
                            origin_str = from_to_match.group(2).strip()
                        else:
                            origin_str = None
                            destination_str = None
                    else:
                        origin_str = from_to_match.group(1).strip()
                        destination_str = from_to_match.group(2).strip()
                    
                    if origin_str and destination_str:
                        logger.info(f"📍 Extracted route: {origin_str} → {destination_str}")
                        
                        # Find route using Transportation RAG
                        route = transport_rag.find_route(origin_str, destination_str)
                        
                        if route:
                            logger.info(f"✅ Route found: {len(route.steps)} steps, {route.transfers} transfers")
                            
                            # Convert route to map_data format
                            # Extract coordinates from stations along the route
                            coordinates = []
                            markers = []
                            
                            # Add origin marker
                            if route.steps and len(route.steps) > 0:
                                first_step = route.steps[0]
                                origin_name = first_step.get('from', origin_str)
                                
                                # Find station coordinates
                                for station_id, station in transport_rag.stations.items():
                                    if station.name.lower() == origin_name.lower():
                                        coordinates.append([station.lat, station.lon])
                                        markers.append({
                                            "lat": station.lat,
                                            "lon": station.lon,
                                            "label": origin_name,
                                            "type": "origin"
                                        })
                                        break
                            
                            # Add waypoints for each step
                            for step in route.steps:
                                if step.get('type') == 'transit':
                                    to_station = step.get('to')
                                    # Find station coordinates
                                    for station_id, station in transport_rag.stations.items():
                                        if station.name.lower() == to_station.lower():
                                            coordinates.append([station.lat, station.lon])
                                            break
                            
                            # Add destination marker
                            if route.steps and len(route.steps) > 0:
                                last_step = route.steps[-1]
                                dest_name = last_step.get('to', destination_str)
                                
                                # Find station coordinates
                                for station_id, station in transport_rag.stations.items():
                                    if station.name.lower() == dest_name.lower():
                                        if [station.lat, station.lon] not in coordinates:
                                            coordinates.append([station.lat, station.lon])
                                        markers.append({
                                            "lat": station.lat,
                                            "lon": station.lon,
                                            "label": dest_name,
                                            "type": "destination"
                                        })
                                        break
                            
                            # Build map_data
                            if coordinates:
                                # Calculate center point
                                center_lat = sum(c[0] for c in coordinates) / len(coordinates)
                                center_lon = sum(c[1] for c in coordinates) / len(coordinates)
                                
                                map_data = {
                                    "type": "route",
                                    "coordinates": coordinates,
                                    "markers": markers,
                                    "center": {"lat": center_lat, "lon": center_lon},
                                    "zoom": 12,
                                    "route_data": {
                                        "distance_km": route.total_distance,
                                        "duration_min": route.total_time,
                                        "transport_mode": ", ".join(route.lines_used),
                                        "lines": route.lines_used,
                                        "transfers": route.transfers
                                    },
                                    "has_origin": True,
                                    "has_destination": True,
                                    "origin_name": origin_str.title(),
                                    "destination_name": destination_str.title()
                                }
                                
                                logger.info(f"✅ Generated map_data with {len(coordinates)} waypoints and {len(markers)} markers")
                        else:
                            logger.warning(f"⚠️ No route found between {origin_str} and {destination_str}")
                    else:
                        logger.info(f"⚠️ Could not extract origin/destination from query")
                        
            except Exception as e:
                logger.error(f"❌ Transportation route visualization error: {e}", exc_info=True)
        
        # DEBUG LOGGING
        logger.info(f"🔍 MAP GENERATION DEBUG:")
        logger.info(f"  - Query: {query}")
        logger.info(f"  - User location: {user_location}")
        logger.info(f"  - Initial map_data from context: {map_data}")
        logger.info(f"  - needs_restaurant: {signals['signals'].get('needs_restaurant')}")
        logger.info(f"  - needs_attraction: {signals['signals'].get('needs_attraction')}")
        logger.info(f"  - needs_hidden_gems: {signals['signals'].get('needs_hidden_gems')}")
        logger.info(f"  - needs_neighborhood: {signals['signals'].get('needs_neighborhood')}")
        
        # If no map_data but we have location-based or transportation signals, generate basic map data
        if not map_data and any([
            signals['signals'].get('needs_transportation'),  # Added for routing queries
            signals['signals'].get('needs_restaurant'),
            signals['signals'].get('needs_attraction'),
            signals['signals'].get('needs_hidden_gems'),
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
    
    # ========================================================================
    # HELPER METHODS - Analytics, Experimentation, etc.
    # ========================================================================
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        return self.analytics.get_summary()
    
    def record_user_feedback(
        self,
        query: str,
        detected_signals: Dict[str, bool],
        confidence_scores: Dict[str, float],
        feedback_type: str,
        feedback_data: Dict[str, Any],
        language: str = "en"
    ):
        """Record user feedback for threshold learning."""
        if self.experimentation:
            self.experimentation.record_feedback(
                query=query,
                detected_signals=detected_signals,
                confidence_scores=confidence_scores,
                feedback_type=feedback_type,
                feedback_data=feedback_data,
                language=language
            )
    
    async def auto_tune_thresholds(
        self,
        language: str = "en",
        force: bool = False
    ) -> Dict[str, Any]:
        """Automatically tune thresholds based on feedback."""
        if self.experimentation:
            return await self.experimentation.auto_tune_thresholds(
                language=language,
                force=force
            )
        return {'status': 'disabled'}
    
    def get_conversation_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if self.conversation_manager:
            return self.conversation_manager.get_history(
                session_id=session_id,
                max_turns=max_turns
            )
        return []
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session."""
        if self.conversation_manager:
            self.conversation_manager.clear_session(session_id)
    
    async def get_query_suggestions(
        self,
        partial_query: str,
        language: str = "en",
        max_suggestions: int = 5
    ) -> List[str]:
        """Get autocomplete suggestions."""
        if self.query_enhancer:
            return await self.query_enhancer.get_suggestions(
                partial_query=partial_query,
                language=language,
                max_suggestions=max_suggestions
            )
        return []
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all subsystems.
        
        Returns health information including:
        - Overall system health
        - Circuit breaker states
        - Timeout metrics
        - Service availability
        - Performance metrics
        
        Returns:
            Dict with health status information
        """
        try:
            # Collect circuit breaker states
            circuit_breaker_status = {}
            all_healthy = True
            
            for service_name, cb in self.circuit_breakers.items():
                state = cb.get_state()
                circuit_breaker_status[service_name] = state
                
                # Mark as unhealthy if circuit is open
                if state['state'] == 'open':
                    all_healthy = False
            
            # Collect timeout metrics
            timeout_metrics = self.timeout_manager.get_metrics()
            
            # Calculate timeout health
            timeout_health = 'healthy'
            if timeout_metrics['timeout_rate'] > 25:  # >25% timeout rate
                timeout_health = 'unhealthy'
                all_healthy = False
            elif timeout_metrics['timeout_rate'] > 10:  # >10% timeout rate
                timeout_health = 'degraded'
                all_healthy = False
            
            # Get analytics summary
            analytics_summary = self.analytics.get_summary() if hasattr(self, 'analytics') else {}
            
            # Build comprehensive health report
            health_status = {
                'status': 'healthy' if all_healthy else 'degraded',
                'timestamp': datetime.now().isoformat(),
                'circuit_breakers': circuit_breaker_status,
                'timeout_metrics': {
                    'health': timeout_health,
                    'total_operations': timeout_metrics['total_operations'],
                    'timeout_rate': f"{timeout_metrics['timeout_rate']:.2f}%",
                    'timeouts_by_stage': timeout_metrics['timeouts_by_stage']
                },
                'services': {
                    'llm': {
                        'available': circuit_breaker_status.get('llm', {}).get('state') != 'open',
                        'circuit_state': circuit_breaker_status.get('llm', {}).get('state', 'unknown')
                    },
                    'database': {
                        'available': circuit_breaker_status.get('database', {}).get('state') != 'open',
                        'circuit_state': circuit_breaker_status.get('database', {}).get('state', 'unknown')
                    },
                    'rag': {
                        'available': circuit_breaker_status.get('rag', {}).get('state') != 'open',
                        'circuit_state': circuit_breaker_status.get('rag', {}).get('state', 'unknown')
                    },
                    'weather': {
                        'available': circuit_breaker_status.get('weather', {}).get('state') != 'open',
                        'circuit_state': circuit_breaker_status.get('weather', {}).get('state', 'unknown')
                    },
                    'events': {
                        'available': circuit_breaker_status.get('events', {}).get('state') != 'open',
                        'circuit_state': circuit_breaker_status.get('events', {}).get('state', 'unknown')
                    }
                },
                'performance': {
                    'total_queries': analytics_summary.get('total_queries', 0),
                    'avg_latency': analytics_summary.get('avg_latency', 0),
                    'cache_hit_rate': analytics_summary.get('cache_hit_rate', 0)
                },
                'subsystems': {
                    'signal_detector': hasattr(self, 'signal_detector'),
                    'context_builder': hasattr(self, 'context_builder'),
                    'prompt_builder': hasattr(self, 'prompt_builder'),
                    'analytics': hasattr(self, 'analytics'),
                    'query_enhancer': hasattr(self, 'query_enhancer'),
                    'conversation_manager': hasattr(self, 'conversation_manager'),
                    'cache_manager': hasattr(self, 'cache_manager'),
                    'experimentation': hasattr(self, 'experimentation')
                }
            }
            
            logger.info(f"Health check completed: {health_status['status']}")
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def test_circuit_breakers(self) -> Dict[str, Any]:
        """
        Test all circuit breakers by making health check calls to services.
        
        This is useful for:
        - Verifying circuit breaker configuration
        - Testing service connectivity
        - Monitoring service health
        
        Returns:
            Dict with test results for each service
        """
        results = {}
        
        logger.info("🧪 Testing circuit breakers...")
        
        # Test LLM service
        try:
            test_prompt = "Say 'OK' if you receive this."
            
            async def _test_llm():
                return await self.llm.generate(
                    prompt=test_prompt,
                    max_tokens=10,
                    temperature=0
                )
            
            await self.circuit_breakers['llm'].call(_test_llm)
            results['llm'] = {'status': 'success', 'message': 'LLM service responsive'}
            
        except CircuitBreakerError:
            results['llm'] = {'status': 'circuit_open', 'message': 'Circuit breaker is open'}
        except Exception as e:
            results['llm'] = {'status': 'error', 'message': str(e)}
        
        # Test Database
        try:
            async def _test_db():
                # Simple query to test connection
                return await self.db.execute("SELECT 1")
            
            await self.circuit_breakers['database'].call(_test_db)
            results['database'] = {'status': 'success', 'message': 'Database responsive'}
            
        except CircuitBreakerError:
            results['database'] = {'status': 'circuit_open', 'message': 'Circuit breaker is open'}
        except Exception as e:
            results['database'] = {'status': 'error', 'message': str(e)}
        
        # Test RAG service (if available)
        if self.config.get('rag_service'):
            try:
                async def _test_rag():
                    return await self.config['rag_service'].search("test", top_k=1)
                
                await self.circuit_breakers['rag'].call(_test_rag)
                results['rag'] = {'status': 'success', 'message': 'RAG service responsive'}
                
            except CircuitBreakerError:
                results['rag'] = {'status': 'circuit_open', 'message': 'Circuit breaker is open'}
            except Exception as e:
                results['rag'] = {'status': 'error', 'message': str(e)}
        
        # Test Weather API (if available)
        if self.config.get('weather_service'):
            try:
                async def _test_weather():
                    return await self.config['weather_service'].get_current_weather("Istanbul")
                
                await self.circuit_breakers['weather'].call(_test_weather)
                results['weather'] = {'status': 'success', 'message': 'Weather API responsive'}
                
            except CircuitBreakerError:
                results['weather'] = {'status': 'circuit_open', 'message': 'Circuit breaker is open'}
            except Exception as e:
                results['weather'] = {'status': 'error', 'message': str(e)}
        
        # Test Events API (if available)
        if self.config.get('events_service'):
            try:
                async def _test_events():
                    return await self.config['events_service'].get_upcoming_events(limit=1)
                
                await self.circuit_breakers['events'].call(_test_events)
                results['events'] = {'status': 'success', 'message': 'Events API responsive'}
                
            except CircuitBreakerError:
                results['events'] = {'status': 'circuit_open', 'message': 'Circuit breaker is open'}
            except Exception as e:
                results['events'] = {'status': 'error', 'message': str(e)}
        
        logger.info(f"Circuit breaker tests completed: {len(results)} services tested")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total_tests': len(results),
                'successful': sum(1 for r in results.values() if r['status'] == 'success'),
                'failed': sum(1 for r in results.values() if r['status'] in ['error', 'circuit_open'])
            }
        }


    # ===================================================================
    # PHASE 2: FEEDBACK & PERSONALIZATION METHODS
    # ===================================================================
    
    async def process_user_feedback(
        self,
        user_id: str,
        query: str,
        response: str,
        feedback_type: str,
        detected_signals: List[str],
        signal_scores: Dict[str, float],
        feedback_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user feedback for continuous improvement (Phase 2).
        
        Args:
            user_id: User identifier
            query: Original query
            response: System response
            feedback_type: 'positive', 'negative', or 'correction'
            detected_signals: Signals that were detected
            signal_scores: Confidence scores for signals
            feedback_details: Additional feedback (issues, corrections, etc.)
            
        Returns:
            Feedback processing result
        """
        logger.info(f"📝 Processing {feedback_type} feedback from user {user_id}")
        
        try:
            # Store feedback in personalization engine
            feedback_record = await self.personalization.process_feedback(
                user_id=user_id,
                query=query,
                response=response,
                feedback_type=feedback_type,
                detected_signals=detected_signals,
                signal_scores=signal_scores,
                feedback_details=feedback_details
            )
            
            # Update auto-tuner metrics
            if feedback_details and 'correct_signals' in feedback_details:
                correct_signals = set(feedback_details['correct_signals'])
                all_possible_signals = set(detected_signals) | correct_signals
                
                for signal in all_possible_signals:
                    was_detected = signal in detected_signals
                    should_be_detected = signal in correct_signals
                    
                    await self.auto_tuner.update_metrics_from_feedback(
                        signal=signal,
                        was_detected=was_detected,
                        should_be_detected=should_be_detected
                    )
            
            # Track in analytics
            self.analytics.track_feedback(user_id, feedback_type, query)
            
            logger.info(f"✅ Feedback processed successfully")
            
            return {
                'status': 'success',
                'feedback_recorded': True,
                'user_profile_updated': True,
                'metrics_updated': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to process feedback: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def record_user_interaction(
        self,
        user_id: str,
        query: str,
        selected_items: List[Dict[str, Any]],
        signals: List[str]
    ) -> Dict[str, Any]:
        """
        Record user interaction with results for preference learning (Phase 2).
        
        Args:
            user_id: User identifier
            query: User's query
            selected_items: Items user clicked/viewed (with type, cuisine, district, etc.)
            signals: Detected signals
            
        Returns:
            Interaction recording result
        """
        logger.info(f"📊 Recording interaction for user {user_id}")
        
        try:
            # Update user profile based on interaction
            await self.personalization.update_profile_from_interaction(
                user_id=user_id,
                query=query,
                selected_items=selected_items,
                signals=signals
            )
            
            logger.info(f"✅ User interaction recorded")
            
            return {
                'status': 'success',
                'profile_updated': True
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to record interaction: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user profile and preferences (Phase 2).
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile data
        """
        try:
            profile = await self.personalization.get_user_profile(user_id)
            return profile.to_dict()
        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            return {}
    
    async def run_auto_tuning(self, signals: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run auto-tuning for signal thresholds (Phase 2).
        
        Args:
            signals: Specific signals to tune (None = all)
            
        Returns:
            Tuning report
        """
        logger.info("🎯 Running auto-tuning...")
        
        try:
            result = await self.auto_tuner.run_auto_tuning(signals_to_tune=signals)
            return result
        except Exception as e:
            logger.error(f"Auto-tuning failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_tuning_report(self) -> Dict[str, Any]:
        """
        Get comprehensive tuning report (Phase 2).
        
        Returns:
            Detailed tuning metrics and history
        """
        try:
            return await self.auto_tuner.get_tuning_report()
        except Exception as e:
            logger.error(f"Failed to get tuning report: {e}")
            return {}
    
    def _generate_map_from_context(
        self,
        context: Dict[str, Any],
        signals: Dict[str, bool],
        user_location: Optional[Dict[str, float]],
        query: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Generate basic map_data from database context for location-based queries.
        
        When the map service doesn't generate routing data, but we have locations
        from database (restaurants, attractions, etc.), we create a simple map_data
        structure with markers.
        
        For "nearby" queries with GPS but no specific locations in database,
        generates map centered on user location.
        
        For routing queries ("how to get to X"), looks up destination coordinates
        from Istanbul Knowledge and creates a route map.
        
        Args:
            context: Built context with database/service data
            signals: Detected signals
            user_location: User GPS location
            query: Original user query (to detect "nearby" keywords)
            
        Returns:
            Map data dict with markers, or None if no locations found
        """
        try:
            import re
            import json
            
            markers = []
            locations = []
            
            # SPECIAL HANDLING FOR ROUTING QUERIES
            # Check if this is a routing query (how to get to X, directions to X, etc.)
            query_lower = query.lower()
            is_routing_query = any([
                'how' in query_lower and any(w in query_lower for w in ['get', 'go', 'reach']),
                'direction' in query_lower,
                'way to' in query_lower,
                'route to' in query_lower,
                'from' in query_lower and 'to' in query_lower,  # Location-to-location routing
                signals.get('needs_gps_routing'),
                signals.get('needs_directions'),
            ])
            
            # Try to extract destination (and origin if specified) from query using Istanbul Knowledge
            if is_routing_query:
                try:
                    from .istanbul_knowledge import IstanbulKnowledge
                    istanbul_kb = IstanbulKnowledge()
                    
                    # Check if this is a "from X to Y" query (location-to-location)
                    origin_coord = None
                    origin_name = None
                    destination_coord = None
                    destination_name = None
                    
                    # Pattern: "from X to Y"
                    if 'from' in query_lower and 'to' in query_lower:
                        # Extract origin and destination
                        # Check neighborhoods for both origin and destination
                        for name, neighborhood in istanbul_kb.neighborhoods.items():
                            if name.lower() in query_lower:
                                if neighborhood.center_location:
                                    # Heuristic: if it appears before "to", it's origin; after "to", it's destination
                                    name_pos = query_lower.find(name.lower())
                                    to_pos = query_lower.find(' to ')
                                    
                                    if name_pos < to_pos and not origin_coord:
                                        origin_coord = neighborhood.center_location
                                        origin_name = name
                                        logger.info(f"🎯 Found origin in neighborhoods: {name} at {origin_coord}")
                                    elif name_pos > to_pos and not destination_coord:
                                        destination_coord = neighborhood.center_location
                                        destination_name = name
                                        logger.info(f"🎯 Found destination in neighborhoods: {name} at {destination_coord}")
                        
                        # Check landmarks for origin/destination
                        if not origin_coord or not destination_coord:
                            for name, landmark in istanbul_kb.landmarks.items():
                                if name.lower() in query_lower or any(syn.lower() in query_lower for syn in landmark.synonyms):
                                    name_pos = query_lower.find(name.lower())
                                    to_pos = query_lower.find(' to ')
                                    
                                    if name_pos < to_pos and not origin_coord:
                                        origin_coord = landmark.location
                                        origin_name = name
                                        logger.info(f"🎯 Found origin in landmarks: {name} at {origin_coord}")
                                    elif name_pos > to_pos and not destination_coord:
                                        destination_coord = landmark.location
                                        destination_name = name
                                        logger.info(f"🎯 Found destination in landmarks: {name} at {destination_coord}")
                        
                        # If we found both origin and destination, use Transportation RAG to find actual route
                        if origin_coord and destination_coord:
                            logger.info(f"🗺️ Found both locations, calling Transportation RAG for route: {origin_name} → {destination_name}")
                            
                            # Try to get actual route from Transportation RAG system
                            try:
                                from services.transportation_rag_system import get_transportation_rag
                                transport_rag = get_transportation_rag()
                                
                                # Find route using the comprehensive transportation database
                                route_result = transport_rag.find_route(origin_name, destination_name)
                                
                                if route_result:
                                    logger.info(f"✅ Transportation RAG found route with {len(route_result.steps)} steps")
                                    
                                    # Convert route steps to waypoints (coordinates for polyline)
                                    coordinates = []
                                    route_markers = []
                                    
                                    # Add origin marker
                                    route_markers.append({
                                        "position": {"lat": origin_coord[0], "lng": origin_coord[1]},
                                        "label": origin_name,
                                        "type": "origin"
                                    })
                                    coordinates.append([origin_coord[0], origin_coord[1]])
                                    
                                    # Add waypoints from route steps
                                    for step in route_result.steps:
                                        if hasattr(step, 'from_station') and step.from_station:
                                            # Add station coordinate
                                            if hasattr(step.from_station, 'lat') and hasattr(step.from_station, 'lng'):
                                                coordinates.append([step.from_station.lat, step.from_station.lng])
                                                
                                                # Add transit line marker
                                                if step.mode in ['metro', 'tram', 'ferry', 'marmaray', 'funicular']:
                                                    route_markers.append({
                                                        "position": {"lat": step.from_station.lat, "lng": step.from_station.lng},
                                                        "label": f"{step.line} - {step.from_station.name}",
                                                        "type": "transit"
                                                    })
                                        
                                        if hasattr(step, 'to_station') and step.to_station:
                                            if hasattr(step.to_station, 'lat') and hasattr(step.to_station, 'lng'):
                                                coordinates.append([step.to_station.lat, step.to_station.lng])
                                    
                                    # Add destination marker
                                    route_markers.append({
                                        "position": {"lat": destination_coord[0], "lng": destination_coord[1]},
                                        "label": destination_name,
                                        "type": "destination"
                                    })
                                    coordinates.append([destination_coord[0], destination_coord[1]])
                                    
                                    # Calculate center point
                                    center_lat = (origin_coord[0] + destination_coord[0]) / 2
                                    center_lon = (origin_coord[1] + destination_coord[1]) / 2
                                    
                                    map_data = {
                                        "type": "route",  # Full route with waypoints
                                        "coordinates": coordinates,  # For polyline
                                        "markers": route_markers,
                                        "center": {"lat": center_lat, "lng": center_lon},
                                        "zoom": 12,
                                        "has_origin": True,
                                        "has_destination": True,
                                        "origin_name": origin_name,
                                        "destination_name": destination_name,
                                        "route_data": {
                                            "distance_km": route_result.total_distance_km,
                                            "duration_min": route_result.total_time_minutes,
                                            "transport_mode": "Public Transit",
                                            "lines": [step.line for step in route_result.steps if hasattr(step, 'line')]
                                        }
                                    }
                                    
                                    logger.info(f"✅ Generated full route map with {len(coordinates)} waypoints")
                                    return map_data
                                else:
                                    logger.warning(f"⚠️ Transportation RAG found no route, using marker-only map")
                            except Exception as e:
                                logger.warning(f"⚠️ Transportation RAG failed: {e}, using marker-only map")
                            
                            # Fallback: marker-only map if route finding fails
                            markers.append({
                                "position": {"lat": origin_coord[0], "lng": origin_coord[1]},
                                "label": origin_name,
                                "type": "origin"
                            })
                            
                            markers.append({
                                "position": {"lat": destination_coord[0], "lng": destination_coord[1]},
                                "label": destination_name,
                                "type": "destination"
                            })
                            
                            # Calculate center point between origin and destination
                            center_lat = (origin_coord[0] + destination_coord[0]) / 2
                            center_lon = (origin_coord[1] + destination_coord[1]) / 2
                            
                            map_data = {
                                "type": "marker",  # Marker-only fallback
                                "markers": markers,
                                "center": {"lat": center_lat, "lng": center_lon},
                                "zoom": 12,
                                "has_origin": True,
                                "has_destination": True,
                                "origin_name": origin_name,
                                "destination_name": destination_name
                            }
                            
                            logger.info(f"✅ Generated location-to-location marker map: {origin_name} → {destination_name}")
                            return map_data
                    
                    # Single destination query with GPS - USE TRANSPORTATION RAG FOR ROUTING
                    if user_location and user_location.get('lat') and user_location.get('lon'):
                        # Check neighborhoods
                        destination_coord = None
                        destination_name = None
                        
                        for name, neighborhood in istanbul_kb.neighborhoods.items():
                            if name.lower() in query_lower:
                                if neighborhood.center_location:
                                    destination_coord = neighborhood.center_location
                                    destination_name = name
                                    logger.info(f"🎯 Found destination in neighborhoods: {name} at {destination_coord}")
                                    break
                        
                        # Check landmarks if no neighborhood found
                        if not destination_coord:
                            for name, landmark in istanbul_kb.landmarks.items():
                                if name.lower() in query_lower or any(syn.lower() in query_lower for syn in landmark.synonyms):
                                    destination_coord = landmark.location
                                    destination_name = name
                                    logger.info(f"🎯 Found destination in landmarks: {name} at {destination_coord}")
                                    break
                        
                        # If we found a destination, try Transportation RAG for actual route
                        if destination_coord:
                            logger.info(f"🗺️ GPS + destination found, attempting Transportation RAG routing to {destination_name}")
                            
                            # Find nearest known location to user's GPS for routing
                            try:
                                from services.transportation_rag_system import get_transportation_rag
                                transport_rag = get_transportation_rag()
                                
                                # Find nearest neighborhood to user location
                                def calc_distance(loc1, loc2):
                                    import math
                                    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
                                
                                user_coords = (user_location['lat'], user_location['lon'])
                                nearest_origin = None
                                nearest_distance = float('inf')
                                
                                for name, neighborhood in istanbul_kb.neighborhoods.items():
                                    if neighborhood.center_location:
                                        dist = calc_distance(user_coords, neighborhood.center_location)
                                        if dist < nearest_distance:
                                            nearest_distance = dist
                                            nearest_origin = name
                                
                                if nearest_origin:
                                    logger.info(f"🎯 Nearest location to GPS: {nearest_origin}")
                                    
                                    # Find route from nearest location to destination
                                    route_result = transport_rag.find_route(nearest_origin, destination_name)
                                    
                                    if route_result:
                                        logger.info(f"✅ Transportation RAG found route from {nearest_origin} to {destination_name}")
                                        
                                        # Build coordinates and markers
                                        coordinates = []
                                        route_markers = []
                                        
                                        # Add user GPS marker
                                        route_markers.append({
                                            "position": {"lat": user_location['lat'], "lng": user_location['lon']},
                                            "label": "Your Location",
                                            "type": "user"
                                        })
                                        coordinates.append([user_location['lat'], user_location['lon']])
                                        
                                        # Add route waypoints
                                        for step in route_result.steps:
                                            if hasattr(step, 'from_station') and step.from_station:
                                                if hasattr(step.from_station, 'lat') and hasattr(step.from_station, 'lng'):
                                                    coordinates.append([step.from_station.lat, step.from_station.lng])
                                                    
                                                    if step.mode in ['metro', 'tram', 'ferry', 'marmaray', 'funicular']:
                                                        route_markers.append({
                                                            "position": {"lat": step.from_station.lat, "lng": step.from_station.lng},
                                                            "label": f"{step.line} - {step.from_station.name}",
                                                            "type": "transit"
                                                        })
                                            
                                            if hasattr(step, 'to_station') and step.to_station:
                                                if hasattr(step.to_station, 'lat') and hasattr(step.to_station, 'lng'):
                                                    coordinates.append([step.to_station.lat, step.to_station.lng])
                                        
                                        # Add destination marker
                                        route_markers.append({
                                            "position": {"lat": destination_coord[0], "lng": destination_coord[1]},
                                            "label": destination_name,
                                            "type": "destination"
                                        })
                                        coordinates.append([destination_coord[0], destination_coord[1]])
                                        
                                        map_data = {
                                            "type": "route",
                                            "coordinates": coordinates,
                                            "markers": route_markers,
                                            "center": {"lat": (user_location['lat'] + destination_coord[0]) / 2, "lng": (user_location['lon'] + destination_coord[1]) / 2},
                                            "zoom": 12,
                                            "has_origin": True,
                                            "has_destination": True,
                                            "origin_name": "Your Location",
                                            "destination_name": destination_name,
                                            "route_data": {
                                                "distance_km": route_result.total_distance_km,
                                                "duration_min": route_result.total_time_minutes,
                                                "transport_mode": "Public Transit",
                                                "lines": [step.line for step in route_result.steps if hasattr(step, 'line')]
                                            }
                                        }
                                        
                                        logger.info(f"✅ Generated GPS-based route map with {len(coordinates)} waypoints")
                                        return map_data
                                else:
                                    logger.warning(f"⚠️ Could not find nearest origin for GPS routing")
                            except Exception as e:
                                logger.warning(f"⚠️ Transportation RAG GPS routing failed: {e}")
                            
                            # Fallback: create marker-only map if routing failed
                            markers.append({
                                "position": {"lat": user_location['lat'], "lng": user_location['lon']},
                                "label": "Your Location",
                                "type": "user"
                            })
                            
                            markers.append({
                                "position": {"lat": destination_coord[0], "lng": destination_coord[1]},
                                "label": destination_name,
                                "type": "destination"
                            })
                            
                            # Calculate center point between origin and destination
                            center_lat = (user_location['lat'] + destination_coord[0]) / 2
                            center_lon = (user_location['lon'] + destination_coord[1]) / 2
                            
                            map_data = {
                                "type": "route",
                                "markers": markers,
                                "center": {"lat": center_lat, "lng": center_lon},
                                "zoom": 12,
                                "has_origin": True,
                                "has_destination": True,
                                "origin_name": "Your Location",
                                "destination_name": destination_name,
                                "route": {
                                    "origin": {"lat": user_location['lat'], "lng": user_location['lon']},
                                    "destination": {"lat": destination_coord[0], "lng": destination_coord[1]}
                                }
                            }
                            
                            logger.info(f"✅ Generated routing map from {user_location['lat']:.4f},{user_location['lon']:.4f} to {destination_name}")
                            return map_data
                
                except Exception as e:
                    logger.error(f"Failed to lookup destination in Istanbul Knowledge: {e}")
            
            # ORIGINAL LOGIC: Try to extract locations from database context string
            db_context = context.get('database', '')
            
            # Pattern to find coordinates in the database context
            # Looking for patterns like: "Coordinates: (41.012, 28.978)" or "lat: 41.012, lon: 28.978"
            coord_patterns = [
                r'Coordinates:\s*\(([0-9.]+),\s*([0-9.]+)\)',
                r'lat(?:itude)?:\s*([0-9.]+)[,\s]+lon(?:gitude)?:\s*([0-9.]+)',
                r'\(([0-9.]+),\s*([0-9.]+)\)',  # Simple tuple format
            ]
            
            # Pattern to find location names (before coordinates)
            name_pattern = r'([A-ZÇĞİÖŞÜ][A-Za-zçğıöşü\s&\'-]+?)(?:\s*[-:]\s*|\s+Coordinates:)'
            
            # Extract all coordinate pairs from database context if available
            if db_context:
                for pattern in coord_patterns:
                    for match in re.finditer(pattern, db_context):
                        try:
                            lat = float(match.group(1))
                            lon = float(match.group(2))
                            
                            # Find the name before this coordinate
                            name = "Location"
                            text_before = db_context[:match.start()]
                            name_match = re.search(name_pattern, text_before[-200:])  # Look in last 200 chars
                            if name_match:
                                name = name_match.group(1).strip()
                            
                            locations.append({
                                'name': name,
                                'lat': lat,
                                'lon': lon
                            })
                        except (ValueError, IndexError):
                            continue
            
            # If we found locations, create markers
            if locations:
                for loc in locations:
                    markers.append({
                        "position": {"lat": loc['lat'], "lng": loc['lon']},
                        "label": loc['name'],
                        "type": "restaurant" if signals.get('needs_restaurant') else "attraction"
                    })
                
                # Calculate center point
                avg_lat = sum(loc['lat'] for loc in locations) / len(locations)
                avg_lon = sum(loc['lon'] for loc in locations) / len(locations)
                
                # Add user location marker if available
                if user_location and isinstance(user_location, dict) and 'lat' in user_location and 'lon' in user_location:
                    markers.append({
                        "position": {"lat": user_location['lat'], "lng": user_location['lon']},
                        "label": "Your Location",
                        "type": "user"
                    })
                    # Recalculate center to include user location
                    avg_lat = (avg_lat * len(locations) + user_location['lat']) / (len(locations) + 1)
                    avg_lon = (avg_lon * len(locations) + user_location['lon']) / (len(locations) + 1)
                
                map_data = {
                    "type": "markers",
                    "markers": markers,
                    "center": {"lat": avg_lat, "lng": avg_lon},
                    "zoom": 13,
                    "has_origin": user_location is not None,
                    "has_destination": False,
                    "origin_name": "Your Location" if user_location else None,
                    "destination_name": None,
                    "locations_count": len(locations)
                }
                
                logger.info(f"✅ Generated map_data with {len(markers)} markers from context")
                return map_data
            
            # No database locations found, but for "nearby" queries with GPS,
            # still generate map centered on user location
            # Check query for "nearby", "near me", "close to me", etc.
            query_lower = query.lower()
            is_nearby_query = any([
                'nearby' in query_lower,
                'near me' in query_lower,
                'close to me' in query_lower,
                'around me' in query_lower,
                'around here' in query_lower,
                signals.get('needs_restaurant'),
                signals.get('needs_attraction'),
                signals.get('needs_hidden_gems')
            ])
            
            # Check if user_location has valid coordinates
            has_valid_location = (user_location and 
                                isinstance(user_location, dict) and 
                                'lat' in user_location and 
                                'lon' in user_location and
                                user_location['lat'] is not None and
                                user_location['lon'] is not None)
            
            if has_valid_location and is_nearby_query:
                # Create map centered on user location
                markers.append({
                    "position": {"lat": user_location['lat'], "lng": user_location['lon']},
                    "label": "Your Location",
                    "type": "user"
                })
                
                map_data = {
                    "type": "user_centered",
                    "markers": markers,
                    "center": {"lat": user_location['lat'], "lng": user_location['lon']},
                    "zoom": 14,
                    "has_origin": True,
                    "has_destination": False,
                    "origin_name": "Your Location",
                    "destination_name": None,
                    "locations_count": 0,
                    "note": "Map centered on your location - results may be shown in text"
                }
                
                logger.info(f"✅ Generated GPS-centered map_data for 'nearby' query (no DB locations)")
                return map_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate map from context: {e}")
            return None

