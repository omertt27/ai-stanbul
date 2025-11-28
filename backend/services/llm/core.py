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
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, AsyncGenerator
from collections import defaultdict

# Import extracted modules
from .signals import SignalDetector
from .context import ContextBuilder
from .prompts import PromptBuilder
from .analytics import AnalyticsManager
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
        self.circuit_breakers = {
            'llm': CircuitBreaker(
                name='LLM Service',
                failure_threshold=self.config.get('llm_failure_threshold', 5),
                success_threshold=2,
                timeout=60.0  # 1 minute before retry
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
        
        # STEP 5: Context Building
        context = await self.context_builder.build_context(
            query=query,
            signals=signals['signals'],
            user_location=user_location,
            language=language
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
        prompt = self.prompt_builder.build_prompt(
            query=query,
            signals=signals['signals'],
            context=context,
            conversation_context=conversation_context,
            language=language
        )
        
        logger.info(f"📝 Prompt built: {len(prompt)} chars")
        
        # STEP 7: LLM Generation with Resilience
        try:
            llm_start = time.time()
            
            # Simplified LLM call with circuit breaker only
            async def _generate_with_llm():
                return await self.llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
            
            # Apply circuit breaker protection
            response_data = await self.circuit_breakers['llm'].call(_generate_with_llm)
            
            llm_latency = time.time() - llm_start
            
            if not response_data or "generated_text" not in response_data:
                raise Exception("Invalid LLM response")
            
            response_text = response_data["generated_text"]
            
            logger.info(f"✅ LLM generated response in {llm_latency:.2f}s")
            
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
                    context=context
                )
        
        except CircuitBreakerError as e:
            logger.error(f"❌ LLM service unavailable (circuit breaker open): {e}")
            self.analytics.track_llm_failure('circuit_breaker_open')
            
            # Graceful degradation - provide informative error
            response_text = GracefulDegradation.create_degraded_response(
                original_query=query,
                available_data=context,
                unavailable_services=['LLM']
            )['metadata']['notice']
            
            result = {
                'response': response_text,
                'map_data': None,
                'signals': signals['signals'],
                'metadata': {
                    'error': 'llm_unavailable',
                    'degraded_mode': True,
                    'cached': False
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            self.analytics.track_error('llm_failure', str(e))
            
            # Fallback to context-based response
            response_text = await self._fallback_response(
                query=query,
                context=context
            )
            llm_latency = 0
        
        # STEP 8: Build Result
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
        context: Dict[str, Any]
    ) -> str:
        """
        Generate fallback response when LLM fails.
        
        Args:
            query: User query
            context: Built context
            
        Returns:
            Fallback response string
        """
        # Try to use RAG context if available
        if context.get('rag'):
            return (
                f"Based on available information:\n\n"
                f"{context['rag'][:500]}\n\n"
                f"Note: This is a simplified response. Please try rephrasing your question."
            )
        
        # Try to use database context
        if context.get('database'):
            return (
                f"Here's what I found:\n\n"
                f"{context['database'][:500]}\n\n"
                f"Note: This is a simplified response. Please try rephrasing your question."
            )
        
        # Ultimate fallback
        return (
            "I apologize, but I'm having trouble generating a response right now. "
            "Please try rephrasing your question or contact support if the issue persists."
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
            if timeout_metrics['timeout_rate'] > 10:  # >10% timeout rate
                timeout_health = 'degraded'
                all_healthy = False
            elif timeout_metrics['timeout_rate'] > 25:  # >25% timeout rate
                timeout_health = 'unhealthy'
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

