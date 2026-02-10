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
import os
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


# ============================================================
# Entity Validation for Hallucination Detection
# ============================================================

class EntityValidator:
    """
    Validates entities mentioned in LLM responses against known data.
    
    Used to detect and flag potential hallucinations for:
    - Restaurant names
    - Attraction names
    - Location/neighborhood names
    - Transit line numbers
    
    IMPORTANT: For transportation queries, entity validation is DISABLED
    to prevent false positives on valid transport entities like "Marmaray", "Sirkeci", etc.
    
    Author: AI Istanbul Team
    Date: December 2025
    """
    
    # Known Istanbul landmarks (always valid)
    KNOWN_LANDMARKS = {
        'hagia sophia', 'ayasofya', 'blue mosque', 'sultanahmet mosque',
        'topkapi palace', 'topkapı sarayı', 'dolmabahce palace', 'dolmabahçe',
        'grand bazaar', 'kapalıçarşı', 'spice bazaar', 'mısır çarşısı',
        'galata tower', 'galata kulesi', 'maiden tower', 'kız kulesi',
        'basilica cistern', 'yerebatan sarnıcı', 'suleymaniye mosque',
        'chora church', 'kariye museum', 'istiklal street', 'taksim square',
        'bosphorus', 'boğaz', 'ortakoy mosque', 'rumeli fortress',
        'yildiz park', 'gulhane park', 'princes islands', 'pierre loti',
        'balat', 'fener', 'eyup sultan mosque', 'miniaturk', 'rahmi koc museum'
    }
    
    # Valid transport entities - NOT hallucinations (Week 3 Fix)
    VALID_TRANSPORT_ENTITIES = {
        # Metro lines
        'm1', 'm1a', 'm1b', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm9', 'm11',
        # Tram lines  
        't1', 't4', 't5',
        # Other transit
        'marmaray', 'f1', 'f2', 'metrobus', 'ferry', 'ido', 'turyol', 'şehir hatları',
        # Common stations that might be flagged
        'sirkeci', 'yenikapı', 'yenikapi', 'ayrılık çeşmesi', 'ayrilik cesmesi',
        'kadıköy', 'kadikoy', 'üsküdar', 'uskudar', 'taksim', 'levent',
        'mecidiyeköy', 'mecidiyekoy', 'şişli', 'sisli', 'osmanbey',
        'kabataş', 'kabatas', 'karaköy', 'karakoy', 'eminönü', 'eminonu',
        'sultanahmet', 'beyazıt', 'beyazit', 'aksaray', 'zeytinburnu',
        'bağcılar', 'bagcilar', 'kirazlı', 'kirazli', 'otogar',
        'bakırköy', 'bakirkoy', 'yeşilköy', 'yesilkoy', 'florya',
        'pendik', 'kartal', 'maltepe', 'bostancı', 'bostanci',
        'beşiktaş', 'besiktas', 'ortaköy', 'ortakoy', 'bebek',
        'hacıosman', 'hacimosman', 'maslak', 'gayrettepe',
        '4. levent', '4 levent', 'levent', 'zincirlikuyu',
        # Islands (ferry destinations)
        'büyükada', 'buyukada', 'heybeliada', 'burgazada', 'kınalıada', 'kinaliada',
        'adalar', 'princes islands', 'sedef adası',
        # Airport stations
        'istanbul havalimanı', 'istanbul havalimani', 'atatürk havalimanı',
        'sabiha gökçen', 'sabiha gokcen',
    }
    
    # Known neighborhoods/districts
    KNOWN_LOCATIONS = {
        'sultanahmet', 'taksim', 'kadikoy', 'kadıköy', 'besiktas', 'beşiktaş',
        'uskudar', 'üsküdar', 'ortakoy', 'ortaköy', 'bebek', 'nisantasi', 'nişantaşı',
        'karakoy', 'karaköy', 'galata', 'cihangir', 'beyoglu', 'beyoğlu',
        'fatih', 'eminonu', 'eminönü', 'sariyer', 'sarıyer', 'bakirkoy', 'bakırköy',
        'balat', 'fener', 'moda', 'bahariye', 'bagdat caddesi', 'bağdat caddesi',
        'levent', 'mecidiyekoy', 'mecidiyeköy', 'sisli', 'şişli', 'etiler',
        'arnavutkoy', 'arnavutköy', 'kuzguncuk', 'cengelkoy', 'çengelköy',
        'atasehir', 'ataşehir', 'umraniye', 'ümraniye', 'maltepe', 'kartal', 'pendik'
    }
    
    # Known transit lines
    KNOWN_TRANSIT = {
        'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm9', 'm11',
        't1', 't4', 't5', 'marmaray', 'f1', 'f2', 'metrobus',
        'havaist', 'iett', 'ido', 'turyol', 'sehir hatlari'
    }
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self._known_restaurants = set()
        self._known_attractions = set(self.KNOWN_LANDMARKS)
        self._known_locations = set(self.KNOWN_LOCATIONS)
        self._loaded_from_db = False
    
    async def load_from_database(self):
        """Load known entities from database."""
        if self._loaded_from_db or not self.db:
            return
        
        try:
            from sqlalchemy import text
            
            # Load restaurants
            result = await self.db.execute(text("SELECT name FROM restaurants"))
            rows = await result.fetchall()
            self._known_restaurants.update(row[0].lower() for row in rows if row[0])
            
            # Load places/attractions
            result = await self.db.execute(text("SELECT name FROM places"))
            rows = await result.fetchall()
            self._known_attractions.update(row[0].lower() for row in rows if row[0])
            
            self._loaded_from_db = True
            logger.info(f"✅ EntityValidator loaded {len(self._known_restaurants)} restaurants, "
                       f"{len(self._known_attractions)} attractions from DB")
            
        except Exception as e:
            logger.warning(f"Failed to load entities from database: {e}")
    
    def validate_location(self, name: str) -> bool:
        """Check if location name is known."""
        name_lower = name.lower().strip()
        
        # Direct match
        if name_lower in self._known_locations:
            return True
        
        # Partial match (for compound names)
        for known in self._known_locations:
            if known in name_lower or name_lower in known:
                return True
        
        return False
    
    def validate_attraction(self, name: str) -> bool:
        """Check if attraction name is known."""
        name_lower = name.lower().strip()
        
        # Direct match
        if name_lower in self._known_attractions:
            return True
        
        # Partial match
        for known in self._known_attractions:
            if known in name_lower or name_lower in known:
                return True
        
        return False
    
    def validate_transit_line(self, line: str) -> bool:
        """Check if transit line is valid."""
        line_lower = line.lower().strip()
        
        # Check direct match
        if line_lower in self.KNOWN_TRANSIT:
            return True
        
        # Check pattern (M1-M11, T1-T5)
        if re.match(r'^m([1-9]|1[0-1])$', line_lower):
            return True
        if re.match(r'^t[1-5]$', line_lower):
            return True
        if re.match(r'^f[1-2]$', line_lower):
            return True
        
        return False
    
    def _is_valid_transport_entity(self, entity: str) -> bool:
        """Check if entity is a valid transport-related term."""
        entity_lower = entity.lower().strip()
        
        # Direct match
        if entity_lower in self.VALID_TRANSPORT_ENTITIES:
            return True
        
        # Check for partial matches
        for valid_entity in self.VALID_TRANSPORT_ENTITIES:
            if valid_entity in entity_lower or entity_lower in valid_entity:
                return True
        
        # Check metro/tram line patterns
        if re.match(r'^m\d{1,2}$', entity_lower):
            return True
        if re.match(r'^t\d$', entity_lower):
            return True
        if re.match(r'^f\d$', entity_lower):
            return True
            
        return False
    
    def find_potentially_hallucinated(self, response: str, is_transport_query: bool = False) -> Dict[str, List[str]]:
        """
        Analyze response for potentially hallucinated entities.
        
        Args:
            response: The LLM-generated response text
            is_transport_query: If True, skip entity validation for transport entities
            
        Returns:
            Dict with lists of potentially fake entities by type
        """
        issues = {
            'restaurants': [],
            'attractions': [],
            'locations': [],
            'transit': []
        }
        
        # WEEK 3 FIX: For transport queries, skip entity validation entirely
        # Transport entities like "Marmaray", "Sirkeci" are NOT hallucinations
        if is_transport_query:
            logger.debug("Skipping entity validation for transportation query")
            return issues
        
        # Extract quoted names (common for entity mentions)
        quoted = re.findall(r'"([^"]+)"', response)
        quoted += re.findall(r"'([^']+)'", response)
        
        # Extract bold names (markdown)
        bold = re.findall(r'\*\*([^*]+)\*\*', response)
        
        # Check for fake metro lines (M15, M20, etc.)
        fake_lines = re.findall(r'\b[MT](\d{2,})\b', response)
        for num in fake_lines:
            if int(num) > 11:  # No metro line above M11
                issues['transit'].append(f"M{num}")
        
        # Check for suspicious line references
        all_line_refs = re.findall(r'\b([MT]\d+)\b', response, re.IGNORECASE)
        for line in all_line_refs:
            if not self.validate_transit_line(line):
                issues['transit'].append(line)
        
        # Check quoted/bold entities
        all_mentioned = set(quoted + bold)
        for entity in all_mentioned:
            if len(entity) < 3:
                continue
            
            # WEEK 3 FIX: Skip valid transport entities
            if self._is_valid_transport_entity(entity):
                continue
            
            # Skip if it's a known entity
            if self.validate_attraction(entity) or self.validate_location(entity):
                continue
            
            # Check if looks like a proper name that might be hallucinated
            # Pattern: Title Case words that aren't common English
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$', entity):
                # Not a known place - could be hallucinated
                # Not a known place - could be hallucinated
                if len(entity.split()) <= 3:  # Short names are more suspicious
                    issues['attractions'].append(entity)
        
        return issues
    
    def get_validation_score(self, response: str, is_transport_query: bool = False) -> Tuple[float, Dict[str, List[str]]]:
        """
        Calculate validation score and return issues.
        
        Args:
            response: The LLM-generated response text
            is_transport_query: If True, skip entity validation for transport entities
        
        Returns:
            Tuple of (score 0.0-1.0, issues dict)
        """
        issues = self.find_potentially_hallucinated(response, is_transport_query=is_transport_query)
        total_issues = sum(len(v) for v in issues.values())
        
        if total_issues == 0:
            return 1.0, issues
        elif total_issues <= 1:
            return 0.85, issues
        elif total_issues <= 3:
            return 0.6, issues
        elif total_issues <= 5:
            return 0.4, issues
        else:
            return 0.2, issues


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
                        OR None to auto-initialize with UnifiedLLMService
            db_connection: Database connection
            config: Configuration dictionary
            services: Service Manager instance with all local services
        """
        # Check if we should use UnifiedLLMService (Week 2 Integration)
        use_unified = os.getenv('USE_UNIFIED_LLM', 'true').lower() == 'true'
        
        if use_unified and llm_client is None:
            # Auto-initialize with UnifiedLLMService
            try:
                import sys
                from pathlib import Path
                unified_path = Path(__file__).parent.parent.parent.parent / "unified_system"
                if str(unified_path) not in sys.path:
                    sys.path.insert(0, str(unified_path))
                
                from services.unified_llm_service import get_unified_llm
                self.llm = get_unified_llm()
                self.using_unified_llm = True
                logger.info("✅ Using UnifiedLLMService (with caching, metrics, circuit breaker)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize UnifiedLLMService: {e}")
                logger.warning("   Falling back to provided llm_client")
                self.llm = llm_client
                self.using_unified_llm = False
        elif llm_client is not None:
            # Use provided LLM client
            self.llm = llm_client
            self.using_unified_llm = False
            if use_unified:
                logger.info("ℹ️  USE_UNIFIED_LLM=true but llm_client provided, using provided client")
        else:
            # No client provided and unified disabled
            raise ValueError("Either provide llm_client or set USE_UNIFIED_LLM=true")
        
        self.llm_client = self.llm  # Alias for compatibility
        self.db = db_connection
        self.config = config or {}
        self.services = services  # Service Manager for local services
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        logger.info("🚀 Pure LLM Core initialized successfully")
        logger.info(f"   🤖 LLM Service: {'UnifiedLLMService' if self.using_unified_llm else 'Legacy RunPodClient'}")
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
        
        # 🚀 Redis Client for LLM Response Caching (PERFORMANCE OPTIMIZATION)
        self.redis_client = None
        try:
            if self.config.get('redis_client'):
                self.redis_client = self.config['redis_client']
                logger.info("✅ Redis client provided via config - LLM caching enabled")
            elif self.services and hasattr(self.services, 'redis_manager') and self.services.redis_manager:
                self.redis_client = self.services.redis_manager.client
                logger.info("✅ Redis client from service manager - LLM caching enabled")
            else:
                # Try to import and get Redis client from global cache
                try:
                    from services.cache_service import get_redis_client
                    self.redis_client = get_redis_client()
                    logger.info("✅ Redis client from global cache service - LLM caching enabled")
                except ImportError:
                    logger.warning("⚠️  No Redis client available - LLM caching disabled")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Redis client: {e} - LLM caching disabled")
        
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
        
        # 11. Entity Validator (hallucination detection) - NEW
        self.entity_validator = EntityValidator(db_connection=self.db)
        
        logger.info("✅ All subsystems initialized (including Phase 2 features)")
    
    # ========================================================================
    # LLM Call Compatibility Wrapper (Week 2 Integration)
    # ========================================================================
    
    async def _llm_call(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        component: str = 'core'
    ) -> Dict[str, Any]:
        """
        Unified LLM call interface - Works with both UnifiedLLMService and legacy clients
        
        This wrapper provides backward compatibility while enabling UnifiedLLMService features:
        - Automatic caching (UnifiedLLMService only)
        - Circuit breaker (UnifiedLLMService only)
        - Enhanced metrics (UnifiedLLMService only)
        - Consistent interface for both services
        
        Args:
            prompt: The prompt to send to LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            component: Component name for metrics tracking
            
        Returns:
            {
                'generated_text': str,      # Main response text
                'finish_reason': str,       # stop, length, or error
                'usage': dict,              # Token usage stats
                'cached': bool,             # True if from cache (UnifiedLLM only)
                'latency_ms': float         # Latency in milliseconds
            }
        """
        if self.using_unified_llm:
            # Use UnifiedLLMService (with caching, metrics, circuit breaker)
            result = await self.llm.complete(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                component=component
            )
            
            # Convert UnifiedLLMService format to legacy format
            return {
                'generated_text': result['text'],
                'finish_reason': result['finish_reason'],
                'usage': result.get('usage', {}),
                'cached': result.get('cached', False),
                'latency_ms': result.get('latency_ms', 0)
            }
        else:
            # Use legacy RunPodLLMClient
            import time
            start = time.time()
            
            result = await self.llm.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            latency_ms = (time.time() - start) * 1000
            
            # Ensure consistent format
            if result:
                result['cached'] = False
                result['latency_ms'] = latency_ms
            
            return result
    
    async def _llm_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        component: str = 'core'
    ):
        """
        Unified LLM streaming interface
        
        Yields:
            Text chunks as they arrive
        """
        if self.using_unified_llm:
            # Use UnifiedLLMService streaming
            async for chunk in self.llm.stream_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                component=component
            ):
                yield chunk
        else:
            # Use legacy streaming if available
            if hasattr(self.llm, 'generate_stream'):
                async for chunk in self.llm.generate_stream(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                ):
                    yield chunk
            else:
                # Fallback: non-streaming
                result = await self._llm_call(prompt, max_tokens, temperature, component)
                if result and 'generated_text' in result:
                    yield result['generated_text']
    
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
            result = await self._llm_call(
                prompt=rewrite_prompt,
                max_tokens=30,  # Drastically reduced to prevent long outputs
                temperature=0.1,  # Very low temperature for consistency
                component='query_rewriter'
            )
            
            rewritten = result['generated_text'].strip()
            
            # Remove any quotes, newlines, or extra formatting
            rewritten = rewritten.strip('"\'').split('\n')[0].strip()
            # ADDITIONAL: Remove any stray quotes at end, beginning, or trailing punctuation issues
            rewritten = rewritten.strip('"\'').rstrip('"\'?').strip()
            if rewritten.endswith('"') or rewritten.endswith("'"):
                rewritten = rewritten[:-1].strip()
            # Fix double question marks or trailing quote-question combos
            rewritten = rewritten.replace('?"', '?').replace('"?', '?').replace("?'", '?')
            
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
    
    def _detect_ambiguous_query(self, query: str, signals: Dict[str, bool], language: str = "en") -> Dict[str, Any]:
        """
        Detect if a query is ambiguous and needs clarification.
        Supports 5 languages: English (en), Turkish (tr), Russian (ru), German (de), Arabic (ar)
        
        Ambiguity indicators:
        - Multiple conflicting signals detected
        - Very short queries without context
        - Generic queries without specifics
        - Queries with unclear intent
        
        Args:
            query: User query
            signals: Detected signals
            language: Response language for clarification questions
            
        Returns:
            {
                'is_ambiguous': bool,
                'ambiguity_type': str,
                'clarification_questions': List[str],
                'confidence': float
            }
        """
        query_lower = query.lower().strip()
        word_count = len(query.split())
        
        result = {
            'is_ambiguous': False,
            'ambiguity_type': None,
            'clarification_questions': [],
            'confidence': 1.0
        }
        
        # ========== GREETINGS & SMALL TALK BYPASS ==========
        # These should NEVER be marked as ambiguous - always let LLM handle them
        greeting_patterns = [
            # Turkish greetings
            'merhaba', 'selam', 'günaydın', 'iyi günler', 'iyi akşamlar', 
            'nasılsın', 'nasılsınız', 'naber', 'ne haber', 'hoşgeldin',
            'teşekkür', 'sağol', 'eyvallah', 'hoşçakal', 'görüşürüz',
            'iyiyim', 'iyi misin', 'sen nasılsın',
            # English greetings
            'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
            'how are you', "how's it going", 'what\'s up', 'howdy',
            'thanks', 'thank you', 'bye', 'goodbye', 'see you',
            'i\'m fine', 'i am fine', 'doing well', 'doing good',
            # Russian greetings
            'привет', 'здравствуй', 'доброе утро', 'добрый день', 'добрый вечер',
            'как дела', 'спасибо', 'пока', 'до свидания',
            # German greetings
            'hallo', 'guten tag', 'guten morgen', 'guten abend',
            'wie geht es', 'danke', 'tschüss', 'auf wiedersehen',
            # Arabic greetings
            'مرحبا', 'السلام عليكم', 'صباح الخير', 'مساء الخير',
            'كيف حالك', 'شكرا', 'مع السلامة'
        ]
        
        # Check if query is a greeting/small talk - if so, DO NOT mark as ambiguous
        if any(pattern in query_lower for pattern in greeting_patterns):
            logger.info(f"👋 Greeting/small talk detected: '{query}' - bypassing ambiguity check")
            result['confidence'] = 0.9  # High confidence - let LLM handle it
            result['is_ambiguous'] = False
            return result
        
        # Multilingual clarification questions
        clarification_questions = {
            'multi_intent_restaurant_transport': {
                'en': [
                    "Are you looking for restaurant recommendations, or do you need directions to a restaurant?",
                    "Would you like me to suggest restaurants nearby, or help you get to a specific place?"
                ],
                'tr': [
                    "Restoran önerisi mi arıyorsunuz yoksa bir restorana nasıl gideceğinizi mi öğrenmek istiyorsunuz?",
                    "Size yakındaki restoranları önermemi mi yoksa belirli bir yere ulaşmanıza yardım etmemi mi istersiniz?"
                ],
                'ru': [
                    "Вы ищете рекомендации по ресторанам или вам нужны указания, как добраться до ресторана?",
                    "Хотите, чтобы я предложил рестораны поблизости, или помочь вам добраться до определенного места?"
                ],
                'de': [
                    "Suchen Sie Restaurantempfehlungen oder brauchen Sie eine Wegbeschreibung zu einem Restaurant?",
                    "Möchten Sie Restaurants in der Nähe vorgeschlagen bekommen oder Hilfe, um zu einem bestimmten Ort zu gelangen?"
                ],
                'ar': [
                    "هل تبحث عن توصيات للمطاعم أم تحتاج إلى الاتجاهات للوصول إلى مطعم؟",
                    "هل تريد أن أقترح عليك مطاعم قريبة أو مساعدتك للوصول إلى مكان محدد؟"
                ]
            },
            'multi_intent_attraction_restaurant': {
                'en': [
                    "Are you looking for attractions to visit, or restaurants to eat at?",
                    "Would you like sightseeing recommendations or dining options?"
                ],
                'tr': [
                    "Ziyaret edilecek mekanlar mı arıyorsunuz yoksa yemek yiyecek restoranlar mı?",
                    "Gezi önerileri mi yoksa yemek seçenekleri mi istersiniz?"
                ],
                'ru': [
                    "Вы ищете достопримечательности или рестораны?",
                    "Хотите рекомендации по осмотру достопримечательностей или варианты питания?"
                ],
                'de': [
                    "Suchen Sie Sehenswürdigkeiten zum Besichtigen oder Restaurants zum Essen?",
                    "Möchten Sie Empfehlungen für Besichtigungen oder Essensoptionen?"
                ],
                'ar': [
                    "هل تبحث عن معالم سياحية لزيارتها أو مطاعم للأكل؟",
                    "هل تريد توصيات لمشاهدة المعالم أو خيارات لتناول الطعام؟"
                ]
            },
            'too_short': {
                'en': [
                    "Could you tell me more about what you're looking for?",
                    "I can help with restaurants, attractions, transportation, and more. What interests you?"
                ],
                'tr': [
                    "Ne aradığınız hakkında biraz daha bilgi verebilir misiniz?",
                    "Restoranlar, turistik yerler, ulaşım ve daha fazlası konusunda yardımcı olabilirim. Sizi ne ilgilendiriyor?"
                ],
                'ru': [
                    "Не могли бы вы рассказать больше о том, что вы ищете?",
                    "Я могу помочь с ресторанами, достопримечательностями, транспортом и многим другим. Что вас интересует?"
                ],
                'de': [
                    "Könnten Sie mir mehr darüber erzählen, wonach Sie suchen?",
                    "Ich kann bei Restaurants, Sehenswürdigkeiten, Transport und mehr helfen. Was interessiert Sie?"
                ],
                'ar': [
                    "هل يمكنك إخباري المزيد عما تبحث عنه؟",
                    "يمكنني المساعدة في المطاعم والمعالم السياحية والمواصلات والمزيد. ما الذي يهمك؟"
                ]
            },
            'generic_restaurant': {
                'en': [
                    "What type of cuisine are you in the mood for? (Turkish, seafood, kebab, etc.)",
                    "Any preferred area or neighborhood in Istanbul?"
                ],
                'tr': [
                    "Ne tür bir mutfak istiyorsunuz? (Türk, deniz ürünleri, kebap, vb.)",
                    "İstanbul'da tercih ettiğiniz bir bölge veya semt var mı?"
                ],
                'ru': [
                    "Какой тип кухни вы предпочитаете? (Турецкая, морепродукты, кебаб и т.д.)",
                    "Есть ли предпочтительный район в Стамбуле?"
                ],
                'de': [
                    "Welche Art von Küche bevorzugen Sie? (Türkisch, Meeresfrüchte, Kebab, usw.)",
                    "Haben Sie einen bevorzugten Bereich oder Stadtteil in Istanbul?"
                ],
                'ar': [
                    "ما نوع المطبخ الذي تفضله؟ (تركي، مأكولات بحرية، كباب، إلخ)",
                    "هل لديك منطقة مفضلة في إسطنبول؟"
                ]
            },
            'generic_sightseeing': {
                'en': [
                    "Are you interested in historical sites, museums, markets, or nature?",
                    "How much time do you have for sightseeing?"
                ],
                'tr': [
                    "Tarihi mekanlar, müzeler, pazarlar veya doğa ile mi ilgileniyorsunuz?",
                    "Gezi için ne kadar zamanınız var?"
                ],
                'ru': [
                    "Вас интересуют исторические места, музеи, рынки или природа?",
                    "Сколько времени у вас есть на осмотр достопримечательностей?"
                ],
                'de': [
                    "Interessieren Sie sich für historische Stätten, Museen, Märkte oder Natur?",
                    "Wie viel Zeit haben Sie für Besichtigungen?"
                ],
                'ar': [
                    "هل أنت مهتم بالمواقع التاريخية أم المتاحف أم الأسواق أم الطبيعة؟",
                    "كم من الوقت لديك للتجول؟"
                ]
            },
            'generic_default': {
                'en': [
                    "Could you be more specific about what you're looking for?",
                    "Are you interested in food, sightseeing, transportation, or something else?"
                ],
                'tr': [
                    "Ne aradığınız konusunda daha spesifik olabilir misiniz?",
                    "Yemek, gezi, ulaşım veya başka bir şeyle mi ilgileniyorsunuz?"
                ],
                'ru': [
                    "Не могли бы вы быть более конкретны в том, что вы ищете?",
                    "Вас интересует еда, достопримечательности, транспорт или что-то другое?"
                ],
                'de': [
                    "Könnten Sie genauer sagen, wonach Sie suchen?",
                    "Interessieren Sie sich für Essen, Besichtigungen, Transport или etwas anderes?"
                ],
                'ar': [
                    "هل يمكنك أن تكون أكثر تحديداً بشأن ما تبحث عنه؟",
                    "هل أنت مهتم بالطعام أو مشاهدة المعالم أو المواصلات أو شيء آخر؟"
                ]
            },
            'unclear_intent': {
                'en': [
                    "I'm not sure I understand. Could you rephrase your question?",
                    "I can help with restaurants, attractions, transportation, weather, and events in Istanbul."
                ],
                'tr': [
                    "Tam anlayamadım. Sorunuzu farklı şekilde sorabilir misiniz?",
                    "İstanbul'da restoranlar, turistik yerler, ulaşım, hava durumu ve etkinlikler konusunda yardımcı olabilirim."
                ],
                'ru': [
                    "Я не уверен, что понял. Не могли бы вы перефразировать свой вопрос?",
                    "Я могу помочь с ресторанами, достопримечательностями, транспортом, погодой и мероприятиями в Стамбуле."
                ],
                'de': [
                    "Ich bin mir nicht sicher, ob ich verstehe. Könnten Sie Ihre Frage umformulieren?",
                    "Ich kann bei Restaurants, Sehenswürdigkeiten, Transport, Wetter und Veranstaltungen in Istanbul helfen."
                ],
                'ar': [
                    "لست متأكداً من فهمي. هل يمكنك إعادة صياغة سؤالك؟",
                    "يمكنني المساعدة في المطاعم والمعالم السياحية والمواصلات والطقس والفعاليات في إسطنبول."
                ]
            }
        }
        
        def get_questions(key: str) -> list:
            """Helper to get language-specific questions"""
            return clarification_questions.get(key, {}).get(language, clarification_questions.get(key, {}).get('en', []))
        
        # Count active signals
        active_signals = [k for k, v in signals.items() if v]
        signal_count = len(active_signals)
        
        # Type 1: Multiple conflicting signals (e.g., restaurant + transportation)
        conflicting_pairs = [
            ('needs_restaurant', 'needs_transportation'),
            ('needs_attraction', 'needs_restaurant'),
            ('needs_weather', 'needs_transportation'),
        ]
        
        has_conflict = any(
            signals.get(a, False) and signals.get(b, False) 
            for a, b in conflicting_pairs
        )
        
        if has_conflict:
            result['is_ambiguous'] = True
            result['ambiguity_type'] = 'multi_intent'
            result['confidence'] = 0.6
            
            if signals.get('needs_restaurant') and signals.get('needs_transportation'):
                result['clarification_questions'] = get_questions('multi_intent_restaurant_transport')
            elif signals.get('needs_attraction') and signals.get('needs_restaurant'):
                result['clarification_questions'] = get_questions('multi_intent_attraction_restaurant')
        
        # Type 2: REMOVED - "too short" queries should go to LLM
        # The LLM can understand short queries like greetings, thanks, etc.
        # We already have a greeting bypass above, and LLM handles everything else
        
        # Type 3: Generic location query
        generic_patterns = [
            r'^where\s+(is|can|should|do)',
            r'^what\s+(is|should|can)',
            r'^(show|tell)\s+me',
            r'^find\s+me',
            r'^i\s+want',
            r'^looking\s+for'
        ]
        
        is_generic = any(re.match(p, query_lower) for p in generic_patterns)
        has_specifics = any([
            # Check for specific locations
            any(loc in query_lower for loc in ['taksim', 'sultanahmet', 'kadikoy', 'besiktas']),
            # Check for specific types
            any(t in query_lower for t in ['kebab', 'fish', 'meze', 'baklava', 'museum', 'mosque']),
        ])
        
        if is_generic and not has_specifics and word_count < 6:
            result['is_ambiguous'] = True
            result['ambiguity_type'] = 'generic'
            result['confidence'] = 0.5
            
            if 'restaurant' in query_lower or 'food' in query_lower or 'eat' in query_lower:
                result['clarification_questions'] = get_questions('generic_restaurant')
            elif 'see' in query_lower or 'visit' in query_lower or 'do' in query_lower:
                result['clarification_questions'] = get_questions('generic_sightseeing')
            else:
                result['clarification_questions'] = get_questions('generic_default')
        
        # Type 4: No signals detected at all
        # REMOVED: "too short" check - LLM can understand any query length
        # The greeting bypass above handles common phrases, and LLM handles everything else
        # Short queries like "nasılsın", "hi", "thanks" should go to LLM, not return errors
        if signal_count == 0:
            # Don't mark as ambiguous - let LLM handle it
            # LLM can understand queries even if regex patterns don't match
            result['is_ambiguous'] = False
            result['confidence'] = 0.7  # Medium confidence - proceed to LLM
        
        return result
    
    def _generate_clarification_response(
        self,
        query: str,
        ambiguity_info: Dict[str, Any],
        language: str = "en"
    ) -> str:
        """
        Generate a clarification response for ambiguous queries.
        Supports 5 languages: English (en), Turkish (tr), Russian (ru), German (de), Arabic (ar)
        
        Args:
            query: Original query
            ambiguity_info: Result from _detect_ambiguous_query
            language: Response language (en/tr/ru/de/ar)
            
        Returns:
            Clarification response string
        """
        questions = ambiguity_info.get('clarification_questions', [])
        ambiguity_type = ambiguity_info.get('ambiguity_type', 'unknown')
        
        # Intro phrases for each language
        intro_phrases_by_lang = {
            'en': {
                'multi_intent': "I can help with several things here:",
                'too_short': "I'd love to help! To give you the best answer:",
                'generic': "To give you better recommendations:",
                'unclear_intent': "I want to make sure I understand correctly:",
                'default': "I'd like to clarify:"
            },
            'tr': {
                'multi_intent': "Birkaç şekilde yardımcı olabilirim:",
                'too_short': "Size daha iyi yardımcı olmak için biraz daha bilgiye ihtiyacım var:",
                'generic': "Daha iyi öneriler sunabilmem için:",
                'unclear_intent': "Sorunuzu tam anlayamadım:",
                'default': "Daha fazla bilgiye ihtiyacım var:"
            },
            'ru': {
                'multi_intent': "Я могу помочь с несколькими вещами:",
                'too_short': "Я бы хотел помочь! Чтобы дать вам лучший ответ:",
                'generic': "Чтобы дать вам лучшие рекомендации:",
                'unclear_intent': "Я хочу убедиться, что правильно понимаю:",
                'default': "Позвольте уточнить:"
            },
            'de': {
                'multi_intent': "Ich kann bei mehreren Dingen helfen:",
                'too_short': "Ich helfe gerne! Um Ihnen die beste Antwort zu geben:",
                'generic': "Um Ihnen bessere Empfehlungen zu geben:",
                'unclear_intent': "Ich möchte sicherstellen, dass ich Sie richtig verstehe:",
                'default': "Lassen Sie mich nachfragen:"
            },
            'ar': {
                'multi_intent': "يمكنني المساعدة في عدة أمور:",
                'too_short': "يسعدني المساعدة! لإعطائك أفضل إجابة:",
                'generic': "لتقديم توصيات أفضل:",
                'unclear_intent': "أريد التأكد من أنني أفهم بشكل صحيح:",
                'default': "اسمحوا لي بالتوضيح:"
            }
        }
        
        # Closing messages for each language
        closing_messages = {
            'en': "💡 I'm your Istanbul guide and can help with restaurants, attractions, transportation, and more!",
            'tr': "💡 İstanbul hakkında restoranlar, turistik yerler, ulaşım ve daha fazlası konusunda yardımcı olabilirim!",
            'ru': "💡 Я ваш гид по Стамбулу и могу помочь с ресторанами, достопримечательностями, транспортом и многим другим!",
            'de': "💡 Ich bin Ihr Istanbul-Führer und kann bei Restaurants, Sehenswürdigkeiten, Transport und mehr helfen!",
            'ar': "💡 أنا دليلك في إسطنبول ويمكنني المساعدة في المطاعم والمعالم السياحية والمواصلات والمزيد!"
        }
        
        # Get phrases for the language (fallback to English)
        intro_phrases = intro_phrases_by_lang.get(language, intro_phrases_by_lang['en'])
        intro = intro_phrases.get(ambiguity_type, intro_phrases['default'])
        
        response = f"{intro}\n\n"
        
        for i, question in enumerate(questions[:2], 1):
            response += f"{i}. {question}\n"
        
        closing = closing_messages.get(language, closing_messages['en'])
        response += f"\n{closing}"
        
        return response

    async def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        max_tokens: int = 768,
        enable_conversation: bool = True,
        conversation_history: Optional[List[Dict[str, str]]] = None
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
            conversation_history: Pre-loaded conversation history from Redis (optional)
            
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
        
        # STEP 3.5: Ambiguity Detection and Clarification Flow
        if self.config.get('enable_clarification_flow', True):
            ambiguity_info = self._detect_ambiguous_query(query, signals['signals'], language=language)
            
            if ambiguity_info['is_ambiguous'] and ambiguity_info['confidence'] < 0.5:
                logger.info(f"❓ Ambiguous query detected: {ambiguity_info['ambiguity_type']} (confidence: {ambiguity_info['confidence']:.2f})")
                
                # Generate clarification response instead of proceeding
                clarification_response = self._generate_clarification_response(
                    query=query,
                    ambiguity_info=ambiguity_info,
                    language=language
                )
                
                return {
                    "status": "clarification_needed",
                    "response": clarification_response,
                    "map_data": None,
                    "signals": signals['signals'],
                    "metadata": {
                        "signals_detected": active_signals,
                        "ambiguity": {
                            "type": ambiguity_info['ambiguity_type'],
                            "confidence": ambiguity_info['confidence'],
                            "questions": ambiguity_info['clarification_questions']
                        },
                        "enhancement": enhancement_metadata,
                        "processing_time": time.time() - start_time,
                        "cached": False,
                        "language": language,
                        "source": "clarification_flow"
                    }
                }
        
        # STEP 4: Conversation Context (if enabled)
        conversation_context = None
        
        # First, use pre-loaded conversation history from Redis (if provided)
        if conversation_history:
            conversation_context = {
                'history': conversation_history,
                'needs_resolution': False,  # Will check below
                'topics': [],
                'entities': {}
            }
            logger.info(f"💭 Using pre-loaded conversation history: {len(conversation_history)} turns")
        
        # Also try local ConversationManager for reference resolution
        if enable_conversation and session_id:
            try:
                local_context = await self.conversation_manager.get_context(
                    session_id=session_id,
                    current_query=query,
                    max_turns=3
                )
                
                # Merge local context with pre-loaded history
                if conversation_context:
                    # Keep the pre-loaded history, but use local context for resolution
                    conversation_context['needs_resolution'] = local_context.get('needs_resolution', False)
                    conversation_context['topics'] = local_context.get('topics', [])
                    conversation_context['entities'] = local_context.get('entities', {})
                else:
                    conversation_context = local_context
                
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
        
        # STEP 5.7: Get User Profile for Personalization (PHASE 1)
        user_profile = None
        if self.config.get('enable_personalization', True) and user_id and user_id != "anonymous":
            try:
                profile_obj = await self.personalization.get_user_profile(user_id)
                
                # Convert UserPreferences dataclass to dict for prompt injection
                user_profile = {
                    'interests': profile_obj.interests,
                    'dietary_restrictions': profile_obj.dietary_restrictions,
                    'cuisine_preferences': profile_obj.preferred_cuisines,
                    'budget_range': profile_obj.preferred_price_range,
                    'preferred_districts': profile_obj.preferred_districts,
                    'favorite_neighborhoods': profile_obj.preferred_districts,  # Alias
                    'preferred_activities': profile_obj.preferred_activities,
                    'travel_style': None,  # Not in PersonalizationEngine profile
                    'group_type': None,    # Not in PersonalizationEngine profile
                    'has_children': False,  # Not in PersonalizationEngine profile
                    'children_ages': [],    # Not in PersonalizationEngine profile
                }
                
                # Filter out empty values
                user_profile = {k: v for k, v in user_profile.items() if v}
                
                if user_profile:
                    logger.info(f"👤 User profile loaded for {user_id}: {len(user_profile)} attributes")
                else:
                    user_profile = None
                    
            except Exception as e:
                logger.warning(f"Failed to load user profile for {user_id}: {e}")
                user_profile = None
        
        # STEP 6: Prompt Engineering (WITH PERSONALIZATION - PHASE 1)
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
            user_profile=user_profile,    # PHASE 1: Pass user profile for personalized recommendations
            enable_intent_classification=enable_intent_classification,  # Priority 2
            signal_confidence=overall_confidence  # Priority 3
        )
        
        logger.info(f"📝 Prompt built: {len(prompt)} chars, signal confidence: {overall_confidence:.2f}")
        if user_location:
            logger.info(f"   📍 GPS location included in prompt: {user_location}")
        if user_profile:
            logger.info(f"   👤 User profile injected into prompt: {list(user_profile.keys())}")
        if overall_confidence < 0.6:
            logger.warning(f"   ⚠️ Low signal confidence - added explicit instructions to prompt")

        
        
        # STEP 7: LLM Generation with Resilience
        try:
            # 🚀 PERFORMANCE OPTIMIZATION: LLM Response Caching
            # Generate cache key from normalized query + context
            import hashlib
            cache_key_data = f"{query.lower().strip()}:{language}:{len(context.get('db_results', []))}:{len(context.get('rag_results', []))}"
            cache_key = f"llm_response:{hashlib.md5(cache_key_data.encode()).hexdigest()}"
            
            # Check Redis cache first (MAJOR PERFORMANCE BOOST)
            cached_response = None
            if hasattr(self, 'redis_client') and self.redis_client:
                try:
                    cached_data = await self.redis_client.get(cache_key)
                    if cached_data:
                        cached_response = json.loads(cached_data)
                        logger.info(f"✅ LLM CACHE HIT! Saved ~9s (key: {cache_key[:16]}...)")
                        
                        # Track cache hit in analytics
                        self.analytics.track_cache_hit()
                        
                        # Return cached response with fresh metadata
                        cached_response['metadata']['cached'] = True
                        cached_response['metadata']['cache_key'] = cache_key
                        cached_response['metadata']['processing_time'] = time.time() - start_time
                        
                        return cached_response
                except Exception as cache_err:
                    logger.warning(f"Cache lookup failed: {cache_err}")
            
            self.analytics.track_cache_miss()
            logger.info(f"💫 LLM CACHE MISS - generating new response (key: {cache_key[:16]}...)")
            
            llm_start = time.time()
            
            # Simplified LLM call with circuit breaker only
            async def _generate_with_llm():
                logger.info(f"📝 Calling LLM...")
                logger.info(f"📏 Prompt length: {len(prompt)} chars")
                logger.info(f"🔚 Prompt ending (last 300 chars): ...{prompt[-300:]}")
                result = await self._llm_call(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    component='main_generation'
                )
                logger.info(f"📥 LLM returned response with keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                if result.get('cached'):
                    logger.info(f"💾 Response from cache (latency: {result.get('latency_ms', 0):.0f}ms)")
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
            
            # === RESPONSE TRUNCATION VALIDATOR ===
            # Detect and handle truncated responses (cut off mid-sentence/word)
            response_text, was_truncated = self._validate_and_fix_truncation(response_text)
            if was_truncated:
                logger.warning(f"⚠️ Response was truncated and cleaned up")
            
            # === FACTUAL GROUNDING CHECK ===
            # If we have no DB context and low RAG results, add disclaimer
            db_context_len = len(context.get('database', ''))
            rag_context_len = len(context.get('rag', ''))
            has_grounding = db_context_len > 50 or rag_context_len > 100
            
            if not has_grounding and signals['signals'].get('needs_restaurant') or signals['signals'].get('needs_attraction'):
                logger.warning(f"⚠️ Low factual grounding (DB: {db_context_len}, RAG: {rag_context_len} chars)")
                # Don't completely block, but add a subtle disclaimer at end
                if "specific" not in response_text.lower() and "recommend checking" not in response_text.lower():
                    response_text = response_text.rstrip()
                    if not response_text.endswith('.'):
                        response_text += '.'
                    response_text += "\n\n💡 *For the most up-to-date info, I recommend verifying details on Google Maps or calling ahead.*"
            
            # === ENTITY VALIDATION (Hallucination Detection) ===
            # Check for potentially hallucinated entities in the response
            # WEEK 3 FIX: Skip entity validation for transport queries to avoid false positives
            is_transport_query = signals['signals'].get('needs_transportation', False)
            
            if self.config.get('enable_entity_validation', True) and not is_transport_query:
                try:
                    validation_score, hallucination_issues = self.entity_validator.get_validation_score(
                        response_text, 
                        is_transport_query=is_transport_query
                    )
                    
                    # Track validation in monitoring system
                    has_hallucinations = validation_score < 0.7
                    try:
                        from .monitoring import get_monitor
                        monitor = get_monitor()
                        monitor.track_entity_validation(validation_score, has_hallucinations)
                    except Exception as mon_err:
                        logger.debug(f"Could not track entity validation in monitor: {mon_err}")
                    
                    if validation_score < 0.7:
                        # Log potentially hallucinated entities
                        total_issues = sum(len(v) for v in hallucination_issues.values())
                        logger.warning(f"⚠️ Entity validation score: {validation_score:.2f}, issues: {total_issues}")
                        
                        for issue_type, entities in hallucination_issues.items():
                            if entities:
                                logger.warning(f"   - Potential hallucinated {issue_type}: {entities}")
                        
                        # Track for analytics
                        self.analytics.track_validation_failure(
                            f"entity_hallucination: {hallucination_issues}"
                        )
                        
                        # If very low score, add a subtle warning
                        if validation_score < 0.4 and "verify" not in response_text.lower():
                            response_text = response_text.rstrip()
                            if not response_text.endswith('.'):
                                response_text += '.'
                            response_text += "\n\n⚠️ *Some details may need verification. Please double-check names and addresses.*"
                    else:
                        logger.debug(f"✅ Entity validation passed: {validation_score:.2f}")
                        
                except Exception as e:
                    logger.warning(f"Entity validation failed: {e}")
            
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
            # WEEK 3 CRITICAL FIX: Make RAG route object AUTHORITATIVE
            # If Transportation RAG has a route, use it - DO NOT let LLM override
            if signals['signals'].get('needs_transportation'):
                logger.info(f"🚇 Applying transportation validation for route query...")
                
                # CRITICAL: Check RAG state FIRST (source of truth)
                # The RAG already computed the route during context building
                rag_has_route = False
                try:
                    from services.transportation_rag_system import get_transportation_rag
                    transport_rag = get_transportation_rag()
                    rag_has_route = transport_rag and transport_rag.last_route is not None
                    if rag_has_route:
                        logger.info(f"✅ RAG has computed route: {transport_rag.last_route.origin} → {transport_rag.last_route.destination}")
                except Exception as e:
                    logger.warning(f"Could not check RAG route state: {e}")
                
                # Extract route data from context (verified facts from Transportation RAG)
                route_data = context.get('route_data')
                
                # Fallback: check context['services'] for legacy compatibility
                if not route_data and context.get('services'):
                    if isinstance(context['services'], dict):
                        route_data = context['services'].get('transportation', {}).get('route')
                    else:
                        for service_item in context['services']:
                            if isinstance(service_item, dict) and 'route' in service_item:
                                route_data = service_item.get('route')
                                break
                
                # WEEK 3 FIX: Trust RAG state over route_data dict
                # If RAG has a route but route_data is missing, rebuild it
                if rag_has_route and not route_data:
                    logger.warning(f"⚠️ RAG has route but route_data missing - rebuilding from RAG")
                    try:
                        route_obj = transport_rag.last_route
                        route_data = {
                            'origin': getattr(route_obj, 'origin', None),
                            'destination': getattr(route_obj, 'destination', None),
                            'steps': getattr(route_obj, 'steps', []),
                            'total_time': getattr(route_obj, 'total_time', None),
                            'total_distance': getattr(route_obj, 'total_distance', None),
                            'transfers': getattr(route_obj, 'transfers', None),
                            'lines_used': getattr(route_obj, 'lines_used', [])
                        }
                        logger.info(f"✅ Rebuilt route_data from RAG: {route_data['origin']} → {route_data['destination']}")
                    except Exception as e:
                        logger.error(f"Failed to rebuild route_data from RAG: {e}")
                
                logger.info(f"🚇 Route data found: {bool(route_data)}, has steps: {bool(route_data and route_data.get('steps'))}")
                logger.info(f"🚇 RAG has route: {rag_has_route}")
                logger.info(f"🚇 DEBUG: route_data = {route_data}")
                logger.info(f"🚇 DEBUG: route_data.get('steps') if route_data else None = {route_data.get('steps') if route_data else None}")
                
                # WEEK 3 CRITICAL RULE:
                # If RAG has a route OR we have route_data with steps -> ALWAYS use template
                # The LLM should NEVER say "no route found" if RAG computed one
                if rag_has_route or (route_data and route_data.get('steps')):
                    logger.info(f"✅ Using template-based transportation response (RAG is authoritative)")
                    response_text = await self._generate_template_transportation_response(
                        route_data=route_data,
                        query=query,
                        language=language
                    )
                else:
                    # No route from RAG - validate LLM response is not hallucinating
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
            
            # === WEATHER-SPECIFIC POST-PROCESSING ===
            # Ensure real weather data is included in weather query responses
            if signals['signals'].get('needs_weather'):
                logger.info(f"🌤️ Applying weather post-processing for weather query...")
                response_text = self._post_process_weather_response(
                    response=response_text,
                    context=context,
                    language=language
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
        # use the Transportation RAG system which already computed the route
        if not map_data and signals['signals'].get('needs_transportation'):
            logger.info(f"🚇 Detected transportation query, getting map_data from RAG...")
            try:
                from services.transportation_rag_system import get_transportation_rag
                
                transport_rag = get_transportation_rag()
                if transport_rag and transport_rag.last_route:
                    # Use the route already computed by the RAG system during context building
                    map_data = transport_rag.get_map_data_for_last_route()
                    if map_data:
                        logger.info(f"✅ Got map_data from Transportation RAG: {map_data.get('route_data', {}).get('origin')} → {map_data.get('route_data', {}).get('destination')}")
                    else:
                        logger.warning("⚠️ Transportation RAG has last_route but get_map_data_for_last_route() returned None")
                else:
                    logger.warning("⚠️ No last_route available from Transportation RAG")
                    
            except Exception as e:
                logger.error(f"❌ Error getting map_data from Transportation RAG: {e}", exc_info=True)
        
        # ==================================================================
        # 3️⃣ LOCATION-BASED QUERIES (Restaurants, Attractions, etc.)
        # ==================================================================
        # Map data for location-based queries is already included via context.get('map_data')
        # The context builder extracts map_data from database/RAG results
        if not map_data and any([
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
            logger.info(f"🗺️ Checking if map_data available from context...")
            
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

        # 🚀 PERFORMANCE: Cache LLM Response for Future Use
        # Store the result in Redis cache (MAJOR PERFORMANCE OPTIMIZATION)
        if not bypass_cache and hasattr(self, 'redis_client') and self.redis_client:
            try:
                cache_ttl = 3600  # Cache for 1 hour
                await self.redis_client.setex(
                    cache_key, 
                    cache_ttl, 
                    json.dumps(result, ensure_ascii=False)
                )
                logger.info(f"💾 Cached LLM response for {cache_ttl}s (key: {cache_key[:16]}...)")
                logger.info(f"   📊 Response size: {len(json.dumps(result))} bytes")
            except Exception as cache_err:
                logger.warning(f"Failed to cache LLM response: {cache_err}")
        
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
        max_tokens: int = 768,
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
            
            # Use unified streaming interface
            async for token in self._llm_stream(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                component='streaming_generation'
            ):
                response_tokens.append(token)
                yield {
                    'type': 'token',
                    'data': token,
                    'cached': False
                }
            
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
    
    def _post_process_weather_response(
        self,
        response: str,
        context: Dict[str, Any],
        language: str = "en"
    ) -> str:
        """
        Post-process weather query responses to ensure real weather data is included.
        
        The LLM sometimes ignores weather context and hallucinates values.
        This method extracts the real weather data from context and ensures
        it's prominently included in the response.
        
        Args:
            response: LLM-generated response
            context: Built context containing weather data
            language: Response language (en/tr)
            
        Returns:
            Post-processed response with verified weather data
        """
        try:
            # Extract weather data from context
            weather_context = context.get('services', {}).get('weather', '')
            
            if not weather_context:
                logger.warning("⚠️ No weather context found for post-processing")
                return response
            
            logger.info(f"🌤️ Weather post-processing - context: {weather_context[:200]}...")
            
            # Parse the weather context to extract real values
            # Format: "Current weather in Istanbul: <condition>, <temp>°C. <description>"
            import re
            
            # Extract temperature
            temp_match = re.search(r'(\d+(?:\.\d+)?)\s*°C', weather_context)
            real_temp = temp_match.group(1) if temp_match else None
            
            # Extract condition - try multiple patterns
            real_condition = None
            
            # Pattern 1: "Current weather in Istanbul: Clouds, 9°C"
            condition_match = re.search(r'weather in Istanbul:\s*([^,]+)', weather_context, re.IGNORECASE)
            if condition_match:
                real_condition = condition_match.group(1).strip()
            
            # Pattern 2: "Currently: Clouds"
            if not real_condition:
                condition_match = re.search(r'Currently:\s*([^\n,]+)', weather_context, re.IGNORECASE)
                real_condition = condition_match.group(1).strip() if condition_match else None
            
            # Pattern 3: Look for common weather conditions
            if not real_condition or real_condition == 'Unknown':
                common_conditions = ['Clear', 'Clouds', 'Cloudy', 'Rain', 'Drizzle', 'Snow', 'Sunny', 
                                     'Partly cloudy', 'Overcast', 'Thunderstorm', 'Mist', 'Fog']
                for cond in common_conditions:
                    if cond.lower() in weather_context.lower():
                        real_condition = cond
                        break
            
            logger.info(f"🌡️ Extracted weather: temp={real_temp}°C, condition={real_condition}")
            
            if not real_temp:
                logger.warning("⚠️ Could not extract temperature from weather context")
                return response
            
            # Check if the response already contains the correct temperature
            if real_temp and real_temp in response:
                logger.info(f"✅ Response already contains correct temperature ({real_temp}°C)")
                return response
            
            # The LLM hallucinated the temperature - we need to fix this
            logger.warning(f"🚨 LLM response does not contain correct weather data, prepending real data")
            
            # Create a weather summary header with multilingual support
            weather_headers = {
                'en': f"""🌤️ **Current Istanbul Weather** (Real-Time Data)
📍 Right now: {real_condition or 'Clear'}, {real_temp}°C

---

""",
                'tr': f"""🌤️ **Güncel İstanbul Havası** (Gerçek Zamanlı)
📍 Şu anda: {real_condition or 'Açık'}, {real_temp}°C

---

""",
                'ru': f"""🌤️ **Текущая погода в Стамбуле** (В реальном времени)
📍 Сейчас: {real_condition or 'Ясно'}, {real_temp}°C

---

""",
                'de': f"""🌤️ **Aktuelles Istanbul-Wetter** (Echtzeit-Daten)
📍 Gerade jetzt: {real_condition or 'Klar'}, {real_temp}°C

---

""",
                'ar': f"""🌤️ **طقس إسطنبول الحالي** (بيانات في الوقت الفعلي)
📍 الآن: {real_condition or 'صافي'}, {real_temp}°C

---

"""
            }
            
            weather_header = weather_headers.get(language, weather_headers['en'])
            
            # Prepend the verified weather header to the response
            corrected_response = weather_header + response
            
            logger.info(f"✅ Weather post-processing complete - prepended verified data")
            return corrected_response
            
        except Exception as e:
            logger.error(f"❌ Weather post-processing failed: {e}")
            return response
    
    def _validate_and_fix_truncation(self, response: str) -> Tuple[str, bool]:
        """
        Detect and fix truncated responses.
        
        Truncation indicators:
        - Ends with incomplete markdown (*, **, -)
        - Ends mid-word
        - Ends with incomplete sentence (no punctuation)
        - Cut off list items
        
        Args:
            response: The generated response text
            
        Returns:
            Tuple of (cleaned_response, was_truncated)
        """
        if not response:
            return response, False
        
        original_response = response
        was_truncated = False
        
        # Strip trailing whitespace first
        response = response.rstrip()
        
        # Pattern 1: Ends with incomplete markdown formatting
        truncation_patterns = [
            r'\*\*[^*]*$',       # Unclosed bold (e.g., "**Beş")
            r'\*[^*]*$',         # Unclosed italic
            r'^\s*[-*]\s*\*\*[^*]*$',  # List item with unclosed bold
            r'^\s*[-*]\s*$',     # Empty list item
            r'^\s*\d+\.\s*$',    # Empty numbered list item
        ]
        
        for pattern in truncation_patterns:
            if re.search(pattern, response, re.MULTILINE):
                was_truncated = True
                # Remove the incomplete line
                lines = response.split('\n')
                while lines and re.search(pattern, lines[-1]):
                    lines.pop()
                response = '\n'.join(lines)
        
        # Pattern 2: Ends mid-sentence (no terminal punctuation)
        # But allow responses ending with emoji or markdown
        valid_endings = '.!?)"\'`>]'
        emoji_pattern = r'[\U0001F300-\U0001F9FF]$'
        
        if response and response[-1] not in valid_endings and not re.search(emoji_pattern, response):
            # Check if it's a cut-off word (no space before last word boundary)
            last_line = response.split('\n')[-1].strip()
            words = last_line.split()
            
            if words:
                last_word = words[-1].rstrip('*_`')
                # If last word is very short or looks incomplete
                if len(last_word) <= 2 and last_word.isalpha():
                    was_truncated = True
                    # Remove the incomplete word
                    response = response[:response.rfind(last_word)].rstrip()
        
        # Pattern 3: Ends with standalone formatting characters
        while response and response[-1] in '*_-:':
            was_truncated = True
            response = response[:-1].rstrip()
        
        # Pattern 4: Ensure response ends cleanly
        # If truncated, add ellipsis to indicate incompleteness (optional)
        if was_truncated and response:
            # Clean up any trailing incomplete structures
            response = response.rstrip(' \t*_-:')
            
            # If response now ends awkwardly, trim to last complete sentence
            if response and response[-1] not in '.!?)':
                # Find last sentence boundary
                for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                    last_punct = response.rfind(punct)
                    if last_punct > len(response) * 0.5:  # Only trim if we keep >50%
                        response = response[:last_punct + 1]
                        break
        
        if was_truncated:
            logger.info(f"🔧 Fixed truncated response: {len(original_response)} → {len(response)} chars")
        
        return response.strip(), was_truncated
    
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
        
        # Check for actual LLM error patterns (not conversational "sorry")
        # Only fail on specific error patterns that indicate LLM failure
        response_lower = response.lower()
        
        # Specific error patterns that indicate actual failures
        error_patterns = [
            "i cannot process",
            "i am unable to",
            "internal error",
            "service unavailable",
            "technical difficulties",
            "something went wrong",
            "failed to generate",
            "error occurred",
            "unable to complete your request",
            "i apologize, but i'm experiencing technical difficulties"
        ]
        
        for pattern in error_patterns:
            if pattern in response_lower:
                return False, f"Error pattern detected: {pattern}"
        
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
            't1', 't4', 't5', 'marmaray', 'f1', 'f2', 'metrobus',
            'havaist', 'iett', 'ido', 'turyol', 'sehir hatlari'
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
            'taksim', 'sultanahmet', 'kadiköy', 'üsküdar', 'beşiktaş',
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
        
        Supports 5 languages: English (en), Turkish (tr), Russian (ru), German (de), Arabic (ar)
        
        Args:
            route_data: Verified route data from Transportation RAG (ground truth)
            query: Original user query
            language: Response language ('en', 'tr', 'ru', 'de', 'ar')
            
        Returns:
            Template-based response text with verified facts only
        """
        try:
            # Extract verified facts from route data
            origin = route_data.get('origin', 'Starting point')
            destination = route_data.get('destination', 'Destination')
            total_time = route_data.get('total_time', 0)
            total_distance = route_data.get('total_distance', 0)
            transfers = route_data.get('transfers', 0)
            lines_used = route_data.get('lines_used', [])
            steps = route_data.get('steps', [])
            
            # =================================================================
            # MULTILINGUAL TEMPLATES (Clean format without Markdown)
            # =================================================================
            templates = {
                'en': {
                    'header': f"🚇 Route: {origin} → {destination}",
                    'duration': f"⏱️ Duration: {total_time} minutes",
                    'distance': f"📏 Distance: {total_distance:.1f} km",
                    'transfers': f"🔄 Transfers: {transfers}",
                    'lines': f"🚇 Lines: {', '.join(lines_used)}",
                    'step_header': "📍 Step-by-Step:",
                    'verified': "✅ This route has been verified in Istanbul's transportation database.",
                    'fallback': "Route information is available but could not be displayed. Please try again.",
                    'time_unit': 'min',
                    'transit': lambda l, f, t: f"Take {l} from {f} to {t}",
                    'transfer': lambda l, s: f"Transfer to {l} at {s}",
                    'walk': lambda s: f"Walk to {s}",
                },
                'tr': {
                    'header': f"🚇 Güzergah: {origin} → {destination}",
                    'duration': f"⏱️ Süre: {total_time} dakika",
                    'distance': f"📏 Mesafe: {total_distance:.1f} km",
                    'transfers': f"🔄 Aktarma: {transfers} aktarma",
                    'lines': f"🚇 Hatlar: {', '.join(lines_used)}",
                    'step_header': "📍 Adım Adım:",
                    'verified': "✅ Bu güzergah İstanbul ulaşım veritabanından doğrulanmıştır.",
                    'fallback': "Güzergah bilgisi mevcut ancak görüntülenemiyor. Lütfen tekrar deneyin.",
                    'time_unit': 'dk',
                    'transit': lambda l, f, t: f"{l} ile {f}'dan {t}'a gidin",
                    'transfer': lambda l, s: f"{s}'da {l} hattına aktarma yapın",
                    'walk': lambda s: f"{s}'a yürüyün",
                },
                'ru': {
                    'header': f"🚇 Маршрут: {origin} → {destination}",
                    'duration': f"⏱️ Время: {total_time} минут",
                    'distance': f"📏 **Расстояние:** {total_distance:.1f} км",
                    'transfers': f"🔄 **Пересадки:** {transfers}",
                    'lines': f"🚇 **Линии:** {', '.join(lines_used)}",
                    'step_header': "**📍 Пошагово:**",
                    'verified': "✅ Этот маршрут проверен в транспортной базе данных Стамбула.",
                    'fallback': "Информация о маршруте доступна, но не может быть отображена. Попробуйте снова.",
                    'time_unit': 'мин',
                    'transit': lambda l, f, t: f"Сядьте на {l} от {f} до {t}",
                    'transfer': lambda l, s: f"Пересядьте на {l} на станции {s}",
                    'walk': lambda s: f"Идите до {s}",
                },
                'de': {
                    'header': f"**Route: {origin} → {destination}**",
                    'duration': f"⏱️ **Dauer:** {total_time} Minuten",
                    'distance': f"📏 **Entfernung:** {total_distance:.1f} km",
                    'transfers': f"� **Umstiege:** {transfers}",
                    'lines': f"🚇 **Linien:** {', '.join(lines_used)}",
                    'step_header': "**📍 Schritt für Schritt:**",
                    'verified': "✅ Diese Route wurde in der Istanbuler Verkehrsdatenbank verifiziert.",
                    'fallback': "Routeninformationen sind verfügbar, können aber nicht angezeigt werden. Bitte versuchen Sie es erneut.",
                    'time_unit': 'Min',
                    'transit': lambda l, f, t: f"Nehmen Sie {l} von {f} nach {t}",
                    'transfer': lambda l, s: f"Umsteigen auf {l} bei {s}",
                    'walk': lambda s: f"Gehen Sie zu {s}",
                },
                'ar': {
                    'header': f"**المسار: {origin} → {destination}**",
                    'duration': f"⏱️ **المدة:** {total_time} دقائق",
                    'distance': f"📏 **المسافة:** {total_distance:.1f} كم",
                    'transfers': f"🔄 **التحويلات:** {transfers}",
                    'lines': f"🚇 **الخطوط:** {', '.join(lines_used)}",
                    'step_header': "**📍 خطوة بخطوة:**",
                    'verified': "✅ تم التحقق من هذا المسار في قاعدة بيانات النقل في اسطنبول.",
                    'fallback': "معلومات المسار متاحة ولكن لا يمكن عرضها. حاول مرة أخرى.",
                    'time_unit': 'د',
                    'transit': lambda l, f, t: f"استقل {l} من {f} إلى {t}",
                    'transfer': lambda l, s: f"انتقل إلى {l} في {s}",
                    'walk': lambda s: f"امشِ إلى {s}",
                },
            }
            
            # Default to English if language not supported
            if language not in templates:
                language = 'en'
            
            t = templates[language]
            
            # Build response
            response = f"{t['header']}\n\n"
            response += f"{t['duration']}\n"
            response += f"{t['distance']}\n"
            response += f"{t['transfers']}\n"
            response += f"{t['lines']}\n\n"
            
            if steps:
                response += f"{t['step_header']}\n\n"
                step_num = 1
                for step in steps:
                    step_type = step.get('type', 'transit')
                    line = step.get('line', '')
                    from_loc = step.get('from', '')
                    to_loc = step.get('to', '')
                    duration = step.get('duration', 0)
                    
                    if step_type == 'transfer':
                        instruction = t['transfer'](line, from_loc)
                        response += f"{step_num}. 🔄 **{instruction}** ({duration:.0f} {t['time_unit']})\n"
                    elif step_type == 'walk':
                        instruction = t['walk'](to_loc)
                        response += f"{step_num}. 🚶 **{instruction}** ({duration:.0f} {t['time_unit']})\n"
                    else:
                        instruction = t['transit'](line, from_loc, to_loc)
                        response += f"{step_num}. 🚇 **{instruction}** ({duration:.0f} {t['time_unit']})\n"
                    step_num += 1
            
            response += f"\n{t['verified']}"
            
            logger.info(f"✅ Generated template-based transportation response (fact-locked, no LLM, lang={language})")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to generate template response: {e}")
            # Ultra-minimal fallback
            fallback_msgs = {
                'en': "Route information is available but could not be displayed. Please try again.",
                'tr': "Güzergah bilgisi mevcut ancak görüntülenemiyor. Lütfen tekrar deneyin.",
                'ru': "Информация о маршруте доступна, но не может быть отображена. Попробуйте снова.",
                'de': "Routeninformationen sind verfügbar, können aber nicht angezeigt werden. Bitte versuchen Sie es erneut.",
                'ar': "معلومات المسار متاحة ولكن لا يمكن عرضها. حاول مرة أخرى.",
            }
            return fallback_msgs.get(language, fallback_msgs['en'])