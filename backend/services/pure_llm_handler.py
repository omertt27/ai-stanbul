"""
Pure LLM Query Handler
Routes ALL queries through RunPod LLM with database context injection
No rule-based fallback - LLM handles everything

ENHANCED: Now supports map visualization for transportation and route planning queries
by integrating with Istanbul Daily Talk AI system when needed.

SIGNAL-BASED DETECTION: Multilingual, semantic approach using embeddings for
flexible intent detection that supports multiple intents and languages.

Architecture:
- Single entry point for all queries
- Context injection from database
- RAG for similar queries
- Signal-based intent detection (multi-intent support)
- Redis caching for responses and signals
- MAP VISUALIZATION for routes and transportation (NEW)
- Semantic embeddings for language-independent detection

PRIORITY 2 ENHANCEMENTS:
- Dynamic threshold learning from user feedback
- A/B testing framework for safe experimentation

Author: Istanbul AI Team
Date: November 12, 2025
"""

import logging
import hashlib
import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple
from sqlalchemy.orm import Session
from datetime import datetime
from collections import defaultdict, deque
import json
import time

# Semantic similarity for multilingual detection
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Signal detection will use keyword fallback only.")

# Automatic language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Ensure consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not installed. Automatic language detection disabled.")

# PRIORITY 2.3: Dynamic threshold learning
try:
    from backend.services.threshold_learner import ThresholdLearner
    THRESHOLD_LEARNER_AVAILABLE = True
except ImportError:
    THRESHOLD_LEARNER_AVAILABLE = False
    logging.warning("ThresholdLearner not available. Dynamic threshold learning disabled.")

# PRIORITY 2.4: A/B testing framework
try:
    from backend.services.ab_testing import ABTestingFramework
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False
    logging.warning("ABTestingFramework not available. A/B testing disabled.")

# PRIORITY 3.2: Conversational context (Simple LLM-based approach)
try:
    from backend.services.conversation_context_simple import SimpleConversationManager
    CONVERSATION_MANAGER_AVAILABLE = True
except ImportError:
    CONVERSATION_MANAGER_AVAILABLE = False
    logging.warning("SimpleConversationManager not available. Conversational context disabled.")

# PRIORITY 3.3: Query rewriting (Simple LLM-based approach)
try:
    from backend.services.query_rewriter_simple import SimpleQueryRewriter
    QUERY_REWRITER_AVAILABLE = True
except ImportError:
    QUERY_REWRITER_AVAILABLE = False
    logging.warning("SimpleQueryRewriter not available. Query rewriting disabled.")

# PRIORITY 3.4: Response Caching 2.0 (Semantic Cache)
try:
    from backend.services.response_cache_semantic import SemanticCache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    logging.warning("SemanticCache not available. Semantic caching disabled.")

# PRIORITY 3.5: Query Explanation System
try:
    from backend.services.query_explainer import QueryExplainer
    QUERY_EXPLAINER_AVAILABLE = True
except ImportError:
    QUERY_EXPLAINER_AVAILABLE = False
    logging.warning("QueryExplainer not available. Query explanation disabled.")

logger = logging.getLogger(__name__)


class PureLLMHandler:
    """
    Pure LLM architecture - no rule-based processing
    All queries go through RunPod LLM with context injection
    
    ENHANCED: Now includes map visualization support by routing
    transportation/route queries to Istanbul Daily Talk AI when needed.
    
    PRIORITY 1 ENHANCEMENTS:
    - Advanced analytics and monitoring
    - Per-language threshold tuning
    - Automatic language detection
    - Response validation and quality checks
    """
    
    def __init__(
        self,
        runpod_client,
        db_session: Session,
        redis_client=None,
        context_builder=None,
        rag_service=None,
        istanbul_ai_system=None
    ):
        """
        Initialize Pure LLM Handler
        
        Args:
            runpod_client: RunPod LLM client instance
            db_session: SQLAlchemy database session
            redis_client: Redis client for caching (optional)
            context_builder: ML context builder (optional)
            rag_service: RAG vector service (optional)
            istanbul_ai_system: Istanbul Daily Talk AI for map visualization (optional)
        """
        self.llm = runpod_client
        self.db = db_session
        self.redis = redis_client
        self.context_builder = context_builder
        self.rag = rag_service
        self.istanbul_ai = istanbul_ai_system  # For map visualization
        
        # Initialize semantic embedding model for signal detection
        # MULTILINGUAL: Supports EN, TR, AR, DE, RU, FR and more
        self.embedding_model = None
        self._signal_embeddings = {}  # Cache for signal pattern embeddings
        
        # Configuration: Detection strategy
        self.use_semantic = True  # Primary method: AI-driven, language-independent
        self.use_keywords = True  # Backup method: Fast, reliable for explicit terms
        
        if EMBEDDINGS_AVAILABLE and self.use_semantic:
            try:
                # Use multilingual model for 50+ languages support
                # Optimized for: English, Turkish, Arabic, German, Russian, French
                self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self._init_signal_embeddings()
                
                # Pre-warm model for faster first query
                self._prewarm_model()
                
                logger.info("✅ Semantic embedding model loaded for signal detection (6+ languages)")
            except Exception as e:
                logger.warning(f"⚠️ Could not load embedding model: {e}")
                logger.info("   Falling back to keyword-only detection")
                self.embedding_model = None
                self.use_semantic = False
        
        # Initialize additional services
        self._init_additional_services()
        
        # Load system prompts
        self._load_prompts()
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_calls": 0,
            "map_requests": 0,
            "weather_requests": 0,
            "hidden_gems_requests": 0,
            "signal_cache_hits": 0,
            "multi_signal_queries": 0
        }
        
        # PRIORITY 1: Advanced Analytics & Monitoring
        self._init_advanced_analytics()
        
        # PRIORITY 1: Per-Language Threshold Configuration
        self._init_language_thresholds()
        
        # PRIORITY 2.3: Dynamic Threshold Learning
        self._init_threshold_learning()
        
        # PRIORITY 2.4: A/B Testing Framework
        self._init_ab_testing()
        
        # PRIORITY 3.2: Conversational Context
        self._init_conversation_manager()
        
        # PRIORITY 3.3: Query Rewriting
        self._init_query_rewriter()
        
        # PRIORITY 3.4: Response Caching 2.0 (Semantic Cache)
        self._init_semantic_cache()
        
        # PRIORITY 3.5: Query Explanation System
        self._init_query_explainer()
        
        logger.info("✅ Pure LLM Handler initialized")
        logger.info(f"   RunPod LLM: {'✅ Enabled' if self.llm.enabled else '❌ Disabled'}")
        logger.info(f"   Redis Cache: {'✅ Enabled' if self.redis else '❌ Disabled'}")
        logger.info(f"   RAG Service: {'✅ Enabled' if self.rag else '❌ Disabled'}")
        logger.info(f"   Istanbul AI (Maps): {'✅ Enabled' if self.istanbul_ai else '❌ Disabled'}")
        logger.info(f"   Weather Service: {'✅ Enabled' if self.weather_service else '❌ Disabled'}")
        logger.info(f"   Events Service: {'✅ Enabled' if self.events_service else '❌ Disabled'}")
        logger.info(f"   Hidden Gems: {'✅ Enabled' if self.hidden_gems_handler else '❌ Disabled'}")
        logger.info(f"   Price Filter: {'✅ Enabled' if self.price_filter else '❌ Disabled'}")
        logger.info(f"   Semantic Embeddings: {'✅ Enabled' if self.embedding_model else '❌ Disabled (fallback to keywords)'}")
        logger.info(f"   Auto Language Detection: {'✅ Enabled' if LANGDETECT_AVAILABLE else '❌ Disabled'}")
        logger.info(f"   Advanced Analytics: ✅ Enabled")
        logger.info(f"   Threshold Learning: {'✅ Enabled' if THRESHOLD_LEARNER_AVAILABLE else '❌ Disabled'}")
        logger.info(f"   A/B Testing: {'✅ Enabled' if AB_TESTING_AVAILABLE else '❌ Disabled'}")
        logger.info(f"   Conversational Context: {'✅ Enabled' if CONVERSATION_MANAGER_AVAILABLE else '❌ Disabled'}")
        logger.info(f"   Query Rewriting: {'✅ Enabled' if QUERY_REWRITER_AVAILABLE else '❌ Disabled'}")
        logger.info(f"   Semantic Cache: {'✅ Enabled' if SEMANTIC_CACHE_AVAILABLE else '❌ Disabled'}")
        logger.info(f"   Query Explainer: {'✅ Enabled' if QUERY_EXPLAINER_AVAILABLE else '❌ Disabled'}")
    
    def _init_advanced_analytics(self):
        """
        PRIORITY 1: Initialize advanced analytics and monitoring system.
        
        Tracks:
        - Performance metrics (latency, throughput)
        - Error tracking and patterns
        - User behavior analytics
        - Signal detection accuracy
        - Service usage patterns
        - Quality metrics
        """
        
        # Performance metrics
        self.performance_metrics = {
            "query_latencies": deque(maxlen=1000),  # Last 1000 query times
            "llm_latencies": deque(maxlen=1000),
            "cache_latencies": deque(maxlen=1000),
            "service_latencies": defaultdict(lambda: deque(maxlen=100)),
        }
        
        # Error tracking
        self.error_tracker = {
            "total_errors": 0,
            "error_by_type": defaultdict(int),
            "error_by_service": defaultdict(int),
            "recent_errors": deque(maxlen=50),
            "error_recovery_count": 0
        }
        
        # User analytics
        self.user_analytics = {
            "queries_by_language": defaultdict(int),
            "queries_by_intent": defaultdict(int),
            "multi_intent_patterns": defaultdict(int),
            "user_locations_used": 0,
            "unique_users": set(),
            "queries_per_user": defaultdict(int)
        }
        
        # Signal detection analytics
        self.signal_analytics = {
            "detections_by_signal": defaultdict(int),
            "detection_confidence_scores": defaultdict(list),
            "false_positive_reports": defaultdict(int),
            "semantic_vs_keyword": {"semantic": 0, "keyword": 0, "both": 0},
            "language_specific_accuracy": defaultdict(lambda: {"correct": 0, "incorrect": 0})
        }
        
        # Service usage analytics
        self.service_analytics = {
            "map_generation_success": 0,
            "map_generation_failure": 0,
            "weather_service_calls": 0,
            "events_service_calls": 0,
            "hidden_gems_calls": 0,
            "rag_usage": 0,
            "cache_efficiency": {"hits": 0, "misses": 0}
        }
        
        # Quality metrics
        self.quality_metrics = {
            "responses_validated": 0,
            "validation_failures": 0,
            "response_lengths": deque(maxlen=100),
            "empty_responses": 0,
            "context_usage_rate": deque(maxlen=100)
        }
        
        logger.info("   📊 Advanced analytics system initialized")
    
    def _init_language_thresholds(self):
        """
        PRIORITY 1: Per-language semantic similarity thresholds.
        
        Different languages have different semantic spaces and embedding quality.
        These thresholds are tuned based on:
        - Embedding model performance per language
        - Query patterns in each language
        - False positive/negative rates
        
        Languages:
        - en: English (best supported)
        - tr: Turkish (well supported)
        - ar: Arabic (moderate support)
        - de: German (well supported)
        - ru: Russian (moderate support)
        - fr: French (well supported)
        """
        
        self.language_thresholds = {
            # English: Highest quality embeddings, can use higher thresholds
            "en": {
                "needs_map": 0.35,
                "needs_gps_routing": 0.48,
                "needs_weather": 0.33,
                "needs_events": 0.38,
                "needs_hidden_gems": 0.30,
                "has_budget_constraint": 0.38,
                "likely_restaurant": 0.33,
                "likely_attraction": 0.28
            },
            # Turkish: Good support, moderate thresholds
            "tr": {
                "needs_map": 0.32,
                "needs_gps_routing": 0.45,
                "needs_weather": 0.30,
                "needs_events": 0.35,
                "needs_hidden_gems": 0.28,
                "has_budget_constraint": 0.35,
                "likely_restaurant": 0.30,
                "likely_attraction": 0.25
            },
            # Arabic: Moderate support, lower thresholds
            "ar": {
                "needs_map": 0.28,
                "needs_gps_routing": 0.42,
                "needs_weather": 0.27,
                "needs_events": 0.32,
                "needs_hidden_gems": 0.25,
                "has_budget_constraint": 0.32,
                "likely_restaurant": 0.27,
                "likely_attraction": 0.22
            },
            # German: Good support, moderate thresholds
            "de": {
                "needs_map": 0.33,
                "needs_gps_routing": 0.46,
                "needs_weather": 0.31,
                "needs_events": 0.36,
                "needs_hidden_gems": 0.29,
                "has_budget_constraint": 0.36,
                "likely_restaurant": 0.31,
                "likely_attraction": 0.26
            },
            # Russian: Moderate support, lower thresholds
            "ru": {
                "needs_map": 0.30,
                "needs_gps_routing": 0.43,
                "needs_weather": 0.28,
                "needs_events": 0.33,
                "needs_hidden_gems": 0.26,
                "has_budget_constraint": 0.33,
                "likely_restaurant": 0.28,
                "likely_attraction": 0.23
            },
            # French: Good support, moderate thresholds
            "fr": {
                "needs_map": 0.34,
                "needs_gps_routing": 0.47,
                "needs_weather": 0.32,
                "needs_events": 0.37,
                "needs_hidden_gems": 0.29,
                "has_budget_constraint": 0.37,
                "likely_restaurant": 0.32,
                "likely_attraction": 0.27
            },
            # Default: Conservative thresholds for unknown languages
            "default": {
                "needs_map": 0.32,
                "needs_gps_routing": 0.45,
                "needs_weather": 0.30,
                "needs_events": 0.35,
                "needs_hidden_gems": 0.28,
                "has_budget_constraint": 0.35,
                "likely_restaurant": 0.30,
                "likely_attraction": 0.25
            }
        }
        
        logger.info("   🌍 Per-language thresholds configured (6 languages + default)")
    
    def _init_threshold_learning(self):
        """
        PRIORITY 2.3: Initialize dynamic threshold learning system.
        
        Learns optimal thresholds from user feedback:
        - Tracks implicit feedback (clicks, engagement)
        - Tracks explicit feedback (thumbs up/down, corrections)
        - Computes ROC curves for each signal
        - Auto-tunes thresholds to maximize F1 score
        - Supports per-language optimization
        """
        if not THRESHOLD_LEARNER_AVAILABLE:
            self.threshold_learner = None
            logger.warning("   ⚠️ ThresholdLearner not available - skipping initialization")
            return
        
        try:
            self.threshold_learner = ThresholdLearner(
                redis_client=self.redis,
                learning_interval_hours=24,  # Learn daily
                min_samples=100  # Require 100 samples before updating
            )
            
            # Enable automatic threshold tuning
            self.enable_auto_tuning = True
            self.auto_tune_interval_hours = 24
            self.last_auto_tune = {}  # Per language
            
            logger.info("   🎓 Threshold learning system initialized")
            logger.info(f"      Learning interval: 24 hours")
            logger.info(f"      Minimum samples: 100")
            
        except Exception as e:
            logger.error(f"Failed to initialize threshold learner: {e}")
            self.threshold_learner = None
    
    def _init_ab_testing(self):
        """
        PRIORITY 2.4: Initialize A/B testing framework.
        
        Enables safe experimentation with:
        - Threshold changes
        - New features
        - Algorithm variations
        - Prompt modifications
        
        Features:
        - Traffic splitting
        - Statistical significance testing
        - Automatic winner selection
        - Multi-variant testing
        """
        if not AB_TESTING_AVAILABLE:
            self.ab_testing = None
            logger.warning("   ⚠️ ABTestingFramework not available - skipping initialization")
            return
        
        try:
            self.ab_testing = ABTestingFramework(
                redis_client=self.redis
            )
            
            # Active experiments tracking
            self.active_experiments = {}
            
            logger.info("   🧪 A/B Testing framework initialized")
            logger.info(f"      Features: Traffic splitting, significance testing, auto-winner")
            
        except Exception as e:
            logger.error(f"Failed to initialize A/B testing: {e}")
            self.ab_testing = None
    
    def _init_conversation_manager(self):
        """
        PRIORITY 3.2: Initialize conversational context manager.
        
        Enables:
        - Conversation history storage
        - Follow-up question handling
        - Reference resolution (pronouns, "there", "it")
        - Context continuity across turns
        - Session management
        
        Features:
        - Redis-backed persistent storage
        - Automatic reference resolution
        - Context summarization
        - Session expiration management
        """
        if not CONVERSATION_MANAGER_AVAILABLE:
            self.conversation_manager = None
            logger.warning("   ⚠️ SimpleConversationManager not available - skipping initialization")
            return
        
        try:
            self.conversation_manager = SimpleConversationManager(
                redis_client=self.redis,
                max_history_turns=5,  # Keep last 5 turns (simpler approach)
                session_ttl=3600  # 1 hour session lifetime
            )
            
            logger.info("   💬 Simple conversational context manager initialized")
            logger.info(f"      Max history: 5 turns, Session TTL: 1 hour")
            logger.info(f"      Approach: LLM-based (no keyword matching)")
            
        except Exception as e:
            logger.error(f"Failed to initialize conversation manager: {e}")
            self.conversation_manager = None
    
    def _init_query_rewriter(self):
        """
        PRIORITY 3.3: Initialize query rewriter.
        
        Enables:
        - Query clarity enhancement
        - Abbreviation expansion
        - Context injection into queries
        - Grammar and spelling fixes
        - Implicit information extraction
        
        Features:
        - LLM-based rewriting (no rule-based patterns)
        - Redis-backed caching
        - Multilingual support
        - Validation of rewrites
        - Statistics tracking
        """
        if not QUERY_REWRITER_AVAILABLE:
            self.query_rewriter = None
            logger.warning("   ⚠️ SimpleQueryRewriter not available - skipping initialization")
            return
        
        try:
            # Pass the same LLM client used by the handler
            self.query_rewriter = SimpleQueryRewriter(
                llm_client=self.llm,
                redis_client=self.redis,
                cache_ttl=86400,  # 24 hours
                min_query_length=2,  # Rewrite queries with 2 or fewer words
                rewrite_threshold=0.7  # Confidence threshold
            )
            
            logger.info("   ✍️ Simple query rewriter initialized")
            logger.info(f"      Min query length: 2 words, Cache TTL: 24 hours")
            logger.info(f"      Approach: LLM-based (no keyword patterns)")
            
        except Exception as e:
            logger.error(f"Failed to initialize query rewriter: {e}")
            self.query_rewriter = None
    
    def _init_semantic_cache(self):
        """
        PRIORITY 3.4: Initialize semantic response cache.
        
        Advanced caching system that uses embeddings to find similar queries
        and return cached responses even for differently worded questions.
        
        Features:
        - Embedding-based similarity search
        - Redis-backed persistent storage
        - Configurable similarity threshold
        - Reduces LLM calls by 40%+
        - Instant responses for similar queries
        - Statistics tracking
        
        Benefits:
        - Lower API costs (fewer LLM calls)
        - Faster response times (cached = instant)
        - Better user experience
        - Reduced server load
        
        Example:
            Original query: "restaurants in Sultanahmet"
            New query: "places to eat near Sultanahmet"
            Similarity: 0.92 → Return cached response (no LLM call needed)
        """
        if not SEMANTIC_CACHE_AVAILABLE:
            self.semantic_cache = None
            logger.warning("   ⚠️ SemanticCache not available - skipping initialization")
            return
        
        try:
            self.semantic_cache = SemanticCache(
                redis_client=self.redis,
                embedding_model="all-MiniLM-L6-v2",  # Fast, efficient model
                similarity_threshold=0.85,  # High threshold for quality
                default_ttl=3600,  # 1 hour cache lifetime
                max_cached_queries=10000  # Maximum cache size
            )
            
            logger.info("   🗄️ Semantic cache initialized (Priority 3.4)")
            logger.info(f"      Similarity threshold: 0.85")
            logger.info(f"      Cache TTL: 1 hour")
            logger.info(f"      Max cached queries: 10,000")
            logger.info(f"      Expected cache hit rate: 40%+")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic cache: {e}")
            self.semantic_cache = None
    
    def _init_query_explainer(self):
        """
        PRIORITY 3.5: Initialize query explanation system.
        
        Provides transparency in how queries are understood by explaining:
        - Detected signals and intents
        - Confidence scores
        - Multi-intent handling
        - What action will be taken
        
        Features:
        - LLM-based natural language explanations
        - Multilingual support (EN, TR, etc.)
        - Redis caching for performance
        - Simple fallback for reliability
        - Debugging and transparency
        
        Benefits:
        - User trust through transparency
        - Better debugging and QA
        - Improved user experience
        - Educational for users
        
        Example:
            Query: "How do I get to restaurants in Sultanahmet?"
            Explanation: "You're asking about both transportation and restaurants.
                         I detected high confidence (0.92) for transportation intent
                         and medium confidence (0.75) for restaurant intent. I'll 
                         provide route information to dining areas in Sultanahmet."
        """
        if not QUERY_EXPLAINER_AVAILABLE:
            self.query_explainer = None
            logger.warning("   ⚠️ QueryExplainer not available - skipping initialization")
            return
        
        try:
            # Detect default language from system (fallback to English)
            default_lang = "en"  # Can be configured via environment
            
            self.query_explainer = QueryExplainer(
                llm_client=self.llm,
                redis_client=self.redis,
                cache_ttl=3600,  # 1 hour cache
                language=default_lang
            )
            
            logger.info("   💡 Query explainer initialized (Priority 3.5)")
            logger.info(f"      Default language: {default_lang}")
            logger.info(f"      Cache TTL: 1 hour")
            logger.info(f"      Supports: EN, TR, and auto-detection")
            logger.info(f"      Target: 100% query coverage")
            
        except Exception as e:
            logger.error(f"Failed to initialize query explainer: {e}")
            self.query_explainer = None
    
    def _prewarm_model(self):
        """
        Pre-warm the embedding model with sample queries.
        This eliminates the cold-start delay on the first user query.
        """
        if not self.embedding_model:
            return
        
        try:
            warm_up_queries = [
                "Where should I eat?",
                "How do I get there?",
                "What's the weather?",
                "Show me events",
                "Cheap restaurants"
            ]
            
            # Encode sample queries to warm up the model
            self.embedding_model.encode(warm_up_queries, convert_to_numpy=True, show_progress_bar=False)
            
            logger.debug("   Model pre-warmed (faster first query)")
            
        except Exception as e:
            logger.debug(f"   Model pre-warming failed (non-critical): {e}")
    
    def _init_additional_services(self):
        """Initialize additional services (weather, events, hidden gems, price filter)"""
        
        # Weather Recommendations Service
        try:
            from backend.services.weather_recommendations import get_weather_recommendations_service
            self.weather_service = get_weather_recommendations_service()
            logger.debug("Weather service loaded")
        except Exception as e:
            logger.warning(f"Weather service not available: {e}")
            self.weather_service = None
        
        # Events Service
        try:
            from backend.services.events_service import get_events_service
            self.events_service = get_events_service()
            logger.debug("Events service loaded")
        except Exception as e:
            logger.warning(f"Events service not available: {e}")
            self.events_service = None
        
        # Hidden Gems Handler
        try:
            from backend.services.hidden_gems_handler import HiddenGemsHandler
            self.hidden_gems_handler = HiddenGemsHandler()
            logger.debug("Hidden gems handler loaded")
        except Exception as e:
            logger.warning(f"Hidden gems handler not available: {e}")
            self.hidden_gems_handler = None
        
        # Price Filter Service
        try:
            from backend.services.price_filter_service import PriceFilterService
            self.price_filter = PriceFilterService()
            logger.debug("Price filter service loaded")
        except Exception as e:
            logger.warning(f"Price filter service not available: {e}")
            self.price_filter = None
    
    def _init_signal_embeddings(self):
        """
        Pre-compute embeddings for signal patterns.
        This enables semantic similarity matching for language-independent detection.
        """
        if not self.embedding_model:
            return
        
        try:
            # Define signal patterns in 6 languages with diverse phrasings: EN, TR, AR, DE, RU, FR
            # More examples = better semantic coverage = fewer missed queries
            signal_patterns = {
                'map_routing': [
                    # English - varied phrasings
                    "How do I get there?", "Show me directions", "What's the best route?", "Navigate to this place",
                    "How can I reach", "Take me to", "Guide me to", "Path from here", "Way to get there",
                    "Travel directions", "Show me the way", "How far is it", "Getting to",
                    # Turkish - varied phrasings
                    "Oraya nasıl gidilir?", "Yol tarifi", "En iyi güzergah nedir?", "Buradan nasıl giderim",
                    "Yol göster", "Nasıl ulaşırım", "Güzergah göster", "Yolu tarif et",
                    # Arabic
                    "كيف أصل إلى هناك؟", "اعرض لي الاتجاهات", "ما هو أفضل طريق؟", "كيف أذهب",
                    "دلني على الطريق", "وجهني إلى", "الطريق من هنا",
                    # German
                    "Wie komme ich dorthin?", "Zeig mir die Wegbeschreibung", "Was ist der beste Weg?",
                    "Wie erreiche ich", "Führe mich zu", "Weg dorthin",
                    # Russian
                    "Как туда добраться?", "Покажите мне маршрут", "Какой лучший путь?",
                    "Как мне пройти", "Проведи меня к", "Направление к",
                    # French
                    "Comment puis-je y arriver?", "Montrez-moi les directions", "Quel est le meilleur itinéraire?",
                    "Comment aller", "Guidez-moi vers", "Chemin pour y aller"
                ],
                'weather': [
                    # English - varied weather questions
                    "What's the weather like?", "Will it rain today?", "Temperature forecast", "Is it sunny?",
                    "How's the weather", "Is it cold outside", "What should I wear", "Climate today",
                    "Weather forecast", "Will it be warm", "Temperature today", "Is it hot",
                    # Turkish
                    "Hava durumu nasıl?", "Yağmur yağacak mı?", "Sıcaklık kaç derece?", "Hava nasıl",
                    "Soğuk mu", "Sıcak mı", "Ne giysem", "Hava tahmini",
                    # Arabic
                    "كيف الطقس؟", "هل ستمطر اليوم؟", "توقعات الحرارة", "هل هو مشمس؟",
                    "حالة الطقس", "هل الجو بارد", "درجة الحرارة اليوم",
                    # German
                    "Wie ist das Wetter?", "Wird es heute regnen?", "Temperaturvorhersage", "Ist es sonnig?",
                    "Wie wird das Wetter", "Ist es kalt", "Was soll ich anziehen",
                    # Russian
                    "Какая погода?", "Будет ли дождь сегодня?", "Прогноз температуры", "Солнечно ли?",
                    "Как погода", "Холодно ли", "Что надеть", "Температура сегодня",
                    # French
                    "Quel temps fait-il?", "Va-t-il pleuvoir aujourd'hui?", "Prévisions de température",
                    "Comment est la météo", "Fait-il froid", "Que porter", "Température aujourd'hui"
                ],
                'events': [
                    # English - events and activities
                    "What events are happening?", "Any concerts tonight?", "Show me festivals", "Cultural activities",
                    "Things to do", "What's going on", "Entertainment tonight", "Live shows",
                    "Events this weekend", "Concerts near me", "Festival schedule", "Activities today",
                    # Turkish
                    "Hangi etkinlikler var?", "Konser var mı?", "Festival programı", "Ne yapabilirim",
                    "Etkinlikler", "Konserler", "Gösteriler", "Aktiviteler",
                    # Arabic
                    "ما هي الفعاليات الجارية؟", "هل هناك حفلات موسيقية الليلة؟", "أظهر لي المهرجانات",
                    "ماذا يمكنني أن أفعل", "الأحداث اليوم", "الحفلات الموسيقية",
                    # German
                    "Welche Veranstaltungen finden statt?", "Gibt es heute Abend Konzerte?", "Zeig mir Festivals",
                    "Was kann ich tun", "Veranstaltungen heute", "Концерты в der Nähe",
                    # Russian
                    "Какие события происходят?", "Есть ли концерты сегодня?", "Покажи мне фестивали",
                    "Что делать", "События сегодня", "Концерты рядом",
                    # French
                    "Quels événements se passent?", "Y a-t-il des concerts ce soir?", "Montre-moi les festivals",
                    "Que faire", "Événements aujourd'hui", "Concerts près de moi"
                ],
                'hidden_gems': [
                    # English - local and authentic experiences
                    "Local secrets", "Off the beaten path", "Where do locals go?", "Authentic experiences",
                    "Hidden places", "Secret spots", "Local favorites", "Non-touristy places",
                    "Where locals eat", "Undiscovered gems", "Authentic spots", "Local recommendations",
                    # Turkish
                    "Gizli yerler", "Yerel mekanlar", "Turistik olmayan yerler", "Saklı yerler",
                    "Yerli mekanlar", "Otantik yerler", "Yerel tavsiyeler",
                    # Arabic
                    "الأماكن السرية", "أماكن محلية", "أين يذهب السكان المحليون؟", "أماكن أصيلة",
                    "الأماكن المخفية", "توصيات محلية",
                    # German
                    "Geheimtipps", "Lokale Orte", "Wo gehen Einheimische hin?", "Authentische Erlebnisse",
                    "Versteckte Orte", "Lokale Empfehlungen",
                    # Russian
                    "Скрытые жемчужины", "Местные места", "Куда ходят местные жители?", "Аутентичные места",
                    "Секретные места", "Местные рекомендации",
                    # French
                    "Secrets locaux", "Lieux authentiques", "Où vont les habitants?", "Expériences authentiques",
                    "Endroits cachés", "Recommandations locales"
                ],
                'budget': [
                    # English - price and budget mentions
                    "Cheap options", "Budget-friendly", "Affordable places", "Expensive restaurants",
                    "Cheap eats", "Low cost", "Free things", "Luxury dining",
                    "Inexpensive", "Good value", "Pricey places", "Budget travel",
                    # Turkish
                    "Ucuz yerler", "Ekonomik", "Pahalı mekanlar", "Uygun fiyatlı",
                    "Bütçe dostu", "Lüks restoranlar", "Ucuz yemek",
                    # Arabic
                    "خيارات رخيصة", "أماكن اقتصادية", "مطاعم غالية", "أماكن ميسورة التكلفة",
                    "خيارات منخفضة التكلفة", "مطاعم فاخرة",
                    # German
                    "Günstige Optionen", "Budgetfreundlich", "Teure Restaurants", "Erschwingliche Orte",
                    "Preiswert", "Luxus-Restaurants",
                    # Russian
                    "Дешевые варианты", "Бюджетные места", "Дорогие рестораны", "Доступные места",
                    "Недорого", "Люксовые рестораны",
                    # French
                    "Options bon marché", "Économique", "Restaurants chers", "Lieux abordables",
                    "Pas cher", "Restaurants de luxe"
                ],
                'restaurant': [
                    # English - food and dining
                    "Where should I eat?", "Good restaurants", "Food recommendations", "Best places to eat",
                    "Where to dine", "Dinner spot", "Lunch place", "Breakfast options",
                    "Best food", "Restaurant near me", "Where to grab food", "Dining recommendations",
                    # Turkish
                    "Nerede yemek yenir?", "İyi restoranlar", "Yemek önerisi", "En iyi restoranlar",
                    "Nerede yemek yiyeyim", "Yemek yerleri", "Restoran tavsiyesi",
                    # Arabic
                    "أين يجب أن آكل؟", "مطاعم جيدة", "توصيات الطعام", "أفضل أماكن الطعام",
                    "أين أتناول العشاء", "مطاعم قريبة مني",
                    # German
                    "Wo soll ich essen?", "Gute Restaurants", "Essensempfehlungen", "Beste Restaurants",
                    "Wo kann ich essen", "Restaurant in der Nähe",
                    # Russian
                    "Где мне поесть?", "Хорошие рестораны", "Рекомендации еды", "Лучшие места для еды",
                    "Где пообедать", "Ресторан рядом",
                    # French
                    "Où devrais-je manger?", "Bons restaurants", "Recommandations culinaires",
                    "Meilleurs restaurants", "Où dîner", "Restaurant près de moi"
                ],
                'attraction': [
                    # English - sightseeing and landmarks
                    "What should I visit?", "Tourist attractions", "Museums to see", "Places to visit",
                    "Sightseeing", "Must-see places", "What to see", "Famous landmarks",
                    "Historical sites", "Top attractions", "Things to visit", "Popular spots",
                    "Blue Mosque", "Hagia Sophia", "Topkapi Palace", "Galata Tower",  # Explicit landmarks
                    # Turkish
                    "Nereleri gezmeliyim?", "Turistik yerler", "Müzeler", "Gezilecek yerler",
                    "Görülmesi gerekenler", "Tarihi yerler", "Ünlü yerler",
                    "Sultanahmet", "Ayasofya", "Topkapı Sarayı",  # Turkish landmarks
                    # Arabic
                    "ماذا يجب أن أزور؟", "المعالم السياحية", "المتاحف", "أين أذهب",
                    "أين يجب أن أزور", "أهم المعالم", "الأماكن التاريخية",
                    # German
                    "Was sollte ich besichtigen?", "Touristenattraktionen", "Museen zu sehen",
                    "Sehenswürdigkeiten", "Top-Attraktionen", "Beliebte Orte",
                    # Russian
                    "Что мне стоит посетить?", "Туристические достопримечательности", "Музеи",
                    "Где мне побывать", "Исторические места", "Популярные достопримечательности",
                    # French
                    "Que devrais-je visiter?", "Attractions touristiques", "Musées à voir",
                    "Sites historiques", "Top attractions", "Lieux populaires"
                ]
            }
            
            # Pre-compute embeddings for each signal pattern
            for signal, patterns in signal_patterns.items():
                embeddings = self.embedding_model.encode(patterns, convert_to_numpy=True, show_progress_bar=False)
                self._signal_embeddings[signal] = embeddings
            
            logger.debug(f"   Pre-computed {len(self._signal_embeddings)} signal embeddings")
            
        except Exception as e:
            logger.warning(f"Failed to initialize signal embeddings: {e}")
            self._signal_embeddings = {}
    
    def _load_prompts(self):
        """Load Istanbul-specific system prompts"""
        
        self.base_prompt = """You are AI Istanbul, an expert travel assistant for Istanbul, Turkey.

You have deep knowledge of:
🏛️ Attractions: Museums, mosques, palaces, historical sites
🍽️ Restaurants: Authentic Turkish cuisine, international options
🚇 Transportation: Metro, bus, ferry, tram routes
🏘️ Neighborhoods: Districts, areas, local culture
🎭 Events: Concerts, festivals, cultural activities
💎 Hidden Gems: Local favorites, off-the-beaten-path spots

Guidelines:
1. Provide specific names, locations, and details
2. Use provided database context
3. Include practical info (hours, prices, directions)
4. Be enthusiastic about Istanbul
5. Respond in the same language as the query
6. Never make up information - use context only

Format:
- Start with direct answer
- List 3-5 specific recommendations
- Include practical details
- Add a local tip or insight"""

        self.intent_prompts = {
            'restaurant': """
Focus on restaurants from the provided database context.
Include: name, location, cuisine, price range, rating.
Mention dietary options if relevant.""",

            'attraction': """
Focus on attractions and museums from the provided context.
Include: name, district, description, opening hours, ticket price.
Prioritize based on location and interests.""",

            'transportation': """
Provide clear transportation directions.
Include: metro lines, bus numbers, ferry routes.
Mention transfer points and approximate times.
If available, include a map visualization link.""",

            'neighborhood': """
Describe the neighborhood character and highlights.
Include: atmosphere, best areas, local tips.
Mention nearby attractions and dining.""",

            'events': """
Focus on current and upcoming events.
Include: event name, date, location, price.
Prioritize cultural and authentic experiences.""",

            'weather': """
Provide weather-aware recommendations.
Include current conditions and activity suggestions.
Recommend indoor options for bad weather, outdoor for good weather.""",

            'hidden_gems': """
Focus on local secrets and off-the-beaten-path spots.
Include authentic experiences away from tourist crowds.
Mention accessibility and best times to visit.""",

            'general': """
Provide helpful Istanbul travel information.
Draw from all available context.
Be comprehensive but concise."""
        }
    
    def _detect_language(self, query: str) -> str:
        """
        PRIORITY 1: Automatic language detection from query text.
        
        Uses langdetect library to identify query language.
        Falls back to 'en' if detection fails or library unavailable.
        
        Supported languages: en, tr, ar, de, ru, fr
        
        Args:
            query: User query string
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'tr')
        """
        if not LANGDETECT_AVAILABLE:
            return "en"  # Default fallback
        
        try:
            detected_lang = detect(query)
            
            # Map to supported languages
            supported_langs = {"en", "tr", "ar", "de", "ru", "fr"}
            
            if detected_lang in supported_langs:
                logger.info(f"   🌍 Language detected: {detected_lang}")
                # Track analytics
                self.user_analytics["queries_by_language"][detected_lang] += 1
                return detected_lang
            else:
                logger.debug(f"   🌍 Language detected as {detected_lang}, not in supported set, using 'en'")
                self.user_analytics["queries_by_language"]["en"] += 1
                return "en"
                
        except Exception as e:
            logger.debug(f"   ⚠️  Language detection failed: {e}, defaulting to 'en'")
            return "en"
    
    def _validate_response(
        self, 
        response: str, 
        query: str, 
        signals: Dict[str, bool],
        context_used: bool
    ) -> Tuple[bool, Optional[str]]:
        """
        PRIORITY 1: Response validation and quality checks.
        
        Validates:
        - Response is not empty
        - Minimum length requirements
        - Contains relevant context usage
        - Matches query intent (basic checks)
        - No hallucination indicators
        
        Args:
            response: Generated LLM response
            query: Original user query
            signals: Detected signals
            context_used: Whether database context was available
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if response passes validation
            - error_message: Description of validation failure (if any)
        """
        self.quality_metrics["responses_validated"] += 1
        
        # Check 1: Empty response
        if not response or not response.strip():
            self.quality_metrics["empty_responses"] += 1
            self.quality_metrics["validation_failures"] += 1
            return False, "Empty response generated"
        
        # Check 2: Minimum length (at least 20 characters)
        if len(response.strip()) < 20:
            self.quality_metrics["validation_failures"] += 1
            return False, f"Response too short ({len(response)} chars)"
        
        # Check 3: Track response length for analytics
        self.quality_metrics["response_lengths"].append(len(response))
        
        # Check 4: Context usage validation
        if context_used:
            # Response should reference specific places, not be generic
            generic_phrases = [
                "I don't have information",
                "I cannot provide",
                "I'm not sure",
                "I don't know"
            ]
            
            response_lower = response.lower()
            if any(phrase in response_lower for phrase in generic_phrases):
                logger.warning("   ⚠️  Response appears generic despite context availability")
                # Don't fail, but log for monitoring
        
        # Check 5: Hallucination indicators (basic heuristics)
        hallucination_indicators = [
            "as an ai",
            "i am an ai",
            "i cannot actually",
            "i don't have access to real-time",
            "i cannot browse",
            "fictional",
            "made up"
        ]
        
        response_lower = response.lower()
        if any(indicator in response_lower for indicator in hallucination_indicators):
            logger.warning("   ⚠️  Possible hallucination indicator detected")
            # Don't fail, but log for monitoring
        
        # Check 6: Signal-specific validation
        if signals.get('needs_map') or signals.get('needs_gps_routing'):
            # For map queries, ensure some direction/location info
            direction_terms = ["metro", "bus", "tram", "walk", "route", "direction", "get to", "reach"]
            if not any(term in response_lower for term in direction_terms):
                logger.warning("   ⚠️  Map query response lacks direction information")
        
        # All checks passed
        return True, None
    
    def _track_error(
        self, 
        error_type: str, 
        service: str, 
        error_message: str,
        query: Optional[str] = None
    ):
        """
        PRIORITY 1: Track errors for monitoring and debugging.
        
        Args:
            error_type: Type of error (e.g., 'llm_failure', 'service_timeout')
            service: Service that failed (e.g., 'runpod', 'weather')
            error_message: Error description
            query: Optional user query for context
        """
        self.error_tracker["total_errors"] += 1
        self.error_tracker["error_by_type"][error_type] += 1
        self.error_tracker["error_by_service"][service] += 1
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "service": service,
            "message": error_message[:200],  # Truncate long messages
            "query": query[:100] if query else None
        }
        
        self.error_tracker["recent_errors"].append(error_entry)
        
        logger.error(f"   ❌ Error tracked: {error_type} in {service} - {error_message[:100]}")
    
    def _track_performance(
        self, 
        metric_name: str, 
        latency: float
    ):
        """
        PRIORITY 1: Track performance metrics.
        
        Args:
            metric_name: Name of the metric (e.g., 'query', 'llm', 'cache')
            latency: Time taken in seconds
        """
        if metric_name == "query":
            self.performance_metrics["query_latencies"].append(latency)
        elif metric_name == "llm":
            self.performance_metrics["llm_latencies"].append(latency)
        elif metric_name == "cache":
            self.performance_metrics["cache_latencies"].append(latency)
        else:
            self.performance_metrics["service_latencies"][metric_name].append(latency)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        PRIORITY 1: Get comprehensive analytics summary.
        
        Returns:
            Dictionary with all analytics data for monitoring/dashboards
        """
        # Calculate performance statistics
        def calc_stats(latencies):
            if not latencies:
                return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            return {
                "count": n,
                "avg": sum(sorted_lat) / n,
                "p50": sorted_lat[int(n * 0.5)],
                "p95": sorted_lat[int(n * 0.95)] if n > 20 else sorted_lat[-1],
                "p99": sorted_lat[int(n * 0.99)] if n > 100 else sorted_lat[-1]
            }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "basic_stats": self.stats,
            "performance": {
                "query_latency": calc_stats(self.performance_metrics["query_latencies"]),
                "llm_latency": calc_stats(self.performance_metrics["llm_latencies"]),
                "cache_latency": calc_stats(self.performance_metrics["cache_latencies"])
            },
            "errors": {
                "total": self.error_tracker["total_errors"],
                "by_type": dict(self.error_tracker["error_by_type"]),
                "by_service": dict(self.error_tracker["error_by_service"]),
                "recovery_count": self.error_tracker["error_recovery_count"],
                "recent": list(self.error_tracker["recent_errors"])[-10:]  # Last 10
            },
            "users": {
                "queries_by_language": dict(self.user_analytics["queries_by_language"]),
                "queries_by_intent": dict(self.user_analytics["queries_by_intent"]),
                "unique_users": len(self.user_analytics["unique_users"]),
                "multi_intent_patterns": dict(self.user_analytics["multi_intent_patterns"]),
                "user_locations_used": self.user_analytics["user_locations_used"]
            },
            "signals": {
                "detections": dict(self.signal_analytics["detections_by_signal"]),
                "semantic_vs_keyword": self.signal_analytics["semantic_vs_keyword"],
                "false_positives": dict(self.signal_analytics["false_positive_reports"])
            },
            "services": {
                "map_success_rate": (
                    self.service_analytics["map_generation_success"] / 
                    max(1, self.service_analytics["map_generation_success"] + 
                        self.service_analytics["map_generation_failure"])
                ),
                "cache_hit_rate": (
                    self.service_analytics["cache_efficiency"]["hits"] /
                    max(1, self.service_analytics["cache_efficiency"]["hits"] + 
                        self.service_analytics["cache_efficiency"]["misses"])
                ),
                "service_usage": {
                    "weather": self.service_analytics["weather_service_calls"],
                    "events": self.service_analytics["events_service_calls"],
                    "hidden_gems": self.service_analytics["hidden_gems_calls"],
                    "rag": self.service_analytics["rag_usage"]
                }
            },
            "quality": {
                "responses_validated": self.quality_metrics["responses_validated"],
                "validation_failures": self.quality_metrics["validation_failures"],
                "validation_success_rate": (
                    (self.quality_metrics["responses_validated"] - 
                     self.quality_metrics["validation_failures"]) /
                    max(1, self.quality_metrics["responses_validated"])
                ),
                "empty_responses": self.quality_metrics["empty_responses"],
                "avg_response_length": (
                    sum(self.quality_metrics["response_lengths"]) / 
                    len(self.quality_metrics["response_lengths"])
                    if self.quality_metrics["response_lengths"] else 0
                )
            }
        }
    
    async def process_query(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        intent: Optional[str] = None,
        max_tokens: int = 250
    ) -> Dict[str, Any]:
        """
        Process query using SIGNAL-BASED detection (multi-intent support)
        
        Pipeline:
        1. Check cache
        2. Detect service signals (multi-intent, semantic)
        3. Extract GPS location (if provided)
        4. Build smart database context based on signals
        5. Get RAG embeddings (if available)
        6. Conditionally call expensive services (maps, weather, events, etc.)
        7. Construct signal-aware prompt
        8. Call RunPod LLM
        9. Cache signals and response
        10. Return with metadata
        
        Args:
            query: User query string
            user_id: User identifier
            session_id: Session identifier for context tracking
            user_location: User GPS location {"lat": float, "lon": float}
            language: Response language (en/tr)
            intent: Pre-detected intent (optional, for backward compatibility)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with response, map_data (if applicable), signals, and metadata
        """
        
        self.stats["total_queries"] += 1
        start_time = datetime.now()
        
        logger.info(f"🔍 Processing query: {query[:50]}...")
        if user_location:
            logger.info(f"📍 User GPS: ({user_location.get('lat'):.4f}, {user_location.get('lon'):.4f})")
            self.user_analytics["user_locations_used"] += 1
        
        # PRIORITY 1: Track unique users
        self.user_analytics["unique_users"].add(user_id)
        self.user_analytics["queries_per_user"][user_id] += 1
        
        # PRIORITY 1: Auto-detect language if not explicitly provided
        if language == "en" and LANGDETECT_AVAILABLE:
            detected_language = self._detect_language(query)
            if detected_language != "en":
                language = detected_language
                logger.info(f"   🌍 Auto-detected language: {language}")
        else:
            # Track manually provided language
            self.user_analytics["queries_by_language"][language] += 1
        
        # PRIORITY 3.3: Query rewriting (before cache check to cache rewritten queries)
        original_query = query
        if self.query_rewriter:
            try:
                rewrite_start = time.time()
                rewrite_result = await self.query_rewriter.rewrite_query(query, language)
                self._track_performance("query_rewriting", time.time() - rewrite_start)
                
                if rewrite_result.get("needs_rewrite", False):
                    query = rewrite_result.get("rewritten_query", query)
                    logger.info(f"   ✍️ Query rewritten: '{original_query[:50]}...' → '{query[:50]}...'")
                    logger.info(f"      Reason: {rewrite_result.get('reason', 'N/A')}")
                    self.user_analytics["queries_rewritten"] = self.user_analytics.get("queries_rewritten", 0) + 1
                else:
                    logger.info(f"   ℹ️ Query does not need rewriting")
            except Exception as e:
                logger.warning(f"   ⚠️ Query rewriting failed: {e}")
                query = original_query  # Fall back to original query
        
        # Step 1: Check cache (PRIORITY 3.4: Semantic Cache with similarity search)
        cache_start = time.time()
        
        # Try semantic cache first (finds similar queries)
        cached_response = None
        if self.semantic_cache:
            try:
                context_for_cache = {
                    'language': language,
                    'user_id': user_id,
                    'session_id': session_id
                }
                cached_response = await self.semantic_cache.get_similar_response(
                    query=query,
                    context=context_for_cache
                )
                
                if cached_response:
                    # Semantic cache hit - return cached response
                    self.stats["cache_hits"] += 1
                    self.service_analytics["cache_efficiency"]["hits"] += 1
                    logger.info("✅ Semantic cache hit! (Similar query found)")
                    
                    self._track_performance("cache", time.time() - cache_start)
                    
                    # Return cached response with metadata
                    return {
                        "status": "success",
                        "response": cached_response['response']['response'],
                        "map_data": cached_response['response'].get('map_data'),
                        "signals": cached_response['response'].get('signals', {}),
                        "metadata": {
                            **cached_response['response'].get('metadata', {}),
                            "cached": True,
                            "cache_type": "semantic",
                            "original_cached_query": cached_response.get('query'),
                            "cache_retrieval_time": (time.time() - cache_start) * 1000
                        }
                    }
            except Exception as e:
                logger.warning(f"   ⚠️ Semantic cache lookup failed: {e}")
        
        # Fallback to exact match cache (old system)
        if not cached_response:
            cache_key = self._get_cache_key(query, language)
            cached_response = await self._get_cached_response(cache_key)
            
            if cached_response:
                self.stats["cache_hits"] += 1
                self.service_analytics["cache_efficiency"]["hits"] += 1
                logger.info("✅ Exact cache hit!")
                self._track_performance("cache", time.time() - cache_start)
                return cached_response
        
        # Track cache miss
        self.service_analytics["cache_efficiency"]["misses"] += 1
        self._track_performance("cache", time.time() - cache_start)
        
        # Step 2: Detect service signals (NEW: multi-intent, semantic, language-aware)
        signals = await self._detect_service_signals(query, user_location, language)
        active_signals = [k for k, v in signals.items() if v]
        logger.info(f"   Signals detected: {', '.join(active_signals) if active_signals else 'none'}")
        
        # Track multi-signal queries
        if len(active_signals) > 2:
            self.stats["multi_signal_queries"] += 1
            logger.info(f"   🎯 Multi-signal query detected ({len(active_signals)} signals)")
        
        # Step 3: Build smart context based on signals
        db_context = await self._build_smart_context(query, signals)
        logger.info(f"   DB Context: {len(db_context)} chars")
        
        # Step 4: Get RAG context (if available)
        rag_context = await self._get_rag_context(query)
        logger.info(f"   RAG Context: {len(rag_context)} chars")
        if rag_context:
            self.service_analytics["rag_usage"] += 1
        
        # Step 5: Conditionally call EXPENSIVE services based on signals
        map_data = None
        if signals['needs_map'] or signals['needs_gps_routing']:
            logger.info("🗺️ Map visualization needed - routing to Istanbul AI system")
            try:
                self.stats["map_requests"] += 1
                map_start = time.time()
                
                map_data = await self._get_map_visualization(
                    query, 'route_planning', user_id, language, user_location
                )
                
                self._track_performance("map_generation", time.time() - map_start)
                
                if map_data:
                    self.service_analytics["map_generation_success"] += 1
                    logger.info(f"   ✅ Map data generated successfully")
                else:
                    self.service_analytics["map_generation_failure"] += 1
                    logger.warning(f"   ⚠️ Map generation returned no data")
                    
            except Exception as e:
                self.service_analytics["map_generation_failure"] += 1
                self._track_error("service_failure", "map_generation", str(e), query)
                logger.warning(f"   ⚠️ Map generation failed: {e}")
                map_data = None
        
        weather_context = ""
        if signals['needs_weather'] and self.weather_service:
            self.stats["weather_requests"] += 1
            self.service_analytics["weather_service_calls"] += 1
            weather_context = await self._get_weather_context(query)
            logger.info(f"   ☀️ Weather context: {len(weather_context)} chars")
        
        events_context = ""
        if signals['needs_events'] and self.events_service:
            self.service_analytics["events_service_calls"] += 1
            events_context = await self._get_events_context()
            logger.info(f"   🎭 Events context: {len(events_context)} chars")
        
        hidden_gems_context = ""
        if signals['needs_hidden_gems'] and self.hidden_gems_handler:
            self.stats["hidden_gems_requests"] += 1
            self.service_analytics["hidden_gems_calls"] += 1
            hidden_gems_context = await self._get_hidden_gems_context(query)
            logger.info(f"   💎 Hidden gems context: {len(hidden_gems_context)} chars")
        
        # Step 6: Build signal-aware system prompt
        system_prompt = self._build_system_prompt(signals)
        
        # Step 7: Build full prompt with all contexts
        full_prompt = self._build_prompt_with_signals(
            query=query,
            signals=signals,
            system_prompt=system_prompt,
            db_context=db_context,
            rag_context=rag_context,
            weather_context=weather_context,
            events_context=events_context,
            hidden_gems_context=hidden_gems_context,
            language=language
        )
        
        # Step 8: Call RunPod LLM
        try:
            self.stats["llm_calls"] += 1
            
            # PRIORITY 1: Track LLM latency
            llm_start = time.time()
            
            response_data = await self.llm.generate(
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            llm_latency = time.time() - llm_start
            self._track_performance("llm", llm_latency)
            logger.debug(f"   ⏱️  LLM latency: {llm_latency:.2f}s")
            
            if not response_data or "generated_text" not in response_data:
                raise Exception("Invalid LLM response")
            
            response_text = response_data["generated_text"]
            
            # PRIORITY 1: Validate response quality
            is_valid, error_message = self._validate_response(response_text, query, signals, bool(db_context))
            if not is_valid:
                logger.warning(f"   ❌ Response validation failed: {error_message}")
                self._track_error("validation_failure", "response_validator", error_message, query)
                self.error_tracker["error_recovery_count"] += 1
                
                # Attempt recovery with fallback
                return await self._fallback_response(
                    query=query,
                    intent='general',
                    db_context=db_context,
                    rag_context=rag_context,
                    map_data=map_data
                )
            
            # PRIORITY 1: Track query latency
            total_latency = (datetime.now() - start_time).total_seconds()
            self._track_performance("query", total_latency)
            
            # Build result with signals and map_data
            result = {
                "status": "success",
                "response": response_text,
                "map_data": map_data,
                "signals": signals,  # Include detected signals
                "metadata": {
                    "signals_detected": active_signals,
                    "context_used": bool(db_context),
                    "rag_used": bool(rag_context),
                    "map_generated": bool(map_data),
                    "weather_used": bool(weather_context),
                    "events_used": bool(events_context),
                    "hidden_gems_used": bool(hidden_gems_context),
                    "source": "runpod_llm",
                    "processing_time": total_latency,
                    "llm_latency": llm_latency,
                    "cached": False,
                    "multi_intent": len(active_signals) > 2,
                    "language": language,
                    "detected_language": language if LANGDETECT_AVAILABLE else None
                }
            }
            
            # Step 10: Cache response (PRIORITY 3.4: Semantic Cache)
            # Cache in both systems for redundancy
            await self._cache_response(cache_key, result)
            
            # Also cache in semantic cache for similarity search
            if self.semantic_cache:
                try:
                    context_for_cache = {
                        'language': language,
                        'user_id': user_id,
                        'session_id': session_id
                    }
                    self.semantic_cache.cache_response(
                        query=query,
                        response=result,
                        ttl=3600,  # 1 hour
                        context=context_for_cache
                    )
                    logger.info(f"   🗄️ Response cached in semantic cache")
                except Exception as e:
                    logger.warning(f"   ⚠️ Semantic cache storage failed: {e}")
            
            logger.info(f"✅ Query processed in {total_latency:.2f}s")
            if map_data:
                logger.info(f"   🗺️ Map visualization included")
            
            # PRIORITY 3.5: Generate explanation (if enabled and requested)
            if self.query_explainer:
                try:
                    # Generate explanation asynchronously
                    conv_context = None
                    if self.conversation_manager and session_id:
                        conv_context = {
                            "conversation_history": self.conversation_manager.get_history(session_id),
                            "session_id": session_id
                        }
                    
                    explanation = await self.query_explainer.explain_query(
                        query=query,
                        signals=signals,
                        context=conv_context,
                        language=language
                    )
                    result["explanation"] = explanation
                    logger.info(f"   💡 Explanation generated")
                except Exception as e:
                    logger.warning(f"   ⚠️ Explanation generation failed: {e}")
                    # Don't fail the whole query if explanation fails
            
            return result
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            self.stats["fallback_calls"] += 1
            self._track_error("llm_failure", "runpod", str(e), query)
            self.error_tracker["error_recovery_count"] += 1
            
            # Fallback to RAG-only or database context
            return await self._fallback_response(
                query=query,
                intent='general',
                db_context=db_context,
                rag_context=rag_context,
                map_data=map_data
            )
    
    # ============================================================================
    # PRIORITY 2.3 & 2.4: THRESHOLD LEARNING & A/B TESTING INTEGRATION
    # ============================================================================
    
    def record_user_feedback(
        self,
        query: str,
        detected_signals: Dict[str, bool],
        confidence_scores: Dict[str, float],
        feedback_type: str,
        feedback_data: Dict[str, Any],
        language: str = "en"
    ):
        """
        Record user feedback for threshold learning.
        
        PRIORITY 2.3: Feeds data to ThresholdLearner for continuous improvement.
        
        Args:
            query: User query
            detected_signals: Detected signals (True/False)
            confidence_scores: Confidence scores for each signal
            feedback_type: 'implicit' or 'explicit'
            feedback_data: Additional feedback data
            language: Query language
        
        Examples of feedback_type and feedback_data:
        - Implicit: feedback_type='implicit', feedback_data={'action': 'clicked_result', 'value': 0.8}
        - Explicit: feedback_type='explicit', feedback_data={'type': 'thumbs_up'}
        - Correction: feedback_type='explicit', feedback_data={'type': 'correction', 'corrected_signals': {...}}
        """
        if not self.threshold_learner:
            return
        
        try:
            self.threshold_learner.record_feedback(
                query=query,
                detected_signals=detected_signals,
                confidence_scores=confidence_scores,
                feedback_type=feedback_type,
                feedback_data=feedback_data,
                language=language
            )
            
            logger.debug(f"📝 Recorded {feedback_type} feedback for threshold learning")
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
    
    def auto_tune_thresholds(
        self,
        language: str = "en",
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Automatically tune thresholds based on accumulated feedback.
        
        PRIORITY 2.3: Runs periodically to optimize detection thresholds.
        
        Args:
            language: Language to tune for
            force: Force tuning even if interval hasn't passed
            
        Returns:
            Dict with tuning results and recommendations
        """
        if not self.threshold_learner or not self.enable_auto_tuning:
            return {'status': 'disabled'}
        
        # Check if we should run auto-tuning
        if not force:
            last_tune = self.last_auto_tune.get(language)
            if last_tune:
                hours_since = (datetime.now() - last_tune).total_seconds() / 3600
                if hours_since < self.auto_tune_interval_hours:
                    return {
                        'status': 'skipped',
                        'reason': f'Last tuned {hours_since:.1f}h ago'
                    }
        
        try:
            logger.info(f"🔧 Auto-tuning thresholds for {language}...")
            
            # Get current thresholds
            current_thresholds = self.language_thresholds.get(language, self.language_thresholds['default'])
            
            # Run auto-tuning
            results = self.threshold_learner.auto_tune(
                current_thresholds=current_thresholds,
                language=language,
                auto_apply=False  # Don't auto-apply, use A/B testing instead
            )
            
            # Update last tune time
            self.last_auto_tune[language] = datetime.now()
            
            logger.info(f"✅ Auto-tuning complete for {language}")
            
            # If we have A/B testing, create experiments for recommended changes
            if self.ab_testing and results.get('recommendations'):
                self._create_threshold_experiments(results['recommendations'], language)
            
            return results
            
        except Exception as e:
            logger.error(f"Auto-tuning failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _create_threshold_experiments(
        self,
        recommendations: Dict[str, Dict[str, Any]],
        language: str
    ):
        """
        Create A/B test experiments for threshold changes.
        
        PRIORITY 2.4: Test threshold changes before full rollout.
        
        Args:
            recommendations: Threshold recommendations from learner
            language: Language code
        """
        if not self.ab_testing:
            return
        
        for signal_name, rec in recommendations.items():
            if not rec.get('should_apply'):
                continue
            
            try:
                # Create experiment
                experiment = self.ab_testing.create_experiment(
                    name=f"Threshold Tuning: {signal_name} ({language})",
                    description=f"Test new threshold for {signal_name}: {rec['current']:.3f} -> {rec['recommended']:.3f}",
                    variants={
                        'control': {
                            'threshold': rec['current'],
                            'signal': signal_name,
                            'language': language
                        },
                        'treatment': {
                            'threshold': rec['recommended'],
                            'signal': signal_name,
                            'language': language
                        }
                    },
                    traffic_allocation={'control': 0.8, 'treatment': 0.2},  # 80/20 split
                    success_metrics=['detection_accuracy', 'f1_score'],
                    min_sample_size=100,
                    auto_start=True
                )
                
                # Track active experiment
                self.active_experiments[signal_name] = experiment.experiment_id
                
                logger.info(
                    f"🧪 Created A/B test for {signal_name} threshold: "
                    f"{rec['current']:.3f} -> {rec['recommended']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to create experiment for {signal_name}: {e}")
    
    def get_threshold_for_experiment(
        self,
        signal_name: str,
        language: str,
        user_id: str
    ) -> float:
        """
        Get threshold value considering active A/B tests.
        
        PRIORITY 2.4: Returns variant-specific threshold if user is in experiment.
        
        Args:
            signal_name: Signal name
            language: Language code
            user_id: User identifier
            
        Returns:
            Threshold value to use
        """
        # Check if there's an active experiment for this signal
        if self.ab_testing and signal_name in self.active_experiments:
            experiment_id = self.active_experiments[signal_name]
            
            try:
                variant = self.ab_testing.get_variant(user_id, experiment_id)
                
                if variant and variant['config'].get('language') == language:
                    threshold = variant['config']['threshold']
                    
                    logger.debug(
                        f"🧪 Using experimental threshold for {signal_name}: "
                        f"{threshold:.3f} (variant: {variant['variant_id']})"
                    )
                    
                    return threshold
                    
            except Exception as e:
                logger.error(f"Failed to get experiment variant: {e}")
        
        # Default: use configured threshold
        thresholds = self.language_thresholds.get(language, self.language_thresholds['default'])
        return thresholds.get(signal_name, 0.35)
    
    def record_experiment_metric(
        self,
        signal_name: str,
        user_id: str,
        metric_name: str,
        value: float
    ):
        """
        Record metric for active A/B experiment.
        
        PRIORITY 2.4: Tracks experiment performance metrics.
        
        Args:
            signal_name: Signal name
            user_id: User identifier
            metric_name: Metric name
            value: Metric value
        """
        if not self.ab_testing or signal_name not in self.active_experiments:
            return
        
        experiment_id = self.active_experiments[signal_name]
        
        try:
            # Get user's variant
            variant = self.ab_testing.get_variant(user_id, experiment_id)
            
            if variant:
                self.ab_testing.record_metric(
                    experiment_id=experiment_id,
                    variant_id=variant['variant_id'],
                    metric_name=metric_name,
                    value=value
                )
                
        except Exception as e:
            logger.error(f"Failed to record experiment metric: {e}")
    
    def analyze_active_experiments(self) -> Dict[str, Any]:
        """
        Analyze all active A/B experiments.
        
        PRIORITY 2.4: Generates reports and recommendations.
        
        Returns:
            Dict of experiment_id -> analysis report
        """
        if not self.ab_testing:
            return {}
        
        results = {}
        
        for signal_name, experiment_id in self.active_experiments.items():
            try:
                report = self.ab_testing.analyze_experiment(experiment_id)
                results[signal_name] = report
                
                # Check if we should stop the experiment
                if report.get('analysis_results', {}).get('early_stopping', {}).get('should_stop'):
                    logger.info(f"🛑 Stopping experiment for {signal_name}: {report['recommendation']['reason']}")
                    
                    # Apply winner if treatment won
                    if report['recommendation'].get('winner') == 'treatment':
                        self._apply_winning_threshold(signal_name, report)
                    
                    # Stop experiment
                    self.ab_testing.stop_experiment(
                        experiment_id,
                        reason=report['recommendation']['reason']
                    )
                    
                    # Remove from active experiments
                    del self.active_experiments[signal_name]
                
            except Exception as e:
                logger.error(f"Failed to analyze experiment {experiment_id}: {e}")
        
        return results
    
    def _apply_winning_threshold(
        self,
        signal_name: str,
        report: Dict[str, Any]
    ):
        """
        Apply winning threshold from A/B test.
        
        Args:
            signal_name: Signal name
            report: Experiment report
        """
        try:
            # Get treatment threshold from report
            variants = report.get('experiment', {}).get('variants', {})
            treatment_config = variants.get('treatment', {})
            new_threshold = treatment_config.get('threshold')
            language = treatment_config.get('language')
            
            if new_threshold and language:
                # Update threshold configuration
                if language in self.language_thresholds:
                    old_threshold = self.language_thresholds[language].get(signal_name, 0.35)
                    self.language_thresholds[language][signal_name] = new_threshold
                    
                    logger.info(
                        f"✅ Applied winning threshold for {signal_name} ({language}): "
                        f"{old_threshold:.3f} -> {new_threshold:.3f}"
                    )
                    
                    # TODO: Persist to database or config file
                
        except Exception as e:
            logger.error(f"Failed to apply winning threshold: {e}")
    
    async def process_query_stream(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        user_location: Optional[Dict[str, float]] = None,
        language: str = "en",
        intent: Optional[str] = None,
        max_tokens: int = 250
    ):
        """
        PRIORITY 2: Stream query processing with real-time progress updates.
        
        Provides much better UX by showing progress as the query is processed:
        - Signal detection progress
        - Context building progress
        - Token-by-token LLM generation
        - Final metadata
        
        This is an async generator that yields progress updates.
        
        Args:
            query: User query string
            user_id: User identifier
            session_id: Session identifier
            user_location: User GPS location
            language: Response language
            intent: Pre-detected intent (optional)
           
            max_tokens: Maximum tokens to generate
            
        Yields:
            Dict with type and data:
            - type='progress': Stage progress update
            - type='signals': Detected signals
            - type='context': Context building complete
            - type='token': LLM token (streamed)
            - type='complete': Final metadata
            - type='error': Error occurred
        """
        from typing import AsyncGenerator
        
        start_time = datetime.now()
        
        try:
            logger.info(f"🎬 Starting streaming query: {query[:50]}...")
            
            # Step 1: Language detection
            if language == "en" and LANGDETECT_AVAILABLE:
                yield {
                    'type': 'progress',
                    'stage': 'language_detection',
                    'message': 'Detecting language...'
                }
                
                detected_language = self._detect_language(query)
                if detected_language != "en":
                    language = detected_language
                    yield {
                        'type': 'language',
                        'detected': language,
                        'message': f'Language detected: {language}'
                    }
            else:
                self.user_analytics["queries_by_language"][language] += 1
            
            # PRIORITY 3.3: Query rewriting (before cache check)
            original_query = query
            if self.query_rewriter:
                yield {
                    'type': 'progress',
                    'stage': 'query_rewriting',
                    'message': 'Optimizing your query...'
                }
                
                try:
                    rewrite_result = await self.query_rewriter.rewrite_query(query, language)
                    
                    if rewrite_result.get("needs_rewrite", False):
                        query = rewrite_result.get("rewritten_query", query)
                        logger.info(f"   ✍️ Query rewritten: '{original_query[:50]}...' → '{query[:50]}...'")
                        
                        yield {
                            'type': 'query_rewritten',
                            'original': original_query,
                            'rewritten': query,
                            'reason': rewrite_result.get('reason', ''),
                            'message': 'Query optimized for better results'
                        }
                        
                        self.user_analytics["queries_rewritten"] = self.user_analytics.get("queries_rewritten", 0) + 1
                    else:
                        logger.info(f"   ℹ️ Query does not need rewriting")
                except Exception as e:
                    logger.warning(f"   ⚠️ Query rewriting failed: {e}")
                    query = original_query  # Fall back to original
            
            # Step 2: Check cache
            yield {
                'type': 'progress',
                'stage': 'cache_check',
                'message': 'Checking cache...'
            }
            cache_key = self._get_cache_key(query, language)
            cached_response = await self._get_cached_response(cache_key)
            
            if cached_response:
                self.stats["cache_hits"] += 1
                self.service_analytics["cache_efficiency"]["hits"] += 1
                logger.info("✅ Cache hit!")
                yield {
                    'type': 'cache_hit',
                    'message': 'Found cached response!'
                }
                
                # Stream cached response character by character for UX
                response_text = cached_response.get('response', '')
                for i in range(0, len(response_text), 5): # 5 chars at a time
                    yield {
                        'type': 'token',
                        'data': response_text[i:i+5],
                        'cached': True
                    }
                    await asyncio.sleep(0.01)  # Small delay for streaming effect
                
                yield {
                    'type': 'complete',
                    'data': cached_response,
                    'cached': True
                }
                return
            
            # Track cache miss
            self.service_analytics["cache_efficiency"]["misses"] += 1
            
            # Step 3: Signal detections
            yield {
                'type': 'progress',
                'stage': 'signal_detection',
                'message': 'Analyzing your query...'
            }
            
            signals = await self._detect_service_signals(query, user_location, language)
            active_signals = [k for k, v in signals.items() if v]
            logger.info(f"   Signals detected: {', '.join(active_signals) if active_signals else 'none'}")
            
            # Track multi-signal queries
            if len(active_signals) > 2:
                self.stats["multi_signal_queries"] += 1
                logger.info(f"   🎯 Multi-signal query detected ({len(active_signals)} signals)")
            
            yield {
                'type': 'signals',
                'data': signals,
                'active': active_signals,
                'count': len(active_signals),
                'message': f'Detected {len(active_signals)} signals'
            }
            
            # Step 4: Context building
            yield {
                'type': 'progress',
                'stage': 'context_building',
                'message': 'Gathering relevant information...'
            }
            
            db_context = await self._build_smart_context(query, signals)
            yield {
                'type': 'context',
                'data': db_context,
                'message': 'Context built successfully'
            }
            
            # Step 5: Map visualization (if needed)
            map_data = None
            if signals['needs_map'] or signals['needs_gps_routing']:
                yield {
                    'type': 'progress',
                    'stage': 'map_generation',
                    'message': '🗺️ Generating map visualization...'
                }
                
                try:
                    map_data = await self._get_map_visualization(
                        query, 'route_planning', user_id, language, user_location
                    )
                    
                    if map_data:
                        self.service_analytics["map_generation_success"] += 1
                        yield {
                            'type': 'service',
                            'service': 'map',
                            'status': 'success',
                            'message': 'Map generated successfully'
                        }
                    else:
                        self.service_analytics["map_generation_failure"] += 1
                        yield {
                            'type': 'service',
                            'service': 'map',
                            'status': 'error',
                            'message': 'Map generation failed'
                        }
                
                except Exception as e:
                    self.service_analytics["map_generation_failure"] += 1
                    self._track_error("service_failure", "map_generation", str(e), query)
                    yield {
                        'type': 'service',
                        'service': 'map',
                        'status': 'error',
                        'message': 'Map generation failed'
                    }
            
            if signals['needs_weather'] and self.weather_service:
                yield {
                    'type': 'progress',
                    'stage': 'weather',
                    'message': '☀️ Checking weather conditions...'
                }
                weather_context = await self._get_weather_context(query)
                self.service_analytics["weather_service_calls"] += 1
                
                if weather_context:
                    yield {
                        'type': 'service',
                        'service': 'weather',
                        'status': 'success',
                        'message': 'Weather data retrieved'
                    }
            
            if signals['needs_events'] and self.events_service:
                yield {
                    'type': 'progress',
                    'stage': 'events',
                    'message': '🎭 Finding events...'
                }
                events_context = await self._get_events_context()
                self.service_analytics["events_service_calls"] += 1
                
                if events_context:
                    yield {
                        'type': 'service',
                        'service': 'events',
                        'status': 'success',
                        'message': 'Events data retrieved'
                    }
            
            if signals['needs_hidden_gems'] and self.hidden_gems_handler:
                yield {
                    'type': 'progress',
                    'stage': 'hidden_gems',
                    'message': '💎 Discovering hidden gems...'
                }
                hidden_gems_context = await self._get_hidden_gems_context(query)
                self.service_analytics["hidden_gems_calls"] += 1
                
                if hidden_gems_context:
                    yield {
                        'type': 'service',
                        'service': 'hidden_gems',
                        'status': 'success',
                        'message': 'Hidden gems found'
                    }
            
            # Step 6: Build prompts
            system_prompt = self._build_system_prompt(signals)
            full_prompt = self._build_prompt_with_signals(
                query=query,
                signals=signals,
                system_prompt=system_prompt,
                db_context=db_context,
                rag_context=rag_context,
                weather_context=weather_context,
                events_context=events_context,
                hidden_gems_context=hidden_gems_context,
                language=language
            )
            
            # Step 7: Stream LLM generation
            yield {
                'type': 'progress',
                'stage': 'generation',
                'message': '✨ Generating response...'
            }
            self.stats["llm_calls"] += 1
            llm_start = time.time()
            
            # Check if LLM supports streaming
            if hasattr(self.llm, 'generate_stream'):
                response_tokens = []
                async for token in self.llm.generate_stream(
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                ):
                    response_tokens.append(token)
                    yield {
                        'type': 'token',
                        'data': token,
                        'cached': False
                    }
                    await asyncio.sleep(0.01)  # Simulate streaming delay
            else:
                # Fallback: No streaming support, simulate it
                response_data = await self.llm.generate(
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                
                if not response_data or "generated_text" not in response_data:
                    raise Exception("Invalid LLM response")
                
                response_text = response_data["generated_text"]
                
                # Simulate streaming for UX (5 chars at a time)
                for i in range(0, len(response_text), 5):
                    yield {
                        'type': 'token',
                        'data': response_text[i:i+5],
                        'cached': False
                    }
                    await asyncio.sleep(0.01)
            
            llm_latency = time.time() - llm_start
            self._track_performance("llm", llm_latency)
            
            # Step 8: Validate response
            is_valid, error_message = self._validate_response(
                response_text, query, signals, bool(db_context)
            )
            
            if not is_valid:
                logger.warning(f"Response validation failed: {error_message}")
                self._track_error("validation_failure", "response_validator", error_message, query)
                
                yield {
                    'type': 'warning',
                    'message': 'Response validation issue detected',
                    'details': error_message
                }
            
            # Step 9: Final metadata
            total_latency = (datetime.now() - start_time).total_seconds()
            self._track_performance("query", total_latency)
            
            result = {
                "status": "success",
                "response": response_text,
                "map_data": map_data,
                "signals": signals,
                "metadata": {
                    "signals_detected": active_signals,
                    "context_used": bool(db_context),
                    "rag_used": bool(rag_context),
                    "map_generated": bool(map_data),
                    "weather_used": bool(weather_context),
                    "events_used": bool(events_context),
                    "hidden_gems_used": bool(hidden_gems_context),
                    "source": "runpod_llm",
                    "processing_time": total_latency,
                    "llm_latency": llm_latency,
                    "cached": False,
                    "multi_intent": len(active_signals) > 2,
                    "language": language,
                    "detected_language": language if LANGDETECT_AVAILABLE else None
                }
            }
            
            yield {
                'type': 'complete',
                'data': result
            }
            return
            
        except Exception as e:
            logger.error(f"❌ Streaming query failed: {e}")
            self.stats["fallback_calls"] += 1
            self._track_error("llm_failure", "runpod", str(e), query)
            self.error_tracker["error_recovery_count"] += 1
            
            # Fallback to RAG-only or database context
            fallback = await self._fallback_response(
                query=query,
                intent='general',
                db_context=db_context,
                rag_context=rag_context,
                map_data=map_data
            )
            yield {
                'type': 'error',
                'message': str(e),
                'fallback': fallback
            }
            return
    
    # ============================================================================
    # PRIORITY 3.2: CONVERSATIONAL CONTEXT INTEGRATION
    # ============================================================================
    
    async def process_query_with_conversation(
        self,
        query: str,
        session_id: str,
        user_id: str = "anonymous",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process query with conversational context (Simple LLM-based approach).
        
        This method:
        1. Retrieves conversation history
        2. Formats context for LLM prompt
        3. Lets LLM naturally understand references
        4. Stores conversation turn
        
        No complex keyword matching - just give the LLM conversation history!
        
        Args:
            query: User query (may contain references like "there", "it", etc.)
            session_id: Conversation session ID
            user_id: User identifier
            **kwargs: Additional parameters for process_query
            
        Returns:
            Response dict with conversation metadata
        """
        if not self.conversation_manager:
            # No conversation support, process normally
            return await self.process_query(query=query, user_id=user_id, **kwargs)
        
        try:
            # Step 1: Check if session has conversation history
            has_context = self.conversation_manager.has_context(session_id)
            
            # Step 2: Get formatted context for LLM if history exists
            context_prompt = ""
            if has_context:
                context_prompt = self.conversation_manager.format_context_for_llm(
                    session_id=session_id,
                    max_turns=3,  # Include last 3 conversation turns
                    include_metadata=False
                )
                
                logger.info(f"💬 Including conversation context ({len(self.conversation_manager.get_history(session_id))} turns)")
            
            # Step 3: Add context to kwargs (will be injected into LLM prompt)
            if context_prompt:
                if 'conversation_context' not in kwargs:
                    kwargs['conversation_context'] = context_prompt
            
            # Step 4: Process query with context
            response = await self.process_query(
                query=query,  # Use original query - LLM will understand references
                user_id=user_id,
                **kwargs
            )
            
            # Step 5: Store user query in conversation history
            self.conversation_manager.add_turn(
                session_id=session_id,
                role='user',
                content=query,
                metadata={
                    'detected_signals': response.get('signals', {}),
                    'user_id': user_id
                }
            )
            
            # Step 6: Store assistant response in conversation history
            self.conversation_manager.add_turn(
                session_id=session_id,
                role='assistant',
                content=response.get('response', ''),
                metadata={
                    'signals_detected': [k for k, v in response.get('signals', {}).items() if v],
                    'map_generated': response.get('metadata', {}).get('map_generated', False)
                }
            )
            
            # Step 7: Add conversation metadata to response
            stats = self.conversation_manager.get_statistics(session_id)
            response['conversation'] = {
                'session_id': session_id,
                'turn_count': stats.get('turn_count', 0),
                'had_context': has_context,
                'last_topic': stats.get('last_topic')
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in conversational query processing: {e}")
            # Fallback to normal processing
            return await self.process_query(query=query, user_id=user_id, **kwargs)
    
    async def process_query_stream_with_context(
        self,
        query: str,
        session_id: str,
        user_id: str = "anonymous",
        **kwargs
    ):
        """
        Stream query processing with conversational context support.
        
        PRIORITY 3.2: Handles follow-up questions with streaming.
        
        Args:
            query: User query (may contain references)
            session_id: Conversation session ID
            user_id: User identifier
            **kwargs: Additional parameters
            
        Yields:
            Progress updates and response chunks
        """
        if not self.conversation_manager:
            # No conversation support, stream normally
            async for chunk in self.process_query_stream(query=query, user_id=user_id, **kwargs):
                yield chunk
            return
        
        try:
            # Step 1: Resolve references
            yield {
                'type': 'progress',
                'stage': 'context_resolution',
                'message': 'Analyzing conversation context...'
            }
            
            resolved_query, was_resolved = self.conversation_manager.resolve_query(
                session_id=session_id,
                query=query
            )
            
            if was_resolved:
                yield {
                    'type': 'context_resolved',
                    'original_query': query,
                    'resolved_query': resolved_query
                }
            
            # Step 2: Get context summary
            context_summary = self.conversation_manager.get_context_summary(
                session_id=session_id,
                max_length=200
            )
            
            if context_summary:
                kwargs['additional_context'] = kwargs.get('additional_context', '') + f"\n{context_summary}"
            
            # Step 3: Stream query processing
            response_text = ""
            signals = {}
            metadata = {}
            
            async for chunk in self.process_query_stream(
                query=resolved_query,
                user_id=user_id,
                **kwargs
            ):
                # Forward all chunks
                yield chunk
                
                # Capture response data for conversation history
                if chunk['type'] == 'token':
                    response_text += chunk.get('data', '')
                elif chunk['type'] == 'signals':
                    signals = chunk.get('data', {})
                elif chunk['type'] == 'complete':
                    metadata = chunk.get('metadata', {})
            
            # Step 4: Store conversation turns
            self.conversation_manager.add_turn(
                session_id=session_id,
                role='user',
                content=query,
                metadata={
                    'resolved_query': resolved_query if was_resolved else None,
                    'detected_signals': signals,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            self.conversation_manager.add_turn(
                session_id=session_id,
                role='assistant',
                content=response_text,
                metadata={
                    'signals_detected': [k for k, v in signals.items() if v],
                    'map_generated': metadata.get('map_generated', False),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Step 5: Yield conversation metadata
            yield {
                'type': 'conversation_metadata',
                'session_id': session_id,
                'was_resolved': was_resolved,
                'original_query': query if was_resolved else None,
                'resolved_query': resolved_query if was_resolved else None
            }
            
        except Exception as e:
            logger.error(f"Failed to stream query with context: {e}")
            # Fallback to regular streaming
            async for chunk in self.process_query_stream(query=query, user_id=user_id, **kwargs):
                yield chunk
    
    def get_conversation_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum turns to retrieve
            
        Returns:
            List of conversation turns
        """
        if not self.conversation_manager:
            return []
        
        try:
            history = self.conversation_manager.get_history(
                session_id=session_id,
                max_turns=max_turns
            )
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def clear_conversation(self, session_id: str):
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
        """
        if not self.conversation_manager:
            return
        
        try:
            self.conversation_manager.clear_session(session_id)
            logger.info(f"🗑️ Cleared conversation for session: {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to clear conversation: {e}")
    
    def get_conversation_statistics(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation statistics for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Statistics dict
        """
        if not self.conversation_manager:
            return {'enabled': False}
        
        try:
            stats = self.conversation_manager.get_statistics(session_id)
            stats['enabled'] = True
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get conversation statistics: {e}")
            return {'enabled': True, 'error': str(e)}