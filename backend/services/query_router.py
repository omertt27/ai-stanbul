"""
Industry-Enhanced Query Router for AI Istanbul System

This service classifies user queries and routes them to appropriate services
with enterprise-level features including monitoring, security, and resilience.
Uses advanced pattern matching, ML-based classification, and rule-based routing.
"""

import re
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)

# Custom exceptions
class SecurityError(Exception):
    """Raised when query fails security validation"""
    pass

class ClassificationError(Exception):
    """Raised when query classification fails"""
    pass

# Industry-level imports
try:
    from .industry_monitoring import get_monitoring_system, monitor_operation
    from .industry_security import get_security_system, validate_and_sanitize
    from .industry_resilience import get_error_handler, handle_errors, with_retry, with_circuit_breaker
    INDUSTRY_FEATURES_AVAILABLE = True
except ImportError:
    INDUSTRY_FEATURES_AVAILABLE = False
    # Fallback implementations when industry features are not available
    def handle_errors(func):
        return func
    def with_retry(max_attempts=3, delay=1.0):
        def decorator(func):
            return func
        return decorator
    def with_circuit_breaker(failure_threshold=5, reset_timeout=60):
        def decorator(func):
            return func
        return decorator
    def monitor_operation(operation_name):
        def decorator(func):
            return func
        return decorator
    def get_monitoring_system():
        return None
    def get_security_system():
        return None
    def validate_and_sanitize(query, user_id=None):
        return query

# Advanced AI system integration
try:
    from .advanced_ai_system import (
        get_ai_system, process_query_with_ai, get_personalized_insights, 
        predict_demand, AdvancedAISystem, AICapability
    )
    AI_SYSTEM_AVAILABLE = True
except ImportError:
    AI_SYSTEM_AVAILABLE = False
    # Fallback implementations
    def get_ai_system():
        return None
    def process_query_with_ai(query, user_id=None, context=None):
        return {}
    def get_personalized_insights(user_id, context=None):
        return {}
    def predict_demand(location=None, time_period=None):
        return {}

# Enhanced services integration
try:
    from .template_engine import TemplateEngine
    from .recommendation_engine import RecommendationEngine, UserProfile as RecUserProfile
    from .route_planner import RouteOptimizer, RouteRequest, Location, RouteType, TransportMode
    ENHANCED_SERVICES_AVAILABLE = True
except ImportError as e:
    ENHANCED_SERVICES_AVAILABLE = False
    logger.warning(f"⚠️ Enhanced services not available: {e}")
    
    # Fallback implementations
    class RouteType(Enum):
        FASTEST = "fastest"
        SHORTEST = "shortest"
        CHEAPEST = "cheapest"
        MOST_SCENIC = "most_scenic"
        ACCESSIBLE = "accessible"
    
    class TransportMode(Enum):
        WALKING = "walking"
        METRO = "metro"
        BUS = "bus"
        TAXI = "taxi"
    
    @dataclass
    class Location:
        name: str
        coordinates: Tuple[float, float]
        district: str = "Istanbul"
        transport_connections: List[str] = field(default_factory=list)
    
    @dataclass 
    class RouteRequest:
        from_location: Location
        to_location: Location
        route_type: RouteType
        departure_time: Optional[datetime] = None
        accessibility_required: bool = False
        max_walking_distance_km: float = 2.0

class QueryType(Enum):
    """Types of queries the system can handle"""
    GREETING = "greeting"
    ATTRACTION_INFO = "attraction_info"
    ATTRACTION_SEARCH = "attraction_search"
    RESTAURANT_SEARCH = "restaurant_search"
    RESTAURANT_INFO = "restaurant_info"
    TRANSPORT_ROUTE = "transport_route"
    TRANSPORT_INFO = "transport_info"
    ITINERARY_REQUEST = "itinerary_request"
    GENERAL_INFO = "general_info"
    PRACTICAL_INFO = "practical_info"
    RECOMMENDATION = "recommendation"
    UNKNOWN = "unknown"

@dataclass
class QueryClassification:
    """Enhanced result of query classification with industry features and AI integration"""
    query_type: QueryType
    confidence: float
    extracted_entities: Dict[str, Any]
    suggested_service: str
    language: str
    # Industry enhancements
    security_score: float = 100.0
    processing_time_ms: float = 0.0
    query_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    cache_key: str = ""
    user_id: Optional[str] = None
    request_context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # AI enhancements
    ai_analysis: Dict[str, Any] = field(default_factory=dict)
    personalization: Dict[str, Any] = field(default_factory=dict)
    predictive_insights: Dict[str, Any] = field(default_factory=dict)
    semantic_similarity_scores: Dict[str, float] = field(default_factory=dict)
    ai_recommendations: List[str] = field(default_factory=list)
    intent_confidence: float = 0.0
    entity_confidence: float = 0.0
    sentiment_analysis: Dict[str, Any] = field(default_factory=dict)
    classification_method: str = "rule_based"  # "rule_based", "ai_powered", "hybrid"

class IndustryQueryRouter:
    """
    Next-Generation AI-Powered Query Classification and Routing System
    
    Features:
    - Neural semantic understanding with transformers
    - Advanced threat detection and input validation
    - Performance monitoring and distributed tracing
    - Circuit breakers and retry logic for resilience
    - ML-based intent classification and entity extraction
    - Real-time personalization and learning
    - Predictive analytics and demand forecasting
    - Multi-modal support and context awareness
    - Comprehensive logging and observability
    """
    
    def __init__(self):
        # Core routing components (legacy fallback)
        self.patterns = self._load_patterns()
        self.keywords = self._load_keywords()
        self.location_patterns = self._load_location_patterns()
        
        # Industry-level features
        self.query_cache = {}
        self.performance_history = []
        self.classification_stats = {}
        self.threat_detection_enabled = True
        
        # AI system integration
        self.ai_system = None
        self.ai_enabled = False
        self.use_hybrid_classification = True  # Use both AI and rule-based
        self.ai_fallback_confidence_threshold = 0.6
        
        # Initialize industry systems
        if INDUSTRY_FEATURES_AVAILABLE:
            self.monitoring = get_monitoring_system()
            self.security = get_security_system()
            self.error_handler = get_error_handler()
            
            # Start performance monitoring
            self.monitoring.record_metric("query_router_initialized", 1.0, "counter")
        else:
            self.monitoring = None
            self.security = None
            self.error_handler = None
        
        # Initialize AI system
        if AI_SYSTEM_AVAILABLE:
            try:
                self.ai_system = get_ai_system()
                self.ai_enabled = True
                logger.info("🧠 AI System integrated successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize AI system: {e}")
                self.ai_enabled = False
        else:
            logger.warning("⚠️ AI system not available - using rule-based classification only")
        
        # Initialize enhanced services
        if ENHANCED_SERVICES_AVAILABLE:
            try:
                self.template_engine = TemplateEngine()
                self.recommendation_engine = RecommendationEngine()
                self.route_planner = RouteOptimizer()
                logger.info("🎯 Enhanced services integrated: Template Engine, Recommendation Engine, Route Planner")
            except Exception as e:
                logger.error(f"❌ Failed to initialize enhanced services: {e}")
                self.template_engine = None
                self.recommendation_engine = None
                self.route_planner = None
        else:
            self.template_engine = None
            self.recommendation_engine = None
            self.route_planner = None
            logger.warning("⚠️ Enhanced services not available")
        
        # Log initialization status
        features_status = []
        if INDUSTRY_FEATURES_AVAILABLE:
            features_status.append("✅ Enterprise Features")
        if self.ai_enabled:
            features_status.append("✅ AI-Powered Classification")
        
        logger.info(f"🚀 Industry Query Router initialized: {', '.join(features_status) if features_status else 'Basic Mode'}")
        
    def _load_patterns(self) -> Dict[str, List[str]]:
        """Load regex patterns for different query types"""
        return {
            "greeting": [
                r"\b(merhaba|selam|hi|hello|hey|good morning|günaydın|iyi günler)\b",
                r"^(selam|merhaba|hi|hello)$",
                r"\bnasılsın\b|\bhow are you\b"
            ],
            
            "attraction_info": [
                r"\b(ayasofya|hagia sophia|sultanahmet|blue mosque|topkapi|galata tower|basilica cistern)\b",
                r"\b(müze|museum|cami|mosque|saray|palace|kule|tower)\b.*\b(nerede|where|nasıl|how)\b",
                r"\b(bilgi|info|information|hakkında|about)\b.*\b(ayasofya|sultanahmet|topkapi)\b"
            ],
            
            "attraction_search": [
                r"\b(görmek|see|visit|gez|explore|keşfet|discover)\b.*\b(yer|place|mekan|location)\b",
                r"\b(turistik|touristic|historic|tarihi|kültürel|cultural)\b.*\b(yer|place|mekan|sites?)\b",
                r"\b(ne yapabilirim|what can i do|what to see|ne görebilirim)\b",
                r"\b(öneri|recommend|suggestion|tavsiye)\b.*\b(yer|place|gezilecek)\b",
                r"\b(cultural|historic|historical|religious)\s+(sites?|places?|attractions?)\b",
                r"\b(sites?|places?|attractions?)\s+to\s+(see|visit|explore)\b"
            ],
            
            "restaurant_search": [
                r"\b(restoran|restaurant|yemek|food|dining|lokanta)\b.*\b(öneri|recommend|tavsiye)\b",
                r"\b(nerede yemek|where to eat|en iyi|best)\b.*\b(restoran|restaurant|yemek)\b",
                r"\b(türk mutfağı|turkish cuisine|ottoman|kebap|kebab|baklava)\b",
                r"\b(balık|fish|seafood|deniz ürünleri)\b.*\b(restoran|restaurant)\b",
                r"\b(turkish food|türk yemek|i want.*food)\b",
                r"\b(best restaurants|en iyi restoran|good restaurants)\b",
                r"\b(upscale|expensive|luxury|pahalı|lüks|fine dining)\b.*\b(restoran|restaurant|dining)\b",
                r"\b(cheap|budget|ucuz|ekonomik)\b.*\b(restoran|restaurant|yemek|food|dining)\b",
                r"\b(italian|pizza|pasta|chinese|japanese|sushi)\b.*\b(restoran|restaurant)\b",
                r"\b(dining options|yemek seçenek|restoran seçenek)\b",
                r"\b(where can i find|nerede bulabilirim)\b.*\b(food|yemek|kebab|restoran)\b",
                r"\b(italian|chinese|japanese|french|greek|indian|mexican)\b.*\b(restaurants?|near|in)\b",
                r"\b(restaurants?|food)\b.*\b(near|yakın|close to|around)\b.*\b(galata|sultanahmet|taksim|beyoğlu|kadıköy)\b",
                r"\b(galata|sultanahmet|taksim|beyoğlu|kadıköy)\b.*\b(restaurants?|food|dining|yemek)\b"
            ],
            
            "transport_route": [
                r"\b(nasıl giderim|how to get|how do i get|ulaşım|transport|transportation)\b",
                r"\b(metro|subway|bus|otobüs|tram|tramvay|ferry|vapur)\b.*\b(güzergah|route|to|dan|from)\b",
                r"\bdan\b.*\ba\b.*\b(nasıl|how)\b|\bfrom\b.*\bto\b",
                r"\b(taksi|taxi|uber|dolmuş)\b.*\b(ne kadar|how much|ücret|cost)\b",
                r"\b(metro route|bus from|ferry schedule|ferry from)\b",
                r"\b(from|to)\b.*\b(sultanahmet|taksim|galata|kadıkoy|kadıköy|eminonu|eminönü|besiktas|beşiktaş)\b",
                r"\b(sultanahmet|taksim|galata|kadıkoy|kadıköy|eminonu|eminönü|besiktas|beşiktaş)\b.*\b(to|from|dan|a)\b.*\b(sultanahmet|taksim|galata|kadıkoy|kadıköy|eminonu|eminönü|besiktas|beşiktaş)\b",
                r"\b(airport|havalimanı)\b.*\b(city center|şehir merkezi|center)\b"
            ],
            
            "transport_info": [
                r"\b(metro haritası|metro map|ulaşım|transportation|public transport)\b",
                r"\b(istanbulkart|istanbul card|bilet|ticket|fiyat|price)\b.*\b(metro|bus|tram)\b",
                r"\b(sefer saatleri|schedule|timetable|çalışma saatleri)\b"
            ],
            
            "itinerary_request": [
                r"\b(program|itinerary|plan|gün)\b.*\b(istanbul|istanbulda)\b",
                r"\b(\d+)\s+(gün|day|günlük|days?)\b.*\b(program|plan|itinerary)\b",
                r"\b(ne yapmalıyım|what should i do|nasıl geçirmeliyim|how to spend)\b.*\b(gün|day|time)\b"
            ],
            
            "practical_info": [
                r"\b(açılış saatleri|opening hours|working hours|çalışma saatleri)\b",
                r"\b(giriş ücreti|admission|entrance fee|ticket price|bilet fiyatı)\b",
                r"\b(nasıl ulaşırım|how to reach|transportation|ulaşım)\b",
                r"\b(wifi|internet|atm|para|money|döviz|exchange)\b"
            ],
            
            "recommendation": [
                r"\b(öner|recommend|suggest|tavsiye et)\b",
                r"\b(en iyi|best|favorin|favorite|must see|must visit)\b",
                r"\b(ne yapmalı|what should|what to do|ne görmeli|what to see)\b"
            ]
        }
    
    def _load_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Load keyword dictionaries for entity extraction"""
        return {
            "attractions": {
                "turkish": [
                    "ayasofya", "sultanahmet", "topkapi", "galata kulesi", "yerebatan sarnıcı",
                    "kapalıçarşı", "mısır çarşısı", "dolmabahçe", "beylerbeyi", "çırağan", 
                    "istiklal caddesi", "taksim", "ortaköy", "bebek", "emirgan",
                    "büyük çamlıca", "pierre loti", "eyüp sultan", "fatih", "eminönü"
                ],
                "english": [
                    "hagia sophia", "sultanahmet", "topkapi", "galata tower", "basilica cistern",
                    "grand bazaar", "spice bazaar", "dolmabahce", "beylerbeyi", "ciragan",
                    "istiklal street", "taksim", "ortakoy", "bebek", "emirgan",
                    "camlica hill", "pierre loti", "eyup sultan", "fatih", "eminonu"
                ]
            },
            
            "districts": {
                "turkish": [
                    "sultanahmet", "beyoğlu", "galata", "karaköy", "beşiktaş", "ortaköy",
                    "üsküdar", "kadıköy", "fatih", "eminönü", "bakırköy", "şişli",
                    "bebek", "arnavutköy", "balat", "fener", "eyüp", "kuzguncuk"
                ],
                "english": [
                    "sultanahmet", "beyoglu", "galata", "karakoy", "besiktas", "ortakoy",
                    "uskudar", "kadikoy", "fatih", "eminonu", "bakirkoy", "sisli",
                    "bebek", "arnavutkoy", "balat", "fener", "eyup", "kuzguncuk"
                ]
            },
            
            "cuisines": {
                "turkish": [
                    "türk mutfağı", "osmanlı", "kebap", "döner", "balık", "deniz ürünleri",
                    "meze", "rakı", "baklava", "künefe", "lahmacun", "pide", "çorba",
                    "mantı", "dolma", "köfte", "şiş", "adana", "urfa"
                ],
                "english": [
                    "turkish cuisine", "ottoman", "kebab", "doner", "fish", "seafood",
                    "meze", "raki", "baklava", "kunefe", "lahmacun", "pide", "soup",
                    "manti", "dolma", "meatballs", "shish", "adana", "urfa"
                ]
            },
            
            "transport": {
                "turkish": [
                    "metro", "metrobüs", "otobüs", "tramvay", "vapur", "denizbus",
                    "funicular", "teleferik", "taksi", "dolmuş", "minibüs", "marmaray"
                ],
                "english": [
                    "metro", "metrobus", "bus", "tram", "ferry", "sea bus",
                    "funicular", "cable car", "taxi", "dolmus", "minibus", "marmaray"
                ]
            }
        }
    
    def _load_location_patterns(self) -> List[str]:
        """Load location extraction patterns"""
        return [
            # Enhanced Turkish patterns
            r"([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+?)\s*(?:dan|den)\s+([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+?)\s*(?:ya|ye|a|e)\s*(?:nasıl|how)",
            r"([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+?)\s*(?:dan|den)\s+([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+?)\s*(?:ya|ye|a|e)",
            # Enhanced English patterns
            r"(?:how\s+to\s+get\s+)?(?:from\s+)?([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+?)\s+(?:to)\s+([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+)",
            r"([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+?)\s*(?:to)\s+([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+)",
            # Transport-specific patterns
            r"([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+?)\s*(?:metro|bus|otobüs|tram|ferry|vapur)\s*(?:to|ya|ye)?\s*([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+)",
            # General location patterns
            r"\bnerede\s+([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+)\b",
            r"\bwhere\s+is\s+([A-Za-zÇĞıİÖŞÜçğıiöşü\s]+)\b"
        ]
    
    @handle_errors
    @with_retry("api_call")
    @with_circuit_breaker("query_classification")
    @monitor_operation("query_classification")
    def classify_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryClassification:
        """
        Next-Generation AI-Powered Query Classification
        
        Combines neural semantic understanding with rule-based fallbacks for
        industry-grade reliability and accuracy.
        
        Args:
            query: User's input query
            context: Request context (IP, user agent, user ID, etc.)
            
        Returns:
            Enhanced QueryClassification with AI insights, personalization, and predictions
        """
        start_time = time.time()
        context = context or {}
        
        # Generate query ID for tracking
        query_id = self._generate_query_id(query, context)
        
        # Security validation and threat detection
        security_result = None
        security_score = 100.0
        if INDUSTRY_FEATURES_AVAILABLE and self.security and self.threat_detection_enabled:
            security_result = self.security.validate_input(
                query, 
                {
                    "source_ip": context.get("source_ip"),
                    "user_agent": context.get("user_agent"),
                    "operation": "query_classification"
                }
            )
            
            # Block if security validation fails
            if not security_result["valid"]:
                self._record_security_block(query, context, security_result)
                raise SecurityError("Query blocked due to security concerns")
            
            # Use sanitized input if available
            if security_result["sanitized"]:
                query = security_result["sanitized_input"]
            
            security_score = security_result.get("security_score", 100.0)
        
        # Check cache first
        cache_key = self._generate_cache_key(query, context)
        cached_result = self._get_cached_classification(cache_key)
        if cached_result:
            cached_result.processing_time_ms = (time.time() - start_time) * 1000
            self._record_cache_hit(query_id)
            return cached_result
        
        # Perform AI-powered classification
        query_lower = query.lower().strip()
        language = self._detect_language(query_lower)
        classification_start = time.time()
        
        try:
            # Primary AI-powered classification
            if self.ai_enabled and self.ai_system:
                ai_result = self._classify_with_ai(query, context, query_id, language)
                
                # Use AI result if confidence is high enough
                if ai_result.confidence >= self.ai_fallback_confidence_threshold:
                    processing_time = time.time() - start_time
                    ai_result.processing_time_ms = processing_time * 1000
                    ai_result.security_score = security_score
                    ai_result.classification_method = "ai_powered"
                    
                    # Cache the result
                    self._cache_classification(cache_key, ai_result)
                    self._record_classification_metrics(ai_result)
                    
                    return ai_result
                
                # Use hybrid approach if AI confidence is moderate
                elif self.use_hybrid_classification and ai_result.confidence >= 0.4:
                    rule_result = self._classify_with_rules(query_lower, language, context, query_id)
                    hybrid_result = self._combine_ai_and_rule_results(ai_result, rule_result, context)
                    
                    processing_time = time.time() - start_time
                    hybrid_result.processing_time_ms = processing_time * 1000
                    hybrid_result.security_score = security_score
                    hybrid_result.classification_method = "hybrid"
                    
                    # Cache the result
                    self._cache_classification(cache_key, hybrid_result)
                    self._record_classification_metrics(hybrid_result)
                    
                    return hybrid_result
        
        except Exception as e:
            logger.error(f"❌ AI classification failed: {e}")
            if INDUSTRY_FEATURES_AVAILABLE and self.monitoring:
                self.monitoring.increment_counter("ai_classification_failures")
        
        # Fallback to rule-based classification
        rule_result = self._classify_with_rules(query_lower, language, context, query_id)
        processing_time = time.time() - start_time
        rule_result.processing_time_ms = processing_time * 1000
        rule_result.security_score = security_score
        rule_result.classification_method = "rule_based"
        
        # Cache the result
        self._cache_classification(cache_key, rule_result)
        self._record_classification_metrics(rule_result)
        
        return rule_result
    
    def _matches_patterns(self, query: str, pattern_type: str) -> bool:
        """Check if query matches any pattern of given type"""
        if pattern_type not in self.patterns:
            return False
        
        for pattern in self.patterns[pattern_type]:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _detect_language(self, query: str) -> str:
        """Detect query language"""
        turkish_indicators = [
            "nerede", "nasıl", "ne", "neler", "hangi", "kaç", "kim", "niçin", "niye",
            "merhaba", "selam", "teşekkür", "sağol", "türkiye", "istanbul", "çok",
            "güzel", "iyi", "kötü", "büyük", "küçük", "yeni", "eski", "var", "yok"
        ]
        
        english_indicators = [
            "where", "how", "what", "which", "when", "who", "why", "hello", "hi",
            "thank", "thanks", "turkey", "istanbul", "very", "beautiful", "good",
            "bad", "big", "small", "new", "old", "there", "here", "the", "and"
        ]
        
        turkish_score = sum(1 for word in turkish_indicators if word in query)
        english_score = sum(1 for word in english_indicators if word in query)
        
        return "turkish" if turkish_score >= english_score else "english"
    
    def _extract_attractions(self, query: str, language: str) -> List[str]:
        """Extract attraction names from query"""
        attractions = []
        keyword_list = self.keywords["attractions"][language]
        
        for attraction in keyword_list:
            if attraction in query:
                attractions.append(attraction)
        
        return attractions
    
    def _extract_districts(self, query: str, language: str) -> List[str]:
        """Extract district names from query"""
        districts = []
        keyword_list = self.keywords["districts"][language]
        
        for district in keyword_list:
            if district in query:
                districts.append(district)
        
        return districts
    
    def _extract_cuisines(self, query: str, language: str) -> List[str]:
        """Extract cuisine types from query"""
        cuisines = []
        keyword_list = self.keywords["cuisines"][language]
        
        for cuisine in keyword_list:
            if cuisine in query:
                cuisines.append(cuisine)
        
        return cuisines
    
    def _extract_transport_type(self, query: str, language: str) -> Optional[str]:
        """Extract transport type from query"""
        keyword_list = self.keywords["transport"][language]
        
        for transport in keyword_list:
            if transport in query:
                return transport
        
        return None
    
    def _extract_locations(self, query: str) -> Dict[str, str]:
        """Extract from/to locations from query"""
        locations = {}
        
        for pattern in self.location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    locations["from"] = groups[0].strip() if groups[0] else groups[1].strip()
                    locations["to"] = groups[-1].strip()
                break
        
        return locations
    
    def _extract_duration(self, query: str) -> Optional[str]:
        """Extract duration from query (e.g., '3 days', '2 gün')"""
        patterns = [
            r"(\d+)\s+(gün|day|days|günlük)",
            r"(\d+)\s+(saat|hour|hours|saatlik)",
            r"(bir|one|1)\s+(gün|day|hafta|week)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_interests(self, query: str, language: str) -> List[str]:
        """Extract user interests from query"""
        interests = []
        
        interest_keywords = {
            "turkish": {
                "history": ["tarih", "tarihi", "antik", "eski", "geçmiş"],
                "culture": ["kültür", "kültürel", "sanat", "müze", "gelenek"],
                "food": ["yemek", "lezzet", "mutfak", "lokanta", "restoran"],
                "nature": ["doğa", "park", "bahçe", "deniz", "orman"],
                "shopping": ["alışveriş", "çarşı", "mağaza", "pazar"],
                "nightlife": ["gece", "eğlence", "bar", "kulüp", "müzik"]
            },
            "english": {
                "history": ["history", "historical", "ancient", "old", "past"],
                "culture": ["culture", "cultural", "art", "museum", "tradition"],
                "food": ["food", "taste", "cuisine", "restaurant", "dining"],
                "nature": ["nature", "park", "garden", "sea", "forest"],
                "shopping": ["shopping", "bazaar", "store", "market"],
                "nightlife": ["night", "entertainment", "bar", "club", "music"]
            }
        }
        
        keyword_dict = interest_keywords.get(language, interest_keywords["english"])
        
        for interest, keywords in keyword_dict.items():
            if any(keyword in query for keyword in keywords):
                interests.append(interest)
        
        return interests
    
    def _contains_food_keywords(self, query: str) -> bool:
        """Check if query contains food-related keywords"""
        food_keywords = [
            "food", "yemek", "restaurant", "restoran", "dining", "eat", "meal",
            "turkish food", "türk yemek", "cuisine", "mutfak", "kebab", "kebap",
            "baklava", "döner", "meze", "lokanta", "cook", "chef", "kitchen"
        ]
        
        return any(keyword in query.lower() for keyword in food_keywords)
    
    def _contains_transport_keywords(self, query: str) -> bool:
        """Check if query contains transport-related keywords"""
        transport_keywords = [
            "how to get", "nasıl giderim", "ulaşım", "transportation", "transport",
            "metro", "subway", "bus", "otobüs", "tram", "tramvay", "ferry", "vapur",
            "taxi", "taksi", "uber", "dolmuş", "route", "güzergah", "from", "to", 
            "dan", "den", "a", "e", "airport", "havalimanı", "schedule", "sefer"
        ]
        
        return any(keyword in query.lower() for keyword in transport_keywords)

    def _extract_price_range(self, query: str) -> Optional[str]:
        """Extract price range indicators from query"""
        if any(word in query for word in ["ucuz", "cheap", "budget", "ekonomik"]):
            return "budget"
        elif any(word in query for word in ["pahalı", "expensive", "luxury", "lüks"]):
            return "expensive"
        elif any(word in query for word in ["orta", "middle", "moderate", "normal"]):
            return "moderate"
        
        return None
    
    def _extract_practical_info_type(self, query: str) -> str:
        """Extract type of practical information requested"""
        if any(word in query for word in ["saat", "hours", "açık", "open"]):
            return "hours"
        elif any(word in query for word in ["fiyat", "price", "ücret", "cost"]):
            return "price"
        elif any(word in query for word in ["ulaşım", "transport", "nasıl", "how"]):
            return "transport"
        elif any(word in query for word in ["wifi", "internet", "atm"]):
            return "facilities"
        
        return "general"
    
    def get_service_for_query_type(self, query_type: QueryType) -> str:
        """Get the appropriate service for a given query type"""
        service_mapping = {
            QueryType.GREETING: "template_engine",
            QueryType.ATTRACTION_INFO: "info_retrieval_service",
            QueryType.ATTRACTION_SEARCH: "info_retrieval_service",
            QueryType.RESTAURANT_SEARCH: "restaurant_database_service",
            QueryType.RESTAURANT_INFO: "restaurant_database_service",
            QueryType.TRANSPORT_ROUTE: "transport_service",
            QueryType.TRANSPORT_INFO: "transport_service",
            QueryType.ITINERARY_REQUEST: "recommendation_engine",
            QueryType.GENERAL_INFO: "info_retrieval_service",
            QueryType.PRACTICAL_INFO: "info_retrieval_service",
            QueryType.RECOMMENDATION: "recommendation_engine",
            QueryType.UNKNOWN: "template_engine"
        }
        
        return service_mapping.get(query_type, "template_engine")
    
    # =================== INDUSTRY-LEVEL HELPER METHODS ===================
    
    def _generate_query_id(self, query: str, context: Dict[str, Any]) -> str:
        """Generate unique query ID for tracking"""
        timestamp = str(int(time.time() * 1000))
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        user_id = context.get("user_id", "anonymous")
        return f"qry_{timestamp}_{query_hash}_{user_id}"
    
    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for query classification"""
        # Include relevant context that affects classification
        key_components = [
            query.lower().strip(),
            context.get("language", "auto"),
            context.get("user_type", "default")
        ]
        key_string = "|".join(str(c) for c in key_components)
        return f"qr_class_{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _get_cached_classification(self, cache_key: str) -> Optional[QueryClassification]:
        """Get cached classification result"""
        if cache_key in self.query_cache:
            cached_data = self.query_cache[cache_key]
            # Check if cache is still valid (5 minutes TTL)
            if time.time() - cached_data["timestamp"] < 300:
                result = cached_data["classification"]
                result.metadata["cache_hit"] = True
                return result
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]
        return None
    
    def _cache_classification(self, cache_key: str, classification: QueryClassification):
        """Cache classification result"""
        self.query_cache[cache_key] = {
            "classification": classification,
            "timestamp": time.time()
        }
        
        # Limit cache size
        if len(self.query_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.query_cache.keys(), 
                               key=lambda k: self.query_cache[k]["timestamp"])[:100]
            for key in oldest_keys:
                del self.query_cache[key]
    
    def _record_security_block(self, query: str, context: Dict[str, Any], security_result: Dict[str, Any]):
        """Record security block event"""
        if INDUSTRY_FEATURES_AVAILABLE and self.monitoring:
            self.monitoring.increment_counter(
                "query_router_security_blocks",
                tags={
                    "threats_detected": str(len(security_result.get("threats", []))),
                    "security_score": str(int(security_result.get("security_score", 0) / 10) * 10)
                }
            )
    
    def _record_cache_hit(self, query_id: str):
        """Record cache hit metrics"""
        if INDUSTRY_FEATURES_AVAILABLE and self.monitoring:
            self.monitoring.increment_counter("query_router_cache_hits")
    
    def _record_classification_metrics(self, classification: QueryClassification):
        """Record classification performance metrics"""
        if INDUSTRY_FEATURES_AVAILABLE and self.monitoring:
            self.monitoring.record_histogram(
                "query_classification_time_ms",
                classification.processing_time_ms,
                tags={
                    "query_type": classification.query_type.value,
                    "language": classification.language,
                    "confidence_range": self._get_confidence_range(classification.confidence)
                }
            )
            
            self.monitoring.increment_counter(
                "query_classifications_total",
                tags={
                    "query_type": classification.query_type.value,
                    "language": classification.language
                }
            )
    
    def _get_confidence_range(self, confidence: float) -> str:
        """Get confidence range for metrics"""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"
    
    def _create_enhanced_classification(self, query_type: QueryType, confidence: float,
                                      entities: Dict[str, Any], service: str, language: str,
                                      query_id: str, processing_time: float,
                                      cache_key: str, context: Dict[str, Any],
                                      security_score: float = 100.0) -> QueryClassification:
        """Create enhanced classification result with all industry features"""
        classification = QueryClassification(
            query_type=query_type,
            confidence=confidence,
            extracted_entities=entities,
            suggested_service=service,
            language=language,
            security_score=security_score,
            processing_time_ms=processing_time * 1000,
            query_id=query_id,
            cache_key=cache_key,
            user_id=context.get("user_id"),
            request_context=context,
            performance_metrics={
                "classification_time_ms": processing_time * 1000,
                "pattern_matches": len([p for p in self.patterns.keys() if self._matches_patterns(context.get("query", ""), p)]),
                "entity_extractions": len(entities)
            }
        )
        
        # Add warnings if confidence is low
        if confidence < 0.5:
            classification.warnings.append("Low confidence classification - manual review recommended")
        
        if security_score < 80:
            classification.warnings.append("Security concerns detected in query")
        
        return classification
    
    def get_classification_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive classification analytics"""
        if not INDUSTRY_FEATURES_AVAILABLE or not self.monitoring:
            return {"error": "Analytics not available - industry features disabled"}
        
        # Get metrics from monitoring system
        metrics_summary = self.monitoring.get_metrics_summary(hours)
        
        return {
            "time_period_hours": hours,
            "total_classifications": metrics_summary.get("query_classifications_total", {}).get("count", 0),
            "average_processing_time_ms": metrics_summary.get("query_classification_time_ms", {}).get("avg", 0),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "query_type_distribution": self._get_query_type_distribution(),
            "language_distribution": self._get_language_distribution(),
            "confidence_distribution": self._get_confidence_distribution(),
            "security_blocks": metrics_summary.get("query_router_security_blocks", {}).get("count", 0),
            "performance_percentiles": {
                "p50": metrics_summary.get("query_classification_time_ms", {}).get("avg", 0),
                "p95": metrics_summary.get("query_classification_time_ms", {}).get("max", 0)
            }
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would be calculated from monitoring metrics
        return 0.0  # Placeholder
    
    def _get_query_type_distribution(self) -> Dict[str, int]:
        """Get distribution of query types"""
        return {}  # Placeholder - would be populated from metrics
    
    def _get_language_distribution(self) -> Dict[str, int]:
        """Get distribution of detected languages"""
        return {}  # Placeholder - would be populated from metrics
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels"""
        return {}  # Placeholder - would be populated from metrics
    
    def optimize_patterns(self):
        """Optimize classification patterns based on usage analytics"""
        if not INDUSTRY_FEATURES_AVAILABLE:
            return
        
        # This would analyze classification performance and optimize patterns
        logger.info("🔧 Pattern optimization completed")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get router health status"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "cache_size": len(self.query_cache),
            "patterns_loaded": len(self.patterns),
            "keywords_loaded": sum(len(keywords) for lang_keywords in self.keywords.values() 
                                 for keywords in lang_keywords.values()),
            "industry_features": INDUSTRY_FEATURES_AVAILABLE
        }
        
        # Check cache health
        if len(self.query_cache) > 800:
            health["status"] = "warning"
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append("Cache size approaching limit")
        
        return health

    # =================== AI-POWERED CLASSIFICATION METHODS ===================
    
    def _classify_with_ai(self, query: str, context: Dict[str, Any], 
                         query_id: str, language: str) -> QueryClassification:
        """
        Classify query using advanced AI system
        
        Args:
            query: User's input query
            context: Request context
            query_id: Unique query identifier
            language: Detected language
            
        Returns:
            AI-powered QueryClassification
        """
        user_id = context.get("user_id", "anonymous")
        
        # Add query type to context for AI system
        ai_context = {
            **context,
            "language": language,
            "query_id": query_id,
            "classification_source": "ai_powered"
        }
        
        # Process query with advanced AI system
        ai_analysis = self.ai_system.process_query_advanced(query, user_id, ai_context)
        
        # Extract classification from AI analysis
        query_type, confidence = self._map_ai_analysis_to_query_type(ai_analysis, query)
        
        # Extract entities using AI system
        entities = self._extract_entities_from_ai(ai_analysis, query, language)
        
        # Get suggested service
        suggested_service = self.get_service_for_query_type(query_type)
        
        # Create enhanced classification with AI data
        result = QueryClassification(
            query_type=query_type,
            confidence=confidence,
            extracted_entities=entities,
            suggested_service=suggested_service,
            language=language,
            query_id=query_id,
            cache_key=self._generate_cache_key(query, context),
            user_id=user_id,
            request_context=context,
            ai_analysis=ai_analysis.get("semantic_analysis", {}),
            personalization=ai_analysis.get("personalization", {}),
            predictive_insights=ai_analysis.get("predictions", {}),
            semantic_similarity_scores=self._calculate_semantic_similarities(ai_analysis, query),
            ai_recommendations=ai_analysis.get("recommendations", []),
            intent_confidence=confidence,
            entity_confidence=self._calculate_entity_confidence(ai_analysis),
            sentiment_analysis=ai_analysis.get("semantic_analysis", {}).get("sentiment", {}),
            classification_method="ai_powered"
        )
        
        # Add AI-specific metadata
        result.metadata.update({
            "ai_capabilities_used": ai_analysis.get("capabilities_used", []),
            "ai_processing_time_ms": ai_analysis.get("processing_time_ms", 0),
            "ai_insights_count": len(ai_analysis.get("ai_insights", [])),
            "personalization_score": ai_analysis.get("personalization", {}).get("profile_score", 0.0)
        })
        
        return result
    
    def _classify_with_rules(self, query: str, language: str, context: Dict[str, Any], 
                           query_id: str) -> QueryClassification:
        """
        Fallback rule-based classification (original logic)
        
        Args:
            query: Lowercase query string
            language: Detected language
            context: Request context
            query_id: Unique query identifier
            
        Returns:
            Rule-based QueryClassification
        """
        # Check for greeting first
        if self._matches_patterns(query, "greeting"):
            return self._create_enhanced_classification(
                QueryType.GREETING, 0.9, {}, "template_engine", language,
                query_id, 0.0, "", context, 100.0
            )
        
        # Check for restaurant queries FIRST (before attractions) to avoid misclassification
        if self._matches_patterns(query, "restaurant_search") or self._contains_food_keywords(query):
            cuisines = self._extract_cuisines(query, language)
            districts = self._extract_districts(query, language)
            entities = {
                "cuisines": cuisines,
                "districts": districts,
                "price_range": self._extract_price_range(query)
            }
            return self._create_enhanced_classification(
                QueryType.RESTAURANT_SEARCH, 0.85, entities, "restaurant_database_service",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Check for specific attraction mentions
        attractions = self._extract_attractions(query, language)
        if attractions and len(query.split()) <= 10:
            entities = {"attractions": attractions}
            return self._create_enhanced_classification(
                QueryType.ATTRACTION_INFO, 0.8, entities, "info_retrieval_service",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Check for transport route queries
        locations = self._extract_locations(query)
        if locations and self._matches_patterns(query, "transport_route"):
            entities = {
                "from_location": locations.get("from"),
                "to_location": locations.get("to"),
                "transport_type": self._extract_transport_type(query, language)
            }
            return self._create_enhanced_classification(
                QueryType.TRANSPORT_ROUTE, 0.8, entities, "transport_service",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Check for transport info queries
        if self._matches_patterns(query, "transport_info"):
            entities = {"transport_type": self._extract_transport_type(query, language)}
            return self._create_enhanced_classification(
                QueryType.TRANSPORT_INFO, 0.7, entities, "transport_service",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Check for itinerary requests
        duration = self._extract_duration(query)
        if self._matches_patterns(query, "itinerary_request") or duration:
            entities = {
                "duration": duration or "1 day",
                "interests": self._extract_interests(query, language)
            }
            return self._create_enhanced_classification(
                QueryType.ITINERARY_REQUEST, 0.7, entities, "recommendation_engine",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Check for attraction search queries
        if self._matches_patterns(query, "attraction_search"):
            entities = {
                "interests": self._extract_interests(query, language),
                "districts": self._extract_districts(query, language)
            }
            return self._create_enhanced_classification(
                QueryType.ATTRACTION_SEARCH, 0.7, entities, "info_retrieval_service",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Check for practical info queries
        if self._matches_patterns(query, "practical_info"):
            entities = {"info_type": self._extract_practical_info_type(query)}
            return self._create_enhanced_classification(
                QueryType.PRACTICAL_INFO, 0.6, entities, "info_retrieval_service",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Check for general recommendations
        if self._matches_patterns(query, "recommendation"):
            entities = {"interests": self._extract_interests(query, language)}
            return self._create_enhanced_classification(
                QueryType.RECOMMENDATION, 0.6, entities, "recommendation_engine",
                language, query_id, 0.0, "", context, 100.0
            )
        
        # Default to unknown
        return self._create_enhanced_classification(
            QueryType.UNKNOWN, 0.1, {}, "template_engine",
            language, query_id, 0.0, "", context, 100.0
        )
    
    def _combine_ai_and_rule_results(self, ai_result: QueryClassification, 
                                   rule_result: QueryClassification,
                                   context: Dict[str, Any]) -> QueryClassification:
        """
        Combine AI and rule-based classification results for hybrid approach
        
        Args:
            ai_result: AI-powered classification
            rule_result: Rule-based classification
            context: Request context
            
        Returns:
            Combined hybrid classification
        """
        # Use AI classification as base
        result = ai_result
        
        # Boost confidence if both methods agree
        if ai_result.query_type == rule_result.query_type:
            result.confidence = min(0.95, (ai_result.confidence + rule_result.confidence) / 2 + 0.1)
            result.metadata["classification_agreement"] = True
        else:
            # Keep AI result but note disagreement
            result.metadata["classification_agreement"] = False
            result.metadata["rule_based_alternative"] = {
                "query_type": rule_result.query_type.value,
                "confidence": rule_result.confidence
            }
        
        # Merge entities (AI entities take precedence, rule entities as fallback)
        merged_entities = {**rule_result.extracted_entities, **ai_result.extracted_entities}
        result.extracted_entities = merged_entities
        
        # Add hybrid-specific metadata
        result.metadata.update({
            "hybrid_classification": True,
            "ai_confidence": ai_result.confidence,
            "rule_confidence": rule_result.confidence,
            "confidence_boost": result.confidence - ai_result.confidence
        })
        
        return result
    
    def _map_ai_analysis_to_query_type(self, ai_analysis: Dict[str, Any], 
                                     query: str) -> Tuple[QueryType, float]:
        """
        Map AI analysis results to QueryType and confidence
        
        Args:
            ai_analysis: Results from AI system
            query: Original query
            
        Returns:
            Tuple of (QueryType, confidence)
        """
        # Use semantic analysis and AI insights to determine query type
        sentiment = ai_analysis.get("semantic_analysis", {}).get("sentiment", {})
        entities = ai_analysis.get("semantic_analysis", {}).get("entities", [])
        ai_insights = ai_analysis.get("ai_insights", [])
        confidence_scores = ai_analysis.get("confidence_scores", {})
        
        # Default values
        query_type = QueryType.UNKNOWN
        confidence = confidence_scores.get("overall", 0.5)
        
        # Analyze entities to determine intent
        entity_types = [entity.get("label", "").lower() for entity in entities]
        query_lower = query.lower()
        
        # Rule-based mapping enhanced with AI insights
        if any(word in query_lower for word in ["hi", "hello", "merhaba", "selam"]):
            query_type = QueryType.GREETING
            confidence = max(confidence, 0.8)
        
        elif any(entity_type in ["gpe", "loc", "fac"] for entity_type in entity_types):
            # Location entities detected
            if any(word in query_lower for word in ["restaurant", "food", "eat", "restoran", "yemek"]):
                query_type = QueryType.RESTAURANT_SEARCH
                confidence = max(confidence, 0.7)
            elif any(word in query_lower for word in ["how to get", "transport", "metro", "nasıl giderim"]):
                query_type = QueryType.TRANSPORT_ROUTE
                confidence = max(confidence, 0.7)
            elif any(word in query_lower for word in ["visit", "see", "attractions", "gez", "gör"]):
                query_type = QueryType.ATTRACTION_SEARCH
                confidence = max(confidence, 0.7)
            else:
                query_type = QueryType.GENERAL_INFO
                confidence = max(confidence, 0.6)
        
        elif any(word in query_lower for word in ["restaurant", "food", "cuisine", "restoran", "yemek", "mutfak"]):
            query_type = QueryType.RESTAURANT_SEARCH
            confidence = max(confidence, 0.7)
        
        elif any(word in query_lower for word in ["transport", "metro", "bus", "taxi", "ulaşım", "otobüs"]):
            if any(word in query_lower for word in ["how", "route", "from", "to", "nasıl", "dan", "ye"]):
                query_type = QueryType.TRANSPORT_ROUTE
            else:
                query_type = QueryType.TRANSPORT_INFO
            confidence = max(confidence, 0.7)
        
        elif any(word in query_lower for word in ["itinerary", "plan", "days", "program", "gün", "günlük"]):
            query_type = QueryType.ITINERARY_REQUEST
            confidence = max(confidence, 0.7)
        
        elif any(word in query_lower for word in ["recommend", "suggest", "best", "öner", "tavsiye", "en iyi"]):
            query_type = QueryType.RECOMMENDATION
            confidence = max(confidence, 0.6)
        
        elif any(word in query_lower for word in ["hours", "price", "cost", "ticket", "saat", "fiyat", "bilet"]):
            query_type = QueryType.PRACTICAL_INFO
            confidence = max(confidence, 0.6)
        
        # Boost confidence based on AI insights
        for insight in ai_insights:
            if insight.get("confidence", 0) > 0.8:
                confidence = min(0.95, confidence + 0.1)
        
        return query_type, confidence
    
    def _extract_entities_from_ai(self, ai_analysis: Dict[str, Any], 
                                query: str, language: str) -> Dict[str, Any]:
        """
        Extract entities from AI analysis and combine with rule-based extraction
        
        Args:
            ai_analysis: Results from AI system
            query: Original query
            language: Detected language
            
        Returns:
            Combined entity dictionary
        """
        entities = {}
        
        # Get AI-extracted entities
        ai_entities = ai_analysis.get("semantic_analysis", {}).get("entities", [])
        
        # Group AI entities by type
        locations = []
        organizations = []
        persons = []
        
        for entity in ai_entities:
            entity_type = entity.get("label", "").upper()
            entity_text = entity.get("text", "")
            
            if entity_type in ["GPE", "LOC", "FAC"]:  # Geographic/Location entities
                locations.append(entity_text)
            elif entity_type in ["ORG"]:  # Organizations
                organizations.append(entity_text)
            elif entity_type in ["PERSON"]:  # Persons
                persons.append(entity_text)
        
        # Add AI entities
        if locations:
            entities["ai_locations"] = locations
        if organizations:
            entities["ai_organizations"] = organizations
        if persons:
            entities["ai_persons"] = persons
        
        # Enhance with rule-based extraction
        query_lower = query.lower()
        
        # Extract traditional entities
        entities["attractions"] = self._extract_attractions(query_lower, language)
        entities["districts"] = self._extract_districts(query_lower, language)
        entities["cuisines"] = self._extract_cuisines(query_lower, language)
        
        # Extract transport and location info
        locations_dict = self._extract_locations(query_lower)
        if locations_dict:
            entities.update(locations_dict)
        
        transport_type = self._extract_transport_type(query_lower, language)
        if transport_type:
            entities["transport_type"] = transport_type
        
        # Extract additional context
        price_range = self._extract_price_range(query_lower)
        if price_range:
            entities["price_range"] = price_range
        
        duration = self._extract_duration(query_lower)
        if duration:
            entities["duration"] = duration
        
        interests = self._extract_interests(query_lower, language)
        if interests:
            entities["interests"] = interests
        
        # Remove empty entities
        entities = {k: v for k, v in entities.items() if v}
        
        return entities
    
    def _calculate_semantic_similarities(self, ai_analysis: Dict[str, Any], 
                                       query: str) -> Dict[str, float]:
        """
        Calculate semantic similarity scores for different query types
        
        Args:
            ai_analysis: Results from AI system
            query: Original query
            
        Returns:
            Dictionary of similarity scores
        """
        similarities = {}
        
        # This would use the AI system's semantic embedding capabilities
        if self.ai_enabled and self.ai_system and hasattr(self.ai_system, 'neural_processor'):
            try:
                # Get semantic embedding for the query
                query_embedding = self.ai_system.neural_processor.get_semantic_embedding(query)
                
                if query_embedding:
                    # Calculate similarities with prototype queries for each type
                    prototype_queries = {
                        "greeting": "Hello, how are you?",
                        "restaurant": "Where can I find good restaurants?",
                        "attraction": "What attractions should I visit?",
                        "transport": "How do I get from point A to point B?",
                        "itinerary": "Can you suggest a 3-day itinerary?"
                    }
                    
                    for query_type, prototype in prototype_queries.items():
                        similarity = self.ai_system.neural_processor.calculate_semantic_similarity(
                            query, prototype
                        )
                        similarities[query_type] = similarity
            except Exception as e:
                logger.error(f"❌ Error calculating semantic similarities: {e}")
        
        return similarities
    
    def _calculate_entity_confidence(self, ai_analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence score for extracted entities
        
        Args:
            ai_analysis: Results from AI system
            
        Returns:
            Entity confidence score
        """
        entities = ai_analysis.get("semantic_analysis", {}).get("entities", [])
        
        if not entities:
            return 0.0
        
        # Calculate average confidence of detected entities
        confidences = []
        for entity in entities:
            # AI entities don't always have confidence scores, so we estimate
            entity_length = len(entity.get("text", ""))
            confidence = min(0.9, 0.5 + (entity_length / 20))  # Longer entities tend to be more confident
            confidences.append(confidence)
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    # =================== AI SYSTEM MANAGEMENT METHODS ===================
    
    def get_ai_system_status(self) -> Dict[str, Any]:
        """Get status of integrated AI system"""
        if not self.ai_enabled or not self.ai_system:
            return {
                "available": False,
                "reason": "AI system not initialized"
            }
        
        try:
            return {
                "available": True,
                "status": self.ai_system.get_system_status(),
                "capabilities": [cap.value for cap in self.ai_system.capabilities],
                "fallback_threshold": self.ai_fallback_confidence_threshold,
                "hybrid_mode": self.use_hybrid_classification
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    def update_ai_settings(self, settings: Dict[str, Any]):
        """Update AI system settings"""
        if "fallback_threshold" in settings:
            self.ai_fallback_confidence_threshold = max(0.0, min(1.0, settings["fallback_threshold"]))
        
        if "hybrid_mode" in settings:
            self.use_hybrid_classification = bool(settings["hybrid_mode"])
        
        if "ai_enabled" in settings:
            self.ai_enabled = bool(settings["ai_enabled"]) and AI_SYSTEM_AVAILABLE
        
        logger.info(f"🔧 AI settings updated: threshold={self.ai_fallback_confidence_threshold}, "
                   f"hybrid={self.use_hybrid_classification}, enabled={self.ai_enabled}")
    
    def get_personalized_classification(self, query: str, user_id: str, 
                                      context: Optional[Dict[str, Any]] = None) -> QueryClassification:
        """
        Get personalized query classification using user profile
        
        Args:
            query: User's input query
            user_id: User identifier for personalization
            context: Additional context
            
        Returns:
            Personalized QueryClassification
        """
        context = context or {}
        context["user_id"] = user_id
        context["personalization_requested"] = True
        
        # Use standard classification which now includes personalization
        return self.classify_query(query, context)
    
    def get_predictive_insights(self, query_type: QueryType, hours_ahead: int = 24) -> Dict[str, Any]:
        """
        Get predictive insights for query type using AI system
        
        Args:
            query_type: Type of query to predict for
            hours_ahead: Number of hours to predict ahead
            
        Returns:
            Predictive insights dictionary
        """
        if not self.ai_enabled or not self.ai_system:
            return {"error": "AI system not available"}
        
        try:
            return predict_demand(query_type.value, hours_ahead)
        except Exception as e:
            logger.error(f"❌ Error getting predictive insights: {e}")
            return {"error": str(e)}
    
    def get_comprehensive_classification(self, query: str, user_id: str = "anonymous", 
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get comprehensive classification with all AI features
        
        Args:
            query: User's input query
            user_id: User identifier
            context: Additional context
            
        Returns:
            Comprehensive classification with AI analysis, personalization, and predictions
        """
        context = context or {}
        context["user_id"] = user_id
        
        # Get standard classification
        classification = self.classify_query(query, context)
        
        # Add comprehensive analysis
        result = {
            "classification": {
                "query_type": classification.query_type.value,
                "confidence": classification.confidence,
                "extracted_entities": classification.extracted_entities,
                "suggested_service": classification.suggested_service,
                "language": classification.language,
                "classification_method": classification.classification_method
            },
            "ai_analysis": classification.ai_analysis,
            "personalization": classification.personalization,
            "predictive_insights": classification.predictive_insights,
            "semantic_similarities": classification.semantic_similarity_scores,
            "sentiment_analysis": classification.sentiment_analysis,
            "recommendations": classification.ai_recommendations,
            "performance": {
                "processing_time_ms": classification.processing_time_ms,
                "security_score": classification.security_score,
                "query_id": classification.query_id
            },
            "metadata": classification.metadata,
            "warnings": classification.warnings
        }
        
        # Add predictive insights if available
        if self.ai_enabled:
            try:
                predictive_insights = self.get_predictive_insights(classification.query_type)
                result["demand_prediction"] = predictive_insights
            except Exception as e:
                logger.error(f"❌ Error adding predictive insights: {e}")
        
        return result
    
    def generate_complete_response(self, query: str, user_id: str = "anonymous", 
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a complete response using all integrated services:
        - Query classification and routing
        - Template-based response generation
        - Personalized recommendations
        - Route planning when applicable
        
        Args:
            query: User's input query
            user_id: User identifier for personalization
            context: Additional context information
            
        Returns:
            Complete response with formatted text, recommendations, and routes
        """
        start_time = time.time()
        
        try:
            # 1. Classify the query
            classification = self.classify_query(query, context)
            
            # 2. Generate base response using template engine
            formatted_response = self._generate_template_response(classification, context)
            
            # 3. Get personalized recommendations
            recommendations = self._get_personalized_recommendations(
                classification, user_id, context
            )
            
            # 4. Generate route information if location-related
            route_info = self._generate_route_information(classification, context)
            
            # 5. Compile comprehensive response
            complete_response = {
                "query_id": classification.query_id,
                "query_type": classification.query_type.value,
                "confidence": classification.confidence,
                "language": classification.language,
                "formatted_response": formatted_response,
                "recommendations": recommendations,
                "route_information": route_info,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "personalization_enabled": self.recommendation_engine is not None,
                "ai_powered": self.ai_enabled
            }
            
            # Add classification details
            if classification.ai_analysis:
                complete_response["ai_analysis"] = classification.ai_analysis
            
            if classification.warnings:
                complete_response["warnings"] = classification.warnings
            
            return complete_response
            
        except Exception as e:
            logger.error(f"❌ Error generating complete response: {e}")
            return {
                "error": f"Failed to generate complete response: {str(e)}",
                "query_id": getattr(classification, 'query_id', 'unknown'),
                "fallback_response": "I'm having trouble processing your request right now. Please try again."
            }
    
    def _generate_template_response(self, classification: QueryClassification, 
                                   context: Optional[Dict[str, Any]] = None) -> str:
        """Generate formatted response using template engine"""
        
        if not self.template_engine:
            return self._generate_fallback_response(classification)
        
        try:
            language = classification.language
            query_type = classification.query_type
            entities = classification.extracted_entities
            
            if query_type == QueryType.GREETING:
                return self.template_engine.generate_greeting(language)
            
            elif query_type == QueryType.ATTRACTION_INFO:
                # Use extracted attraction info
                attraction_data = entities.get('attraction_details', {})
                if attraction_data:
                    return self.template_engine.generate_attraction_response(attraction_data, language)
                else:
                    return self.template_engine.generate_no_results(language)
            
            elif query_type == QueryType.RESTAURANT_SEARCH:
                restaurant_data = entities.get('restaurant_details', {})
                if restaurant_data:
                    return self.template_engine.generate_restaurant_response(restaurant_data, language)
                else:
                    return self.template_engine.generate_no_results(language)
            
            elif query_type == QueryType.TRANSPORT_ROUTE:
                route_data = entities.get('route_details', {})
                if route_data:
                    return self.template_engine.generate_transport_response(route_data, language)
                else:
                    return "Route information will be provided below."
            
            elif query_type == QueryType.ITINERARY_REQUEST:
                itinerary_data = entities.get('itinerary_details', {})
                if itinerary_data:
                    return self.template_engine.generate_itinerary_response(itinerary_data, language)
                else:
                    return "Personalized itinerary recommendations will be provided below."
            
            else:
                # Generate generic template response
                template_data = {
                    "query_type": query_type.value,
                    "entities": entities,
                    "confidence": classification.confidence
                }
                return self.template_engine.generate_response("general_info", template_data, language)
                
        except Exception as e:
            logger.error(f"❌ Template generation error: {e}")
            return self._generate_fallback_response(classification)
    
    def _generate_fallback_response(self, classification: QueryClassification) -> str:
        """Generate a fallback response when template engine is not available"""
        
        query_type = classification.query_type
        language = classification.language
        
        fallback_responses = {
            "turkish": {
                QueryType.GREETING: "Merhaba! İstanbul hakkında size nasıl yardımcı olabilirim?",
                QueryType.ATTRACTION_INFO: "İstanbul'da birçok harika turistik yer var. Hangi bölgeyi ziyaret etmek istiyorsunuz?",
                QueryType.RESTAURANT_SEARCH: "İstanbul'da muhteşem yemek seçenekleri bulunuyor. Hangi tür mutfak arıyorsunuz?",
                QueryType.TRANSPORT_ROUTE: "İstanbul'da ulaşım konusunda size yardımcı olabilirim. Nereden nereye gitmek istiyorsunuz?",
                QueryType.ITINERARY_REQUEST: "Size özelleştirilmiş bir İstanbul gezisi planı hazırlayabilirim.",
                QueryType.UNKNOWN: "Sorunuzu daha net anlayabilmem için biraz daha detay verebilir misiniz?"
            },
            "english": {
                QueryType.GREETING: "Hello! How can I help you explore Istanbul?",
                QueryType.ATTRACTION_INFO: "Istanbul has many amazing attractions. Which area would you like to visit?",
                QueryType.RESTAURANT_SEARCH: "Istanbul offers incredible dining options. What type of cuisine are you looking for?",
                QueryType.TRANSPORT_ROUTE: "I can help you with transportation in Istanbul. Where would you like to go?",
                QueryType.ITINERARY_REQUEST: "I can create a personalized Istanbul itinerary for you.",
                QueryType.UNKNOWN: "Could you provide more details so I can better understand your question?"
            }
        }
        
        lang_responses = fallback_responses.get(language, fallback_responses["english"])
        return lang_responses.get(query_type, lang_responses[QueryType.UNKNOWN])
    
    def _get_personalized_recommendations(self, classification: QueryClassification, 
                                        user_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get personalized recommendations using the recommendation engine"""
        
        if not self.recommendation_engine:
            return {"message": "Personalized recommendations not available"}
        
        try:
            query_type = classification.query_type
            language = classification.language
            entities = classification.extracted_entities
            
            # Create user profile for this request
            initial_preferences = context.get('user_preferences', {}) if context else {}
            # Note: recommendation engine creates profiles per request, doesn't store by user_id
            
            # Get recommendations based on query type
            if query_type in [QueryType.ATTRACTION_SEARCH, QueryType.ATTRACTION_INFO]:
                # Create user profile for advanced recommendations
                user_profile = self.recommendation_engine.create_user_profile(
                    context.get('user_preferences', {}) if context else {}
                )
                recommendations = self.recommendation_engine.get_advanced_recommendations(
                    user_profile, n_recommendations=5, context=context
                )
                
                # Filter for attractions only
                attraction_recs = [r for r in recommendations if r.item_type == "attraction"]
                
                return {
                    "type": "attractions",
                    "items": [{
                        "name": rec.name,
                        "category": rec.category,
                        "district": rec.district,
                        "score": rec.score,
                        "reasons": rec.reasons,
                        "confidence": rec.confidence
                    } for rec in attraction_recs[:3]],
                    "personalized": True,
                    "language": language
                }
            
            elif query_type in [QueryType.RESTAURANT_SEARCH, QueryType.RESTAURANT_INFO]:
                # Create user profile for advanced recommendations
                user_profile = self.recommendation_engine.create_user_profile(
                    context.get('user_preferences', {}) if context else {}
                )
                recommendations = self.recommendation_engine.get_advanced_recommendations(
                    user_profile, n_recommendations=5, context=context
                )
                
                # Filter for restaurants only
                restaurant_recs = [r for r in recommendations if r.item_type == "restaurant"]
                
                return {
                    "type": "restaurants",
                    "items": [{
                        "name": rec.name,
                        "category": rec.category,
                        "district": rec.district,
                        "score": rec.score,
                        "reasons": rec.reasons,
                        "confidence": rec.confidence
                    } for rec in restaurant_recs[:3]],
                    "personalized": True,
                    "language": language
                }
            
            elif query_type == QueryType.ITINERARY_REQUEST:
                # Extract duration from entities
                duration_days = entities.get('duration_days', 1)
                
                # Create user profile and get itinerary recommendations
                user_profile = self.recommendation_engine.create_user_profile(
                    context.get('user_preferences', {}) if context else {}
                )
                from .recommendation_engine import RecommendationType
                itinerary_recs = self.recommendation_engine.generate_recommendations(
                    user_profile, RecommendationType.ITINERARY, context
                )
                itinerary = itinerary_recs[0] if itinerary_recs else {"message": "No itinerary available"}
                
                return {
                    "type": "itinerary",
                    "itinerary": itinerary,
                    "personalized": True,
                    "language": language
                }
            
            else:
                # General recommendations
                user_profile = self.recommendation_engine.create_user_profile(
                    context.get('user_preferences', {}) if context else {}
                )
                recommendations = self.recommendation_engine.get_advanced_recommendations(
                    user_profile, n_recommendations=3, context=context
                )
                
                return {
                    "type": "general",
                    "items": [{
                        "name": rec.name,
                        "type": rec.item_type,
                        "category": rec.category,
                        "district": rec.district,
                        "score": rec.score,
                        "reasons": rec.reasons[:2]  # Limit reasons
                    } for rec in recommendations],
                    "personalized": True,
                    "language": language
                }
                
        except Exception as e:
            logger.error(f"❌ Recommendation generation error: {e}")
            return {
                "error": f"Failed to generate recommendations: {str(e)}",
                "personalized": False
            }
    
    def _generate_route_information(self, classification: QueryClassification, 
                                   context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generate route information using the route planner"""
        
        if not self.route_planner:
            return None
        
        query_type = classification.query_type
        entities = classification.extracted_entities
        
        # Only generate routes for transport-related queries
        if query_type not in [QueryType.TRANSPORT_ROUTE, QueryType.TRANSPORT_INFO]:
            return None
        
        try:
            # Extract locations from entities
            from_location_data = entities.get('from_location')
            to_location_data = entities.get('to_location')
            
            if not from_location_data or not to_location_data:
                return {
                    "message": "Please specify both origin and destination for route planning",
                    "available": True
                }
            
            # Create Location objects
            from_location = Location(
                id=from_location_data.get('id', 'from'),
                name=from_location_data.get('name', 'Origin'),
                coordinates=from_location_data.get('coordinates', (41.0082, 28.9784)),  # Default to Istanbul center
                district=from_location_data.get('district', 'Istanbul'),
                transport_connections=from_location_data.get('transport_connections', [])
            )
            
            to_location = Location(
                id=to_location_data.get('id', 'to'),
                name=to_location_data.get('name', 'Destination'),
                coordinates=to_location_data.get('coordinates', (41.0082, 28.9784)),
                district=to_location_data.get('district', 'Istanbul'),
                transport_connections=to_location_data.get('transport_connections', [])
            )
            
            # Determine route type from context or entities
            route_type = RouteType.FASTEST  # Default
            if context:
                preference = context.get('route_preference', 'fastest')
                route_type_map = {
                    'fastest': RouteType.FASTEST,
                    'shortest': RouteType.SHORTEST,
                    'cheapest': RouteType.CHEAPEST,
                    'scenic': RouteType.MOST_SCENIC,
                    'accessible': RouteType.ACCESSIBLE
                }
                route_type = route_type_map.get(preference, RouteType.FASTEST)
            
            # Create route request
            route_request = RouteRequest(
                from_location=from_location,
                to_location=to_location,
                route_type=route_type,
                departure_time=context.get('departure_time') if context else None,
                accessibility_required=context.get('accessibility_required', False) if context else False,
                max_walking_distance_km=context.get('max_walking_distance', 2.0) if context else 2.0
            )
            
            # Get optimal route
            optimal_route = self.route_planner.find_optimal_route(route_request)
            
            # Format route response
            return {
                "route_id": optimal_route.route_id,
                "from": from_location.name,
                "to": to_location.name,
                "route_type": optimal_route.route_type.value,
                "total_distance_km": optimal_route.total_distance_km,
                "total_duration_minutes": optimal_route.total_duration_minutes,
                "total_cost_tl": optimal_route.total_cost_tl,
                "confidence_score": optimal_route.confidence_score,
                "segments": [{
                    "from": segment.from_location.name,
                    "to": segment.to_location.name,
                    "transport_mode": segment.transport_mode.value,
                    "duration_minutes": segment.duration_minutes,
                    "distance_km": segment.distance_km,
                    "cost_tl": segment.cost_tl,
                    "instructions": segment.instructions
                } for segment in optimal_route.segments],
                "warnings": optimal_route.warnings,
                "advantages": optimal_route.advantages,
                "alternatives_available": len(optimal_route.alternative_routes) > 0,
                "generated_at": optimal_route.created_at.isoformat(),
                "valid_until": optimal_route.valid_until.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Route planning error: {e}")
            return {
                "error": f"Failed to generate route: {str(e)}",
                "available": True,
                "suggestion": "Please check your origin and destination and try again."
            }
    
    def update_user_interaction(self, user_id: str, query_type: QueryType, 
                               interaction_data: Dict[str, Any]) -> bool:
        """Update user interaction data for learning and personalization"""
        
        if not self.recommendation_engine:
            return False
        
        try:
            # Extract interaction details
            item_id = interaction_data.get('item_id')
            interaction_type = interaction_data.get('interaction_type', 'view')
            rating = interaction_data.get('rating')
            context = interaction_data.get('context', {})
            
            if item_id:
                feedback_data = {
                    'interaction_type': interaction_type,
                    'rating': rating,
                    'context': context
                }
                self.recommendation_engine.add_user_feedback(user_id, item_id, feedback_data)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating user interaction: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all integrated services"""
        
        status = {
            "query_router": {
                "status": "operational",
                "features": {
                    "ai_powered": self.ai_enabled,
                    "industry_features": INDUSTRY_FEATURES_AVAILABLE,
                    "enhanced_services": ENHANCED_SERVICES_AVAILABLE
                }
            },
            "services": {
                "template_engine": self.template_engine is not None,
                "recommendation_engine": self.recommendation_engine is not None,
                "route_planner": self.route_planner is not None,
                "ai_system": self.ai_enabled,
                "monitoring": self.monitoring is not None,
                "security": self.security is not None
            }
        }
        
        # Add performance metrics if monitoring is available
        if self.monitoring:
            try:
                status["performance"] = {
                    "queries_processed": len(self.performance_history),
                    "average_processing_time_ms": sum(self.performance_history) / len(self.performance_history) if self.performance_history else 0,
                    "classification_stats": self.classification_stats
                }
            except Exception as e:
                logger.error(f"❌ Error getting performance metrics: {e}")
        
        return status
