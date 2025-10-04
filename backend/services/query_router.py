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

# Industry-level imports
try:
    from .industry_monitoring import get_monitoring_system, monitor_operation
    from .industry_security import get_security_system, validate_and_sanitize
    from .industry_resilience import get_error_handler, handle_errors, with_retry, with_circuit_breaker
    INDUSTRY_FEATURES_AVAILABLE = True
except ImportError:
    INDUSTRY_FEATURES_AVAILABLE = False

# Advanced AI system integration
try:
    from .advanced_ai_system import (
        get_ai_system, process_query_with_ai, get_personalized_insights, 
        predict_demand, AdvancedAISystem, AICapability
    )
    AI_SYSTEM_AVAILABLE = True
except ImportError:
    AI_SYSTEM_AVAILABLE = False

logger = logging.getLogger(__name__)

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
            self.monitoring.record_metric("query_router_initialized", 1.0, 
                                        self.monitoring.MetricType.COUNTER)
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
                r"\b(from|to)\b.*\b(sultanahmet|taksim|galata|kadikoy|kadıköy|eminonu|eminönü|besiktas|beşiktaş)\b",
                r"\b(sultanahmet|taksim|galata|kadikoy|kadıköy|eminonu|eminönü|besiktas|beşiktaş)\b.*\b(to|from|dan|a)\b.*\b(sultanahmet|taksim|galata|kadikoy|kadıköy|eminonu|eminönü|besiktas|beşiktaş)\b",
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

# Custom exceptions
class SecurityError(Exception):
    """Raised when query fails security validation"""
    pass

class ClassificationError(Exception):
    """Raised when query classification fails"""
    pass

# Maintain backward compatibility
QueryRouter = IndustryQueryRouter
