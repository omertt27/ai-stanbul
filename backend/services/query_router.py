"""
Query Router for AI Istanbul System

This service classifies user queries and routes them to appropriate services
without using GPT. Uses keyword matching, patterns, and rule-based classification.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from dataclasses import dataclass

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
    """Result of query classification"""
    query_type: QueryType
    confidence: float
    extracted_entities: Dict[str, Any]
    suggested_service: str
    language: str

class QueryRouter:
    """
    Advanced query classification and routing system that replaces GPT
    with deterministic pattern matching and keyword analysis.
    """
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.keywords = self._load_keywords()
        self.location_patterns = self._load_location_patterns()
        
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
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a user query and extract relevant entities
        
        Args:
            query: User's input query
            
        Returns:
            QueryClassification with type, confidence, and entities
        """
        query_lower = query.lower().strip()
        language = self._detect_language(query_lower)
        
        # Check for greeting first
        if self._matches_patterns(query_lower, "greeting"):
            return QueryClassification(
                query_type=QueryType.GREETING,
                confidence=0.9,
                extracted_entities={},
                suggested_service="template_engine",
                language=language
            )
        
        # Check for restaurant queries FIRST (before attractions) to avoid misclassification
        if self._matches_patterns(query_lower, "restaurant_search") or self._contains_food_keywords(query_lower):
            cuisines = self._extract_cuisines(query_lower, language)
            districts = self._extract_districts(query_lower, language)
            return QueryClassification(
                query_type=QueryType.RESTAURANT_SEARCH,
                confidence=0.85,
                extracted_entities={
                    "cuisines": cuisines,
                    "districts": districts,
                    "price_range": self._extract_price_range(query_lower)
                },
                suggested_service="restaurant_database_service",
                language=language
            )
        
        # Check for specific attraction mentions (only if not a restaurant query)
        attractions = self._extract_attractions(query_lower, language)
        if attractions and len(query_lower.split()) <= 10:  # Short queries about specific places
            return QueryClassification(
                query_type=QueryType.ATTRACTION_INFO,
                confidence=0.8,
                extracted_entities={"attractions": attractions},
                suggested_service="info_retrieval_service",
                language=language
            )
        
        # Check for transport route queries
        locations = self._extract_locations(query_lower)
        if locations and self._matches_patterns(query_lower, "transport_route"):
            return QueryClassification(
                query_type=QueryType.TRANSPORT_ROUTE,
                confidence=0.8,
                extracted_entities={
                    "from_location": locations.get("from"),
                    "to_location": locations.get("to"),
                    "transport_type": self._extract_transport_type(query_lower, language)
                },
                suggested_service="transport_service",
                language=language
            )
        
        # Check for transport info queries
        if self._matches_patterns(query_lower, "transport_info"):
            return QueryClassification(
                query_type=QueryType.TRANSPORT_INFO,
                confidence=0.7,
                extracted_entities={
                    "transport_type": self._extract_transport_type(query_lower, language)
                },
                suggested_service="transport_service",
                language=language
            )
        
        # Check for itinerary requests
        duration = self._extract_duration(query_lower)
        if self._matches_patterns(query_lower, "itinerary_request") or duration:
            return QueryClassification(
                query_type=QueryType.ITINERARY_REQUEST,
                confidence=0.7,
                extracted_entities={
                    "duration": duration or "1 day",
                    "interests": self._extract_interests(query_lower, language)
                },
                suggested_service="recommendation_engine",
                language=language
            )
        
        # Check for attraction search queries
        if self._matches_patterns(query_lower, "attraction_search"):
            return QueryClassification(
                query_type=QueryType.ATTRACTION_SEARCH,
                confidence=0.7,
                extracted_entities={
                    "interests": self._extract_interests(query_lower, language),
                    "districts": self._extract_districts(query_lower, language)
                },
                suggested_service="info_retrieval_service",
                language=language
            )
        
        # Check for practical info queries
        if self._matches_patterns(query_lower, "practical_info"):
            return QueryClassification(
                query_type=QueryType.PRACTICAL_INFO,
                confidence=0.6,
                extracted_entities={
                    "info_type": self._extract_practical_info_type(query_lower)
                },
                suggested_service="info_retrieval_service",
                language=language
            )
        
        # Check for general recommendations
        if self._matches_patterns(query_lower, "recommendation"):
            return QueryClassification(
                query_type=QueryType.RECOMMENDATION,
                confidence=0.6,
                extracted_entities={
                    "interests": self._extract_interests(query_lower, language)
                },
                suggested_service="recommendation_engine",
                language=language
            )
        
        # Default to unknown
        return QueryClassification(
            query_type=QueryType.UNKNOWN,
            confidence=0.1,
            extracted_entities={},
            suggested_service="template_engine",
            language=language
        )
    
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
