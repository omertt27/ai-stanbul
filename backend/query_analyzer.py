#!/usr/bin/env python3
"""
Advanced Query Analysis and Location Context System
=================================================

This module provides intelligent query parsing, location detection, and response templating
to improve query relevance from 2.04/5 to 3.5+/5.
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

class QueryType(Enum):
    """Types of queries for better response targeting"""
    RESTAURANT_SPECIFIC = "restaurant_specific"
    RESTAURANT_GENERAL = "restaurant_general" 
    TRANSPORTATION = "transportation"
    CULTURAL_SITES = "cultural_sites"
    SHOPPING = "shopping"
    ACCOMMODATION = "accommodation"
    NIGHTLIFE = "nightlife"
    WEATHER = "weather"
    PRACTICAL_INFO = "practical_info"
    CULTURAL_ETIQUETTE = "cultural_etiquette"
    LANGUAGE_HELP = "language_help"
    EMERGENCY = "emergency"
    GENERIC = "generic"

class LocationContext(Enum):
    """Istanbul neighborhoods and areas for location-specific responses"""
    SULTANAHMET = "sultanahmet"
    BEYOGLU = "beyoglu"
    KARAKOY = "karakoy" 
    KADIKOY = "kadikoy"
    GALATA = "galata"
    TAKSIM = "taksim"
    EMINONU = "eminonu"
    BALAT = "balat"
    FENER = "fener"
    ORTAKOY = "ortakoy"
    BESIKTAS = "besiktas"
    USKUDAR = "uskudar"
    FATIH = "fatih"
    SISLI = "sisli"
    LEVENT = "levent"
    BOSPHORUS = "bosphorus"
    GOLDEN_HORN = "golden_horn"
    ASIAN_SIDE = "asian_side"
    EUROPEAN_SIDE = "european_side"
    HISTORIC_PENINSULA = "historic_peninsula"
    NONE = "none"

@dataclass
class QueryAnalysis:
    """Structured query analysis result"""
    query_type: QueryType
    location_context: LocationContext
    specific_requests: List[str]
    dietary_restrictions: List[str]
    time_context: Optional[str]
    budget_context: Optional[str]
    group_size: Optional[str]
    keywords: List[str]
    confidence_score: float
    response_template_id: str

class QueryAnalyzer:
    """Advanced query analysis system for Istanbul tourism chatbot"""
    
    def __init__(self):
        self.location_patterns = self._build_location_patterns()
        self.query_type_patterns = self._build_query_type_patterns()
        self.dietary_patterns = self._build_dietary_patterns()
        self.time_patterns = self._build_time_patterns()
        self.budget_patterns = self._build_budget_patterns()
        
    def _build_location_patterns(self) -> Dict[LocationContext, List[str]]:
        """Build comprehensive location detection patterns"""
        return {
            LocationContext.SULTANAHMET: [
                r'\bsultanahmet\b', r'\bsultan ahmet\b', r'\bblue mosque\b', r'\bhagia sophia\b',
                r'\btopkapi\b', r'\bbasilica cistern\b', r'\bhippodrome\b', r'\bgrand bazaar\b'
            ],
            LocationContext.BEYOGLU: [
                r'\bbeyoğlu\b', r'\bbeyoglu\b', r'\bistiklal\b', r'\bgalata tower\b',
                r'\bpera\b', r'\btünel\b', r'\btunel\b', r'\bcihangir\b'
            ],
            LocationContext.KARAKOY: [
                r'\bkaraköy\b', r'\bkarakoy\b', r'\bgalata bridge\b', r'\bkarakoy port\b'
            ],
            LocationContext.KADIKOY: [
                r'\bkadıköy\b', r'\bkadikoy\b', r'\bmoda\b', r'\bbahariye\b', r'\basian side\b'
            ],
            LocationContext.GALATA: [
                r'\bgalata\b', r'\bkarakoy\b', r'\bgalata tower\b', r'\bgalata bridge\b'
            ],
            LocationContext.TAKSIM: [
                r'\btaksim\b', r'\btaksim square\b', r'\bgezi park\b'
            ],
            LocationContext.EMINONU: [
                r'\beminönü\b', r'\beminonu\b', r'\bspice bazaar\b', r'\begyptian bazaar\b',
                r'\bmisir carsisi\b', r'\bgolden horn\b', r'\bferry terminal\b'
            ],
            LocationContext.BALAT: [
                r'\bbalat\b', r'\bfener\b', r'\bjewish quarter\b', r'\bcolorful houses\b'
            ],
            LocationContext.ORTAKOY: [
                r'\bortaköy\b', r'\bortakoy\b', r'\bbosphorus\b', r'\bortakoy mosque\b'
            ],
            LocationContext.BESIKTAS: [
                r'\bbeşiktaş\b', r'\bbesiktas\b', r'\bdolmabahce\b', r'\bbarboros\b'
            ],
            LocationContext.USKUDAR: [
                r'\büsküdar\b', r'\buskudar\b', r'\bmaiden tower\b', r'\basian side\b'
            ],
            LocationContext.BOSPHORUS: [
                r'\bbosphorus\b', r'\bbosporus\b', r'\bstrait\b', r'\bferry\b', r'\bcruise\b'
            ]
        }
    
    def _build_query_type_patterns(self) -> Dict[QueryType, List[str]]:
        """Build query type detection patterns"""
        return {
            QueryType.RESTAURANT_SPECIFIC: [
                r'\brestaurant\b.*\bin\b', r'\beat\b.*\bin\b', r'\bfood\b.*\bnear\b',
                r'\bdining\b.*\bin\b', r'\bmeal\b.*\bin\b', r'\bbest.*restaurant\b'
            ],
            QueryType.RESTAURANT_GENERAL: [
                r'\brestaurant\b', r'\bfood\b', r'\bbreakfast\b', r'\blunch\b', r'\bdinner\b',
                r'\bstreet food\b', r'\bturkish cuisine\b', r'\bseafood\b', r'\bvegetarian\b'
            ],
            QueryType.TRANSPORTATION: [
                r'\btransport\b', r'\bmetro\b', r'\bbus\b', r'\btram\b', r'\bferry\b',
                r'\btaxi\b', r'\bget to\b', r'\bhow to reach\b', r'\btravel from\b', r'\broute\b'
            ],
            QueryType.CULTURAL_SITES: [
                r'\bmuseum\b', r'\bmosque\b', r'\bchurch\b', r'\bpalace\b', r'\bhistoric\b',
                r'\bsites\b', r'\battraction\b', r'\bmonument\b', r'\btour\b', r'\bvisit\b'
            ],
            QueryType.SHOPPING: [
                r'\bshopping\b', r'\bbazaar\b', r'\bmarket\b', r'\bbuy\b', r'\bsouvenir\b',
                r'\bcarpet\b', r'\bceramics\b', r'\bfashion\b', r'\bmall\b'
            ],
            QueryType.ACCOMMODATION: [
                r'\bhotel\b', r'\bhostel\b', r'\bstay\b', r'\baccommodation\b', r'\bloading\b'
            ],
            QueryType.NIGHTLIFE: [
                r'\bnightlife\b', r'\bbar\b', r'\bclub\b', r'\bnight\b', r'\bdrink\b', r'\bevening\b'
            ],
            QueryType.WEATHER: [
                r'\bweather\b', r'\btemperature\b', r'\brain\b', r'\bsnow\b', r'\bclimate\b',
                r'\bpack\b', r'\bclothing\b'
            ],
            QueryType.PRACTICAL_INFO: [
                r'\btip\b', r'\bmoney\b', r'\bcurrency\b', r'\bcard\b', r'\bwifi\b', r'\bsim\b',
                r'\bopen\b.*\bhours\b', r'\bschedule\b', r'\bprice\b', r'\bcost\b'
            ],
            QueryType.CULTURAL_ETIQUETTE: [
                r'\betiquette\b', r'\bcustoms\b', r'\btradition\b', r'\bculture\b', r'\brespect\b',
                r'\bmistake\b', r'\boffensive\b', r'\bappropriate\b'
            ],
            QueryType.LANGUAGE_HELP: [
                r'\bturkish\b.*\bphrase\b', r'\blanguage\b', r'\bspeak\b', r'\btranslate\b',
                r'\benglish\b.*\bspoken\b', r'\bcommunicate\b'
            ]
        }
    
    def _build_dietary_patterns(self) -> List[Tuple[str, str]]:
        """Build dietary restriction detection patterns"""
        return [
            (r'\bvegetarian\b', 'vegetarian'),
            (r'\bvegan\b', 'vegan'),
            (r'\bgluten.*free\b', 'gluten-free'),
            (r'\bhalal\b', 'halal'),
            (r'\bkosher\b', 'kosher'),
            (r'\ballergy\b', 'allergy'),
            (r'\bnut.*allergy\b', 'nut allergy'),
            (r'\bseafood.*allergy\b', 'seafood allergy'),
            (r'\blactose.*intolerant\b', 'lactose intolerant'),
            (r'\bdiabetic\b', 'diabetic')
        ]
    
    def _build_time_patterns(self) -> List[Tuple[str, str]]:
        """Build time context detection patterns"""
        return [
            (r'\btonight\b', 'tonight'),
            (r'\btomorrow\b', 'tomorrow'),
            (r'\btoday\b', 'today'),
            (r'\bthis.*evening\b', 'this evening'),
            (r'\bthis.*morning\b', 'this morning'),
            (r'\bthis.*afternoon\b', 'this afternoon'),
            (r'\bweekend\b', 'weekend'),
            (r'\b(\d+).*hours?\b', 'specific duration'),
            (r'\b(\d+).*days?\b', 'multi-day'),
            (r'\bnow\b', 'immediate')
        ]
    
    def _build_budget_patterns(self) -> List[Tuple[str, str]]:
        """Build budget context detection patterns"""
        return [
            (r'\bbudget\b', 'budget-conscious'),
            (r'\bcheap\b', 'budget-conscious'),
            (r'\binexpensive\b', 'budget-conscious'),
            (r'\baffordable\b', 'budget-conscious'),
            (r'\bluxury\b', 'luxury'),
            (r'\bupscale\b', 'upscale'),
            (r'\bexpensive\b', 'high-end'),
            (r'\bhigh.*end\b', 'high-end'),
            (r'\bmid.*range\b', 'mid-range'),
            (r'\bmoderate\b', 'moderate')
        ]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Perform comprehensive query analysis"""
        query_lower = query.lower().strip()
        
        # Detect location context
        location_context = self._detect_location(query_lower)
        
        # Detect query type  
        query_type = self._detect_query_type(query_lower)
        
        # Extract specific requests
        specific_requests = self._extract_specific_requests(query_lower)
        
        # Detect dietary restrictions
        dietary_restrictions = self._detect_dietary_restrictions(query_lower)
        
        # Detect time context
        time_context = self._detect_time_context(query_lower)
        
        # Detect budget context
        budget_context = self._detect_budget_context(query_lower)
        
        # Extract group size
        group_size = self._detect_group_size(query_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            location_context, query_type, specific_requests, keywords
        )
        
        # Determine response template ID
        response_template_id = self._get_response_template_id(
            query_type, location_context, specific_requests
        )
        
        return QueryAnalysis(
            query_type=query_type,
            location_context=location_context,
            specific_requests=specific_requests,
            dietary_restrictions=dietary_restrictions,
            time_context=time_context,
            budget_context=budget_context,
            group_size=group_size,
            keywords=keywords,
            confidence_score=confidence_score,
            response_template_id=response_template_id
        )
    
    def _detect_location(self, query: str) -> LocationContext:
        """Detect specific Istanbul location mentioned in query"""
        for location, patterns in self.location_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return location
        return LocationContext.NONE
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of query for better response targeting"""
        query_scores = {}
        
        for query_type, patterns in self.query_type_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            query_scores[query_type] = score
        
        if not any(query_scores.values()):
            return QueryType.GENERIC
        
        return max(query_scores, key=query_scores.get)
    
    def _extract_specific_requests(self, query: str) -> List[str]:
        """Extract specific requests from the query"""
        specific_requests = []
        
        # Look for question words and their objects
        question_patterns = [
            (r'\bwhat.*best\b.*?(?:\?|$)', 'recommendation request'),
            (r'\bwhere.*can.*i\b.*?(?:\?|$)', 'location request'),
            (r'\bhow.*to\b.*?(?:\?|$)', 'instruction request'),
            (r'\bwhen.*is.*best\b.*?(?:\?|$)', 'timing request'),
            (r'\bcan.*you.*recommend\b.*?(?:\?|$)', 'recommendation request'),
            (r'\bi.*want.*to\b.*?(?:\?|$)', 'desire statement'),
            (r'\bi.*need\b.*?(?:\?|$)', 'need statement'),
            (r'\bi.*looking.*for\b.*?(?:\?|$)', 'search request')
        ]
        
        for pattern, request_type in question_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                specific_requests.append(request_type)
        
        return specific_requests
    
    def _detect_dietary_restrictions(self, query: str) -> List[str]:
        """Detect dietary restrictions and allergies"""
        restrictions = []
        
        for pattern, restriction in self.dietary_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                restrictions.append(restriction)
        
        return restrictions
    
    def _detect_time_context(self, query: str) -> Optional[str]:
        """Detect time-related context in the query"""
        for pattern, time_context in self.time_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return time_context
        return None
    
    def _detect_budget_context(self, query: str) -> Optional[str]:
        """Detect budget-related context in the query"""
        for pattern, budget_context in self.budget_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return budget_context
        return None
    
    def _detect_group_size(self, query: str) -> Optional[str]:
        """Detect group size from the query"""
        group_patterns = [
            (r'\balone\b', 'solo'),
            (r'\bcouple\b', 'couple'),
            (r'\bfamily\b', 'family'),
            (r'\bgroup\b', 'group'),
            (r'\bfriends\b', 'friends'),
            (r'\b(\d+)\s*people\b', 'specific number'),
            (r'\b(\d+)\s*person\b', 'specific number')
        ]
        
        for pattern, group_type in group_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return group_type
        return None
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Remove common words and extract meaningful terms
        stop_words = {
            'i', 'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'shall', 'what', 'when', 'where', 'who',
            'why', 'how', 'this', 'that', 'these', 'those', 'my', 'your', 'his',
            'her', 'its', 'our', 'their'
        }
        
        # Extract words, filter stop words, and find meaningful keywords
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _calculate_confidence_score(self, location_context: LocationContext, 
                                  query_type: QueryType, specific_requests: List[str],
                                  keywords: List[str]) -> float:
        """Calculate confidence score for the analysis"""
        score = 0.0
        
        # Location context adds confidence
        if location_context != LocationContext.NONE:
            score += 0.3
        
        # Specific query type adds confidence
        if query_type != QueryType.GENERIC:
            score += 0.3
        
        # Specific requests add confidence
        if specific_requests:
            score += 0.2
        
        # Keywords add confidence
        if len(keywords) >= 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_response_template_id(self, query_type: QueryType, 
                                location_context: LocationContext,
                                specific_requests: List[str]) -> str:
        """Determine the appropriate response template ID"""
        template_id = f"{query_type.value}"
        
        if location_context != LocationContext.NONE:
            template_id += f"_{location_context.value}"
        
        if 'recommendation request' in specific_requests:
            template_id += "_recommendations"
        elif 'instruction request' in specific_requests:
            template_id += "_instructions"
        elif 'location request' in specific_requests:
            template_id += "_locations"
        
        return template_id

class ResponseTemplateEngine:
    """Generate targeted responses based on query analysis"""
    
    def __init__(self):
        self.templates = self._build_response_templates()
        self.location_data = self._build_location_data()
    
    def _build_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive response templates"""
        return {
            "restaurant_specific_sultanahmet": {
                "intro": "For dining in Sultanahmet, here are the best options in the historic district:",
                "focus": "authentic Ottoman cuisine, historic setting, tourist-friendly",
                "include_walking_distances": True,
                "include_landmarks": ["Hagia Sophia", "Blue Mosque", "Topkapi Palace"],
                "practical_tips": ["Make reservations", "Try traditional Turkish breakfast", "Check if halal certified"]
            },
            "restaurant_specific_beyoglu": {
                "intro": "Beyoğlu offers fantastic dining with modern Turkish and international cuisine:",
                "focus": "modern dining, rooftop terraces, trendy atmosphere", 
                "include_walking_distances": True,
                "include_landmarks": ["Istiklal Street", "Galata Tower", "Taksim Square"],
                "practical_tips": ["Evening reservations recommended", "Try meyhane culture", "Check dress code for upscale places"]
            },
            "transportation_specific": {
                "intro": "Here's the best way to get around Istanbul:",
                "focus": "metro routes, ferry schedules, practical steps",
                "include_walking_distances": True,
                "include_istanbulkart": True,
                "practical_tips": ["Get Istanbulkart for all public transport", "Download BiTaksi app", "Ferries offer scenic routes"]
            },
            "cultural_sites_sultanahmet": {
                "intro": "Sultanahmet is the heart of historic Istanbul with world-famous sites:",
                "focus": "Byzantine and Ottoman history, opening hours, ticket info",
                "include_walking_distances": True,
                "include_landmarks": ["Hagia Sophia", "Blue Mosque", "Topkapi Palace", "Basilica Cistern"],
                "practical_tips": ["Start early to avoid crowds", "Dress modestly for mosques", "Museum pass saves money"]
            },
            "shopping_traditional": {
                "intro": "For authentic Turkish shopping experiences:",
                "focus": "traditional crafts, negotiation tips, authentic products",
                "include_walking_distances": True,
                "practical_tips": ["Bargaining is expected", "Check authenticity certificates", "Compare prices between shops"]
            }
        }
    
    def _build_location_data(self) -> Dict[LocationContext, Dict[str, Any]]:
        """Build location-specific data for responses"""
        return {
            LocationContext.SULTANAHMET: {
                "description": "Historic district and tourist center",
                "metro_stations": ["Sultanahmet", "Eminönü"],
                "tram_stops": ["Sultanahmet", "Gülhane", "Eminönü"],
                "walking_distances": {
                    "Hagia Sophia to Blue Mosque": "2 minutes",
                    "Blue Mosque to Topkapi Palace": "5 minutes",
                    "Sultanahmet to Grand Bazaar": "8 minutes"
                },
                "local_tips": ["Very crowded in summer", "Many tourist restaurants", "Lots of historical sites within walking distance"]
            },
            LocationContext.BEYOGLU: {
                "description": "Modern cultural district with nightlife",
                "metro_stations": ["Şişhane", "Vezneciler"],
                "funicular": ["Tünel"],
                "walking_distances": {
                    "Taksim to Istiklal Street": "0 minutes (same location)",
                    "Istiklal Street to Galata Tower": "10 minutes",
                    "Karaköy to Galata Tower": "5 minutes uphill"
                },
                "local_tips": ["Great nightlife", "Trendy restaurants", "Art galleries and bookshops"]
            },
            LocationContext.KADIKOY: {
                "description": "Authentic Asian side neighborhood",
                "ferry_terminal": "Kadıköy",
                "metro_stations": ["Kadıköy"],
                "walking_distances": {
                    "Kadıköy ferry to Bahariye Street": "3 minutes",
                    "Kadıköy to Moda": "15 minutes",
                    "Moda to seaside": "5 minutes"
                },
                "local_tips": ["More local, less touristy", "Great for authentic food", "Beautiful Moda neighborhood"]
            }
        }
    
    def enhance_gpt_prompt(self, query: str, analysis: QueryAnalysis) -> str:
        """Enhance GPT prompt with query analysis insights"""
        base_prompt = f"Answer this specific question about Istanbul: {query}\n\n"
        
        # Add location context
        if analysis.location_context != LocationContext.NONE:
            location_data = self.location_data.get(analysis.location_context, {})
            base_prompt += f"FOCUS AREA: {analysis.location_context.value.title()}\n"
            base_prompt += f"Location context: {location_data.get('description', '')}\n"
            
            # Add walking distances if available
            if 'walking_distances' in location_data:
                base_prompt += "Include relevant walking distances:\n"
                for route, time in location_data['walking_distances'].items():
                    base_prompt += f"- {route}: {time}\n"
            
            # Add transportation info
            if 'metro_stations' in location_data:
                base_prompt += f"Nearby metro: {', '.join(location_data['metro_stations'])}\n"
            if 'tram_stops' in location_data:
                base_prompt += f"Tram stops: {', '.join(location_data['tram_stops'])}\n"
                
        # Add query type specific instructions
        if analysis.query_type == QueryType.RESTAURANT_SPECIFIC:
            base_prompt += "\nProvide specific restaurant recommendations with:\n"
            base_prompt += "- Exact names and brief descriptions\n"
            base_prompt += "- Walking distances to major landmarks\n"
            base_prompt += "- Cuisine type and atmosphere\n"
            base_prompt += "- Practical dining tips\n"
            
        elif analysis.query_type == QueryType.TRANSPORTATION:
            base_prompt += "\nProvide step-by-step transportation instructions:\n"
            base_prompt += "- Specific metro/tram/bus routes\n"
            base_prompt += "- Station names and transfer points\n"
            base_prompt += "- Approximate travel times\n"
            base_prompt += "- Alternative routes\n"
            
        elif analysis.query_type == QueryType.CULTURAL_SITES:
            base_prompt += "\nProvide cultural site information with:\n"
            base_prompt += "- Historical context and significance\n"
            base_prompt += "- Opening hours and ticket information\n"
            base_prompt += "- Visiting tips and etiquette\n"
            base_prompt += "- Best times to visit\n"
            
        # Add dietary restrictions
        if analysis.dietary_restrictions:
            base_prompt += f"\nIMPORTANT: Address these dietary needs: {', '.join(analysis.dietary_restrictions)}\n"
            
        # Add time context
        if analysis.time_context:
            base_prompt += f"\nTime context: {analysis.time_context} - provide time-appropriate suggestions\n"
            
        # Add budget context
        if analysis.budget_context:
            base_prompt += f"\nBudget preference: {analysis.budget_context} - focus on appropriate price ranges\n"
            
        # Add specific request instructions
        if 'recommendation request' in analysis.specific_requests:
            base_prompt += "\nProvide clear, ranked recommendations with reasons why each is good.\n"
        if 'instruction request' in analysis.specific_requests:
            base_prompt += "\nProvide clear, step-by-step instructions.\n"
        if 'location request' in analysis.specific_requests:
            base_prompt += "\nProvide specific locations with addresses/directions.\n"
            
        base_prompt += "\nMake your response directly relevant to the specific question asked. Be precise and practical."
        
        return base_prompt

# Global analyzer instance
query_analyzer = QueryAnalyzer()
response_template_engine = ResponseTemplateEngine()

def analyze_and_enhance_query(query: str) -> Tuple[QueryAnalysis, str]:
    """Convenience function to analyze query and get enhanced prompt"""
    analysis = query_analyzer.analyze_query(query)
    enhanced_prompt = response_template_engine.enhance_gpt_prompt(query, analysis)
    return analysis, enhanced_prompt

# Test the analyzer
if __name__ == "__main__":
    test_queries = [
        "I'm staying in Sultanahmet and want authentic Turkish breakfast. What are the best traditional places nearby?",
        "How do I get from Istanbul Airport to Sultanahmet using public transportation?",
        "I'm vegetarian and staying in Beyoğlu. Where can I find good vegetarian Turkish food?",
        "What are the must-try street foods in Istanbul?"
    ]
    
    print("🔍 QUERY ANALYSIS TEST RESULTS")
    print("=" * 50)
    
    for query in test_queries:
        analysis, enhanced_prompt = analyze_and_enhance_query(query)
        print(f"\n📝 Query: {query}")
        print(f"🎯 Type: {analysis.query_type.value}")
        print(f"📍 Location: {analysis.location_context.value}")
        print(f"🔍 Requests: {analysis.specific_requests}")
        print(f"🥗 Dietary: {analysis.dietary_restrictions}")
        print(f"⏰ Time: {analysis.time_context}")
        print(f"💰 Budget: {analysis.budget_context}")
        print(f"📊 Confidence: {analysis.confidence_score:.2f}")
        print(f"🎫 Template: {analysis.response_template_id}")
        print("-" * 40)
