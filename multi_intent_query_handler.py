#!/usr/bin/env python3
"""
Multi-Intent Query Handling System for AI Istanbul
Advanced system for detecting and handling multiple intents in complex queries
ENHANCED WITH ATTRACTIONS SUPPORT FOR 78+ ISTANBUL ATTRACTIONS
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime
import logging
from copy import deepcopy

# Initialize logger first
logger = logging.getLogger(__name__)

# Import lightweight deep learning system for enhanced intent classification
try:
    from lightweight_deep_learning import (
        DeepLearningMultiIntentIntegration, 
        LearningContext, 
        LearningMode,
        create_lightweight_deep_learning_system
    )
    DEEP_LEARNING_AVAILABLE = True
    logger.info("🧠 Lightweight Deep Learning System available!")
except ImportError as e:
    logger.warning(f"Deep Learning System not available: {e}")
    DEEP_LEARNING_AVAILABLE = False

# Import attractions system for comprehensive Istanbul attractions support
try:
    from istanbul_attractions_system import IstanbulAttractionsSystem, AttractionCategory, WeatherPreference, BudgetCategory
    ATTRACTIONS_AVAILABLE = True
    logger.info("🏛️ Istanbul Attractions System integrated successfully!")
except ImportError as e:
    logger.warning(f"Istanbul Attractions System not available: {e}")
    ATTRACTIONS_AVAILABLE = False

# Import İKSV Events System for cultural event recommendations
try:
    from monthly_events_scheduler import MonthlyEventsScheduler, get_cached_events, fetch_monthly_events, check_if_fetch_needed
    IKSV_EVENTS_AVAILABLE = True
    logger.info("🎭 İKSV Events System integrated successfully!")
except ImportError as e:
    logger.warning(f"İKSV Events System not available: {e}")
    IKSV_EVENTS_AVAILABLE = False

class IntentType(Enum):
    """Different types of intents that can be detected"""
    LOCATION_SEARCH = "location_search"
    RECOMMENDATION = "recommendation"
    INFORMATION_REQUEST = "information_request"
    ROUTE_PLANNING = "route_planning"
    COMPARISON = "comparison"
    TIME_QUERY = "time_query"
    PRICE_QUERY = "price_query"
    REVIEW_REQUEST = "review_request"
    ACTIVITY_PLANNING = "activity_planning"
    # New attraction-specific intents
    ATTRACTION_SEARCH = "attraction_search"
    CULTURAL_QUERY = "cultural_query"
    FAMILY_ACTIVITY = "family_activity"
    ROMANTIC_SPOT = "romantic_spot"
    HIDDEN_GEM = "hidden_gem"
    EVENTS_QUERY = "events_query"
    CULTURAL_EVENTS = "cultural_events"

@dataclass
@dataclass
class Intent:
    """Individual intent with confidence and parameters"""
    type: IntentType
    confidence: float
    parameters: Dict[str, Any]
    text_span: Tuple[int, int]  # Start and end positions in original text
    priority: int = 1  # 1=high, 2=medium, 3=low
    dependencies: List[str] = None  # Other intent IDs this depends on
    priority_score: float = 0.0  # Boosted score for prioritization (can exceed 1.0)

@dataclass
class MultiIntentResult:
    """Result of multi-intent analysis"""
    primary_intent: Intent
    secondary_intents: List[Intent]
    query_complexity: float
    execution_plan: List[Dict[str, Any]]
    confidence_score: float
    processing_strategy: str
    detected_language: str = 'english'
    response_text: str = ""
    original_query: str = ""  # Store original query for intent handlers

class MultiIntentQueryHandler:
    """
    Advanced multi-intent query handling system
    Detects, prioritizes, and orchestrates multiple intents in complex queries
    """
    
    def __init__(self):
        global ATTRACTIONS_AVAILABLE
        # Initialize lightweight deep learning system for enhanced intent classification
        self.deep_learning_system = None
        if DEEP_LEARNING_AVAILABLE:
            try:
                self.deep_learning_system = create_lightweight_deep_learning_system()
                logger.info("🧠 Lightweight Deep Learning System initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize deep learning system: {e}")
                self.deep_learning_system = None
        
        # Initialize attractions system for comprehensive Istanbul attractions support
        self.attractions_system = None
        if ATTRACTIONS_AVAILABLE:
            try:
                self.attractions_system = IstanbulAttractionsSystem()
                logger.info("🏛️ Istanbul Attractions System initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize attractions system: {e}")
                ATTRACTIONS_AVAILABLE = False
        
        # Initialize İKSV Events System for cultural event recommendations
        global IKSV_EVENTS_AVAILABLE
        self.events_system = None
        if IKSV_EVENTS_AVAILABLE:
            try:
                self.events_system = MonthlyEventsScheduler()
                logger.info("🎭 İKSV Events System initialized successfully!")
            except Exception as e:
                logger.error(f"Failed to initialize İKSV Events System: {e}")
                IKSV_EVENTS_AVAILABLE = False
        
        # Intent detection patterns with priorities
        self.intent_patterns = {
            IntentType.LOCATION_SEARCH: {
                'patterns': [
                    # Strong location-specific patterns (prioritize over recommendation)
                    r'\b(where\s+is|where\s+are|where\s+can\s+i\s+(find|get|eat))\b',
                    r'\b(location\s+of|address\s+of|directions\s+to)\b',
                    r'\b(near|close\s+to|around|vicinity\s+of)\b',
                    r'\b(show\s+me.*around|show\s+me.*near)\b',
                    # "Where" questions that are clearly location-focused
                    r'\b(where.*in\s+(beyoğlu|sultanahmet|kadıköy|taksim|galata))\b',
                    r'\b(where.*near\s+(the|a))\b',
                    # Enhanced disambiguation patterns
                    r'\b(where\s+to\s+(eat|go|find))\b',  # "Where to eat" is location search
                    r'\b(show\s+me\s+(restaurants|places)\s+(around|near))\b',  # "Show me restaurants around" is location
                    r'\b(where\s+can\s+i\s+get)\b',  # "Where can I get" is location-focused
                    r'\b(where.*in\s+the\s+(city|area|historic))\b'  # "Where ... in the city/area" is location
                ],
                'keywords': ['where', 'location', 'address', 'directions', 'near', 'close', 'around', 'vicinity', 'show'],
                'priority': 1,
                # Disambiguation rules to distinguish from recommendation
                'boost_patterns': [
                    r'\b(where\s+(can|do|to)\s+i)\b',  # "Where can I" / "Where to" vs "What are the best"
                    r'\b(show\s+me)\b',  # "Show me" is location-focused
                    r'\b(address|directions|location)\b',  # Explicit location words
                    r'\b(where\s+can\s+i\s+eat)\b',  # Strong location indicator
                    r'\b(where\s+to\s+eat)\b'  # "Where to eat" should be location primary
                ],
                # Patterns that should NOT trigger location intent (favor recommendation instead)
                'negative_patterns': [
                    r'\b(best|good|recommend|suggest|top|great|excellent)\b',  # Quality indicators favor recommendation
                    r'\b(find\s+the\s+best|what\s+are\s+the\s+best)\b',  # "Find the best" is recommendation
                    r'\b(recommend.*in|suggest.*in|good.*in)\b'  # "Recommend X in Y" is recommendation
                ]
            },
            IntentType.RECOMMENDATION: {
                'patterns': [
                    # English patterns only
                    r'\b(recommend|suggest|best|top|good|popular|excellent|amazing)\b',
                    r'\b(should\s+(i|we)\s+(visit|go|see|try|eat|dine))\b',
                    r'\b(what.*worth\s+(visiting|seeing|trying|eating))\b',
                    r'\b(restaurant|food|dining|eat|meal|lunch|dinner|breakfast)\b',
                    r'\b(cafe|bistro|eatery|places?\s+to\s+eat)\b',
                    r'\b(cuisine|culinary|chef|menu|dish)\b',
                    r'\b(vegetarian|vegan|halal|kosher|gluten.free)\b',
                    r'\b(budget|cheap|expensive|affordable|mid.range|luxury)\b',
                    r'\b(seafood|turkish|italian|japanese|mediterranean|ottoman)\b',
                    # Enhanced recommendation patterns that override location
                    r'\b(find\s+(the\s+)?(best|good|great))\b',  # "Find the best" is recommendation
                    r'\b(what\s+are\s+(the\s+)?(best|good|top))\b',  # "What are the best" is recommendation
                    r'\b((best|good)\s+(restaurants|places|food))\b'  # Quality-focused queries
                ],
                'keywords': [
                           # English keywords only
                           'recommend', 'suggest', 'best', 'top', 'good', 'popular', 'should', 'worth',
                           'restaurant', 'food', 'dining', 'eat', 'meal', 'lunch', 'dinner', 'breakfast',
                           'cafe', 'bistro', 'eatery', 'places', 'cuisine', 'culinary', 'chef', 'menu',
                           'vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free', 'budget', 'cheap',
                           'expensive', 'affordable', 'seafood', 'turkish', 'italian', 'japanese',
                           'find', 'what', 'great'
                           ],
                'priority': 1,
                'restaurant_specific': True,  # Flag for restaurant-specific patterns
                # Strong recommendation indicators that should override location patterns
                'boost_patterns': [
                    r'\b(best|good|recommend|suggest|top|great|excellent)\b',  # Quality indicators
                    r'\b(find\s+(the\s+)?(best|good|great))\b',  # "Find the best/good" 
                    r'\b(what\s+are\s+the\s+(best|top|good))\b',  # "What are the best" questions
                    r'\b(where\s+can\s+i\s+get\s+(good|best))\b',  # "Where can I get good" is recommendation
                    r'\b(where.*good|where.*best)\b'  # "Where" + quality = recommendation
                ]
            },
            IntentType.INFORMATION_REQUEST: {
                'patterns': [
                    r'\b(what\s+is|tell\s+me\s+about|information\s+(about|on))\b',
                    r'\b(explain|describe|details\s+(about|of))\b',
                    r'\b(history\s+of|story\s+(behind|of))\b'
                ],
                'keywords': ['what', 'tell', 'information', 'explain', 'describe', 'details', 'history'],
                'priority': 2
            },
            IntentType.ROUTE_PLANNING: {
                'patterns': [
                    # Enhanced route patterns with location keywords (English only)
                    r'\b(how\s+to\s+get|directions?\s+to|route\s+to|way\s+to)\b',
                    r'\b(travel\s+from.*to|go\s+from.*to|drive\s+from.*to)\b',
                    r'\b(plan.*trip|itinerary|journey|path|navigation)\b',
                    # Transportation method patterns
                    r'\b(by\s+(bus|metro|taxi|car|foot|walking|subway|tram))\b',
                    r'\b(public\s+transport|transportation|transit)\b',
                    # Distance and location patterns
                    r'\b(distance\s+(from|to|between)|how\s+far|walking\s+distance)\b',
                    r'\b(nearest\s+(station|stop|metro|bus))\b',
                    r'\b(from\s+[A-Z][a-z]+\s+to\s+[A-Z][a-z]+)\b'  # From Place to Place
                ],
                'keywords': ['directions', 'route', 'travel', 'journey', 'trip', 'itinerary', 'plan', 
                           'distance', 'far', 'nearest', 'transport', 'bus', 'metro', 'taxi', 'walking'],
                'priority': 1,
                # Location-based correction rules
                'correction_rules': {
                    'location_keywords': ['from', 'to', 'near', 'at', 'in', 'around', 'between'],
                    'transport_modes': ['bus', 'metro', 'taxi', 'car', 'walking', 'foot', 'drive']
                }
            },
            IntentType.COMPARISON: {
                'patterns': [
                    # High-priority comparison patterns (boosted detection)
                    r'\b(compare|vs\.?|versus|difference\s+between|which\s+is\s+better)\b',
                    r'\b(better\s+than|worse\s+than|which\s+(one|is)\s+better)\b',
                    r'\b(choose\s+between|decide\s+between|pick\s+between)\b',
                    r'\b(what\'s\s+the\s+difference|how\s+do\s+they\s+compare)\b',
                    r'\b(pros\s+and\s+cons|advantages?\s+(and|vs)\s+disadvantages?)\b',
                    r'\b(which\s+(should|would|do)\s+(i|you|we)\s+(choose|pick|visit))\b',
                    # Enhanced multi-restaurant comparison patterns
                    r'\b(.+)\s+(vs\.?|versus|or)\s+(.+)\s+(restaurant|place|spot|food)\b',
                    r'\b(between\s+.+\s+and\s+.+)\b',
                    # Strong comparison indicators
                    r'\b([a-zA-Z\s]+)\s+(vs\.?|versus)\s+([a-zA-Z\s]+)\b',  # Any "A vs B" pattern
                    r'\b(turkish\s+vs\s+italian|italian\s+vs\s+turkish)\b'  # Cuisine comparisons
                ],
                'keywords': ['compare', 'vs', 'versus', 'difference', 'better', 'worse', 'which', 'choose',
                           'between', 'pros', 'cons', 'advantages', 'disadvantages'],
                'priority': 1,  # Elevated priority for better detection
                # Sub-intents for detailed comparison
                'sub_intents': {
                    'cuisine_comparison': r'\b(cuisine|food|menu|dish|taste)\b',
                    'price_comparison': r'\b(price|cost|expensive|cheap|budget|affordable)\b',
                    'location_comparison': r'\b(location|area|neighborhood|district|near)\b',
                    'rating_comparison': r'\b(rating|review|score|star|quality)\b',
                    'atmosphere_comparison': r'\b(atmosphere|ambiance|vibe|mood|setting)\b',
                    'service_comparison': r'\b(service|staff|waiters|hospitality)\b'
                }
            },
            IntentType.TIME_QUERY: {
                'patterns': [
                    # High-priority time query patterns (English only)
                    r'\b(what\s+time|when\s+do|when\s+does|what.*hours|opening\s+times?|closing\s+times?)\b',
                    # Indirect time patterns for meals and dining times
                    r'\b(for\s+breakfast|breakfast\s+(at|in)|morning\s+(at|in))\b',
                    r'\b(for\s+lunch|lunch\s+(at|in)|afternoon\s+(at|in))\b', 
                    r'\b(for\s+dinner|dinner\s+(at|in)|evening\s+(at|in))\b',
                    r'\b(for\s+brunch|brunch\s+(at|in)|weekend\s+(at|in))\b',
                    # Time-specific context patterns
                    r'\b(weekend\s+hours|weekday\s+hours|holiday\s+hours)\b',
                    r'\b(late\s+night|early\s+morning|all\s+night)\b',
                    r'\b(tonight|today|tomorrow|this\s+weekend)\b',
                    r'\b(schedule|timetable|timing|business\s+hours|hours\s+of\s+operation)\b',
                    r'\b(open\s+(at|until|from)|close\s+(at|until|from)|closed\s+on)\b',
                    r'\b(operating\s+hours|open\s+hours|closing\s+time|opening\s+time)\b',
                    # Strong temporal question patterns
                    r'\b(when\s+do.*open|when\s+do.*close|what\s+time.*open|what\s+time.*close)\b',
                    # Specific temporal references with question context
                    r'\b(open\s+(today|tomorrow|tonight|now|currently))\b',
                    r'\b(hours.*on\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b',
                    r'\b([0-9]{1,2}:\d{2}|[0-9]{1,2}\s*(am|pm|o\'clock))\b',  # Time formats
                    # Enhanced indirect time query patterns
                    r'\b(open\s+for\s+(breakfast|lunch|dinner|brunch))\b',  # "open for breakfast"
                    r'\b((breakfast|lunch|dinner|brunch)\s+(hours|time|timing))\b',  # meal time queries
                    r'\b((places|spots|restaurants)\s+open\s+for\s+(breakfast|lunch|dinner))\b',  # "places open for breakfast"
                    r'\b(early\s+(morning|hours)|late\s+(night|evening|hours))\b',  # time period queries
                    r'\b(weekend\s+(hours|timing)|weekday\s+(hours|timing))\b',  # weekend/weekday hours
                    r'\b(morning\s+(hours|schedule)|evening\s+(hours|schedule))\b',  # time-of-day hours
                    r'\b(24\s*hour|24/7|all\s+night|round\s+the\s+clock)\b',  # 24-hour operations
                    r'\b(right\s+now|at\s+the\s+moment|currently\s+open)\b'  # immediate availability
                ],
                'keywords': ['when', 'time', 'hours', 'open', 'close', 'schedule', 'timing', 'available',
                           'today', 'tomorrow', 'tonight', 'morning', 'afternoon', 'evening', 'weekend',
                           'breakfast', 'lunch', 'dinner', 'brunch', 'weekday', 'holiday',
                           'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                           'now', 'currently', 'am', 'pm', 'oclock', 'breakfast', 'lunch', 'dinner', 'brunch',
                           'early', 'late', 'weekday', 'moment', '24', 'hour', 'round', 'clock'],
                'priority': 1,  # Boosted from 2 to 1 for better time query detection
                # Boost patterns for indirect time queries
                'boost_patterns': [
                    r'\b(for\s+(breakfast|lunch|dinner|brunch))\b',  # "for breakfast" strongly indicates time query
                    r'\b(weekend|weekday|holiday)\s+(hours|timing)\b',  # Weekend/weekday hours
                    r'\b(what\s+time|when|schedule|hours)\b',  # Direct time question words
                    r'\b(open\s+for|places.*open|spots.*open)\b',  # "open for X" patterns
                    r'\b(early|late)\s+(morning|night|dinner|breakfast)\b'  # "early morning", "late dinner"
                ],
                # Time-related correction rules
                'correction_rules': {
                    'temporal_keywords': ['now', 'today', 'tonight', 'tomorrow', 'this', 'next'],
                    'time_formats': [r'\b([0-9]{1,2}:\d{2})\b', r'\b([0-9]{1,2}\s*(am|pm))\b'],
                    'meal_periods': ['breakfast', 'lunch', 'dinner', 'brunch'],
                    'time_periods': ['morning', 'afternoon', 'evening', 'night', 'weekend', 'weekday']
                }
            },
            IntentType.PRICE_QUERY: {
                'patterns': [
                    r'\b(how\s+much|cost|price|fee|ticket\s+price)\b',
                    r'\b(expensive|cheap|budget|affordable)\b',
                    r'\b(entrance\s+fee|admission)\b'
                ],
                'keywords': ['cost', 'price', 'expensive', 'cheap', 'budget', 'fee', 'ticket'],
                'priority': 2
            },

            IntentType.REVIEW_REQUEST: {
                'patterns': [
                    r'\b(review|rating|opinion|experience)\b',
                    r'\b(worth\s+it|recommend\s+it|good\s+or\s+bad)\b',
                    r'\b(feedback|testimonial)\b'
                ],
                'keywords': ['review', 'rating', 'opinion', 'experience', 'worth', 'feedback'],
                'priority': 3
            },
            IntentType.ACTIVITY_PLANNING: {
                'patterns': [
                    r'\b(plan.*day|itinerary|schedule.*visit)\b',
                    r'\b(things\s+to\s+do|activities|attractions)\b',
                    r'\b(spend.*time|visit.*places)\b'
                ],
                'keywords': ['plan', 'itinerary', 'activities', 'attractions', 'things', 'visit'],
                'priority': 1
            },
            # Attraction-specific intent patterns
            IntentType.ATTRACTION_SEARCH: {
                'patterns': [
                    r'\b(what\s+to\s+see|attractions\s+in|visit\s+to|explore\s+in)\b',
                    r'\b(istanbul\s+attractions|sights\s+in|things\s+to\s+see\s+in)\b',
                    r'\b(popular\s+attractions|top\s+sights|must-see\s+places)\b',
                    r'\b(cultural\s+sites|historical\s+places|tourist\s+spots)\b',
                    r'\b(hidden\s+gems|off-the-beaten-path\s+places)\b'
                ],
                'keywords': ['attractions', 'see', 'visit', 'explore', 'istanbul', 'sights', 'things', 'cultural', 'historical', 'tourist', 'hidden', 'gems'],
                'priority': 1,
                # Boost patterns for specific attraction types
                'boost_patterns': [
                    r'\b(cultural\s+sites|historical\s+places|tourist\s+spots)\b',
                    r'\b(hidden\s+gems|off-the-beaten-path\s+places)\b'
                ],
                # Patterns that should NOT trigger attraction intent (favor other intents)
                'negative_patterns': [
                    r'\b(best|good|recommend|suggest|top|great|excellent)\b',  # Quality indicators favor other intents
                    r'\b(find\s+the\s+best|what\s+are\s+the\s+best)\b',  # "Find the best" is other intents
                    r'\b(recommend.*in|suggest.*in|good.*in)\b'  # "Recommend X in Y" is other intents
                ]
            },
            IntentType.CULTURAL_QUERY: {
                'patterns': [
                    r'\b(culture|cultural|historical|heritage|museum|art)\b',
                    r'\b(learn\s+about|discover\s+|explore\s+)\b',
                    r'\b(istanbul\s+culture|local\s+customs|traditions)\b'
                ],
                'keywords': ['culture', 'cultural', 'historical', 'heritage', 'museum', 'art', 'learn', 'discover', 'explore', 'istanbul', 'customs', 'traditions'],
                'priority': 1
            },
            IntentType.FAMILY_ACTIVITY: {
                'patterns': [
                    r'\b(family\s+activities|things\s+to\s+do\s+with\s+family|kid-friendly\s+places)\b',
                    r'\b(family\s+fun|children\s+activities|family\s+attractions)\b'
                ],
                'keywords': ['family', 'activities', 'things', 'do', 'with', 'fun', 'children', 'attractions'],
                'priority': 1
            },
            IntentType.ROMANTIC_SPOT: {
                'patterns': [
                    r'\b(romantic\s+restaurants|dinner\s+for\s+two|couples\s+activities)\b',
                    r'\b(romantic\s+getaways|honeymoon\s+destinations|valentine\'s\s+day\s+ideas)\b'
                ],
                'keywords': ['romantic', 'restaurants', 'dinner', 'for', 'two', 'couples', 'activities', 'getaways', 'honeymoon', 'destinations', 'valentine\'s', 'day', 'ideas'],
                'priority': 1
            },
            IntentType.HIDDEN_GEM: {
                'patterns': [
                    r'\b(hidden\s+gems|off-the-beaten-path\s+places|secret\s+spots)\b',
                    r'\b(local\s+favorites|insider\s+tips|unique\s+experiences)\b'
                ],
                'keywords': ['hidden', 'gems', 'off-the-beaten-path', 'places', 'secret', 'spots', 'local', 'favorites', 'insider', 'tips', 'unique', 'experiences'],
                'priority': 1
            },
            IntentType.EVENTS_QUERY: {
                'patterns': [
                    r'\b(events?\s+(in|at|happening|today|tonight|this\s+week|this\s+month))\b',
                    r'\b(what\'s\s+(on|happening|going\s+on)\s+(today|tonight|this\s+week))\b',
                    r'\b(concerts?|shows?|performances?|exhibitions?|festivals?)\b',
                    r'\b(cultural\s+(events?|activities?|programs?))\b',
                    r'\b(theatre|theater|ballet|opera|dance\s+performances?)\b',
                    r'\b(art\s+(exhibitions?|shows?|galleries?))\b',
                    r'\b(what\'s\s+(happening|on)\s+(at|in)\s+(zorlu|İKSV|iksv))\b',
                    r'\b(İKSV|iksv)\b'
                ],
                'keywords': ['events', 'concerts', 'shows', 'performances', 'exhibitions', 'festivals', 
                           'cultural', 'theatre', 'theater', 'ballet', 'opera', 'dance', 'art', 
                           'happening', 'zorlu', 'İKSV', 'iksv', 'what\'s', 'on', 'today', 'tonight'],
                'priority': 1
            },
            IntentType.CULTURAL_EVENTS: {
                'patterns': [
                    r'\b(cultural\s+(events?|shows?|performances?|exhibitions?))\b',
                    r'\b(arts?\s+(events?|shows?|exhibitions?|festivals?))\b',
                    r'\b(İKSV\s+(events?|shows?|concerts?|performances?))\b',
                    r'\b(zorlu\s+(psm|center|events?|shows?))\b',
                    r'\b(istanbul\s+(theatre|theater|festival|cultural\s+center))\b',
                    r'\b(akm|atatürk\s+cultural\s+center)\b'
                ],
                'keywords': ['cultural', 'arts', 'İKSV', 'iksv', 'zorlu', 'psm', 'istanbul', 
                           'theatre', 'theater', 'festival', 'akm', 'atatürk', 'cultural', 'center'],
                'priority': 1
            }
        }
        
        # Entity patterns for parameter extraction (English only)
        self.entity_patterns = {
            'locations': r'\b([A-Z][a-z]+\s+(?:Mosque|Palace|Tower|Museum|Square|Bridge|Market|Bazaar))\b',
            'time_references': r'\b(morning|afternoon|evening|night|today|tomorrow|weekend|weekday)\b',
            'price_ranges': r'\b(cheap|expensive|budget|luxury|affordable|high-end)\b',
            'food_types': r'\b(Turkish|Ottoman|Mediterranean|seafood|vegetarian|halal)\b',
            'activity_types': r'\b(cultural|historical|shopping|entertainment|nightlife|romantic)\b'
        }
        
        # Intent relationships and dependencies
        self.intent_dependencies = {
            IntentType.ROUTE_PLANNING: [IntentType.LOCATION_SEARCH],
            IntentType.COMPARISON: [IntentType.RECOMMENDATION, IntentType.INFORMATION_REQUEST]
        }
        
        # Query complexity indicators
        self.complexity_indicators = [
            r'\b(and|also|but|however|while|then|after|before)\b',  # Conjunctions
            r'\b(first|second|third|finally|lastly)\b',  # Sequence indicators
            r'\?.*\?',  # Multiple questions
            r'\b(compare|vs|versus)\b',  # Comparison indicators
        ]
    
    def analyze_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> MultiIntentResult:
        """Analyze a query for multiple intents with deep learning enhancement"""
        
        logging.info(f"🔍 Analyzing multi-intent query: {query}")
        
        # Detect query language
        detected_language = self._detect_language(query)
        
        # Create learning context for deep learning system
        learning_context = self._create_learning_context(query, context)
        
        # Detect all intents with deep learning enhancement
        detected_intents = self._detect_intents_enhanced(query, learning_context)
        
        # Validate context alignment
        detected_intents = self._validate_context_alignment(query, detected_intents)
        
        # Extract parameters for each intent (merge with existing parameters from corrections)
        for intent in detected_intents:
            extracted_params = self._extract_parameters(query, intent)
            # Merge extracted parameters with existing ones (corrections take precedence)
            intent.parameters.update(extracted_params)
        
        # Calculate query complexity
        complexity = self._calculate_query_complexity(query, detected_intents)
        
        # Prioritize and organize intents
        primary_intent, secondary_intents = self._prioritize_intents(detected_intents)
        
        # Create execution plan
        execution_plan = self._create_execution_plan(primary_intent, secondary_intents, context)
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(detected_intents)
        
        # Determine processing strategy
        strategy = self._determine_processing_strategy(complexity, len(detected_intents))
        
        result = MultiIntentResult(
            primary_intent=primary_intent,
            secondary_intents=secondary_intents,
            query_complexity=complexity,
            execution_plan=execution_plan,
            confidence_score=confidence,
            processing_strategy=strategy,
            detected_language=detected_language
        )
        
        # Store original query for intent handlers
        result.original_query = query
        
        # Generate response in detected language
        result.response_text = self.generate_response(result, detected_language)
        
        logging.info(f"✅ Multi-intent analysis complete: {len(detected_intents)} intents detected, language: {detected_language}")
        return result
    
    def _detect_intents(self, query: str) -> List[Intent]:
        """Detect all intents in the query with hierarchical rules and multi-label support"""
        
        # Use hybrid approach for better intent classification
        return self._hybrid_intent_classification(query)
    
    def _detect_intents_rule_based(self, query: str) -> List[Intent]:
        """Rule-based intent detection (original method)"""
        
        detected_intents = []
        query_lower = query.lower()
        
        # Multi-label detection: Allow multiple intents to be detected
        intent_scores = {}  # Store all intent scores for comparison
        
        for intent_type, config in self.intent_patterns.items():
            confidence = 0.0
            text_spans = []
            
            # Check negative patterns first (hierarchical rule)
            negative_patterns = config.get('negative_patterns', [])
            has_negative = False
            for neg_pattern in negative_patterns:
                if re.search(neg_pattern, query_lower):
                    has_negative = True
                    break
            
            # If negative pattern found, reduce confidence significantly
            negative_penalty = 0.7 if has_negative else 0.0
            
            # Check pattern matches
            pattern_matches = 0
            for pattern in config['patterns']:
                matches = list(re.finditer(pattern, query_lower))
                if matches:
                    pattern_matches += 1
                    for match in matches:
                        text_spans.append((match.start(), match.end()))
            
            # Check keyword matches
            keyword_matches = 0
            for keyword in config['keywords']:
                if keyword in query_lower:
                    keyword_matches += 1
            
            # Calculate base confidence
            if pattern_matches > 0 or keyword_matches > 0:
                pattern_confidence = min(1.0, pattern_matches * 0.4)
                keyword_confidence = min(0.8, keyword_matches * 0.2)
                confidence = max(pattern_confidence, keyword_confidence)
                
                # Boost confidence for multiple matches
                if pattern_matches > 1 or keyword_matches > 2:
                    confidence = min(1.0, confidence + 0.2)
                
                # PRIORITY BOOSTING LOGIC - Use priority score system (allows > 1.0)
                priority_boost = 0.0
                
                # 1. TIME_QUERY Priority Boost
                if intent_type == IntentType.TIME_QUERY:
                    time_priority_patterns = [r'\b(when\s+do|when\s+does|what\s+time|hours.*open|hours.*close)\b']
                    if any(re.search(pattern, query_lower) for pattern in time_priority_patterns):
                        priority_boost = 0.5  # Strong time boost
                
                # 2. LOCATION_SEARCH Priority Boost (vs recommendation disambiguation)
                elif intent_type == IntentType.LOCATION_SEARCH:
                    boost_patterns = config.get('boost_patterns', [])
                    if any(re.search(pattern, query_lower) for pattern in boost_patterns):
                        priority_boost = 0.4  # Location boost
                
                # 3. COMPARISON Priority Boost
                elif intent_type == IntentType.COMPARISON:
                    comparison_boost_patterns = [r'\b(vs\.?|versus|compare)\b']
                    if any(re.search(pattern, query_lower) for pattern in comparison_boost_patterns):
                        priority_boost = 0.6  # Strong comparison boost
                
                # Apply priority boost (allow exceeding 1.0 for prioritization)
                confidence = confidence + priority_boost
                
                # Restaurant-specific enhancement
                if intent_type == IntentType.RECOMMENDATION and config.get('restaurant_specific', False):
                    restaurant_keywords = ['restaurant', 'food', 'dining', 'eat', 'meal', 'cuisine', 
                                         'vegetarian', 'vegan', 'halal', 'seafood', 'turkish']
                    restaurant_matches = sum(1 for keyword in restaurant_keywords if keyword in query_lower)
                    
                    if restaurant_matches > 0:
                        restaurant_boost = min(0.4, restaurant_matches * 0.15)
                        confidence = min(1.0, confidence + restaurant_boost)
                        
                        # Dietary/cuisine specificity boost
                        specific_terms = ['vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free',
                                        'seafood', 'turkish', 'italian', 'japanese', 'budget', 'luxury']
                        specific_matches = sum(1 for term in specific_terms if term in query_lower)
                        if specific_matches > 0:
                            confidence = min(1.0, confidence + 0.15)
                
                # Apply negative penalty after all boosts
                confidence = max(0.0, confidence - negative_penalty)
                
                # Store intent score for multi-label consideration
                intent_scores[intent_type] = confidence
                
                # Adjusted threshold for better sensitivity
                min_threshold = 0.25 if intent_type in [IntentType.COMPARISON, IntentType.TIME_QUERY] else 0.3
                
                if confidence >= min_threshold:
                    text_span = text_spans[0] if text_spans else (0, len(query))
                    
                    intent = Intent(
                        type=intent_type,
                        confidence=min(1.0, confidence),  # Cap displayed confidence at 1.0
                        parameters={},
                        text_span=text_span,
                        priority=config['priority'],
                        priority_score=confidence  # Store full score for prioritization
                    )
                    detected_intents.append(intent)
        
        # Multi-intent detection: If multiple high-confidence intents exist, keep them
        if len(intent_scores) > 1:
            # Sort by confidence
            sorted_scores = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Special handling for Price/Hours multi-intent scenarios
            has_price_and_time = (
                any(intent_type in [IntentType.PRICE_QUERY, IntentType.TIME_QUERY] for intent_type, _ in sorted_scores) and
                len([it for it, _ in sorted_scores if it in [IntentType.PRICE_QUERY, IntentType.TIME_QUERY]]) >= 1
            )
            
            if has_price_and_time:
                # For Price/Hours scenarios, be more permissive with secondary intents
                primary_score = sorted_scores[0][1]
                for intent_type, score in sorted_scores[1:]:
                    if intent_type in [IntentType.PRICE_QUERY, IntentType.TIME_QUERY, IntentType.RECOMMENDATION]:
                        if score >= 0.3 and score >= primary_score * 0.5:  # Lower threshold for Price/Hours
                            if not any(i.type == intent_type for i in detected_intents):
                                intent = Intent(
                                    type=intent_type,
                                    confidence=min(1.0, score),
                                    parameters={},
                                    text_span=(0, len(query)),
                                    priority=self.intent_patterns[intent_type]['priority'],
                                    priority_score=score
                                )
                                detected_intents.append(intent)
            else:
                # Standard multi-intent logic for other scenarios
                primary_score = sorted_scores[0][1]
                for intent_type, score in sorted_scores[1:]:
                    if score >= 0.4 and score >= primary_score * 0.6:  # Allow strong secondary intents
                        # Check if this intent is already in detected_intents
                        if not any(i.type == intent_type for i in detected_intents):
                            intent = Intent(
                                type=intent_type,
                                confidence=min(1.0, score),
                                parameters={},
                                text_span=(0, len(query)),
                                priority=self.intent_patterns[intent_type]['priority'],
                                priority_score=score
                            )
                            detected_intents.append(intent)
        
        return detected_intents
    
    def _validate_context_alignment(self, query: str, intents: List[Intent]) -> List[Intent]:
        """Validate and adjust intents based on context rules"""
        
        if not intents:
            return intents
        
        validated_intents = []
        query_lower = query.lower()
        
        # Context validation rules
        for intent in intents:
            should_keep = True
            adjusted_confidence = intent.confidence
            
            # Rule 1: Location vs Recommendation disambiguation
            if intent.type == IntentType.LOCATION_SEARCH:
                # Boost if clear location indicators present
                location_boost_patterns = [
                    r'\b(where\s+can\s+i\s+(eat|find|get))\b',
                    r'\b(show\s+me\s+(restaurants|places)\s+(around|near))\b',
                    r'\b(where\s+to\s+(eat|go))\b',
                    r'\b(where.*in\s+the\s+(city|area|historic))\b'
                ]
                if any(re.search(pattern, query_lower) for pattern in location_boost_patterns):
                    adjusted_confidence = min(1.0, adjusted_confidence + 0.6)  # Increased boost
                    # Extra boost for "where to eat" specifically
                    if re.search(r'\b(where\s+to\s+eat)\b', query_lower):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.3)
                
                # Reduce confidence if recommendation indicators present (but less aggressive)
                if any(word in query_lower for word in ['best', 'good', 'recommend', 'suggest', 'top']):
                    adjusted_confidence = max(0.1, adjusted_confidence - 0.2)  # Reduced penalty
            
            elif intent.type == IntentType.RECOMMENDATION:
                # Boost if clear recommendation indicators present
                recommendation_boost_patterns = [
                    r'\b(find\s+(the\s+)?(best|good))\b',
                    r'\b(what\s+are\s+(the\s+)?(best|good|top))\b',
                    r'\b(recommend|suggest)\b'
                ]
                if any(re.search(pattern, query_lower) for pattern in recommendation_boost_patterns):
                    adjusted_confidence = min(1.0, adjusted_confidence + 0.4)
            
            # Rule 2: Enhanced Time query validation with indirect patterns
            elif intent.type == IntentType.TIME_QUERY:
                time_indicators = ['time', 'hours', 'open', 'close', 'schedule']
                indirect_time_indicators = ['breakfast', 'lunch', 'dinner', 'brunch', 'morning', 'evening', 'weekend']
                
                if not any(indicator in query_lower for indicator in time_indicators + indirect_time_indicators):
                    should_keep = False
                else:
                    # Boost confidence for specific time questions
                    if re.search(r'\b(what\s+time)\b', query_lower):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.3)
                    # Boost for indirect time queries
                    if any(re.search(rf'\b(open\s+for\s+{meal}|{meal}\s+(hours|time))\b', query_lower) 
                           for meal in ['breakfast', 'lunch', 'dinner', 'brunch']):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.4)
                    
                    # Strong boost for "right now" / immediate availability queries
                    if re.search(r'\b(right\s+now|open\s+now|currently\s+open|at\s+the\s+moment)\b', query_lower):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.6)
                    
                    # Boost for specific meal timing patterns
                    meal_timing_patterns = [
                        r'\b(places\s+open\s+for\s+(breakfast|lunch|dinner))\b',
                        r'\b((breakfast|lunch|dinner)\s+spots?\s+open)\b'
                    ]
                    if any(re.search(pattern, query_lower) for pattern in meal_timing_patterns):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.5)
            
            # Rule 3: Comparison intent validation
            elif intent.type == IntentType.COMPARISON:
                comparison_indicators = ['vs', 'versus', 'compare', 'better']
                if not any(indicator in query_lower for indicator in comparison_indicators):
                    should_keep = False
                else:
                    # Strong comparison patterns boost confidence
                    if re.search(r'\b(which.*better)\b', query_lower):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.25)
            
            # Rule 4: Recommendation vs Information disambiguation
            elif intent.type == IntentType.RECOMMENDATION:
                # If query has clear recommendation indicators, boost confidence
                rec_indicators = ['best', 'recommend', 'suggest']
                if any(indicator in query_lower for indicator in rec_indicators):
                    adjusted_confidence = min(1.0, adjusted_confidence + 0.2)
                
                # If it's clearly asking for information (not recommendation), penalize
                info_only_patterns = [r'\b(what\s+is)\b']
                if any(re.search(pattern, query_lower) for pattern in info_only_patterns):
                    if not any(indicator in query_lower for indicator in rec_indicators):
                        adjusted_confidence *= 0.6  # Significant penalty
            
            elif intent.type == IntentType.INFORMATION_REQUEST:
                # Boost information requests with clear info indicators
                info_indicators = ['what', 'tell', 'explain']
                if any(indicator in query_lower for indicator in info_indicators):
                    adjusted_confidence = min(1.0, adjusted_confidence + 0.15)
                
                # But penalize if there are strong recommendation indicators
                rec_indicators = ['best', 'recommend']
                if any(indicator in query_lower for indicator in rec_indicators):
                    adjusted_confidence *= 0.7
            
            # Update confidence and add to validated list
            if should_keep and adjusted_confidence >= 0.2:
                intent.confidence = adjusted_confidence
                validated_intents.append(intent)
        
        return validated_intents
    
    def _extract_parameters(self, query: str, intent: Intent) -> Dict[str, Any]:
        """Extract parameters for a specific intent"""
        
        parameters = {}
        query_lower = query.lower()
        
        # Extract entities based on intent type
        if intent.type in [IntentType.LOCATION_SEARCH, IntentType.ROUTE_PLANNING]:
            # Extract location entities
            locations = []
            for match in re.finditer(self.entity_patterns['locations'], query):
                locations.append(match.group(1))
            parameters['locations'] = locations
        
        if intent.type == IntentType.TIME_QUERY:
            # Extract time references
            time_refs = []
            for match in re.finditer(self.entity_patterns['time_references'], query_lower):
                time_refs.append(match.group(1))
            parameters['time_references'] = time_refs
        
        if intent.type in [IntentType.PRICE_QUERY, IntentType.RECOMMENDATION]:
            # Extract price ranges
            price_ranges = []
            for match in re.finditer(self.entity_patterns['price_ranges'], query_lower):
                price_ranges.append(match.group(1))
            parameters['price_ranges'] = price_ranges
        
        if intent.type == IntentType.RECOMMENDATION:
            # Extract food types and activity types
            food_types = []
            for match in re.finditer(self.entity_patterns['food_types'], query_lower):
                food_types.append(match.group(1))
            parameters['food_types'] = food_types
            
            activity_types = []
            for match in re.finditer(self.entity_patterns['activity_types'], query_lower):
                activity_types.append(match.group(1))
            parameters['activity_types'] = activity_types
        
        # Extract numbers (quantities, ratings, etc.)
        numbers = re.findall(r'\b\d+\b', query)
        if numbers:
            parameters['numbers'] = [int(n) for n in numbers]
        
        return parameters
    
    def _calculate_query_complexity(self, query: str, intents: List[Intent]) -> float:
        """Calculate the complexity of the query"""
        
        complexity = 0.0
        
        # Base complexity from number of intents
        complexity += len(intents) * 0.2
        
        # Complexity from query length
        word_count = len(query.split())
        complexity += min(0.3, word_count * 0.02)
        
        # Complexity from structural indicators
        for pattern in self.complexity_indicators:
            matches = len(re.findall(pattern, query.lower()))
            complexity += matches * 0.1
        
        # Complexity from intent dependencies
        for intent in intents:
            if intent.type in self.intent_dependencies:
                complexity += 0.15
        
        return min(1.0, complexity)
    
    def _prioritize_intents(self, intents: List[Intent]) -> Tuple[Intent, List[Intent]]:
        """Prioritize intents and separate primary from secondary"""
        
        if not intents:
            # Create a default general intent
            default_intent = Intent(
                type=IntentType.INFORMATION_REQUEST,
                confidence=0.5,
                parameters={},
                text_span=(0, 0),
                priority=1
            )
            return default_intent, []
        
        # Sort by priority (lower number = higher priority) and confidence
        sorted_intents = sorted(intents, key=lambda x: (x.priority, -x.confidence))
        
        primary_intent = sorted_intents[0]
        secondary_intents = sorted_intents[1:5]  # Limit to top 5 secondary intents
        
        return primary_intent, secondary_intents
    
    def _create_execution_plan(self, primary_intent: Intent, 
                             secondary_intents: List[Intent],
                             context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Create an execution plan for handling multiple intents"""
        
        execution_plan = []
        
        # Add primary intent execution step
        primary_step = {
            'step': 1,
            'intent_type': primary_intent.type.value,
            'action': self._get_action_for_intent(primary_intent),
            'parameters': primary_intent.parameters,
            'confidence': primary_intent.confidence,
            'priority': 'high',
            'dependencies': []
        }
        execution_plan.append(primary_step)
        
        # Add secondary intent steps
        for i, intent in enumerate(secondary_intents):
            # Check if this intent depends on others
            dependencies = []
            if intent.type in self.intent_dependencies:
                for dep_type in self.intent_dependencies[intent.type]:
                    # Check if dependency is satisfied by primary or previous intents
                    if (primary_intent.type == dep_type or 
                        any(si.type == dep_type for si in secondary_intents[:i])):
                        dependencies.append(dep_type.value)
            
            step = {
                'step': i + 2,
                'intent_type': intent.type.value,
                'action': self._get_action_for_intent(intent),
                'parameters': intent.parameters,
                'confidence': intent.confidence,
                'priority': 'medium' if intent.priority <= 2 else 'low',
                'dependencies': dependencies
            }
            execution_plan.append(step)
        
        return execution_plan
    
    def _get_action_for_intent(self, intent: Intent) -> str:
        """Get the appropriate action for an intent type"""
        
        actions = {
            IntentType.LOCATION_SEARCH: "search_locations",
            IntentType.RECOMMENDATION: "generate_recommendations",
            IntentType.INFORMATION_REQUEST: "provide_information",
            IntentType.ROUTE_PLANNING: "plan_route",
            IntentType.COMPARISON: "compare_options",
            IntentType.TIME_QUERY: "check_schedules",
            IntentType.PRICE_QUERY: "get_pricing_info",
            IntentType.REVIEW_REQUEST: "fetch_reviews",
            IntentType.ACTIVITY_PLANNING: "plan_activities",
            # Attraction-specific actions
            IntentType.ATTRACTION_SEARCH: "search_attractions",
            IntentType.CULTURAL_QUERY: "provide_cultural_info",
            IntentType.FAMILY_ACTIVITY: "suggest_family_activities",
            IntentType.ROMANTIC_SPOT: "suggest_romantic_spots",
            IntentType.HIDDEN_GEM: "suggest_hidden_gems",
            # Events-specific actions
            IntentType.EVENTS_QUERY: "search_events",
            IntentType.CULTURAL_EVENTS: "search_cultural_events"
        }
        
        return actions.get(intent.type, "general_response")
    
    def _calculate_overall_confidence(self, intents: List[Intent]) -> float:
        """Calculate overall confidence score for the analysis"""
        
        if not intents:
            return 0.0
        
        # Weight the confidence by intent priority
        total_weighted_confidence = 0
        total_weight = 0
        
        for intent in intents:
            weight = 1.0 / intent.priority  # Higher priority = higher weight
            total_weighted_confidence += intent.confidence * weight
            total_weight += weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _determine_processing_strategy(self, complexity: float, intent_count: int) -> str:
        """Determine the processing strategy based on complexity and intent count"""
        
        if complexity > 0.8 or intent_count > 3:
            return "complex_multi_step"
        elif complexity > 0.5 or intent_count > 1:
            return "standard_multi_intent"
        else:
            return "simple_single_intent"
    
    def execute_multi_intent_plan(self, result: MultiIntentResult, 
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the multi-intent plan (placeholder for actual execution)"""
        
        execution_results = {
            'strategy': result.processing_strategy,
            'steps_completed': 0,
            'results': {},
            'errors': [],
            'total_confidence': result.confidence_score
        }
        
        # Simulate execution of each step
        for step in result.execution_plan:
            step_id = f"step_{step['step']}"
            
            # Check dependencies
            dependencies_met = True
            for dep in step['dependencies']:
                if dep not in execution_results['results']:
                    dependencies_met = False
                    execution_results['errors'].append(
                        f"Dependency not met for {step['intent_type']}: {dep}"
                    )
            
            if dependencies_met:
                # Simulate step execution
                step_result = {
                    'intent_type': step['intent_type'],
                    'action': step['action'],
                    'parameters': step['parameters'],
                    'confidence': step['confidence'],
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat()
                }
                
                execution_results['results'][step_id] = step_result
                execution_results['steps_completed'] += 1
            else:
                step_result = {
                    'intent_type': step['intent_type'],
                    'status': 'failed',
                    'reason': 'dependencies_not_met'
                }
                execution_results['results'][step_id] = step_result
        
        return execution_results
    
    def _detect_language(self, query: str) -> str:
        """Detect the language of the query (English only)"""
        # Always return English since we've removed multilingual support
        return 'english'
    
    def _get_response_templates(self, language: str) -> Dict[str, str]:
        """Get response templates (English only)"""
        
        templates = {
            'recommendation': "🍽️ I'd love to help you find some amazing restaurants! Here are my top recommendations that I think you'll really enjoy:",
            'time_query': "⏰ Let me help you with the timing! Here's what I found about restaurant hours and schedules:", 
            'price_query': "💰 Great question about pricing! Here's what you can expect to spend at these restaurants:",
            'location_search': "📍 I'll help you find exactly where to go! Here are the locations and directions you need:",
            'comparison': "🤔 Let me break down the differences for you! Here's a helpful comparison of your options:",
            'information_request': "ℹ️ I'm happy to share what I know! Here's the information you're looking for:",
            'route_planning': "🗺️ Let me help you get there! Here are the best directions and travel options:",
            'greeting': "👋 Hello! I'm your friendly Istanbul dining guide, and I'm excited to help you discover the city's incredible food scene!",
            # Attraction-specific templates
            'attraction_search': "🏛️ Exploring Istanbul's attractions is a great idea! Here are some top sights and hidden gems you might love:",
            'cultural_query': "🎭 Istanbul is rich in culture and history. Here are some cultural sites and museums you may find interesting:",
            'family_activity': "👨‍👩‍👧‍👦 Family time in Istanbul can be fun and exciting! Here are some family-friendly activities and places:",
            'romantic_spot': "❤️ Looking for a romantic spot? Here are some lovely restaurants and places perfect for couples:",
            'hidden_gem': "💎 Everyone loves a hidden gem! Here are some lesser-known but amazing places to check out in Istanbul:",
            # Events-specific templates  
            'events_query': "🎭 There's always something exciting happening in Istanbul! Here are current events and cultural activities:",
            'cultural_events': "🎨 Istanbul's cultural scene is vibrant! Here are current İKSV and cultural events you might enjoy:"
        }
        
        return templates
    
    def generate_response(self, result: MultiIntentResult, language: str = 'english') -> str:
        """Generate a friendly, contextual response with real data execution"""
        
        templates = self._get_response_templates(language)
        primary_intent = result.primary_intent.type.value
        
        # Get the base template
        if primary_intent in templates:
            response = templates[primary_intent]
        else:
            response = templates.get('information_request', templates['greeting'])
        
        # ✨ NEW: Execute actual handlers for specific intents to get real data
        real_data_response = self._execute_intent_handlers(result)
        if real_data_response:
            response = real_data_response
        
        # Add contextual information based on detected parameters
        context_additions = self._generate_context_additions(result)
        if context_additions:
            response += f"\n\n{context_additions}"
        
        # Add helpful next steps or tips
        next_steps = self._generate_helpful_tips(result)
        if next_steps:
            response += f"\n\n💡 {next_steps}"
        
        # Add confidence indicator for development/testing (but make it friendly)
        confidence_text = f"\n\n✨ I'm {result.confidence_score:.0%} confident this matches what you're looking for!"
        
        return response + confidence_text
    
    def _generate_context_additions(self, result: MultiIntentResult) -> str:
        """Generate contextual additions based on detected parameters and secondary intents"""
        
        additions = []
        primary_intent = result.primary_intent
        
        # Add context based on detected parameters
        if 'locations' in primary_intent.parameters:
            locations = primary_intent.parameters['locations']
            if locations:
                additions.append(f"I notice you're interested in the {locations[0]} area - that's a fantastic choice for dining!")
        
        if 'food_types' in primary_intent.parameters:
            food_types = primary_intent.parameters['food_types']
            if food_types:
                additions.append(f"You mentioned {', '.join(food_types)} - I've focused on places that excel in these cuisines.")
        
        if 'price_ranges' in primary_intent.parameters:
            price_ranges = primary_intent.parameters['price_ranges']
            if 'budget' in str(price_ranges).lower() or 'cheap' in str(price_ranges).lower():
                additions.append("I've prioritized great value options that won't break the bank.")
            elif 'luxury' in str(price_ranges).lower() or 'expensive' in str(price_ranges).lower():
                additions.append("I've selected upscale dining experiences for a special occasion.")
        
        # Add context for secondary intents
        if result.secondary_intents:
            intent_types = [intent.type.value for intent in result.secondary_intents]
            if 'time_query' in intent_types:
                additions.append("I'll also include timing information since you asked about hours.")
            if 'location_search' in intent_types:
                additions.append("I'll make sure to provide clear directions and location details.")
            if 'price_query' in intent_types:
                additions.append("I'll include pricing information to help you plan your budget.")
        
        return ' '.join(additions) if additions else ""
    
    def _generate_helpful_tips(self, result: MultiIntentResult) -> str:
        """Generate helpful tips based on the intent and context"""
        
        primary_intent = result.primary_intent.type
        
        tips = {
            IntentType.RECOMMENDATION: "Would you like more details about any of these restaurants, or do you need directions to get there?",
            IntentType.TIME_QUERY: "Keep in mind that hours can vary on holidays or special occasions. I'd recommend checking their current status before visiting!",
            IntentType.PRICE_QUERY: "Remember that prices in Istanbul can vary by season and location. These are general estimates to help you plan.",
            IntentType.LOCATION_SEARCH: "Istanbul traffic can be unpredictable, so allow extra time for your journey, especially during rush hours!",
            IntentType.COMPARISON: "Each option has its unique charm - would you like more specific details about any of these to help you decide?",
            IntentType.ROUTE_PLANNING: "Don't forget to check the latest public transport schedules, as they can change seasonally!",
            # Attraction-specific tips
            IntentType.ATTRACTION_SEARCH: "Exploring attractions can be exciting! Consider visiting a mix of popular sights and hidden gems.",
            IntentType.CULTURAL_QUERY: "Istanbul has a rich cultural heritage. Don't miss the chance to visit its famous museums and historical sites.",
            IntentType.FAMILY_ACTIVITY: "Istanbul offers many family-friendly activities. Would you like suggestions for indoor or outdoor activities?",
            IntentType.ROMANTIC_SPOT: "For a romantic outing, consider a dinner with a view or a stroll in one of Istanbul's beautiful parks.",
            IntentType.HIDDEN_GEM: "Istanbul is full of hidden gems. Be sure to explore some lesser-known spots for a unique experience.",
            # Events-specific tips
            IntentType.EVENTS_QUERY: "Events in Istanbul are always changing! Check venue websites for the latest schedules and booking information.",
            IntentType.CULTURAL_EVENTS: "İKSV events are very popular - I recommend booking tickets in advance, especially for weekend performances."
        }
        
        return tips.get(primary_intent, "Feel free to ask me anything else about Istanbul's amazing food scene!")
    
    def _detect_sub_intents(self, query: str, intent_type: IntentType) -> List[str]:
        """Detect sub-intents for more granular classification"""
        
        sub_intents = []
        query_lower = query.lower()
        
        # Get sub-intent patterns for this intent type
        config = self.intent_patterns.get(intent_type, {})
        sub_intent_patterns = config.get('sub_intents', {})
        
        for sub_intent_name, pattern in sub_intent_patterns.items():
            if re.search(pattern, query_lower):
                sub_intents.append(sub_intent_name)
        
        return sub_intents
    
    def _apply_rule_based_corrections(self, query: str, detected_intents: List[Intent]) -> List[Intent]:
        """Apply rule-based corrections for time/route logic and other patterns"""
        
        corrected_intents = []
        query_lower = query.lower()
        
        for intent in detected_intents:
            intent_type = intent.type
            config = self.intent_patterns.get(intent_type, {})
            correction_rules = config.get('correction_rules', {})
            
            # Create a copy to modify
            corrected_intent = deepcopy(intent)
            
            # Apply temporal keyword corrections for TIME_QUERY
            if intent_type == IntentType.TIME_QUERY:
                temporal_keywords = correction_rules.get('temporal_keywords', [])
                has_temporal = any(keyword in query_lower for keyword in temporal_keywords)
                
                if has_temporal:
                    # Boost confidence for queries with clear temporal indicators
                    corrected_intent.confidence = min(1.0, intent.confidence + 0.2)
                    corrected_intent.parameters['temporal_context'] = True
                
                # Check for time formats
                time_formats = correction_rules.get('time_formats', [])
                for time_pattern in time_formats:
                    if re.search(time_pattern, query_lower):
                        corrected_intent.confidence = min(1.0, intent.confidence + 0.15)
                        corrected_intent.parameters['specific_time'] = True
                        break
            
            # Apply location keyword corrections for ROUTE_PLANNING
            elif intent_type == IntentType.ROUTE_PLANNING:
                location_keywords = correction_rules.get('location_keywords', [])
                transport_modes = correction_rules.get('transport_modes', [])
                
                has_location = any(keyword in query_lower for keyword in location_keywords)
                has_transport = any(mode in query_lower for mode in transport_modes)
                
                if has_location:
                    corrected_intent.confidence = min(1.0, intent.confidence + 0.25)
                    corrected_intent.parameters['location_context'] = True
                
                if has_transport:
                    corrected_intent.confidence = min(1.0, intent.confidence + 0.15)
                    corrected_intent.parameters['transport_specified'] = True
            
            # Apply comparison corrections
            elif intent_type == IntentType.COMPARISON:
                # Detect sub-intents for comparison
                sub_intents = self._detect_sub_intents(query, intent_type)
                corrected_intent.parameters['comparison_aspects'] = sub_intents
                
                # Boost confidence if multiple comparison aspects detected
                if len(sub_intents) > 1:
                    corrected_intent.confidence = min(1.0, intent.confidence + 0.2)
            
            corrected_intents.append(corrected_intent)
        
        return corrected_intents
    
    def _hybrid_intent_classification(self, query: str) -> List[Intent]:
        """Hybrid model: combine neural classifier with rule-based layer"""
        
        # Step 1: Apply rule-based intent detection (existing method)
        rule_based_intents = self._detect_intents_rule_based(query)
        
        # Step 2: Apply rule-based corrections
        corrected_intents = self._apply_rule_based_corrections(query, rule_based_intents)
        
        # Step 3: Neural enhancement (placeholder for future neural model integration)
        enhanced_intents = self._neural_intent_enhancement(query, corrected_intents)
        
        return enhanced_intents
    
    def _neural_intent_enhancement(self, query: str, rule_intents: List[Intent]) -> List[Intent]:
        """Neural enhancement layer for intent classification (placeholder for future implementation)"""
        
        # For now, apply semantic similarity boosting
        enhanced_intents = []
        
        for intent in rule_intents:
            enhanced_intent = deepcopy(intent)
            
            # Apply semantic similarity boosting for restaurant-related queries
            if 'restaurant' in query.lower() or 'food' in query.lower() or 'eat' in query.lower():
                if intent.type == IntentType.RECOMMENDATION:
                    enhanced_intent.confidence = min(1.0, intent.confidence + 0.1)
                    enhanced_intent.parameters['semantic_boost'] = 'restaurant_context'
            
            # Apply context-aware boosting for location queries
            location_indicators = ['where', 'near', 'at', 'in', 'around', 'close', 'vicinity']
            if any(indicator in query.lower() for indicator in location_indicators):
                if intent.type == IntentType.LOCATION_SEARCH:
                    enhanced_intent.confidence = min(1.0, intent.confidence + 0.15)
                    enhanced_intent.parameters['semantic_boost'] = 'location_context'
            
            enhanced_intents.append(enhanced_intent)
        
        return enhanced_intents
    
    def handle_attraction_query(self, intent: Intent, query: str) -> Dict[str, Any]:
        """Handle attraction-specific queries using the integrated attractions system"""
        
        if not self.attractions_system:
            return {
                'status': 'error',
                'message': 'Attractions system not available',
                'attractions': []
            }
        
        try:
            # Extract query parameters
            query_lower = query.lower()
            
            # Determine query type and parameters
            if intent.type == IntentType.ATTRACTION_SEARCH:
                return self._handle_general_attraction_search(query_lower)
            
            elif intent.type == IntentType.CULTURAL_QUERY:
                return self._handle_cultural_attraction_query(query_lower)
            
            elif intent.type == IntentType.FAMILY_ACTIVITY:
                return self._handle_family_attraction_query(query_lower)
            
            elif intent.type == IntentType.ROMANTIC_SPOT:
                return self._handle_romantic_attraction_query(query_lower)
            
            elif intent.type == IntentType.HIDDEN_GEM:
                return self._handle_hidden_gem_query(query_lower)
            
            elif intent.type == IntentType.EVENTS_QUERY:
                return self._handle_events_query(query_lower)
            
            elif intent.type == IntentType.CULTURAL_EVENTS:
                return self._handle_cultural_events_query(query_lower)
            
            else:
                # General attraction search
                return self._handle_general_attraction_search(query_lower)
                
        except Exception as e:
            logger.error(f"Error handling attraction query: {e}")
            return {
                'status': 'error',
                'message': f'Error processing attractions: {str(e)}',
                'attractions': []
            }
    
    def _handle_general_attraction_search(self, query: str) -> Dict[str, Any]:
        """Handle general attraction search queries"""
        
        # Search attractions based on query
        search_results = self.attractions_system.search_attractions(query)
        
        # Get top attractions if no specific search results
        if not search_results:
            # Get top attractions by category
            categories = [AttractionCategory.HISTORICAL_MONUMENT, AttractionCategory.MUSEUM, 
                         AttractionCategory.RELIGIOUS_SITE, AttractionCategory.VIEWPOINT]
            attractions = []
            for category in categories:
                category_attractions = self.attractions_system.get_attractions_by_category(category)
                attractions.extend(category_attractions[:2])  # Top 2 from each category
        else:
            attractions = [result[0] for result in search_results[:8]]  # Top 8 results
        
        return {
            'status': 'success',
            'message': f'Found {len(attractions)} attractions matching your search',
            'attractions': [self._format_attraction_response(attr) for attr in attractions],
            'total_count': len(attractions)
        }
    
    def _handle_cultural_attraction_query(self, query: str) -> Dict[str, Any]:
        """Handle cultural and historical attraction queries"""
        
        # Get cultural attractions
        cultural_categories = [AttractionCategory.HISTORICAL_MONUMENT, AttractionCategory.MUSEUM, 
                              AttractionCategory.RELIGIOUS_SITE, AttractionCategory.CULTURAL_CENTER]
        
        attractions = []
        for category in cultural_categories:
            category_attractions = self.attractions_system.get_attractions_by_category(category)
            attractions.extend(category_attractions)
        
        # Sort by cultural significance
        attractions = sorted(attractions, key=lambda x: len(x.cultural_significance), reverse=True)
        
        return {
            'status': 'success',
            'message': f'Found {len(attractions)} cultural and historical attractions',
            'attractions': [self._format_attraction_response(attr) for attr in attractions[:10]],
            'total_count': len(attractions)
        }
    
    def _handle_family_attraction_query(self, query: str) -> Dict[str, Any]:
        """Handle family-friendly attraction queries"""
        
        # Get family-friendly attractions
        family_attractions = self.attractions_system.get_family_friendly_attractions()
        
        # Prioritize family attraction category
        family_specific = self.attractions_system.get_attractions_by_category(AttractionCategory.FAMILY_ATTRACTION)
        
        # Combine and deduplicate
        all_attractions = family_specific + [attr for attr in family_attractions if attr not in family_specific]
        
        return {
            'status': 'success',
            'message': f'Found {len(all_attractions)} family-friendly attractions',
            'attractions': [self._format_attraction_response(attr) for attr in all_attractions[:10]],
            'total_count': len(all_attractions)
        }
    
    def _handle_romantic_attraction_query(self, query: str) -> Dict[str, Any]:
        """Handle romantic spot queries"""
        
        # Get romantic attractions
        romantic_attractions = self.attractions_system.get_romantic_attractions()
        
        # Prioritize romantic spot category
        romantic_specific = self.attractions_system.get_attractions_by_category(AttractionCategory.ROMANTIC_SPOT)
        
        # Combine and deduplicate
        all_attractions = romantic_specific + [attr for attr in romantic_attractions if attr not in romantic_specific]
        
        return {
            'status': 'success',
            'message': f'Found {len(all_attractions)} romantic spots',
            'attractions': [self._format_attraction_response(attr) for attr in all_attractions[:8]],
            'total_count': len(all_attractions)
        }
    
    def _handle_hidden_gem_query(self, query: str) -> Dict[str, Any]:
        """Handle hidden gem queries"""
        
        # Get hidden gems
        hidden_gems = self.attractions_system.get_hidden_gems()
        
        # Also get attractions from hidden gem category
        hidden_category = self.attractions_system.get_attractions_by_category(AttractionCategory.HIDDEN_GEM)
        
        # Combine and deduplicate
        all_hidden = hidden_gems + [attr for attr in hidden_category if attr not in hidden_gems]
        
        return {
            'status': 'success',
            'message': f'Found {len(all_hidden)} hidden gems',
            'attractions': [self._format_attraction_response(attr) for attr in all_hidden[:8]],
            'total_count': len(all_hidden)
        }
    
    def _handle_events_query(self, query: str) -> Dict[str, Any]:
        """Handle general events queries"""
        
        if not IKSV_EVENTS_AVAILABLE or not self.events_system:
            return {
                'status': 'error',
                'message': 'İKSV Events system not available',
                'events': []
            }
        
        try:
            # Get cached events
            cached_events = get_cached_events()
            
            if not cached_events and check_if_fetch_needed():
                # Try to fetch fresh events if needed
                try:
                    import asyncio
                    fresh_events = asyncio.run(fetch_monthly_events())
                    if fresh_events:
                        cached_events = fresh_events
                except Exception as e:
                    logger.warning(f"Failed to fetch fresh events: {e}")
            
            if not cached_events:
                return {
                    'status': 'no_events',
                    'message': 'No current İKSV events available',
                    'events': []
                }
            
            # Filter events based on query keywords
            relevant_events = self._filter_events_by_query(cached_events, query)
            
            return {

                'status': 'success',
                'message': f'Found {len(relevant_events)} İKSV events',
                'events': [self._format_event_response(event) for event in relevant_events[:5]],
                'total_count': len(relevant_events),
                'source': 'İKSV Monthly Events'
            }
            
        except Exception as e:
            logger.error(f"Error handling events query: {e}")
            return {
                'status': 'error',
                'message': f'Error processing events: {str(e)}',
                'events': []
            }
    
    def _handle_cultural_events_query(self, query: str) -> Dict[str, Any]:
        """Handle cultural events specific queries"""
        
        if not IKSV_EVENTS_AVAILABLE or not self.events_system:
            return {
                'status': 'error',
                'message': 'İKSV Events system not available',
                'events': []
            }
        
        try:
            # Get cached events
            cached_events = get_cached_events()
            
            if not cached_events:
                return {
                    'status': 'no_events',
                    'message': 'No current İKSV cultural events available',
                    'events': []
                }
            
            # Filter for cultural events (theater, concerts, exhibitions, etc.)
            cultural_keywords = ['concert', 'theatre', 'theater', 'ballet', 'opera', 'dance', 
                               'exhibition', 'art', 'cultural', 'festival', 'performance', 'show']
            
            cultural_events = []
            for event in cached_events:
                event_text = (event.get('title', '') + ' ' + event.get('description', '')).lower()
                if any(keyword in event_text for keyword in cultural_keywords):
                    cultural_events.append(event)
            
            return {
                'status': 'success',
                'message': f'Found {len(cultural_events)} İKSV cultural events',
                'events': [self._format_event_response(event) for event in cultural_events[:5]],
                'total_count': len(cultural_events),
                'source': 'İKSV Cultural Events'
            }
            
        except Exception as e:
            logger.error(f"Error handling cultural events query: {e}")
            return {
                'status': 'error',  
                'message': f'Error processing cultural events: {str(e)}',
                'events': []
            }
    
    def _filter_events_by_query(self, events: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Filter events based on query keywords"""
        
        query_lower = query.lower()
        relevant_events = []
        
        # Extract keywords from query
        event_keywords = ['concert', 'show', 'performance', 'exhibition', 'festival', 
                         'theatre', 'theater', 'ballet', 'opera', 'dance', 'art', 'music']
        
        venue_keywords = ['zorlu', 'psm', 'akm', 'atatürk', 'cultural', 'center', 'salon']
        
        for event in events:
            relevance_score = 0
            event_text = (event.get('title', '') + ' ' + event.get('description', '') + 
                         ' ' + event.get('venue', '')).lower()
            
            # Check for event type keywords
            for keyword in event_keywords:
                if keyword in query_lower and keyword in event_text:
                    relevance_score += 2
            
            # Check for venue keywords
            for keyword in venue_keywords:
                if keyword in query_lower and keyword in event_text:
                    relevance_score += 1
            
            # Check for general event terms
            if any(term in event_text for term in ['event', 'show', 'performance']):
                relevance_score += 1
            
            # Include events with relevance score > 0 or if query is very general
            if relevance_score > 0 or len(query_lower.split()) <= 3:
                event['relevance_score'] = relevance_score
                relevant_events.append(event)
        
        # Sort by relevance score
        relevant_events.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return relevant_events
    
    def _format_event_response(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Format event data for response"""
        
        return {
            'title': event.get('title', 'Untitled Event'),
            'venue': event.get('venue', 'İKSV Venue'),
            'date_str': event.get('date_str', 'Date TBA'),
            'description': event.get('description', ''),
            'category': event.get('category', 'Cultural Event'),
            'price': event.get('price', 'Price varies'),
            'booking_info': event.get('booking_info', 'Contact venue for booking'),
            'venue_district': self._get_venue_district(event.get('venue', '')),
            'event_type': self._classify_event_type(event),
            'relevance_score': event.get('relevance_score', 0)
        }
    
    def _get_venue_district(self, venue_name: str) -> str:
        """Get district for a venue name"""
        
        venue_districts = {
            'zorlu psm': 'Beşiktaş',
            'zorlu psm turkcell stage': 'Beşiktaş', 
            'zorlu psm turkcell platinum stage': 'Beşiktaş',
            'salon iksv': 'Beyoğlu',
            'salon İKSV': 'Beyoğlu',
            'harbiye muhsin ertuğrul stage': 'Şişli',
            'cemal reşit rey concert hall': 'Şişli',
            'lütfi kırdar convention center': 'Şişli',
            'atatürk cultural center': 'Beyoğlu',
            'akm': 'Beyoğlu',
            '29th istanbul theatre festival': 'İstanbul',
            'iksv venue': 'İstanbul',
            'multiple venues': 'İstanbul'
        }
        
        venue_key = venue_name.lower().strip()
        return venue_districts.get(venue_key, 'İstanbul')
    
    def _classify_event_type(self, event: Dict[str, Any]) -> str:
        """Classify event type based on title and description"""
        
        event_text = (event.get('title', '') + ' ' + event.get('description', '')).lower()
        
        if any(term in event_text for term in ['concert', 'music', 'symphony', 'orchestra']):
            return 'Concert'
        elif any(term in event_text for term in ['theatre', 'theater', 'play', 'drama']):
            return 'Theatre'
        elif any(term in event_text for term in ['ballet', 'dance', 'choreography']):
            return 'Dance'
        elif any(term in event_text for term in ['opera', 'operetta']):
            return 'Opera'
        elif any(term in event_text for term in ['exhibition', 'gallery', 'art', 'painting']):
            return 'Exhibition'
        elif any(term in event_text for term in ['festival', 'celebration']):
            return 'Festival'
        else:
            return 'Cultural Event'
    
    def _create_learning_context(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Create learning context for deep learning system"""
        if not DEEP_LEARNING_AVAILABLE or not self.deep_learning_system:
            return None
        
        try:
            # This would create a learning context for the deep learning system
            # For now, return None to use fallback methods
            return None
        except Exception as e:
            logger.error(f"Error creating learning context: {e}")
            return None
    
    def _detect_intents_enhanced(self, query: str, learning_context) -> List[Intent]:
        """Detect intents with enhanced deep learning if available"""
        
        # Use rule-based detection as primary method
        rule_based_intents = self._detect_intents_rule_based(query)
        
        # Enhance with deep learning if available
        if DEEP_LEARNING_AVAILABLE and self.deep_learning_system and learning_context:
            try:
                # This would use deep learning enhancement
                # For now, just return the rule-based results
                return rule_based_intents
            except Exception as e:
                logger.warning(f"Deep learning enhancement failed: {e}")
                return rule_based_intents
        
        return rule_based_intents
    
    def _prioritize_intents(self, intents: List[Intent]) -> Tuple[Intent, List[Intent]]:
        """Prioritize intents and separate primary from secondary"""
        
        if not intents:
            # Create a default general intent
            default_intent = Intent(
                type=IntentType.INFORMATION_REQUEST,
                confidence=0.5,
                parameters={},
                text_span=(0, 0),
                priority=1
            )
            return default_intent, []
        
        # Sort by priority (lower number = higher priority) and confidence
        sorted_intents = sorted(intents, key=lambda x: (x.priority, -x.confidence))
        
        primary_intent = sorted_intents[0]
        secondary_intents = sorted_intents[1:5]  # Limit to top 5 secondary intents
        
        return primary_intent, secondary_intents
    
    def _calculate_overall_confidence(self, intents: List[Intent]) -> float:
        """Calculate overall confidence score for the analysis"""
        
        if not intents:
            return 0.0
        
        # Weight the confidence by intent priority
        total_weighted_confidence = 0
        total_weight = 0
        
        for intent in intents:
            weight = 1.0 / intent.priority  # Higher priority = higher weight
            total_weighted_confidence += intent.confidence * weight
            total_weight += weight
        
        return total_weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _determine_processing_strategy(self, complexity: float, intent_count: int) -> str:
        """Determine the processing strategy based on complexity and intent count"""
        
        if complexity > 0.8 or intent_count > 3:
            return "complex_multi_step"
        elif complexity > 0.5 or intent_count > 1:
            return "standard_multi_intent"
        else:
            return "simple_single_intent"
    
    def _execute_intent_handlers(self, result: MultiIntentResult) -> Optional[str]:
        """Execute actual intent handlers to get real data and format response"""
        
        primary_intent = result.primary_intent
        intent_type = primary_intent.type
        
        try:
            # Handle events-related intents with actual data
            if intent_type == IntentType.EVENTS_QUERY:
                events_data = self._handle_events_query(result.original_query if hasattr(result, 'original_query') else "")
                return self._format_events_response(events_data, "🎭 Here are the current events happening in Istanbul:")
            
            elif intent_type == IntentType.CULTURAL_EVENTS:
                cultural_events_data = self._handle_cultural_events_query(result.original_query if hasattr(result, 'original_query') else "")
                return self._format_events_response(cultural_events_data, "🎨 Here are the current İKSV cultural events you might enjoy:")
            
            # Handle attraction-related intents with actual data
            elif intent_type == IntentType.ATTRACTION_SEARCH:
                attraction_data = self._handle_general_attraction_search(result.original_query if hasattr(result, 'original_query') else "")
                return self._format_attraction_response_text(attraction_data, "🏛️ Here are some amazing Istanbul attractions for you:")
            
            elif intent_type == IntentType.CULTURAL_QUERY:
                cultural_data = self._handle_cultural_attraction_query(result.original_query if hasattr(result, 'original_query') else "")
                return self._format_attraction_response_text(cultural_data, "🎭 Here are Istanbul's rich cultural and historical sites:")
            
            elif intent_type == IntentType.FAMILY_ACTIVITY:
                family_data = self._handle_family_attraction_query(result.original_query if hasattr(result, 'original_query') else "")
                return self._format_attraction_response_text(family_data, "👨‍👩‍👧‍👦 Here are some fantastic family-friendly activities in Istanbul:")
            
            elif intent_type == IntentType.ROMANTIC_SPOT:
                romantic_data = self._handle_romantic_attraction_query(result.original_query if hasattr(result, 'original_query') else "")
                return self._format_attraction_response_text(romantic_data, "❤️ Here are some romantic spots perfect for couples:")
            
            elif intent_type == IntentType.HIDDEN_GEM:
                gem_data = self._handle_hidden_gem_query(result.original_query if hasattr(result, 'original_query') else "")
                return self._format_attraction_response_text(gem_data, "💎 Here are some hidden gems you'll love discovering:")
            
            # For other intents, return None to use template-based responses
            return None
            
        except Exception as e:
            logger.error(f"Error executing intent handlers: {e}")
            return None

    def _format_events_response(self, events_data: Dict[str, Any], intro_text: str) -> str:
        """Format events data into a readable response"""
        
        if events_data.get('status') == 'error':
            return f"{intro_text}\n\n❌ Sorry, I'm having trouble accessing event information right now. Please try again later."
        
        if events_data.get('status') == 'no_events' or not events_data.get('events'):
            return f"{intro_text}\n\n📅 I don't see any current events available at the moment. Events are updated regularly, so please check back soon!"
        
        events = events_data.get('events', [])
        response = intro_text
        
        # Add events information
        response += f"\n\n📋 **{len(events)} Events Found:**\n"
        
        for i, event in enumerate(events[:5], 1):  # Show top 5 events
            title = event.get('title', 'Untitled Event')
            venue = event.get('venue', 'İKSV Venue')
            date_str = event.get('date_str', 'Date TBA')
            description = event.get('description', '')
            
            response += f"\n**{i}. {title}**\n"
            response += f"   📍 **Venue:** {venue}\n"
            response += f"   📅 **Date:** {date_str}\n"
            
            if description and len(description) > 10:
                # Truncate long descriptions
                desc_preview = description[:100] + "..." if len(description) > 100 else description
                response += f"   📝 **About:** {desc_preview}\n"
        
        # Add helpful information
        response += f"\n\n💡 **Tips:**"
        response += f"\n• Book tickets early as İKSV events are very popular"
        response += f"\n• Check venue websites for the latest schedules"
        response += f"\n• Consider arriving early for better seating"
        
        if len(events) > 5:
            response += f"\n\n📝 *Showing 5 of {len(events)} total events*"
        
        return response

    def _format_attraction_response(self, attraction) -> Dict[str, Any]:
        """Format individual attraction data for response"""
        
        return {
            'name': getattr(attraction, 'name', 'Unknown Attraction'),
            'district': getattr(attraction, 'district', 'Istanbul'),
            'category': getattr(attraction, 'category', 'Attraction').value if hasattr(getattr(attraction, 'category', None), 'value') else str(getattr(attraction, 'category', 'Attraction')),
            'rating': getattr(attraction, 'user_rating', 0),
            'description': getattr(attraction, 'description', ''),
            'cultural_significance': getattr(attraction, 'cultural_significance', []),
            'visiting_hours': getattr(attraction, 'visiting_hours', {}),
            'location': {
                'latitude': getattr(attraction, 'latitude', 0),
                'longitude': getattr(attraction, 'longitude', 0)
            },
            'accessibility': getattr(attraction, 'accessibility_features', []),
            'entrance_fee': getattr(attraction, 'entrance_fee', 'Varies')
        }
    
    def _format_attraction_response_text(self, attraction_data: Dict[str, Any], intro_text: str) -> str:
        """Format attraction data into a readable response"""
        
        if attraction_data.get('status') == 'error':
            return f"{intro_text}\n\n❌ Sorry, I'm having trouble accessing attraction information right now. Please try again later."
        
        attractions = attraction_data.get('attractions', [])
        if not attractions:
            return f"{intro_text}\n\n🔍 I couldn't find specific attractions matching your request, but I'd be happy to help you explore Istanbul's amazing sights!"
        
        response = intro_text
        response += f"\n\n📋 **{len(attractions)} Great Options:**\n"
        
        for i, attraction in enumerate(attractions[:5], 1):  # Show top 5 attractions
            name = attraction.get('name', 'Unknown Attraction')
            district = attraction.get('district', 'Istanbul')
            category = attraction.get('category', '')
            rating = attraction.get('rating', 0)
            
            response += f"\n**{i}. {name}**\n"
            response += f"   📍 **Location:** {district}\n"
            
            if category:
                response += f"   🏷️ **Type:** {category}\n"
            
            if rating and rating > 0:
                stars = "⭐" * min(5, int(rating))
                response += f"   {stars} **Rating:** {rating}/5\n"
            
            # Add brief description if available
            description = attraction.get('description', '')
            if description and len(description) > 10:
                desc_preview = description[:80] + "..." if len(description) > 80 else description
                response += f"   📝 {desc_preview}\n"
        
        # Add helpful tips
        response += f"\n\n💡 **Tips:**"
        response += f"\n• Check opening hours before visiting"
        response += f"\n• Consider getting the Museum Pass Istanbul for discounts"
        response += f"\n• Visit early in the day to avoid crowds"
        
        if len(attractions) > 5:
            response += f"\n\n📝 *Showing 5 of {len(attractions)} total attractions*"
        
        return response
