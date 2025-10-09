#!/usr/bin/env python3
"""
Multi-Intent Query Handling System for AI Istanbul
Advanced system for detecting and handling multiple intents in complex queries
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

class IntentType(Enum):
    """Different types of intents that can be detected"""
    LOCATION_SEARCH = "location_search"
    RECOMMENDATION = "recommendation"
    INFORMATION_REQUEST = "information_request"
    ROUTE_PLANNING = "route_planning"
    COMPARISON = "comparison"
    BOOKING = "booking"
    TIME_QUERY = "time_query"
    PRICE_QUERY = "price_query"
    REVIEW_REQUEST = "review_request"
    ACTIVITY_PLANNING = "activity_planning"

@dataclass
class Intent:
    """Individual intent with confidence and parameters"""
    type: IntentType
    confidence: float
    parameters: Dict[str, Any]
    text_span: Tuple[int, int]  # Start and end positions in original text
    priority: int = 1  # 1=high, 2=medium, 3=low
    dependencies: List[str] = None  # Other intent IDs this depends on

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

class MultiIntentQueryHandler:
    """
    Advanced multi-intent query handling system
    Detects, prioritizes, and orchestrates multiple intents in complex queries
    """
    
    def __init__(self):
        # Intent detection patterns with priorities
        self.intent_patterns = {
            IntentType.LOCATION_SEARCH: {
                'patterns': [
                    r'\b(where\s+is|find|locate|location\s+of)\b',
                    r'\b(near|close\s+to|around|vicinity)\b',
                    r'\b(address|directions\s+to)\b'
                ],
                'keywords': ['where', 'find', 'locate', 'near', 'close', 'around', 'address'],
                'priority': 1
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
                    r'\b(seafood|turkish|italian|japanese|mediterranean|ottoman)\b'
                ],
                'keywords': [
                           # English keywords only
                           'recommend', 'suggest', 'best', 'top', 'good', 'popular', 'should', 'worth',
                           'restaurant', 'food', 'dining', 'eat', 'meal', 'lunch', 'dinner', 'breakfast',
                           'cafe', 'bistro', 'eatery', 'places', 'cuisine', 'culinary', 'chef', 'menu',
                           'vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free', 'budget', 'cheap',
                           'expensive', 'affordable', 'seafood', 'turkish', 'italian', 'japanese'
                           ],
                'priority': 1,
                'restaurant_specific': True  # Flag for restaurant-specific patterns
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
                    # Primary comparison patterns
                    r'\b(compare|vs|versus|difference\s+between|which\s+is\s+better)\b',
                    r'\b(better\s+than|worse\s+than|which\s+(one|is)\s+better)\b',
                    r'\b(choose\s+between|decide\s+between|pick\s+between)\b',
                    r'\b(what\'s\s+the\s+difference|how\s+do\s+they\s+compare)\b',
                    r'\b(pros\s+and\s+cons|advantages?\s+(and|vs)\s+disadvantages?)\b',
                    r'\b(which\s+(should|would|do)\s+(i|you|we)\s+(choose|pick|visit))\b',
                    # Multi-restaurant comparison patterns
                    r'\b(.+)\s+(vs|versus|or)\s+(.+)\s+(restaurant|place|spot)\b',
                    r'\b(between\s+.+\s+and\s+.+)\b'
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
                    # Enhanced time patterns with temporal keywords (English only)
                    r'\b(when|what\s+time|hours|opening\s+times?|closing\s+times?)\b',
                    r'\b(schedule|timetable|timing|business\s+hours)\b',
                    r'\b(open|close|closed|available|operating)\b',
                    # Specific temporal references
                    r'\b(today|tomorrow|tonight|morning|afternoon|evening|weekend)\b',
                    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                    r'\b(now|currently|at\s+the\s+moment|right\s+now)\b',
                    r'\b([0-9]{1,2}:\d{2}|[0-9]{1,2}\s*(am|pm|o\'clock))\b'  # Time formats
                ],
                'keywords': ['when', 'time', 'hours', 'open', 'close', 'schedule', 'timing', 'available',
                           'today', 'tomorrow', 'tonight', 'morning', 'afternoon', 'evening', 'weekend',
                           'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                           'now', 'currently', 'am', 'pm', 'oclock'],
                'priority': 2,
                # Time-related correction rules
                'correction_rules': {
                    'temporal_keywords': ['now', 'today', 'tonight', 'tomorrow', 'this', 'next'],
                    'time_formats': [r'\b([0-9]{1,2}:\d{2})\b', r'\b([0-9]{1,2}\s*(am|pm))\b']
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
            IntentType.BOOKING: {
                'patterns': [
                    # English patterns only
                    r'\b(book|reserve|reservation|appointment)\b',
                    r'\b(table\s+for|room\s+for|ticket\s+for)\b',
                    r'\b(availability|available\s+slots?)\b'
                ],
                'keywords': [
                           # English only
                           'book', 'reserve', 'reservation', 'table', 'room', 'ticket', 'availability'
                           ],
                'priority': 1
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
            IntentType.BOOKING: [IntentType.LOCATION_SEARCH, IntentType.TIME_QUERY],
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
        """Analyze a query for multiple intents"""
        
        logging.info(f"🔍 Analyzing multi-intent query: {query}")
        
        # Detect query language
        detected_language = self._detect_language(query)
        
        # Detect all intents
        detected_intents = self._detect_intents(query)
        
        # Validate context alignment
        detected_intents = self._validate_context_alignment(query, detected_intents)
        
        # Extract parameters for each intent
        for intent in detected_intents:
            intent.parameters = self._extract_parameters(query, intent)
        
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
                min_threshold = 0.25 if intent_type in [IntentType.COMPARISON, IntentType.BOOKING, IntentType.TIME_QUERY] else 0.3
                
                if confidence >= min_threshold:
                    text_span = text_spans[0] if text_spans else (0, len(query))
                    
                    intent = Intent(
                        type=intent_type,
                        confidence=confidence,
                        parameters={},
                        text_span=text_span,
                        priority=config['priority']
                    )
                    detected_intents.append(intent)
        
        # Multi-intent detection: If multiple high-confidence intents exist, keep them
        if len(intent_scores) > 1:
            # Sort by confidence
            sorted_scores = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Allow secondary intents if they're within reasonable range of primary
            primary_score = sorted_scores[0][1]
            for intent_type, score in sorted_scores[1:]:
                if score >= 0.4 and score >= primary_score * 0.6:  # Allow strong secondary intents
                    # Check if this intent is already in detected_intents
                    if not any(i.type == intent_type for i in detected_intents):
                        intent = Intent(
                            type=intent_type,
                            confidence=score,
                            parameters={},
                            text_span=(0, len(query)),
                            priority=self.intent_patterns[intent_type]['priority']
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
            
            # Rule 1: Booking intent validation
            if intent.type == IntentType.BOOKING:
                booking_indicators = ['reservation', 'book', 'table']
                if not any(indicator in query_lower for indicator in booking_indicators):
                    should_keep = False
                else:
                    # Strong booking indicators boost confidence
                    time_indicators = ['tonight', 'today', 'tomorrow']
                    if any(indicator in query_lower for indicator in time_indicators):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.2)
            
            # Rule 2: Time query validation
            elif intent.type == IntentType.TIME_QUERY:
                time_indicators = ['time', 'hours', 'open', 'close']
                if not any(indicator in query_lower for indicator in time_indicators):
                    should_keep = False
                else:
                    # Boost confidence if specific time questions
                    if re.search(r'\b(what\s+time)\b', query_lower):
                        adjusted_confidence = min(1.0, adjusted_confidence + 0.3)
            
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
        """Prioritize intents and identify primary vs secondary"""
        
        if not intents:
            # Create a default general intent
            default_intent = Intent(
                type=IntentType.INFORMATION_REQUEST,
                confidence=0.5,
                parameters={},
                text_span=(0, 0),
                priority=2
            )
            return default_intent, []
        
        # Sort by priority (lower number = higher priority) then by confidence
        sorted_intents = sorted(intents, key=lambda x: (x.priority, -x.confidence))
        
        primary_intent = sorted_intents[0]
        secondary_intents = sorted_intents[1:]
        
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
            IntentType.BOOKING: "handle_booking",
            IntentType.REVIEW_REQUEST: "fetch_reviews",
            IntentType.ACTIVITY_PLANNING: "plan_activities"
        }
        
        return actions.get(intent.type, "general_response")
    
    def _calculate_overall_confidence(self, intents: List[Intent]) -> float:
        """Calculate enhanced overall confidence score with boosters"""
        
        if not intents:
            return 0.5
        
        # Base weighted average of intent confidences
        total_weight = sum(1.0 / intent.priority for intent in intents)
        weighted_sum = sum(intent.confidence / intent.priority for intent in intents)
        base_confidence = weighted_sum / total_weight
        
        # Apply confidence boosters for better accuracy
        boosters = 0.0
        
        # Multiple intents detected (shows sophisticated understanding)
        if len(intents) > 1:
            boosters += 0.25
        
        # High-priority intents boost confidence
        if any(intent.priority == 1 for intent in intents):
            boosters += 0.15
        
        # High individual intent confidence
        max_individual_confidence = max(intent.confidence for intent in intents)
        if max_individual_confidence > 0.8:
            boosters += 0.2
        elif max_individual_confidence > 0.6:
            boosters += 0.1
        
        # Istanbul-specific intents get bonus (domain expertise)
        istanbul_intents = ['restaurant_search', 'restaurant_info', 'attraction_search', 'attraction_info']
        if any(intent.type.value in istanbul_intents for intent in intents):
            boosters += 0.15
        
        # Apply boosters with cap at 1.0
        enhanced_confidence = min(1.0, base_confidence + boosters)
        
        return enhanced_confidence
    
    def _determine_processing_strategy(self, complexity: float, num_intents: int) -> str:
        """Determine the best processing strategy"""
        
        if complexity < 0.3 and num_intents == 1:
            return "simple_single_intent"
        elif complexity < 0.5 and num_intents <= 2:
            return "sequential_processing"
        elif complexity < 0.7:
            return "parallel_processing"
        else:
            return "complex_orchestration"
    
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
            'recommendation': "Here are some great restaurant recommendations for you:",
            'booking': "I can help you book a table. Let me find available options:",
            'time_query': "Here are the opening hours and schedule information:", 
            'price_query': "Here's the pricing information you requested:",
            'location_search': "Here's the location information:",
            'comparison': "Here's a comparison of your options:",
            'information_request': "Here's the information you requested:",
            'route_planning': "Here are the directions and route information:",
            'greeting': "Hello! I'm here to help you discover the best of Istanbul."
        }
        
        return templates
    
    def generate_response(self, result: MultiIntentResult, language: str = 'english') -> str:
        """Generate a response in the detected language"""
        
        templates = self._get_response_templates(language)
        primary_intent = result.primary_intent.type.value
        
        # Get the appropriate template
        if primary_intent in templates:
            response = templates[primary_intent]
        else:
            response = templates.get('information_request', templates['greeting'])
        
        # Add confidence indicator for development/testing
        confidence_text = f" (Confidence: {result.confidence_score:.1%})"
        
        return response + confidence_text
    
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

# Example usage and testing
def test_multi_intent_query_handler():
    """Test the multi-intent query handling system"""
    
    print("🎯 Testing Multi-Intent Query Handler...")
    
    handler = MultiIntentQueryHandler()
    
    # Test complex queries
    test_queries = [
        "Where is Hagia Sophia and what are the opening hours?",
        "I want to find good Turkish restaurants near Sultanahmet and also need directions to get there",
        "Compare the Blue Mosque and Hagia Sophia, and tell me which one is better to visit first",
        "What's the best route from Taksim to Galata Tower and how much does it cost?",
        "Recommend some romantic restaurants with Bosphorus view and help me book a table for tonight",
        "Plan a full day itinerary including museums, lunch, and shopping, starting from my hotel"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        result = handler.analyze_query(query)
        
        print(f"   Primary Intent: {result.primary_intent.type.value} (confidence: {result.primary_intent.confidence:.2f})")
        print(f"   Secondary Intents: {[i.type.value for i in result.secondary_intents]}")
        print(f"   Complexity: {result.query_complexity:.2f}")
        print(f"   Strategy: {result.processing_strategy}")
        print(f"   Execution Steps: {len(result.execution_plan)}")
        
        # Show execution plan
        for step in result.execution_plan:
            print(f"     Step {step['step']}: {step['action']} ({step['priority']} priority)")

if __name__ == "__main__":
    test_multi_intent_query_handler()
