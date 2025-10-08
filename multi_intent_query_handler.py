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
                    r'\b(recommend|suggest|best|top|good|popular)\b',
                    r'\b(should\s+(i|we)\s+(visit|go|see|try))\b',
                    r'\b(what.*worth\s+(visiting|seeing|trying))\b'
                ],
                'keywords': ['recommend', 'suggest', 'best', 'top', 'good', 'popular', 'should', 'worth'],
                'priority': 1
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
                    r'\b(how\s+to\s+get|directions?\s+to|route\s+to)\b',
                    r'\b(travel\s+from.*to|go\s+from.*to)\b',
                    r'\b(plan.*trip|itinerary|journey)\b'
                ],
                'keywords': ['directions', 'route', 'travel', 'journey', 'trip', 'itinerary', 'plan'],
                'priority': 1
            },
            IntentType.COMPARISON: {
                'patterns': [
                    r'\b(compare|vs|versus|difference\s+between)\b',
                    r'\b(better\s+than|which\s+(one|is)\s+better)\b',
                    r'\b(choose\s+between|decide\s+between)\b'
                ],
                'keywords': ['compare', 'vs', 'versus', 'difference', 'better', 'which', 'choose'],
                'priority': 2
            },
            IntentType.TIME_QUERY: {
                'patterns': [
                    r'\b(when|what\s+time|hours|opening\s+times?)\b',
                    r'\b(schedule|timetable|timing)\b',
                    r'\b(open|close|closed|available)\b'
                ],
                'keywords': ['when', 'time', 'hours', 'open', 'schedule', 'timing'],
                'priority': 2
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
                    r'\b(book|reserve|reservation|appointment)\b',
                    r'\b(table\s+for|room\s+for|ticket\s+for)\b',
                    r'\b(availability|available\s+slots?)\b'
                ],
                'keywords': ['book', 'reserve', 'reservation', 'table', 'room', 'ticket', 'availability'],
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
        
        # Entity patterns for parameter extraction
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
        
        # Detect all intents
        detected_intents = self._detect_intents(query)
        
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
            processing_strategy=strategy
        )
        
        logging.info(f"✅ Multi-intent analysis complete: {len(detected_intents)} intents detected")
        return result
    
    def _detect_intents(self, query: str) -> List[Intent]:
        """Detect all intents in the query"""
        
        detected_intents = []
        query_lower = query.lower()
        
        for intent_type, config in self.intent_patterns.items():
            confidence = 0.0
            text_spans = []
            
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
            
            # Calculate confidence based on matches
            if pattern_matches > 0 or keyword_matches > 0:
                pattern_confidence = min(1.0, pattern_matches * 0.4)
                keyword_confidence = min(0.8, keyword_matches * 0.2)
                confidence = max(pattern_confidence, keyword_confidence)
                
                # Boost confidence for multiple matches
                if pattern_matches > 1 or keyword_matches > 2:
                    confidence = min(1.0, confidence + 0.2)
                
                if confidence >= 0.3:  # Minimum threshold
                    # Find the best text span
                    text_span = text_spans[0] if text_spans else (0, len(query))
                    
                    intent = Intent(
                        type=intent_type,
                        confidence=confidence,
                        parameters={},
                        text_span=text_span,
                        priority=config['priority']
                    )
                    detected_intents.append(intent)
        
        return detected_intents
    
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
        """Calculate overall confidence score"""
        
        if not intents:
            return 0.5
        
        # Weighted average of intent confidences
        total_weight = sum(1.0 / intent.priority for intent in intents)
        weighted_sum = sum(intent.confidence / intent.priority for intent in intents)
        
        return min(1.0, weighted_sum / total_weight)
    
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
