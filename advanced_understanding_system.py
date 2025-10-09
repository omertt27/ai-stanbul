#!/usr/bin/env python3
"""
Advanced Understanding System Integration for AI Istanbul
Combines Semantic Similarity Engine, Enhanced Context Memory, and Multi-Intent Query Handling
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

# Import our three advanced components
from semantic_similarity_engine import SemanticSimilarityEngine, QueryContext
from enhanced_context_memory import EnhancedContextMemory, ContextType
from multi_intent_query_handler import MultiIntentQueryHandler, MultiIntentResult

@dataclass
class AdvancedUnderstandingResult:
    """Comprehensive result from advanced understanding system"""
    
    # Input information
    original_query: str
    user_id: Optional[str]
    session_id: str
    
    # Multi-intent analysis
    multi_intent_result: MultiIntentResult
    
    # Semantic analysis
    semantic_matches: List[Dict[str, Any]]
    contextual_suggestions: List[Dict[str, Any]]
    
    # Context information
    relevant_contexts: List[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    session_summary: Dict[str, Any]
    
    # Understanding metrics
    understanding_confidence: float
    query_complexity: float
    processing_strategy: str
    
    # Recommendations for response generation
    response_strategy: str
    key_information_needed: List[str]
    suggested_actions: List[Dict[str, Any]]
    
    # Metadata
    processing_time_ms: float
    timestamp: str

class AdvancedUnderstandingSystem:
    """
    Integrated advanced understanding system that combines:
    1. Semantic Similarity Engine - for deep semantic understanding
    2. Enhanced Context Memory - for conversation and user context
    3. Multi-Intent Query Handler - for complex query decomposition
    """
    
    def __init__(self, redis_client=None):
        # Initialize all three components
        self.semantic_engine = SemanticSimilarityEngine()
        self.context_memory = EnhancedContextMemory(redis_client)
        self.multi_intent_handler = MultiIntentQueryHandler()
        
        # System configuration
        self.config = {
            'min_understanding_threshold': 0.4,
            'max_processing_time_ms': 5000,
            'context_relevance_threshold': 0.3,
            'semantic_match_threshold': 0.4
        }
        
        # Response strategy mapping
        self.response_strategies = {
            'simple_single_intent': 'direct_answer',
            'sequential_processing': 'structured_response',
            'parallel_processing': 'multi_part_response',
            'complex_orchestration': 'comprehensive_analysis'
        }
        
        logging.info("✅ Advanced Understanding System initialized")
    
    def understand_query(self, query: str, user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        location_context: Optional[Dict[str, Any]] = None) -> AdvancedUnderstandingResult:
        """
        Comprehensive understanding of a user query using all three systems
        """
        
        start_time = datetime.now()
        logging.info(f"🧠 Advanced understanding analysis for: {query}")
        
        # Store query for confidence calculation
        self._current_query = query
        
        # Ensure we have a session
        if not session_id:
            session_id = self.context_memory.start_new_session(user_id)
        
        # Step 1: Multi-Intent Analysis
        logging.debug("🎯 Analyzing multiple intents...")
        multi_intent_result = self.multi_intent_handler.analyze_query(query)
        
        # Step 2: Get Relevant Context
        logging.debug("🧠 Retrieving relevant context...")
        relevant_contexts = self.context_memory.get_relevant_context(
            query, 
            intent=multi_intent_result.primary_intent.type.value,
            max_items=10
        )
        
        conversation_history = self.context_memory.get_conversation_history(max_turns=5)
        session_summary = self.context_memory.get_session_summary(session_id)
        
        # Step 3: Semantic Analysis with Context
        logging.debug("🔍 Performing semantic analysis...")
        query_context = QueryContext(
            user_query=query,
            location_context=location_context,
            user_preferences=session_summary.get('user_preferences', {}),
            conversation_history=[turn.user_query for turn in conversation_history]
        )
        
        semantic_analysis = self.semantic_engine.analyze_query_semantics(query_context)
        
        # Step 4: Cross-System Integration and Enhancement
        logging.debug("🔄 Integrating cross-system insights...")
        
        # Enhance multi-intent confidence with semantic analysis
        enhanced_confidence = self._enhance_intent_confidence(
            multi_intent_result, semantic_analysis, relevant_contexts
        )
        
        # Calculate overall understanding confidence
        understanding_confidence = self._calculate_understanding_confidence(
            multi_intent_result, semantic_analysis, relevant_contexts, query
        )
        
        # Determine response strategy
        response_strategy = self._determine_response_strategy(
            multi_intent_result, understanding_confidence, len(relevant_contexts)
        )
        
        # Generate key information requirements
        key_info_needed = self._identify_key_information_needs(
            multi_intent_result, semantic_analysis, relevant_contexts
        )
        
        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(
            multi_intent_result, semantic_analysis, relevant_contexts
        )
        
        # Step 5: Update Context Memory
        logging.debug("💾 Updating context memory...")
        self._update_context_memory(
            query, multi_intent_result, semantic_analysis, session_id
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Create comprehensive result
        result = AdvancedUnderstandingResult(
            original_query=query,
            user_id=user_id,
            session_id=session_id,
            multi_intent_result=multi_intent_result,
            semantic_matches=semantic_analysis.get('semantic_matches', []),
            contextual_suggestions=semantic_analysis.get('contextual_suggestions', []),
            relevant_contexts=[{
                'type': ctx.type.value,
                'content': ctx.content,
                'confidence': ctx.confidence,
                'relevance_score': ctx.relevance_score
            } for ctx in relevant_contexts],
            conversation_history=[{
                'query': turn.user_query,
                'intent': turn.intent,
                'timestamp': turn.timestamp.isoformat()
            } for turn in conversation_history],
            session_summary=session_summary,
            understanding_confidence=understanding_confidence,
            query_complexity=multi_intent_result.query_complexity,
            processing_strategy=multi_intent_result.processing_strategy,
            response_strategy=response_strategy,
            key_information_needed=key_info_needed,
            suggested_actions=suggested_actions,
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logging.info(f"✅ Advanced understanding complete: {understanding_confidence:.2f} confidence, {processing_time:.1f}ms")
        return result
    
    def _enhance_intent_confidence(self, multi_intent_result: MultiIntentResult,
                                 semantic_analysis: Dict[str, Any],
                                 relevant_contexts: List) -> float:
        """Enhance intent confidence using semantic and context information"""
        
        base_confidence = multi_intent_result.confidence_score
        
        # Boost confidence if semantic analysis agrees
        semantic_confidence = semantic_analysis.get('overall_confidence', 0.5)
        if semantic_confidence > 0.7:
            base_confidence += 0.1
        
        # Boost confidence if we have relevant context
        if relevant_contexts and len(relevant_contexts) > 2:
            context_boost = min(0.15, len(relevant_contexts) * 0.03)
            base_confidence += context_boost
        
        # Boost confidence if primary intent matches semantic primary intent
        primary_intent = multi_intent_result.primary_intent.type.value
        semantic_intent = semantic_analysis.get('primary_intent', '')
        
        if primary_intent in semantic_intent or semantic_intent in primary_intent:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_understanding_confidence(self, multi_intent_result: MultiIntentResult,
                                          semantic_analysis: Dict[str, Any],
                                          relevant_contexts: List, 
                                          original_query: str = "") -> float:
        """Calculate overall understanding confidence with restaurant query enhancements"""
        
        # Base confidence sources
        intent_confidence = multi_intent_result.confidence_score * 0.4
        semantic_confidence = semantic_analysis.get('overall_confidence', 0.5) * 0.3
        
        # Context confidence based on relevance and quantity
        context_confidence = 0.0
        if relevant_contexts:
            avg_relevance = sum(ctx.relevance_score for ctx in relevant_contexts) / len(relevant_contexts)
            context_confidence = min(0.3, avg_relevance * 0.3)
        
        # ENHANCEMENT: Restaurant query confidence boost
        restaurant_boost = 0.0
        primary_intent = multi_intent_result.primary_intent.type.value
        
        if primary_intent == "recommendation":
            # Check if this is a restaurant-related query using the original query
            query_lower = original_query.lower() if original_query else ""
            restaurant_keywords = ['restaurant', 'food', 'dining', 'eat', 'meal', 'cuisine',
                                 'vegetarian', 'vegan', 'halal', 'seafood', 'turkish', 'lunch', 'dinner']
            
            restaurant_matches = sum(1 for keyword in restaurant_keywords if keyword in query_lower)
            
            if restaurant_matches > 0:
                # Boost confidence for restaurant queries
                restaurant_boost = min(0.25, restaurant_matches * 0.08)
                
                # Additional boost for specific dietary/cuisine terms
                specific_terms = ['vegetarian', 'vegan', 'halal', 'kosher', 'gluten-free',
                                'seafood', 'turkish', 'italian', 'japanese', 'budget', 'luxury']
                specific_matches = sum(1 for term in specific_terms if term in query_lower)
                
                if specific_matches > 0:
                    restaurant_boost += min(0.15, specific_matches * 0.05)
        
        # Reduced complexity penalty for restaurant queries (they're naturally complex)
        complexity_penalty = multi_intent_result.query_complexity * 0.1
        if restaurant_boost > 0:
            complexity_penalty *= 0.5  # Reduce penalty for restaurant queries
        
        total_confidence = intent_confidence + semantic_confidence + context_confidence + restaurant_boost - complexity_penalty
        return max(0.1, min(1.0, total_confidence))
    
    def _determine_response_strategy(self, multi_intent_result: MultiIntentResult,
                                   understanding_confidence: float,
                                   context_count: int) -> str:
        """Determine the best response strategy"""
        
        base_strategy = self.response_strategies.get(
            multi_intent_result.processing_strategy, 'direct_answer'
        )
        
        # Modify strategy based on understanding confidence
        if understanding_confidence < 0.4:
            return 'clarification_request'
        elif understanding_confidence > 0.8 and context_count > 5:
            return 'personalized_comprehensive'
        
        return base_strategy
    
    def _identify_key_information_needs(self, multi_intent_result: MultiIntentResult,
                                      semantic_analysis: Dict[str, Any],
                                      relevant_contexts: List) -> List[str]:
        """Identify what key information is needed for a complete response"""
        
        key_info = []
        
        # Based on primary intent
        primary_intent = multi_intent_result.primary_intent.type.value
        
        if 'location' in primary_intent:
            key_info.extend(['current_location', 'target_locations', 'distance_info'])
        
        if 'recommendation' in primary_intent:
            key_info.extend(['user_preferences', 'rating_data', 'availability_info'])
        
        if 'route' in primary_intent or 'planning' in primary_intent:
            key_info.extend(['transportation_options', 'schedule_data', 'estimated_time'])
        
        if 'price' in primary_intent or 'booking' in primary_intent:
            key_info.extend(['pricing_info', 'availability', 'booking_requirements'])
        
        # Based on missing context
        if not any(ctx.type.value == 'location' for ctx in relevant_contexts):
            key_info.append('location_context')
        
        if not any(ctx.type.value == 'preference' for ctx in relevant_contexts):
            key_info.append('user_preferences')
        
        # Based on semantic entities
        entities = semantic_analysis.get('extracted_entities', {})
        if not entities.get('landmarks'):
            key_info.append('landmark_identification')
        
        return list(set(key_info))  # Remove duplicates
    
    def _generate_suggested_actions(self, multi_intent_result: MultiIntentResult,
                                  semantic_analysis: Dict[str, Any],
                                  relevant_contexts: List) -> List[Dict[str, Any]]:
        """Generate suggested actions based on understanding"""
        
        actions = []
        
        # Actions from execution plan
        for step in multi_intent_result.execution_plan:
            action = {
                'type': 'intent_action',
                'action': step['action'],
                'intent': step['intent_type'],
                'priority': step['priority'],
                'parameters': step['parameters']
            }
            actions.append(action)
        
        # Actions from contextual suggestions
        for suggestion in semantic_analysis.get('contextual_suggestions', []):
            action = {
                'type': 'contextual_suggestion',
                'action': suggestion.get('suggested_action', ''),
                'category': suggestion.get('category', ''),
                'confidence': suggestion.get('confidence', 0.5)
            }
            actions.append(action)
        
        # Context-based actions
        if relevant_contexts:
            for context in relevant_contexts[:3]:  # Top 3 most relevant
                if context.type == ContextType.LOCATION:
                    actions.append({
                        'type': 'location_action',
                        'action': 'use_location_context',
                        'location': context.content,
                        'relevance': context.relevance_score
                    })
        
        return actions
    
    def _update_context_memory(self, query: str, multi_intent_result: MultiIntentResult,
                             semantic_analysis: Dict[str, Any], session_id: str):
        """Update context memory with new information from this interaction"""
        
        # Add conversation turn (placeholder response for now)
        self.context_memory.add_conversation_turn(
            user_query=query,
            ai_response="[Response will be generated]",
            extracted_entities=semantic_analysis.get('extracted_entities', {}),
            intent=multi_intent_result.primary_intent.type.value,
            confidence=multi_intent_result.confidence_score
        )
        
        # Add intent as context
        self.context_memory.add_context_item(
            ContextType.INTENT,
            {
                'primary_intent': multi_intent_result.primary_intent.type.value,
                'secondary_intents': [i.type.value for i in multi_intent_result.secondary_intents],
                'complexity': multi_intent_result.query_complexity
            },
            confidence=multi_intent_result.confidence_score,
            source="multi_intent_analysis"
        )
        
        # Add semantic entities as context
        entities = semantic_analysis.get('extracted_entities', {})
        for entity_type, entity_list in entities.items():
            if entity_list:
                self.context_memory.add_context_item(
                    ContextType.ENTITY,
                    {'type': entity_type, 'entities': entity_list},
                    confidence=0.8,
                    source="semantic_analysis"
                )
    
    def get_understanding_insights(self, result: AdvancedUnderstandingResult) -> Dict[str, Any]:
        """Get insights about the understanding process for debugging/improvement"""
        
        return {
            'query_analysis': {
                'complexity': result.query_complexity,
                'understanding_confidence': result.understanding_confidence,
                'processing_time_ms': result.processing_time_ms,
                'processing_strategy': result.processing_strategy
            },
            'intent_analysis': {
                'primary_intent': result.multi_intent_result.primary_intent.type.value,
                'secondary_intents': [i.type.value for i in result.multi_intent_result.secondary_intents],
                'total_intents': 1 + len(result.multi_intent_result.secondary_intents),
                'intent_confidence': result.multi_intent_result.confidence_score
            },
            'context_analysis': {
                'relevant_contexts_found': len(result.relevant_contexts),
                'conversation_history_length': len(result.conversation_history),
                'session_duration_minutes': result.session_summary.get('duration_minutes', 0),
                'has_location_context': any(ctx['type'] == 'location' for ctx in result.relevant_contexts)
            },
            'semantic_analysis': {
                'semantic_matches_found': len(result.semantic_matches),
                'contextual_suggestions': len(result.contextual_suggestions),
                'entities_extracted': sum(len(entities) for entities in 
                                        result.multi_intent_result.primary_intent.parameters.values() 
                                        if isinstance(entities, list))
            },
            'response_guidance': {
                'strategy': result.response_strategy,
                'key_info_needed': result.key_information_needed,
                'suggested_actions': len(result.suggested_actions)
            }
        }

# Example usage and comprehensive testing
def test_advanced_understanding_system():
    """Test the complete advanced understanding system"""
    
    print("🚀 Testing Advanced Understanding System Integration...")
    
    system = AdvancedUnderstandingSystem()
    
    # Test complex queries
    test_scenarios = [
        {
            'query': "I'm staying near Sultanahmet and want to visit Hagia Sophia tomorrow morning, then find a good Turkish restaurant for lunch, and finally get directions to Galata Tower",
            'location': {'name': 'Sultanahmet', 'lat': 41.0082, 'lng': 28.9784},
            'description': "Complex multi-intent with location context"
        },
        {
            'query': "What's the difference between Blue Mosque and Hagia Sophia, and which one should I visit first?",
            'location': None,
            'description': "Comparison intent with decision support"
        },
        {
            'query': "I need budget-friendly restaurants with vegetarian options near my location",
            'location': {'name': 'Taksim', 'lat': 41.0369, 'lng': 28.9850},
            'description': "Recommendation with constraints"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*60}")
        print(f"🔍 Test Scenario {i}: {scenario['description']}")
        print(f"Query: {scenario['query']}")
        print(f"{'='*60}")
        
        # Analyze query
        result = system.understand_query(
            query=scenario['query'],
            user_id=f"test_user_{i}",
            location_context=scenario['location']
        )
        
        # Print results
        print(f"\n📊 Understanding Results:")
        print(f"   Understanding Confidence: {result.understanding_confidence:.2f}")
        print(f"   Query Complexity: {result.query_complexity:.2f}")
        print(f"   Processing Time: {result.processing_time_ms:.1f}ms")
        print(f"   Response Strategy: {result.response_strategy}")
        
        print(f"\n🎯 Intent Analysis:")
        print(f"   Primary Intent: {result.multi_intent_result.primary_intent.type.value}")
        print(f"   Secondary Intents: {[i.type.value for i in result.multi_intent_result.secondary_intents]}")
        
        print(f"\n🧠 Context & Semantic Info:")
        print(f"   Relevant Contexts: {len(result.relevant_contexts)}")
        print(f"   Semantic Matches: {len(result.semantic_matches)}")
        print(f"   Contextual Suggestions: {len(result.contextual_suggestions)}")
        
        print(f"\n📋 Response Guidance:")
        print(f"   Key Info Needed: {', '.join(result.key_information_needed)}")
        print(f"   Suggested Actions: {len(result.suggested_actions)}")
        
        # Get insights
        insights = system.get_understanding_insights(result)
        print(f"\n🔍 System Insights:")
        print(f"   Total Processing Elements: {insights['intent_analysis']['total_intents']} intents, "
              f"{insights['context_analysis']['relevant_contexts_found']} contexts, "
              f"{insights['semantic_analysis']['semantic_matches_found']} semantic matches")

if __name__ == "__main__":
    test_advanced_understanding_system()
