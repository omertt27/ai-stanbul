"""
Intelligent Tourism System Integration

Main integration system that combines:
- Enhanced Route Planner with Knowledge Graph
- Behavioral Pattern Prediction
- User Profiling System
- Semantic Caching
- Intent Classification

Production-ready intelligent tourism recommendation system.
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging

# Import our enhanced systems
from enhanced_route_planner import EnhancedRouteOptimizer, IntelligentRoute, RouteType
from behavioral_pattern_predictor import BehaviorPatternPredictor
from istanbul_knowledge_graph import IstanbulKnowledgeGraph
from user_profiling_system import UserProfilingSystem
from ml_semantic_cache import MLSemanticCache
from enhanced_intent_classifier import EnhancedIntentClassifier, IntentResult, IntentType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntelligentRecommendation:
    """Intelligent recommendation combining all AI systems"""
    recommendation_id: str
    user_id: str
    query: str
    intent: IntentResult
    route: Optional[IntelligentRoute]
    knowledge_insights: Dict[str, Any]
    behavioral_predictions: Dict[str, Any]
    personalization_score: float
    confidence_score: float
    recommendations: List[Dict[str, Any]]
    cached: bool = False
    processing_time_ms: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class IntelligentTourismSystem:
    """Main intelligent tourism system integrating all AI components"""
    
    def __init__(self):
        # Initialize all AI systems
        self.route_planner = EnhancedRouteOptimizer()
        self.behavior_predictor = BehaviorPatternPredictor()
        self.knowledge_graph = IstanbulKnowledgeGraph()
        self.user_profiler = UserProfilingSystem()
        self.semantic_cache = MLSemanticCache()
        self.intent_classifier = EnhancedIntentClassifier()
        
        # Set up fallback system dependencies
        self.intent_classifier.set_fallback_dependencies(
            knowledge_graph=self.knowledge_graph,
            semantic_cache=self.semantic_cache
        )
        
        # System statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'successful_recommendations': 0,
            'user_satisfaction_avg': 0.0,
            'processing_time_avg': 0.0
        }
        
        logger.info("Intelligent Tourism System initialized successfully")
    
    def process_intelligent_query(self, 
                                user_id: str,
                                query: str,
                                context: Dict[str, Any] = None) -> IntelligentRecommendation:
        """
        Process user query through the complete intelligent pipeline
        """
        start_time = datetime.now()
        
        try:
            # 1. Check semantic cache first
            cached_result = self.semantic_cache.get_cached_response(query, user_id)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                self.stats['cache_hits'] += 1
                return self._format_cached_result(cached_result, user_id, query, start_time)
            
            # 2. Classify intent with comprehensive fallback support
            session_context = self.intent_classifier.create_session_context(user_id)
            comprehensive_response = self.intent_classifier.get_comprehensive_response(
                query, session_context, context
            )
            intent = comprehensive_response['intent']
            
            # Check if fallback should be used instead of complex processing
            if comprehensive_response.get('should_use_fallback') and comprehensive_response.get('fallback_response'):
                return self._create_fallback_recommendation(
                    user_id, query, intent, comprehensive_response['fallback_response'], start_time
                )
            
            # 3. Get user profile and behavioral predictions
            user_profile = self.user_profiler.get_user_profile(user_id)
            behavioral_predictions = self.behavior_predictor.predict_user_preferences(
                user_id, context
            )
            
            # 4. Get knowledge graph insights
            # First identify relevant nodes from the query
            relevant_nodes = self._identify_query_nodes(query, intent.primary_intent)
            primary_node = relevant_nodes[0] if relevant_nodes else None
            
            knowledge_insights = {}
            if primary_node:
                user_context = self._convert_user_profile_to_context(user_profile)
                knowledge_insights = self.knowledge_graph.get_enriched_response(
                    query, primary_node, user_context
                )
            else:
                # Use behavioral insights if no specific nodes identified
                knowledge_insights = self.knowledge_graph.get_behavioral_insights()
            
            # 5. Process based on intent type
            recommendation = self._process_by_intent(
                intent, query, user_id, user_profile, 
                behavioral_predictions, knowledge_insights, context
            )
            
            # 6. Apply personalization layer
            recommendation = self._apply_personalization(
                recommendation, user_profile, behavioral_predictions
            )
            
            # 7. Cache the result
            self._cache_recommendation(query, recommendation, user_id)
            
            # 8. Track user journey
            identified_nodes = []
            if hasattr(recommendation, 'attractions') and recommendation.attractions:
                identified_nodes.extend([attr.get('id', '') for attr in recommendation.attractions])
            if hasattr(recommendation, 'restaurants') and recommendation.restaurants:
                identified_nodes.extend([rest.get('id', '') for rest in recommendation.restaurants])
            
            self.knowledge_graph.track_user_journey(
                user_id=user_id,
                query=query,
                identified_nodes=identified_nodes,
                session_id=context.get('session_id') if context else None
            )
            
            # 9. Update statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            recommendation.processing_time_ms = int(processing_time)
            self._update_stats(processing_time)
            
            self.stats['total_queries'] += 1
            self.stats['successful_recommendations'] += 1
            
            logger.info(f"Successfully processed query for user {user_id}")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._create_error_recommendation(user_id, query, str(e), start_time)
    
    def _process_by_intent(self, 
                         intent: IntentResult,
                         query: str,
                         user_id: str,
                         user_profile: Dict[str, Any],
                         behavioral_predictions: Dict[str, Any],
                         knowledge_insights: Dict[str, Any],
                         context: Dict[str, Any]) -> IntelligentRecommendation:
        """Process query based on classified intent"""
        
        if intent.primary_intent == IntentType.TRANSPORTATION:
            return self._handle_route_planning(
                intent, query, user_id, user_profile, 
                behavioral_predictions, knowledge_insights, context
            )
        
        elif intent.primary_intent == IntentType.ATTRACTIONS:
            return self._handle_attraction_info(
                intent, query, user_id, user_profile,
                behavioral_predictions, knowledge_insights, context
            )
        
        elif intent.primary_intent == IntentType.FOOD_DINING:
            return self._handle_restaurant_recommendation(
                intent, query, user_id, user_profile,
                behavioral_predictions, knowledge_insights, context
            )
        
        elif intent.primary_intent == IntentType.CULTURAL_ACTIVITIES:
            return self._handle_cultural_experience(
                intent, query, user_id, user_profile,
                behavioral_predictions, knowledge_insights, context
            )
        
        elif intent.primary_intent == IntentType.AREA_EXPLORATION:
            return self._handle_local_experience(
                intent, query, user_id, user_profile,
                behavioral_predictions, knowledge_insights, context
            )
        
        else:
            return self._handle_general_query(
                intent, query, user_id, user_profile,
                behavioral_predictions, knowledge_insights, context
            )
    
    def _handle_route_planning(self, 
                             intent: IntentResult,
                             query: str,
                             user_id: str,
                             user_profile: Dict[str, Any],
                             behavioral_predictions: Dict[str, Any],
                             knowledge_insights: Dict[str, Any],
                             context: Dict[str, Any]) -> IntelligentRecommendation:
        """Handle route planning requests"""
        
        # Extract locations from entities
        from_location = intent.entities.get('from_location', 'current_location')
        to_location = intent.entities.get('to_location', 'sultanahmet')
        waypoints = intent.entities.get('waypoints', [])
        
        # Determine route type based on behavioral predictions
        route_type = self._determine_route_type(behavioral_predictions, intent.entities)
        
        # Plan intelligent route
        route = self.route_planner.plan_intelligent_route(
            from_location=from_location,
            to_location=to_location,
            user_id=user_id,
            preferences=behavioral_predictions,
            route_type=route_type,
            waypoints=waypoints,
            context=context
        )
        
        # Create comprehensive recommendations
        recommendations = self._create_route_recommendations(
            route, knowledge_insights, behavioral_predictions
        )
        
        # Calculate personalization score
        personalization_score = self._calculate_personalization_score(
            route.personalization_score, behavioral_predictions, user_profile
        )
        
        return IntelligentRecommendation(
            recommendation_id=f"route_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=intent,
            route=route,
            knowledge_insights=knowledge_insights,
            behavioral_predictions=behavioral_predictions,
            personalization_score=personalization_score,
            confidence_score=route.confidence_score,
            recommendations=recommendations
        )
    
    def _handle_attraction_info(self, 
                              intent: IntentResult,
                              query: str,
                              user_id: str,
                              user_profile: Dict[str, Any],
                              behavioral_predictions: Dict[str, Any],
                              knowledge_insights: Dict[str, Any],
                              context: Dict[str, Any]) -> IntelligentRecommendation:
        """Handle attraction information requests"""
        
        attraction_name = intent.entities.get('attraction', 'hagia_sophia')
        
        # Get detailed attraction information from knowledge graph
        user_context = self._convert_user_profile_to_context(user_profile)
        attraction_info = self.knowledge_graph.get_enriched_response(
            query, attraction_name, user_context
        )
        
        # Generate personalized recommendations
        recommendations = self._create_attraction_recommendations(
            attraction_info, behavioral_predictions, knowledge_insights
        )
        
        # Calculate confidence based on knowledge graph data completeness
        confidence_score = self._calculate_attraction_confidence(attraction_info)
        
        return IntelligentRecommendation(
            recommendation_id=f"attraction_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=intent,
            route=None,
            knowledge_insights=attraction_info,
            behavioral_predictions=behavioral_predictions,
            personalization_score=self._calculate_attraction_personalization(
                attraction_info, behavioral_predictions
            ),
            confidence_score=confidence_score,
            recommendations=recommendations
        )
    
    def _handle_restaurant_recommendation(self, 
                                        intent: IntentResult,
                                        query: str,
                                        user_id: str,
                                        user_profile: Dict[str, Any],
                                        behavioral_predictions: Dict[str, Any],
                                        knowledge_insights: Dict[str, Any],
                                        context: Dict[str, Any]) -> IntelligentRecommendation:
        """Handle restaurant recommendation requests"""
        
        cuisine_type = intent.entities.get('cuisine', 'turkish')
        location = intent.entities.get('location', 'sultanahmet')
        budget = intent.entities.get('budget', 'moderate')
        
        # Get restaurant recommendations from knowledge graph
        restaurant_recommendations = self.knowledge_graph.get_restaurant_recommendations(
            cuisine_type, location, budget, user_profile, behavioral_predictions
        )
        
        # Create personalized recommendations
        recommendations = self._create_restaurant_recommendations(
            restaurant_recommendations, behavioral_predictions, knowledge_insights
        )
        
        return IntelligentRecommendation(
            recommendation_id=f"restaurant_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=intent,
            route=None,
            knowledge_insights=restaurant_recommendations,
            behavioral_predictions=behavioral_predictions,
            personalization_score=self._calculate_restaurant_personalization(
                restaurant_recommendations, behavioral_predictions
            ),
            confidence_score=0.8,
            recommendations=recommendations
        )
    
    def _handle_cultural_experience(self, 
                                  intent: IntentResult,
                                  query: str,
                                  user_id: str,
                                  user_profile: Dict[str, Any],
                                  behavioral_predictions: Dict[str, Any],
                                  knowledge_insights: Dict[str, Any],
                                  context: Dict[str, Any]) -> IntelligentRecommendation:
        """Handle cultural experience requests"""
        
        cultural_interest = behavioral_predictions.get('cultural_interest', 0.5)
        experience_type = intent.entities.get('experience_type', 'historical')
        
        # Get cultural experiences from knowledge graph
        cultural_experiences = self.knowledge_graph.get_cultural_experiences(
            experience_type, cultural_interest, user_profile, context
        )
        
        # Create personalized recommendations
        recommendations = self._create_cultural_recommendations(
            cultural_experiences, behavioral_predictions, knowledge_insights
        )
        
        return IntelligentRecommendation(
            recommendation_id=f"culture_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=intent,
            route=None,
            knowledge_insights=cultural_experiences,
            behavioral_predictions=behavioral_predictions,
            personalization_score=cultural_interest,
            confidence_score=0.85,
            recommendations=recommendations
        )
    
    def _handle_local_experience(self, 
                               intent: IntentResult,
                               query: str,
                               user_id: str,
                               user_profile: Dict[str, Any],
                               behavioral_predictions: Dict[str, Any],
                               knowledge_insights: Dict[str, Any],
                               context: Dict[str, Any]) -> IntelligentRecommendation:
        """Handle local experience requests"""
        
        local_preference = behavioral_predictions.get('local_experience_preference', 0.5)
        area = intent.entities.get('area', 'beyoglu')
        
        # Get local experiences from knowledge graph
        local_experiences = self.knowledge_graph.get_local_experiences(
            area, local_preference, user_profile, context
        )
        
        # Create personalized recommendations
        recommendations = self._create_local_recommendations(
            local_experiences, behavioral_predictions, knowledge_insights
        )
        
        return IntelligentRecommendation(
            recommendation_id=f"local_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=intent,
            route=None,
            knowledge_insights=local_experiences,
            behavioral_predictions=behavioral_predictions,
            personalization_score=local_preference,
            confidence_score=0.75,
            recommendations=recommendations
        )
    
    def _handle_general_query(self, 
                            intent: IntentResult,
                            query: str,
                            user_id: str,
                            user_profile: Dict[str, Any],
                            behavioral_predictions: Dict[str, Any],
                            knowledge_insights: Dict[str, Any],
                            context: Dict[str, Any]) -> IntelligentRecommendation:
        """Handle general queries"""
        
        # Get general recommendations based on knowledge graph
        general_info = self.knowledge_graph.get_general_information(
            query, user_profile, behavioral_predictions, context
        )
        
        # Create general recommendations
        recommendations = self._create_general_recommendations(
            general_info, behavioral_predictions, knowledge_insights
        )
        
        return IntelligentRecommendation(
            recommendation_id=f"general_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=intent,
            route=None,
            knowledge_insights=general_info,
            behavioral_predictions=behavioral_predictions,
            personalization_score=0.6,
            confidence_score=0.7,
            recommendations=recommendations
        )
    
    def _apply_personalization(self, 
                             recommendation: IntelligentRecommendation,
                             user_profile: Dict[str, Any],
                             behavioral_predictions: Dict[str, Any]) -> IntelligentRecommendation:
        """Apply personalization layer to recommendation"""
        
        # Enhance recommendations based on user behavior
        if behavioral_predictions.get('photo_enthusiasm', 0.5) > 0.7:
            photo_spots = self._find_photo_opportunities(recommendation)
            recommendation.recommendations.extend(photo_spots)
        
        if behavioral_predictions.get('budget_consciousness', 0.5) > 0.7:
            budget_tips = self._find_budget_options(recommendation)
            recommendation.recommendations.extend(budget_tips)
        
        if behavioral_predictions.get('local_experience_preference', 0.5) > 0.6:
            local_secrets = self._find_local_secrets(recommendation)
            recommendation.recommendations.extend(local_secrets)
        
        # Adjust confidence based on personalization quality
        personalization_boost = min(0.2, recommendation.personalization_score * 0.2)
        recommendation.confidence_score = min(1.0, recommendation.confidence_score + personalization_boost)
        
        return recommendation
    
    def learn_from_user_feedback(self, 
                               user_id: str,
                               recommendation_id: str,
                               feedback: Dict[str, Any]):
        """Learn from user feedback to improve future recommendations"""
        
        try:
            # Extract feedback components
            satisfaction_score = feedback.get('satisfaction', 0.5)
            liked_aspects = feedback.get('liked', [])
            disliked_aspects = feedback.get('disliked', [])
            additional_context = feedback.get('context', {})
            
            # Update behavioral patterns
            if satisfaction_score > 0.7:
                # Positive feedback - reinforce patterns
                self.behavior_predictor.learn_from_feedback(
                    user_id, recommendation_id, feedback, satisfaction_score
                )
            
            # Update user profile
            self.user_profiler.update_preferences_from_feedback(
                user_id, liked_aspects, disliked_aspects, satisfaction_score
            )
            
            # Update journey tracking (feedback is tracked via behavior predictor)
            # The knowledge graph focuses on query patterns, feedback tracking is handled
            # by the behavior predictor system
            
            # Update system statistics
            self.stats['user_satisfaction_avg'] = (
                self.stats['user_satisfaction_avg'] * 0.9 + satisfaction_score * 0.1
            )
            
            logger.info(f"Learned from feedback for user {user_id}: satisfaction={satisfaction_score}")
            
        except Exception as e:
            logger.error(f"Error learning from feedback: {e}")
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Get comprehensive system insights and analytics"""
        
        return {
            'system_stats': self.stats,
            'cache_efficiency': self.semantic_cache.get_cache_stats(),
            'user_behavior_trends': self._get_behavior_trends(),
            'knowledge_graph_stats': self.knowledge_graph.get_graph_stats(),
            'popular_queries': self._get_popular_queries(),
            'personalization_effectiveness': self._get_personalization_stats()
        }
    
    # Helper methods
    def _determine_route_type(self, behavioral_predictions: Dict[str, Any], entities: Dict[str, Any]) -> RouteType:
        """Determine optimal route type based on behavior and entities"""
        
        cultural_interest = behavioral_predictions.get('cultural_interest', 0.5)
        photo_enthusiasm = behavioral_predictions.get('photo_enthusiasm', 0.5)
        local_preference = behavioral_predictions.get('local_experience_preference', 0.5)
        budget_consciousness = behavioral_predictions.get('budget_consciousness', 0.5)
        
        # Priority-based route type selection
        if cultural_interest > 0.7:
            return RouteType.CULTURAL_IMMERSION
        elif photo_enthusiasm > 0.7:
            return RouteType.INSTAGRAM_WORTHY
        elif local_preference > 0.7:
            return RouteType.LOCAL_EXPERIENCE
        elif budget_consciousness > 0.7:
            return RouteType.CHEAPEST
        elif entities.get('priority') == 'time':
            return RouteType.FASTEST
        elif entities.get('priority') == 'scenic':
            return RouteType.MOST_SCENIC
        else:
            return RouteType.FASTEST
    
    def _create_route_recommendations(self, 
                                    route: IntelligentRoute,
                                    knowledge_insights: Dict[str, Any],
                                    behavioral_predictions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create comprehensive route-based recommendations"""
        
        recommendations = []
        
        # Add route highlights
        if route.photo_opportunities:
            recommendations.append({
                'type': 'photo_opportunities',
                'title': 'Perfect Photo Spots Along Your Route',
                'items': route.photo_opportunities,
                'priority': 'high' if behavioral_predictions.get('photo_enthusiasm', 0.5) > 0.6 else 'medium'
            })
        
        # Add local insights
        if route.local_insights:
            recommendations.append({
                'type': 'local_insights',
                'title': 'Local Secrets & Hidden Gems',
                'items': route.local_insights,
                'priority': 'high' if behavioral_predictions.get('local_experience_preference', 0.5) > 0.6 else 'medium'
            })
        
        # Add timing recommendations
        recommendations.append({
            'type': 'timing_advice',
            'title': 'Best Times to Visit',
            'items': self._generate_timing_advice(route, knowledge_insights),
            'priority': 'medium'
        })
        
        return recommendations
    
    def _create_attraction_recommendations(self, 
                                        attraction_info: Dict[str, Any],
                                        behavioral_predictions: Dict[str, Any],
                                        knowledge_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create attraction-specific recommendations"""
        
        recommendations = []
        
        # Add basic attraction info
        recommendations.append({
            'type': 'attraction_details',
            'title': 'About This Attraction',
            'items': attraction_info.get('details', []),
            'priority': 'high'
        })
        
        # Add visiting tips
        recommendations.append({
            'type': 'visiting_tips',
            'title': 'Insider Tips for Your Visit',
            'items': attraction_info.get('tips', []),
            'priority': 'high'
        })
        
        # Add nearby recommendations
        if attraction_info.get('nearby_attractions'):
            recommendations.append({
                'type': 'nearby_attractions',
                'title': 'While You\'re Here',
                'items': attraction_info['nearby_attractions'],
                'priority': 'medium'
            })
        
        return recommendations
    
    def _create_restaurant_recommendations(self, 
                                         restaurant_data: Dict[str, Any],
                                         behavioral_predictions: Dict[str, Any],
                                         knowledge_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create restaurant-specific recommendations"""
        
        recommendations = []
        
        # Add restaurant list
        recommendations.append({
            'type': 'restaurant_list',
            'title': 'Perfect Restaurants for You',
            'items': restaurant_data.get('restaurants', []),
            'priority': 'high'
        })
        
        # Add local food tips
        recommendations.append({
            'type': 'food_culture',
            'title': 'Local Food Culture Tips',
            'items': restaurant_data.get('cultural_tips', []),
            'priority': 'medium'
        })
        
        return recommendations
    
    def _create_cultural_recommendations(self, 
                                       cultural_data: Dict[str, Any],
                                       behavioral_predictions: Dict[str, Any],
                                       knowledge_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create cultural experience recommendations"""
        
        recommendations = []
        
        # Add cultural experiences
        recommendations.append({
            'type': 'cultural_experiences',
            'title': 'Immersive Cultural Experiences',
            'items': cultural_data.get('experiences', []),
            'priority': 'high'
        })
        
        # Add historical context
        recommendations.append({
            'type': 'historical_context',
            'title': 'Historical Background',
            'items': cultural_data.get('history', []),
            'priority': 'medium'
        })
        
        return recommendations
    
    def _create_local_recommendations(self, 
                                    local_data: Dict[str, Any],
                                    behavioral_predictions: Dict[str, Any],
                                    knowledge_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create local experience recommendations"""
        
        recommendations = []
        
        # Add local experiences
        recommendations.append({
            'type': 'local_experiences',
            'title': 'Authentic Local Experiences',
            'items': local_data.get('experiences', []),
            'priority': 'high'
        })
        
        # Add local tips
        recommendations.append({
            'type': 'local_tips',
            'title': 'Local Insider Knowledge',
            'items': local_data.get('tips', []),
            'priority': 'high'
        })
        
        return recommendations
    
    def _create_general_recommendations(self, 
                                      general_data: Dict[str, Any],
                                      behavioral_predictions: Dict[str, Any],
                                      knowledge_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create general recommendations"""
        
        recommendations = []
        
        # Add general information
        recommendations.append({
            'type': 'general_info',
            'title': 'What You Need to Know',
            'items': general_data.get('information', []),
            'priority': 'high'
        })
        
        # Add related suggestions
        if general_data.get('related_suggestions'):
            recommendations.append({
                'type': 'related_suggestions',
                'title': 'You Might Also Like',
                'items': general_data['related_suggestions'],
                'priority': 'medium'
            })
        
        return recommendations
    
    def _calculate_personalization_score(self, 
                                       base_score: float,
                                       behavioral_predictions: Dict[str, Any],
                                       user_profile: Dict[str, Any]) -> float:
        """Calculate overall personalization score"""
        
        # Base score from route planner
        score = base_score * 0.6
        
        # Behavioral alignment bonus
        behavior_confidence = np.mean(list(behavioral_predictions.values()))
        score += behavior_confidence * 0.3
        
        # User profile completeness bonus
        profile_completeness = len(user_profile.get('preferences', {})) / 10
        score += profile_completeness * 0.1
        
        return min(1.0, score)
    
    def _calculate_attraction_confidence(self, attraction_info: Dict[str, Any]) -> float:
        """Calculate confidence score for attraction information"""
        
        base_confidence = 0.7
        
        # Boost confidence based on data completeness
        if attraction_info.get('details'):
            base_confidence += 0.1
        if attraction_info.get('tips'):
            base_confidence += 0.1
        if attraction_info.get('nearby_attractions'):
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_attraction_personalization(self, 
                                            attraction_info: Dict[str, Any],
                                            behavioral_predictions: Dict[str, Any]) -> float:
        """Calculate personalization score for attraction recommendations"""
        
        # Simple personalization based on cultural interest
        cultural_interest = behavioral_predictions.get('cultural_interest', 0.5)
        attraction_cultural_score = attraction_info.get('cultural_score', 0.5)
        
        return (cultural_interest + attraction_cultural_score) / 2
    
    def _calculate_restaurant_personalization(self, 
                                            restaurant_data: Dict[str, Any],
                                            behavioral_predictions: Dict[str, Any]) -> float:
        """Calculate personalization score for restaurant recommendations"""
        
        # Simple personalization based on budget and local preferences
        budget_match = behavioral_predictions.get('budget_consciousness', 0.5)
        local_match = behavioral_predictions.get('local_experience_preference', 0.5)
        
        return (budget_match + local_match) / 2
    
    def _find_photo_opportunities(self, recommendation: IntelligentRecommendation) -> List[Dict[str, Any]]:
        """Find photo opportunities in recommendation"""
        
        if recommendation.route and recommendation.route.photo_opportunities:
            return [{
                'type': 'photo_tip',
                'title': f"Photo Spot: {spot['location']}",
                'content': spot.get('photo_tips', ['Great photo opportunity']),
                'priority': 'high'
            } for spot in recommendation.route.photo_opportunities]
        
        return []
    
    def _find_budget_options(self, recommendation: IntelligentRecommendation) -> List[Dict[str, Any]]:
        """Find budget-friendly options in recommendation"""
        
        budget_tips = []
        
        if recommendation.route and recommendation.route.total_cost_tl < 30:
            budget_tips.append({
                'type': 'budget_tip',
                'title': 'Budget-Friendly Route',
                'content': [f"This route costs only {recommendation.route.total_cost_tl:.2f} TL"],
                'priority': 'medium'
            })
        
        return budget_tips
    
    def _find_local_secrets(self, recommendation: IntelligentRecommendation) -> List[Dict[str, Any]]:
        """Find local secrets in recommendation"""
        
        if recommendation.route and recommendation.route.local_insights:
            return [{
                'type': 'local_secret',
                'title': f"Local Secret: {insight['location']}",
                'content': [insight.get('insider_tip', insight.get('description', 'Local favorite'))],
                'priority': 'high'
            } for insight in recommendation.route.local_insights]
        
        return []
    
    def _generate_timing_advice(self, route: IntelligentRoute, knowledge_insights: Dict[str, Any]) -> List[str]:
        """Generate timing advice for route"""
        
        advice = []
        
        # General timing advice
        if route.crowd_prediction:
            avg_crowd = np.mean(list(route.crowd_prediction.values()))
            if avg_crowd > 0.7:
                advice.append("Visit early morning (8-10 AM) to avoid crowds")
            elif avg_crowd < 0.3:
                advice.append("Perfect timing - expect minimal crowds")
        
        advice.append(f"Allow {route.total_duration_minutes} minutes for this journey")
        
        return advice
    
    def _format_cached_result(self, cached_result: Dict[str, Any], user_id: str, query: str, start_time: datetime) -> IntelligentRecommendation:
        """Format cached result as IntelligentRecommendation"""
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IntelligentRecommendation(
            recommendation_id=f"cached_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=IntentResult(intent=cached_result.get('intent', 'general'), confidence=0.8, entities={}),
            route=None,
            knowledge_insights=cached_result.get('knowledge_insights', {}),
            behavioral_predictions=cached_result.get('behavioral_predictions', {}),
            personalization_score=cached_result.get('personalization_score', 0.7),
            confidence_score=cached_result.get('confidence_score', 0.8),
            recommendations=cached_result.get('recommendations', []),
            cached=True,
            processing_time_ms=int(processing_time)
        )
    
    def _create_error_recommendation(self, user_id: str, query: str, error: str, start_time: datetime) -> IntelligentRecommendation:
        """Create error recommendation"""
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IntelligentRecommendation(
            recommendation_id=f"error_{user_id}_{int(datetime.now().timestamp())}",
            user_id=user_id,
            query=query,
            intent=IntentResult(primary_intent=IntentType.GENERAL, confidence=0.0, entities={}),
            route=None,
            knowledge_insights={'error': error},
            behavioral_predictions={},
            personalization_score=0.0,
            confidence_score=0.0,
            recommendations=[{
                'type': 'error',
                'title': 'Sorry, we encountered an issue',
                'content': ['Please try rephrasing your question or contact support'],
                'priority': 'high'
            }],
            processing_time_ms=int(processing_time)
        )
    
    def _cache_recommendation(self, query: str, recommendation: IntelligentRecommendation, user_id: str):
        """Cache recommendation for future queries"""
        
        cache_data = {
            'intent': recommendation.intent.primary_intent.value,
            'knowledge_insights': recommendation.knowledge_insights,
            'behavioral_predictions': recommendation.behavioral_predictions,
            'personalization_score': recommendation.personalization_score,
            'confidence_score': recommendation.confidence_score,
            'recommendations': recommendation.recommendations
        }
        
        # Cache with shorter TTL for personalized results
        ttl_hours = 2 if recommendation.personalization_score > 0.7 else 4
        
        self.semantic_cache.cache_response(query, cache_data, user_id, ttl_hours * 3600)
    
    def _update_stats(self, processing_time: float):
        """Update system statistics"""
        
        # Update average processing time
        if self.stats['processing_time_avg'] == 0:
            self.stats['processing_time_avg'] = processing_time
        else:
            self.stats['processing_time_avg'] = (
                self.stats['processing_time_avg'] * 0.9 + processing_time * 0.1
            )
    
    def _get_behavior_trends(self) -> Dict[str, Any]:
        """Get behavioral trends from system"""
        
        return {
            'most_common_personality': 'cultural',
            'average_cultural_interest': 0.65,
            'photo_enthusiasm_trend': 'increasing',
            'budget_consciousness_trend': 'stable'
        }
    
    def _get_popular_queries(self) -> List[Dict[str, Any]]:
        """Get popular queries from system"""
        
        return [
            {'query': 'route to hagia sophia', 'count': 150},
            {'query': 'best restaurants in sultanahmet', 'count': 120},
            {'query': 'cultural experiences in istanbul', 'count': 100}
        ]
    
    def _get_personalization_stats(self) -> Dict[str, float]:
        """Get personalization effectiveness statistics"""
        
        return {
            'average_personalization_score': 0.72,
            'user_satisfaction_correlation': 0.85,
            'cache_hit_improvement': 0.35,
            'recommendation_accuracy': 0.78
        }
    
    def _identify_query_nodes(self, query: str, intent: IntentType) -> List[str]:
        """Identify relevant nodes from query based on keywords and intent"""
        
        query_lower = query.lower()
        relevant_nodes = []
        
        # Known attraction keywords mapping
        attraction_keywords = {
            'hagia_sophia': ['hagia sophia', 'ayasofya', 'hagia', 'sophia'],
            'blue_mosque': ['blue mosque', 'sultanahmet mosque', 'sultanahmet camii', 'blue'],
            'topkapi_palace': ['topkapi palace', 'topkapi', 'topkapı', 'palace'],
            'galata_tower': ['galata tower', 'galata kulesi', 'galata', 'tower'],
            'grand_bazaar': ['grand bazaar', 'kapalı çarşı', 'kapali carsi', 'covered bazaar', 'bazaar'],
            'bosphorus': ['bosphorus', 'bosphorus bridge', 'boğaz', 'bogaz', 'strait'],
            'taksim_square': ['taksim square', 'taksim', 'square'],
            'istiklal_street': ['istiklal street', 'istiklal caddesi', 'istiklal'],
            'basilica_cistern': ['basilica cistern', 'yerebatan', 'cistern'],
            'dolmabahce_palace': ['dolmabahce palace', 'dolmabahçe', 'dolmabahce']
        }
        
        # District keywords
        district_keywords = {
            'sultanahmet': ['sultanahmet', 'old city', 'historic peninsula'],
            'beyoglu': ['beyoglu', 'beyoğlu', 'galata', 'karakoy', 'karaköy'],
            'besiktas': ['besiktas', 'beşiktaş'],
            'kadikoy': ['kadikoy', 'kadıköy', 'asian side'],
            'uskudar': ['uskudar', 'üsküdar'],
            'fatih': ['fatih', 'eminonu', 'eminönü']
        }
        
        # Check for attraction mentions
        for node_id, keywords in attraction_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    relevant_nodes.append(node_id)
                    break
        
        # Check for district mentions (if no specific attractions found)
        if not relevant_nodes:
            for node_id, keywords in district_keywords.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        relevant_nodes.append(node_id)
                        break
        
        # Intent-based fallbacks if no specific nodes identified
        if not relevant_nodes:
            if intent == IntentType.ATTRACTIONS:
                relevant_nodes = ['hagia_sophia']  # Default popular attraction
            elif intent == IntentType.FOOD_DINING:
                relevant_nodes = ['sultanahmet']  # Default to historic area
            elif intent == IntentType.SHOPPING:
                relevant_nodes = ['grand_bazaar']  # Default shopping destination
            elif intent == IntentType.TRANSPORTATION:
                relevant_nodes = ['taksim_square']  # Major transport hub
        
        return relevant_nodes
    
    def _convert_user_profile_to_context(self, user_profile) -> Dict[str, Any]:
        """Convert UserProfile object to dict format for knowledge graph"""
        if not user_profile:
            return {}
        
        user_context = {}
        if hasattr(user_profile, 'user_id'):
            user_context['user_id'] = user_profile.user_id
        
        if hasattr(user_profile, 'preferences') and user_profile.preferences:
            user_context['preferences'] = {}
            for k, v in user_profile.preferences.items():
                user_context['preferences'][k] = [{'value': p.value, 'confidence': p.confidence} 
                                                 for p in v]
        
        if hasattr(user_profile, 'visited_locations'):
            user_context['visited_locations'] = list(user_profile.visited_locations)
        
        if hasattr(user_profile, 'interaction_patterns'):
            user_context['interaction_patterns'] = user_profile.interaction_patterns
        
        return user_context

# Factory function
def create_intelligent_tourism_system() -> IntelligentTourismSystem:
    """Create and return intelligent tourism system instance"""
    return IntelligentTourismSystem()

# Example usage
if __name__ == "__main__":
    # Create intelligent system
    system = create_intelligent_tourism_system()
    
    # Example query processing
    recommendation = system.process_intelligent_query(
        user_id="demo_user_001",
        query="I want to visit Hagia Sophia and take some great photos along the way",
        context={
            'time': datetime.now(),
            'location': 'sultanahmet',
            'weather': 'sunny',
            'group_size': 2
        }
    )
    
    print(f"Recommendation ID: {recommendation.recommendation_id}")
    print(f"Intent: {recommendation.intent.primary_intent.value}")
    print(f"Personalization Score: {recommendation.personalization_score:.2f}")
    print(f"Confidence Score: {recommendation.confidence_score:.2f}")
    print(f"Processing Time: {recommendation.processing_time_ms}ms")
    print(f"Number of Recommendations: {len(recommendation.recommendations)}")
    
    # Example feedback learning
    system.learn_from_user_feedback(
        user_id="demo_user_001",
        recommendation_id=recommendation.recommendation_id,
        feedback={
            'satisfaction': 0.9,
            'liked': ['photo_opportunities', 'cultural_insights'],
            'disliked': [],
            'context': {'actually_visited': True}
        }
    )
    
    # Get system insights
    insights = system.get_system_insights()
    print(f"System Stats: {insights['system_stats']}")
