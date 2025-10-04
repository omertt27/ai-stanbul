"""
Custom AI System Orchestrator for Istanbul Guide

This is the main orchestrator that replaces GPT with deterministic, 
template-based responses using the custom services we've built.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .query_router import QueryRouter, QueryType, QueryClassification
from .template_engine import TemplateEngine
from .info_retrieval_service import InfoRetrievalService
from .transport_service import TransportService
from .recommendation_engine import RecommendationEngine, RecommendationType, UserProfile
from .restaurant_database_service import RestaurantDatabaseService

class CustomAISystemOrchestrator:
    """
    Main orchestrator that coordinates all custom services to provide
    intelligent responses without using GPT. This is the replacement
    for all GPT-based functionality.
    """
    
    def __init__(self):
        # Initialize all services
        self.query_router = QueryRouter()
        self.template_engine = TemplateEngine()
        self.info_service = InfoRetrievalService()
        self.transport_service = TransportService()
        self.recommendation_engine = RecommendationEngine()
        self.restaurant_service = RestaurantDatabaseService()
        
        # Session management
        self.user_sessions = {}
        
        # Performance tracking
        self.response_times = []
        self.query_counts = {"total": 0, "successful": 0, "fallback": 0}
        
    def process_query(self, query: str, user_id: str = "default", 
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main query processing function that replaces all GPT calls
        
        Args:
            query: User's input query
            user_id: User identifier for session management
            context: Additional context information
            
        Returns:
            Response dictionary with answer, confidence, and metadata
        """
        start_time = datetime.now()
        context = context or {}
        
        try:
            # Step 1: Classify the query
            classification = self.query_router.classify_query(query)
            
            # Step 2: Get or create user profile
            user_profile = self._get_user_profile(user_id)
            
            # Step 3: Route to appropriate service and generate response
            response = self._route_and_process(classification, user_profile, context)
            
            # Step 4: Post-process and enhance response
            enhanced_response = self._enhance_response(response, classification, context)
            
            # Step 5: Update user profile based on interaction
            self._update_user_profile(user_id, classification, response)
            
            # Step 6: Track performance
            processing_time = (datetime.now() - start_time).total_seconds()
            self._track_performance(processing_time, True)
            
            return enhanced_response
            
        except Exception as e:
            # Fallback response
            print(f"Error in query processing: {e}")
            self._track_performance(0, False)
            
            return {
                "response": self.template_engine.generate_response(
                    "error", {}, classification.language if 'classification' in locals() else "turkish"
                ),
                "confidence": 0.1,
                "query_type": "error",
                "language": "turkish",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "fallback": True
            }
    
    def _route_and_process(self, classification: QueryClassification, 
                          user_profile: UserProfile, 
                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Route query to appropriate service and generate response"""
        
        query_type = classification.query_type
        entities = classification.extracted_entities
        language = classification.language
        
        # Handle different query types
        if query_type == QueryType.GREETING:
            return self._handle_greeting(language, context)
            
        elif query_type == QueryType.ATTRACTION_INFO:
            return self._handle_attraction_info(entities, language, context)
            
        elif query_type == QueryType.ATTRACTION_SEARCH:
            return self._handle_attraction_search(entities, language, user_profile, context)
            
        elif query_type == QueryType.RESTAURANT_SEARCH:
            return self._handle_restaurant_search(entities, language, user_profile, context)
            
        elif query_type == QueryType.TRANSPORT_ROUTE:
            return self._handle_transport_route(entities, language, context)
            
        elif query_type == QueryType.TRANSPORT_INFO:
            return self._handle_transport_info(entities, language, context)
            
        elif query_type == QueryType.ITINERARY_REQUEST:
            return self._handle_itinerary_request(entities, language, user_profile, context)
            
        elif query_type == QueryType.PRACTICAL_INFO:
            return self._handle_practical_info(entities, language, context)
            
        elif query_type == QueryType.RECOMMENDATION:
            return self._handle_recommendation(entities, language, user_profile, context)
            
        else:
            # Unknown query type
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.2,
                "suggestions": self._generate_suggestions(language)
            }
    
    def _handle_greeting(self, language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle greeting queries"""
        greeting = self.template_engine.generate_greeting(language)
        
        # Add personalized suggestions based on time of day
        current_hour = datetime.now().hour
        suggestions = []
        
        if current_hour < 12:
            suggestions = [
                "Ayasofya'yı ziyaret etmek" if language == "turkish" else "Visit Hagia Sophia",
                "Sultanahmet bölgesini gezme" if language == "turkish" else "Explore Sultanahmet area"
            ]
        elif current_hour < 17:
            suggestions = [
                "Kapalıçarşı'da alışveriş" if language == "turkish" else "Shopping in Grand Bazaar",
                "Boğaz turu" if language == "turkish" else "Bosphorus cruise"
            ]
        else:
            suggestions = [
                "Galata Kulesi'nden gün batımı" if language == "turkish" else "Sunset from Galata Tower",
                "İstiklal Caddesi'nde akşam yürüyüşü" if language == "turkish" else "Evening walk on Istiklal Street"
            ]
        
        return {
            "response": greeting,
            "confidence": 0.9,
            "query_type": "greeting",
            "suggestions": suggestions
        }
    
    def _handle_attraction_info(self, entities: Dict[str, Any], 
                               language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle specific attraction information requests"""
        attractions = entities.get("attractions", [])
        
        if not attractions:
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.3
            }
        
        # Get detailed information for the first attraction
        attraction_name = attractions[0]
        attraction_result = self.info_service.get_attraction_info(attraction_name)
        attraction_info = attraction_result.content if attraction_result else None
        
        if attraction_result and attraction_result.confidence > 0.5:
            return {
                "response": attraction_result.content,
                "confidence": attraction_result.confidence,
                "query_type": "attraction_info",
                "source": attraction_result.source,
                "suggestions": []
            }
        else:
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.3
            }
    
    def _handle_attraction_search(self, entities: Dict[str, Any], 
                                 language: str, user_profile: UserProfile,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle attraction search and recommendations"""
        # Use recommendation engine to find attractions
        context.update(entities)
        recommendations = self.recommendation_engine.generate_recommendations(
            user_profile, RecommendationType.ATTRACTION, context
        )
        
        if not recommendations:
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.3
            }
        
        # Format top recommendations
        top_recs = recommendations[:3]
        attraction_data = [rec.metadata for rec in top_recs]
        
        response = self.template_engine.create_safe_list_response(
            attraction_data, "attractions", language
        )
        
        # Add reasons for recommendations
        reasons = []
        for rec in top_recs:
            if rec.reasons:
                reasons.extend(rec.reasons[:2])  # Top 2 reasons per recommendation
        
        return {
            "response": response,
            "confidence": 0.8,
            "query_type": "attraction_search",
            "recommendations": [rec.__dict__ for rec in top_recs],
            "reasons": reasons[:5]  # Top 5 overall reasons
        }
    
    def _handle_restaurant_search(self, entities: Dict[str, Any], 
                                 language: str, user_profile: UserProfile,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle restaurant search using the restaurant database service"""
        try:
            # Extract query parameters from entities
            original_query = context.get("original_query", "")
            
            # Use our restaurant database service for structured search
            response = self.restaurant_service.search_restaurants(original_query)
            
            return {
                "response": response,
                "confidence": 0.9,  # High confidence since we have real data
                "query_type": "restaurant_search",
                "source": "restaurant_database",
                "data_freshness": "google_places_api"
            }
            
        except Exception as e:
            print(f"Error in restaurant search: {e}")
            # Fallback to recommendation engine
            context.update(entities)
            recommendations = self.recommendation_engine.generate_recommendations(
                user_profile, RecommendationType.RESTAURANT, context
            )
            
            if not recommendations:
                return {
                    "response": self.template_engine.generate_no_results(language),
                    "confidence": 0.3
                }
            
            top_recs = recommendations[:3]
            restaurant_data = [rec.metadata for rec in top_recs]
            
            response = self.template_engine.create_safe_list_response(
                restaurant_data, "restaurants", language
            )
            
            return {
                "response": response,
                "confidence": 0.6,  # Lower confidence for fallback
                "query_type": "restaurant_search",
                "source": "fallback_recommendation_engine"
            }
    
    def _handle_transport_route(self, entities: Dict[str, Any], 
                               language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transportation routing queries"""
        from_location = entities.get("from_location")
        to_location = entities.get("to_location")
        
        # If no specific locations, use search_transport for general queries
        if not from_location or not to_location:
            original_query = context.get("original_query", "")
            response = self.transport_service.search_transport(original_query)
            
            return {
                "response": response,
                "confidence": 0.8,
                "query_type": "transport_route",
                "source": "transport_service"
            }
        
        # Get route information
        routes = self.transport_service.get_route_info(from_location, to_location)
        
        if routes:
            response = self.transport_service.format_route_response(routes)
            
            return {
                "response": response,
                "confidence": 0.8,
                "query_type": "transport_route",
                "route_data": [route.__dict__ for route in routes],
                "source": "transport_service"
            }
        else:
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.3
            }
    
    def _handle_transport_info(self, entities: Dict[str, Any], 
                              language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general transportation information"""
        original_query = context.get("original_query", "")
        
        # Use search_transport for general transport info
        response = self.transport_service.search_transport(original_query)
        
        return {
            "response": response,
            "confidence": 0.8,
            "query_type": "transport_info",
            "source": "transport_service"
        }
    
    def _handle_itinerary_request(self, entities: Dict[str, Any], 
                                 language: str, user_profile: UserProfile,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle itinerary planning requests"""
        duration = entities.get("duration", "1 day")
        interests = entities.get("interests", [])
        
        # Update context with duration and interests
        context.update({"duration": duration, "interests": interests})
        
        # Get itinerary recommendations
        recommendations = self.recommendation_engine.generate_recommendations(
            user_profile, RecommendationType.ITINERARY, context
        )
        
        if recommendations:
            itinerary = recommendations[0]  # Best recommendation
            response = self.template_engine.generate_itinerary_response(
                itinerary.metadata, language
            )
            
            return {
                "response": response,
                "confidence": 0.8,
                "query_type": "itinerary_request",
                "itinerary_data": itinerary.metadata
            }
        else:
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.3
            }
    
    def _handle_practical_info(self, entities: Dict[str, Any], 
                              language: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle practical information requests"""
        info_type = entities.get("info_type", "general")
        
        practical_info = self.info_service.get_practical_info(
            info_type, language, context
        )
        
        if practical_info:
            return {
                "response": practical_info["response"],
                "confidence": 0.8,
                "query_type": "practical_info",
                "info_data": practical_info
            }
        else:
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.3
            }
    
    def _handle_recommendation(self, entities: Dict[str, Any], 
                              language: str, user_profile: UserProfile,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general recommendation requests"""
        interests = entities.get("interests", [])
        
        # Get both attraction and restaurant recommendations
        context.update({"interests": interests})
        
        attraction_recs = self.recommendation_engine.generate_recommendations(
            user_profile, RecommendationType.ATTRACTION, context
        )[:2]
        
        restaurant_recs = self.recommendation_engine.generate_recommendations(
            user_profile, RecommendationType.RESTAURANT, context
        )[:2]
        
        # Combine recommendations
        all_recs = attraction_recs + restaurant_recs
        
        if all_recs:
            # Create combined response
            response_parts = []
            
            if attraction_recs:
                attractions_text = "🏛️ **Gezilecek Yerler:**\n" if language == "turkish" else "🏛️ **Places to Visit:**\n"
                for rec in attraction_recs:
                    attractions_text += f"• {rec.name}: {rec.reasons[0] if rec.reasons else ''}\n"
                response_parts.append(attractions_text)
            
            if restaurant_recs:
                restaurants_text = "🍽️ **Restoran Önerileri:**\n" if language == "turkish" else "🍽️ **Restaurant Recommendations:**\n"
                for rec in restaurant_recs:
                    restaurants_text += f"• {rec.name}: {rec.reasons[0] if rec.reasons else ''}\n"
                response_parts.append(restaurants_text)
            
            response = "\n\n".join(response_parts)
            
            return {
                "response": response,
                "confidence": 0.8,
                "query_type": "recommendation",
                "recommendations": [rec.__dict__ for rec in all_recs]
            }
        else:
            return {
                "response": self.template_engine.generate_no_results(language),
                "confidence": 0.3
            }
    
    def _enhance_response(self, response: Dict[str, Any], 
                         classification: QueryClassification,
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and enhance the response"""
        enhanced = response.copy()
        
        # Add metadata
        enhanced.update({
            "language": classification.language,
            "query_type": classification.query_type.value,
            "classification_confidence": classification.confidence,
            "timestamp": datetime.now().isoformat(),
            "system_version": "custom_v1.0"
        })
        
        # Add usage stats if needed
        if "suggestions" not in enhanced and enhanced.get("confidence", 0) < 0.5:
            enhanced["suggestions"] = self._generate_suggestions(classification.language)
        
        return enhanced
    
    def _generate_suggestions(self, language: str) -> List[str]:
        """Generate helpful suggestions for low-confidence responses"""
        if language == "turkish":
            return [
                "Belirli bir yeri sormayı deneyin (örn: Ayasofya)",
                "Hangi bölgeyi ziyaret etmek istediğinizi belirtin",
                "Yemek önerisi için mutfak türünü belirtin",
                "Ulaşım için nereden nereye gitmek istediğinizi söyleyin"
            ]
        else:
            return [
                "Try asking about a specific place (e.g., Hagia Sophia)",
                "Specify which district you'd like to visit",
                "For food recommendations, mention cuisine type",
                "For transport, specify your origin and destination"
            ]
    
    def _get_user_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_sessions:
            # Create default profile
            self.user_sessions[user_id] = self.recommendation_engine.create_user_profile({
                "interests": ["culture", "history"],
                "visited_places": [],
                "duration": "1 day",
                "budget": "moderate",
                "districts": [],
                "travel_style": "cultural"
            })
        
        return self.user_sessions[user_id]
    
    def _update_user_profile(self, user_id: str, classification: QueryClassification,
                           response: Dict[str, Any]) -> None:
        """Update user profile based on interaction"""
        if user_id not in self.user_sessions:
            return
        
        profile = self.user_sessions[user_id]
        
        # Update interests based on query
        if classification.extracted_entities.get("interests"):
            new_interests = classification.extracted_entities["interests"]
            profile.interests = list(set(profile.interests + new_interests))
        
        # Update visited places if attraction info was requested
        if (classification.query_type == QueryType.ATTRACTION_INFO and 
            "attraction_data" in response):
            attraction_id = response["attraction_data"].get("id")
            if attraction_id and attraction_id not in profile.visited_places:
                profile.visited_places.append(attraction_id)
        
        # Update preferred districts
        if classification.extracted_entities.get("districts"):
            new_districts = classification.extracted_entities["districts"]
            profile.preferred_districts = list(set(profile.preferred_districts + new_districts))
    
    def _track_performance(self, processing_time: float, success: bool) -> None:
        """Track system performance"""
        self.response_times.append(processing_time)
        self.query_counts["total"] += 1
        
        if success:
            self.query_counts["successful"] += 1
        else:
            self.query_counts["fallback"] += 1
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        if not self.response_times:
            return {"status": "no_data"}
        
        avg_response_time = sum(self.response_times) / len(self.response_times)
        success_rate = (self.query_counts["successful"] / 
                       max(1, self.query_counts["total"]))
        
        return {
            "total_queries": self.query_counts["total"],
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "active_users": len(self.user_sessions),
            "system_status": "operational"
        }
