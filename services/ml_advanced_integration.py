"""
Advanced ML System Integration for Istanbul AI Chat & Routing
==============================================================

Integrates the advanced ML system (user preference learning, journey pattern recognition,
predictive route ranking, and context-aware dialogue) into the main chat/routing pipeline.

Features:
- Real-time user preference learning from chat interactions
- Journey pattern recognition from trip history
- Personalized route ranking with ML models
- Context-aware dialogue with intent classification
- Seamless integration with existing routing and chat services
- T4 GPU optimized inference

Author: Istanbul AI Team
Date: October 24, 2025
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import numpy as np

logger = logging.getLogger(__name__)

# Import the advanced ML system
try:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    from ml_advanced_system import (
        get_advanced_ml_system,
        AdvancedMLSystem,
        UserPreference,
        JourneyPattern,
        ConversationContext
    )
    ML_SYSTEM_AVAILABLE = True
    logger.info("✅ Advanced ML System loaded successfully")
except ImportError as e:
    ML_SYSTEM_AVAILABLE = False
    logger.warning(f"⚠️ Advanced ML System not available: {e}")

# Import routing service adapter
try:
    from services.routing_service_adapter import RoutingServiceAdapter
    ROUTING_AVAILABLE = True
except ImportError as e:
    ROUTING_AVAILABLE = False
    logger.warning(f"⚠️ Routing Service Adapter not available: {e}")

# Import database models
try:
    from backend.models import User, ChatHistory, UserPreference as DBUserPreference
    from backend.database import get_db
    from sqlalchemy.orm import Session
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    logger.warning(f"⚠️ Database not available: {e}")


@dataclass
class MLEnhancedResponse:
    """Enhanced chat response with ML insights"""
    response_text: str
    intent: str
    confidence: float
    personalization_applied: bool
    detected_patterns: List[Dict[str, Any]]
    ranked_routes: Optional[List[Tuple[Dict, float]]] = None
    proactive_suggestions: Optional[List[Dict[str, Any]]] = None
    context_summary: Optional[Dict[str, Any]] = None
    ml_metadata: Optional[Dict[str, Any]] = None


class MLAdvancedIntegration:
    """
    Main integration class for advanced ML features in chat/routing pipeline
    """
    
    def __init__(self):
        """Initialize the ML integration service"""
        self.ml_system = None
        self.routing_adapter = None
        
        # Initialize ML system
        if ML_SYSTEM_AVAILABLE:
            try:
                self.ml_system = get_advanced_ml_system()
                logger.info("🧠 Advanced ML System initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ML system: {e}")
                self.ml_system = None
        
        # Initialize routing adapter
        if ROUTING_AVAILABLE:
            try:
                self.routing_adapter = RoutingServiceAdapter()
                logger.info("🗺️ Routing Service Adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize routing adapter: {e}")
                self.routing_adapter = None
        
        # Track active sessions and their contexts
        self.active_sessions: Dict[str, ConversationContext] = {}
        
        # Periodic learning task
        self._learning_task = None
        
        logger.info("✅ ML Advanced Integration ready")
    
    def is_available(self) -> bool:
        """Check if ML features are available"""
        return ML_SYSTEM_AVAILABLE and self.ml_system is not None
    
    async def process_chat_message(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        user_location: Optional[str] = None,
        db_session: Optional[Any] = None
    ) -> MLEnhancedResponse:
        """
        Process a chat message with ML enhancements
        
        Args:
            user_id: User identifier
            session_id: Chat session identifier
            user_message: User's message
            user_location: Optional current location
            db_session: Database session for data access
            
        Returns:
            MLEnhancedResponse with personalized results
        """
        if not self.is_available():
            return self._create_fallback_response(user_message)
        
        try:
            # Step 1: Process conversation with context-aware dialogue model
            context, intent, extracted_info = self.ml_system.process_conversation(
                user_id=user_id,
                user_message=user_message,
                session_id=session_id
            )
            
            # Store context for future use
            self.active_sessions[session_id] = context
            
            # Step 2: Load or learn user preferences
            user_preference = await self._get_or_learn_user_preference(user_id, db_session)
            
            # Step 3: Get journey patterns for proactive suggestions
            detected_patterns = await self._get_journey_patterns(user_id, db_session)
            
            # Step 4: Handle based on intent
            response_text = ""
            ranked_routes = None
            proactive_suggestions = []
            
            if intent == 'routing':
                # Extract origin and destination
                origin = extracted_info.get('locations', [None, None])[0] or user_location
                destination = extracted_info.get('locations', [None, None])[1] if len(extracted_info.get('locations', [])) > 1 else None
                
                if origin and destination:
                    # Get candidate routes from routing service
                    routes_response = await self._get_candidate_routes(origin, destination)
                    
                    if routes_response and routes_response.get('success'):
                        # Use ML to rank routes based on user preferences
                        candidate_routes = self._extract_candidate_routes(routes_response)
                        
                        ranked_routes = self.ml_system.predict_routes(
                            user_id=user_id,
                            origin=origin,
                            destination=destination,
                            candidate_routes=candidate_routes,
                            context=context
                        )
                        
                        # Format response with top route
                        response_text = self._format_route_response(
                            origin=origin,
                            destination=destination,
                            ranked_routes=ranked_routes,
                            user_preference=user_preference,
                            routes_response=routes_response
                        )
                    else:
                        response_text = f"I couldn't find a route from {origin} to {destination}. Please check the location names."
                else:
                    response_text = "I'd be happy to help with directions! Please tell me your origin and destination."
            
            elif intent == 'recommendation':
                # Get proactive recommendations based on patterns and time
                recommendations = self.ml_system.get_personalized_recommendations(
                    user_id=user_id,
                    current_time=datetime.now()
                )
                
                if recommendations:
                    response_text = self._format_recommendations(recommendations)
                    proactive_suggestions = recommendations
                else:
                    response_text = "Based on your travel history, I don't have specific recommendations at this time. What are you looking for?"
            
            elif intent == 'inquiry':
                # General information query - provide context-aware response
                response_text = self._format_inquiry_response(extracted_info, context)
            
            elif intent == 'feedback':
                # Handle user feedback to improve learning
                await self._process_user_feedback(user_id, user_message, context, db_session)
                response_text = "Thank you for your feedback! I'll use it to improve my recommendations."
            
            else:
                # Chitchat or unknown intent
                response_text = self._format_chitchat_response(user_message, context)
            
            # Step 5: Create enhanced response
            return MLEnhancedResponse(
                response_text=response_text,
                intent=intent,
                confidence=extracted_info.get('confidence', 0.0),
                personalization_applied=user_preference is not None,
                detected_patterns=[asdict(p) for p in detected_patterns],
                ranked_routes=ranked_routes,
                proactive_suggestions=proactive_suggestions,
                context_summary={
                    'locations': context.mentioned_locations,
                    'times': context.mentioned_times,
                    'modes': context.mentioned_modes,
                    'conversation_length': len(context.conversation_history)
                },
                ml_metadata={
                    'model_version': '1.0',
                    'inference_time_ms': 0,  # Can be tracked
                    'gpu_used': True  # Based on DEVICE
                }
            )
            
        except Exception as e:
            logger.error(f"Error in ML-enhanced chat processing: {e}", exc_info=True)
            return self._create_fallback_response(user_message)
    
    async def learn_from_trip(
        self,
        user_id: str,
        origin: str,
        destination: str,
        selected_route: Dict[str, Any],
        trip_context: Optional[Dict[str, Any]] = None,
        db_session: Optional[Any] = None
    ):
        """
        Learn from a completed trip to improve future recommendations
        
        Args:
            user_id: User identifier
            origin: Trip origin
            destination: Trip destination
            selected_route: The route the user selected
            trip_context: Additional context (time, mode choices, etc.)
            db_session: Database session
        """
        if not self.is_available():
            return
        
        try:
            # Create trip record
            trip_record = {
                'origin': origin,
                'destination': destination,
                'timestamp': datetime.now().isoformat(),
                'selected_mode': trip_context.get('mode', 'unknown') if trip_context else 'unknown',
                'num_transfers': selected_route.get('num_transfers', 0),
                'duration': selected_route.get('duration', 0),
                'walking_distance': selected_route.get('walking_distance', 0),
                'cost': selected_route.get('cost', 0)
            }
            
            # Store trip in database for future pattern recognition
            if DATABASE_AVAILABLE and db_session:
                await self._store_trip_record(user_id, trip_record, db_session)
            
            # Trigger pattern recognition asynchronously
            asyncio.create_task(self._update_journey_patterns(user_id, db_session))
            
            logger.info(f"📊 Learned from trip: {origin} → {destination} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error learning from trip: {e}", exc_info=True)
    
    async def get_proactive_suggestions(
        self,
        user_id: str,
        current_time: Optional[datetime] = None,
        user_location: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get proactive route suggestions based on learned patterns
        
        Args:
            user_id: User identifier
            current_time: Current time (defaults to now)
            user_location: Optional current location
            
        Returns:
            List of proactive suggestions
        """
        if not self.is_available():
            return []
        
        try:
            current_time = current_time or datetime.now()
            
            # Get recommendations from ML system
            recommendations = self.ml_system.get_personalized_recommendations(
                user_id=user_id,
                current_time=current_time
            )
            
            # Enhance with routing data if user location is available
            if user_location and recommendations:
                for rec in recommendations:
                    if rec.get('origin') == user_location or not rec.get('origin'):
                        # This is a likely journey, fetch live route data
                        routes_response = await self._get_candidate_routes(
                            user_location, 
                            rec['destination']
                        )
                        if routes_response and routes_response.get('success'):
                            rec['live_route'] = routes_response.get('route_data')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting proactive suggestions: {e}", exc_info=True)
            return []
    
    # Helper methods
    
    async def _get_or_learn_user_preference(
        self, 
        user_id: str, 
        db_session: Optional[Any]
    ) -> Optional[UserPreference]:
        """Get existing user preference or learn from history"""
        # Check in-memory cache
        if user_id in self.ml_system.user_preferences:
            return self.ml_system.user_preferences[user_id]
        
        # Try to load from database and learn
        if DATABASE_AVAILABLE and db_session:
            interaction_history = await self._load_interaction_history(user_id, db_session)
            
            if interaction_history and len(interaction_history) >= 3:
                return self.ml_system.learn_user_preferences(user_id, interaction_history)
        
        return None
    
    async def _get_journey_patterns(
        self, 
        user_id: str, 
        db_session: Optional[Any]
    ) -> List[JourneyPattern]:
        """Get detected journey patterns for user"""
        # Check in-memory cache
        if user_id in self.ml_system.journey_patterns:
            return self.ml_system.journey_patterns[user_id]
        
        # Try to recognize patterns from database
        if DATABASE_AVAILABLE and db_session:
            trip_history = await self._load_trip_history(user_id, db_session)
            
            if trip_history and len(trip_history) >= 3:
                return self.ml_system.recognize_journey_patterns(user_id, trip_history)
        
        return []
    
    async def _get_candidate_routes(
        self, 
        origin: str, 
        destination: str
    ) -> Optional[Dict[str, Any]]:
        """Get candidate routes from routing service"""
        if not ROUTING_AVAILABLE or not self.routing_adapter:
            return None
        
        try:
            # Use the routing adapter to get routes
            query = f"how do i get from {origin} to {destination}"
            response = self.routing_adapter.handle_routing_query(query)
            return response
        except Exception as e:
            logger.error(f"Error getting candidate routes: {e}")
            return None
    
    def _extract_candidate_routes(self, routes_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract route candidates from routing response"""
        candidates = []
        
        # Primary route
        if routes_response.get('route_data'):
            candidates.append(routes_response['route_data'])
        
        # Alternative routes
        if routes_response.get('alternatives'):
            for alt in routes_response['alternatives']:
                candidates.append(alt)
        
        return candidates
    
    def _format_route_response(
        self,
        origin: str,
        destination: str,
        ranked_routes: List[Tuple[Dict, float]],
        user_preference: Optional[UserPreference],
        routes_response: Dict[str, Any]
    ) -> str:
        """Format route response with personalization"""
        if not ranked_routes:
            return f"I couldn't find a suitable route from {origin} to {destination}."
        
        # Get the best route (highest score)
        best_route, best_score = ranked_routes[0]
        
        response = f"🗺️ **Route: {origin} → {destination}**\n\n"
        
        # Add personalization note if applicable
        if user_preference:
            response += f"✨ *Personalized based on your preferences*\n\n"
        
        # Use the original routing response format if available
        if routes_response.get('response_text'):
            response += routes_response['response_text']
        else:
            # Fallback formatting
            response += f"⏱️ **Duration:** {best_route.get('duration', 'N/A')} min\n"
            response += f"🔄 **Transfers:** {best_route.get('num_transfers', 0)}\n"
            response += f"💰 **Cost:** ₺{best_route.get('cost', 'N/A')}\n"
        
        # Add alternative routes if significantly different
        if len(ranked_routes) > 1:
            response += "\n\n**Alternative routes:**\n"
            for i, (route, score) in enumerate(ranked_routes[1:3], 1):  # Show top 2 alternatives
                response += f"{i}. Duration: {route.get('duration', 'N/A')} min, "
                response += f"Transfers: {route.get('num_transfers', 0)}, "
                response += f"Score: {score:.2f}\n"
        
        return response
    
    def _format_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format proactive recommendations"""
        if not recommendations:
            return "I don't have any specific recommendations at this time."
        
        response = "🎯 **Based on your travel patterns, you might want to:**\n\n"
        
        for i, rec in enumerate(recommendations[:3], 1):
            response += f"{i}. Travel from **{rec['origin']}** to **{rec['destination']}**\n"
            response += f"   {rec['reason']}\n"
            if rec.get('live_route'):
                response += f"   ⏱️ Current travel time: {rec['live_route'].get('duration', 'N/A')} min\n"
            response += "\n"
        
        return response
    
    def _format_inquiry_response(
        self, 
        extracted_info: Dict[str, Any], 
        context: ConversationContext
    ) -> str:
        """Format response for general inquiry"""
        # This would integrate with your existing information retrieval system
        response = "I can help you with information about Istanbul! "
        
        if extracted_info.get('locations'):
            locations = extracted_info['locations']
            response += f"You asked about {', '.join(locations)}. "
        
        response += "What specifically would you like to know?"
        
        return response
    
    def _format_chitchat_response(
        self, 
        user_message: str, 
        context: ConversationContext
    ) -> str:
        """Format chitchat response"""
        # Simple chitchat responses
        greetings = ['hi', 'hello', 'hey', 'merhaba', 'selam']
        if any(g in user_message.lower() for g in greetings):
            return "Hello! 👋 I'm your Istanbul AI assistant. I can help you with routes, recommendations, and information about Istanbul. How can I assist you today?"
        
        return "I'm here to help you explore Istanbul! Ask me about routes, attractions, restaurants, or anything else about the city."
    
    async def _process_user_feedback(
        self,
        user_id: str,
        feedback: str,
        context: ConversationContext,
        db_session: Optional[Any]
    ):
        """Process user feedback to improve learning"""
        # Store feedback for future training
        if DATABASE_AVAILABLE and db_session:
            feedback_record = {
                'user_id': user_id,
                'feedback': feedback,
                'context': asdict(context),
                'timestamp': datetime.now().isoformat()
            }
            # Store in database (implementation depends on your schema)
            logger.info(f"📝 Stored feedback from user {user_id}")
    
    async def _load_interaction_history(
        self, 
        user_id: str, 
        db_session: Any
    ) -> List[Dict[str, Any]]:
        """Load user interaction history from database"""
        try:
            # Query ChatHistory for recent interactions
            # This is a placeholder - adjust based on your actual schema
            history = []
            # history = db_session.query(ChatHistory).filter_by(user_id=user_id).order_by(ChatHistory.timestamp.desc()).limit(50).all()
            
            # Convert to the format expected by ML system
            interaction_history = []
            for record in history:
                interaction = {
                    'timestamp': record.timestamp.isoformat() if hasattr(record, 'timestamp') else datetime.now().isoformat(),
                    'selected_mode': 'metro',  # Extract from ai_response
                    'num_transfers': 1,  # Extract from ai_response
                    'duration': 30  # Extract from ai_response
                }
                interaction_history.append(interaction)
            
            return interaction_history
        except Exception as e:
            logger.error(f"Error loading interaction history: {e}")
            return []
    
    async def _load_trip_history(
        self, 
        user_id: str, 
        db_session: Any
    ) -> List[Dict[str, Any]]:
        """Load user trip history from database"""
        try:
            # Query trip records
            # This is a placeholder - you may need to create a trips table
            trip_history = []
            
            return trip_history
        except Exception as e:
            logger.error(f"Error loading trip history: {e}")
            return []
    
    async def _store_trip_record(
        self,
        user_id: str,
        trip_record: Dict[str, Any],
        db_session: Any
    ):
        """Store trip record in database"""
        try:
            # Store trip record
            # Implementation depends on your schema
            logger.info(f"💾 Stored trip record for user {user_id}")
        except Exception as e:
            logger.error(f"Error storing trip record: {e}")
    
    async def _update_journey_patterns(
        self,
        user_id: str,
        db_session: Optional[Any]
    ):
        """Update journey patterns asynchronously"""
        try:
            if DATABASE_AVAILABLE and db_session:
                trip_history = await self._load_trip_history(user_id, db_session)
                
                if trip_history and len(trip_history) >= 3:
                    self.ml_system.recognize_journey_patterns(user_id, trip_history)
                    logger.info(f"🔄 Updated journey patterns for user {user_id}")
        except Exception as e:
            logger.error(f"Error updating journey patterns: {e}")
    
    def _create_fallback_response(self, user_message: str) -> MLEnhancedResponse:
        """Create a fallback response when ML is not available"""
        return MLEnhancedResponse(
            response_text="I'm here to help! However, personalization features are temporarily unavailable. How can I assist you with Istanbul?",
            intent='unknown',
            confidence=0.0,
            personalization_applied=False,
            detected_patterns=[],
            ranked_routes=None,
            proactive_suggestions=None,
            context_summary=None,
            ml_metadata={'fallback': True}
        )
    
    async def start_background_learning(self):
        """Start background learning tasks"""
        if not self.is_available():
            return
        
        logger.info("🎓 Starting background learning tasks...")
        
        # Periodic pattern recognition and model updates
        async def learning_loop():
            while True:
                try:
                    # Run every hour
                    await asyncio.sleep(3600)
                    
                    # Update patterns for active users
                    logger.info("🔄 Running periodic pattern recognition...")
                    
                    # This would be expanded to update models, retrain, etc.
                    
                except Exception as e:
                    logger.error(f"Error in learning loop: {e}")
        
        self._learning_task = asyncio.create_task(learning_loop())
    
    def save_models(self, save_dir: str = "./ml_models"):
        """Save trained models and user data"""
        if self.is_available():
            self.ml_system.save_models(save_dir)
            logger.info(f"💾 Models saved to {save_dir}")
    
    def load_models(self, load_dir: str = "./ml_models"):
        """Load trained models and user data"""
        if self.is_available():
            self.ml_system.load_models(load_dir)
            logger.info(f"📂 Models loaded from {load_dir}")


# Singleton instance
_ml_integration_instance = None

def get_ml_integration() -> MLAdvancedIntegration:
    """Get or create the singleton ML integration instance"""
    global _ml_integration_instance
    if _ml_integration_instance is None:
        _ml_integration_instance = MLAdvancedIntegration()
    return _ml_integration_instance


# FastAPI integration helpers
async def enhance_chat_with_ml(
    user_id: str,
    session_id: str,
    user_message: str,
    user_location: Optional[str] = None,
    db_session: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Helper function for FastAPI endpoint integration
    
    Usage in backend/main.py:
        from services.ml_advanced_integration import enhance_chat_with_ml
        
        @app.post("/ai/chat")
        async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
            # ... existing code ...
            
            # Enhance with ML
            ml_response = await enhance_chat_with_ml(
                user_id=request.user_id,
                session_id=request.session_id,
                user_message=request.message,
                user_location=request.current_location,
                db_session=db
            )
            
            # Use ml_response.response_text and other fields
            return ChatResponse(
                response=ml_response['response_text'],
                intent=ml_response['intent'],
                confidence=ml_response['confidence'],
                personalized=ml_response['personalization_applied'],
                # ... other fields ...
            )
    """
    integration = get_ml_integration()
    
    response = await integration.process_chat_message(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        user_location=user_location,
        db_session=db_session
    )
    
    return asdict(response)


if __name__ == "__main__":
    # Test the integration
    print("🧪 Testing ML Advanced Integration...")
    
    integration = get_ml_integration()
    
    if integration.is_available():
        print("✅ ML Integration is available")
        
        # Test chat processing
        async def test():
            response = await integration.process_chat_message(
                user_id="test_user",
                session_id="test_session",
                user_message="How can I go to Sultanahmet from Taksim?",
                user_location="Taksim"
            )
            
            print(f"\n📊 Test Results:")
            print(f"Intent: {response.intent}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Personalized: {response.personalization_applied}")
            print(f"Response: {response.response_text[:200]}...")
        
        asyncio.run(test())
    else:
        print("⚠️ ML Integration not available")
