"""
Example Integration: Advanced ML System in Chat Endpoint
=========================================================

This file shows exactly how to integrate the advanced ML system
into your existing backend/main.py chat endpoint.

Copy the relevant sections into your main.py file.
"""

# ============================================================================
# STEP 1: Add Imports at the Top of backend/main.py
# ============================================================================

from services.ml_advanced_integration import (
    enhance_chat_with_ml, 
    get_ml_integration,
    MLEnhancedResponse
)

# ============================================================================
# STEP 2: Initialize ML System in Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Istanbul AI Backend...")
    
    # ... your existing startup code ...
    
    # Initialize ML Advanced System (NEW!)
    try:
        ml_integration = get_ml_integration()
        if ml_integration.is_available():
            logger.info("🧠 ML Advanced System initialized successfully")
            
            # Start background learning tasks
            await ml_integration.start_background_learning()
            logger.info("🎓 Background learning tasks started")
            
            # Load pre-trained models if available
            try:
                ml_integration.load_models("./ml_models")
                logger.info("📂 Pre-trained models loaded")
            except Exception as e:
                logger.info(f"ℹ️ No pre-trained models found (will train from scratch): {e}")
        else:
            logger.warning("⚠️ ML Advanced System not available - using fallback")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML Advanced System: {e}")


# ============================================================================
# STEP 3: Add Shutdown Handler to Save Models
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Save models and clean up on shutdown"""
    logger.info("🛑 Shutting down Istanbul AI Backend...")
    
    # Save trained models (NEW!)
    try:
        ml_integration = get_ml_integration()
        if ml_integration.is_available():
            ml_integration.save_models("./ml_models")
            logger.info("💾 ML models saved successfully")
    except Exception as e:
        logger.error(f"❌ Failed to save ML models: {e}")


# ============================================================================
# STEP 4: Update Your Chat Endpoint
# ============================================================================

@app.post("/ai/chat", response_model=ChatResponse, tags=["AI Chat"])
async def chat_endpoint(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Main AI chat endpoint with ML enhancements
    
    This is an example of how to integrate ML into your existing endpoint.
    Adapt the field names to match your actual ChatRequest and ChatResponse models.
    """
    
    try:
        # Extract request data
        user_input = request.message
        session_id = request.session_id or str(uuid.uuid4())
        user_id = request.user_id or session_id
        current_location = getattr(request, 'current_location', None)
        
        logger.info(f"💬 Chat request from user {user_id[:8]}...")
        
        # ====================================================================
        # ML ENHANCEMENT (NEW!)
        # ====================================================================
        
        ml_response_dict = await enhance_chat_with_ml(
            user_id=user_id,
            session_id=session_id,
            user_message=user_input,
            user_location=current_location,
            db_session=db
        )
        
        # Extract ML-enhanced data
        response_text = ml_response_dict['response_text']
        intent = ml_response_dict['intent']
        confidence = ml_response_dict['confidence']
        personalization_applied = ml_response_dict['personalization_applied']
        detected_patterns = ml_response_dict['detected_patterns']
        proactive_suggestions = ml_response_dict.get('proactive_suggestions', [])
        
        logger.info(f"🧠 ML Intent: {intent} (confidence: {confidence:.2f}, personalized: {personalization_applied})")
        
        # ====================================================================
        # Your Existing Logic (Adapted)
        # ====================================================================
        
        # If you have existing intent classification, you can:
        # 1. Replace it entirely with ML intent (recommended for routing)
        # 2. Use ML as a fallback when existing system has low confidence
        # 3. Use both and combine results
        
        # Example: Use ML intent for routing, keep existing for other intents
        if intent == 'routing':
            # ML has already formatted a route response in response_text
            final_response = response_text
            
            # If you need the route data for other purposes:
            ranked_routes = ml_response_dict.get('ranked_routes', [])
            if ranked_routes:
                best_route, best_score = ranked_routes[0]
                logger.info(f"🗺️ Best route score: {best_score:.2f}")
        
        elif intent == 'recommendation':
            # ML has proactive suggestions based on learned patterns
            final_response = response_text
            
            # You can enhance with your existing recommendation system
            if proactive_suggestions:
                logger.info(f"🎯 {len(proactive_suggestions)} proactive suggestions provided")
        
        else:
            # For other intents, you might want to use your existing system
            # and fall back to ML response if needed
            final_response = response_text
        
        # ====================================================================
        # Store Interaction History for Future Learning
        # ====================================================================
        
        # Save chat history (your existing code)
        chat_record = ChatHistory(
            session_id=session_id,
            user_message=user_input,
            ai_response=final_response,
            timestamp=datetime.utcnow()
        )
        db.add(chat_record)
        db.commit()
        
        # ====================================================================
        # Build Response
        # ====================================================================
        
        # Adapt these fields to match your actual ChatResponse model
        return ChatResponse(
            response=final_response,
            intent=intent,
            confidence=confidence,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat(),
            
            # NEW ML-enhanced fields (add to your ChatResponse model if not present)
            personalized=personalization_applied,
            detected_patterns=detected_patterns,
            proactive_suggestions=proactive_suggestions,
            
            # Your existing fields...
            # suggestions=[...],
            # has_map_data=...,
            # etc.
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}", exc_info=True)
        
        # Fallback response
        return ChatResponse(
            response="I apologize, but I encountered an error. Please try again.",
            intent="error",
            confidence=0.0,
            session_id=session_id,
            timestamp=datetime.utcnow().isoformat()
        )


# ============================================================================
# STEP 5: Add New Endpoints for ML Features
# ============================================================================

@app.post("/api/trips/complete", tags=["Trips"])
async def complete_trip(
    trip_data: TripCompletionRequest,
    db: Session = Depends(get_db)
):
    """
    Record a completed trip for ML learning
    
    This endpoint should be called when a user completes a journey.
    The ML system will learn from this to improve future recommendations.
    """
    
    try:
        user_id = trip_data.user_id
        
        # ... your existing trip completion logic ...
        
        # Learn from completed trip (NEW!)
        ml_integration = get_ml_integration()
        if ml_integration.is_available():
            await ml_integration.learn_from_trip(
                user_id=user_id,
                origin=trip_data.origin,
                destination=trip_data.destination,
                selected_route={
                    'duration': trip_data.duration_minutes,
                    'num_transfers': trip_data.num_transfers,
                    'cost': trip_data.cost_tl,
                    'walking_distance': trip_data.walking_distance_meters,
                    'modes': trip_data.transport_modes
                },
                trip_context={
                    'mode': trip_data.primary_transport_mode,
                    'time': trip_data.completion_time,
                    'rating': getattr(trip_data, 'rating', None),
                    'weather': getattr(trip_data, 'weather', None)
                },
                db_session=db
            )
            
            logger.info(f"📊 Learned from trip: {trip_data.origin} → {trip_data.destination}")
        
        return {
            "status": "success",
            "message": "Trip recorded and learning updated",
            "trip_id": trip_data.trip_id
        }
        
    except Exception as e:
        logger.error(f"❌ Error recording trip: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/suggestions/proactive", tags=["Suggestions"])
async def get_proactive_suggestions(
    user_id: str = Query(..., description="User ID"),
    current_location: Optional[str] = Query(None, description="Current location")
):
    """
    Get proactive journey suggestions based on learned patterns
    
    This endpoint provides personalized suggestions for journeys the user
    might want to take based on their historical patterns and current context.
    
    Example: If user typically travels to work at 8 AM on weekdays, this will
    suggest that journey when called around that time.
    """
    
    try:
        ml_integration = get_ml_integration()
        
        if not ml_integration.is_available():
            return {
                "suggestions": [],
                "message": "Proactive suggestions not available"
            }
        
        suggestions = await ml_integration.get_proactive_suggestions(
            user_id=user_id,
            user_location=current_location
        )
        
        return {
            "suggestions": suggestions,
            "count": len(suggestions),
            "generated_at": datetime.utcnow().isoformat(),
            "location_context": current_location
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting proactive suggestions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/user-profile/{user_id}", tags=["ML Insights"])
async def get_ml_user_profile(user_id: str):
    """
    Get ML-learned user profile (preferences and patterns)
    
    Useful for debugging and showing users what the system has learned.
    """
    
    try:
        ml_integration = get_ml_integration()
        
        if not ml_integration.is_available():
            return {"error": "ML system not available"}
        
        ml_system = ml_integration.ml_system
        
        # Get user preference
        user_preference = ml_system.user_preferences.get(user_id)
        
        # Get journey patterns
        journey_patterns = ml_system.journey_patterns.get(user_id, [])
        
        return {
            "user_id": user_id,
            "has_learned_preference": user_preference is not None,
            "preference": {
                "preferred_modes": user_preference.preferred_modes if user_preference else [],
                "speed_priority": user_preference.speed_priority if user_preference else None,
                "comfort_priority": user_preference.comfort_priority if user_preference else None,
                "avoid_transfers": user_preference.avoid_transfers if user_preference else None,
                "last_updated": user_preference.last_updated.isoformat() if user_preference and user_preference.last_updated else None
            } if user_preference else None,
            "journey_patterns": [
                {
                    "origin": p.origin,
                    "destination": p.destination,
                    "frequency": p.frequency,
                    "typical_time": p.typical_time,
                    "typical_days": p.typical_days
                }
                for p in journey_patterns
            ],
            "pattern_count": len(journey_patterns)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting ML user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 6: Update Your Pydantic Models (If Needed)
# ============================================================================

"""
Add these fields to your ChatResponse model if not already present:

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    session_id: str
    timestamp: str
    
    # NEW ML-enhanced fields
    personalized: bool = False
    detected_patterns: List[Dict[str, Any]] = []
    proactive_suggestions: List[Dict[str, Any]] = []
    
    # Your other existing fields...

class TripCompletionRequest(BaseModel):
    user_id: str
    trip_id: str
    origin: str
    destination: str
    duration_minutes: int
    num_transfers: int
    cost_tl: float
    walking_distance_meters: int
    transport_modes: List[str]
    primary_transport_mode: str
    completion_time: datetime
    rating: Optional[int] = None
    weather: Optional[str] = None
"""

# ============================================================================
# Testing the Integration
# ============================================================================

if __name__ == "__main__":
    """
    Quick test to verify ML integration works
    """
    import asyncio
    from services.ml_advanced_integration import get_ml_integration
    
    async def test():
        print("🧪 Testing ML Integration...")
        
        integration = get_ml_integration()
        
        if not integration.is_available():
            print("❌ ML Integration not available")
            return
        
        print("✅ ML Integration available")
        
        # Test chat processing
        print("\n📊 Testing chat message processing...")
        response_dict = await enhance_chat_with_ml(
            user_id="test_user_123",
            session_id="test_session_123",
            user_message="How can I go to Sultanahmet from Taksim?",
            user_location="Taksim",
            db_session=None  # Would be actual DB session in production
        )
        
        print(f"Intent: {response_dict['intent']}")
        print(f"Confidence: {response_dict['confidence']:.2f}")
        print(f"Personalized: {response_dict['personalization_applied']}")
        print(f"Response: {response_dict['response_text'][:100]}...")
        
        print("\n✅ Test completed successfully!")
    
    asyncio.run(test())
