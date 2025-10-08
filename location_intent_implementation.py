#!/usr/bin/env python3
"""
PRACTICAL IMPLEMENTATION: Location Intent Detection for AI Istanbul

This file contains the exact code modifications you need to add to your main.py
to enable automatic detection of location-based queries (restaurants, museums, etc.)
"""

# ============================================================================
# 1. ADD THESE IMPORTS TO THE TOP OF YOUR main.py
# ============================================================================

# Add this import after your existing imports in main.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'load-testing'))

from location_intent_detector import LocationIntentDetector, LocationIntentType

# ============================================================================
# 2. ADD THIS GLOBAL VARIABLE AFTER YOUR FastAPI APP INITIALIZATION
# ============================================================================

# Add this line after: app = FastAPI(title="AI Istanbul Backend")
location_detector = LocationIntentDetector()

# ============================================================================
# 3. REPLACE YOUR EXISTING chat_with_ai FUNCTION WITH THIS ENHANCED VERSION
# ============================================================================

@app.post("/ai", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai_enhanced(
    request: ChatRequest,
    user_request: Request
):
    """
    Enhanced AI chat endpoint with location intent detection
    
    This version automatically detects when users want location-based
    recommendations and responds appropriately.
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get user IP for context
        user_ip = user_request.client.host if user_request.client else None
        
        # Sanitize and validate input
        user_message = sanitize_user_input(request.message)
        if not user_message:
            return ChatResponse(
                response="I need a valid message to help you with Istanbul information.",
                session_id=session_id,
                success=False,
                system_type="ultra_specialized_istanbul_ai"
            )
        
        # Process location context if available
        location_info = None
        if request.location_context and request.location_context.has_location:
            location_info = {
                'latitude': request.location_context.latitude,
                'longitude': request.location_context.longitude,
                'district': request.location_context.district,
                'nearby_pois': request.location_context.nearby_pois or [],
                'accuracy': request.location_context.accuracy
            }
            print(f"🌍 Location-aware request - District: {location_info.get('district')}, POIs: {len(location_info.get('nearby_pois', []))}")
        
        print(f"🏛️ AI Chat Request - Session: {session_id[:8]}..., Message: '{user_message[:50]}...', Location: {bool(location_info)}")
        
        # ===== NEW: LOCATION INTENT DETECTION =====
        detected_intents = location_detector.detect_intent(user_message, location_info)
        
        if detected_intents:
            primary_intent = detected_intents[0]  # Highest confidence intent
            print(f"🎯 Location intent detected: {primary_intent.intent_type.value} (confidence: {primary_intent.confidence:.2f})")
            
            # If we have high confidence in a location intent, handle it specially
            if primary_intent.confidence >= 0.4:  # Lowered threshold for more responsiveness
                
                # Generate response configuration
                response_config = location_detector.generate_location_response(primary_intent, location_info)
                
                # Handle restaurant requests
                if primary_intent.intent_type == LocationIntentType.RESTAURANTS:
                    response = await handle_restaurant_intent(primary_intent, location_info, response_config, user_message)
                    if response:
                        return ChatResponse(
                            response=response,
                            session_id=session_id,
                            success=True,
                            system_type="location_intent_restaurants"
                        )
                
                # Handle museum requests
                elif primary_intent.intent_type == LocationIntentType.MUSEUMS:
                    response = await handle_museum_intent(primary_intent, location_info, response_config, user_message)
                    if response:
                        return ChatResponse(
                            response=response,
                            session_id=session_id,
                            success=True,
                            system_type="location_intent_museums"
                        )
                
                # Handle route planning requests
                elif primary_intent.intent_type == LocationIntentType.ROUTE_PLANNING:
                    response = await handle_route_intent(primary_intent, location_info, response_config, user_message)
                    if response:
                        return ChatResponse(
                            response=response,
                            session_id=session_id,
                            success=True,
                            system_type="location_intent_routes"
                        )
                
                # For other intents, enhance the AI context
                else:
                    enhanced_context = build_intent_context(primary_intent, location_info)
                    ai_result = await get_istanbul_ai_response_with_quality(
                        user_message, session_id, user_ip, 
                        location_context=location_info,
                        enhanced_prompt=enhanced_context
                    )
                    
                    if ai_result and ai_result.get('success'):
                        return ChatResponse(
                            response=ai_result['response'],
                            session_id=session_id,
                            success=True,
                            system_type="location_intent_enhanced"
                        )
        
        # Fall back to standard AI response
        ai_result = await get_istanbul_ai_response_with_quality(user_message, session_id, user_ip, location_context=location_info)
        
        if ai_result and ai_result.get('success'):
            return ChatResponse(
                response=ai_result['response'],
                session_id=session_id,
                success=True,
                system_type=ai_result.get('system_type', 'ultra_specialized_istanbul_ai'),
                quality_assessment=ai_result.get('quality_assessment')
            )
        else:
            # Fallback response
            fallback_response = (
                "I'm here to help you explore Istanbul! You can ask me about:\\n\\n"
                "• Restaurants and local cuisine\\n"
                "• Museums and cultural attractions\\n"
                "• Neighborhoods and districts\\n"
                "• Transportation and getting around\\n"
                "• Shopping and entertainment\\n"
                "• Daily activities and local tips\\n\\n"
                "What would you like to know about Istanbul?"
            )
            
            return ChatResponse(
                response=fallback_response,
                session_id=session_id,
                success=True,
                system_type="ultra_specialized_istanbul_ai_fallback"
            )
            
    except Exception as e:
        print(f"❌ AI Chat endpoint error: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            response="I'm sorry, I encountered an issue processing your request. Please try again.",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            system_type="error_fallback"
        )

# ============================================================================
# 4. ADD THESE HELPER FUNCTIONS TO YOUR main.py
# ============================================================================

async def handle_restaurant_intent(intent, location_info, response_config, original_message):
    """Handle restaurant-specific intents"""
    try:
        if not location_info:
            return generate_restaurant_response_without_location(intent, original_message)
        
        # Try to get restaurants from your location API
        search_params = response_config['search_params']
        
        # You can make an API call to your restaurant service here
        # For now, generate a smart response based on the intent
        
        response_parts = []
        response_parts.append(f"🍽️ Looking for restaurants in {location_info.get('district', 'your area')}...")
        
        if intent.specific_requirements.get('cuisine'):
            cuisines = ', '.join(intent.specific_requirements['cuisine'])
            response_parts.append(f"\\nI see you're interested in {cuisines} cuisine.")
        
        if intent.specific_requirements.get('dining_style'):
            styles = ', '.join(intent.specific_requirements['dining_style'])
            response_parts.append(f"Looking for {styles} dining options.")
        
        if intent.distance_preference:
            response_parts.append(f"Within {intent.distance_preference} as requested.")
        
        response_parts.append("\\nLet me suggest some great options:")
        
        # Add some sample restaurants based on district
        district = location_info.get('district', '').lower()
        if 'sultanahmet' in district:
            response_parts.append("\\n• **Pandeli**: Historic Ottoman restaurant in Spice Bazaar")
            response_parts.append("• **Deraliye**: Traditional Ottoman cuisine near Sultanahmet")
            response_parts.append("• **Balıkçı Sabahattin**: Fresh seafood in historic setting")
        elif 'beyoğlu' in district or 'taksim' in district:
            response_parts.append("\\n• **Mikla**: Modern Turkish with Bosphorus view")
            response_parts.append("• **Karaköy Lokantası**: Contemporary Turkish in stylish setting")
            response_parts.append("• **Çiya Sofrası**: Authentic Anatolian dishes")
        else:
            response_parts.append("\\n• **Hamdi Restaurant**: Famous for kebabs near Galata Bridge")
            response_parts.append("• **Sunset Grill & Bar**: International cuisine with amazing views")
            response_parts.append("• **Nusr-Et**: World-famous steakhouse experience")
        
        response_parts.append("\\nWould you like more specific recommendations or details about any of these places?")
        
        return '\\n'.join(response_parts)
        
    except Exception as e:
        print(f"Error handling restaurant intent: {e}")
        return None

async def handle_museum_intent(intent, location_info, response_config, original_message):
    """Handle museum-specific intents"""
    try:
        if not location_info:
            return generate_museum_response_without_location(intent, original_message)
        
        response_parts = []
        response_parts.append(f"🏛️ Finding museums near {location_info.get('district', 'your location')}...")
        
        if intent.specific_requirements.get('specific_sites'):
            sites = ', '.join(intent.specific_requirements['specific_sites'])
            response_parts.append(f"\\nI see you're interested in {sites}.")
        
        response_parts.append("\\nHere are some excellent museums in your area:")
        
        # Add museums based on district
        district = location_info.get('district', '').lower()
        if 'sultanahmet' in district:
            response_parts.append("\\n• **Hagia Sophia**: Iconic Byzantine and Ottoman architecture")
            response_parts.append("• **Topkapi Palace**: Former Ottoman imperial palace")
            response_parts.append("• **Istanbul Archaeology Museums**: World-class ancient artifacts")
            response_parts.append("• **Basilica Cistern**: Ancient underground marvel")
        elif 'beyoğlu' in district:
            response_parts.append("\\n• **Galata Tower**: Historic tower with panoramic views")
            response_parts.append("• **Istanbul Modern**: Contemporary Turkish art")
            response_parts.append("• **Pera Museum**: European art and Anatolian weights")
        else:
            response_parts.append("\\n• **Dolmabahçe Palace**: 19th-century Ottoman palace")
            response_parts.append("• **Turkish and Islamic Arts Museum**: Islamic art collection")
            response_parts.append("• **Chora Church**: Byzantine mosaics and frescoes")
        
        response_parts.append("\\nWould you like opening hours, ticket information, or directions to any of these museums?")
        
        return '\\n'.join(response_parts)
        
    except Exception as e:
        print(f"Error handling museum intent: {e}")
        return None

async def handle_route_intent(intent, location_info, response_config, original_message):
    """Handle route planning intents"""
    try:
        response_parts = []
        
        if location_info:
            response_parts.append(f"🗺️ Planning routes from {location_info.get('district', 'your location')}...")
        else:
            response_parts.append("🗺️ I can help you with directions in Istanbul...")
        
        response_parts.append("\\nTo get better route recommendations, could you please tell me:")
        response_parts.append("• Where you want to go?")
        response_parts.append("• Your preferred transportation (metro, bus, taxi, walking)?")
        response_parts.append("• Any specific requirements or preferences?")
        
        response_parts.append("\\nMeanwhile, here are some general transportation tips:")
        response_parts.append("• **Metro**: Fast and efficient for longer distances")
        response_parts.append("• **Ferry**: Scenic routes across the Bosphorus")
        response_parts.append("• **Tram**: Great for tourist areas like Sultanahmet")
        response_parts.append("• **Walking**: Best way to explore historic neighborhoods")
        
        return '\\n'.join(response_parts)
        
    except Exception as e:
        print(f"Error handling route intent: {e}")
        return None

def generate_restaurant_response_without_location(intent, original_message):
    """Generate restaurant response when location is not available"""
    response_parts = []
    response_parts.append("🍽️ I'd love to recommend restaurants! For the best suggestions, could you share your location or tell me which area of Istanbul you're in?")
    
    if intent.specific_requirements.get('cuisine'):
        cuisines = ', '.join(intent.specific_requirements['cuisine'])
        response_parts.append(f"\\nI see you're interested in {cuisines} cuisine - Istanbul has amazing options!")
    
    response_parts.append("\\nMeanwhile, here are some top-rated restaurants across Istanbul:")
    response_parts.append("• **Pandeli** (Eminönü): Historic Ottoman cuisine")
    response_parts.append("• **Mikla** (Beyoğlu): Modern Turkish with city views")
    response_parts.append("• **Çiya Sofrası** (Kadıköy): Authentic regional dishes")
    response_parts.append("• **Hamdi Restaurant** (Eminönü): Famous for kebabs")
    
    return '\\n'.join(response_parts)

def generate_museum_response_without_location(intent, original_message):
    """Generate museum response when location is not available"""
    response_parts = []
    response_parts.append("🏛️ I'd be happy to recommend museums! For location-specific suggestions, please let me know which area you're in or planning to visit.")
    
    response_parts.append("\\nHere are Istanbul's must-visit museums:")
    response_parts.append("• **Hagia Sophia** (Sultanahmet): Byzantine and Ottoman marvel")
    response_parts.append("• **Topkapi Palace** (Sultanahmet): Ottoman imperial palace")
    response_parts.append("• **Istanbul Modern** (Beyoğlu): Contemporary Turkish art")
    response_parts.append("• **Dolmabahçe Palace** (Beşiktaş): 19th-century Ottoman palace")
    
    return '\\n'.join(response_parts)

def build_intent_context(intent, location_info):
    """Build enhanced context for AI responses"""
    context_parts = []
    context_parts.append(f"USER_INTENT: {intent.intent_type.value}")
    context_parts.append(f"INTENT_CONFIDENCE: {intent.confidence:.2f}")
    context_parts.append(f"MATCHED_KEYWORDS: {', '.join(intent.keywords_matched)}")
    
    if location_info:
        context_parts.append(f"USER_DISTRICT: {location_info.get('district', 'Unknown')}")
        context_parts.append(f"USER_COORDINATES: {location_info.get('latitude')}, {location_info.get('longitude')}")
    
    if intent.specific_requirements:
        for req_type, values in intent.specific_requirements.items():
            context_parts.append(f"{req_type.upper()}_PREFERENCE: {', '.join(values)}")
    
    if intent.distance_preference:
        context_parts.append(f"DISTANCE_PREFERENCE: {intent.distance_preference}")
    
    enhanced_prompt = f"""
LOCATION INTENT DETECTED:
{chr(10).join(context_parts)}

Please provide a location-aware response that:
1. Acknowledges their specific intent and location context
2. Provides relevant, practical recommendations
3. Includes specific details like opening hours, directions, or booking info
4. Suggests complementary activities or nearby attractions
5. Maintains a friendly, helpful tone
"""
    
    return enhanced_prompt

# ============================================================================
# 5. TESTING THE INTEGRATION
# ============================================================================

if __name__ == "__main__":
    print("Location Intent Detection Integration - Ready!")
    print("Add the above code to your main.py to enable automatic detection of:")
    print("• Restaurant requests ('I'm hungry', 'good Turkish food nearby')")
    print("• Museum queries ('museums near me', 'historical sites')")
    print("• Route planning ('how to get to', 'directions to')")
    print("• General location queries ('what's around here')")
