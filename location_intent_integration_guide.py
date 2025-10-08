#!/usr/bin/env python3
"""
Integration Guide: How to Use Location Intent Detection in AI Istanbul Chat System

This file shows you exactly how to integrate the LocationIntentDetector 
into your main chat system to automatically detect when users want 
location-based recommendations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'load-testing'))

from location_intent_detector import LocationIntentDetector, LocationIntentType

class ChatIntentIntegration:
    """
    Integration class showing how to use location intent detection
    in the AI Istanbul chat system
    """
    
    def __init__(self):
        self.location_detector = LocationIntentDetector()
    
    def process_user_message(self, user_message: str, user_location: dict = None) -> dict:
        """
        This is the main integration function that you would call in your 
        chat endpoint before generating the AI response.
        
        Args:
            user_message: The user's chat message
            user_location: User's location data (lat, lng, district, etc.)
            
        Returns:
            dict: Processing result with intent detection and response strategy
        """
        
        # Step 1: Detect location intents
        print(f"🔍 Analyzing message: '{user_message}'")
        detected_intents = self.location_detector.detect_intent(user_message, user_location)
        
        result = {
            'original_message': user_message,
            'user_location': user_location,
            'detected_intents': detected_intents,
            'response_strategy': 'default',
            'api_calls_needed': [],
            'enhanced_context': None
        }
        
        if detected_intents:
            # Step 2: Process the highest confidence intent
            primary_intent = detected_intents[0]  # Already sorted by confidence
            
            print(f"✅ Primary intent detected: {primary_intent.intent_type.value} (confidence: {primary_intent.confidence:.2f})")
            
            # Step 3: Generate response configuration
            response_config = self.location_detector.generate_location_response(primary_intent, user_location)
            
            # Step 4: Determine response strategy
            result.update({
                'response_strategy': 'location_aware',
                'primary_intent': primary_intent,
                'response_config': response_config,
                'api_calls_needed': self._get_required_api_calls(response_config),
                'enhanced_context': self._build_enhanced_context(primary_intent, response_config, user_location)
            })
            
            print(f"🎯 Response strategy: {result['response_strategy']}")
            print(f"🔧 API calls needed: {result['api_calls_needed']}")
            
        else:
            print("ℹ️ No location intent detected - using default chat response")
        
        return result
    
    def _get_required_api_calls(self, response_config: dict) -> list:
        """Determine what API calls are needed based on the intent"""
        api_calls = []
        
        search_type = response_config.get('search_type')
        search_params = response_config.get('search_params', {})
        user_location = response_config.get('user_location')
        
        if search_type == 'restaurants':
            api_calls.append({
                'endpoint': '/api/location/recommendations',
                'method': 'GET',
                'params': {
                    'type': 'restaurants',
                    'lat': user_location.get('latitude') if user_location else None,
                    'lng': user_location.get('longitude') if user_location else None,
                    'radius': search_params.get('radius', 2000),
                    'cuisine': ','.join(search_params.get('cuisine', [])) if search_params.get('cuisine') else None,
                    'style': ','.join(search_params.get('dining_style', [])) if search_params.get('dining_style') else None
                }
            })
        
        elif search_type == 'museums':
            api_calls.append({
                'endpoint': '/api/location/recommendations',
                'method': 'GET', 
                'params': {
                    'type': 'museums',
                    'lat': user_location.get('latitude') if user_location else None,
                    'lng': user_location.get('longitude') if user_location else None,
                    'radius': search_params.get('radius', 3000),
                    'category': 'museum,cultural_site,historical_site'
                }
            })
        
        elif search_type == 'routes':
            api_calls.append({
                'endpoint': '/api/location/route',
                'method': 'GET',
                'params': {
                    'from_lat': user_location.get('latitude') if user_location else None,
                    'from_lng': user_location.get('longitude') if user_location else None,
                    'transport_modes': ','.join(search_params.get('transport_preferences', ['walking', 'metro']))
                }
            })
        
        elif search_type == 'general':
            api_calls.append({
                'endpoint': '/api/location/recommendations',
                'method': 'GET',
                'params': {
                    'type': 'general',
                    'lat': user_location.get('latitude') if user_location else None,
                    'lng': user_location.get('longitude') if user_location else None,
                    'radius': search_params.get('radius', 1500),
                    'categories': ','.join(search_params.get('categories', ['restaurant', 'museum', 'attraction']))
                }
            })
        
        return api_calls
    
    def _build_enhanced_context(self, intent, response_config: dict, user_location: dict) -> str:
        """Build enhanced context for the AI system"""
        context_parts = []
        
        # Add intent information
        context_parts.append(f"USER_INTENT: {intent.intent_type.value}")
        context_parts.append(f"CONFIDENCE: {intent.confidence:.2f}")
        context_parts.append(f"KEYWORDS: {', '.join(intent.keywords_matched)}")
        
        # Add location context
        if user_location:
            if user_location.get('district'):
                context_parts.append(f"USER_DISTRICT: {user_location['district']}")
            context_parts.append(f"COORDINATES: {user_location.get('latitude', 'N/A')}, {user_location.get('longitude', 'N/A')}")
        
        # Add specific requirements
        if intent.specific_requirements:
            for req_type, values in intent.specific_requirements.items():
                context_parts.append(f"{req_type.upper()}_PREFERENCE: {', '.join(values)}")
        
        # Add distance preference
        if intent.distance_preference:
            context_parts.append(f"DISTANCE_PREFERENCE: {intent.distance_preference}")
        
        return '\n'.join(context_parts)
    
    def generate_response_prompt(self, processing_result: dict) -> str:
        """
        Generate an enhanced prompt for the AI system that includes
        location intent context
        """
        base_prompt = """You are KAM, an AI assistant specializing in Istanbul tourism and local recommendations. 
You provide helpful, accurate, and friendly advice about Istanbul's attractions, restaurants, culture, and activities."""
        
        if processing_result['response_strategy'] == 'location_aware':
            intent = processing_result['primary_intent']
            enhanced_context = processing_result['enhanced_context']
            
            location_prompt = f"""

LOCATION CONTEXT DETECTED:
{enhanced_context}

RESPONSE INSTRUCTIONS:
- The user is specifically looking for {intent.intent_type.value} recommendations
- Provide location-aware suggestions based on their area ({processing_result['user_location'].get('district', 'current location') if processing_result['user_location'] else 'their location'})
- Include specific details about accessibility, opening hours, and practical information
- Mention nearby attractions or complementary activities
- Be conversational and helpful, acknowledging their location context

"""
            return base_prompt + location_prompt
        
        return base_prompt


def demonstrate_integration():
    """Demonstrate how the integration works with example scenarios"""
    
    print("=" * 80)
    print("AI ISTANBUL LOCATION INTENT INTEGRATION DEMONSTRATION")
    print("=" * 80)
    
    integrator = ChatIntentIntegration()
    
    # Example scenarios
    scenarios = [
        {
            'user_message': 'I\'m hungry, any good Turkish restaurants nearby?',
            'user_location': {
                'latitude': 41.0082,
                'longitude': 28.9784,
                'district': 'Sultanahmet',
                'accuracy': 50
            }
        },
        {
            'user_message': 'Show me museums within walking distance',
            'user_location': {
                'latitude': 41.0255,
                'longitude': 28.9744,
                'district': 'Beyoğlu',
                'accuracy': 30
            }
        },
        {
            'user_message': 'What are the best attractions in Istanbul?',
            'user_location': None  # No location available
        },
        {
            'user_message': 'Any romantic restaurants with Bosphorus view?',
            'user_location': {
                'latitude': 41.0039,
                'longitude': 29.0058,
                'district': 'Üsküdar',
                'accuracy': 25
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📝 SCENARIO {i}:")
        print(f"User Message: '{scenario['user_message']}'")
        if scenario['user_location']:
            print(f"User Location: {scenario['user_location']['district']} ({scenario['user_location']['latitude']}, {scenario['user_location']['longitude']})")
        else:
            print("User Location: Not available")
        
        print("-" * 60)
        
        # Process the message
        result = integrator.process_user_message(
            scenario['user_message'], 
            scenario['user_location']
        )
        
        # Show the results
        if result['response_strategy'] == 'location_aware':
            print(f"🎯 INTENT DETECTED: {result['primary_intent'].intent_type.value}")
            print(f"📊 CONFIDENCE: {result['primary_intent'].confidence:.2f}")
            print(f"🔑 KEYWORDS: {', '.join(result['primary_intent'].keywords_matched)}")
            
            if result['primary_intent'].specific_requirements:
                print("🎛️ REQUIREMENTS:")
                for req_type, values in result['primary_intent'].specific_requirements.items():
                    print(f"   • {req_type}: {', '.join(values)}")
            
            print("\n🌐 API CALLS TO MAKE:")
            for api_call in result['api_calls_needed']:
                params_str = ', '.join([f"{k}={v}" for k, v in api_call['params'].items() if v is not None])
                print(f"   • {api_call['method']} {api_call['endpoint']}?{params_str}")
            
            print("\n🤖 ENHANCED AI PROMPT:")
            prompt = integrator.generate_response_prompt(result)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            
        else:
            print("ℹ️ No location intent detected - standard response")
        
        print("=" * 60)


# Integration into your main chat system
def integrate_into_main_chat_system():
    """
    This is how you would integrate it into your main.py chat endpoint
    """
    
    print("\n" + "=" * 80)
    print("INTEGRATION CODE FOR YOUR MAIN.PY")
    print("=" * 80)
    
    integration_code = '''
# Add this import at the top of your main.py
from location_intent_detector import LocationIntentDetector

# Add this to your FastAPI app initialization
location_detector = LocationIntentDetector()

# Modify your chat_with_ai function like this:
@app.post("/ai", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest, user_request: Request):
    """Modified to include location intent detection"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        user_ip = user_request.client.host if user_request.client else None
        user_message = sanitize_user_input(request.message)
        
        # Process location context
        location_info = None
        if request.location_context and request.location_context.has_location:
            location_info = {
                'latitude': request.location_context.latitude,
                'longitude': request.location_context.longitude,
                'district': request.location_context.district,
                'nearby_pois': request.location_context.nearby_pois or [],
                'accuracy': request.location_context.accuracy
            }
        
        # NEW: Detect location intents
        detected_intents = location_detector.detect_intent(user_message, location_info)
        
        if detected_intents:
            primary_intent = detected_intents[0]
            response_config = location_detector.generate_location_response(primary_intent, location_info)
            
            print(f"🎯 Location intent detected: {primary_intent.intent_type.value} (confidence: {primary_intent.confidence:.2f})")
            
            # If high confidence location intent, use location-specific response
            if primary_intent.confidence >= 0.5:
                if primary_intent.intent_type == LocationIntentType.RESTAURANTS:
                    # Make API call to your restaurant service
                    restaurants = await get_nearby_restaurants(location_info, response_config['search_params'])
                    response = format_restaurant_recommendations(restaurants, primary_intent)
                    
                elif primary_intent.intent_type == LocationIntentType.MUSEUMS:
                    # Make API call to your museum service  
                    museums = await get_nearby_museums(location_info, response_config['search_params'])
                    response = format_museum_recommendations(museums, primary_intent)
                    
                else:
                    # Use enhanced context for general AI response
                    enhanced_context = build_location_context(primary_intent, location_info)
                    ai_result = await get_istanbul_ai_response_with_quality(
                        user_message, session_id, user_ip, 
                        location_context=location_info,
                        intent_context=enhanced_context
                    )
                    response = ai_result['response'] if ai_result['success'] else fallback_response
                
                return ChatResponse(
                    response=response,
                    session_id=session_id,
                    success=True,
                    system_type="location_intent_aware"
                )
        
        # Fall back to standard AI response for non-location queries
        ai_result = await get_istanbul_ai_response_with_quality(user_message, session_id, user_ip, location_context=location_info)
        
        if ai_result and ai_result.get('success'):
            return ChatResponse(
                response=ai_result['response'],
                session_id=session_id,
                success=True,
                system_type=ai_result.get('system_type', 'ultra_specialized_istanbul_ai')
            )
            
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return ChatResponse(
            response="I'm sorry, I encountered an issue. Please try again.",
            session_id=request.session_id or str(uuid.uuid4()),
            success=False,
            system_type="error_fallback"
        )
    '''
    
    print(integration_code)


if __name__ == "__main__":
    demonstrate_integration()
    integrate_into_main_chat_system()
