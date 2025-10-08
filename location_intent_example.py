#!/usr/bin/env python3
"""
Example: How to use Location Intent Detection in AI Istanbul
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'load-testing'))

from location_intent_detector import LocationIntentDetector, LocationIntentType

def main():
    # Initialize the detector
    detector = LocationIntentDetector()
    
    # Simulate user location (Sultanahmet area)
    user_location = {
        'latitude': 41.0082,
        'longitude': 28.9784,
        'district': 'Sultanahmet',
        'accuracy': 50
    }
    
    print("=== AI Istanbul Location Intent Detection Demo ===\n")
    
    # Example user queries
    queries = [
        "I'm hungry, any good restaurants nearby?",
        "Show me Turkish museums within walking distance",
        "What vegetarian places are around here?",
        "How do I get to Galata Tower?",
        "Any romantic restaurants with Bosphorus view?",
        "Museums about Ottoman history close by",
        "What's interesting around Taksim?",
        "Coffee shops within 500 meters"
    ]
    
    for query in queries:
        print(f"User Query: '{query}'")
        print("-" * 50)
        
        # Detect intents
        intents = detector.detect_intent(query, user_location)
        
        if intents:
            for i, intent in enumerate(intents, 1):
                print(f"  📍 Intent {i}: {intent.intent_type.value.upper()}")
                print(f"     Confidence: {intent.confidence:.2f}")
                print(f"     Keywords: {', '.join(intent.keywords_matched)}")
                
                if intent.specific_requirements:
                    print(f"     Special Requirements:")
                    for req_type, req_values in intent.specific_requirements.items():
                        print(f"       - {req_type}: {', '.join(req_values)}")
                
                if intent.distance_preference:
                    print(f"     Distance Preference: {intent.distance_preference}")
                
                # Generate response configuration
                response_config = detector.generate_location_response(intent, user_location)
                print(f"     🔧 Search Config:")
                print(f"       - Type: {response_config['search_type']}")
                print(f"       - Radius: {response_config['search_params'].get('radius', 'N/A')}m")
                
                if 'cuisine' in response_config['search_params']:
                    cuisine = response_config['search_params']['cuisine']
                    if cuisine:
                        print(f"       - Cuisine: {', '.join(cuisine)}")
                
                print(f"       - Template: {response_config.get('response_template', 'default')}")
                
                # Show what API call you would make
                print(f"     🌐 Recommended API Call:")
                if intent.intent_type == LocationIntentType.RESTAURANTS:
                    print(f"       GET /api/location/recommendations?type=restaurants&lat={user_location['latitude']}&lng={user_location['longitude']}&radius={response_config['search_params']['radius']}")
                elif intent.intent_type == LocationIntentType.MUSEUMS:
                    print(f"       GET /api/location/recommendations?type=museums&lat={user_location['latitude']}&lng={user_location['longitude']}&radius={response_config['search_params']['radius']}")
                elif intent.intent_type == LocationIntentType.ROUTE_PLANNING:
                    print(f"       GET /api/location/route?from_lat={user_location['latitude']}&from_lng={user_location['longitude']}&to=destination")
                else:
                    print(f"       GET /api/location/recommendations?type=general&lat={user_location['latitude']}&lng={user_location['longitude']}&radius={response_config['search_params']['radius']}")
        else:
            print("  ❌ No location intent detected")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
