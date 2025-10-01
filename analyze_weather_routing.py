#!/usr/bin/env python3
"""
Analyze why some queries trigger different response types
"""

import requests
import json

def analyze_response_routing():
    """Analyze why different queries get different response types"""
    
    backend_url = "http://localhost:8000"
    
    # Test queries that triggered different response types
    test_cases = [
        # These got ai_response (good)
        ("What should I wear today in Istanbul?", "Expected: ai_response"),
        ("Is it good weather for walking around Istanbul?", "Expected: ai_response"),
        ("What's the weather like in Istanbul?", "Expected: ai_response"),
        
        # These got default/transportation_info (not ideal)
        ("Do I need a jacket?", "Got: default - should be ai_response"),
        ("Should I bring an umbrella?", "Got: default - should be ai_response"), 
        ("Should I go to the Grand Bazaar or outdoor markets?", "Got: transportation_info - should be ai_response"),
        ("Is it good weather for the ferry to Princes' Islands?", "Got: transportation_info - should be ai_response"),
        
        # Test edge cases
        ("Weather in Istanbul", "Short query test"),
        ("Istanbul weather today", "Short query test"),
        ("Umbrella needed?", "Very short query"),
        ("Jacket or no jacket in Istanbul today?", "Specific clothing query"),
    ]
    
    print("🔍 RESPONSE ROUTING ANALYSIS")
    print("=" * 60)
    
    for i, (query, note) in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: '{query}'")
        print(f"📋 Note: {note}")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{backend_url}/api/chat",
                json={"message": query},
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                response_type = data.get('type', 'unknown')
                response_text = data.get('response', '')
                
                print(f"📊 Response Type: {response_type}")
                
                # Analyze keywords in the query
                query_lower = query.lower()
                
                # Weather keywords from the backend
                weather_keywords = ['weather', 'temperature', 'rain', 'sunny', 'cold', 'hot', 'today', 'wear', 'clothing', 'outdoor', 'indoor', 'cruise', 'activities', 'climate', 'warm', 'cool', 'cloudy', 'windy', 'humidity', 'forecast', 'season']
                general_keywords = ['istanbul', 'turkey', 'visit', 'tourist', 'attraction', 'recommendation', 'what should i', 'what to do', 'where to go']
                
                weather_matches = [kw for kw in weather_keywords if kw in query_lower]
                general_matches = [kw for kw in general_keywords if kw in query_lower]
                
                print(f"🌤️  Weather keywords found: {weather_matches}")
                print(f"🏛️  General keywords found: {general_matches}")
                print(f"🎯 Should trigger AI: {bool(weather_matches or general_matches)}")
                
                # Check if it's categorized properly
                if response_type == 'ai_response':
                    print("✅ CORRECT: Routed to AI system")
                elif response_type in ['default', 'transportation_info', 'museum_list']:
                    print(f"⚠️  ISSUE: Routed to {response_type} instead of AI")
                    
                # Check content quality
                response_lower = response_text.lower()
                has_weather_advice = any(word in response_lower for word in ['weather', 'temperature', 'wear', 'layer', 'jacket', 'umbrella', 'outdoor', 'indoor'])
                
                print(f"🎯 Contains weather-related advice: {'✅' if has_weather_advice else '❌'}")
                
                # Show first sentence to understand the response style
                first_sentence = response_text.split('.')[0] if '.' in response_text else response_text[:100]
                print(f"📄 First sentence: {first_sentence}...")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
    
    print(f"\n🔍 KEYWORD MATCHING ANALYSIS")
    print("=" * 60)
    
    # Test keyword edge cases
    keyword_tests = [
        ("jacket", "Single clothing word"),
        ("umbrella", "Single weather accessory"),
        ("weather Istanbul", "Weather + location"),
        ("Istanbul weather", "Location + weather"),
        ("wear today", "Clothing + time"),
        ("outdoor activities", "Activity + weather context"),
    ]
    
    for query, description in keyword_tests:
        query_lower = query.lower()
        weather_keywords = ['weather', 'temperature', 'rain', 'sunny', 'cold', 'hot', 'today', 'wear', 'clothing', 'outdoor', 'indoor', 'cruise', 'activities', 'climate', 'warm', 'cool', 'cloudy', 'windy', 'humidity', 'forecast', 'season']
        general_keywords = ['istanbul', 'turkey', 'visit', 'tourist', 'attraction', 'recommendation', 'what should i', 'what to do', 'where to go']
        
        weather_matches = [kw for kw in weather_keywords if kw in query_lower]
        general_matches = [kw for kw in general_keywords if kw in query_lower]
        
        should_trigger = bool(weather_matches or general_matches)
        
        print(f"'{query}' ({description})")
        print(f"  Weather matches: {weather_matches}")
        print(f"  General matches: {general_matches}")
        print(f"  Should trigger AI: {should_trigger}")
        print()

def test_weather_context_delivery():
    """Test if weather context is actually being delivered to AI"""
    
    print(f"\n🌤️ TESTING WEATHER CONTEXT DELIVERY")
    print("=" * 60)
    
    # Test a query we know should work
    query = "What should I wear today in Istanbul?"
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={"message": query},
            timeout=20
        )
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            
            print(f"📝 Query: {query}")
            print(f"📊 Response Type: {data.get('type')}")
            print(f"📄 Response Length: {len(response_text)} characters")
            print()
            
            # Look for specific weather information that should be in the prompt
            weather_clues = {
                "Temperature references": ['20°c', '20 degrees', 'temperature', 'warm', 'cool', 'mild'],
                "Weather conditions": ['sunny', 'cloudy', 'partly cloudy', 'overcast', 'clear'],
                "Rain status": ['rain', 'no rain', 'dry', 'precipitation'],
                "Weather advice": ['layer', 'light jacket', 'sweater', 'comfortable', 'appropriate for weather']
            }
            
            print("🔍 Analyzing response for weather context:")
            for category, keywords in weather_clues.items():
                found = [kw for kw in keywords if kw in response_text.lower()]
                print(f"  {category}: {found if found else 'None found'}")
            
            print(f"\n📄 Full Response:")
            print("-" * 40)
            print(response_text)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    analyze_response_routing()
    test_weather_context_delivery()
