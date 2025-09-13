#!/usr/bin/env python3
"""
Test script to verify the chatbot improvements:
1. Emoji removal from responses
2. Weather information inclusion for place recommendations
3. Cost/pricing information removal
"""

import requests
import re
import json
import time
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
AI_ENDPOINT = f"{BASE_URL}/ai"

def test_ai_response(query: str, description: str = "") -> Dict[str, Any]:
    """Send a query to the AI endpoint and return the response."""
    print(f"\n{'='*50}")
    print(f"TEST: {description or query}")
    print(f"{'='*50}")
    print(f"Query: {query}")
    
    payload = {
        "query": query,
        "session_id": f"test_session_{int(time.time())}"
    }
    
    try:
        response = requests.post(AI_ENDPOINT, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            message = data.get('message', '')
            print(f"Response: {message}")
            
            # Analysis
            analysis = []
            
            # Check for emojis
            emoji_found = any(ord(char) > 127 for char in message if ord(char) in [
                *range(0x1F600, 0x1F64F),  # emoticons
                *range(0x1F300, 0x1F5FF),  # symbols & pictographs
                *range(0x1F680, 0x1F6FF),  # transport & map symbols
                *range(0x1F1E0, 0x1F1FF),  # flags
                *range(0x2702, 0x27B0),    # dingbats
                *range(0x24C2, 0x1F251),
                *range(0x1F900, 0x1F9FF),  # supplemental symbols
                *range(0x1FA70, 0x1FAFF),  # extended symbols
                *range(0x2600, 0x26FF),    # miscellaneous symbols
                *range(0x2700, 0x27BF)     # dingbats
            ])
            
            if emoji_found:
                analysis.append("❌ EMOJIS FOUND - emojis still present in response")
            else:
                analysis.append("✅ NO EMOJIS - response is emoji-free")
            
            # Check for weather information (for place/recommendation queries)
            weather_keywords = ['weather', 'temperature', 'current', 'rain', 'sunny', 'cloudy', 'degrees', 'today', 'feels like']
            daily_weather_patterns = [
                'today\'s istanbul weather:', 'today\'s weather', 'feels like', '°c', 
                'today.*\\d+°c', 'current.*weather.*istanbul'
            ]
            
            has_weather = any(keyword in message.lower() for keyword in weather_keywords)
            
            # Enhanced daily weather detection
            has_daily_weather = False
            message_lower = message.lower()
            
            # Check for our specific daily weather format patterns
            for pattern in daily_weather_patterns:
                if re.search(pattern, message_lower):
                    has_daily_weather = True
                    break
            
            # Also check for temperature with degrees pattern
            if re.search(r'\d+°c.*feels.*like.*\d+°c', message_lower):
                has_daily_weather = True
            
            # Count weather indicators for backup detection
            weather_count = sum(1 for keyword in weather_keywords if keyword in message_lower)
            if weather_count >= 3:
                has_daily_weather = True
            
            if 'recommend' in query.lower() or 'place' in query.lower() or 'visit' in query.lower() or 'do' in query.lower():
                if has_daily_weather:
                    analysis.append("✅ DAILY WEATHER - today's weather information included in recommendation")
                elif has_weather:
                    analysis.append("⚠️  GENERAL WEATHER - weather mentioned but not today's specific conditions")
                else:
                    analysis.append("⚠️  NO WEATHER - weather information missing from recommendation")
            
            # Check for cost/pricing information
            cost_patterns = [
                'tl', 'lira', 'cost', 'price', 'expensive', 'cheap', 'budget', 
                'admission', 'fee', 'ticket', 'free entry', 'affordable'
            ]
            has_costs = any(pattern in message.lower() for pattern in cost_patterns)
            
            if has_costs:
                analysis.append("❌ COSTS FOUND - pricing/cost information still present")
            else:
                analysis.append("✅ NO COSTS - response is cost-free")
            
            # Check response quality
            if len(message) > 50:
                analysis.append("✅ GOOD LENGTH - substantial response")
            else:
                analysis.append("⚠️  SHORT RESPONSE - response might be too brief")
            
            # Check Istanbul focus
            istanbul_keywords = ['istanbul', 'turkey', 'turkish', 'bosphorus', 'galata', 'sultanahmet', 'beyoğlu', 'kadıköy']
            has_istanbul = any(keyword in message.lower() for keyword in istanbul_keywords)
            
            if has_istanbul:
                analysis.append("✅ ISTANBUL FOCUS - response mentions Istanbul/Turkey")
            else:
                analysis.append("⚠️  NO ISTANBUL - response lacks Istanbul context")
            
            print(f"\nAnalysis:")
            for item in analysis:
                print(f"  {item}")
            
            return {
                'success': True,
                'message': message,
                'analysis': analysis,
                'has_emojis': emoji_found,
                'has_daily_weather': has_daily_weather,
                'has_costs': has_costs,
                'has_istanbul': has_istanbul
            }
            
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            print(f"Error: {error_msg}")
            return {'success': False, 'error': error_msg}
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {str(e)}"
        print(f"Error: {error_msg}")
        return {'success': False, 'error': error_msg}

def main():
    """Run comprehensive tests for chatbot improvements."""
    print("🧪 CHATBOT IMPROVEMENTS TEST SUITE")
    print("Testing: Emoji removal, Weather integration, Cost removal")
    print("\nChecking if server is running...")
    
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server health check failed")
            return
    except:
        print("❌ Server is not accessible. Please start the backend server first.")
        return
    
    # Test cases designed to verify all improvements
    test_cases = [
        {
            'query': 'Hello, can you recommend some places to visit in Istanbul?',
            'description': 'Place recommendations (should include weather, no emojis, no costs)'
        },
        {
            'query': 'What are the best restaurants in Beyoğlu?',
            'description': 'Restaurant recommendations (should include weather, no emojis, no costs)'
        },
        {
            'query': 'Tell me about Hagia Sophia',
            'description': 'Attraction information (no emojis, no costs)'
        },
        {
            'query': 'How do I get from Taksim to Sultanahmet?',
            'description': 'Transportation (no emojis, no costs)'
        },
        {
            'query': 'What should I do in Istanbul today?',
            'description': 'Activity recommendations (should include weather, no emojis, no costs)'
        },
        {
            'query': 'Where can I go shopping in Istanbul?',
            'description': 'Shopping recommendations (no emojis, no costs)'
        },
        {
            'query': 'Tell me about Istanbul\'s history',
            'description': 'Historical information (no emojis)'
        },
        {
            'query': 'What\'s the weather like in Istanbul?',
            'description': 'Weather query (should include current weather)'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📍 Test {i}/{len(test_cases)}")
        result = test_ai_response(test_case['query'], test_case['description'])
        results.append(result)
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    successful_tests = [r for r in results if r.get('success', False)]
    total_tests = len(results)
    success_rate = len(successful_tests) / total_tests * 100 if total_tests > 0 else 0
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {len(successful_tests)} ({success_rate:.1f}%)")
    print(f"Failed: {total_tests - len(successful_tests)}")
    
    if successful_tests:
        # Analyze improvement metrics
        emoji_free_count = sum(1 for r in successful_tests if not r.get('has_emojis', True))
        cost_free_count = sum(1 for r in successful_tests if not r.get('has_costs', True))
        weather_count = sum(1 for r in successful_tests if r.get('has_weather', False))
        istanbul_count = sum(1 for r in successful_tests if r.get('has_istanbul', False))
        
        print(f"\n🎯 IMPROVEMENT METRICS:")
        print(f"  Emoji-free responses: {emoji_free_count}/{len(successful_tests)} ({emoji_free_count/len(successful_tests)*100:.1f}%)")
        print(f"  Cost-free responses: {cost_free_count}/{len(successful_tests)} ({cost_free_count/len(successful_tests)*100:.1f}%)")
        print(f"  Weather included: {weather_count}/{len(successful_tests)} ({weather_count/len(successful_tests)*100:.1f}%)")
        print(f"  Istanbul-focused: {istanbul_count}/{len(successful_tests)} ({istanbul_count/len(successful_tests)*100:.1f}%)")
        
        # Overall assessment
        if emoji_free_count == len(successful_tests) and cost_free_count == len(successful_tests):
            print(f"\n🎉 SUCCESS: All improvements implemented correctly!")
            if weather_count > 0:
                print(f"✅ Weather integration is working")
            if istanbul_count >= len(successful_tests) * 0.8:
                print(f"✅ Strong Istanbul focus maintained")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: Some improvements need attention")
            if emoji_free_count < len(successful_tests):
                print(f"❌ Emojis still found in some responses")
            if cost_free_count < len(successful_tests):
                print(f"❌ Cost information still found in some responses")

if __name__ == "__main__":
    main()
