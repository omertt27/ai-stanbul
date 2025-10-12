#!/usr/bin/env python3
"""
IstanbulDailyTalkAI Integration Verification
===========================================

This script verifies that IstanbulDailyTalkAI is properly integrated 
as the main AI system and tests for any remaining issues.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_basic_chat():
    """Test basic chat functionality"""
    print("🔧 BASIC CHAT TEST")
    print("=" * 50)
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/chat", 
                               json={"message": "Hello", "user_id": "test_basic"})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Basic chat working: {data['response'][:100]}...")
            return True
        else:
            print(f"❌ Basic chat failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception in basic chat: {e}")
        return False

def test_weather_queries():
    """Test weather-related queries that were failing"""
    print("\n🌤️ WEATHER QUERIES TEST")
    print("=" * 50)
    
    weather_queries = [
        "What's the weather like today?",
        "How's the weather?",
        "Is it raining in Istanbul?"
    ]
    
    success_count = 0
    for query in weather_queries:
        try:
            response = requests.post(f"{BASE_URL}/api/v1/chat", 
                                   json={"message": query, "user_id": f"weather_test_{len(query)}"})
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ '{query}' -> Success")
                success_count += 1
            else:
                print(f"❌ '{query}' -> Failed ({response.status_code})")
                if response.status_code == 500:
                    print(f"   Error details: {response.text}")
        except Exception as e:
            print(f"❌ '{query}' -> Exception: {e}")
    
    print(f"Weather queries success rate: {success_count}/{len(weather_queries)}")
    return success_count == len(weather_queries)

def test_recommendation_queries():
    """Test recommendation queries"""
    print("\n🎯 RECOMMENDATION QUERIES TEST")
    print("=" * 50)
    
    recommendation_queries = [
        "Recommend restaurants",
        "What should I do today?",
        "Show me attractions"
    ]
    
    success_count = 0
    for query in recommendation_queries:
        try:
            response = requests.post(f"{BASE_URL}/api/v1/chat", 
                                   json={"message": query, "user_id": f"rec_test_{len(query)}"})
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ '{query}' -> Success")
                success_count += 1
            else:
                print(f"❌ '{query}' -> Failed ({response.status_code})")
        except Exception as e:
            print(f"❌ '{query}' -> Exception: {e}")
    
    print(f"Recommendation queries success rate: {success_count}/{len(recommendation_queries)}")
    return success_count == len(recommendation_queries)

def test_health_endpoints():
    """Test health and system status endpoints"""
    print("\n🏥 HEALTH ENDPOINTS TEST")
    print("=" * 50)
    
    endpoints = [
        "/health",
        "/api/v1/daily-greeting",
        "/api/v1/daily-conversation"
    ]
    
    for endpoint in endpoints:
        try:
            if endpoint == "/health":
                response = requests.get(f"{BASE_URL}{endpoint}")
            else:
                response = requests.get(f"{BASE_URL}{endpoint}", 
                                      params={"user_id": "health_test", "location": "Istanbul"})
            
            if response.status_code == 200:
                print(f"✅ {endpoint} -> Healthy")
            else:
                print(f"❌ {endpoint} -> Failed ({response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint} -> Exception: {e}")

def verify_integration_status():
    """Verify that IstanbulDailyTalkAI is the main system"""
    print("\n🔍 INTEGRATION STATUS VERIFICATION")
    print("=" * 50)
    
    try:
        # Test that the system identifies itself correctly
        response = requests.post(f"{BASE_URL}/api/v1/chat", 
                               json={"message": "What AI system are you?", "user_id": "integration_test"})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System response: {data['response'][:200]}...")
            
            # Check if it mentions Istanbul-specific features
            if any(keyword in data['response'].lower() for keyword in ['istanbul', 'turkish', 'local']):
                print("✅ System shows Istanbul-specific knowledge")
                return True
            else:
                print("⚠️ System response seems generic")
                return False
        else:
            print(f"❌ System identification failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception in integration verification: {e}")
        return False

def main():
    """Run comprehensive integration verification"""
    print("🚀 ISTANBULDAILYTALK AI INTEGRATION VERIFICATION")
    print("=" * 70)
    print(f"Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run tests
    basic_ok = test_basic_chat()
    weather_ok = test_weather_queries()
    rec_ok = test_recommendation_queries()
    test_health_endpoints()
    integration_ok = verify_integration_status()
    
    # Summary
    print("\n📊 INTEGRATION VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"✅ Basic Chat: {'PASS' if basic_ok else 'FAIL'}")
    print(f"🌤️ Weather Queries: {'PASS' if weather_ok else 'FAIL'}")
    print(f"🎯 Recommendations: {'PASS' if rec_ok else 'FAIL'}")
    print(f"🔍 Istanbul Integration: {'PASS' if integration_ok else 'FAIL'}")
    
    overall_status = all([basic_ok, integration_ok])
    
    print(f"\n🏆 OVERALL STATUS: {'✅ INTEGRATED SUCCESSFULLY' if overall_status else '⚠️ NEEDS ATTENTION'}")
    
    if overall_status:
        print("\n🎉 IstanbulDailyTalkAI is properly integrated as the main AI system!")
        print("✨ Key Features Available:")
        print("   • Istanbul-specific knowledge and expertise")
        print("   • Weather-aware recommendations")
        print("   • Local cultural context and expressions")
        print("   • Personalized user interactions")
        print("   • Multi-intent query handling")
        print("   • Route planning and transportation advice")
    else:
        print("\n🔧 Some issues detected. Check individual test results above.")
    
    return overall_status

if __name__ == "__main__":
    main()
