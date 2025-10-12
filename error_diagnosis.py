#!/usr/bin/env python3
"""
Error Diagnosis for Complex Queries
===================================

This script systematically tests the queries that were causing 500 errors
to identify the root cause and implement fixes.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_failing_query(query, user_id, test_name):
    """Test a specific query that was failing"""
    print(f"\n🔍 Testing: {test_name}")
    print(f"Query: {query}")
    print("-" * 50)
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/chat", 
                               json={"message": query, "user_id": user_id},
                               timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS")
            print(f"Response Length: {len(data['response'])} chars")
            print(f"Response Preview: {data['response'][:200]}...")
            return True
        else:
            print(f"❌ FAILED - HTTP {response.status_code}")
            print(f"Response Text: {response.text}")
            
            # Try to parse error details
            try:
                error_data = response.json()
                print(f"Error Details: {json.dumps(error_data, indent=2)}")
            except:
                print("Could not parse error response as JSON")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT - Query took too long")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Server may be down")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_weather_queries():
    """Test weather-related queries that were failing"""
    print("🌤️ WEATHER QUERIES DIAGNOSIS")
    print("=" * 60)
    
    weather_queries = [
        ("What's the weather like today?", "weather_test_1", "Simple Weather Query"),
        ("How's the weather?", "weather_test_2", "Basic Weather Query"),
        ("Tell me about today's weather in Istanbul", "weather_test_3", "Detailed Weather Query"),
        ("Should I bring an umbrella today?", "weather_test_4", "Weather-Based Recommendation"),
        ("Is it good weather for sightseeing?", "weather_test_5", "Weather Activity Query")
    ]
    
    results = []
    for query, user_id, test_name in weather_queries:
        success = test_failing_query(query, user_id, test_name)
        results.append((test_name, success))
        time.sleep(1)  # Prevent rate limiting
    
    return results

def test_restaurant_queries():
    """Test restaurant queries that were failing"""
    print("\n🍽️ RESTAURANT QUERIES DIAGNOSIS")
    print("=" * 60)
    
    restaurant_queries = [
        ("Recommend restaurants", "restaurant_test_1", "Simple Restaurant Request"),
        ("I want authentic Turkish breakfast", "restaurant_test_2", "Specific Food Request"),
        ("Show me good restaurants near Sultanahmet", "restaurant_test_3", "Location-Based Restaurant Query"),
        ("What are the best local restaurants?", "restaurant_test_4", "Local Restaurant Query"),
        ("I'm looking for vegetarian restaurants", "restaurant_test_5", "Dietary Restaurant Query")
    ]
    
    results = []
    for query, user_id, test_name in restaurant_queries:
        success = test_failing_query(query, user_id, test_name)
        results.append((test_name, success))
        time.sleep(1)
    
    return results

def test_complex_queries():
    """Test complex multi-intent queries"""
    print("\n🧠 COMPLEX QUERIES DIAGNOSIS")
    print("=" * 60)
    
    complex_queries = [
        ("I'm feeling curious - any hidden gems to explore?", "complex_test_1", "Mood + Hidden Gems Query"),
        ("What local activities would you recommend for tonight?", "complex_test_2", "Time + Activity Query"),
        ("I want to explore Istanbul like a local today", "complex_test_3", "Cultural Immersion Query"),
        ("Show me hidden gems that locals love, not touristy places", "complex_test_4", "Anti-Tourist Query"),
        ("I'm traveling with family, what neighborhoods are best for us?", "complex_test_5", "Family + Neighborhood Query")
    ]
    
    results = []
    for query, user_id, test_name in complex_queries:
        success = test_failing_query(query, user_id, test_name)
        results.append((test_name, success))
        time.sleep(1)
    
    return results

def test_server_health():
    """Test server health and system status"""
    print("\n🏥 SERVER HEALTH CHECK")
    print("=" * 60)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server Health: OK")
            print(f"Services Status: {data.get('services', {})}")
            return True
        else:
            print(f"❌ Server Health: FAILED ({response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Server Health Check Exception: {e}")
        return False

def analyze_results(weather_results, restaurant_results, complex_results):
    """Analyze test results to identify patterns"""
    print("\n📊 ERROR PATTERN ANALYSIS")
    print("=" * 60)
    
    all_results = weather_results + restaurant_results + complex_results
    total_tests = len(all_results)
    successful_tests = sum(1 for _, success in all_results if success)
    failed_tests = total_tests - successful_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
    
    # Categorize failures
    print("\n🔍 FAILURE ANALYSIS:")
    failed_weather = sum(1 for _, success in weather_results if not success)
    failed_restaurant = sum(1 for _, success in restaurant_results if not success)
    failed_complex = sum(1 for _, success in complex_results if not success)
    
    print(f"Weather Query Failures: {failed_weather}/{len(weather_results)}")
    print(f"Restaurant Query Failures: {failed_restaurant}/{len(restaurant_results)}")
    print(f"Complex Query Failures: {failed_complex}/{len(complex_results)}")
    
    # Identify patterns
    patterns = []
    if failed_weather > len(weather_results) * 0.5:
        patterns.append("Weather service integration issues")
    if failed_restaurant > len(restaurant_results) * 0.5:
        patterns.append("Restaurant recommendation system issues")
    if failed_complex > len(complex_results) * 0.5:
        patterns.append("Complex query processing issues")
    
    if patterns:
        print(f"\n🎯 IDENTIFIED PATTERNS:")
        for pattern in patterns:
            print(f"   • {pattern}")
    else:
        print(f"\n✅ No clear failure patterns identified")
    
    return {
        'total': total_tests,
        'successful': successful_tests,
        'failed': failed_tests,
        'success_rate': (successful_tests/total_tests)*100,
        'patterns': patterns
    }

def main():
    """Run comprehensive error diagnosis"""
    print("🔧 COMPLEX QUERY ERROR DIAGNOSIS")
    print("=" * 70)
    print(f"Diagnosis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Check server health first
    if not test_server_health():
        print("❌ Server health check failed. Cannot proceed with diagnosis.")
        return
    
    # Run systematic tests
    weather_results = test_weather_queries()
    restaurant_results = test_restaurant_queries()
    complex_results = test_complex_queries()
    
    # Analyze results
    analysis = analyze_results(weather_results, restaurant_results, complex_results)
    
    # Generate recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("=" * 60)
    
    if analysis['success_rate'] > 80:
        print("✅ System is performing well overall")
        print("🔧 Minor optimizations needed for edge cases")
    elif analysis['success_rate'] > 60:
        print("⚠️ Moderate issues detected")
        print("🔧 Specific service components need attention")
    else:
        print("❌ Significant issues detected")
        print("🔧 Major system components need investigation")
    
    # Specific recommendations based on patterns
    for pattern in analysis['patterns']:
        if "Weather service" in pattern:
            print("   → Check weather API configuration and error handling")
        elif "Restaurant recommendation" in pattern:
            print("   → Review restaurant database and recommendation logic")
        elif "Complex query" in pattern:
            print("   → Optimize multi-intent processing and error recovery")
    
    print(f"\n📋 DIAGNOSIS COMPLETE")
    print(f"Success Rate: {analysis['success_rate']:.1f}%")

if __name__ == "__main__":
    main()
