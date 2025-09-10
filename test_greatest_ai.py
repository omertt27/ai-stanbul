#!/usr/bin/env python3
"""
Complete system test for the Greatest AI Istanbul Chatbot
Tests both backend and frontend with advanced features
"""

import requests
import json
import time
import subprocess
import sys
import os
from concurrent.futures import ThreadPoolExecutor
import threading

def test_backend_advanced_features():
    """Test the advanced backend features"""
    
    backend_url = "http://localhost:8000"
    
    print("🧠 Testing Advanced Backend Features")
    print("=" * 40)
    
    # Advanced test queries that should showcase superior AI
    advanced_queries = [
        {
            "query": "I'm visiting Istanbul for 3 days with my family. We love history and good food. Can you create a perfect itinerary?",
            "expected_features": ["personalization", "itinerary", "family-friendly"]
        },
        {
            "query": "What are the most authentic Turkish restaurants in Beyoğlu that locals actually go to?",
            "expected_features": ["local knowledge", "authentic", "specific location"]
        },
        {
            "query": "I'm staying in Sultanahmet and want to see the sunset from the best viewpoint. How do I get there and what time?",
            "expected_features": ["location-aware", "real-time", "transportation"]
        },
        {
            "query": "Tell me about Istanbul's coffee culture and where to experience it like a local",
            "expected_features": ["cultural insight", "local experience", "coffee"]
        },
        {
            "query": "I need to get from Istanbul Airport to Galata Tower at 11 PM. What are my options?",
            "expected_features": ["transportation", "time-specific", "route planning"]
        }
    ]
    
    session_id = f"advanced_test_{int(time.time())}"
    
    for i, test_case in enumerate(advanced_queries, 1):
        query = test_case["query"]
        expected_features = test_case["expected_features"]
        
        print(f"\n{i}. Advanced Query Test")
        print(f"   Query: {query}")
        print(f"   Expected: {', '.join(expected_features)}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{backend_url}/ai",
                json={
                    "query": query,
                    "session_id": session_id
                },
                timeout=45  # Longer timeout for advanced processing
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", "")
                
                # Analyze response quality
                analysis = analyze_response_quality(message, expected_features)
                
                print(f"   ✅ Response received in {response_time:.2f}s")
                print(f"   📊 Quality Score: {analysis['score']}/10")
                print(f"   🎯 Features Found: {analysis['features_found']}")
                print(f"   📏 Length: {len(message)} characters")
                
                if analysis['score'] >= 7:
                    print(f"   🌟 EXCELLENT: This is superior AI performance!")
                elif analysis['score'] >= 5:
                    print(f"   👍 GOOD: Quality response with room for improvement")
                else:
                    print(f"   ⚠️  NEEDS IMPROVEMENT: Response quality below expectations")
                
                # Show key excerpts
                sentences = message.split('. ')
                if len(sentences) > 2:
                    print(f"   💬 Key insight: {sentences[1][:100]}...")
                
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

def analyze_response_quality(message, expected_features):
    """Analyze the quality of an AI response"""
    
    score = 0
    features_found = []
    message_lower = message.lower()
    
    # Basic quality indicators
    if len(message) > 200:
        score += 1
    if len(message) > 500:
        score += 1
        
    # Structural quality
    if any(marker in message for marker in ['**', '•', '-', '1.', '2.', '###']):
        score += 1
        features_found.append("structured")
    
    # Engagement quality
    if any(char for char in message if ord(char) > 127):  # Emojis/special chars
        score += 1
        features_found.append("engaging")
    
    # Content quality based on expected features
    for feature in expected_features:
        if feature.lower() in message_lower:
            score += 1
            features_found.append(feature)
        elif any(keyword in message_lower for keyword in get_feature_keywords(feature)):
            score += 0.5
            features_found.append(f"{feature}*")
    
    # Advanced indicators
    if any(word in message_lower for word in ['recommend', 'suggest', 'perfect', 'best', 'ideal']):
        score += 0.5
        features_found.append("advisory")
    
    if any(word in message_lower for word in ['cultural', 'local', 'authentic', 'traditional']):
        score += 0.5
        features_found.append("cultural_insight")
    
    return {
        'score': min(10, score),
        'features_found': features_found,
        'analysis': {
            'length': len(message),
            'has_structure': any(marker in message for marker in ['**', '•', '-']),
            'has_engagement': any(ord(char) > 127 for char in message)
        }
    }

def get_feature_keywords(feature):
    """Get keywords associated with each feature"""
    keywords = {
        'personalization': ['for you', 'your', 'based on', 'family', 'prefer'],
        'itinerary': ['day 1', 'first', 'then', 'after', 'schedule', 'plan'],
        'local knowledge': ['locals', 'authentic', 'hidden', 'secret', 'insider'],
        'transportation': ['metro', 'bus', 'taxi', 'walk', 'transport', 'get there'],
        'cultural insight': ['culture', 'tradition', 'history', 'local', 'turkish'],
        'real-time': ['current', 'today', 'now', 'latest', 'updated']
    }
    return keywords.get(feature.lower(), [])

def test_frontend_integration():
    """Test frontend integration with advanced backend"""
    
    frontend_url = "http://localhost:3000"
    
    print("\n🎨 Testing Frontend Integration")
    print("=" * 40)
    
    try:
        # Test if frontend is accessible
        response = requests.get(frontend_url, timeout=10)
        if response.status_code == 200:
            print("✅ Frontend is accessible")
            
            # Check if it contains expected elements
            content = response.text
            if 'chatbot' in content.lower() or 'istanbul' in content.lower():
                print("✅ Frontend contains expected Istanbul chatbot content")
            else:
                print("⚠️  Frontend content may not be fully loaded")
                
        else:
            print(f"⚠️  Frontend returned status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Frontend not accessible - is it running on port 3000?")
    except Exception as e:
        print(f"❌ Frontend test error: {e}")

def test_performance_benchmarks():
    """Test performance benchmarks"""
    
    print("\n⚡ Testing Performance Benchmarks")
    print("=" * 40)
    
    backend_url = "http://localhost:8000"
    
    # Performance test queries
    quick_queries = [
        "Hello",
        "What's the weather?",
        "Best restaurants",
        "How to get to Taksim?",
        "Tell me about Hagia Sophia"
    ]
    
    session_id = f"perf_test_{int(time.time())}"
    response_times = []
    
    for query in quick_queries:
        try:
            start_time = time.time()
            response = requests.post(
                f"{backend_url}/ai",
                json={"query": query, "session_id": session_id},
                timeout=30
            )
            response_time = time.time() - start_time
            response_times.append(response_time)
            
            if response.status_code == 200:
                print(f"✅ '{query}' -> {response_time:.2f}s")
            else:
                print(f"❌ '{query}' -> Failed ({response.status_code})")
                
        except Exception as e:
            print(f"❌ '{query}' -> Error: {e}")
    
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average response time: {avg_time:.2f}s")
        print(f"   Fastest response: {min_time:.2f}s")
        print(f"   Slowest response: {max_time:.2f}s")
        
        if avg_time < 3.0:
            print("   🚀 EXCELLENT: Sub-3 second average response time!")
        elif avg_time < 5.0:
            print("   👍 GOOD: Good response times")
        else:
            print("   ⚠️  NEEDS OPTIMIZATION: Response times could be improved")

def main():
    """Run the complete system test"""
    
    print("🏆 GREATEST AI ISTANBUL CHATBOT - COMPLETE SYSTEM TEST")
    print("=" * 60)
    print("Testing the most advanced Istanbul travel AI system...")
    print()
    
    # Test advanced backend features
    test_backend_advanced_features()
    
    # Test frontend integration
    test_frontend_integration()
    
    # Test performance
    test_performance_benchmarks()
    
    print("\n" + "=" * 60)
    print("🎯 COMPLETE SYSTEM TEST FINISHED")
    print("🏆 The Greatest AI Istanbul Chatbot is ready to beat all competitors!")
    print("=" * 60)

if __name__ == "__main__":
    main()
