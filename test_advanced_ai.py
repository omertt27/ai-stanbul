#!/usr/bin/env python3
"""
Test script for the Advanced AI Orchestrator integration
"""

import requests
import json
import time
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_advanced_ai_integration():
    """Test the advanced AI integration in the chatbot"""
    
    backend_url = "http://localhost:8000"
    
    # Test queries to verify advanced AI is working
    test_queries = [
        "What are the best restaurants in Sultanahmet?",
        "Tell me about Istanbul's cultural heritage",
        "How do I get from Taksim to Kadıköy?",
        "What are some hidden gems in Beyoğlu?",
        "Plan a perfect day in Istanbul",
        "What's the weather like in Istanbul today?",
        "Recommend some nightlife spots",
        "Best museums to visit in Istanbul"
    ]
    
    print("🧪 Testing Advanced AI Integration")
    print("=" * 50)
    
    # Test health endpoint first
    try:
        health_response = requests.get(f"{backend_url}/health", timeout=10)
        if health_response.status_code == 200:
            print("✅ Backend is healthy")
        else:
            print(f"⚠️  Backend health check returned: {health_response.status_code}")
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return
    
    # Test advanced AI endpoint
    try:
        advanced_health = requests.get(f"{backend_url}/ai/enhanced/health", timeout=10)
        if advanced_health.status_code == 200:
            data = advanced_health.json()
            print("✅ Enhanced AI features are active:")
            for feature, status in data.get("enhanced_features", {}).items():
                print(f"   - {feature}: {status}")
        else:
            print(f"⚠️  Enhanced AI health check returned: {advanced_health.status_code}")
    except Exception as e:
        print(f"⚠️  Enhanced AI health check failed: {e}")
    
    print("\n🚀 Testing Advanced AI Responses")
    print("-" * 30)
    
    session_id = f"test_session_{int(time.time())}"
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Testing: '{query}'")
        
        try:
            response = requests.post(
                f"{backend_url}/ai",
                json={
                    "query": query,
                    "session_id": session_id
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", "")
                
                # Analyze response quality
                response_length = len(message)
                has_emoji = any(ord(char) > 127 for char in message)
                has_structure = any(marker in message for marker in ['**', '•', '-', '1.', '2.'])
                
                print(f"   ✅ Response received ({response_length} chars)")
                print(f"   📊 Quality indicators:")
                print(f"      - Structured content: {'✓' if has_structure else '✗'}")
                print(f"      - Engaging format: {'✓' if has_emoji else '✗'}")
                print(f"      - Substantial length: {'✓' if response_length > 200 else '✗'}")
                
                # Check for advanced features
                if any(indicator in message.lower() for indicator in [
                    'real-time', 'latest', 'current', 'today', 'updated'
                ]):
                    print("   🚀 Contains real-time information indicators")
                
                if any(indicator in message.lower() for indicator in [
                    'personalized', 'based on your', 'for you specifically'
                ]):
                    print("   🎯 Contains personalization indicators")
                
                # Show preview
                preview = message[:150] + "..." if len(message) > 150 else message
                print(f"   📝 Preview: {preview}")
                
            else:
                print(f"   ❌ Request failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"   ⏰ Request timed out")
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    print("\n🔄 Testing Streaming Endpoint")
    print("-" * 30)
    
    try:
        stream_response = requests.post(
            f"{backend_url}/ai/stream",
            json={
                "user_input": "What's the best way to experience Istanbul culture?",
                "session_id": session_id
            },
            timeout=30,
            stream=True
        )
        
        if stream_response.status_code == 200:
            print("✅ Streaming endpoint responding")
            chunks_received = 0
            for line in stream_response.iter_lines():
                if line:
                    chunks_received += 1
                    if chunks_received <= 3:  # Show first few chunks
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            try:
                                chunk_data = json.loads(decoded_line[6:])
                                print(f"   📦 Chunk {chunks_received}: {str(chunk_data)[:100]}...")
                            except:
                                print(f"   📦 Chunk {chunks_received}: {decoded_line[:100]}...")
            
            print(f"   📊 Total chunks received: {chunks_received}")
        else:
            print(f"❌ Streaming failed: {stream_response.status_code}")
    
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
    
    print("\n🎯 Advanced AI Integration Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_advanced_ai_integration()
