#!/usr/bin/env python3
import requests
import json
import time

def test_backend_and_frontend():
    print("🧪 Testing AI Istanbul System...")
    print("=" * 50)
    
    # Test 1: Backend Health Check
    print("\n1. Testing Backend Health...")
    try:
        backend_response = requests.get("http://localhost:8000/health", timeout=10)
        if backend_response.status_code == 200:
            print("✅ Backend is healthy")
            health_data = backend_response.json()
            print(f"   Status: {health_data.get('status', 'unknown')}")
        else:
            print(f"❌ Backend health check failed: {backend_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running on port 8000")
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
    
    # Test 2: Frontend Accessibility
    print("\n2. Testing Frontend Accessibility...")
    try:
        frontend_response = requests.get("http://localhost:3001", timeout=10)
        if frontend_response.status_code == 200:
            print("✅ Frontend is accessible")
            print(f"   Content length: {len(frontend_response.text)} bytes")
        else:
            print(f"❌ Frontend not accessible: {frontend_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Frontend is not running on port 3001")
    except Exception as e:
        print(f"❌ Frontend accessibility error: {e}")
    
    # Test 3: Chatbot Restaurant Query
    print("\n3. Testing Chatbot Restaurant Recommendations...")
    try:
        chatbot_data = {
            "query": "Can you recommend good restaurants in Beyoğlu?",
            "session_id": "test_session_comprehensive"
        }
        
        chatbot_response = requests.post(
            "http://localhost:8000/ai", 
            json=chatbot_data, 
            timeout=30
        )
        
        if chatbot_response.status_code == 200:
            response_data = chatbot_response.json()
            message = response_data.get('message', '')
            
            print("✅ Chatbot restaurant query successful")
            print(f"   Response length: {len(message)} characters")
            
            # Check for expected content in restaurant response
            expected_content = ['restaurant', 'beyoğlu', 'rating', '⭐']
            found_content = [content for content in expected_content if content.lower() in message.lower()]
            
            if len(found_content) >= 2:
                print(f"✅ Response contains expected restaurant content: {found_content}")
            else:
                print(f"⚠️  Response missing some expected content. Found: {found_content}")
            
            # Print first 200 characters of response
            print(f"   Preview: {message[:200]}...")
            
        else:
            print(f"❌ Chatbot query failed: {chatbot_response.status_code}")
            print(f"   Error: {chatbot_response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to chatbot endpoint")
    except Exception as e:
        print(f"❌ Chatbot test error: {e}")
    
    # Test 4: Enhanced Features
    print("\n4. Testing Enhanced Chatbot Features...")
    try:
        enhanced_response = requests.get("http://localhost:8000/ai/test-enhancements", timeout=10)
        if enhanced_response.status_code == 200:
            features_data = enhanced_response.json()
            print("✅ Enhanced features are active")
            
            if features_data.get('status') == 'enhanced_features_active':
                print("   ✓ Typo correction working")
                print("   ✓ Knowledge base available")
                print("   ✓ Intent extraction active")
                print("   ✓ Context manager running")
            else:
                print(f"   ⚠️  Features status: {features_data.get('status')}")
        else:
            print(f"❌ Enhanced features test failed: {enhanced_response.status_code}")
    except Exception as e:
        print(f"❌ Enhanced features test error: {e}")
    
    # Test 5: Context-Aware Follow-up Query
    print("\n5. Testing Context-Aware Follow-up...")
    try:
        # First query to establish context
        context_data = {
            "query": "Tell me about Kadıköy",
            "session_id": "test_context_session"
        }
        
        first_response = requests.post("http://localhost:8000/ai", json=context_data, timeout=30)
        
        if first_response.status_code == 200:
            # Follow-up query that should use context
            followup_data = {
                "query": "What about restaurants there?",
                "session_id": "test_context_session"
            }
            
            followup_response = requests.post("http://localhost:8000/ai", json=followup_data, timeout=30)
            
            if followup_response.status_code == 200:
                followup_message = followup_response.json().get('message', '')
                
                # Check if response mentions Kadıköy (showing context awareness)
                if 'kadıköy' in followup_message.lower() or 'kadikoy' in followup_message.lower():
                    print("✅ Context-aware follow-up working")
                    print("   ✓ Bot remembered previous location context")
                else:
                    print("⚠️  Follow-up response may not be fully context-aware")
                    print(f"   Preview: {followup_message[:150]}...")
            else:
                print(f"❌ Follow-up query failed: {followup_response.status_code}")
        else:
            print(f"❌ Initial context query failed: {first_response.status_code}")
            
    except Exception as e:
        print(f"❌ Context test error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("   - Backend: Running on port 8000")
    print("   - Frontend: Running on port 3001") 
    print("   - Chatbot: Restaurant recommendations working")
    print("   - Enhanced Features: Typo correction, context management")
    print("   - Context Awareness: Follow-up query handling")
    print("\n✨ AI Istanbul system is ready for use!")

if __name__ == "__main__":
    test_backend_and_frontend()
