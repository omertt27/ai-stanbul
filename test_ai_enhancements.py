#!/usr/bin/env python3
"""
Test script for AI Istanbul enhanced intelligence features
Tests session management, preference learning, intent recognition, and personalized recommendations
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
AI_ENDPOINT = f"{BASE_URL}/ai"

def test_session_and_preferences():
    """Test session-based conversation and preference learning"""
    print("🧪 Testing Session Management & Preference Learning")
    print("=" * 60)
    
    # Create a session ID for this test
    session_id = "test_session_001"
    
    # Test queries that should teach the system about user preferences
    test_queries = [
        {
            "query": "I love Turkish cuisine, show me restaurants in Kadıköy",
            "expected": "restaurant recommendations",
            "description": "Learning cuisine preference (Turkish) and location preference (Kadıköy)"
        },
        {
            "query": "What about some budget-friendly restaurants?",
            "expected": "budget restaurant suggestions",
            "description": "Learning budget preference (budget-friendly)"
        },
        {
            "query": "I'm interested in historical attractions",
            "expected": "historical sites",
            "description": "Learning interest (historical attractions)"
        },
        {
            "query": "Show me more restaurants in Kadıköy", 
            "expected": "personalized recommendations",
            "description": "Should show personalized results based on learned preferences"
        }
    ]
    
    conversation_history = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Query: {test['query']}")
        
        payload = {
            "query": test["query"],
            "session_id": session_id,
            "conversation_history": conversation_history
        }
        
        try:
            response = requests.post(AI_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "No message returned")
                print(f"✅ Response received ({len(message)} chars)")
                print(f"Preview: {message[:200]}...")
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": test["query"]})
                conversation_history.append({"role": "assistant", "content": message})
                
                # Keep history manageable
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        time.sleep(2)  # Be nice to the API

def test_intent_recognition():
    """Test intelligent intent recognition and entity extraction"""
    print("\n\n🎯 Testing Intent Recognition & Entity Extraction")
    print("=" * 60)
    
    session_id = "test_session_002"
    
    # Test different types of queries to see intent recognition
    intent_tests = [
        {
            "query": "restaraunts in beyoglu",  # Typo + location
            "expected_intent": "restaurant_search",
            "expected_entities": ["beyoglu"]
        },
        {
            "query": "museums near galata tower",
            "expected_intent": "museum_query", 
            "expected_entities": ["galata"]
        },
        {
            "query": "how to go from taksim to kadikoy",
            "expected_intent": "transportation_query",
            "expected_entities": ["taksim", "kadikoy"]
        },
        {
            "query": "nightlife in besiktas",
            "expected_intent": "nightlife_query",
            "expected_entities": ["besiktas"]
        },
        {
            "query": "cheap eats budget food",
            "expected_intent": "restaurant_search",
            "expected_entities": ["budget"]
        }
    ]
    
    for i, test in enumerate(intent_tests, 1):
        print(f"\n--- Intent Test {i} ---")
        print(f"Query: {test['query']}")
        print(f"Expected Intent: {test['expected_intent']}")
        print(f"Expected Entities: {test['expected_entities']}")
        
        payload = {
            "query": test["query"],
            "session_id": session_id
        }
        
        try:
            response = requests.post(AI_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "No message returned")
                print(f"✅ Response type determined and processed")
                print(f"Preview: {message[:150]}...")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            
        time.sleep(1.5)

def test_personalized_recommendations():
    """Test personalized recommendation engine"""
    print("\n\n🌟 Testing Personalized Recommendations")
    print("=" * 60)
    
    session_id = "test_session_003"
    
    # Build up preferences through multiple queries
    preference_building = [
        "I love seafood restaurants",
        "Show me places in Kadıköy", 
        "I prefer mid-range dining options",
        "restaurants in kadikoy",  # This should show personalized results
    ]
    
    conversation_history = []
    
    for i, query in enumerate(preference_building, 1):
        print(f"\n--- Preference Building Step {i} ---")
        print(f"Query: {query}")
        
        payload = {
            "query": query,
            "session_id": session_id,
            "conversation_history": conversation_history
        }
        
        try:
            response = requests.post(AI_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "")
                
                if i == len(preference_building):
                    print("🎯 Final personalized recommendations:")
                    print(f"Full response: {message}")
                    
                    # Look for personalization indicators
                    if any(word in message.lower() for word in ['seafood', 'kadikoy', 'moderate']):
                        print("✅ Personalization appears to be working!")
                    else:
                        print("⚠️ Personalization not clearly evident")
                else:
                    print(f"✅ Step {i} completed, building preferences...")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": query})
                conversation_history.append({"role": "assistant", "content": message})
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            
        time.sleep(2)

def test_follow_up_context():
    """Test context-aware follow-up questions"""
    print("\n\n💭 Testing Follow-up Context Awareness")
    print("=" * 60)
    
    session_id = "test_session_004" 
    conversation_history = []
    
    # Test context-aware conversation
    context_tests = [
        {
            "query": "Show me restaurants in Sultanahmet",
            "description": "Initial restaurant query"
        },
        {
            "query": "What about the budget options?",
            "description": "Follow-up should understand 'budget restaurants in Sultanahmet'"
        },
        {
            "query": "Any nearby attractions?", 
            "description": "Should understand 'attractions near Sultanahmet'"
        },
        {
            "query": "How do I get there from Taksim?",
            "description": "Should understand 'Taksim to Sultanahmet transportation'"
        }
    ]
    
    for i, test in enumerate(context_tests, 1):
        print(f"\n--- Context Test {i}: {test['description']} ---")
        print(f"Query: {test['query']}")
        
        payload = {
            "query": test["query"],
            "session_id": session_id,
            "conversation_history": conversation_history
        }
        
        try:
            response = requests.post(AI_ENDPOINT, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "")
                print(f"✅ Response: {message[:200]}...")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": test["query"]})
                conversation_history.append({"role": "assistant", "content": message})
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            
        time.sleep(2)

def main():
    """Run all AI enhancement tests"""
    print("🚀 AI Istanbul Enhancement Test Suite")
    print("=" * 60)
    print("Testing enhanced AI intelligence features:")
    print("- Session-based conversation memory")
    print("- User preference learning") 
    print("- Intelligent intent recognition")
    print("- Personalized recommendations")
    print("- Context-aware follow-ups")
    print("\nStarting tests...\n")
    
    try:
        # Test basic connectivity first
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code != 200:
            print(f"❌ Server not responding correctly: {response.status_code}")
            return
        print("✅ Server is running")
        
        # Run test suites
        test_session_and_preferences()
        test_intent_recognition()
        test_personalized_recommendations()
        test_follow_up_context()
        
        print("\n\n🎉 Test Suite Completed!")
        print("Check the output above to verify AI enhancement functionality.")
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("Make sure the backend server is running on port 8000")
    except Exception as e:
        print(f"❌ Test suite failed: {e}")

if __name__ == "__main__":
    main()
