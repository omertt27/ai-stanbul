#!/usr/bin/env python3

"""
Test the improved chatbot with various queries
"""

import requests
import json

def test_improved_chatbot():
    """Test various scenarios with the improved chatbot"""
    
    base_url = "http://localhost:8001"
    
    test_scenarios = [
        {
            "name": "Greeting Test",
            "query": "hello",
            "expected": "greeting response"
        },
        {
            "name": "Restaurant with Typo",
            "query": "show me restaurnts in beyoglu",
            "expected": "restaurant recommendations"
        },
        {
            "name": "Follow-up Restaurant Query", 
            "query": "what about kadikoy restaurants",
            "expected": "restaurant recommendations"
        },
        {
            "name": "Simple Location Query",
            "query": "kadikoy",
            "expected": "places in kadikoy"
        },
        {
            "name": "Transportation Query",
            "query": "how to get from kadikoy to beyoglu",
            "expected": "transportation info"
        }
    ]
    
    print("=" * 70)
    print("TESTING IMPROVED CHATBOT")
    print("=" * 70)
    
    session_id = "test_session_123"
    
    for scenario in test_scenarios:
        print(f"\n🧪 {scenario['name']}")
        print(f"Query: '{scenario['query']}'")
        print("-" * 50)
        
        try:
            response = requests.post(
                f"{base_url}/ai",
                json={"query": scenario['query'], "session_id": session_id},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                message = data.get('message', 'No message')
                
                print(f"✅ Status: {response.status_code}")
                print(f"📝 Response Preview: {message[:150]}...")
                
                # Check response quality
                if len(message) > 50 and len(message) < 1000:
                    print("📏 Length: Good (concise but informative)")
                elif len(message) > 1000:
                    print("⚠️  Length: Too long (might overwhelm user)")
                else:
                    print("⚠️  Length: Too short (might not be helpful)")
                
                # Check for conversational elements
                if any(word in message.lower() for word in ['!', '?', 'you', 'your', 'great', 'perfect']):
                    print("💬 Tone: Conversational ✅")
                else:
                    print("📋 Tone: Formal/robotic")
                    
                # Check for actionable content
                if any(word in message.lower() for word in ['try', 'search', 'ask', 'want more', 'tip']):
                    print("🎯 Actionable: Has suggestions ✅")
                else:
                    print("🎯 Actionable: Limited guidance")
                    
            else:
                print(f"❌ Error: Status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*70}")
    print("✅ Test completed! Check results above for chatbot quality.")

if __name__ == "__main__":
    test_improved_chatbot()
