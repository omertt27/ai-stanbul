#!/usr/bin/env python3

"""
Test the enhanced conversational chatbot
"""

import requests
import json

def test_enhanced_chatbot():
    """Test the enhanced conversational features"""
    
    base_url = "http://localhost:8001"
    session_id = "enhanced_test_456"
    
    test_scenarios = [
        {
            "name": "Enhanced Greeting",
            "query": "hello",
            "expected": "More engaging greeting"
        },
        {
            "name": "Follow-up Greeting",  
            "query": "hi again",
            "expected": "Reference to previous conversation"
        },
        {
            "name": "Restaurant Query",
            "query": "restaurants in beyoglu",
            "expected": "Better restaurant response with smart suggestions"
        },
        {
            "name": "Transportation Query",
            "query": "how to get around",
            "expected": "Concise transportation info"
        },
        {
            "name": "Shopping Query",
            "query": "where to shop",
            "expected": "Improved shopping response"
        }
    ]
    
    print("=" * 70)
    print("TESTING ENHANCED CONVERSATIONAL CHATBOT")
    print("=" * 70)
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n🧪 Test {i+1}: {scenario['name']}")
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
                print(f"📝 Response Preview: {message[:200]}...")
                
                # Check for conversational elements
                conversational_indicators = [
                    '!', '?', 'you', 'your', 'great', 'perfect', 'amazing', 
                    'try asking', 'want more', 'interested in', 'what sounds'
                ]
                conversational_count = sum(1 for indicator in conversational_indicators if indicator.lower() in message.lower())
                
                if conversational_count >= 3:
                    print("💬 Conversational: Excellent ✅")
                elif conversational_count >= 2:
                    print("💬 Conversational: Good ✅")
                else:
                    print("💬 Conversational: Needs improvement")
                
                # Check response length
                if 150 <= len(message) <= 800:
                    print("📏 Length: Perfect (engaging but digestible) ✅")
                elif len(message) < 150:
                    print("📏 Length: Too short")
                else:
                    print("📏 Length: Too long")
                
                # Check for follow-up suggestions
                if any(phrase in message.lower() for phrase in ['try asking', 'want more', 'interested in', 'need help with']):
                    print("🎯 Engagement: Has follow-up suggestions ✅")
                else:
                    print("🎯 Engagement: Limited follow-up")
                    
                # Check for emojis and visual elements
                emoji_count = sum(1 for char in message if ord(char) > 127)
                if emoji_count >= 3:
                    print("🎨 Visual: Good use of emojis ✅")
                else:
                    print("🎨 Visual: Could use more visual elements")
                    
            else:
                print(f"❌ Error: Status {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Add small delay between requests
        import time
        time.sleep(1)
    
    print(f"\n{'='*70}")
    print("🎉 Enhanced chatbot testing completed!")
    print("Check the conversational quality, engagement, and visual elements above.")

if __name__ == "__main__":
    test_enhanced_chatbot()
