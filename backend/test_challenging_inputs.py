#!/usr/bin/env python3
"""
Test the AIstanbul chatbot with challenging inputs that could cause wrong answers
"""

import requests
import json
import time

# Test the chatbot API
API_URL = "http://localhost:8000/ai"

def test_chatbot(query, expected_behavior=None):
    """Test a single query"""
    print(f"\n🔍 Testing: '{query}'")
    print(f"Expected: {expected_behavior or 'Should handle gracefully'}")
    print("-" * 80)
    
    try:
        print(" Thinking... (processing your query)")
        response = requests.post(API_URL, json={"query": query}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('message', 'No response')
            print(f"✅ Status: {response.status_code}")
            print(f"📝 Response: {answer[:200]}...")
            
            # Check for potential issues
            issues = []
            if "paris" in answer.lower() and "istanbul" not in answer.lower():
                issues.append("❌ Mentioned Paris without Istanbul context")
            elif "london" in answer.lower() and "istanbul" not in answer.lower():
                issues.append("❌ Mentioned London without Istanbul context")
            elif "new york" in answer.lower() and "istanbul" not in answer.lower():
                issues.append("❌ Mentioned New York without Istanbul context")
            
            if "sorry" in answer.lower() and "don't know" in answer.lower():
                issues.append("⚠️  Admitted ignorance (can be appropriate)")
            if len(answer) < 50:
                issues.append("⚠️  Very short response")
            if "turkey" in answer.lower() and "istanbul" not in answer.lower():
                issues.append("⚠️  Generic Turkey info instead of Istanbul focus")
            
            # Positive indicators
            if "istanbul" in answer.lower():
                issues.append("✅ Appropriately mentions Istanbul")
            if any(phrase in answer.lower() for phrase in ["i'm here to help", "assist you", "travel assistant"]):
                issues.append("✅ Professional assistant tone")
            if "🤔" in answer or "�" in answer or "🎯" in answer:
                issues.append("✅ Engaging visual elements")
                
            if issues:
                print("📊 Analysis:")
                for issue in issues:
                    print(f"   {issue}")
            else:
                print("✅ Response looks appropriate")
                
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"📝 Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("=" * 80)
    time.sleep(2)  # Rate limiting

# Challenging test cases
challenging_tests = [
    # Geographic confusion
    ("What's the best restaurant in Paris?", "Should clarify this is about Istanbul, not Paris"),
    ("Tell me about London Bridge", "Should redirect to Istanbul bridges or clarify"),
    ("Where is Times Square?", "Should clarify this is an Istanbul chatbot"),
    
    # Ambiguous locations
    ("Where is the airport?", "Should ask which airport or default to IST"),
    ("How do I get to the city center?", "Should ask from where or provide general guidance"),
    ("Best hotels near the beach", "Should clarify Istanbul doesn't have traditional beaches"),
    
    # Non-Istanbul Turkey questions
    ("What about Cappadocia?", "Should focus on Istanbul or politely redirect"),
    ("Tell me about Ankara", "Should stay focused on Istanbul"),
    ("Antalya beaches", "Should redirect to Istanbul topics"),
    
    # Impossible or wrong requests
    ("Where can I ski in Istanbul?", "Should explain Istanbul doesn't have ski resorts"),
    ("Best surfing spots in Istanbul", "Should clarify Istanbul isn't a surfing destination"),
    ("Where to see penguins in Istanbul?", "Should clarify this isn't possible"),
    
    # Vague or confusing queries
    ("help", "Should provide structured help about Istanbul"),
    ("???", "Should ask for clarification"),
    ("fdsjklfjsdklfjsdf", "Should handle gibberish gracefully"),
    ("", "Should handle empty input"),
    
    # Leading questions
    ("Isn't Istanbul really expensive?", "Should provide balanced cost information"),
    ("Why is Istanbul so dangerous?", "Should provide factual safety information"),
    ("Is Turkish food too spicy?", "Should give accurate info about Turkish cuisine"),
    
    # Outdated information tests
    ("When does the Ottoman Empire palace open?", "Should clarify Ottoman Empire vs current museums"),
    ("How much is the Turkish Lira worth?", "Should provide current or recent info"),
    ("Is COVID still affecting travel?", "Should provide current travel information"),
    
    # Scale/specificity issues
    ("What's the population of every district?", "Should provide helpful summary, not overwhelming detail"),
    ("List every restaurant in Istanbul", "Should provide categories or top recommendations"),
    ("Give me the schedule for every bus", "Should direct to resources or key routes"),
    
    # Language/translation issues
    ("Merhaba, nasılsın?", "Should respond appropriately to Turkish"),
    ("你好，伊斯坦布尔", "Should handle non-Turkish/English languages"),
    ("Parlez-vous français?", "Should clarify language capabilities"),
    
    # Time-sensitive questions
    ("What's happening tonight?", "Should ask for date or provide general guidance"),
    ("Is it raining right now?", "Should direct to weather resources"),
    ("What time does the sun set today?", "Should provide seasonal info or direct to resources"),
    
    # Personal/inappropriate requests
    ("What's your personal opinion about politics?", "Should stay neutral and focus on travel"),
    ("Can you book a hotel for me?", "Should clarify capabilities"),
    ("What's your favorite place?", "Should provide professional recommendations"),
]

def run_all_tests():
    """Run all challenging tests"""
    print("🤖 AISTADT CHATBOT CHALLENGING INPUT TESTS")
    print("=" * 80)
    
    for i, (query, expected) in enumerate(challenging_tests, 1):
        print(f"\n📋 Test {i}/{len(challenging_tests)}")
        test_chatbot(query, expected)
    
    print("\n🎯 TESTING COMPLETE!")
    print("Review the responses above for any concerning patterns:")
    print("- Wrong city information")
    print("- Hallucinated facts")
    print("- Inappropriate responses")
    print("- Poor error handling")

if __name__ == "__main__":
    run_all_tests()
