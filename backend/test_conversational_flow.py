#!/usr/bin/env python3
"""
Comprehensive test script for conversational flow and multi-turn questions.
Tests the AIstanbul chatbot's ability to handle follow-up questions and 
maintain context like a professional Istanbul guide AI.
"""

import requests
import json
import time
from typing import List, Dict, Any

class ConversationTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.conversation_id = None
        
    def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message to the chatbot and return the response."""
        url = f"{self.base_url}/chat"
        payload = {"message": message}
        
        if self.conversation_id:
            payload["conversation_id"] = self.conversation_id
            
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Store conversation ID for follow-up questions
            if "conversation_id" in result:
                self.conversation_id = result["conversation_id"]
                
            return result
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def check_response_quality(self, response: Dict[str, Any], criteria: List[str]) -> Dict[str, bool]:
        """Check if response meets quality criteria."""
        if "error" in response:
            return {criterion: False for criterion in criteria}
            
        text = response.get("response", "").lower()
        
        results = {}
        for criterion in criteria:
            if criterion == "no_emojis":
                # Check for common emoji patterns
                emoji_patterns = ["🏛️", "🍽️", "☀️", "🌧️", "❄️", "🔥", "💰", "🎨", "🍕", "🥙"]
                results[criterion] = not any(emoji in response.get("response", "") for emoji in emoji_patterns)
            elif criterion == "has_weather":
                weather_keywords = ["weather", "temperature", "sunny", "rainy", "cloudy", "celsius", "fahrenheit"]
                results[criterion] = any(keyword in text for keyword in weather_keywords)
            elif criterion == "no_pricing":
                pricing_keywords = ["$", "€", "₺", "price", "cost", "fee", "entrance", "ticket", "lira", "euro", "dollar"]
                results[criterion] = not any(keyword in text for keyword in pricing_keywords)
            elif criterion == "relevant_content":
                istanbul_keywords = ["istanbul", "turkey", "bosphorus", "galata", "sultanahmet", "hagia sophia", "blue mosque"]
                results[criterion] = any(keyword in text for keyword in istanbul_keywords)
            elif criterion == "helpful_tone":
                helpful_indicators = ["recommend", "suggest", "best", "great", "wonderful", "visit", "explore"]
                results[criterion] = any(indicator in text for indicator in helpful_indicators)
            else:
                results[criterion] = True
                
        return results

def test_conversation_scenarios():
    """Test various conversation scenarios."""
    tester = ConversationTester()
    
    # Test server availability
    try:
        health_check = requests.get(f"{tester.base_url}/health", timeout=5)
        if health_check.status_code != 200:
            print("❌ Server is not running. Please start the backend server first.")
            return
    except:
        print("❌ Cannot connect to server. Please start the backend server first.")
        return
        
    print("🧪 Testing AIstanbul Chatbot Conversational Flow")
    print("=" * 60)
    
    # Scenario 1: Museum recommendations with follow-ups
    print("\n📚 Scenario 1: Museum Recommendations with Follow-ups")
    print("-" * 50)
    
    tester.conversation_id = None  # Reset conversation
    
    # Initial question
    response1 = tester.send_message("What museums should I visit in Istanbul?")
    quality1 = tester.check_response_quality(response1, ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: What museums should I visit in Istanbul?")
    print(f"A1: {response1.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality1}")
    
    time.sleep(1)
    
    # Follow-up question 1
    response2 = tester.send_message("Which one is best for kids?")
    quality2 = tester.check_response_quality(response2, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ2: Which one is best for kids?")
    print(f"A2: {response2.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality2}")
    
    time.sleep(1)
    
    # Follow-up question 2
    response3 = tester.send_message("How do I get there from Taksim?")
    quality3 = tester.check_response_quality(response3, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ3: How do I get there from Taksim?")
    print(f"A3: {response3.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality3}")
    
    # Scenario 2: Restaurant recommendations with dietary preferences
    print("\n\n🍽️ Scenario 2: Restaurant Recommendations with Dietary Preferences")
    print("-" * 50)
    
    tester.conversation_id = None  # Reset conversation
    
    response4 = tester.send_message("I'm looking for good restaurants in Istanbul")
    quality4 = tester.check_response_quality(response4, ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: I'm looking for good restaurants in Istanbul")
    print(f"A1: {response4.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality4}")
    
    time.sleep(1)
    
    response5 = tester.send_message("I'm vegetarian, do any of these have good vegetarian options?")
    quality5 = tester.check_response_quality(response5, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ2: I'm vegetarian, do any of these have good vegetarian options?")
    print(f"A2: {response5.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality5}")
    
    time.sleep(1)
    
    response6 = tester.send_message("What about the atmosphere? I want somewhere romantic")
    quality6 = tester.check_response_quality(response6, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ3: What about the atmosphere? I want somewhere romantic")
    print(f"A3: {response6.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality6}")
    
    # Scenario 3: Complex multi-part question
    print("\n\n🗺️ Scenario 3: Complex Multi-part Planning")
    print("-" * 50)
    
    tester.conversation_id = None  # Reset conversation
    
    response7 = tester.send_message("I have 2 days in Istanbul. Can you help me plan an itinerary that includes historical sites, good food, and some shopping?")
    quality7 = tester.check_response_quality(response7, ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: I have 2 days in Istanbul. Can you help me plan an itinerary that includes historical sites, good food, and some shopping?")
    print(f"A1: {response7.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality7}")
    
    time.sleep(1)
    
    response8 = tester.send_message("That sounds great! But I'm staying near Galata Tower. Can you adjust the plan based on my location?")
    quality8 = tester.check_response_quality(response8, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ2: That sounds great! But I'm staying near Galata Tower. Can you adjust the plan based on my location?")
    print(f"A2: {response8.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality8}")
    
    # Scenario 4: Challenging/Edge case questions
    print("\n\n⚡ Scenario 4: Challenging Questions")
    print("-" * 50)
    
    tester.conversation_id = None  # Reset conversation
    
    # Typo/misspelling test
    response9 = tester.send_message("whre can i fnd gud restarnts in istambul?")
    quality9 = tester.check_response_quality(response9, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: whre can i fnd gud restarnts in istambul? (typos)")
    print(f"A1: {response9.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality9}")
    
    time.sleep(1)
    
    # Vague question test
    response10 = tester.send_message("something fun")
    quality10 = tester.check_response_quality(response10, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ2: something fun (vague)")
    print(f"A2: {response10.get('response', 'Error')[:200]}...")
    print(f"Quality: {quality10}")
    
    # Calculate overall scores
    all_qualities = [quality1, quality2, quality3, quality4, quality5, quality6, quality7, quality8, quality9, quality10]
    
    print("\n\n📊 OVERALL PERFORMANCE SUMMARY")
    print("=" * 60)
    
    criteria_scores = {}
    for criterion in ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"]:
        total_tests = sum(1 for q in all_qualities if criterion in q)
        passed_tests = sum(1 for q in all_qualities if q.get(criterion, False))
        if total_tests > 0:
            criteria_scores[criterion] = (passed_tests / total_tests) * 100
    
    for criterion, score in criteria_scores.items():
        status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
        print(f"{status} {criterion.replace('_', ' ').title()}: {score:.1f}%")
    
    overall_score = sum(criteria_scores.values()) / len(criteria_scores) if criteria_scores else 0
    print(f"\n🎯 Overall Performance: {overall_score:.1f}%")
    
    if overall_score >= 85:
        print("🌟 EXCELLENT: Ready to compete with top Istanbul guide AIs!")
    elif overall_score >= 70:
        print("👍 GOOD: Strong performance with room for minor improvements")
    elif overall_score >= 50:
        print("⚠️ FAIR: Needs some improvements before production")
    else:
        print("❌ POOR: Significant improvements needed")

if __name__ == "__main__":
    test_conversation_scenarios()
