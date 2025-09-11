#!/usr/bin/env python3
"""
Direct test of the chatbot functionality without requiring a running server.
Tests conversational flow by directly calling the chat endpoint function.
"""

import sys
import os
import asyncio
import json
from typing import Dict, Any

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import the main app and chat function
try:
    from main import chat_endpoint, ChatRequest
    from database import get_db
    print("✅ Successfully imported main chatbot functions")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

class DirectChatTester:
    """Test chatbot directly without HTTP requests."""
    
    def __init__(self):
        self.conversation_id = None
    
    async def send_message(self, message: str) -> Dict[str, Any]:
        """Send a message directly to the chat function."""
        try:
            # Create request object
            request = ChatRequest(
                message=message,
                conversation_id=self.conversation_id
            )
            
            # Call the chat endpoint directly
            response = await chat_endpoint(request)
            
            # Update conversation ID
            if hasattr(response, 'conversation_id') and response.conversation_id:
                self.conversation_id = response.conversation_id
            
            return {
                "response": response.response if hasattr(response, 'response') else str(response),
                "conversation_id": getattr(response, 'conversation_id', None)
            }
            
        except Exception as e:
            return {"error": f"Chat function failed: {str(e)}"}
    
    def check_response_quality(self, response: Dict[str, Any], criteria: list) -> Dict[str, bool]:
        """Check if response meets quality criteria."""
        if "error" in response:
            return {criterion: False for criterion in criteria}
            
        text = response.get("response", "").lower()
        original_text = response.get("response", "")
        
        results = {}
        for criterion in criteria:
            if criterion == "no_emojis":
                # Check for common emoji patterns
                emoji_patterns = ["🏛️", "🍽️", "☀️", "🌧️", "❄️", "🔥", "💰", "🎨", "🍕", "🥙", "😊", "👍", "🌟"]
                results[criterion] = not any(emoji in original_text for emoji in emoji_patterns)
            elif criterion == "has_weather":
                weather_keywords = ["weather", "temperature", "sunny", "rainy", "cloudy", "celsius", "fahrenheit", "°c", "°f"]
                results[criterion] = any(keyword in text for keyword in weather_keywords)
            elif criterion == "no_pricing":
                pricing_keywords = ["$", "€", "₺", "price", "cost", "fee", "entrance", "ticket", "lira", "euro", "dollar"]
                results[criterion] = not any(keyword in text for keyword in pricing_keywords)
            elif criterion == "relevant_content":
                istanbul_keywords = ["istanbul", "turkey", "turkish", "bosphorus", "galata", "sultanahmet", "hagia sophia", "blue mosque", "topkapi"]
                results[criterion] = any(keyword in text for keyword in istanbul_keywords)
            elif criterion == "helpful_tone":
                helpful_indicators = ["recommend", "suggest", "best", "great", "wonderful", "visit", "explore", "should", "try"]
                results[criterion] = any(indicator in text for indicator in helpful_indicators)
            else:
                results[criterion] = True
                
        return results

async def run_conversational_tests():
    """Run comprehensive conversational flow tests."""
    print("🧪 Testing AIstanbul Chatbot Conversational Flow (Direct)")
    print("=" * 60)
    
    tester = DirectChatTester()
    
    # Test 1: Museum recommendations with follow-ups
    print("\n📚 Test 1: Museum Recommendations with Follow-ups")
    print("-" * 50)
    
    tester.conversation_id = None
    
    # Initial question
    response1 = await tester.send_message("What museums should I visit in Istanbul?")
    quality1 = tester.check_response_quality(response1, ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: What museums should I visit in Istanbul?")
    print(f"A1: {response1.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality1}")
    
    # Follow-up question
    response2 = await tester.send_message("Which one is best for kids?")
    quality2 = tester.check_response_quality(response2, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ2: Which one is best for kids?")
    print(f"A2: {response2.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality2}")
    
    # Test 2: Restaurant recommendations
    print("\n\n🍽️ Test 2: Restaurant Recommendations")
    print("-" * 50)
    
    tester.conversation_id = None
    
    response3 = await tester.send_message("I'm looking for good restaurants in Istanbul")
    quality3 = tester.check_response_quality(response3, ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: I'm looking for good restaurants in Istanbul")
    print(f"A1: {response3.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality3}")
    
    response4 = await tester.send_message("I'm vegetarian, do any of these have vegetarian options?")
    quality4 = tester.check_response_quality(response4, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ2: I'm vegetarian, do any of these have vegetarian options?")
    print(f"A2: {response4.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality4}")
    
    # Test 3: Complex planning
    print("\n\n🗺️ Test 3: Complex Itinerary Planning")
    print("-" * 50)
    
    tester.conversation_id = None
    
    response5 = await tester.send_message("I have 2 days in Istanbul. Can you help me plan an itinerary?")
    quality5 = tester.check_response_quality(response5, ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: I have 2 days in Istanbul. Can you help me plan an itinerary?")
    print(f"A1: {response5.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality5}")
    
    # Test 4: Challenging inputs
    print("\n\n⚡ Test 4: Challenging Inputs")
    print("-" * 50)
    
    tester.conversation_id = None
    
    # Typos
    response6 = await tester.send_message("whre can i fnd gud restarnts in istambul?")
    quality6 = tester.check_response_quality(response6, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"Q1: whre can i fnd gud restarnts in istambul? (typos)")
    print(f"A1: {response6.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality6}")
    
    # Vague question
    response7 = await tester.send_message("something fun")
    quality7 = tester.check_response_quality(response7, ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"])
    
    print(f"\nQ2: something fun (vague)")
    print(f"A2: {response7.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality7}")
    
    # Test 5: Inappropriate content filtering
    print("\n\n🚫 Test 5: Content Filtering")
    print("-" * 50)
    
    response8 = await tester.send_message("Tell me about nightlife and adult entertainment")
    quality8 = tester.check_response_quality(response8, ["no_emojis", "no_pricing"])
    
    print(f"Q1: Tell me about nightlife and adult entertainment")
    print(f"A1: {response8.get('response', 'Error')[:150]}...")
    print(f"Quality: {quality8}")
    
    # Calculate overall performance
    all_qualities = [quality1, quality2, quality3, quality4, quality5, quality6, quality7, quality8]
    
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
    
    # Detailed analysis
    print(f"\n📋 DETAILED ANALYSIS")
    print(f"- Emoji Removal: {'✅ Working' if criteria_scores.get('no_emojis', 0) >= 90 else '⚠️ Needs improvement'}")
    print(f"- Weather Integration: {'✅ Working' if criteria_scores.get('has_weather', 0) >= 70 else '⚠️ Needs improvement'}")
    print(f"- Cost Info Removal: {'✅ Working' if criteria_scores.get('no_pricing', 0) >= 90 else '⚠️ Needs improvement'}")
    print(f"- Content Relevance: {'✅ Working' if criteria_scores.get('relevant_content', 0) >= 80 else '⚠️ Needs improvement'}")
    print(f"- Helpful Tone: {'✅ Working' if criteria_scores.get('helpful_tone', 0) >= 80 else '⚠️ Needs improvement'}")
    
    if overall_score >= 85:
        print("\n🌟 VERDICT: EXCELLENT - Ready to compete with top Istanbul guide AIs!")
        print("   The chatbot demonstrates robust conversational abilities,")
        print("   proper content filtering, and helpful recommendations.")
    elif overall_score >= 70:
        print("\n👍 VERDICT: GOOD - Strong performance with minor areas for improvement")
        print("   The chatbot handles most scenarios well but could benefit from")
        print("   fine-tuning in specific areas.")
    elif overall_score >= 50:
        print("\n⚠️ VERDICT: FAIR - Functional but needs improvements before production")
        print("   The chatbot works but requires optimization for better user experience.")
    else:
        print("\n❌ VERDICT: POOR - Significant improvements needed")
        print("   The chatbot needs substantial work before deployment.")
    
    return overall_score

if __name__ == "__main__":
    try:
        # Run the async test function
        score = asyncio.run(run_conversational_tests())
        print(f"\n🎯 Final Score: {score:.1f}%")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
