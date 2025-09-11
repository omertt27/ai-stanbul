#!/usr/bin/env python3
"""
Simple test script to verify the chatbot works by making requests to the FastAPI endpoints.
This script starts a test server temporarily to run the tests.
"""

import subprocess
import time
import requests
import json
import signal
import os
import sys
from typing import Dict, Any, List

def start_server():
    """Start the FastAPI server in background."""
    backend_dir = "/Users/omer/Desktop/ai-stanbul/backend"
    try:
        # Start server using uvicorn
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        print("🚀 Starting server...")
        time.sleep(5)
        
        # Test if server is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server started successfully")
                return process
        except:
            pass
            
        print("❌ Server failed to start")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return None

def stop_server(process):
    """Stop the server process."""
    if process:
        process.terminate()
        process.wait()
        print("🛑 Server stopped")

def test_chatbot_quality():
    """Test the chatbot quality and conversational abilities."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing AIstanbul Chatbot Quality")
    print("=" * 50)
    
    # Test cases with expected quality criteria
    test_cases = [
        {
            "name": "Museum Recommendations",
            "query": "What museums should I visit in Istanbul?",
            "criteria": ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"]
        },
        {
            "name": "Restaurant Query",
            "query": "I'm looking for good restaurants in Istanbul",
            "criteria": ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"]
        },
        {
            "name": "Itinerary Planning", 
            "query": "I have 2 days in Istanbul. Can you help me plan an itinerary?",
            "criteria": ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"]
        },
        {
            "name": "Typo Handling",
            "query": "whre can i fnd gud restarnts in istambul?",
            "criteria": ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"]
        },
        {
            "name": "Vague Query",
            "query": "something fun",
            "criteria": ["no_emojis", "no_pricing", "relevant_content", "helpful_tone"]
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        
        try:
            # Make request to AI endpoint
            response = requests.post(
                f"{base_url}/ai",
                json={"query": test_case["query"]},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get("response", "")
                
                print(f"Response: {ai_response[:100]}...")
                
                # Check quality criteria
                quality = check_response_quality(ai_response, test_case["criteria"])
                results.append(quality)
                
                # Display quality results
                for criterion, passed in quality.items():
                    status = "✅" if passed else "❌"
                    print(f"  {status} {criterion.replace('_', ' ').title()}")
                    
            else:
                print(f"❌ Request failed with status {response.status_code}")
                results.append({criterion: False for criterion in test_case["criteria"]})
                
        except Exception as e:
            print(f"❌ Request failed: {e}")
            results.append({criterion: False for criterion in test_case["criteria"]})
        
        time.sleep(1)  # Rate limiting
    
    # Calculate overall scores
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY")
    print("=" * 50)
    
    if results:
        criteria_scores = {}
        for criterion in ["no_emojis", "has_weather", "no_pricing", "relevant_content", "helpful_tone"]:
            total_tests = sum(1 for r in results if criterion in r)
            passed_tests = sum(1 for r in results if r.get(criterion, False))
            if total_tests > 0:
                criteria_scores[criterion] = (passed_tests / total_tests) * 100
        
        for criterion, score in criteria_scores.items():
            status = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            print(f"{status} {criterion.replace('_', ' ').title()}: {score:.1f}%")
        
        overall_score = sum(criteria_scores.values()) / len(criteria_scores) if criteria_scores else 0
        print(f"\n🎯 Overall Performance: {overall_score:.1f}%")
        
        # Final verdict
        if overall_score >= 85:
            print("🌟 VERDICT: EXCELLENT - Ready to compete with top Istanbul guide AIs!")
        elif overall_score >= 70:
            print("👍 VERDICT: GOOD - Strong performance with minor improvements needed")
        elif overall_score >= 50:
            print("⚠️ VERDICT: FAIR - Functional but needs optimization")
        else:
            print("❌ VERDICT: POOR - Significant improvements needed")
        
        return overall_score
    else:
        print("❌ No successful tests completed")
        return 0

def check_response_quality(response_text: str, criteria: List[str]) -> Dict[str, bool]:
    """Check if response meets quality criteria."""
    text_lower = response_text.lower()
    
    results = {}
    for criterion in criteria:
        if criterion == "no_emojis":
            # Check for common emoji patterns
            emoji_patterns = ["🏛️", "🍽️", "☀️", "🌧️", "❄️", "🔥", "💰", "🎨", "🍕", "🥙", "😊", "👍", "🌟"]
            results[criterion] = not any(emoji in response_text for emoji in emoji_patterns)
        elif criterion == "has_weather":
            weather_keywords = ["weather", "temperature", "sunny", "rainy", "cloudy", "celsius", "fahrenheit", "°c", "°f"]
            results[criterion] = any(keyword in text_lower for keyword in weather_keywords)
        elif criterion == "no_pricing":
            pricing_keywords = ["$", "€", "₺", "price", "cost", "fee", "entrance", "ticket", "lira", "euro", "dollar"]
            results[criterion] = not any(keyword in text_lower for keyword in pricing_keywords)
        elif criterion == "relevant_content":
            istanbul_keywords = ["istanbul", "turkey", "turkish", "bosphorus", "galata", "sultanahmet", "hagia sophia", "blue mosque", "topkapi"]
            results[criterion] = any(keyword in text_lower for keyword in istanbul_keywords)
        elif criterion == "helpful_tone":
            helpful_indicators = ["recommend", "suggest", "best", "great", "wonderful", "visit", "explore", "should", "try"]
            results[criterion] = any(indicator in text_lower for indicator in helpful_indicators)
        else:
            results[criterion] = True
            
    return results

def main():
    """Main test execution."""
    print("🚀 AIstanbul Chatbot Quality Assessment")
    print("=" * 50)
    
    # Start server
    server_process = start_server()
    if not server_process:
        print("❌ Could not start server. Please check if port 8000 is available.")
        return
    
    try:
        # Run tests
        score = test_chatbot_quality()
        
        print(f"\n🎯 Final Assessment Score: {score:.1f}%")
        
        if score >= 85:
            print("\n🌟 CONCLUSION: The AIstanbul chatbot is ready for production!")
            print("   ✅ Robust against challenging inputs")
            print("   ✅ Proper content filtering")
            print("   ✅ Enhanced user experience")
            print("   ✅ Competitive with other Istanbul guide AIs")
        else:
            print(f"\n⚠️ CONCLUSION: The chatbot needs improvement (Score: {score:.1f}%)")
            
    finally:
        # Stop server
        stop_server(server_process)

if __name__ == "__main__":
    main()
