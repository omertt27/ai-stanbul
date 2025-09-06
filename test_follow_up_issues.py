#!/usr/bin/env python3
"""
Test script for problematic follow-up questions that could lead to wrong answers.
This tests edge cases, ambiguous contexts, and potential failure scenarios.
"""

import requests
import time
import json

# Test configuration
BASE_URL = "http://localhost:8001"

def test_query(query: str, session_id: str, description: str = "", expected_issue: str = ""):
    """Send a test query and analyze the response for potential issues"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {description}")
    print(f"Query: '{query}'")
    print(f"Session: {session_id}")
    if expected_issue:
        print(f"🚨 Expected Issue: {expected_issue}")
    print("-" * 80)
    
    try:
        response = requests.post(
            f"{BASE_URL}/ai",
            json={"query": query, "session_id": session_id},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", "No message")
            print(f"✅ Response:\n{message[:500]}{'...' if len(message) > 500 else ''}")
            
            # Analyze response for potential issues
            analyze_response_issues(message, query, expected_issue)
            
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def analyze_response_issues(response: str, query: str, expected_issue: str):
    """Analyze response for potential issues"""
    issues_found = []
    
    # Check for generic/non-specific responses
    if "Here are some" in response and len(response.split('\n')) < 10:
        issues_found.append("🔶 Response seems too generic")
    
    # Check for wrong location context
    locations = ['sultanahmet', 'beyoglu', 'kadikoy', 'galata', 'taksim', 'besiktas']
    query_lower = query.lower()
    response_lower = response.lower()
    
    for location in locations:
        if location in query_lower and location not in response_lower:
            # Check if query mentioned location but response doesn't reference it
            if any(word in query_lower for word in ['restaurant', 'place', 'visit', 'attraction']):
                issues_found.append(f"🔶 Query mentioned '{location}' but response doesn't reference it")
    
    # Check for contradictory information
    if "sultanahmet" in response_lower and "beyoglu" in response_lower and "sultanahmet" not in query_lower and "beyoglu" not in query_lower:
        issues_found.append("🔶 Response mentions multiple locations when query was specific")
    
    # Check for missing context awareness
    if any(word in query_lower for word in ['also', 'more', 'too', 'as well']) and "based on" not in response_lower and "perfect" not in response_lower:
        issues_found.append("🔶 Follow-up query not properly acknowledged")
    
    # Check for restaurant vs places confusion
    if "restaurant" in query_lower and "places to visit" in response_lower:
        issues_found.append("🚨 CRITICAL: Restaurant query returned places instead of restaurants")
    
    if "places" in query_lower and "restaurant" in response_lower and "places to visit" not in response_lower:
        issues_found.append("🚨 CRITICAL: Places query returned restaurants instead of places")
    
    # Print issues found
    if issues_found:
        print(f"\n🚨 ISSUES DETECTED:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print(f"\n✅ No obvious issues detected")

def main():
    print("🚀 Testing Problematic Follow-up Questions")
    print("="*80)
    
    # Test Case 1: Location confusion in follow-ups
    session1 = f"confusion_test_{int(time.time())}"
    test_query(
        "places in sultanahmet", 
        session1,
        "Setup: Places in Sultanahmet",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "restaurants in beyoglu", 
        session1,
        "Different location - should not use Sultanahmet context",
        "Should mention Beyoglu specifically, not Sultanahmet"
    )
    
    # Test Case 2: Ambiguous follow-up pronouns
    session2 = f"pronoun_test_{int(time.time())}"
    test_query(
        "kadikoy attractions", 
        session2,
        "Setup: Kadıköy attractions",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "what about restaurants there?", 
        session2,
        "Ambiguous 'there' - should resolve to Kadıköy",
        "Should understand 'there' refers to Kadıköy"
    )
    
    # Test Case 3: Mixed query types causing confusion
    session3 = f"mixed_test_{int(time.time())}"
    test_query(
        "taksim square", 
        session3,
        "Setup: Taksim Square (place)",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "also restaurants", 
        session3,
        "Follow-up for restaurants - context should be Taksim/Beyoglu",
        "Should provide restaurants in Taksim/Beyoglu area"
    )
    
    # Test Case 4: Contradictory follow-ups
    session4 = f"contradiction_test_{int(time.time())}"
    test_query(
        "places in galata", 
        session4,
        "Setup: Places in Galata",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "no, I meant restaurants in sultanahmet", 
        session4,
        "Contradictory follow-up - user changes mind completely",
        "Should switch to Sultanahmet restaurants, not use Galata context"
    )
    
    # Test Case 5: Overly generic follow-ups
    session5 = f"generic_test_{int(time.time())}"
    test_query(
        "beyoglu nightlife", 
        session5,
        "Setup: Beyoğlu nightlife",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "what else?", 
        session5,
        "Very generic follow-up",
        "Should ask for clarification rather than assume"
    )
    
    # Test Case 6: Cross-category confusion
    session6 = f"category_test_{int(time.time())}"
    test_query(
        "museums in sultanahmet", 
        session6,
        "Setup: Museums in Sultanahmet",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "show me places to visit", 
        session6,
        "Follow-up asking for places (not museums)",
        "Should show general places, not just museums"
    )
    
    # Test Case 7: Temporal confusion
    session7 = f"temporal_test_{int(time.time())}"
    test_query(
        "restaurants in kadikoy", 
        session7,
        "Setup: Restaurants in Kadıköy",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "what about tomorrow?", 
        session7,
        "Temporal follow-up - unclear what 'tomorrow' refers to",
        "Should ask for clarification about what they want to know for tomorrow"
    )
    
    # Test Case 8: Multiple location confusion
    session8 = f"multi_location_test_{int(time.time())}"
    test_query(
        "compare sultanahmet and beyoglu", 
        session8,
        "Setup: Comparison query with multiple locations",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "restaurants also", 
        session8,
        "Follow-up after multi-location query",
        "Should ask which location or provide both"
    )
    
    # Test Case 9: Typos in follow-ups
    session9 = f"typo_test_{int(time.time())}"
    test_query(
        "places in galata", 
        session9,
        "Setup: Places in Galata",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "resturants also", 
        session9,
        "Follow-up with typo in 'restaurants'",
        "Should still understand despite typo"
    )
    
    # Test Case 10: Wrong assumptions about user intent
    session10 = f"intent_test_{int(time.time())}"
    test_query(
        "I'm going to istanbul", 
        session10,
        "Setup: General travel statement",
        ""
    )
    
    time.sleep(2)
    
    test_query(
        "where should I go?", 
        session10,
        "Follow-up asking for recommendations",
        "Should ask about preferences rather than assume specific type"
    )
    
    print(f"\n{'='*80}")
    print("🎯 SUMMARY")
    print("="*80)
    print("The test cases above check for common issues with follow-up questions:")
    print("1. ❌ Location context confusion (wrong district)")
    print("2. ❌ Ambiguous pronouns (there, it, that)")
    print("3. ❌ Mixed query types (places vs restaurants)")
    print("4. ❌ Contradictory follow-ups (user changes mind)")
    print("5. ❌ Overly generic responses")
    print("6. ❌ Cross-category confusion (museums vs general places)")
    print("7. ❌ Temporal confusion (tomorrow, later, etc.)")
    print("8. ❌ Multiple location context issues")
    print("9. ❌ Typo handling in follow-ups")
    print("10. ❌ Wrong assumptions about user intent")
    print("\n✅ Review the responses above for any issues marked with 🚨 or 🔶")

if __name__ == "__main__":
    main()
