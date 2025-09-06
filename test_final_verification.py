#!/usr/bin/env python3
"""
Final verification test for follow-up query improvements
Tests the most problematic scenarios that could lead to wrong answers
"""

import requests
import json
import random
import time

def test_api(query, session_id):
    """Test API with a query and return response"""
    try:
        response = requests.post(
            'http://localhost:8001/ai',
            json={'query': query, 'session_id': session_id},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['message']
        else:
            return f"ERROR {response.status_code}: {response.text}"
    except Exception as e:
        return f"EXCEPTION: {str(e)}"

def run_conversation_test(test_name, conversation_steps, session_id):
    """Run a multi-step conversation test"""
    print(f"\n" + "="*80)
    print(f"🔬 CONVERSATION TEST: {test_name}")
    print("="*80)
    
    for i, (query, expected_context) in enumerate(conversation_steps, 1):
        print(f"\n📋 Step {i}: '{query}'")
        print(f"Expected context: {expected_context}")
        print("-" * 60)
        
        response = test_api(query, session_id)
        print(f"Response: {response[:300]}...")
        
        # Check for expected context clues
        response_lower = response.lower()
        if expected_context:
            context_found = any(ctx.lower() in response_lower for ctx in expected_context)
            if context_found:
                print("✅ Expected context found")
            else:
                print(f"🚨 ISSUE: Expected context {expected_context} not found")
        
        time.sleep(1)  # Small delay between requests

def main():
    print("🧪 Final Verification Tests for Follow-up Query Improvements")
    print("="*80)
    
    # Test 1: Typo + Follow-up Context Preservation
    run_conversation_test(
        "Typo + Follow-up Context",
        [
            ("places in galata", ["galata", "beyoglu"]),
            ("resturants also", ["galata", "beyoglu", "perfect"])  # Should acknowledge context despite typo
        ],
        f"typo_context_{random.randint(1000000, 9999999)}"
    )
    
    # Test 2: Complex Context Switching
    run_conversation_test(
        "Complex Context Switching", 
        [
            ("kadikoy attractions", ["kadikoy"]),
            ("restaurants also", ["kadikoy", "perfect"]),
            ("no, restaurants in sultanahmet", ["sultanahmet"])  # Should switch completely
        ],
        f"context_switch_{random.randint(1000000, 9999999)}"
    )
    
    # Test 3: Ambiguous Pronouns with Multiple Locations
    run_conversation_test(
        "Ambiguous Pronouns with Multiple Locations",
        [
            ("compare sultanahmet and beyoglu", ["sultanahmet", "beyoglu"]),
            ("restaurants there", ["beyoglu"])  # Should pick one or ask for clarification
        ],
        f"ambiguous_{random.randint(1000000, 9999999)}"
    )
    
    # Test 4: Generic Follow-ups
    run_conversation_test(
        "Generic Follow-ups Should Ask for Clarification",
        [
            ("museums in sultanahmet", ["museum", "sultanahmet"]),
            ("what else?", ["what would you like", "specific", "clarification"])  # Should ask for clarification
        ],
        f"generic_{random.randint(1000000, 9999999)}"
    )
    
    # Test 5: Cross-category Follow-ups
    run_conversation_test(
        "Cross-category Follow-ups",
        [
            ("museums in beyoglu", ["museum", "beyoglu"]),
            ("places to visit also", ["places", "beyoglu"])  # Should show general places, not just museums
        ],
        f"cross_category_{random.randint(1000000, 9999999)}"
    )
    
    # Test 6: Input Validation Edge Cases
    print(f"\n" + "="*80)
    print("🛡️ INPUT VALIDATION TESTS")
    print("="*80)
    
    validation_tests = [
        ("I'm going to istanbul", "Should work with apostrophes"),
        ("What's the best restaurant?", "Should work with apostrophes"),
        ("restaurants; DROP TABLE", "Should block SQL injection"),
        ("restaurant recommendations", "Should work normally")
    ]
    
    for query, description in validation_tests:
        print(f"\n🧪 {description}")
        print(f"Query: '{query}'")
        response = test_api(query, f"validation_{random.randint(1000, 9999)}")
        if "ERROR" in response or "EXCEPTION" in response:
            print(f"🚨 FAILED: {response}")
        else:
            print(f"✅ PASSED: {response[:100]}...")
    
    print(f"\n" + "="*80)
    print("✅ Final verification tests completed!")
    print("Review the outputs above for any remaining issues with context handling.")
    print("="*80)

if __name__ == "__main__":
    main()
