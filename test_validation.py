#!/usr/bin/env python3
"""
Test script for enhanced query validation in AIstanbul chatbot
"""

import sys
sys.path.append('/Users/omer/Desktop/ai-stanbul/backend')

from enhanced_chatbot import EnhancedQueryUnderstanding, ValidationErrorHandler

def test_problematic_queries():
    """Test various problematic queries that should be caught"""
    
    query_understander = EnhancedQueryUnderstanding()
    
    test_cases = [
        # Geographic errors
        "restaurants in Paris Istanbul",
        "how to get to Eiffel Tower from Sultanahmet",
        "best museums in Manhattan Turkey",
        
        # Logical contradictions
        "vegetarian steakhouse in Beyoglu",
        "kosher restaurant serving pork",
        "underwater mountaintop restaurant",
        
        # Temporal errors
        "Ottoman restaurants in 2025",
        "Byzantine museums in year 3000",
        
        # Budget unrealistic
        "free luxury restaurants",
        "million dollar meal in Istanbul",
        
        # Fictional content
        "Hogwarts school in Istanbul",
        "flying car transport to Galata Tower",
        
        # Ambiguous queries
        "good",
        "it",
        "what about that",
        "",
        "restaurant museum transport expensive cheap",
        
        # Valid queries (should pass)
        "restaurants in Beyoglu",
        "how to get from Taksim to Kadikoy",
        "best museums in Sultanahmet",
        "vegetarian restaurants in Istanbul"
    ]
    
    print("🧪 Testing Enhanced Query Validation\n")
    print("=" * 60)
    
    for query in test_cases:
        print(f"\n📝 Testing: '{query}'")
        
        # Test validation
        validation_result = query_understander.validate_query_logic(query)
        
        # Test ambiguity detection
        ambiguity_result = query_understander.detect_ambiguous_queries(query)
        
        # Test intent extraction
        intent_result = query_understander.extract_intent_and_entities(query)
        
        # Show results
        if not validation_result['is_valid']:
            print(f"❌ VALIDATION ERROR: {validation_result['error_type']}")
            print(f"   Issue: {validation_result['issues'][0] if validation_result['issues'] else 'Unknown'}")
            error_response = ValidationErrorHandler.generate_error_response(validation_result)
            print(f"   Response: {error_response[:100]}...")
        
        elif ambiguity_result['is_ambiguous']:
            print(f"⚠️  AMBIGUOUS: {ambiguity_result['ambiguity_type']}")
            print(f"   Clarification: {ambiguity_result['clarification_needed']}")
            clarification_response = ValidationErrorHandler.generate_clarification_response(ambiguity_result)
            print(f"   Response: {clarification_response[:100]}...")
        
        else:
            print(f"✅ VALID: Intent = {intent_result['intent']}, Confidence = {intent_result['confidence']}")
        
        print("-" * 40)
    
    print("\n🎯 Validation testing complete!")

if __name__ == "__main__":
    test_problematic_queries()
