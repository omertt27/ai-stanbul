#!/usr/bin/env python3
"""
Test script for the Advanced Language Processing module
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api_clients.language_processing import (
    AdvancedLanguageProcessor,
    process_user_query,
    extract_intent_and_entities,
    is_followup
)


def test_intent_recognition():
    """Test intent recognition functionality"""
    print("🧠 Testing Intent Recognition")
    print("=" * 40)
    
    test_queries = [
        "I want to find a good Turkish restaurant",
        "Where can I visit Hagia Sophia?",
        "How do I get from Taksim to Sultanahmet?",
        "What's the weather like today?",
        "I need a hotel in Beyoğlu",
        "Where can I buy souvenirs?",
        "Any good bars for nightlife?",
        "Tell me about Istanbul"
    ]
    
    processor = AdvancedLanguageProcessor()
    
    for query in test_queries:
        result = processor.recognize_intent(query)
        print(f"Query: '{query}'")
        print(f"  Intent: {result.intent} (confidence: {result.confidence:.2f})")
        print(f"  Entities: {result.entities}")
        print(f"  Sentiment: {result.entities.get('sentiment', 'neutral')}")
        print()


def test_entity_extraction():
    """Test entity extraction functionality"""
    print("🔍 Testing Entity Extraction")
    print("=" * 40)
    
    test_queries = [
        "I want cheap Turkish food in Sultanahmet",
        "Show me expensive restaurants near Galata Tower",
        "Looking for vegetarian options in Kadıköy for dinner",
        "Historical museums in Beyoğlu during morning hours",
        "Budget-friendly shopping in Grand Bazaar"
    ]
    
    processor = AdvancedLanguageProcessor()
    
    for query in test_queries:
        entities = processor.extract_entities(query)
        print(f"Query: '{query}'")
        print(f"  Locations: {entities.locations}")
        print(f"  Cuisines: {entities.cuisines}")
        print(f"  Budget: {entities.budget_indicators}")
        print(f"  Time: {entities.time_references}")
        print(f"  Interests: {entities.interests}")
        print()


def test_followup_detection():
    """Test follow-up question detection"""
    print("🔄 Testing Follow-up Detection")
    print("=" * 40)
    
    test_cases = [
        ("I want restaurants in Sultanahmet", False),
        ("What about there?", True),
        ("Any others?", True),
        ("More options", True),
        ("Also in that area", True),
        ("Tell me about museums", False),
        ("Near there", True)
    ]
    
    for query, expected in test_cases:
        result = is_followup(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{query}' -> {result} (expected: {expected})")
    
    print()


def test_comprehensive_analysis():
    """Test comprehensive query understanding"""
    print("🔬 Testing Comprehensive Analysis")
    print("=" * 40)
    
    test_query = "I'm looking for a nice Turkish restaurant in Sultanahmet for dinner tonight, preferably not too expensive"
    
    result = process_user_query(test_query)
    
    print(f"Query: '{test_query}'")
    print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
    print(f"Entities: {result['entities']}")
    print(f"Ambiguity Score: {result['ambiguity_score']:.2f}")
    print(f"Suggestions: {result['suggestions']}")
    print(f"Clarifications: {result['clarifications']}")
    print(f"Is Follow-up: {result['is_followup']}")
    print()


def test_clarification_generation():
    """Test clarification question generation"""
    print("❓ Testing Clarification Questions")
    print("=" * 40)
    
    processor = AdvancedLanguageProcessor()
    
    # Test ambiguous queries
    test_cases = [
        ("I want food", "restaurant_search"),
        ("Where can I go?", "attraction_search"),
        ("How do I get there?", "transportation")
    ]
    
    for query, intent in test_cases:
        entities = processor.extract_entities(query)
        clarifications = processor.generate_clarification_questions(intent, entities.__dict__)
        
        print(f"Query: '{query}' (Intent: {intent})")
        print(f"Clarifications: {clarifications}")
        print()


def main():
    """Run all tests"""
    print("🚀 Advanced Language Processing Tests")
    print("=" * 50)
    print()
    
    try:
        test_intent_recognition()
        test_entity_extraction()
        test_followup_detection()
        test_comprehensive_analysis()
        test_clarification_generation()
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
