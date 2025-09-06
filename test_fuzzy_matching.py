#!/usr/bin/env python3

"""
Test script to verify fuzzy matching for restaurant typos
"""

import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

try:
    import main  # type: ignore
    from fuzzywuzzy import process  # type: ignore
    
    # Import the specific functions we need
    correct_typos = main.correct_typos
    create_fuzzy_keywords = main.create_fuzzy_keywords
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct directory and dependencies are installed")
    print(f"Looking for backend at: {backend_path}")
    sys.exit(1)

def test_fuzzy_matching():
    """Test the fuzzy matching functionality"""
    print("=" * 60)
    print("TESTING FUZZY MATCHING FOR RESTAURANT TYPOS")
    print("=" * 60)
    
    # Test cases with typos
    test_cases = [
        "give me restaurnts in beyoglu",
        "show me restaurnt near taksim", 
        "find restarants in kadikoy",
        "best restrants in sultanahmet",
        "restaurents in galata"
    ]
    
    keywords = create_fuzzy_keywords()
    print(f"\nRestaurant keywords: {keywords['restaurants']}")
    print()
    
    for test_input in test_cases:
        print(f"Input: '{test_input}'")
        
        # Test typo correction
        corrected = correct_typos(test_input)
        print(f"Corrected: '{corrected}'")
        print("-" * 40)

def test_individual_words():
    """Test individual word matching"""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL WORD MATCHING")
    print("=" * 60)
    
    keywords = create_fuzzy_keywords()
    typo_words = ['restaurnts', 'restaurnt', 'restarants', 'restrants', 'restaurents']
    
    for word in typo_words:
        print(f"\nTesting word: '{word}'")
        
        # Test against restaurant keywords
        match = process.extractOne(word, keywords['restaurants'])
        print(f"Best match: {match}")
        
        # Test threshold levels
        for threshold in [60, 70, 80, 90]:
            if match and match[1] >= threshold:
                print(f"  ✅ Passes threshold {threshold}")
            else:
                print(f"  ❌ Fails threshold {threshold}")

if __name__ == "__main__":
    test_fuzzy_matching()
    test_individual_words()
