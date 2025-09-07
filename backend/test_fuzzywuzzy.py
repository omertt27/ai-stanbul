#!/usr/bin/env python3
"""
Test script to verify fuzzywuzzy functionality in the backend
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fuzzywuzzy_import():
    """Test if fuzzywuzzy can be imported correctly"""
    print("🧪 Testing fuzzywuzzy import...")
    try:
        from fuzzywuzzy import fuzz, process
        print("✅ fuzzywuzzy imported successfully")
        return True, fuzz, process
    except ImportError as e:
        print(f"❌ Failed to import fuzzywuzzy: {e}")
        return False, None, None

def test_fuzzywuzzy_functionality(fuzz, process):
    """Test fuzzywuzzy core functionality"""
    print("\n🧪 Testing fuzzywuzzy functionality...")
    
    # Test basic ratio
    ratio = fuzz.ratio("restaurant", "restaurnt")
    print(f"✅ fuzz.ratio('restaurant', 'restaurnt') = {ratio}")
    
    # Test token sort ratio
    token_ratio = fuzz.token_sort_ratio("good restaurants in istanbul", "restaurants istanbul good")
    print(f"✅ fuzz.token_sort_ratio = {token_ratio}")
    
    # Test process.extractOne
    keywords = ["restaurant", "museum", "hotel", "attraction", "nightlife"]
    match = process.extractOne("restaurnt", keywords)
    print(f"✅ process.extractOne('restaurnt', keywords) = {match}")
    
    # Test with actual typos
    test_typos = [
        ("restaurnts", ["restaurant", "restaurants", "museum", "hotel"]),
        ("musem", ["restaurant", "museum", "hotel", "attraction"]),
        ("hotal", ["restaurant", "museum", "hotel", "attraction"]),
        ("istambul", ["istanbul", "ankara", "izmir", "bodrum"])
    ]
    
    print("\n🔍 Testing typo correction scenarios:")
    for typo, candidates in test_typos:
        match = process.extractOne(typo, candidates)
        if match:
            print(f"  '{typo}' -> '{match[0]}' (confidence: {match[1]}%)")
        else:
            print(f"  '{typo}' -> No match found")
    
    return True

def test_backend_typo_correction():
    """Test the backend's typo correction function"""
    print("\n🧪 Testing backend typo correction function...")
    
    try:
        # Import the main module to test typo correction
        from main import correct_typos
        
        test_cases = [
            "show me restaurnts in istanbul",
            "best musems to visit",
            "hotals near taksim",
            "what are the best restarants",
            "good nightlife spots"
        ]
        
        for test_case in test_cases:
            corrected = correct_typos(test_case)
            print(f"  Input: '{test_case}'")
            print(f"  Output: '{corrected}'")
            print()
        
        return True
    except Exception as e:
        print(f"❌ Error testing backend typo correction: {e}")
        return False

def main():
    print("🚀 Starting fuzzywuzzy functionality test\n")
    
    # Test 1: Import
    success, fuzz, process = test_fuzzywuzzy_import()
    if not success:
        print("\n❌ fuzzywuzzy import failed. Attempting to install...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy==0.18.0", "python-levenshtein==0.20.9"])
            print("✅ fuzzywuzzy installed successfully")
            success, fuzz, process = test_fuzzywuzzy_import()
        except Exception as e:
            print(f"❌ Failed to install fuzzywuzzy: {e}")
            return False
    
    # Test 2: Functionality
    if success and fuzz and process:
        test_fuzzywuzzy_functionality(fuzz, process)
    
    # Test 3: Backend integration
    test_backend_typo_correction()
    
    print("\n🎉 fuzzywuzzy test completed!")
    return True

if __name__ == "__main__":
    main()
