#!/usr/bin/env python3
"""
Quick Analysis of Neighborhood Guide Responses
"""

import requests
import json

def test_neighborhood_queries():
    """Test specific neighborhood queries and analyze responses"""
    
    endpoint = "http://localhost:8000/ai/chat"
    
    test_queries = [
        "What's the character and atmosphere of Sultanahmet district?",
        "Describe the vibe and personality of Beyoğlu neighborhood", 
        "Best time to visit Kadıköy district?",
        "Hidden gems in Galata area?",
        "How to get around Sultanahmet using public transport?"
    ]
    
    print("🔍 NEIGHBORHOOD GUIDES ANALYSIS")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔸 Query {i}: {query}")
        
        try:
            response = requests.post(
                endpoint,
                json={"user_input": query},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                ai_response = data.get('response', '')
                
                # Quick analysis
                word_count = len(ai_response.split())
                has_practical_info = any(term in ai_response.lower() for term in ['hours', 'transport', 'metro', 'tram', 'ferry', 'walk'])
                has_cultural_context = any(term in ai_response.lower() for term in ['culture', 'character', 'atmosphere', 'authentic', 'local'])
                has_specific_details = any(term in ai_response.lower() for term in ['district', 'neighborhood', 'area', 'street', 'square'])
                
                print(f"   📝 Length: {word_count} words")
                print(f"   🚇 Practical Info: {'Yes' if has_practical_info else 'No'}")
                print(f"   🏛️  Cultural Context: {'Yes' if has_cultural_context else 'No'}")
                print(f"   📍 Specific Details: {'Yes' if has_specific_details else 'No'}")
                
                # Show first 150 characters
                preview = ai_response[:150] + "..." if len(ai_response) > 150 else ai_response
                print(f"   📄 Preview: {preview}")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_neighborhood_queries()
