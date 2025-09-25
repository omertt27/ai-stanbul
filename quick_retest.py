#!/usr/bin/env python3
"""
Quick retest of previously failed cases to verify improvements
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Test previously failed cases
failed_cases = [
    "Tell me about Turkish breakfast culture",
    "Where can I eat with a Bosphorus view?",
    "What's the tipping culture in Istanbul restaurants?",
    "How do I get from Sultanahmet to Galata Tower?",
    "What's the difference between metro and metrobus?",
    "What's the best transport app for Istanbul?",
    "Are there any food tours in Istanbul?",
]

def test_query(query):
    try:
        response = requests.post(
            f"{BASE_URL}/ai",
            json={"user_input": query, "session_id": f"retest_{hash(query)}"},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response")
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error: {e}"

def analyze_response(query, response):
    # Define expected keywords for each query type
    keywords_map = {
        "breakfast": ["kahvaltı", "cheese", "olives", "tea", "bread"],
        "bosphorus": ["bosphorus", "view", "restaurant", "waterfront", "ortaköy"],
        "tipping": ["tip", "10%", "15%", "service", "cash"],
        "sultanahmet galata": ["tram", "metro", "walk", "karaköy", "bridge"],
        "metro metrobus": ["metro", "metrobus", "underground", "BRT", "difference"],
        "transport app": ["app", "moovit", "google", "bitaksi", "mobile"],
        "food tours": ["tour", "food", "guided", "culinary", "tasting"],
    }
    
    # Find which keyword set applies
    query_lower = query.lower()
    response_lower = response.lower()
    
    for key, keywords in keywords_map.items():
        if any(word in query_lower for word in key.split()):
            found = sum(1 for kw in keywords if kw in response_lower)
            total = len(keywords)
            coverage = found / total * 100
            return coverage, found, total, keywords
    
    return 0, 0, 0, []

print("🔍 Testing Previously Failed Cases After Improvements")
print("=" * 60)

for i, query in enumerate(failed_cases, 1):
    print(f"\n[{i}] {query}")
    response = test_query(query)
    
    if "Error:" in response:
        print(f"    ❌ {response}")
        continue
    
    coverage, found, total, expected = analyze_response(query, response)
    
    if coverage >= 60:
        print(f"    ✅ IMPROVED - Coverage: {coverage:.1f}% ({found}/{total} keywords)")
    else:
        print(f"    ❌ Still issues - Coverage: {coverage:.1f}% ({found}/{total} keywords)")
    
    print(f"    Response preview: {response[:100]}...")
    
    if found < total:
        missing = [kw for kw in expected if kw not in response.lower()]
        print(f"    Missing keywords: {', '.join(missing[:3])}...")

print("\n" + "=" * 60)
