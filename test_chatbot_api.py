#!/usr/bin/env python3

"""
Test the actual chatbot endpoint with restaurant typos
"""

import requests
import json

def test_restaurant_typo_queries():
    """Test restaurant queries with typos against the actual API"""
    
    base_url = "http://localhost:8001"
    
    test_queries = [
        "give me restaurnts in beyoglu",
        "show me restaurnt near taksim", 
        "find restarants in kadikoy",
        "best restrants in sultanahmet"
    ]
    
    print("=" * 60)
    print("TESTING CHATBOT API WITH RESTAURANT TYPOS")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🧪 Testing: '{query}'")
        print("-" * 40)
        
        try:
            response = requests.post(
                f"{base_url}/ai",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status: {response.status_code}")
                print(f"📝 Response: {data.get('message', 'No message field')[:200]}...")
                
                # Check if it contains restaurant data
                response_text = data.get('message', '').lower()
                if 'restaurant' in response_text or 'dining' in response_text or 'cuisine' in response_text:
                    print("🍽️  Contains restaurant-related content: ✅")
                else:
                    print("❌ Does NOT contain restaurant-related content")
                    
            else:
                print(f"❌ Error: Status {response.status_code}")
                print(f"Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the server is running on localhost:8001")
        except requests.exceptions.Timeout:
            print("❌ Timeout: Server took too long to respond")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_restaurant_typo_queries()
