#!/usr/bin/env python3
import requests
import json

def test_chatbot():
    url = "http://localhost:8001/ai"
    
    # Test restaurant recommendation query
    test_data = {
        "query": "Can you recommend some good restaurants in Beyoğlu?",
        "user_id": "test_user"
    }
    
    try:
        print("Testing chatbot endpoint...")
        print(f"URL: {url}")
        print(f"Request: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"\nResponse Data:")
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
        else:
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - is the server running on port 8001?")
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_chatbot()
