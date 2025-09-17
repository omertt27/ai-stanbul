#!/usr/bin/env python3
"""
Interactive Transportation Query Tester
Quick testing tool for individual transportation queries
"""

import requests
import json
import sys

def test_query(query: str):
    """Test a single query and display formatted results"""
    
    url = "http://localhost:8000/ai"
    payload = {"query": query, "session_id": "interactive_test"}
    
    try:
        print(f"🔍 Testing: {query}")
        print("=" * 60)
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", "")
            
            print("✅ SUCCESS")
            print(f"📱 Response ({len(message)} chars):")
            print("-" * 40)
            print(message)
            print("-" * 40)
            
            # Check for transportation keywords
            transport_keywords = ["ferry", "metro", "bus", "taxi", "transport", "route", "station", "terminal"]
            found_keywords = [kw for kw in transport_keywords if kw.lower() in message.lower()]
            
            if found_keywords:
                print(f"🎯 Transport keywords found: {', '.join(found_keywords)}")
            else:
                print("⚠️  No transportation keywords detected")
                
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
    
    print("\n" + "=" * 60 + "\n")

def interactive_mode():
    """Interactive testing mode"""
    
    print("🚇 Istanbul Transportation Chatbot Tester")
    print("Type 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            query = input("Enter transportation query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not query:
                print("Please enter a query!")
                continue
                
            test_query(query)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            break

def run_quick_tests():
    """Run a few quick predefined tests"""
    
    quick_tests = [
        "how can I go kadikoy from beyoglu",
        "transportation in istanbul",
        "metro to airport",
        "ferry routes",
        "where to buy istanbulkart"
    ]
    
    print("🚀 Running Quick Transportation Tests")
    print("=" * 50)
    
    for query in quick_tests:
        test_query(query)

if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding correctly")
            sys.exit(1)
    except:
        print("❌ Server not accessible at http://localhost:8000")
        print("Please start the backend server first:")
        print("cd /Users/omer/Desktop/ai-stanbul/backend && uvicorn main:app --reload --port 8000")
        sys.exit(1)
    
    print("✅ Server is running\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            run_quick_tests()
        elif sys.argv[1] == "--interactive":
            interactive_mode()
        else:
            # Test the provided query
            test_query(" ".join(sys.argv[1:]))
    else:
        # Default to interactive mode
        interactive_mode()
