#!/usr/bin/env python3
"""
Test script for the new advanced AI endpoints
"""
import requests
import json
import os
from pathlib import Path

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_real_time_data():
    """Test the real-time data endpoint"""
    print("🧪 Testing real-time data endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ai/real-time-data", params={
            "include_events": True,
            "include_crowds": True,
            "include_traffic": False
        })
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Real-time data endpoint working")
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Real-time data endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing real-time data: {e}")
    print()

def test_predictive_analytics():
    """Test the predictive analytics endpoint"""
    print("🧪 Testing predictive analytics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ai/predictive-analytics", params={
            "locations": "hagia sophia,blue mosque,grand bazaar",
            "user_preferences": json.dumps({"interests": ["history", "culture"], "budget": "medium"})
        })
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Predictive analytics endpoint working")
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Predictive analytics endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing predictive analytics: {e}")
    print()

def test_enhanced_recommendations():
    """Test the enhanced recommendations endpoint"""
    print("🧪 Testing enhanced recommendations endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/ai/enhanced-recommendations", params={
            "query": "I want to visit historical places in Istanbul",
            "include_realtime": True,
            "include_predictions": True,
            "session_id": "test_session_123"
        })
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced recommendations endpoint working")
            print(f"Response: {json.dumps(data, indent=2)}")
        else:
            print(f"❌ Enhanced recommendations endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Error testing enhanced recommendations: {e}")
    print()

def test_image_analysis():
    """Test the image analysis endpoint with a sample image"""
    print("🧪 Testing image analysis endpoint...")
    try:
        # Create a simple test image (1x1 pixel PNG)
        import io
        from PIL import Image
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"image": ("test.png", img_bytes, "image/png")}
        data = {"context": "Test image for location analysis"}
        
        response = requests.post(f"{BASE_URL}/ai/analyze-image", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image analysis endpoint working")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Image analysis endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except ImportError:
        print("⚠️ PIL not available, skipping image analysis test")
    except Exception as e:
        print(f"❌ Error testing image analysis: {e}")
    print()

def test_menu_analysis():
    """Test the menu analysis endpoint with a sample image"""
    print("🧪 Testing menu analysis endpoint...")
    try:
        # Create a simple test image (1x1 pixel PNG)
        import io
        from PIL import Image
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        files = {"image": ("menu.png", img_bytes, "image/png")}
        data = {"dietary_restrictions": "vegetarian"}
        
        response = requests.post(f"{BASE_URL}/ai/analyze-menu", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Menu analysis endpoint working")
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Menu analysis endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except ImportError:
        print("⚠️ PIL not available, skipping menu analysis test")
    except Exception as e:
        print(f"❌ Error testing menu analysis: {e}")
    print()

def main():
    """Run all tests"""
    print("🚀 Testing Advanced AI Endpoints")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server not running at http://localhost:8000")
            print("Please start the server first with: python start-server.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Server not running at http://localhost:8000")
        print("Please start the server first with: python start-server.py")
        return
    
    print("✅ Server is running")
    print()
    
    # Run all tests
    test_real_time_data()
    test_predictive_analytics()
    test_enhanced_recommendations()
    test_image_analysis()
    test_menu_analysis()
    
    print("🎉 All endpoint tests completed!")

if __name__ == "__main__":
    main()
