#!/usr/bin/env python3
"""
Real API Verification Script
This script helps you verify and configure your real API keys.
"""

import os
import sys
import requests
from datetime import datetime

def check_google_places_api():
    """Check if Google Places API key is working"""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    
    if not api_key or api_key == "your_google_places_key_here":
        return {
            "status": "missing",
            "message": "Google Places API key not set or still has placeholder value",
            "has_real_key": False
        }
    
    # Test the API with a simple request
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": "restaurant in Istanbul",
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "OK":
                return {
                    "status": "working",
                    "message": f"✅ Google Places API working! Found {len(data.get('results', []))} restaurants",
                    "has_real_key": True,
                    "sample_result": data.get('results', [{}])[0].get('name', 'N/A') if data.get('results') else 'No results'
                }
            else:
                return {
                    "status": "error",
                    "message": f"❌ Google Places API error: {data.get('status')} - {data.get('error_message', 'Unknown error')}",
                    "has_real_key": True
                }
        else:
            return {
                "status": "error", 
                "message": f"❌ HTTP error: {response.status_code}",
                "has_real_key": True
            }
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"❌ Request failed: {str(e)}",
            "has_real_key": True
        }

def check_openweather_api():
    """Check if OpenWeatherMap API key is working"""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    
    if not api_key or api_key == "your_openweather_key_here":
        return {
            "status": "missing",
            "message": "OpenWeatherMap API key not set or still has placeholder value",
            "has_real_key": False
        }
    
    # Test the API
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": "Istanbul,TR",
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            temp = data.get('main', {}).get('temp', 'N/A')
            description = data.get('weather', [{}])[0].get('description', 'N/A')
            return {
                "status": "working",
                "message": f"✅ OpenWeatherMap API working! Istanbul: {temp}°C, {description}",
                "has_real_key": True,
                "current_weather": f"{temp}°C, {description}"
            }
        else:
            return {
                "status": "error",
                "message": f"❌ HTTP error: {response.status_code}",
                "has_real_key": True
            }
    except requests.RequestException as e:
        return {
            "status": "error", 
            "message": f"❌ Request failed: {str(e)}",
            "has_real_key": True
        }

def main():
    print("🔍 Real API Verification Script")
    print("=" * 50)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n1. 🏪 Checking Google Places API...")
    google_result = check_google_places_api()
    print(f"   {google_result['message']}")
    if google_result['status'] == 'working':
        print(f"   Sample restaurant: {google_result.get('sample_result', 'N/A')}")
    
    print("\n2. 🌤️ Checking OpenWeatherMap API...")
    weather_result = check_openweather_api()
    print(f"   {weather_result['message']}")
    if weather_result['status'] == 'working':
        print(f"   Current Istanbul weather: {weather_result.get('current_weather', 'N/A')}")
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    google_working = google_result['status'] == 'working'
    weather_working = weather_result['status'] == 'working'
    
    if google_working and weather_working:
        print("🎉 EXCELLENT! Both APIs are working with real data!")
        print("\n✅ Your AI Istanbul is now using:")
        print("   • Real restaurant data from Google Places")
        print("   • Real weather data from OpenWeatherMap")
        print("   • Enhanced recommendations with live information")
        
        print("\n🚀 Next steps:")
        print("   1. Your backend is ready for real data")
        print("   2. Test with: python test_chatbot_quick.py")
        print("   3. Try queries like 'Turkish restaurants in Sultanahmet'")
        
    elif google_working or weather_working:
        print("⚡ PARTIAL SUCCESS! Some APIs working:")
        if google_working:
            print("   ✅ Google Places API: Working")
        else:
            print("   ❌ Google Places API: Not working")
        if weather_working:
            print("   ✅ OpenWeatherMap API: Working")
        else:
            print("   ❌ OpenWeatherMap API: Not working")
        
        print("\n🔧 Fix the non-working APIs for complete real data integration")
        
    else:
        print("❌ No real APIs detected. Using fallback mock data.")
        print("\n🔑 To add real API keys:")
        print("   1. Edit .env file")
        print("   2. Replace placeholder values with real keys:")
        print("      GOOGLE_PLACES_API_KEY=your_actual_google_key")
        print("      OPENWEATHERMAP_API_KEY=your_actual_weather_key")
        print("   3. Restart backend: python backend/main.py")
    
    print(f"\n⏰ Verification completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # Install dotenv if not available
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
        from dotenv import load_dotenv
    
    main()
