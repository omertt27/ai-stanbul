
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def search_restaurants(location, query):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    
    # Get API key at runtime
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    if not google_maps_api_key:
        print("ERROR: GOOGLE_MAPS_API_KEY not found in environment variables")
        return {"results": [], "status": "NO_API_KEY"}
    
    # Always search in Istanbul, Turkey
    params = {
        "query": f"restaurant {query} in Istanbul, Turkey",
        "key": google_maps_api_key
    }
    
    print(f"Making Google Places API request: {params['query']}")
    response = requests.get(url, params=params)
    result = response.json()
    
    print(f"Google Places API response status: {result.get('status')}")
    print(f"Found {len(result.get('results', []))} results")
    
    return result
