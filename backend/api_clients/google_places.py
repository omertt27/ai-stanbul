
import os
import requests

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def search_restaurants(location, query):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    # Always search in Istanbul, Turkey
    params = {
        "query": f"restaurant {query} in Istanbul, Turkey",
        "key": GOOGLE_MAPS_API_KEY
    }
    response = requests.get(url, params=params)
    return response.json()
