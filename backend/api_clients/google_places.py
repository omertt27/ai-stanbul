
import requests

GOOGLE_API_KEY = "R3g!on9$Plx"
BASE_URL = "https://maps.googleapis.com/maps/api/place/"

def search_restaurants(location, radius=1000, type="restaurant"):
    url = f"{BASE_URL}nearbysearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": location,   # "lat,lng"
        "radius": radius,
        "type": type
    }
    response = requests.get(url, params=params)
    return response.json()
