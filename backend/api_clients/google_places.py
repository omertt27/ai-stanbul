import requests

GOOGLE_API_KEY = "R3g!on9$Plx"
PLACES_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"

def fetch_restaurants(city: str, query: str = "restaurant", limit: int = 20):
    params = {
        "query": f"{query} in {city}",
        "key": GOOGLE_API_KEY
    }
    response = requests.get(PLACES_SEARCH_URL, params=params)
    response.raise_for_status()
    data = response.json()

    restaurants = []
    for item in data.get("results", [])[:limit]:
        restaurant = {
            "name": item.get("name"),
            "cuisine": query,  # optionally parse types for more detailed cuisine
            "location": item.get("formatted_address"),
            "rating": item.get("rating"),
            "source": "Google Places"
        }
        restaurants.append(restaurant)
    return restaurants