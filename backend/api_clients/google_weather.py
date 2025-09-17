import requests
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class GoogleWeatherClient:
    """Google Weather API client using Google Maps Weather services."""
    
    def __init__(self, api_key: Optional[str] = None):
        # Use Google Weather API key or fallback to Google Maps key
        self.api_key = (api_key or 
                       os.getenv("GOOGLE_WEATHER_API_KEY") or 
                       os.getenv("GOOGLE_MAPS_API_KEY") or 
                       os.getenv("GOOGLE_PLACES_API_KEY"))
        self.has_api_key = bool(self.api_key)
        self.use_real_apis = os.getenv("USE_REAL_APIS", "true").lower() == "true"
        
        # Google doesn't have a direct weather API, so we'll use Places + geocoding
        self.geocoding_url = "https://maps.googleapis.com/maps/api/geocode/json"
        self.places_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        
        # Simple cache for weather data
        self._cache = {}
        self.cache_duration = 15  # minutes
        
        if not self.has_api_key or not self.use_real_apis:
            logger.warning("Google Weather: Using fallback mode with enhanced mock data.")
        else:
            logger.info("Google Weather: Ready for location-based weather integration!")
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for weather request."""
        sorted_params = sorted(kwargs.items())
        return f"google_weather_{method}:{hash(str(sorted_params))}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached weather response if not expired."""
        if cache_key not in self._cache:
            return None
        
        cached_data, timestamp = self._cache[cache_key]
        if datetime.now() - timestamp < timedelta(minutes=self.cache_duration):
            return cached_data
        else:
            del self._cache[cache_key]
            return None
    
    def _cache_response(self, cache_key: str, data: Dict) -> None:
        """Cache weather response."""
        self._cache[cache_key] = (data, datetime.now())
    
    def get_location_coordinates(self, city: str = "Istanbul, Turkey") -> Dict:
        """Get coordinates for a city using Google Geocoding API."""
        cache_key = self._get_cache_key("geocoding", city=city)
        
        # Check cache first
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        if not self.has_api_key or not self.use_real_apis:
            return self._get_mock_coordinates(city)
        
        try:
            params = {
                'address': city,
                'key': self.api_key
            }
            
            response = requests.get(self.geocoding_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') == 'OK' and data.get('results'):
                location = data['results'][0]['geometry']['location']
                result = {
                    'lat': location['lat'],
                    'lng': location['lng'],
                    'formatted_address': data['results'][0]['formatted_address']
                }
                
                self._cache_response(cache_key, result)
                logger.info(f"📍 Google Geocoding: Found coordinates for {city}")
                return result
            else:
                logger.warning(f"Google Geocoding API error: {data.get('status', 'Unknown')}")
                return self._get_mock_coordinates(city)
                
        except Exception as e:
            logger.error(f"Google Geocoding API failed: {e}")
            return self._get_mock_coordinates(city)
    
    def get_current_weather(self, city: str = "Istanbul") -> Dict:
        """Get current weather conditions (using mock data since Google doesn't have direct weather API)."""
        cache_key = self._get_cache_key("current", city=city)
        
        # Check cache first
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        # Get location coordinates first
        location = self.get_location_coordinates(f"{city}, Turkey")
        
        # Since Google doesn't have a direct weather API, we'll use enhanced mock data
        # with location awareness
        result = self._get_enhanced_mock_weather(city, location)
        
        self._cache_response(cache_key, result)
        return result
    
    def get_forecast(self, city: str = "Istanbul", days: int = 5) -> Dict:
        """Get weather forecast (using enhanced mock data)."""
        cache_key = self._get_cache_key("forecast", city=city, days=days)
        
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            return cached_result
        
        location = self.get_location_coordinates(f"{city}, Turkey")
        result = self._get_enhanced_mock_forecast(city, location, days)
        
        self._cache_response(cache_key, result)
        return result
    
    def get_api_status(self) -> Dict:
        """Get Google Weather API status and capabilities."""
        return {
            'service': 'Google Weather (via Maps API)',
            'has_api_key': self.has_api_key,
            'using_real_data': self.has_api_key and self.use_real_apis,
            'capabilities': [
                'Location coordinates',
                'Enhanced mock weather data',
                'Activity recommendations',
                'Clothing advice',
                'Seasonal weather simulation'
            ],
            'note': 'Google Weather uses Maps API for geocoding + enhanced mock weather data'
        }
    
    def _get_mock_coordinates(self, city: str) -> Dict:
        """Mock coordinates for major cities."""
        coordinates = {
            "istanbul": {"lat": 41.0082, "lng": 28.9784, "formatted_address": "Istanbul, Turkey"},
            "ankara": {"lat": 39.9334, "lng": 32.8597, "formatted_address": "Ankara, Turkey"},
            "izmir": {"lat": 38.4192, "lng": 27.1287, "formatted_address": "Izmir, Turkey"},
        }
        
        city_key = city.lower().split(',')[0].strip()
        return coordinates.get(city_key, coordinates["istanbul"])
    
    def _get_enhanced_mock_weather(self, city: str, location: Dict) -> Dict:
        """Enhanced mock weather data with seasonal variation."""
        # Simulate realistic Istanbul weather based on current date
        now = datetime.now()
        month = now.month
        
        # Seasonal temperature ranges for Istanbul
        if month in [12, 1, 2]:  # Winter
            base_temp = 8
            condition = "scattered clouds"
            description = "Cool and partly cloudy"
        elif month in [3, 4, 5]:  # Spring
            base_temp = 18
            condition = "partly cloudy"
            description = "Pleasant spring weather"
        elif month in [6, 7, 8]:  # Summer
            base_temp = 28
            condition = "clear sky"
            description = "Warm and sunny"
        else:  # Fall
            base_temp = 15
            condition = "overcast clouds"
            description = "Mild autumn weather"
        
        # Add some daily variation
        temp_variation = (now.hour - 12) * 0.5  # Temperature varies by time of day
        current_temp = base_temp + temp_variation
        
        return {
            "coord": location,
            "weather": [
                {
                    "id": 801,
                    "main": condition.title(),
                    "description": description,
                    "icon": "02d"
                }
            ],
            "main": {
                "temp": round(current_temp, 1),
                "feels_like": round(current_temp - 1, 1),
                "temp_min": round(current_temp - 3, 1),
                "temp_max": round(current_temp + 3, 1),
                "pressure": 1013,
                "humidity": 65
            },
            "visibility": 10000,
            "wind": {
                "speed": 3.5,
                "deg": 230
            },
            "dt": int(now.timestamp()),
            "sys": {
                "country": "TR",
                "sunrise": int((now.replace(hour=6, minute=30)).timestamp()),
                "sunset": int((now.replace(hour=19, minute=45)).timestamp())
            },
            "name": city.title(),
            "activity_recommendations": self._get_weather_activity_recommendations(current_temp, condition),
            "clothing_advice": self._get_clothing_advice(current_temp, condition)
        }
    
    def _get_enhanced_mock_forecast(self, city: str, location: Dict, days: int) -> Dict:
        """Enhanced mock forecast data."""
        now = datetime.now()
        forecast_list = []
        
        for day in range(days):
            future_date = now + timedelta(days=day)
            
            # Simulate realistic temperature variation
            base_temp = 15 + day * 2  # Slight trend
            temp_with_season = base_temp + (10 if future_date.month in [6, 7, 8] else 0)
            
            forecast_list.append({
                "dt": int(future_date.timestamp()),
                "main": {
                    "temp": round(temp_with_season, 1),
                    "temp_min": round(temp_with_season - 2, 1),
                    "temp_max": round(temp_with_season + 4, 1),
                    "humidity": 60 + day * 2
                },
                "weather": [
                    {
                        "main": "Clear" if day % 2 == 0 else "Clouds",
                        "description": "sunny" if day % 2 == 0 else "partly cloudy",
                        "icon": "01d" if day % 2 == 0 else "02d"
                    }
                ],
                "dt_txt": future_date.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {
            "cod": "200",
            "message": 0,
            "cnt": len(forecast_list),
            "list": forecast_list,
            "city": {
                "id": 745044,
                "name": city.title(),
                "coord": location,
                "country": "TR"
            }
        }
    
    def _get_weather_activity_recommendations(self, temp: float, condition: str) -> List[str]:
        """Get activity recommendations based on weather."""
        recommendations = []
        
        if temp > 25:
            recommendations.extend([
                "Perfect weather for exploring outdoor attractions",
                "Great time for Bosphorus boat tour",
                "Ideal for walking in parks and gardens"
            ])
        elif temp > 15:
            recommendations.extend([
                "Comfortable weather for sightseeing",
                "Good time for walking tours",
                "Perfect for visiting outdoor markets"
            ])
        else:
            recommendations.extend([
                "Great weather for visiting museums",
                "Perfect time for indoor attractions",
                "Good for exploring covered bazaars"
            ])
        
        if "rain" in condition.lower():
            recommendations.append("Consider indoor activities like museums or shopping")
        elif "clear" in condition.lower() or "sun" in condition.lower():
            recommendations.append("Excellent visibility for photography and sightseeing")
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _get_clothing_advice(self, temp: float, condition: str) -> str:
        """Get clothing advice based on weather."""
        if temp > 25:
            advice = "Light clothing, shorts, t-shirt, comfortable walking shoes"
        elif temp > 15:
            advice = "Light layers, long pants, light jacket, comfortable shoes"
        else:
            advice = "Warm clothing, jacket or coat, closed shoes, layers"
        
        if "rain" in condition.lower():
            advice += ", umbrella or rain jacket"
        elif "wind" in condition.lower():
            advice += ", windproof jacket"
        
        return advice
    
    def format_weather_info(self, weather_data: Dict) -> str:
        """Format weather data for user display."""
        if not weather_data or "main" not in weather_data:
            return "Weather information not available"
        
        temp = weather_data["main"]["temp"]
        condition = weather_data["weather"][0]["description"]
        feels_like = weather_data["main"]["feels_like"]
        
        info = f"🌡️ {temp}°C ({condition}), feels like {feels_like}°C"
        
        if "activity_recommendations" in weather_data:
            recommendations = weather_data["activity_recommendations"][:2]
            info += f"\\n🎯 {', '.join(recommendations)}"
        
        if "clothing_advice" in weather_data:
            info += f"\\n👕 Clothing: {weather_data['clothing_advice']}"
        
        return info
