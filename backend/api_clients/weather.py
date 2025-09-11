import requests
import os
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class WeatherClient:
    """OpenWeatherMap API client for fetching Istanbul weather information."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            # Use a mock weather response if no API key is available
            logger.warning("OpenWeatherMap API key not found. Using mock weather data.")
            self.use_mock = True
        else:
            self.use_mock = False
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_istanbul_weather(self) -> Dict:
        """
        Get current weather information for Istanbul.
        
        Returns:
            Dictionary containing weather information
        """
        if self.use_mock:
            return self._get_mock_weather()
        
        try:
            params = {
                "q": "Istanbul,TR",
                "appid": self.api_key,
                "units": "metric"  # Celsius
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant weather information
            weather_info = {
                "temperature": round(data["main"]["temp"]),
                "feels_like": round(data["main"]["feels_like"]),
                "description": data["weather"][0]["description"].title(),
                "humidity": data["main"]["humidity"],
                "wind_speed": round(data["wind"]["speed"], 1),
                "visibility": data.get("visibility", 10000) // 1000,  # Convert to km
                "is_raining": any("rain" in w["main"].lower() for w in data["weather"]),
                "is_cloudy": any("cloud" in w["main"].lower() for w in data["weather"]),
                "condition": data["weather"][0]["main"].lower()
            }
            
            return {
                "status": "success",
                "weather": weather_info,
                "city": "Istanbul"
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._get_mock_weather()
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}")
            return self._get_mock_weather()
    
    def _get_mock_weather(self) -> Dict:
        """Return mock weather data when API is unavailable."""
        return {
            "status": "mock",
            "weather": {
                "temperature": 18,
                "feels_like": 16,
                "description": "Partly Cloudy",
                "humidity": 65,
                "wind_speed": 12.0,
                "visibility": 8,
                "is_raining": False,
                "is_cloudy": True,
                "condition": "clouds"
            },
            "city": "Istanbul"
        }
    
    def format_weather_info(self, weather_data: Dict) -> str:
        """
        Format current daily weather information for inclusion in chatbot responses.
        
        Args:
            weather_data: Weather data from get_istanbul_weather()
            
        Returns:
            Formatted daily weather string for responses
        """
        if weather_data["status"] not in ["success", "mock"]:
            return "Current weather information is unavailable."
        
        weather = weather_data["weather"]
        temp = weather["temperature"]
        feels_like = weather["feels_like"]
        description = weather["description"]
        humidity = weather["humidity"]
        wind_speed = weather["wind_speed"]
        
        # Create today's weather description
        weather_text = f"Today's Istanbul weather: {temp}°C (feels like {feels_like}°C), {description}"
        
        # Add relevant daily activity recommendations based on conditions
        if weather["is_raining"]:
            weather_text += ". Perfect day for indoor attractions like museums, covered bazaars, or traditional Turkish baths (hamams)"
        elif weather["condition"] == "clear" and temp >= 18:
            weather_text += ". Excellent weather for outdoor sightseeing, Bosphorus ferry rides, and walking tours"
        elif temp >= 25:
            weather_text += ". Great weather for outdoor dining, rooftop bars, and waterfront activities"
        elif temp <= 10:
            weather_text += ". Bundle up warmly for outdoor activities - consider cozy indoor experiences like traditional tea houses"
        elif weather["is_cloudy"]:
            weather_text += ". Pleasant weather for exploring neighborhoods and outdoor markets"
        else:
            weather_text += ". Good weather for most outdoor and indoor activities"
        
        # Add humidity/wind context if relevant
        if humidity > 80:
            weather_text += ". High humidity today - stay hydrated"
        elif wind_speed > 15:
            weather_text += ". Breezy conditions - perfect for Bosphorus views"
        
        return weather_text

# Global weather client instance
weather_client = WeatherClient()
