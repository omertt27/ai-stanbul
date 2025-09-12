import requests
import os
from typing import Dict, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GoogleWeatherClient:
    """Google Weather API client for fetching Istanbul weather information."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_WEATHER_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            logger.warning("Google Weather API key not found. Using mock weather data.")
            self.use_mock = True
        else:
            self.use_mock = False
        # Google Weather API endpoint (part of Google Maps Platform)
        self.base_url = "https://maps.googleapis.com/maps/api/weather/current"
    
    def get_istanbul_weather(self) -> Dict:
        """
        Get current weather information for Istanbul using Google Weather API.
        
        Returns:
            Dictionary containing weather information
        """
        if self.use_mock:
            return self._get_mock_weather()
        
        try:
            # Istanbul coordinates
            params = {
                "location": "41.0082,28.9784",  # Istanbul lat,lng
                "key": self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Parse Google Weather API response
            weather_info = {
                "temperature": round(data.get("current", {}).get("temperature", 18)),
                "feels_like": round(data.get("current", {}).get("feelsLike", 16)),
                "description": data.get("current", {}).get("condition", "Partly Cloudy").title(),
                "humidity": data.get("current", {}).get("humidity", 65),
                "wind_speed": round(data.get("current", {}).get("windSpeed", 15), 1),
                "visibility": data.get("current", {}).get("visibility", 10),
                "is_raining": "rain" in data.get("current", {}).get("condition", "").lower(),
                "cloud_cover": data.get("current", {}).get("cloudCover", 50)
            }
            
            return weather_info
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Google Weather API request failed: {e}")
            return self._get_mock_weather()
        except Exception as e:
            logger.error(f"Error parsing Google Weather API response: {e}")
            return self._get_mock_weather()
    
    def _get_mock_weather(self) -> Dict:
        """Return mock weather data when API is unavailable"""
        return {
            "temperature": 18,
            "feels_like": 16,
            "description": "Partly Cloudy",
            "humidity": 65,
            "wind_speed": 15.0,
            "visibility": 10,
            "is_raining": False,
            "cloud_cover": 50
        }
    
    def format_weather_info(self, weather_data: Dict) -> str:
        """
        Format weather information for use in GPT context.
        
        Args:
            weather_data: Weather information dictionary
            
        Returns:
            Formatted weather string for GPT context
        """
        temp = weather_data.get("temperature", "N/A")
        feels_like = weather_data.get("feels_like", "N/A")
        description = weather_data.get("description", "Unknown")
        humidity = weather_data.get("humidity", "N/A")
        wind_speed = weather_data.get("wind_speed", "N/A")
        is_raining = weather_data.get("is_raining", False)
        
        rain_status = "Currently raining" if is_raining else "No rain"
        
        weather_summary = (
            f"Today's Istanbul weather: {temp}°C (feels like {feels_like}°C), {description}. "
            f"Humidity: {humidity}%, Wind: {wind_speed} km/h. {rain_status}."
        )
        
        return weather_summary

class WeatherClient:
    """Enhanced weather client that supports both OpenWeatherMap and Google Weather APIs."""
    
    def __init__(self, api_key: Optional[str] = None, provider: str = "auto"):
        """
        Initialize weather client with fallback support.
        
        Args:
            api_key: API key for the weather service
            provider: "openweather", "google", or "auto" for automatic selection
        """
        self.provider = provider
        self.openweather_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        self.google_key = api_key or os.getenv("GOOGLE_WEATHER_API_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
        
        # Auto-select provider based on available keys
        if provider == "auto":
            if self.google_key:
                self.provider = "google"
                self.api_key = self.google_key
            elif self.openweather_key:
                self.provider = "openweather"
                self.api_key = self.openweather_key
            else:
                self.provider = "mock"
                self.api_key = None
        elif provider == "google":
            self.api_key = self.google_key
        elif provider == "openweather":
            self.api_key = self.openweather_key
        else:
            self.provider = "mock"
            self.api_key = None
        
        # Set up the appropriate client
        if self.provider == "google" and self.api_key:
            self.client = GoogleWeatherClient(self.api_key)
        elif self.provider == "openweather" and self.api_key:
            self.client = OpenWeatherMapClient(self.api_key)
        else:
            logger.warning("No weather API key found. Using mock weather data.")
            self.client = MockWeatherClient()
    
    def get_istanbul_weather(self) -> Dict:
        """Get weather data using the configured provider"""
        return self.client.get_istanbul_weather()
    
    def format_weather_info(self, weather_data: Dict) -> str:
        """Format weather information for GPT context"""
        return self.client.format_weather_info(weather_data)

class OpenWeatherMapClient:
    """OpenWeatherMap API client for fetching Istanbul weather information."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_istanbul_weather(self) -> Dict:
        """Get current weather information for Istanbul using OpenWeatherMap API."""
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
                "cloud_cover": data["clouds"]["all"]
            }
            
            return weather_info
            
        except Exception as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            return MockWeatherClient().get_istanbul_weather()
    
    def format_weather_info(self, weather_data: Dict) -> str:
        """Format weather information for use in GPT context."""
        temp = weather_data.get("temperature", "N/A")
        feels_like = weather_data.get("feels_like", "N/A")
        description = weather_data.get("description", "Unknown")
        humidity = weather_data.get("humidity", "N/A")
        wind_speed = weather_data.get("wind_speed", "N/A")
        is_raining = weather_data.get("is_raining", False)
        
        rain_status = "Currently raining" if is_raining else "No rain"
        
        weather_summary = (
            f"Today's Istanbul weather: {temp}°C (feels like {feels_like}°C), {description}. "
            f"Humidity: {humidity}%, Wind: {wind_speed} km/h. {rain_status}."
        )
        
        return weather_summary

class MockWeatherClient:
    """Mock weather client for when no API key is available"""
    
    def get_istanbul_weather(self) -> Dict:
        """Return realistic mock weather data for Istanbul"""
        return {
            "temperature": 18,
            "feels_like": 16,
            "description": "Partly Cloudy",
            "humidity": 65,
            "wind_speed": 15.0,
            "visibility": 10,
            "is_raining": False,
            "cloud_cover": 50
        }
    
    def format_weather_info(self, weather_data: Dict) -> str:
        """Format mock weather information"""
        temp = weather_data.get("temperature", 18)
        feels_like = weather_data.get("feels_like", 16)
        description = weather_data.get("description", "Partly Cloudy")
        humidity = weather_data.get("humidity", 65)
        wind_speed = weather_data.get("wind_speed", 15.0)
        
        weather_summary = (
            f"Today's Istanbul weather: {temp}°C (feels like {feels_like}°C), {description}. "
            f"Humidity: {humidity}%, Wind: {wind_speed} km/h. No rain."
        )
        
        return weather_summary
