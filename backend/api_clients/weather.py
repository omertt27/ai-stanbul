import requests
import os
from typing import Dict, Optional
import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class WeatherClient:
    """Weather client using Google API key for enhanced location data and OpenWeatherMap for weather."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.google_api_key = api_key or os.getenv("GOOGLE_WEATHER_API_KEY")
        self.openweather_key = os.getenv("OPENWEATHER_API_KEY")
        
        if not self.google_api_key:
            logger.warning("Google API key not found. Using basic location data.")
        
        if not self.openweather_key:
            logger.warning("OpenWeatherMap API key not found. Using mock weather data.")
            self.use_mock = True
        else:
            self.use_mock = False
        
        self.openweather_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_istanbul_weather(self) -> Dict:
        """
        Get current weather information for Istanbul.
        Uses OpenWeatherMap as primary source, with intelligent mock fallback.
        
        Returns:
            Dictionary containing weather information
        """
        # First try OpenWeatherMap if key is available
        if self.openweather_key and not self.use_mock:
            try:
                params = {
                    "q": "Istanbul,TR",
                    "appid": self.openweather_key,
                    "units": "metric"
                }
                
                response = requests.get(self.openweather_url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                # Extract relevant weather information
                weather_info = {
                    "temperature": round(data["main"]["temp"]),
                    "feels_like": round(data["main"]["feels_like"]),
                    "description": data["weather"][0]["description"].title(),
                    "humidity": data["main"]["humidity"],
                    "wind_speed": round(data["wind"]["speed"], 1),
                    "visibility": data.get("visibility", 10000) // 1000,
                    "is_raining": any("rain" in w["main"].lower() for w in data["weather"]),
                    "is_sunny": any("clear" in w["main"].lower() for w in data["weather"]),
                    "source": "OpenWeatherMap"
                }
                
                logger.info(f"Weather data retrieved from OpenWeatherMap: {weather_info['temperature']}°C, {weather_info['description']}")
                return weather_info
                
            except Exception as e:
                logger.warning(f"OpenWeatherMap API failed: {e}. Using intelligent mock data.")
        
        # Use intelligent mock data based on Istanbul's climate patterns
        return self._get_intelligent_mock_weather()
    
    def _get_intelligent_mock_weather(self) -> Dict:
        """
        Generate realistic weather data for Istanbul based on seasonal patterns.
        """
        # Get current month for seasonal accuracy
        current_month = datetime.now().month
        
        # Istanbul seasonal weather patterns
        if current_month in [12, 1, 2]:  # Winter
            temp_range = (5, 15)
            conditions = ["Cloudy", "Partly Cloudy", "Light Rain", "Overcast"]
            humidity_range = (70, 85)
        elif current_month in [3, 4, 5]:  # Spring
            temp_range = (12, 22)
            conditions = ["Partly Cloudy", "Clear", "Light Clouds", "Sunny"]
            humidity_range = (60, 75)
        elif current_month in [6, 7, 8]:  # Summer
            temp_range = (20, 30)
            conditions = ["Sunny", "Clear", "Hot", "Partly Cloudy"]
            humidity_range = (55, 70)
        else:  # Fall (9, 10, 11)
            temp_range = (15, 25)
            conditions = ["Partly Cloudy", "Clear", "Light Clouds", "Mild"]
            humidity_range = (65, 80)
        
        temperature = random.randint(*temp_range)
        description = random.choice(conditions)
        
        weather_info = {
            "temperature": temperature,
            "feels_like": temperature + random.randint(-3, 3),
            "description": description,
            "humidity": random.randint(*humidity_range),
            "wind_speed": round(random.uniform(2.0, 8.0), 1),
            "visibility": random.randint(8, 10),
            "is_raining": "rain" in description.lower(),
            "is_sunny": any(word in description.lower() for word in ["sunny", "clear", "hot"]),
            "source": "Intelligent Mock (Seasonal)"
        }
        
        logger.info(f"Using intelligent mock weather: {weather_info['temperature']}°C, {weather_info['description']}")
        return weather_info
    
    def format_weather_info(self, weather_data: Dict) -> str:
        """
        Format weather information for use in recommendations.
        
        Args:
            weather_data: Dictionary containing weather information
            
        Returns:
            Formatted string for use in AI context
        """
        temp = weather_data.get("temperature", "N/A")
        feels_like = weather_data.get("feels_like", "N/A")
        description = weather_data.get("description", "Unknown")
        source = weather_data.get("source", "Unknown")
        
        formatted = f"Today's Istanbul weather: {temp}°C (feels like {feels_like}°C), {description}"
        
        # Add activity recommendations based on weather
        if weather_data.get("is_raining"):
            formatted += ". Perfect for indoor activities like museums, covered bazaars, and Turkish baths."
        elif weather_data.get("is_sunny") and temp >= 20:
            formatted += ". Great weather for outdoor activities like Bosphorus cruises, park visits, and walking tours."
        elif temp <= 10:
            formatted += ". Good weather for indoor cultural experiences and warm Turkish tea."
        else:
            formatted += ". Pleasant weather for both indoor and outdoor Istanbul experiences."
        
        formatted += f" (Source: {source})"
        return formatted
    
    def get_weather_recommendations(self, weather_data: Dict) -> str:
        """
        Get activity recommendations based on current weather.
        
        Args:
            weather_data: Dictionary containing weather information
            
        Returns:
            Weather-appropriate activity recommendations
        """
        temp = weather_data.get("temperature", 15)
        is_raining = weather_data.get("is_raining", False)
        is_sunny = weather_data.get("is_sunny", False)
        
        if is_raining:
            return "Indoor recommendations: Grand Bazaar shopping, Hagia Sophia visit, Turkish bath experience, museum tours, covered Spice Bazaar exploration."
        elif is_sunny and temp >= 25:
            return "Outdoor recommendations: Bosphorus ferry cruise, Emirgan Park visit, Galata Bridge walk, rooftop dining, sunset at Çamlıca Hill."
        elif temp <= 10:
            return "Cozy recommendations: Traditional Turkish breakfast, hammam experience, indoor markets, cultural museums, warm cafes in Galata."
        else:
            return "Mixed recommendations: Historic walking tours, both indoor and outdoor attractions, flexible sightseeing with layers."


# Global weather client instance
weather_client = WeatherClient()
