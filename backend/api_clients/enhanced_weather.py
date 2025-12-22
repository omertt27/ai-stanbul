import requests
import os
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

class EnhancedWeatherClient:
    """Enhanced Weather API client with real OpenWeatherMap integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHERMAP_API_KEY")
        self.has_api_key = bool(self.api_key)
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.use_real_apis = os.getenv("USE_REAL_APIS", "true").lower() == "true"
        
        # In-memory cache as fallback
        self._memory_cache = {}
        self.cache_duration = 60  # minutes (1 hour)
        
        # Try to use Redis cache for persistence across restarts
        self._redis_cache = None
        try:
            from services.redis_cache import get_redis_cache
            self._redis_cache = get_redis_cache()
            logger.info("✅ Weather API: Using Redis cache (persistent across restarts)")
        except Exception as e:
            logger.warning(f"⚠️ Weather API: Redis not available, using memory cache: {e}")
        
        if not self.has_api_key or not self.use_real_apis:
            logger.warning("Weather API: Using fallback mode with enhanced mock data.")
        else:
            logger.info("Weather API: Ready for live weather integration!")
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for weather request."""
        sorted_params = sorted(kwargs.items())
        return f"weather_{method}:{hash(str(sorted_params))}"
    
    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached weather response if not expired (tries Redis first, then memory)."""
        # Try Redis cache first (persistent)
        if self._redis_cache:
            try:
                cached_data = self._redis_cache.get(cache_key)
                if cached_data:
                    # Parse JSON string back to dict
                    if isinstance(cached_data, str):
                        cached_data = json.loads(cached_data)
                    logger.info(f"✅ Weather cache HIT (Redis): {cache_key[:50]}...")
                    return cached_data
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        # Fallback to memory cache
        if cache_key not in self._memory_cache:
            return None
        
        cached_data, timestamp = self._memory_cache[cache_key]
        if datetime.now() - timestamp < timedelta(minutes=self.cache_duration):
            logger.info(f"✅ Weather cache HIT (memory): {cache_key[:50]}...")
            return cached_data
        else:
            del self._memory_cache[cache_key]
            return None
    
    def _cache_response(self, cache_key: str, data: Dict) -> None:
        """Cache weather response (in both Redis and memory)."""
        # Cache in Redis with 1-hour TTL (persistent across restarts)
        if self._redis_cache:
            try:
                self._redis_cache.set(
                    cache_key,
                    json.dumps(data),
                    ttl=self.cache_duration * 60  # Convert minutes to seconds
                )
                logger.info(f"💾 Weather cached in Redis for {self.cache_duration} minutes")
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
        
        # Also cache in memory as fallback
        self._memory_cache[cache_key] = (data, datetime.now())
    
    def get_current_weather(self, city: str = "Istanbul", country: str = "TR") -> Dict:
        """Get current weather conditions with real API data when available."""
        cache_key = self._get_cache_key("current", city=city, country=country)
        
        # Try cache first (1-hour cache to reduce API calls)
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            logger.info(f"🔄 Using cached weather data for {city} (refreshes every hour)")
            return cached_result
        
        # Use real API if available
        if self.has_api_key and self.use_real_apis:
            try:
                logger.info(f"🌐 Fetching fresh weather data from API for {city}...")
                result = self._get_current_weather_real_api(city, country)
                self._cache_response(cache_key, result)
                logger.info(f"✅ REAL WEATHER: Current conditions for {city} (cached for 1 hour)")
                return result
            except Exception as e:
                logger.error(f"Real weather API failed, using mock data: {e}")
        
        # Fallback to enhanced mock data
        result = self._get_mock_current_weather(city)
        logger.info(f"📝 MOCK WEATHER: Using enhanced fallback data for {city}")
        return result
    
    def _get_current_weather_real_api(self, city: str, country: str) -> Dict:
        """Get real current weather from OpenWeatherMap."""
        url = f"{self.base_url}/weather"
        params = {
            "q": f"{city},{country}",
            "appid": self.api_key,
            "units": "metric"  # Celsius
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Enhance with Istanbul-specific recommendations
        enhanced_data = self._enhance_weather_data(data)
        enhanced_data["data_source"] = "real_api"
        enhanced_data["timestamp"] = datetime.now().isoformat()
        
        return enhanced_data
    
    def get_forecast(self, city: str = "Istanbul", country: str = "TR", days: int = 5) -> Dict:
        """Get weather forecast with real API data when available."""
        cache_key = self._get_cache_key("forecast", city=city, country=country, days=days)
        
        # Try cache first (1-hour cache to reduce API calls)
        cached_result = self._get_cached_response(cache_key)
        if cached_result:
            logger.info(f"🔄 Using cached forecast for {city} (refreshes every hour)")
            return cached_result
        
        # Use real API if available
        if self.has_api_key and self.use_real_apis:
            try:
                logger.info(f"🌐 Fetching fresh forecast from API for {city}...")
                result = self._get_forecast_real_api(city, country, days)
                self._cache_response(cache_key, result)
                logger.info(f"✅ REAL FORECAST: {days}-day forecast for {city} (cached for 1 hour)")
                return result
            except Exception as e:
                logger.error(f"Real forecast API failed, using mock data: {e}")
        
        # Fallback to enhanced mock data
        result = self._get_mock_forecast(city, days)
        logger.info(f"📝 MOCK FORECAST: Using enhanced fallback data for {city}")
        return result
    
    def _get_forecast_real_api(self, city: str, country: str, days: int) -> Dict:
        """Get real weather forecast from OpenWeatherMap."""
        url = f"{self.base_url}/forecast"
        params = {
            "q": f"{city},{country}",
            "appid": self.api_key,
            "units": "metric",
            "cnt": days * 8  # 8 forecasts per day (3-hour intervals)
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Process and enhance forecast data
        enhanced_forecast = self._process_forecast_data(data)
        enhanced_forecast["data_source"] = "real_api"
        enhanced_forecast["timestamp"] = datetime.now().isoformat()
        
        return enhanced_forecast
    
    def _enhance_weather_data(self, weather_data: Dict) -> Dict:
        """Enhance weather data with Istanbul-specific recommendations."""
        temp = weather_data["main"]["temp"]
        condition = weather_data["weather"][0]["main"].lower()
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data.get("wind", {}).get("speed", 0) * 3.6  # Convert m/s to km/h
        
        # Add activity recommendations based on weather
        recommendations = self._get_activity_recommendations(temp, condition, humidity, wind_speed)
        
        # Add clothing suggestions
        clothing = self._get_clothing_suggestions(temp, condition, wind_speed)
        
        # Add Istanbul-specific insights
        istanbul_insights = self._get_istanbul_weather_insights(temp, condition)
        
        # Add normalized fields for easy access
        weather_data.update({
            # Normalized field names for main_system.py
            "temperature": temp,
            "feels_like": weather_data["main"].get("feels_like", temp),
            "description": weather_data["weather"][0]["description"],
            "humidity": humidity,
            "wind_speed": round(wind_speed, 1),
            "pressure": weather_data["main"].get("pressure", 1013),
            "visibility": weather_data.get("visibility", 10000) / 1000,  # Convert m to km
            
            # Enhanced fields
            "activity_recommendations": recommendations,
            "clothing_suggestions": clothing,
            "istanbul_insights": istanbul_insights,
            "comfort_level": self._calculate_comfort_level(temp, humidity, wind_speed)
        })
        
        return weather_data
    
    def _get_activity_recommendations(self, temp: float, condition: str, humidity: int, wind_speed: float) -> List[str]:
        """Get activity recommendations based on weather conditions."""
        recommendations = []
        
        if temp >= 25 and condition not in ["rain", "storm"]:
            recommendations.extend([
                "Perfect for Bosphorus cruise",
                "Great day for exploring Sultanahmet outdoors",
                "Ideal for walking across Galata Bridge",
                "Perfect weather for Prince Islands ferry trip"
            ])
        elif 15 <= temp < 25:
            recommendations.extend([
                "Good for visiting outdoor attractions",
                "Perfect for exploring Balat and Fener neighborhoods",
                "Great for walking through Gülhane Park",
                "Ideal for rooftop restaurant dining"
            ])
        elif temp < 15:
            recommendations.extend([
                "Perfect for visiting indoor museums",
                "Great for Turkish bath (hammam) experience",
                "Ideal for shopping in Grand Bazaar",
                "Perfect for cozy café visits in Beyoğlu"
            ])
        
        if condition == "rain":
            recommendations.extend([
                "Visit covered areas like Grand Bazaar",
                "Perfect time for museum tours",
                "Ideal for underground Basilica Cistern",
                "Great for Turkish coffee houses"
            ])
        elif wind_speed > 10:
            recommendations.append("Be cautious near the Bosphorus - windy conditions")
        
        return recommendations[:4]  # Limit to top 4
    
    def _get_clothing_suggestions(self, temp: float, condition: str, wind_speed: float) -> List[str]:
        """Get clothing suggestions based on weather."""
        suggestions = []
        
        if temp >= 25:
            suggestions.extend(["Light, breathable clothing", "Sunhat recommended", "Comfortable walking shoes"])
        elif 15 <= temp < 25:
            suggestions.extend(["Light layers", "Comfortable jacket", "Closed shoes recommended"])
        else:
            suggestions.extend(["Warm layers", "Jacket or coat essential", "Warm, waterproof shoes"])
        
        if condition == "rain":
            suggestions.extend(["Umbrella essential", "Waterproof jacket", "Non-slip shoes"])
        elif wind_speed > 10:
            suggestions.append("Windproof outer layer")
        
        return suggestions
    
    def _get_istanbul_weather_insights(self, temp: float, condition: str) -> List[str]:
        """Get Istanbul-specific weather insights."""
        insights = []
        
        if temp > 30:
            insights.append("Hot day in Istanbul - many locals head to Princes' Islands for cooler air")
        elif temp < 5:
            insights.append("Cold day - perfect time to experience authentic Turkish breakfast indoors")
        
        if condition == "rain":
            insights.append("Rainy weather brings out the beauty of Istanbul's reflective streets")
        elif condition == "clear":
            insights.append("Clear skies offer stunning views from Galata Tower and Pierre Loti Hill")
        
        # Seasonal insights
        month = datetime.now().month
        if 6 <= month <= 8:  # Summer
            insights.append("Summer in Istanbul - evening strolls along the Bosphorus are magical")
        elif 12 <= month <= 2:  # Winter
            insights.append("Winter charm - hot chestnuts and tea vendors are everywhere")
        
        return insights
    
    def _calculate_comfort_level(self, temp: float, humidity: int, wind_speed: float) -> str:
        """Calculate overall comfort level."""
        score = 0
        
        # Temperature comfort (optimal 18-25°C)
        if 18 <= temp <= 25:
            score += 3
        elif 15 <= temp < 18 or 25 < temp <= 28:
            score += 2
        elif 10 <= temp < 15 or 28 < temp <= 32:
            score += 1
        
        # Humidity comfort (optimal 40-60%)
        if 40 <= humidity <= 60:
            score += 2
        elif 30 <= humidity < 40 or 60 < humidity <= 70:
            score += 1
        
        # Wind comfort (optimal < 15 km/h)
        if wind_speed < 15:
            score += 1
        
        if score >= 5:
            return "excellent"
        elif score >= 3:
            return "good"
        elif score >= 2:
            return "fair"
        else:
            return "poor"
    
    def _process_forecast_data(self, forecast_data: Dict) -> Dict:
        """Process raw forecast data into daily summaries."""
        daily_forecasts = []
        current_date = None
        daily_data = []
        
        for item in forecast_data["list"]:
            date = datetime.fromtimestamp(item["dt"]).date()
            
            if current_date != date:
                if daily_data:
                    # Process previous day
                    daily_summary = self._create_daily_summary(daily_data)
                    daily_forecasts.append(daily_summary)
                
                current_date = date
                daily_data = [item]
            else:
                daily_data.append(item)
        
        # Process last day
        if daily_data:
            daily_summary = self._create_daily_summary(daily_data)
            daily_forecasts.append(daily_summary)
        
        return {
            "city": forecast_data["city"],
            "daily_forecasts": daily_forecasts,
            "data_source": "real_api"
        }
    
    def _create_daily_summary(self, daily_data: List[Dict]) -> Dict:
        """Create daily weather summary from 3-hour intervals."""
        temps = [item["main"]["temp"] for item in daily_data]
        conditions = [item["weather"][0]["main"] for item in daily_data]
        
        # Most common condition
        condition = max(set(conditions), key=conditions.count)
        
        return {
            "date": datetime.fromtimestamp(daily_data[0]["dt"]).strftime("%Y-%m-%d"),
            "temp_min": round(min(temps), 1),
            "temp_max": round(max(temps), 1),
            "condition": condition,
            "description": daily_data[len(daily_data)//2]["weather"][0]["description"],
            "humidity": daily_data[len(daily_data)//2]["main"]["humidity"],
            "recommendations": self._get_activity_recommendations(
                sum(temps)/len(temps), condition.lower(), 
                daily_data[len(daily_data)//2]["main"]["humidity"], 0
            )[:2]
        }
    
    def _get_mock_current_weather(self, city: str) -> Dict:
        """Get enhanced mock current weather data."""
        # Realistic Istanbul weather based on season
        month = datetime.now().month
        
        if 6 <= month <= 8:  # Summer
            temp = 28
            condition = "clear"
            humidity = 65
        elif 12 <= month <= 2:  # Winter
            temp = 8
            condition = "clouds"
            humidity = 78
        elif 3 <= month <= 5:  # Spring
            temp = 18
            condition = "clear"
            humidity = 60
        else:  # Fall
            temp = 15
            condition = "clouds"
            humidity = 70
        
        mock_data = {
            "weather": [{"main": condition.title(), "description": f"{condition} sky"}],
            "main": {
                "temp": temp,
                "feels_like": temp - 2,
                "humidity": humidity,
                "pressure": 1013
            },
            "wind": {"speed": 5.2},
            "name": city,
            "data_source": "mock_data",
            "timestamp": datetime.now().isoformat(),
            "info_message": "🔄 Using mock weather data. Add OPENWEATHERMAP_API_KEY for real-time weather."
        }
        
        return self._enhance_weather_data(mock_data)
    
    def _get_mock_forecast(self, city: str, days: int) -> Dict:
        """Get enhanced mock forecast data."""
        daily_forecasts = []
        base_temp = 20
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            temp_variation = (-5 + i * 2) if i < 3 else (5 - i)
            
            daily_forecasts.append({
                "date": date,
                "temp_min": base_temp + temp_variation - 3,
                "temp_max": base_temp + temp_variation + 3,
                "condition": "Clear" if i % 2 == 0 else "Clouds",
                "description": "clear sky" if i % 2 == 0 else "scattered clouds",
                "humidity": 60 + (i * 5),
                "recommendations": ["Great for outdoor activities", "Perfect sightseeing weather"]
            })
        
        return {
            "city": {"name": city},
            "daily_forecasts": daily_forecasts,
            "data_source": "mock_data",
            "timestamp": datetime.now().isoformat(),
            "info_message": "🔄 Using mock forecast data. Add OPENWEATHERMAP_API_KEY for real forecasts."
        }
    
    def get_api_status(self) -> Dict:
        """Get weather API status."""
        return {
            "has_api_key": self.has_api_key,
            "use_real_apis": self.use_real_apis,
            "cache_entries": len(self._cache),
            "data_source": "real_api" if (self.has_api_key and self.use_real_apis) else "mock_data"
        }
