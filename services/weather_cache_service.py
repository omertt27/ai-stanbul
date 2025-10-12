#!/usr/bin/env python3
"""
Weather Cache Service for Istanbul Daily Talk AI
Provides accurate, up-to-date weather for Istanbul with smart caching and notifications
"""

import asyncio
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Istanbul coordinates
ISTANBUL_LAT = 41.0082
ISTANBUL_LON = 28.9784
ISTANBUL_CITY_ID = 745044

@dataclass
class WeatherData:
    """Weather data structure"""
    location: str
    latitude: float
    longitude: float
    current_temp: float
    feels_like: float
    humidity: int
    pressure: float
    visibility: float
    uv_index: Optional[float]
    condition: str
    description: str
    icon: str
    wind_speed: float
    wind_direction: int
    clouds: int
    sunrise: datetime
    sunset: datetime
    timestamp: datetime
    
    # Additional Istanbul-specific data
    bosphorus_wind: Optional[float] = None  # Wind conditions for Bosphorus
    rainfall_1h: Optional[float] = None
    rainfall_3h: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'location': self.location,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'current_temp': self.current_temp,
            'feels_like': self.feels_like,
            'humidity': self.humidity,
            'pressure': self.pressure,
            'visibility': self.visibility,
            'uv_index': self.uv_index,
            'condition': self.condition,
            'description': self.description,
            'icon': self.icon,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'clouds': self.clouds,
            'sunrise': self.sunrise.isoformat(),
            'sunset': self.sunset.isoformat(),
            'timestamp': self.timestamp.isoformat(),
            'bosphorus_wind': self.bosphorus_wind,
            'rainfall_1h': self.rainfall_1h,
            'rainfall_3h': self.rainfall_3h
        }

@dataclass
class HourlyForecast:
    """Hourly weather forecast"""
    datetime: datetime
    temp: float
    feels_like: float
    condition: str
    description: str
    icon: str
    precipitation_chance: float
    precipitation_amount: float
    wind_speed: float
    humidity: int
    clouds: int
    visibility: float

@dataclass
class WeatherAlert:
    """Weather alert/warning"""
    alert_type: str  # 'rain', 'wind', 'temperature', 'storm', 'fog'
    severity: str    # 'low', 'medium', 'high', 'extreme'
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    affected_areas: List[str]


class WeatherAPIClient:
    """Weather API client with multiple providers"""
    
    def __init__(self):
        self.openweather_api_key = os.getenv('OPENWEATHER_API_KEY', 'demo_key')
        self.meteosource_api_key = os.getenv('METEOSOURCE_API_KEY', 'demo_key')
        self.session = requests.Session()
        self.session.timeout = 10
        
    def get_current_weather_openweather(self) -> Optional[WeatherData]:
        """Get current weather from OpenWeatherMap"""
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': ISTANBUL_LAT,
                'lon': ISTANBUL_LON,
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            main = data['main']
            weather = data['weather'][0]
            wind = data.get('wind', {})
            clouds = data.get('clouds', {})
            sys = data['sys']
            rain = data.get('rain', {})
            
            return WeatherData(
                location="Istanbul, Turkey",
                latitude=ISTANBUL_LAT,
                longitude=ISTANBUL_LON,
                current_temp=main['temp'],
                feels_like=main['feels_like'],
                humidity=main['humidity'],
                pressure=main['pressure'],
                visibility=data.get('visibility', 10000) / 1000,  # Convert to km
                uv_index=None,  # Requires separate API call
                condition=weather['main'],
                description=weather['description'],
                icon=weather['icon'],
                wind_speed=wind.get('speed', 0),
                wind_direction=wind.get('deg', 0),
                clouds=clouds.get('all', 0),
                sunrise=datetime.fromtimestamp(sys['sunrise']),
                sunset=datetime.fromtimestamp(sys['sunset']),
                timestamp=datetime.now(),
                bosphorus_wind=wind.get('speed', 0) * 1.2,  # Bosphorus typically windier
                rainfall_1h=rain.get('1h', 0),
                rainfall_3h=rain.get('3h', 0)
            )
            
        except Exception as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            return None
    
    def get_hourly_forecast_openweather(self, hours: int = 24) -> List[HourlyForecast]:
        """Get hourly forecast from OpenWeatherMap"""
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast"
            params = {
                'lat': ISTANBUL_LAT,
                'lon': ISTANBUL_LON,
                'appid': self.openweather_api_key,
                'units': 'metric',
                'cnt': min(hours, 40)  # API limit
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            for item in data['list'][:hours]:
                main = item['main']
                weather = item['weather'][0]
                wind = item.get('wind', {})
                clouds = item.get('clouds', {})
                rain = item.get('rain', {})
                
                forecasts.append(HourlyForecast(
                    datetime=datetime.fromtimestamp(item['dt']),
                    temp=main['temp'],
                    feels_like=main['feels_like'],
                    condition=weather['main'],
                    description=weather['description'],
                    icon=weather['icon'],
                    precipitation_chance=item.get('pop', 0) * 100,
                    precipitation_amount=rain.get('3h', 0),
                    wind_speed=wind.get('speed', 0),
                    humidity=main['humidity'],
                    clouds=clouds.get('all', 0),
                    visibility=item.get('visibility', 10000) / 1000
                ))
            
            return forecasts
            
        except Exception as e:
            logger.error(f"OpenWeatherMap forecast API error: {e}")
            return []
    
    def get_current_weather_fallback(self) -> Optional[WeatherData]:
        """Fallback weather data (mock or alternative API)"""
        try:
            # Mock weather data for development/fallback
            now = datetime.now()
            return WeatherData(
                location="Istanbul, Turkey",
                latitude=ISTANBUL_LAT,
                longitude=ISTANBUL_LON,
                current_temp=18.5,
                feels_like=19.2,
                humidity=65,
                pressure=1013.2,
                visibility=10.0,
                uv_index=3.0,
                condition="Partly Cloudy",
                description="partly cloudy",
                icon="02d",
                wind_speed=3.2,
                wind_direction=235,
                clouds=25,
                sunrise=now.replace(hour=6, minute=45, second=0, microsecond=0),
                sunset=now.replace(hour=18, minute=30, second=0, microsecond=0),
                timestamp=now,
                bosphorus_wind=4.1,
                rainfall_1h=0.0,
                rainfall_3h=0.0
            )
        except Exception as e:
            logger.error(f"Fallback weather error: {e}")
            return None


class WeatherCache:
    """Smart weather caching system"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "weather_cache.json"
        self.forecast_file = self.cache_dir / "forecast_cache.json"
        
        self.api_client = WeatherAPIClient()
        self.current_weather: Optional[WeatherData] = None
        self.hourly_forecast: List[HourlyForecast] = []
        self.last_update: Optional[datetime] = None
        self.update_interval = 3600  # 1 hour in seconds
        
        # Load existing cache
        self._load_cache()
        
    def _load_cache(self):
        """Load weather cache from disk"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    if data.get('timestamp'):
                        # Check if cache is still valid (within 2 hours)
                        cache_time = datetime.fromisoformat(data['timestamp'])
                        if datetime.now() - cache_time < timedelta(hours=2):
                            self.current_weather = self._dict_to_weather_data(data)
                            self.last_update = cache_time
                            logger.info("✅ Loaded weather cache from disk")
            
            if self.forecast_file.exists():
                with open(self.forecast_file, 'r') as f:
                    forecast_data = json.load(f)
                    if forecast_data.get('forecasts'):
                        self.hourly_forecast = [
                            self._dict_to_hourly_forecast(item) 
                            for item in forecast_data['forecasts']
                        ]
                        logger.info(f"✅ Loaded {len(self.hourly_forecast)} forecast entries from cache")
                        
        except Exception as e:
            logger.warning(f"Failed to load weather cache: {e}")
    
    def _save_cache(self):
        """Save weather cache to disk"""
        try:
            if self.current_weather:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.current_weather.to_dict(), f, indent=2)
            
            if self.hourly_forecast:
                forecast_data = {
                    'updated_at': datetime.now().isoformat(),
                    'forecasts': [asdict(forecast) for forecast in self.hourly_forecast]
                }
                # Convert datetime objects to strings
                for forecast in forecast_data['forecasts']:
                    forecast['datetime'] = forecast['datetime'].isoformat()
                
                with open(self.forecast_file, 'w') as f:
                    json.dump(forecast_data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Failed to save weather cache: {e}")
    
    def _dict_to_weather_data(self, data: Dict) -> WeatherData:
        """Convert dictionary to WeatherData object"""
        return WeatherData(
            location=data['location'],
            latitude=data['latitude'],
            longitude=data['longitude'],
            current_temp=data['current_temp'],
            feels_like=data['feels_like'],
            humidity=data['humidity'],
            pressure=data['pressure'],
            visibility=data['visibility'],
            uv_index=data.get('uv_index'),
            condition=data['condition'],
            description=data['description'],
            icon=data['icon'],
            wind_speed=data['wind_speed'],
            wind_direction=data['wind_direction'],
            clouds=data['clouds'],
            sunrise=datetime.fromisoformat(data['sunrise']),
            sunset=datetime.fromisoformat(data['sunset']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            bosphorus_wind=data.get('bosphorus_wind'),
            rainfall_1h=data.get('rainfall_1h'),
            rainfall_3h=data.get('rainfall_3h')
        )
    
    def _dict_to_hourly_forecast(self, data: Dict) -> HourlyForecast:
        """Convert dictionary to HourlyForecast object"""
        return HourlyForecast(
            datetime=datetime.fromisoformat(data['datetime']),
            temp=data['temp'],
            feels_like=data['feels_like'],
            condition=data['condition'],
            description=data['description'],
            icon=data['icon'],
            precipitation_chance=data['precipitation_chance'],
            precipitation_amount=data['precipitation_amount'],
            wind_speed=data['wind_speed'],
            humidity=data['humidity'],
            clouds=data['clouds'],
            visibility=data['visibility']
        )
    
    def needs_update(self) -> bool:
        """Check if weather data needs updating"""
        if not self.last_update or not self.current_weather:
            return True
        
        time_since_update = datetime.now() - self.last_update
        return time_since_update.total_seconds() > self.update_interval
    
    async def update_weather(self, force: bool = False) -> bool:
        """Update weather data from API"""
        if not force and not self.needs_update():
            return True
        
        logger.info("🌤️ Updating weather data...")
        
        try:
            # Get current weather
            weather_data = self.api_client.get_current_weather_openweather()
            if not weather_data:
                logger.warning("Primary weather API failed, trying fallback...")
                weather_data = self.api_client.get_current_weather_fallback()
            
            if weather_data:
                self.current_weather = weather_data
                self.last_update = datetime.now()
                logger.info(f"✅ Updated current weather: {weather_data.condition}, {weather_data.current_temp}°C")
            
            # Get hourly forecast
            forecast_data = self.api_client.get_hourly_forecast_openweather(24)
            if forecast_data:
                self.hourly_forecast = forecast_data
                logger.info(f"✅ Updated hourly forecast: {len(forecast_data)} hours")
            
            # Save to cache
            self._save_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update weather: {e}")
            return False
    
    def get_current_weather(self) -> Optional[WeatherData]:
        """Get current weather (from cache)"""
        return self.current_weather
    
    def get_hourly_forecast(self, hours: int = 24) -> List[HourlyForecast]:
        """Get hourly forecast (from cache)"""
        return self.hourly_forecast[:hours]
    
    def get_weather_summary(self) -> Dict[str, Any]:
        """Get weather summary for AI system"""
        if not self.current_weather:
            return {"error": "No weather data available"}
        
        w = self.current_weather
        now = datetime.now()
        
        # Determine time of day
        is_daytime = w.sunrise <= now <= w.sunset
        
        # Weather conditions analysis
        conditions = {
            'temperature_category': self._categorize_temperature(w.current_temp),
            'comfort_level': self._assess_comfort(w.current_temp, w.feels_like, w.humidity, w.wind_speed),
            'precipitation_status': self._assess_precipitation(w.rainfall_1h, w.rainfall_3h),
            'visibility_status': self._assess_visibility(w.visibility),
            'wind_status': self._assess_wind(w.wind_speed),
            'outdoor_suitability': self._assess_outdoor_conditions(w),
            'is_daytime': is_daytime
        }
        
        return {
            'current': w.to_dict(),
            'conditions': conditions,
            'recommendations': self._generate_weather_recommendations(w, conditions),
            'alerts': self._check_weather_alerts(w),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'cache_age_minutes': (now - self.last_update).total_seconds() / 60 if self.last_update else None
        }
    
    def _categorize_temperature(self, temp: float) -> str:
        """Categorize temperature for Istanbul context"""
        if temp < 0:
            return "freezing"
        elif temp < 10:
            return "cold"
        elif temp < 20:
            return "cool"
        elif temp < 28:
            return "comfortable"
        elif temp < 35:
            return "warm"
        else:
            return "hot"
    
    def _assess_comfort(self, temp: float, feels_like: float, humidity: int, wind_speed: float) -> str:
        """Assess overall comfort level"""
        temp_comfort = 15 <= temp <= 25
        feels_like_comfort = 15 <= feels_like <= 25
        humidity_comfort = 30 <= humidity <= 70
        wind_comfort = wind_speed < 10
        
        comfort_score = sum([temp_comfort, feels_like_comfort, humidity_comfort, wind_comfort])
        
        if comfort_score >= 3:
            return "excellent"
        elif comfort_score >= 2:
            return "good"
        elif comfort_score >= 1:
            return "fair"
        else:
            return "poor"
    
    def _assess_precipitation(self, rain_1h: Optional[float], rain_3h: Optional[float]) -> str:
        """Assess precipitation status"""
        if not rain_1h and not rain_3h:
            return "dry"
        
        recent_rain = rain_1h or 0
        if recent_rain == 0:
            return "dry"
        elif recent_rain < 0.5:
            return "light_rain"
        elif recent_rain < 2.0:
            return "moderate_rain"
        else:
            return "heavy_rain"
    
    def _assess_visibility(self, visibility: float) -> str:
        """Assess visibility conditions"""
        if visibility >= 10:
            return "excellent"
        elif visibility >= 5:
            return "good"
        elif visibility >= 2:
            return "moderate"
        else:
            return "poor"
    
    def _assess_wind(self, wind_speed: float) -> str:
        """Assess wind conditions"""
        if wind_speed < 2:
            return "calm"
        elif wind_speed < 5:
            return "light"
        elif wind_speed < 10:
            return "moderate"
        elif wind_speed < 15:
            return "strong"
        else:
            return "very_strong"
    
    def _assess_outdoor_conditions(self, weather: WeatherData) -> str:
        """Assess overall outdoor suitability"""
        factors = []
        
        # Temperature factor
        if 15 <= weather.current_temp <= 28:
            factors.append(1)
        elif 10 <= weather.current_temp <= 32:
            factors.append(0.5)
        else:
            factors.append(0)
        
        # Precipitation factor
        if (weather.rainfall_1h or 0) == 0:
            factors.append(1)
        elif (weather.rainfall_1h or 0) < 1:
            factors.append(0.5)
        else:
            factors.append(0)
        
        # Wind factor
        if weather.wind_speed < 10:
            factors.append(1)
        elif weather.wind_speed < 15:
            factors.append(0.5)
        else:
            factors.append(0)
        
        # Visibility factor
        if weather.visibility >= 5:
            factors.append(1)
        else:
            factors.append(0.5)
        
        score = sum(factors) / len(factors)
        
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_weather_recommendations(self, weather: WeatherData, conditions: Dict) -> List[str]:
        """Generate weather-based recommendations"""
        recommendations = []
        
        # Temperature recommendations
        if conditions['temperature_category'] == 'cold':
            recommendations.append("Dress warmly and consider indoor attractions")
        elif conditions['temperature_category'] == 'hot':
            recommendations.append("Stay hydrated and seek shade during midday")
        
        # Precipitation recommendations
        if conditions['precipitation_status'] in ['light_rain', 'moderate_rain']:
            recommendations.append("Bring an umbrella and consider covered attractions")
            recommendations.append("Indoor markets like Grand Bazaar are perfect for rainy weather")
        elif conditions['precipitation_status'] == 'heavy_rain':
            recommendations.append("Best time for museum visits and indoor activities")
        
        # Wind recommendations
        if conditions['wind_status'] in ['strong', 'very_strong']:
            recommendations.append("Be cautious near the Bosphorus - extra windy conditions")
            recommendations.append("Avoid outdoor boat tours in strong winds")
        
        # General outdoor recommendations
        if conditions['outdoor_suitability'] == 'excellent':
            recommendations.append("Perfect weather for exploring outdoor attractions!")
            recommendations.append("Great time for Bosphorus cruise and walking tours")
        elif conditions['outdoor_suitability'] == 'poor':
            recommendations.append("Consider indoor activities like museums and mosques")
        
        return recommendations
    
    def _check_weather_alerts(self, weather: WeatherData) -> List[WeatherAlert]:
        """Check for weather alerts that should trigger notifications"""
        alerts = []
        now = datetime.now()
        
        # Temperature alerts
        if weather.current_temp > 35:
            alerts.append(WeatherAlert(
                alert_type="temperature",
                severity="high",
                title="🌡️ Heat Warning",
                description=f"Very hot weather ({weather.current_temp}°C). Stay hydrated and avoid midday sun.",
                start_time=now,
                end_time=now + timedelta(hours=4),
                affected_areas=["All Istanbul"]
            ))
        elif weather.current_temp < 0:
            alerts.append(WeatherAlert(
                alert_type="temperature",
                severity="high",
                title="🧊 Freezing Alert",
                description=f"Freezing temperatures ({weather.current_temp}°C). Dress very warmly.",
                start_time=now,
                end_time=now + timedelta(hours=6),
                affected_areas=["All Istanbul"]
            ))
        
        # Rain alerts
        if (weather.rainfall_1h or 0) > 2:
            alerts.append(WeatherAlert(
                alert_type="rain",
                severity="medium",
                title="🌧️ Heavy Rain Alert",
                description="Heavy rainfall detected. Consider indoor activities and bring waterproof gear.",
                start_time=now,
                end_time=now + timedelta(hours=2),
                affected_areas=["All Istanbul"]
            ))
        
        # Wind alerts
        if weather.wind_speed > 15:
            alerts.append(WeatherAlert(
                alert_type="wind",
                severity="medium",
                title="💨 Strong Wind Warning",
                description=f"Strong winds ({weather.wind_speed:.1f} m/s). Be cautious near waterfront areas.",
                start_time=now,
                end_time=now + timedelta(hours=3),
                affected_areas=["Bosphorus", "Golden Horn", "Coastal Areas"]
            ))
        
        # Visibility alerts
        if weather.visibility < 2:
            alerts.append(WeatherAlert(
                alert_type="fog",
                severity="medium",
                title="🌫️ Low Visibility Alert",
                description=f"Poor visibility ({weather.visibility} km). Exercise caution when traveling.",
                start_time=now,
                end_time=now + timedelta(hours=4),
                affected_areas=["All Istanbul"]
            ))
        
        return alerts


# Global weather cache instance
weather_cache = WeatherCache()

# Weather cache management functions
async def update_weather_cache():
    """Update weather cache - called by scheduler"""
    try:
        success = await weather_cache.update_weather()
        if success:
            logger.info("🌤️ Weather cache updated successfully")
        else:
            logger.warning("⚠️ Weather cache update failed")
        return success
    except Exception as e:
        logger.error(f"Weather cache update error: {e}")
        return False

def get_current_weather() -> Optional[Dict[str, Any]]:
    """Get current weather for API endpoints"""
    return weather_cache.get_weather_summary()

def get_weather_for_ai() -> Dict[str, Any]:
    """Get weather summary optimized for AI system integration"""
    summary = weather_cache.get_weather_summary()
    if 'error' in summary:
        return summary
    
    # Extract key info for AI
    current = summary['current']
    conditions = summary['conditions']
    
    return {
        'temperature': current['current_temp'],
        'feels_like': current['feels_like'],
        'condition': current['condition'],
        'description': current['description'],
        'comfort_level': conditions['comfort_level'],
        'outdoor_suitability': conditions['outdoor_suitability'],
        'precipitation': conditions['precipitation_status'],
        'recommendations': summary['recommendations'][:3],  # Top 3 recommendations
        'is_good_weather': conditions['outdoor_suitability'] in ['excellent', 'good'],
        'needs_umbrella': conditions['precipitation_status'] in ['light_rain', 'moderate_rain', 'heavy_rain'],
        'is_very_hot': current['current_temp'] > 30,
        'is_cold': current['current_temp'] < 10
    }

# Export for use in other modules
__all__ = [
    'WeatherCache',
    'WeatherData', 
    'HourlyForecast',
    'WeatherAlert',
    'weather_cache',
    'update_weather_cache',
    'get_current_weather',
    'get_weather_for_ai'
]
