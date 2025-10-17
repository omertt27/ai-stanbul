"""
Robust İBB API Wrapper
======================

Wraps the İBB API client with comprehensive error handling, retries, and fallbacks.
Ensures all transport API calls are resilient to failures.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transportation.utils.api_resilience import (
    APIResilientClient,
    CircuitBreaker,
    validate_response,
    safe_get,
    safe_list_get,
    create_fallback_response,
    batch_resilient_calls,
    APIHealthMonitor
)

try:
    from real_ibb_api_integration import RealIBBAPIClient
    IBB_API_AVAILABLE = True
except ImportError as e:
    logging.warning(f"RealIBBAPIClient not available: {e}")
    IBB_API_AVAILABLE = False


logger = logging.getLogger(__name__)


class RobustIBBAPIWrapper:
    """Robust wrapper for İBB API with comprehensive error handling"""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize robust İBB API wrapper
        
        Args:
            redis_url: Optional Redis URL for caching
        """
        self.redis_url = redis_url
        self.resilient_client = APIResilientClient(
            max_retries=3,
            retry_delay=1.0,
            timeout=15.0,
            circuit_breaker=CircuitBreaker(failure_threshold=5, reset_timeout=60)
        )
        
        # Health monitors for different services
        self.health_monitors = {
            'metro': APIHealthMonitor(window_size=50),
            'bus': APIHealthMonitor(window_size=50),
            'ferry': APIHealthMonitor(window_size=50),
            'weather': APIHealthMonitor(window_size=30)
        }
        
        # Cache for fallback data
        self.last_successful_responses = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize GTFS fallback data
        self._initialize_gtfs_fallback()
    
    def _initialize_gtfs_fallback(self):
        """Initialize GTFS dataset fallback"""
        self.gtfs_fallback = {
            'metro_lines': {
                'M1A': {
                    'name': 'Yenikapı-Atatürk Havalimanı',
                    'stations': ['Yenikapı', 'Aksaray', 'Emniyet-Fatih', 'Topkapı-Ulubatlı', 
                                'Bayrampaşa-Maltepe', 'Sağmalcılar', 'Kocatepe', 'Otogar',
                                'Terazidere', 'Davutpaşa-YTÜ', 'Merter', 'Zeytinburnu',
                                'Bakırköy-İncirli', 'Bahçelievler', 'Ataköy-Şirinevler',
                                'Yenibosna', 'DTM-İstanbul Fuar Merkezi', 'Atatürk Havalimanı'],
                    'headway_minutes': {'peak': 3, 'off_peak': 5, 'evening': 8},
                    'operating_hours': {'first': '06:00', 'last': '00:30'}
                },
                'M2': {
                    'name': 'Yenikapı-Hacıosman',
                    'stations': ['Yenikapı', 'Vezneciler-İÜ', 'Haliç', 'Şişhane', 'Taksim',
                                'Osmanbey', 'Şişli-Mecidiyeköy', 'Gayrettepe', 'Levent',
                                'Sanayi Mahallesi', '4. Levent', 'Ayazağa', 'Atatürk Oto Sanayi',
                                'İTÜ Ayazağa', 'Seyrantepe', 'Darüşşafaka', 'Hacıosman'],
                    'headway_minutes': {'peak': 2, 'off_peak': 4, 'evening': 6},
                    'operating_hours': {'first': '06:00', 'last': '01:00'}
                },
                'M4': {
                    'name': 'Kadıköy-Sabiha Gökçen Havalimanı',
                    'stations': ['Kadıköy', 'Ayrılık Çeşmesi', 'Acıbadem', 'Ünalan', 'Göztepe',
                                'Yenisahra', 'Kozyatağı', 'Bostancı', 'Küçükyalı', 'Maltepe',
                                'Huzurevi', 'Gülsuyu', 'Esenkent', 'Hastane-Adliye',
                                'Soğanlık', 'Kartal', 'Yakacık-Adnan Kahveci', 'Pendik',
                                'Tavşantepe', 'Fevzi Çakmak', 'Yayalar', 'Kurtköy',
                                'Sabiha Gökçen Havalimanı'],
                    'headway_minutes': {'peak': 4, 'off_peak': 6, 'evening': 10},
                    'operating_hours': {'first': '06:00', 'last': '00:30'}
                },
                'M11': {
                    'name': 'İstanbul Havalimanı-Gayrettepe',
                    'stations': ['İstanbul Havalimanı', 'Gayrettepe'],
                    'headway_minutes': {'peak': 10, 'off_peak': 15, 'evening': 20},
                    'operating_hours': {'first': '06:00', 'last': '01:00'}
                },
                'T1': {
                    'name': 'Kabataş-Bağcılar',
                    'stations': ['Kabataş', 'Fındıklı', 'Tophane', 'Karaköy', 'Eminönü',
                                'Gülhane', 'Sultanahmet', 'Beyazıt-Kapalıçarşı', 'Laleli-Üniversite',
                                'Aksaray', 'Yusufpaşa', 'Haseki', 'Merkez Efendi', 'Pazartekke',
                                'Çapa-Şehremini', 'Davutpaşa-YTÜ', 'Topkapı', 'Cevizlibağ',
                                'Zeytinburnu', 'Seyitnizam', 'Soğanlı', 'Bağcılar'],
                    'headway_minutes': {'peak': 4, 'off_peak': 6, 'evening': 10},
                    'operating_hours': {'first': '06:00', 'last': '00:30'}
                }
            },
            'bus_routes': {
                '28': {'name': 'Beşiktaş-Edirnekapı', 'headway': 10},
                '36': {'name': 'Taksim-Kadıköy', 'headway': 15},
                '74': {'name': 'Mecidiyeköy-Söğütlü', 'headway': 12},
                '110': {'name': 'Taksim-Atatürk Havalimanı', 'headway': 20}
            },
            'ferry_routes': {
                'Eminönü-Üsküdar': {'duration': 15, 'frequency': 20},
                'Karaköy-Kadıköy': {'duration': 12, 'frequency': 15},
                'Beşiktaş-Üsküdar': {'duration': 8, 'frequency': 30},
                'Kabataş-Üsküdar': {'duration': 10, 'frequency': 25},
                'Kabataş-Kadıköy': {'duration': 20, 'frequency': 30}
            }
        }
    
    async def get_metro_data(self) -> Dict[str, Any]:
        """
        Get metro real-time data with robust error handling
        
        Returns:
            Metro data dictionary with status, crowding, and schedule info
        """
        start_time = datetime.now()
        
        try:
            if not IBB_API_AVAILABLE:
                raise ImportError("İBB API client not available")
            
            # Define the API call
            async def api_call():
                async with RealIBBAPIClient(self.redis_url) as client:
                    data = await client.get_metro_real_time_data()
                    # Validate response
                    validate_response(data, required_fields=['lines', 'timestamp'])
                    return data
            
            # Execute with resilience
            result = await self.resilient_client.resilient_call(
                api_call,
                fallback=self._get_metro_fallback
            )
            
            # Post-process and validate
            result = self._post_process_metro_data(result)
            
            # Record success
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['metro'].record_call(True, response_time)
            
            # Cache successful response
            self.last_successful_responses['metro'] = result
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['metro'].record_call(False, response_time)
            
            self.logger.error(f"Failed to get metro data: {e}")
            
            # Try to use last successful response if available
            if 'metro' in self.last_successful_responses:
                cached = self.last_successful_responses['metro'].copy()
                cached['source'] = 'cached_fallback'
                cached['message'] = 'Using last successful response due to API failure'
                return cached
            
            # Fall back to GTFS data
            return self._get_metro_fallback()
    
    async def get_bus_data(self, stop_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get bus real-time data with robust error handling
        
        Args:
            stop_id: Optional specific stop ID to query
            
        Returns:
            Bus data dictionary with routes, arrivals, and crowding
        """
        start_time = datetime.now()
        
        try:
            if not IBB_API_AVAILABLE:
                raise ImportError("İBB API client not available")
            
            async def api_call():
                async with RealIBBAPIClient(self.redis_url) as client:
                    data = await client.get_bus_real_time_data(stop_id)
                    validate_response(data, required_fields=['routes', 'timestamp'])
                    return data
            
            result = await self.resilient_client.resilient_call(
                api_call,
                fallback=self._get_bus_fallback,
                fallback_args=(stop_id,)
            )
            
            result = self._post_process_bus_data(result)
            
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['bus'].record_call(True, response_time)
            
            self.last_successful_responses['bus'] = result
            
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['bus'].record_call(False, response_time)
            
            self.logger.error(f"Failed to get bus data: {e}")
            
            if 'bus' in self.last_successful_responses:
                cached = self.last_successful_responses['bus'].copy()
                cached['source'] = 'cached_fallback'
                return cached
            
            return self._get_bus_fallback(stop_id)
    
    async def get_ferry_data(self) -> Dict[str, Any]:
        """
        Get ferry schedule data with robust error handling
        
        Returns:
            Ferry data dictionary with schedules and real-time status
        """
        start_time = datetime.now()
        
        try:
            if not IBB_API_AVAILABLE:
                raise ImportError("İBB API client not available")
            
            async def api_call():
                async with RealIBBAPIClient(self.redis_url) as client:
                    data = await client.get_ferry_schedule_data()
                    validate_response(data, required_fields=['routes', 'timestamp'])
                    return data
            
            result = await self.resilient_client.resilient_call(
                api_call,
                fallback=self._get_ferry_fallback
            )
            
            result = self._post_process_ferry_data(result)
            
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['ferry'].record_call(True, response_time)
            
            self.last_successful_responses['ferry'] = result
            
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['ferry'].record_call(False, response_time)
            
            self.logger.error(f"Failed to get ferry data: {e}")
            
            if 'ferry' in self.last_successful_responses:
                cached = self.last_successful_responses['ferry'].copy()
                cached['source'] = 'cached_fallback'
                return cached
            
            return self._get_ferry_fallback()
    
    async def get_weather_data(self) -> Dict[str, Any]:
        """
        Get weather data with robust error handling
        
        Returns:
            Weather data dictionary
        """
        start_time = datetime.now()
        
        try:
            if not IBB_API_AVAILABLE:
                raise ImportError("İBB API client not available")
            
            async def api_call():
                async with RealIBBAPIClient(self.redis_url) as client:
                    data = await client.get_weather_data()
                    validate_response(data, required_fields=['temperature', 'timestamp'])
                    return data
            
            result = await self.resilient_client.resilient_call(
                api_call,
                fallback=self._get_weather_fallback
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['weather'].record_call(True, response_time)
            
            self.last_successful_responses['weather'] = result
            
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.health_monitors['weather'].record_call(False, response_time)
            
            self.logger.error(f"Failed to get weather data: {e}")
            
            if 'weather' in self.last_successful_responses:
                cached = self.last_successful_responses['weather'].copy()
                cached['source'] = 'cached_fallback'
                return cached
            
            return self._get_weather_fallback()
    
    async def get_all_transport_data(self) -> Dict[str, Any]:
        """
        Get all transportation data concurrently with error handling
        
        Returns:
            Dictionary with all transport data
        """
        # Execute all calls concurrently
        metro_task = self.get_metro_data()
        bus_task = self.get_bus_data()
        ferry_task = self.get_ferry_data()
        weather_task = self.get_weather_data()
        
        metro_data, bus_data, ferry_data, weather_data = await asyncio.gather(
            metro_task, bus_task, ferry_task, weather_task,
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(metro_data, Exception):
            self.logger.error(f"Metro data exception: {metro_data}")
            metro_data = self._get_metro_fallback()
        
        if isinstance(bus_data, Exception):
            self.logger.error(f"Bus data exception: {bus_data}")
            bus_data = self._get_bus_fallback()
        
        if isinstance(ferry_data, Exception):
            self.logger.error(f"Ferry data exception: {ferry_data}")
            ferry_data = self._get_ferry_fallback()
        
        if isinstance(weather_data, Exception):
            self.logger.error(f"Weather data exception: {weather_data}")
            weather_data = self._get_weather_fallback()
        
        return {
            'metro': metro_data,
            'bus': bus_data,
            'ferry': ferry_data,
            'weather': weather_data,
            'timestamp': datetime.now().isoformat(),
            'health_status': self.get_health_status()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all API endpoints"""
        return {
            service: monitor.get_health_status()
            for service, monitor in self.health_monitors.items()
        }
    
    # Post-processing methods
    def _post_process_metro_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and validate metro data"""
        lines = safe_get(data, 'lines', default={}, expected_type=dict)
        
        # Ensure all lines have required fields
        for line_code, line_data in lines.items():
            if not isinstance(line_data, dict):
                continue
            
            line_data.setdefault('name', f'Metro Line {line_code}')
            line_data.setdefault('status', 'operational')
            line_data.setdefault('delays', [])
            line_data.setdefault('crowding', 0.5)
            line_data.setdefault('last_updated', datetime.now().isoformat())
        
        data['lines'] = lines
        return data
    
    def _post_process_bus_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and validate bus data"""
        routes = safe_get(data, 'routes', default={}, expected_type=dict)
        
        for route_code, route_data in routes.items():
            if not isinstance(route_data, dict):
                continue
            
            route_data.setdefault('name', f'Bus Route {route_code}')
            route_data.setdefault('crowding', 0.5)
            route_data.setdefault('next_arrival', 10)
            route_data.setdefault('status', 'active')
            route_data.setdefault('delays', 0)
        
        data['routes'] = routes
        return data
    
    def _post_process_ferry_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and validate ferry data"""
        routes = safe_get(data, 'routes', default={}, expected_type=dict)
        
        for route_name, route_data in routes.items():
            if not isinstance(route_data, dict):
                continue
            
            route_data.setdefault('next_departure', 15)
            route_data.setdefault('frequency_minutes', 20)
            route_data.setdefault('duration_minutes', 15)
            route_data.setdefault('status', 'operational')
            route_data.setdefault('delays', 0)
        
        data['routes'] = routes
        return data
    
    # Fallback methods using GTFS data
    def _get_metro_fallback(self) -> Dict[str, Any]:
        """Get metro fallback data from GTFS dataset"""
        current_time = datetime.now()
        hour = current_time.hour
        
        # Determine time period
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            period = 'peak'
        elif 20 <= hour <= 23 or 0 <= hour <= 5:
            period = 'evening'
        else:
            period = 'off_peak'
        
        lines = {}
        for line_code, line_info in self.gtfs_fallback['metro_lines'].items():
            # Calculate crowding based on time
            crowding = self._estimate_crowding_by_time(hour, line_code)
            
            lines[line_code] = {
                'name': line_info['name'],
                'status': 'operational',
                'delays': [],
                'crowding': crowding,
                'headway_minutes': line_info['headway_minutes'][period],
                'next_arrival': line_info['headway_minutes'][period] // 2,
                'stations': line_info['stations'],
                'last_updated': current_time.isoformat()
            }
        
        return {
            'lines': lines,
            'timestamp': current_time.isoformat(),
            'source': 'gtfs_fallback',
            'message': 'Using İBB GTFS static dataset'
        }
    
    def _get_bus_fallback(self, stop_id: Optional[str] = None) -> Dict[str, Any]:
        """Get bus fallback data from GTFS dataset"""
        current_time = datetime.now()
        hour = current_time.hour
        
        routes = {}
        for route_code, route_info in self.gtfs_fallback['bus_routes'].items():
            crowding = self._estimate_crowding_by_time(hour, route_code, 'bus')
            
            routes[route_code] = {
                'name': route_info['name'],
                'crowding': crowding,
                'next_arrival': route_info['headway'] // 2,
                'status': 'active',
                'delays': 0,
                'headway_minutes': route_info['headway']
            }
        
        return {
            'routes': routes,
            'timestamp': current_time.isoformat(),
            'source': 'gtfs_fallback',
            'message': 'Using İBB GTFS static dataset'
        }
    
    def _get_ferry_fallback(self) -> Dict[str, Any]:
        """Get ferry fallback data from GTFS dataset"""
        current_time = datetime.now()
        
        routes = {}
        for route_name, route_info in self.gtfs_fallback['ferry_routes'].items():
            # Calculate next departure
            minutes_past_hour = current_time.minute
            freq = route_info['frequency']
            next_dep = freq - (minutes_past_hour % freq)
            if next_dep == freq:
                next_dep = 0
            
            routes[route_name] = {
                'next_departure': next_dep,
                'frequency_minutes': freq,
                'duration_minutes': route_info['duration'],
                'status': 'operational',
                'delays': 0
            }
        
        return {
            'routes': routes,
            'timestamp': current_time.isoformat(),
            'source': 'gtfs_fallback',
            'message': 'Using İBB GTFS static dataset'
        }
    
    def _get_weather_fallback(self) -> Dict[str, Any]:
        """Get weather fallback data based on seasonal averages"""
        current_time = datetime.now()
        month = current_time.month
        
        # Istanbul seasonal temperature averages
        seasonal_temps = {
            12: 10, 1: 8, 2: 10,      # Winter
            3: 13, 4: 18, 5: 23,      # Spring
            6: 27, 7: 30, 8: 29,      # Summer
            9: 25, 10: 19, 11: 14     # Fall
        }
        
        temp = seasonal_temps.get(month, 20)
        
        return {
            'temperature': temp,
            'feels_like': temp - 1,
            'humidity': 65,
            'description': 'partly cloudy',
            'wind_speed': 8,
            'timestamp': current_time.isoformat(),
            'source': 'seasonal_fallback',
            'message': 'Using seasonal average weather data'
        }
    
    def _estimate_crowding_by_time(
        self,
        hour: int,
        line_code: str,
        transport_type: str = 'metro'
    ) -> float:
        """Estimate crowding based on time of day"""
        base_crowding = 0.3
        
        # Rush hour multipliers
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            base_crowding += 0.4
        elif 10 <= hour <= 16:
            base_crowding += 0.2
        elif 20 <= hour <= 22:
            base_crowding += 0.3
        
        # Line-specific adjustments
        busy_lines = ['M2', 'T1', '28', '74', '36']
        if line_code in busy_lines:
            base_crowding += 0.1
        
        return min(base_crowding, 1.0)


if __name__ == "__main__":
    # Test the robust wrapper
    async def test_wrapper():
        """Test robust İBB API wrapper"""
        print("🧪 Testing Robust İBB API Wrapper...")
        
        wrapper = RobustIBBAPIWrapper()
        
        # Test individual endpoints
        print("\n📊 Testing Metro Data...")
        metro_data = await wrapper.get_metro_data()
        print(f"Metro lines: {len(metro_data.get('lines', {}))}")
        print(f"Source: {metro_data.get('source', 'unknown')}")
        
        print("\n🚌 Testing Bus Data...")
        bus_data = await wrapper.get_bus_data()
        print(f"Bus routes: {len(bus_data.get('routes', {}))}")
        print(f"Source: {bus_data.get('source', 'unknown')}")
        
        print("\n⛴️ Testing Ferry Data...")
        ferry_data = await wrapper.get_ferry_data()
        print(f"Ferry routes: {len(ferry_data.get('routes', {}))}")
        print(f"Source: {ferry_data.get('source', 'unknown')}")
        
        print("\n🌤️ Testing Weather Data...")
        weather_data = await wrapper.get_weather_data()
        print(f"Temperature: {weather_data.get('temperature', 'N/A')}°C")
        print(f"Source: {weather_data.get('source', 'unknown')}")
        
        print("\n📦 Testing All Transport Data...")
        all_data = await wrapper.get_all_transport_data()
        print(f"Got all transport data")
        
        print("\n💚 Health Status:")
        health = wrapper.get_health_status()
        for service, status in health.items():
            print(f"  {service}: {status['status']} ({status['success_rate']:.1%} success)")
        
        print("\n✅ Robust İBB API Wrapper Test Complete!")
    
    asyncio.run(test_wrapper())
