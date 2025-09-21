"""
Comprehensive tests for Real-time Data Service
Tests weather, traffic, events, and Istanbul-specific real-time data
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, Any, List
from datetime import datetime, timedelta

class TestRealTimeDataService:
    """Test Real-time Data Service functionality."""
    
    @pytest.fixture
    def mock_aiohttp_session(self):
        """Mock aiohttp session for API calls."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock()
        
        mock_session = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        return mock_session, mock_response
    
    @pytest.fixture
    def realtime_service(self):
        """Create real-time data service instance."""
        from backend.api_clients.realtime_data import RealTimeDataAggregator
        return RealTimeDataAggregator()
    
    @pytest.mark.asyncio
    async def test_get_weather_data_success(self, realtime_service, mock_aiohttp_session):
        """Test successful weather data retrieval."""
        mock_session, mock_response = mock_aiohttp_session
        mock_response.json.return_value = {
            "current": {
                "temperature_2m": 22.5,
                "weather_code": 0,
                "wind_speed_10m": 5.2,
                "humidity": 65
            },
            "daily": {
                "temperature_2m_max": [25.0],
                "temperature_2m_min": [18.0],
                "weather_code": [1]
            }
        }
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            weather_data = await realtime_service.get_weather_data("Istanbul")
        
        assert weather_data is not None
        assert "current" in weather_data
        assert weather_data["current"]["temperature"] == 22.5
        assert "forecast" in weather_data
    
    @pytest.mark.asyncio
    async def test_get_traffic_data_success(self, realtime_service, mock_aiohttp_session):
        """Test successful traffic data retrieval."""
        mock_session, mock_response = mock_aiohttp_session
        mock_response.json.return_value = {
            "routes": [
                {
                    "summary": {
                        "distance": 15000,
                        "duration": 1800,
                        "traffic_delay": 300
                    }
                }
            ]
        }
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            traffic_data = await realtime_service.get_traffic_data("Sultanahmet", "Taksim")
        
        assert traffic_data is not None
        assert "status" in traffic_data
        assert "duration" in traffic_data
        assert traffic_data["duration"] == 1800
    
    @pytest.mark.asyncio
    async def test_get_events_data_success(self, realtime_service, mock_aiohttp_session):
        """Test successful events data retrieval."""
        mock_session, mock_response = mock_aiohttp_session
        mock_response.json.return_value = {
            "events": [
                {
                    "name": "Istanbul Music Festival",
                    "start_time": "2025-09-22T20:00:00Z",
                    "venue": "Zorlu Center",
                    "category": "music"
                },
                {
                    "name": "Art Exhibition",
                    "start_time": "2025-09-23T10:00:00Z",
                    "venue": "Istanbul Modern",
                    "category": "art"
                }
            ]
        }
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            events_data = await realtime_service.get_events_data("Istanbul")
        
        assert events_data is not None
        assert "events" in events_data
        assert len(events_data["events"]) == 2
        assert events_data["events"][0]["name"] == "Istanbul Music Festival"
    
    @pytest.mark.asyncio
    async def test_get_transportation_status(self, realtime_service, mock_aiohttp_session):
        """Test public transportation status retrieval."""
        mock_session, mock_response = mock_aiohttp_session
        mock_response.json.return_value = {
            "metro": {
                "status": "operational",
                "delays": ["M2 Line: 5 minute delay"]
            },
            "ferry": {
                "status": "operational",
                "weather_affected": False
            },
            "bus": {
                "status": "limited",
                "affected_routes": ["15F", "28T"]
            }
        }
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            transport_data = await realtime_service.get_transportation_status()
        
        assert transport_data is not None
        assert "metro" in transport_data
        assert "ferry" in transport_data
        assert "bus" in transport_data
        assert transport_data["metro"]["status"] == "operational"
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_real_time_data(self, realtime_service):
        """Test comprehensive real-time data aggregation."""
        with patch.object(realtime_service, 'get_weather_data') as mock_weather, \
             patch.object(realtime_service, 'get_traffic_data') as mock_traffic, \
             patch.object(realtime_service, 'get_events_data') as mock_events, \
             patch.object(realtime_service, 'get_transportation_status') as mock_transport:
            
            # Mock return values
            mock_weather.return_value = {
                "current": {"temperature": 20, "condition": "sunny"},
                "forecast": [{"day": "tomorrow", "high": 25, "low": 15}]
            }
            mock_traffic.return_value = {
                "status": "moderate",
                "duration": 1200,
                "alternative_routes": ["Galata Bridge route"]
            }
            mock_events.return_value = {
                "events": [{"name": "Concert at Zorlu", "time": "20:00"}]
            }
            mock_transport.return_value = {
                "metro": {"status": "operational"},
                "ferry": {"status": "operational"}
            }
            
            result = await realtime_service.get_comprehensive_real_time_data(
                location="Sultanahmet",
                user_preferences={"interests": ["museums", "food"]}
            )
        
        assert result is not None
        assert "weather" in result
        assert "traffic" in result
        assert "events" in result
        assert "transportation" in result
        assert result["weather"]["current"]["temperature"] == 20
    
    @pytest.mark.asyncio
    async def test_error_handling_api_failure(self, realtime_service, mock_aiohttp_session):
        """Test error handling when external APIs fail."""
        mock_session, mock_response = mock_aiohttp_session
        mock_response.status = 500
        mock_response.json.side_effect = Exception("API Error")
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            weather_data = await realtime_service.get_weather_data("Istanbul")
        
        # Should handle errors gracefully
        assert weather_data is not None
        assert "error" in weather_data or "status" in weather_data
    
    @pytest.mark.asyncio
    async def test_error_handling_network_timeout(self, realtime_service):
        """Test handling of network timeouts."""
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = MagicMock()
            mock_session.get.side_effect = asyncio.TimeoutError("Request timeout")
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session
            
            result = await realtime_service.get_weather_data("Istanbul")
        
        # Should handle timeout gracefully
        assert result is not None
        assert "error" in result or "status" in result
    
    def test_data_caching_mechanism(self, realtime_service):
        """Test data caching for performance optimization."""
        cache_key = "weather_istanbul"
        test_data = {"temperature": 22, "condition": "sunny"}
        
        # Test cache storage
        realtime_service._cache_data(cache_key, test_data, ttl=300)
        
        # Test cache retrieval
        cached_data = realtime_service._get_cached_data(cache_key)
        assert cached_data == test_data
        
        # Test cache expiration
        expired_data = realtime_service._get_cached_data(cache_key, max_age=0)
        assert expired_data is None
    
    @pytest.mark.asyncio
    async def test_data_freshness_validation(self, realtime_service):
        """Test validation of data freshness."""
        # Test with fresh data
        fresh_timestamp = datetime.utcnow()
        is_fresh = realtime_service._is_data_fresh(fresh_timestamp, max_age_minutes=30)
        assert is_fresh is True
        
        # Test with stale data
        stale_timestamp = datetime.utcnow() - timedelta(hours=2)
        is_stale = realtime_service._is_data_fresh(stale_timestamp, max_age_minutes=30)
        assert is_stale is False
    
    @pytest.mark.asyncio
    async def test_location_based_filtering(self, realtime_service):
        """Test filtering data based on user location."""
        events_data = {
            "events": [
                {"name": "Sultanahmet Concert", "location": "Sultanahmet", "distance": 0.5},
                {"name": "Beyoğlu Festival", "location": "Beyoğlu", "distance": 5.2},
                {"name": "Kadıköy Market", "location": "Kadıköy", "distance": 15.8}
            ]
        }
        
        filtered_events = realtime_service._filter_by_location(
            events_data, 
            user_location="Sultanahmet", 
            max_distance_km=10
        )
        
        assert len(filtered_events["events"]) == 2  # Should exclude Kadıköy event
        assert all(event["distance"] <= 10 for event in filtered_events["events"])
    
    @pytest.mark.asyncio
    async def test_user_preference_filtering(self, realtime_service):
        """Test filtering data based on user preferences."""
        events_data = {
            "events": [
                {"name": "Classical Concert", "category": "music", "type": "classical"},
                {"name": "Rock Festival", "category": "music", "type": "rock"},
                {"name": "Art Exhibition", "category": "art", "type": "contemporary"},
                {"name": "Food Festival", "category": "food", "type": "street_food"}
            ]
        }
        
        user_preferences = {
            "interests": ["music", "art"],
            "exclude": ["rock"]
        }
        
        filtered_events = realtime_service._filter_by_preferences(
            events_data, 
            user_preferences
        )
        
        # Should include music and art, but exclude rock
        assert len(filtered_events["events"]) == 2
        categories = [event["category"] for event in filtered_events["events"]]
        assert "music" in categories
        assert "art" in categories
        assert not any(event["type"] == "rock" for event in filtered_events["events"])
    
    @pytest.mark.asyncio
    async def test_concurrent_data_fetching(self, realtime_service):
        """Test concurrent fetching of multiple data sources."""
        with patch.object(realtime_service, 'get_weather_data') as mock_weather, \
             patch.object(realtime_service, 'get_events_data') as mock_events, \
             patch.object(realtime_service, 'get_traffic_data') as mock_traffic:
            
            # Mock async return values
            mock_weather.return_value = {"temperature": 22}
            mock_events.return_value = {"events": []}
            mock_traffic.return_value = {"status": "light"}
            
            start_time = asyncio.get_event_loop().time()
            
            # Fetch data concurrently
            result = await realtime_service.get_comprehensive_real_time_data("Istanbul")
            
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            # Should complete quickly due to concurrent execution
            assert duration < 1.0  # Should be much faster than sequential
            assert result is not None
    
    def test_rate_limiting_compliance(self, realtime_service):
        """Test rate limiting for external API calls."""
        # Test rate limiter initialization
        assert hasattr(realtime_service, '_rate_limiter')
        
        # Test rate limit checking
        can_call = realtime_service._can_make_api_call("weather_api")
        assert isinstance(can_call, bool)
        
        # Test rate limit updating
        realtime_service._update_rate_limit("weather_api")
        
        # Verify rate limits are respected
        assert realtime_service._get_remaining_calls("weather_api") >= 0
    
    @pytest.mark.asyncio
    async def test_data_validation_and_sanitization(self, realtime_service):
        """Test validation and sanitization of received data."""
        invalid_weather_data = {
            "current": {
                "temperature_2m": "invalid_temp",
                "weather_code": None,
                "wind_speed_10m": -999
            }
        }
        
        sanitized_data = realtime_service._validate_and_sanitize_weather(invalid_weather_data)
        
        assert sanitized_data is not None
        assert isinstance(sanitized_data["current"]["temperature"], (int, float))
        assert sanitized_data["current"]["wind_speed"] >= 0
    
    @pytest.mark.asyncio
    async def test_emergency_alerts_integration(self, realtime_service, mock_aiohttp_session):
        """Test integration with emergency alert systems."""
        mock_session, mock_response = mock_aiohttp_session
        mock_response.json.return_value = {
            "alerts": [
                {
                    "type": "weather_warning",
                    "severity": "moderate",
                    "message": "Strong winds expected in Bosphorus area",
                    "valid_until": "2025-09-22T18:00:00Z"
                }
            ]
        }
        
        with patch('aiohttp.ClientSession', return_value=mock_session):
            alerts = await realtime_service.get_emergency_alerts("Istanbul")
        
        assert alerts is not None
        assert "alerts" in alerts
        assert len(alerts["alerts"]) == 1
        assert alerts["alerts"][0]["type"] == "weather_warning"
    
    def test_data_aggregation_accuracy(self, realtime_service):
        """Test accuracy of data aggregation and merging."""
        weather_data = {"temperature": 22, "condition": "sunny"}
        traffic_data = {"status": "moderate", "delay": 5}
        events_data = {"count": 3, "upcoming": ["Concert", "Exhibition"]}
        
        aggregated = realtime_service._aggregate_data(
            weather=weather_data,
            traffic=traffic_data,
            events=events_data
        )
        
        assert "weather" in aggregated
        assert "traffic" in aggregated
        assert "events" in aggregated
        assert aggregated["weather"]["temperature"] == 22
        assert aggregated["traffic"]["status"] == "moderate"
        assert aggregated["events"]["count"] == 3
