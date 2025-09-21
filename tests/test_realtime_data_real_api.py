"""
Realtime Data Tests - Testing REAL API Methods Only
Tests only async methods that actually exist in realtime_data.py
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from backend.api_clients.realtime_data import (
    RealTimeEventClient,
    RealTimeCrowdClient, 
    RealTimeTrafficClient,
    RealTimeDataAggregator,
    EventData,
    CrowdData,
    TrafficData
)


class TestRealTimeDataRealAPI:
    """Test the Real-time Data functionality - only real methods."""
    
    @pytest.fixture
    def event_client(self):
        """Create real-time event client for testing."""
        return RealTimeEventClient()
    
    @pytest.fixture
    def crowd_client(self):
        """Create real-time crowd client for testing."""
        return RealTimeCrowdClient()
    
    @pytest.fixture
    def traffic_client(self):
        """Create real-time traffic client for testing."""
        return RealTimeTrafficClient()
    
    @pytest.fixture
    def data_aggregator(self):
        """Create real-time data aggregator for testing."""
        return RealTimeDataAggregator()
    
    def test_event_client_initialization(self, event_client):
        """Test real-time event client initialization."""
        assert event_client is not None
        assert hasattr(event_client, 'ticketmaster_key')
        assert hasattr(event_client, 'eventbrite_token')
        assert hasattr(event_client, 'facebook_token')
        assert hasattr(event_client, 'mock_events')
        assert isinstance(event_client.mock_events, list)
    
    def test_crowd_client_initialization(self, crowd_client):
        """Test real-time crowd client initialization."""
        assert crowd_client is not None
        assert hasattr(crowd_client, 'google_maps_key')
        assert hasattr(crowd_client, 'foursquare_key')
        assert hasattr(crowd_client, 'monitored_locations')
        assert isinstance(crowd_client.monitored_locations, list)
    
    def test_traffic_client_initialization(self, traffic_client):
        """Test real-time traffic client initialization."""
        assert traffic_client is not None
        assert hasattr(traffic_client, 'google_maps_key')
        assert hasattr(traffic_client, 'mapbox_token')
    
    def test_data_aggregator_initialization(self, data_aggregator):
        """Test real-time data aggregator initialization."""
        assert data_aggregator is not None
        assert hasattr(data_aggregator, 'event_client')
        assert hasattr(data_aggregator, 'crowd_client')
        assert hasattr(data_aggregator, 'traffic_client')
    
    @pytest.mark.asyncio
    async def test_get_live_events_no_api_keys(self, event_client):
        """Test getting live events without API keys (should return mock data)."""
        events = await event_client.get_live_events(date_range=7)
        
        assert isinstance(events, list)
        assert len(events) <= 20  # Should limit to 20 events
        
        if events:
            # Check first event structure
            event = events[0]
            assert isinstance(event, EventData)
            assert hasattr(event, 'event_id')
            assert hasattr(event, 'name')
            assert hasattr(event, 'location')
            assert hasattr(event, 'start_time')
            assert hasattr(event, 'category')
    
    @pytest.mark.asyncio
    async def test_get_live_events_with_categories(self, event_client):
        """Test getting live events with category filter."""
        categories = ["music", "cultural"]
        events = await event_client.get_live_events(date_range=3, categories=categories)
        
        assert isinstance(events, list)
        # Should return filtered results or mock data
        assert len(events) <= 20
    
    def test_generate_mock_events(self, event_client):
        """Test mock events generation."""
        mock_events = event_client._generate_mock_events()
        
        assert isinstance(mock_events, list)
        assert len(mock_events) > 0
        
        # Check event structure
        event = mock_events[0]
        assert isinstance(event, EventData)
        assert event.location is not None
        assert event.name is not None
        assert isinstance(event.start_time, datetime)
    
    def test_get_intelligent_mock_events(self, event_client):
        """Test intelligent mock events generation."""
        mock_events = event_client._get_intelligent_mock_events(7, ["music"])
        
        assert isinstance(mock_events, list)
        
        # Check if any events returned
        for event in mock_events:
            assert isinstance(event, EventData)
            assert event.name is not None
    
    @pytest.mark.asyncio
    async def test_get_crowd_levels_no_api_keys(self, crowd_client):
        """Test getting crowd levels without API keys (should return mock data)."""
        location_ids = ["hagia_sophia", "blue_mosque", "grand_bazaar"]
        crowd_data = await crowd_client.get_crowd_levels(location_ids)
        
        assert isinstance(crowd_data, list)
        
        if crowd_data:
            crowd = crowd_data[0]
            assert isinstance(crowd, CrowdData)
            assert hasattr(crowd, 'location_id')
            assert hasattr(crowd, 'location_name')
            assert hasattr(crowd, 'current_crowd_level')
            assert hasattr(crowd, 'wait_time_minutes')
            assert crowd.current_crowd_level in ["low", "moderate", "high", "very_high"]
    
    def test_generate_mock_crowd_data(self, crowd_client):
        """Test mock crowd data generation."""
        test_location = {"id": "hagia_sophia", "name": "Hagia Sophia", "place_id": "test_place_id"}
        mock_data = crowd_client._generate_mock_crowd_data(test_location)
        
        assert isinstance(mock_data, CrowdData)
        assert mock_data.location_id == "hagia_sophia"
        assert mock_data.location_name == "Hagia Sophia"
        assert mock_data.current_crowd_level in ["low", "moderate", "high", "very_high"]
        assert isinstance(mock_data.wait_time_minutes, int)
        assert isinstance(mock_data.peak_times, list)
    
    @pytest.mark.asyncio
    async def test_get_traffic_aware_route_no_api_keys(self, traffic_client):
        """Test getting traffic aware route without API keys (should return mock data)."""
        origin = "Sultanahmet"
        destination = "Galata Tower"
        
        traffic_data = await traffic_client.get_traffic_aware_route(origin, destination)
        
        assert isinstance(traffic_data, TrafficData)
        assert traffic_data.origin == origin
        assert traffic_data.destination == destination
        assert hasattr(traffic_data, 'duration_normal')
        assert hasattr(traffic_data, 'duration_current')
        assert hasattr(traffic_data, 'traffic_level')
        assert traffic_data.traffic_level in ["light", "moderate", "heavy", "severe"]
    
    def test_generate_mock_traffic_data(self, traffic_client):
        """Test mock traffic data generation."""
        origin = "Sultanahmet"
        destination = "Taksim"
        
        traffic_data = traffic_client._generate_mock_traffic_data(origin, destination, "driving")
        
        assert isinstance(traffic_data, TrafficData)
        assert traffic_data.origin == origin
        assert traffic_data.destination == destination
        assert isinstance(traffic_data.duration_normal, int)
        assert isinstance(traffic_data.duration_current, int)
        assert traffic_data.duration_current >= traffic_data.duration_normal
        assert traffic_data.traffic_level in ["light", "moderate", "heavy", "severe"]
        assert isinstance(traffic_data.alternative_routes, list)
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_real_time_data_no_api_keys(self, data_aggregator):
        """Test getting comprehensive real-time data without API keys."""
        data = await data_aggregator.get_comprehensive_real_time_data(
            include_events=True,
            include_crowds=True,
            include_traffic=False
        )
        
        assert isinstance(data, dict)
        assert "events" in data
        assert "crowd_levels" in data
        
        # Events should be a list
        assert isinstance(data["events"], list)
        
        # Crowd levels should be a list
        assert isinstance(data["crowd_levels"], list)
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_data_with_traffic(self, data_aggregator):
        """Test getting comprehensive data including traffic."""
        data = await data_aggregator.get_comprehensive_real_time_data(
            include_events=False,
            include_crowds=False,
            include_traffic=True,
            origin="Taksim",
            destination="Sultanahmet"
        )
        
        assert isinstance(data, dict)
        assert "traffic" in data
        
        traffic_info = data["traffic"]
        assert isinstance(traffic_info, dict)
        assert "origin" in traffic_info
        assert "destination" in traffic_info
        assert "duration_current" in traffic_info
    
    def test_data_classes_creation(self):
        """Test creation of data class instances."""
        # Test EventData
        event = EventData(
            event_id="test_123",
            name="Istanbul Music Festival",
            location="Zorlu Center",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=3),
            category="music",
            price_range=(50.0, 200.0),
            crowd_level="moderate",
            description="Annual music festival",
            venue_capacity=5000,
            tickets_available=True
        )
        
        assert event.event_id == "test_123"
        assert event.name == "Istanbul Music Festival"
        assert event.category == "music"
        assert event.tickets_available is True
        
        # Test CrowdData
        crowd = CrowdData(
            location_id="hagia_sophia",
            location_name="Hagia Sophia",
            current_crowd_level="moderate",
            peak_times=["10:00-12:00", "14:00-16:00"],
            wait_time_minutes=15,
            last_updated=datetime.now(),
            confidence_score=0.8
        )
        
        assert crowd.location_id == "hagia_sophia"
        assert crowd.current_crowd_level == "moderate"
        assert crowd.wait_time_minutes == 15
        assert crowd.confidence_score == 0.8
        
        # Test TrafficData
        traffic = TrafficData(
            origin="Taksim",
            destination="Sultanahmet",
            duration_normal=25,
            duration_current=35,
            traffic_level="moderate",
            recommended_route="via Kennedy Avenue",
            alternative_routes=[{"route": "via Golden Horn", "duration": 40}],
            last_updated=datetime.now()
        )
        
        assert traffic.origin == "Taksim"
        assert traffic.destination == "Sultanahmet"
        assert traffic.duration_normal == 25
        assert traffic.duration_current == 35
        assert traffic.traffic_level == "moderate"
        assert len(traffic.alternative_routes) == 1
    
    def test_helper_methods(self, event_client):
        """Test helper methods in event client."""
        # Test price range extraction
        price_ranges = [{"min": 50, "max": 100}, {"min": 30, "max": 150}]
        result = event_client._extract_price_range(price_ranges)
        assert isinstance(result, tuple)
        assert result[0] == 30  # min
        assert result[1] == 150  # max
        
        # Test category mapping
        categories = event_client._map_to_ticketmaster_categories(["music", "sports"])
        assert isinstance(categories, list)
        assert "Music" in categories
        assert "Sports" in categories
    
    def test_crowd_data_conversion(self, data_aggregator):
        """Test crowd data conversion methods."""
        crowd = CrowdData(
            location_id="test_location",
            location_name="Test Location",
            current_crowd_level="high",
            peak_times=["10:00", "15:00"],
            wait_time_minutes=20,
            last_updated=datetime.now(),
            confidence_score=0.9
        )
        
        crowd_dict = data_aggregator._crowd_to_dict(crowd)
        assert isinstance(crowd_dict, dict)
        assert crowd_dict["location_id"] == "test_location"
        assert crowd_dict["current_crowd_level"] == "high"
        assert crowd_dict["wait_time_minutes"] == 20
    
    def test_event_data_conversion(self, data_aggregator):
        """Test event data conversion methods."""
        event = EventData(
            event_id="test_event",
            name="Test Event",
            location="Test Location",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2),
            category="test",
            price_range=(10.0, 50.0),
            crowd_level="moderate",
            description="Test event description",
            venue_capacity=100,
            tickets_available=True
        )
        
        event_dict = data_aggregator._event_to_dict(event)
        assert isinstance(event_dict, dict)
        assert event_dict["id"] == "test_event"
        assert event_dict["name"] == "Test Event"
        assert event_dict["category"] == "test"
        assert event_dict["tickets_available"] is True
    
    def test_traffic_data_conversion(self, data_aggregator):
        """Test traffic data conversion methods."""
        traffic = TrafficData(
            origin="Origin",
            destination="Destination",
            duration_normal=30,
            duration_current=45,
            traffic_level="heavy",
            recommended_route="Test Route",
            alternative_routes=[{"summary": "Alt Route", "duration": 40}],
            last_updated=datetime.now()
        )
        
        traffic_dict = data_aggregator._traffic_to_dict(traffic)
        assert isinstance(traffic_dict, dict)
        assert traffic_dict["origin"] == "Origin"
        assert traffic_dict["destination"] == "Destination"
        assert traffic_dict["duration_normal"] == 30
        assert traffic_dict["duration_current"] == 45
        assert traffic_dict["traffic_level"] == "heavy"
    
    @pytest.mark.asyncio
    async def test_integration_full_realtime_workflow(self, data_aggregator):
        """Test complete real-time data workflow."""
        # Test comprehensive data gathering
        location_ids = ["hagia_sophia", "blue_mosque", "grand_bazaar"]
        
        # Get events
        events = await data_aggregator.event_client.get_live_events(date_range=7)
        assert isinstance(events, list)
        
        # Get crowd data
        crowd_data = await data_aggregator.crowd_client.get_crowd_levels(location_ids)
        assert isinstance(crowd_data, list)
        
        # Get traffic data
        traffic_data = await data_aggregator.traffic_client.get_traffic_aware_route(
            "Taksim", "Sultanahmet"
        )
        assert isinstance(traffic_data, TrafficData)
        
        # Get comprehensive data
        comprehensive_data = await data_aggregator.get_comprehensive_real_time_data(
            include_events=True,
            include_crowds=True,
            include_traffic=True,
            origin="Taksim",
            destination="Sultanahmet"
        )
        
        assert isinstance(comprehensive_data, dict)
        if "events" in comprehensive_data:
            assert isinstance(comprehensive_data["events"], list)
        if "crowd_levels" in comprehensive_data:
            assert isinstance(comprehensive_data["crowd_levels"], list)
        if "traffic" in comprehensive_data:
            assert isinstance(comprehensive_data["traffic"], dict)
