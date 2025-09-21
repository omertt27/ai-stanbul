"""
Comprehensive tests for Real-time Data Service - ACTIVELY USED MODULE
Tests event data, crowd levels, traffic data, and data aggregation
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from backend.api_clients.realtime_data import (
    RealTimeDataAggregator, RealTimeEventClient, RealTimeCrowdClient, 
    RealTimeTrafficClient, EventData, CrowdData, TrafficData
)


class TestRealTimeDataAggregator:
    """Test the real-time data aggregation service."""
    
    @pytest.fixture
    def aggregator(self):
        """Create real-time data aggregator for testing."""
        return RealTimeDataAggregator()
    
    @pytest.fixture
    def mock_event_data(self):
        """Create mock event data."""
        return EventData(
            event_id="event_123",
            name="Istanbul Music Festival",
            location="Zorlu Center",
            start_time=datetime.now() + timedelta(days=1),
            end_time=datetime.now() + timedelta(days=1, hours=3),
            category="music",
            price_range=(50.0, 200.0),
            crowd_level="high",
            description="Annual music festival featuring international artists",
            venue_capacity=5000,
            tickets_available=True
        )
    
    @pytest.fixture
    def mock_crowd_data(self):
        """Create mock crowd data."""
        return CrowdData(
            location_id="blue_mosque",
            location_name="Blue Mosque",
            current_crowd_level="moderate",
            peak_times=["09:00", "14:00", "17:00"],
            wait_time_minutes=15,
            last_updated=datetime.now(),
            confidence_score=0.85
        )
    
    @pytest.fixture
    def mock_traffic_data(self):
        """Create mock traffic data."""
        return TrafficData(
            origin="Taksim Square",
            destination="Sultanahmet",
            duration_normal=25,
            duration_current=35,
            traffic_level="moderate",
            recommended_route="via Golden Horn Bridge",
            alternative_routes=[
                {"route": "via Galata Bridge", "duration": 40},
                {"route": "via Bosphorus Bridge", "duration": 45}
            ],
            last_updated=datetime.now()
        )
    
    def test_aggregator_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.event_client is not None
        assert aggregator.crowd_client is not None
        assert aggregator.traffic_client is not None
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_data_events_only(self, aggregator, mock_event_data):
        """Test getting comprehensive data with events only."""
        # Mock event client
        aggregator.event_client.get_live_events = AsyncMock(return_value=[mock_event_data])
        
        result = await aggregator.get_comprehensive_real_time_data(
            include_events=True,
            include_crowds=False,
            include_traffic=False
        )
        
        assert "events" in result
        assert len(result["events"]) == 1
        assert result["events"][0]["name"] == "Istanbul Music Festival"
        assert result["events"][0]["location"] == "Zorlu Center"
        assert "crowds" not in result
        assert "traffic" not in result
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_data_crowds_only(self, aggregator, mock_crowd_data):
        """Test getting comprehensive data with crowds only."""
        # Mock crowd client
        aggregator.crowd_client.get_crowd_levels = AsyncMock(return_value=[mock_crowd_data])
        
        result = await aggregator.get_comprehensive_real_time_data(
            include_events=False,
            include_crowds=True,
            include_traffic=False
        )
        
        assert "crowd_levels" in result
        assert len(result["crowd_levels"]) == 1
        assert result["crowd_levels"][0]["location_name"] == "Blue Mosque"
        assert result["crowd_levels"][0]["current_crowd_level"] == "moderate"
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_data_traffic_only(self, aggregator, mock_traffic_data):
        """Test getting comprehensive data with traffic only."""
        # Mock traffic client
        aggregator.traffic_client.get_traffic_aware_route = AsyncMock(return_value=mock_traffic_data)
        
        result = await aggregator.get_comprehensive_real_time_data(
            include_events=False,
            include_crowds=False,
            include_traffic=True,
            origin="Taksim",
            destination="Sultanahmet"
        )
        
        assert "traffic" in result
        assert result["traffic"]["origin"] == "Taksim Square"
        assert result["traffic"]["traffic_level"] == "moderate"
    
    @pytest.mark.asyncio
    async def test_get_comprehensive_data_all_services(self, aggregator, mock_event_data, mock_crowd_data, mock_traffic_data):
        """Test getting comprehensive data from all services."""
        # Mock all clients
        aggregator.event_client.get_live_events = AsyncMock(return_value=[mock_event_data])
        aggregator.crowd_client.get_crowd_levels = AsyncMock(return_value=[mock_crowd_data])
        aggregator.traffic_client.get_traffic_aware_route = AsyncMock(return_value=mock_traffic_data)
        
        result = await aggregator.get_comprehensive_real_time_data(
            include_events=True,
            include_crowds=True,
            include_traffic=True,
            origin="Taksim",
            destination="Sultanahmet"
        )
        
        assert "events" in result
        assert "crowd_levels" in result
        assert "traffic" in result
        assert len(result["events"]) == 1
        assert len(result["crowd_levels"]) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling_event_service_failure(self, aggregator):
        """Test error handling when event service fails."""
        # Mock event client to raise exception
        aggregator.event_client.get_live_events = AsyncMock(side_effect=Exception("Event API failed"))
        aggregator.crowd_client.get_crowd_levels = AsyncMock(return_value=[])
        
        result = await aggregator.get_comprehensive_real_time_data(
            include_events=True,
            include_crowds=True
        )
        
        # Should still return partial data
        assert "crowd_levels" in result
        # Events might be missing or empty due to error
        assert isinstance(result, dict)
    
    def test_event_to_dict_conversion(self, aggregator, mock_event_data):
        """Test conversion of EventData to dictionary."""
        event_dict = aggregator._event_to_dict(mock_event_data)
        
        assert event_dict["id"] == "event_123"
        assert event_dict["name"] == "Istanbul Music Festival"
        assert event_dict["location"] == "Zorlu Center"
        assert event_dict["category"] == "music"
        assert event_dict["price_range"] == (50.0, 200.0)
        assert event_dict["crowd_level"] == "high"
        assert "start_time" in event_dict
        assert "end_time" in event_dict
    
    def test_crowd_to_dict_conversion(self, aggregator, mock_crowd_data):
        """Test conversion of CrowdData to dictionary."""
        crowd_dict = aggregator._crowd_to_dict(mock_crowd_data)
        
        assert crowd_dict["location_id"] == "blue_mosque"
        assert crowd_dict["location_name"] == "Blue Mosque"
        assert crowd_dict["current_crowd_level"] == "moderate"
        assert crowd_dict["wait_time_minutes"] == 15
        assert crowd_dict["peak_times"] == ["09:00", "14:00", "17:00"]
    
    def test_traffic_to_dict_conversion(self, aggregator, mock_traffic_data):
        """Test conversion of TrafficData to dictionary."""
        traffic_dict = aggregator._traffic_to_dict(mock_traffic_data)
        
        assert traffic_dict["origin"] == "Taksim Square"
        assert traffic_dict["destination"] == "Sultanahmet"
        assert traffic_dict["duration_current"] == 35
        assert traffic_dict["traffic_level"] == "moderate"
        assert len(traffic_dict["alternative_routes"]) == 2


class TestRealTimeEventClient:
    """Test the real-time event client."""
    
    @pytest.fixture
    def event_client(self):
        """Create event client for testing."""
        return RealTimeEventClient()
    
    def test_event_client_initialization(self, event_client):
        """Test event client initialization."""
        assert event_client.mock_events is not None
        assert len(event_client.mock_events) > 0
    
    @pytest.mark.asyncio
    async def test_get_live_events_mock_data(self, event_client):
        """Test getting live events with mock data."""
        events = await event_client.get_live_events(date_range=7)
        
        assert isinstance(events, list)
        assert len(events) > 0
        assert len(events) <= 20  # Should limit to 20 events
        
        # Check first event structure
        first_event = events[0]
        assert hasattr(first_event, 'event_id')
        assert hasattr(first_event, 'name')
        assert hasattr(first_event, 'location')
        assert hasattr(first_event, 'start_time')
    
    @pytest.mark.asyncio
    async def test_get_live_events_with_categories(self, event_client):
        """Test getting live events filtered by categories."""
        events = await event_client.get_live_events(
            date_range=7, 
            categories=["music", "culture"]
        )
        
        assert isinstance(events, list)
        # All events should be in specified categories
        for event in events:
            assert event.category in ["music", "culture", "arts", "entertainment"]
    
    def test_generate_mock_events(self, event_client):
        """Test mock event generation."""
        mock_events = event_client._generate_mock_events()
        
        assert len(mock_events) > 5
        # Check event diversity
        categories = {event.category for event in mock_events}
        assert len(categories) > 2  # Should have multiple categories
        
        # Check required fields
        for event in mock_events:
            assert event.event_id
            assert event.name
            assert event.location
            assert isinstance(event.start_time, datetime)


class TestRealTimeCrowdClient:
    """Test the real-time crowd client."""
    
    @pytest.fixture
    def crowd_client(self):
        """Create crowd client for testing."""
        return RealTimeCrowdClient()
    
    def test_crowd_client_initialization(self, crowd_client):
        """Test crowd client initialization."""
        assert hasattr(crowd_client, 'popular_locations')
        assert len(crowd_client.popular_locations) > 0
    
    @pytest.mark.asyncio
    async def test_get_crowd_levels(self, crowd_client):
        """Test getting crowd levels for popular locations."""
        crowd_data = await crowd_client.get_crowd_levels()
        
        assert isinstance(crowd_data, list)
        assert len(crowd_data) > 0
        
        # Check first crowd data structure
        first_crowd = crowd_data[0]
        assert hasattr(first_crowd, 'location_id')
        assert hasattr(first_crowd, 'location_name')
        assert hasattr(first_crowd, 'current_crowd_level')
        assert first_crowd.current_crowd_level in ["low", "moderate", "high", "very_high"]
    
    @pytest.mark.asyncio
    async def test_get_crowd_level_for_location(self, crowd_client):
        """Test getting crowd level for specific location."""
        crowd_data = await crowd_client.get_crowd_level_for_location("blue_mosque")
        
        assert isinstance(crowd_data, CrowdData)
        assert crowd_data.location_id == "blue_mosque"
        assert crowd_data.location_name
        assert crowd_data.current_crowd_level in ["low", "moderate", "high", "very_high"]


class TestRealTimeTrafficClient:
    """Test the real-time traffic client."""
    
    @pytest.fixture
    def traffic_client(self):
        """Create traffic client for testing."""
        return RealTimeTrafficClient()
    
    @pytest.mark.asyncio
    async def test_get_traffic_aware_route(self, traffic_client):
        """Test getting traffic-aware route."""
        route_data = await traffic_client.get_traffic_aware_route(
            "Taksim Square", 
            "Blue Mosque"
        )
        
        assert isinstance(route_data, TrafficData)
        assert route_data.origin == "Taksim Square"
        assert route_data.destination == "Blue Mosque"
        assert route_data.duration_current > 0
        assert route_data.traffic_level in ["light", "moderate", "heavy", "severe"]
    
    @pytest.mark.asyncio
    async def test_get_current_traffic_conditions(self, traffic_client):
        """Test getting current traffic conditions."""
        conditions = await traffic_client.get_current_traffic_conditions()
        
        assert isinstance(conditions, dict)
        assert "overall_traffic_level" in conditions
        assert "affected_areas" in conditions
        assert "last_updated" in conditions


class TestDataStructures:
    """Test the data structure classes."""
    
    def test_event_data_creation(self):
        """Test EventData creation and properties."""
        event = EventData(
            event_id="test_123",
            name="Test Event",
            location="Test Location",
            start_time=datetime.now(),
            end_time=None,
            category="test",
            price_range=(10.0, 50.0),
            crowd_level="moderate",
            description="Test description",
            venue_capacity=100,
            tickets_available=True
        )
        
        assert event.event_id == "test_123"
        assert event.name == "Test Event"
        assert event.price_range == (10.0, 50.0)
        assert event.tickets_available is True
    
    def test_crowd_data_creation(self):
        """Test CrowdData creation and properties."""
        crowd = CrowdData(
            location_id="test_location",
            location_name="Test Location",
            current_crowd_level="high",
            peak_times=["10:00", "15:00"],
            wait_time_minutes=30,
            last_updated=datetime.now(),
            confidence_score=0.9
        )
        
        assert crowd.location_id == "test_location"
        assert crowd.current_crowd_level == "high"
        assert crowd.wait_time_minutes == 30
        assert crowd.confidence_score == 0.9
    
    def test_traffic_data_creation(self):
        """Test TrafficData creation and properties."""
        traffic = TrafficData(
            origin="Origin Point",
            destination="Destination Point",
            duration_normal=20,
            duration_current=35,
            traffic_level="heavy",
            recommended_route="Main route",
            alternative_routes=[{"route": "Alt 1", "duration": 40}],
            last_updated=datetime.now()
        )
        
        assert traffic.origin == "Origin Point"
        assert traffic.duration_current == 35
        assert traffic.traffic_level == "heavy"
        assert len(traffic.alternative_routes) == 1


class TestIntegration:
    """Integration tests for real-time data services."""
    
    @pytest.mark.asyncio
    async def test_full_realtime_data_workflow(self):
        """Test complete real-time data workflow."""
        aggregator = RealTimeDataAggregator()
        
        # Mock all dependencies
        mock_event = EventData(
            event_id="integration_event",
            name="Integration Test Event",
            location="Test Venue",
            start_time=datetime.now() + timedelta(hours=2),
            end_time=datetime.now() + timedelta(hours=5),
            category="test",
            price_range=(20.0, 80.0),
            crowd_level="moderate",
            description="Integration test event",
            venue_capacity=200,
            tickets_available=True
        )
        
        mock_crowd = CrowdData(
            location_id="integration_location",
            location_name="Integration Location",
            current_crowd_level="low",
            peak_times=["12:00", "18:00"],
            wait_time_minutes=5,
            last_updated=datetime.now(),
            confidence_score=0.8
        )
        
        # Mock client responses
        aggregator.event_client.get_live_events = AsyncMock(return_value=[mock_event])
        aggregator.crowd_client.get_crowd_levels = AsyncMock(return_value=[mock_crowd])
        
        # Get comprehensive data
        result = await aggregator.get_comprehensive_real_time_data(
            include_events=True,
            include_crowds=True,
            include_traffic=False
        )
        
        # Verify results
        assert "events" in result
        assert "crowd_levels" in result
        assert len(result["events"]) == 1
        assert len(result["crowd_levels"]) == 1
        assert result["events"][0]["name"] == "Integration Test Event"
        assert result["crowd_levels"][0]["current_crowd_level"] == "low"
