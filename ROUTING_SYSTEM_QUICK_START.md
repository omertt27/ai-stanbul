# Industry-Level Routing System - Quick Start Guide
## Istanbul AI Transportation

**For Developers:** How to use the new graph-based routing system

---

## 🚀 Quick Start

### Basic Journey Planning

```python
from services.route_network_builder import TransportationNetwork
from services.journey_planner import JourneyPlanner

# 1. Initialize network (in production, load from İBB data)
network = TransportationNetwork()
# ... load network data from İBB API ...

# 2. Create journey planner
planner = JourneyPlanner(network)

# 3. Plan a journey
plan = planner.plan_journey_simple("Taksim", "Kadıköy")

# 4. Get results
if plan:
    print(plan.get_summary())
    journey_data = plan.to_dict()  # For JSON API
```

---

## 📋 Common Use Cases

### 1. Simple Route Finding

```python
# Find route with default preferences
plan = planner.plan_journey_simple(
    origin="Taksim",
    destination="Sultanahmet",
    minimize_transfers=True
)

# Access route details
print(f"Duration: {plan.primary_journey.total_duration_minutes} min")
print(f"Transfers: {plan.primary_journey.total_transfers}")
print(f"Cost: ₺{plan.primary_journey.estimated_cost_tl}")
```

### 2. Route with Custom Preferences

```python
from services.intelligent_route_finder import RoutePreferences
from services.journey_planner import JourneyRequest

# Define preferences
preferences = RoutePreferences(
    minimize_transfers=True,      # Minimize number of transfers
    minimize_walking=True,         # Minimize walking distance
    minimize_time=False,           # Not prioritizing speed
    prefer_metro=True,             # Prefer metro over bus
    max_transfers=2,               # Maximum 2 transfers
    max_walking_meters=500,        # Maximum 500m walking
    wheelchair_accessible=False    # Accessibility requirements
)

# Create request
request = JourneyRequest(
    origin="Taksim",
    destination="Kadıköy",
    preferences=preferences,
    include_alternatives=True,     # Get alternative routes
    max_alternatives=3             # Up to 3 alternatives
)

# Plan journey
plan = planner.plan_journey(request)

# Check alternatives
for i, alt in enumerate(plan.alternative_journeys, 1):
    print(f"Alternative {i}: {alt.total_duration_minutes}min, "
          f"{alt.total_transfers} transfers")
```

### 3. Accessible Routes

```python
# Find wheelchair-accessible route
accessible_plan = planner.get_accessible_route("Taksim", "Kadıköy")

if accessible_plan:
    print("Accessible route found!")
else:
    print("No accessible route available")
```

### 4. Explore Nearby Transport

```python
# Find all transport options near a location
transport_options = planner.explore_area(
    location="Taksim",
    max_distance_km=0.5  # Within 500m
)

# Print options by type
for transport_type, stops in transport_options['transport_options'].items():
    print(f"\n{transport_type.upper()}:")
    for stop in stops:
        print(f"  - {stop['name']} ({stop['distance_km']}km)")
        print(f"    Lines: {', '.join(stop['lines'])}")
```

### 5. Location Matching

```python
from services.location_matcher import LocationMatcher

matcher = LocationMatcher(network)

# Find stops by name (fuzzy matching)
matches = matcher.find_stops_by_name("taksi", max_results=3)
for match in matches:
    print(f"{match.stop_name} (confidence: {match.confidence:.2f})")

# Find stops by coordinates
nearby = matcher.find_stops_by_coordinates(
    latitude=41.0370,
    longitude=28.9857,
    max_distance_km=0.5
)
for stop in nearby:
    print(f"{stop.stop_name} - {stop.distance_km:.2f}km away")

# Smart matching (handles both names and coordinates)
match = matcher.match_location("41.0370,28.9857")
if match:
    print(f"Matched to: {match.stop_name}")
```

### 6. Direct Route Finding

```python
from services.intelligent_route_finder import IntelligentRouteFinder, RoutePreferences

route_finder = IntelligentRouteFinder(network)

# Find route between specific stops
journey = route_finder.find_optimal_route(
    origin_id="M1_TAK",      # Stop ID
    destination_id="M1_KAD",  # Stop ID
    preferences=RoutePreferences(minimize_transfers=True),
    use_astar=True  # Use A* algorithm (faster for large networks)
)

if journey:
    # Print route segments
    for segment in journey.segments:
        print(f"{segment.from_stop_name} → {segment.to_stop_name}")
        print(f"  Line: {segment.line_name}")
        print(f"  Duration: {segment.duration_minutes} min")
    
    # Print transfers
    for transfer in journey.transfers:
        print(f"Transfer: {transfer.from_stop_name} → {transfer.to_stop_name}")
        print(f"  Walking: {transfer.distance_meters}m ({transfer.duration_minutes} min)")
```

### 7. Compare Multiple Routes

```python
# Get multiple routes
plan = planner.plan_journey_simple("Taksim", "Kadıköy")

# Compare all options
if plan.alternative_journeys:
    all_journeys = [plan.primary_journey] + plan.alternative_journeys
    comparison = planner.compare_routes(all_journeys)
    
    print("Comparison:")
    print(f"Fastest: {comparison['fastest']['duration']} min")
    print(f"Least transfers: {comparison['least_transfers']['transfers']}")
    print(f"Cheapest: ₺{comparison['cheapest']['cost_tl']}")
```

### 8. Multi-Destination Planning

```python
# Plan routes to multiple destinations
destinations = ["Sultanahmet", "Kadıköy", "Beşiktaş", "Üsküdar"]

multi_plan = planner.get_multi_destination_plan(
    origin="Taksim",
    destinations=destinations
)

# Results sorted by duration
for dest, info in multi_plan['destinations'].items():
    print(f"{dest}: {info['duration_min']}min, {info['transfers']} transfers")
```

---

## 🔧 Building a Network

### From İBB Data (Production)

```python
from ibb_real_time_api import IBBOpenDataAPI
from services.route_network_builder import RouteNetworkBuilder

# Initialize İBB API
ibb_api = IBBOpenDataAPI(use_mock_data=False)

# Build network
builder = RouteNetworkBuilder(ibb_api)
network = await builder.build_network(force_rebuild=False)

# Network is automatically cached for fast loading
print(f"Network loaded: {len(network.stops)} stops")
```

### Manual Network Construction (Testing)

```python
from services.route_network_builder import (
    TransportationNetwork, TransportStop, TransportLine
)

# Create network
network = TransportationNetwork()

# Add stops
network.add_stop(TransportStop(
    stop_id="M1_TAK",
    name="Taksim",
    lat=41.0370,
    lon=28.9857,
    transport_type="metro"
))

network.add_stop(TransportStop(
    stop_id="M1_SIS",
    name="Şişli",
    lat=41.0602,
    lon=28.9879,
    transport_type="metro"
))

# Add lines
network.add_line(TransportLine(
    line_id="M1",
    name="M1 Metro",
    transport_type="metro",
    stops=["M1_TAK", "M1_SIS"]
))

# Build connections
network.build_network()

# Add transfers (optional)
network.add_transfer(
    from_stop_id="M1_TAK",
    to_stop_id="B1_TAK",
    transfer_type="same_station",
    walking_meters=50,
    duration_minutes=2
)
```

---

## 🎯 Integration with Chat System

### Replace Hardcoded Routes

**Before (Old):**
```python
# Hardcoded route
route = {
    "steps": ["Take M2 to Yenikapı", "Transfer to T1"],
    "duration": 25
}
```

**After (New):**
```python
# Dynamic routing
planner = JourneyPlanner(network)
plan = planner.plan_journey_simple(origin_location, destination_location)

if plan:
    # Generate natural language description
    description = []
    for segment in plan.primary_journey.segments:
        description.append(
            f"Take {segment.line_name} from {segment.from_stop_name} "
            f"to {segment.to_stop_name}"
        )
    
    for transfer in plan.primary_journey.transfers:
        description.append(
            f"Transfer: Walk {transfer.distance_meters}m "
            f"({transfer.duration_minutes} min)"
        )
    
    return {
        "steps": description,
        "duration": plan.primary_journey.total_duration_minutes,
        "transfers": plan.primary_journey.total_transfers,
        "cost": plan.primary_journey.estimated_cost_tl,
        "alternatives": len(plan.alternative_journeys)
    }
```

### Natural Language Query Processing

```python
def process_transport_query(user_query: str, planner: JourneyPlanner):
    """
    Process natural language transport queries
    """
    # Extract preferences from query
    preferences = RoutePreferences()
    
    if "fast" in user_query.lower() or "quick" in user_query.lower():
        preferences.minimize_time = True
        preferences.minimize_transfers = False
    
    if "no transfer" in user_query.lower() or "direct" in user_query.lower():
        preferences.minimize_transfers = True
        preferences.max_transfers = 0
    
    if "wheelchair" in user_query.lower() or "accessible" in user_query.lower():
        preferences.wheelchair_accessible = True
    
    if "walk" in user_query.lower():
        if "no walk" in user_query.lower() or "minimal walk" in user_query.lower():
            preferences.minimize_walking = True
            preferences.max_walking_meters = 200
    
    # Extract origin and destination
    # (This would use NLP to extract locations from query)
    origin = extract_origin(user_query)
    destination = extract_destination(user_query)
    
    # Plan journey
    request = JourneyRequest(
        origin=origin,
        destination=destination,
        preferences=preferences,
        include_alternatives=True
    )
    
    return planner.plan_journey(request)
```

---

## 📊 Response Formats

### JSON API Response

```python
plan = planner.plan_journey_simple("Taksim", "Kadıköy")
json_response = plan.to_dict()

# Returns:
{
    "request_time": "2025-10-24T18:00:00",
    "origin": {
        "stop_id": "M1_TAK",
        "name": "Taksim Metro",
        "transport_type": "metro",
        "location": {"lat": 41.0370, "lon": 28.9857}
    },
    "destination": {
        "stop_id": "F1_KAD",
        "name": "Kadıköy Ferry",
        "transport_type": "ferry",
        "location": {"lat": 40.9905, "lon": 29.0250}
    },
    "primary_route": {
        "total_duration": 45,
        "total_distance_km": 12.5,
        "transfers": 2,
        "walking_distance_m": 300,
        "cost_tl": 21.0,
        "transport_types": ["metro", "bus", "ferry"],
        "quality_score": 0.85,
        "segments": [
            {
                "type": "transport",
                "line": "M1 Metro",
                "transport": "metro",
                "from": "Taksim Metro",
                "to": "Yenikapı Metro",
                "stops": 4,
                "duration": 15
            },
            {
                "type": "transfer",
                "from": "Yenikapı Metro",
                "to": "Yenikapı Ferry",
                "transfer_type": "walking",
                "duration": 5,
                "walking": true
            },
            {
                "type": "transport",
                "line": "Kadıköy Ferry",
                "transport": "ferry",
                "from": "Yenikapı Ferry",
                "to": "Kadıköy Ferry",
                "stops": 1,
                "duration": 25
            }
        ]
    },
    "alternative_routes": [...],
    "total_options": 3
}
```

### Human-Readable Summary

```python
print(plan.get_summary())

# Outputs:
# Journey from Taksim Metro to Kadıköy Ferry
#
# Primary Route:
#   Duration: 45 minutes
#   Transfers: 2
#   Transport: metro, ferry
#   Quality: 85.0%
#   Cost: ₺21.00
#
# 2 Alternative Routes Available
#   Option 1: 52min, 1 transfers, metro, bus
#   Option 2: 50min, 3 transfers, metro, tram, ferry
```

---

## 🐛 Error Handling

```python
def safe_journey_planning(origin: str, destination: str):
    """
    Journey planning with proper error handling
    """
    try:
        planner = JourneyPlanner(network)
        plan = planner.plan_journey_simple(origin, destination)
        
        if plan is None:
            return {
                "success": False,
                "error": "No route found",
                "message": f"Could not find a route from {origin} to {destination}"
            }
        
        return {
            "success": True,
            "plan": plan.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Journey planning failed: {e}")
        return {
            "success": False,
            "error": "Planning failed",
            "message": str(e)
        }
```

---

## ⚡ Performance Tips

### 1. Network Caching
```python
# Network is automatically cached on first build
network = await builder.build_network(force_rebuild=False)

# Force rebuild only when data changes
network = await builder.build_network(force_rebuild=True)
```

### 2. Reuse Instances
```python
# Create once, use many times
planner = JourneyPlanner(network)
route_finder = IntelligentRouteFinder(network)
matcher = LocationMatcher(network)

# Use for multiple queries
for query in queries:
    plan = planner.plan_journey_simple(query.origin, query.dest)
```

### 3. Limit Alternatives
```python
# Fewer alternatives = faster response
plan = planner.plan_journey_simple(origin, destination)  # No alternatives

# Or specify max alternatives
request = JourneyRequest(
    origin=origin,
    destination=destination,
    include_alternatives=True,
    max_alternatives=2  # Only 2 alternatives
)
```

---

## 📚 See Also

- **`INDUSTRY_LEVEL_ROUTING_IMPLEMENTATION_COMPLETE.md`** - Complete technical documentation
- **`INDUSTRY_LEVEL_ROUTING_ENHANCEMENT_PLAN.md`** - Project roadmap
- **`test_industry_routing_system.py`** - Working examples and test cases
- **`services/intelligent_route_finder.py`** - Route finding API documentation
- **`services/location_matcher.py`** - Location matching API documentation
- **`services/journey_planner.py`** - Journey planning API documentation

---

**Need Help?** Check the test file for working examples or refer to inline code documentation.
