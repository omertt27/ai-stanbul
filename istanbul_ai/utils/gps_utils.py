"""
GPS Location Utilities for Istanbul AI System
Provides GPS-based calculations and helpers for route planning and transportation recommendations.
"""

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransportHub:
    """Represents a major transport hub in Istanbul"""
    name: str
    coordinates: Tuple[float, float]  # (latitude, longitude)
    transport_types: List[str]  # metro, tram, ferry, bus, etc.
    district: str
    description: str


# Istanbul's major transport hubs
ISTANBUL_TRANSPORT_HUBS = [
    TransportHub(
        name="Taksim",
        coordinates=(41.0369, 28.9850),
        transport_types=["metro", "funicular", "bus", "taxi"],
        district="Beyoğlu",
        description="Major transport hub with M2 metro, funicular to Kabataş, and extensive bus network"
    ),
    TransportHub(
        name="Sultanahmet",
        coordinates=(41.0086, 28.9802),
        transport_types=["tram", "bus"],
        district="Fatih",
        description="Historic center with T1 tram line connecting to major attractions"
    ),
    TransportHub(
        name="Kadıköy",
        coordinates=(40.9904, 29.0254),
        transport_types=["metro", "ferry", "bus", "dolmuş"],
        district="Kadıköy",
        description="Asian side hub with M4 metro, ferries to European side, and Marmaray connection"
    ),
    TransportHub(
        name="Beşiktaş",
        coordinates=(41.0421, 29.0067),
        transport_types=["ferry", "bus", "dolmuş"],
        district="Beşiktaş",
        description="Bosphorus ferry terminal with connections to Asian side and islands"
    ),
    TransportHub(
        name="Eminönü",
        coordinates=(41.0172, 28.9736),
        transport_types=["tram", "ferry", "bus"],
        district="Fatih",
        description="Historic transport hub with ferries, T1 tram, and access to Grand Bazaar"
    ),
    TransportHub(
        name="Kabataş",
        coordinates=(41.0287, 28.9838),
        transport_types=["tram", "funicular", "ferry", "bus"],
        district="Beyoğlu",
        description="Major interchange with T1 tram, funicular to Taksim, and ferry terminal"
    ),
    TransportHub(
        name="Üsküdar",
        coordinates=(41.0255, 29.0145),
        transport_types=["metro", "ferry", "bus", "marmaray"],
        district="Üsküdar",
        description="Asian side ferry terminal with M5 metro and Marmaray tunnel connection"
    ),
    TransportHub(
        name="Mecidiyeköy",
        coordinates=(41.0695, 28.9950),
        transport_types=["metro", "metrobüs", "bus"],
        district="Şişli",
        description="Business district hub with M2 metro and metrobüs connections"
    ),
    TransportHub(
        name="Yenikapı",
        coordinates=(41.0029, 28.9506),
        transport_types=["metro", "marmaray", "ferry"],
        district="Fatih",
        description="Major interchange with M1, M2 metro lines and Marmaray tunnel"
    ),
    TransportHub(
        name="Karaköy",
        coordinates=(41.0250, 28.9741),
        transport_types=["tram", "ferry", "funicular", "bus"],
        district="Beyoğlu",
        description="Historic port area with T1 tram, ferries, and Tünel funicular"
    )
]


def calculate_distance(gps1: Tuple[float, float], gps2: Tuple[float, float]) -> float:
    """
    Calculate distance between two GPS coordinates using Haversine formula.
    
    Args:
        gps1: First GPS coordinate (latitude, longitude)
        gps2: Second GPS coordinate (latitude, longitude)
    
    Returns:
        Distance in meters
    """
    lat1, lon1 = gps1
    lat2, lon2 = gps2
    
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance


def estimate_walking_time(distance_m: float, pace: str = "normal") -> int:
    """
    Estimate walking time based on distance.
    
    Args:
        distance_m: Distance in meters
        pace: Walking pace - "slow" (4 km/h), "normal" (5 km/h), "fast" (6 km/h)
    
    Returns:
        Estimated walking time in minutes
    """
    # Walking speeds in meters per minute
    pace_speeds = {
        "slow": 66.67,    # 4 km/h
        "normal": 83.33,  # 5 km/h
        "fast": 100.0     # 6 km/h
    }
    
    speed = pace_speeds.get(pace, pace_speeds["normal"])
    return int(math.ceil(distance_m / speed))


def find_nearest_hub(user_gps: Tuple[float, float], 
                     hubs: Optional[List[TransportHub]] = None,
                     max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Find nearest transport hubs to user's GPS location.
    
    Args:
        user_gps: User's GPS coordinates (latitude, longitude)
        hubs: List of transport hubs to search (defaults to ISTANBUL_TRANSPORT_HUBS)
        max_results: Maximum number of results to return
    
    Returns:
        List of nearest hubs with distance and walking time information
    """
    if hubs is None:
        hubs = ISTANBUL_TRANSPORT_HUBS
    
    nearest_hubs = []
    
    for hub in hubs:
        distance_m = calculate_distance(user_gps, hub.coordinates)
        walking_time = estimate_walking_time(distance_m)
        
        nearest_hubs.append({
            'name': hub.name,
            'coordinates': hub.coordinates,
            'transport_types': hub.transport_types,
            'district': hub.district,
            'description': hub.description,
            'distance_m': distance_m,
            'distance_km': round(distance_m / 1000, 2),
            'walking_time_minutes': walking_time,
            'is_walkable': distance_m <= 1500  # Within 15-20 min walk
        })
    
    # Sort by distance and return top results
    nearest_hubs.sort(key=lambda x: x['distance_m'])
    return nearest_hubs[:max_results]


def format_gps_coordinates(gps: Tuple[float, float], precision: int = 4) -> str:
    """
    Format GPS coordinates for display.
    
    Args:
        gps: GPS coordinates (latitude, longitude)
        precision: Number of decimal places
    
    Returns:
        Formatted coordinate string
    """
    lat, lon = gps
    return f"{lat:.{precision}f}°N, {lon:.{precision}f}°E"


def get_transport_recommendations(
    user_gps: Tuple[float, float],
    destination_gps: Optional[Tuple[float, float]] = None,
    destination_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get smart transport mode recommendations based on GPS location.
    
    Args:
        user_gps: User's GPS coordinates
        destination_gps: Destination GPS coordinates (optional)
        destination_name: Destination name (optional)
    
    Returns:
        Dictionary with transport recommendations
    """
    # Find nearest hubs
    nearest_hubs = find_nearest_hub(user_gps, max_results=3)
    nearest = nearest_hubs[0] if nearest_hubs else None
    
    recommendations = {
        'user_location': {
            'coordinates': user_gps,
            'formatted': format_gps_coordinates(user_gps)
        },
        'nearest_hubs': nearest_hubs,
        'primary_hub': nearest,
        'transport_options': []
    }
    
    if not nearest:
        return recommendations
    
    distance_to_hub = nearest['distance_m']
    
    # Walking option (if hub is close)
    if distance_to_hub <= 1500:
        recommendations['transport_options'].append({
            'mode': 'walking',
            'priority': 1,
            'to_hub': nearest['name'],
            'duration_minutes': nearest['walking_time_minutes'],
            'cost_tl': 0,
            'description': f"Walk {nearest['distance_km']} km to {nearest['name']}",
            'recommendation': "Best for short distances and enjoying the city"
        })
    
    # Taxi option (always available)
    taxi_time = max(5, int(distance_to_hub / 300))  # Approx 18 km/h in city traffic
    taxi_cost = max(45, 15 + (distance_to_hub / 1000) * 10)  # Base + distance rate
    
    recommendations['transport_options'].append({
        'mode': 'taxi',
        'priority': 2 if distance_to_hub <= 1500 else 1,
        'to_hub': nearest['name'],
        'duration_minutes': taxi_time,
        'cost_tl': round(taxi_cost, 0),
        'description': f"Taxi to {nearest['name']} (~{taxi_time} min)",
        'recommendation': "Fast and convenient, especially with luggage"
    })
    
    # Public transport from hub
    if nearest['transport_types']:
        recommendations['transport_options'].append({
            'mode': 'public_transport',
            'priority': 1,
            'from_hub': nearest['name'],
            'available_types': nearest['transport_types'],
            'cost_tl': 15,  # Istanbulkart single journey
            'description': f"Public transport from {nearest['name']}",
            'recommendation': "Most economical and reliable option"
        })
    
    # If destination provided, calculate total distance
    if destination_gps:
        total_distance = calculate_distance(user_gps, destination_gps)
        recommendations['destination'] = {
            'name': destination_name,
            'coordinates': destination_gps,
            'total_distance_km': round(total_distance / 1000, 2),
            'direct_walking_time': estimate_walking_time(total_distance)
        }
        
        # Direct walking if very close
        if total_distance <= 2000:
            recommendations['transport_options'].insert(0, {
                'mode': 'direct_walking',
                'priority': 1,
                'duration_minutes': estimate_walking_time(total_distance),
                'cost_tl': 0,
                'description': f"Walk directly to {destination_name}",
                'recommendation': "Perfect weather for a pleasant walk!"
            })
    
    # Sort by priority
    recommendations['transport_options'].sort(key=lambda x: x['priority'])
    
    return recommendations


def is_within_istanbul(gps: Tuple[float, float]) -> bool:
    """
    Check if GPS coordinates are within Istanbul boundaries.
    
    Istanbul approximate boundaries:
    - Latitude: 40.8 to 41.4
    - Longitude: 28.5 to 29.5
    
    Args:
        gps: GPS coordinates (latitude, longitude)
    
    Returns:
        True if within Istanbul, False otherwise
    """
    lat, lon = gps
    return (40.8 <= lat <= 41.4) and (28.5 <= lon <= 29.5)


def format_distance(distance_m: float) -> str:
    """
    Format distance in human-readable format.
    
    Args:
        distance_m: Distance in meters
    
    Returns:
        Formatted distance string
    """
    if distance_m < 1000:
        return f"{int(distance_m)}m"
    else:
        return f"{distance_m / 1000:.1f} km"


def get_walking_directions_summary(user_gps: Tuple[float, float], hub: TransportHub) -> str:
    """
    Generate simple walking directions summary to a transport hub.
    
    Args:
        user_gps: User's GPS coordinates
        hub: Target transport hub
    
    Returns:
        Walking directions summary
    """
    distance_m = calculate_distance(user_gps, hub.coordinates)
    walking_time = estimate_walking_time(distance_m)
    
    # Determine general direction
    user_lat, user_lon = user_gps
    hub_lat, hub_lon = hub.coordinates
    
    lat_diff = hub_lat - user_lat
    lon_diff = hub_lon - user_lon
    
    # Primary direction (N/S)
    if abs(lat_diff) > abs(lon_diff):
        primary = "north" if lat_diff > 0 else "south"
        secondary = "east" if lon_diff > 0 else "west"
    else:
        primary = "east" if lon_diff > 0 else "west"
        secondary = "north" if lat_diff > 0 else "south"
    
    direction = f"{primary}-{secondary}" if abs(lat_diff) > 0.001 and abs(lon_diff) > 0.001 else primary
    
    return f"Head {direction} for approximately {format_distance(distance_m)} (~{walking_time} min walk)"


# Example usage and testing
if __name__ == "__main__":
    # Test with sample GPS location (near Taksim)
    user_location = (41.0350, 28.9830)
    
    print("🧪 Testing GPS Utilities\n")
    print(f"User Location: {format_gps_coordinates(user_location)}")
    print(f"Within Istanbul: {is_within_istanbul(user_location)}\n")
    
    # Test nearest hub finding
    print("📍 Nearest Transport Hubs:")
    nearest = find_nearest_hub(user_location, max_results=3)
    for i, hub in enumerate(nearest, 1):
        print(f"\n{i}. {hub['name']} ({hub['district']})")
        print(f"   Distance: {format_distance(hub['distance_m'])}")
        print(f"   Walking: ~{hub['walking_time_minutes']} minutes")
        print(f"   Transport: {', '.join(hub['transport_types'])}")
        print(f"   Walkable: {'✅ Yes' if hub['is_walkable'] else '❌ Too far'}")
    
    # Test transport recommendations
    print("\n\n🚇 Transport Recommendations:")
    recommendations = get_transport_recommendations(user_location)
    
    for i, option in enumerate(recommendations['transport_options'], 1):
        print(f"\n{i}. {option['mode'].upper()}")
        print(f"   {option['description']}")
        print(f"   Duration: ~{option.get('duration_minutes', 'N/A')} min")
        print(f"   Cost: {option.get('cost_tl', 0)} TL")
        print(f"   💡 {option['recommendation']}")


# ==============================================================================
# ATTRACTION-RELATED FUNCTIONS (Sprint 2)
# ==============================================================================

def load_attractions_database() -> List[Dict]:
    """
    Load attractions from the database file.
    
    Returns:
        List of attraction dictionaries
    """
    import json
    import os
    
    try:
        # Get the path to the attractions database
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), 'data')
        db_path = os.path.join(data_dir, 'attractions_database.json')
        
        if os.path.exists(db_path):
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"✅ Loaded {len(data['attractions'])} attractions from database")
                return data['attractions']
        else:
            logger.warning(f"⚠️ Attractions database not found at {db_path}")
            return []
    except Exception as e:
        logger.error(f"❌ Error loading attractions database: {e}")
        return []


def get_attractions_in_radius(
    user_lat: float,
    user_lon: float,
    radius_km: float = 2.0,
    categories: Optional[List[str]] = None,
    max_results: int = 10
) -> List[Dict]:
    """
    Find all attractions within a specified radius of user's location.
    
    Args:
        user_lat: User's latitude
        user_lon: User's longitude
        radius_km: Search radius in kilometers (default: 2.0)
        categories: Filter by categories (e.g., ['culture', 'food'])
        max_results: Maximum number of results to return
        
    Returns:
        List of attractions sorted by distance, each with distance_km added
    """
    attractions = load_attractions_database()
    
    if not attractions:
        return []
    
    nearby_attractions = []
    
    for attraction in attractions:
        attr_lat, attr_lon = attraction['gps']
        distance_km = calculate_distance(user_lat, user_lon, attr_lat, attr_lon)
        
        # Check if within radius
        if distance_km <= radius_km:
            # Filter by categories if specified
            if categories:
                # Check if attraction has any of the requested categories
                if not any(cat in attraction['category'] for cat in categories):
                    continue
            
            # Add distance information
            attraction_with_distance = attraction.copy()
            attraction_with_distance['distance_km'] = distance_km
            attraction_with_distance['walking_time_min'] = estimate_walking_time(distance_km)
            
            nearby_attractions.append(attraction_with_distance)
    
    # Sort by distance
    nearby_attractions.sort(key=lambda x: x['distance_km'])
    
    # Limit results
    return nearby_attractions[:max_results]


def format_attraction_response(attractions: List[Dict], user_location: Tuple[float, float]) -> str:
    """
    Format a list of attractions into a user-friendly response.
    
    Args:
        attractions: List of attractions with distance information
        user_location: (lat, lon) tuple for user's current location
        
    Returns:
        Formatted string response
    """
    if not attractions:
        return (
            "🗺️ **No attractions found nearby**\n\n"
            "Try:\n"
            "• Expanding your search radius\n"
            "• Exploring a different area\n"
            "• Asking for specific types of attractions"
        )
    
    user_lat, user_lon = user_location
    response_parts = []
    
    # Header
    response_parts.append(
        f"📍 **Attractions Near You** ({format_gps_coordinates(user_lat, user_lon)})\n"
        f"Found {len(attractions)} attraction{'s' if len(attractions) > 1 else ''} nearby:\n"
    )
    
    # List each attraction
    for i, attr in enumerate(attractions, 1):
        emoji_map = {
            'culture': '🏛️',
            'history': '🏺',
            'shopping': '🛍️',
            'food': '🍽️',
            'nightlife': '🌃',
            'nature': '🌳',
            'activity': '🎯',
            'museum': '🖼️',
            'religious': '🕌',
            'sightseeing': '👀'
        }
        
        # Get emoji for primary category
        emoji = emoji_map.get(attr['category'][0], '📍')
        
        response_parts.append(
            f"\n**{i}. {emoji} {attr['name']}**"
        )
        
        # Distance and walking time
        distance = attr['distance_km']
        walking_time = attr['walking_time_min']
        
        if distance < 0.5:
            distance_text = "Very close!"
        elif distance < 1.0:
            distance_text = f"{distance:.1f} km away"
        else:
            distance_text = f"{distance:.1f} km away"
        
        response_parts.append(
            f"📏 {distance_text} • 🚶 ~{walking_time} min walk"
        )
        
        # Description
        response_parts.append(f"ℹ️ {attr['description']}")
        
        # Entry fee and hours
        if attr['entry_fee_tl'] > 0:
            response_parts.append(f"💵 {attr['entry_fee_tl']} TL")
        else:
            response_parts.append("💵 Free entry")
        
        response_parts.append(f"🕐 {attr['opening_hours']}")
        
        # Rating
        if attr.get('rating'):
            stars = '⭐' * int(attr['rating'])
            response_parts.append(f"⭐ {attr['rating']}/5.0 {stars}")
        
        # Best time to visit
        if attr.get('best_time'):
            response_parts.append(f"💡 Best time: {attr['best_time']}")
    
    # Footer with tips
    response_parts.append(
        f"\n\n💡 **Tips:**\n"
        f"• Use public transport for attractions > 2 km away\n"
        f"• Visit popular sites early morning or late afternoon\n"
        f"• Get an Istanbulkart for easy transport access\n\n"
        f"Ask me for directions to any of these attractions!"
    )
    
    return "\n".join(response_parts)


def get_attraction_categories() -> List[str]:
    """
    Get list of available attraction categories.
    
    Returns:
        List of category strings
    """
    return [
        'culture', 'history', 'shopping', 'food', 
        'nightlife', 'nature', 'activity', 'museum',
        'architecture', 'religious', 'sightseeing'
    ]


def filter_attractions_by_preferences(
    attractions: List[Dict],
    preferences: Dict[str, Any]
) -> List[Dict]:
    """
    Filter attractions based on user preferences.
    
    Args:
        attractions: List of attractions to filter
        preferences: Dict with filtering criteria:
            - max_entry_fee: Maximum entry fee in TL
            - min_rating: Minimum rating (0-5)
            - exclude_categories: Categories to exclude
            - max_duration: Maximum visit duration in minutes
            
    Returns:
        Filtered list of attractions
    """
    filtered = attractions.copy()
    
    # Filter by max entry fee
    if 'max_entry_fee' in preferences:
        max_fee = preferences['max_entry_fee']
        filtered = [a for a in filtered if a['entry_fee_tl'] <= max_fee]
    
    # Filter by minimum rating
    if 'min_rating' in preferences:
        min_rating = preferences['min_rating']
        filtered = [a for a in filtered if a.get('rating', 0) >= min_rating]
    
    # Exclude certain categories
    if 'exclude_categories' in preferences:
        exclude = preferences['exclude_categories']
        filtered = [
            a for a in filtered 
            if not any(cat in a['category'] for cat in exclude)
        ]
    
    # Filter by visit duration
    if 'max_duration' in preferences:
        max_dur = preferences['max_duration']
        filtered = [a for a in filtered if a['visit_duration_min'] <= max_dur]
    
    return filtered
