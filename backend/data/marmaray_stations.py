"""
Complete Marmaray Station Data
Marmaray is the undersea rail tunnel connecting European and Asian Istanbul

Updated: November 5, 2025
Source: Official İBB/TCDD transportation data
"""

from typing import Dict, List, Any

# Complete Marmaray line with all stations
MARMARAY_STATIONS = {
    'european_side': [
        # Western European side stations
        {'name': 'Halkalı', 'lat': 41.0058, 'lon': 28.6756, 'district': 'Küçükçekmece'},
        {'name': 'Mustafa Kemal', 'lat': 41.0045, 'lon': 28.7125, 'district': 'Küçükçekmece'},
        {'name': 'Florya Akvaryum', 'lat': 40.9858, 'lon': 28.7845, 'district': 'Bakırköy'},
        {'name': 'Florya', 'lat': 40.9831, 'lon': 28.7911, 'district': 'Bakırköy'},
        {'name': 'Yeşilyurt', 'lat': 40.9923, 'lon': 28.8156, 'district': 'Bakırköy'},
        {'name': 'Ataköy', 'lat': 40.9778, 'lon': 28.8489, 'district': 'Bakırköy'},
        {'name': 'Yenimahalle', 'lat': 40.9856, 'lon': 28.8667, 'district': 'Bakırköy'},
        {'name': 'Bakırköy', 'lat': 40.9781, 'lon': 28.8733, 'district': 'Bakırköy'},
        {'name': 'Yenibosna', 'lat': 40.9931, 'lon': 28.8289, 'district': 'Bahçelievler'},
        {'name': 'Zeytinburnu', 'lat': 41.0019, 'lon': 28.9025, 'district': 'Zeytinburnu'},
        {'name': 'Kazlıçeşme', 'lat': 40.9944, 'lon': 28.9278, 'district': 'Zeytinburnu'},
        
        # Central European side - approaching tunnel
        {'name': 'Yenikapı', 'lat': 41.0046, 'lon': 28.9512, 'district': 'Fatih'},
        {'name': 'Sirkeci', 'lat': 41.0171, 'lon': 28.9765, 'district': 'Eminönü'},
    ],
    
    'tunnel': [
        # Undersea tunnel stations
        {'name': 'Yenikapı', 'lat': 41.0046, 'lon': 28.9512, 'district': 'Fatih', 'type': 'tunnel_entrance_eu'},
        {'name': 'Sirkeci', 'lat': 41.0171, 'lon': 28.9765, 'district': 'Eminönü', 'type': 'tunnel_entrance_eu'},
        {'name': 'Üsküdar', 'lat': 41.0226, 'lon': 29.0150, 'district': 'Üsküdar', 'type': 'tunnel_exit_as'},
        {'name': 'Ayrılık Çeşmesi', 'lat': 41.0165, 'lon': 29.0252, 'district': 'Üsküdar', 'type': 'tunnel_exit_as'},
    ],
    
    'asian_side': [
        # Central Asian side - after tunnel
        {'name': 'Üsküdar', 'lat': 41.0226, 'lon': 29.0150, 'district': 'Üsküdar'},
        {'name': 'Ayrılık Çeşmesi', 'lat': 41.0165, 'lon': 29.0252, 'district': 'Üsküdar'},
        
        # Eastern Asian side stations
        {'name': 'Söğütlüçeşme', 'lat': 41.0178, 'lon': 29.0403, 'district': 'Kadıköy'},
        {'name': 'Feneryolu', 'lat': 40.9678, 'lon': 29.0531, 'district': 'Kadıköy'},
        {'name': 'Göztepe', 'lat': 40.9756, 'lon': 29.0689, 'district': 'Kadıköy'},
        {'name': 'Erenköy', 'lat': 40.9739, 'lon': 29.0819, 'district': 'Kadıköy'},
        {'name': 'Suadiye', 'lat': 40.9672, 'lon': 29.0947, 'district': 'Kadıköy'},
        {'name': 'Bostancı', 'lat': 40.9631, 'lon': 29.0882, 'district': 'Kadıköy'},
        {'name': 'Küçükyalı', 'lat': 40.9456, 'lon': 29.1147, 'district': 'Maltepe'},
        {'name': 'İdealtepe', 'lat': 40.9389, 'lon': 29.1356, 'district': 'Maltepe'},
        {'name': 'Maltepe', 'lat': 40.9317, 'lon': 29.1489, 'district': 'Maltepe'},
        {'name': 'Cevizli', 'lat': 40.9178, 'lon': 29.1678, 'district': 'Maltepe'},
        {'name': 'Atalar', 'lat': 40.9067, 'lon': 29.1856, 'district': 'Kartal'},
        {'name': 'Kartal', 'lat': 40.8978, 'lon': 29.1978, 'district': 'Kartal'},
        {'name': 'Yakacık-Pendik', 'lat': 40.8856, 'lon': 29.2167, 'district': 'Pendik'},
        {'name': 'Pendik', 'lat': 40.8775, 'lon': 29.2336, 'district': 'Pendik'},
        {'name': 'Güzelyalı', 'lat': 40.8667, 'lon': 29.2489, 'district': 'Pendik'},
        {'name': 'Esenkent', 'lat': 40.8556, 'lon': 29.2678, 'district': 'Pendik'},
        {'name': 'Çayırova', 'lat': 40.8278, 'lon': 29.3589, 'district': 'Kocaeli'},
        {'name': 'Tersane', 'lat': 40.8167, 'lon': 29.3789, 'district': 'Kocaeli'},
        {'name': 'Gebze', 'lat': 40.7995, 'lon': 29.4305, 'district': 'Kocaeli'},
    ]
}

# Key transfer stations (connections to other lines)
MARMARAY_TRANSFER_STATIONS = {
    'Yenikapı': {
        'connections': ['M1A', 'M1B', 'M2'],
        'transfer_time_minutes': 5,
        'description': 'Major hub - Metro M1A/M1B (Airport), M2 (Taksim)'
    },
    'Sirkeci': {
        'connections': ['T1', 'Ferry'],
        'transfer_time_minutes': 3,
        'description': 'Tram T1 (Sultanahmet), Eminönü Ferry Terminal'
    },
    'Üsküdar': {
        'connections': ['M5', 'Ferry'],
        'transfer_time_minutes': 4,
        'description': 'Metro M5 (Ümraniye), Üsküdar Ferry Terminal'
    },
    'Ayrılık Çeşmesi': {
        'connections': ['M4'],
        'transfer_time_minutes': 3,
        'description': 'Metro M4 (Kadıköy-Tavşantepe)'
    },
    'Bakırköy': {
        'connections': ['M1A'],
        'transfer_time_minutes': 5,
        'description': 'Metro M1A (Airport-Yenikapı)'
    },
    'Halkalı': {
        'connections': ['Regional Trains'],
        'transfer_time_minutes': 5,
        'description': 'Western terminus, connections to regional trains'
    },
    'Gebze': {
        'connections': ['Regional Trains'],
        'transfer_time_minutes': 5,
        'description': 'Eastern terminus, connections to regional trains and Kocaeli'
    }
}

# Travel times between key stations (in minutes)
MARMARAY_TRAVEL_TIMES = {
    ('Yenikapı', 'Üsküdar'): 4,  # Undersea crossing
    ('Sirkeci', 'Üsküdar'): 3,  # Shortest undersea
    ('Yenikapı', 'Ayrılık Çeşmesi'): 6,
    ('Sirkeci', 'Ayrılık Çeşmesi'): 5,
    ('Halkalı', 'Yenikapı'): 32,
    ('Yenikapı', 'Gebze'): 73,  # Full line
    ('Halkalı', 'Gebze'): 105,  # Complete west to east
    ('Kadıköy', 'Taksim'): 35,  # via Ayrılık Çeşmesi + Yenikapı transfer
}

# Operational information
MARMARAY_INFO = {
    'total_length_km': 76.6,
    'tunnel_length_km': 13.6,
    'undersea_section_km': 1.4,
    'depth_meters': 60,  # Below sea level at deepest point
    'operational_hours': {
        'weekday_start': '06:00',
        'weekday_end': '00:30',
        'weekend_start': '06:30',
        'weekend_end': '00:30'
    },
    'frequency_minutes': {
        'peak': 5,  # Every 5 minutes during rush hour
        'off_peak': 10,  # Every 10 minutes
        'late_night': 15  # Every 15 minutes after 22:00
    },
    'advantages': [
        'Weather-independent (underground)',
        'Avoids Bosphorus traffic',
        'Fixed schedule, reliable timing',
        'Fast underwater crossing (4 minutes)',
        'Connects to major metro lines',
        'Most direct route across Bosphorus'
    ],
    'when_to_use': [
        'Rainy or snowy weather',
        'Heavy traffic hours',
        'Quick cross-Bosphorus transit',
        'Connecting European and Asian metro lines',
        'Airport to Asian side travel'
    ]
}


def get_all_stations() -> List[Dict[str, Any]]:
    """Get complete list of all Marmaray stations"""
    all_stations = []
    all_stations.extend(MARMARAY_STATIONS['european_side'])
    all_stations.extend(MARMARAY_STATIONS['asian_side'])
    # Don't duplicate tunnel stations as they're already in european/asian lists
    return all_stations


def get_tunnel_stations() -> List[Dict[str, Any]]:
    """Get only the undersea tunnel stations"""
    return MARMARAY_STATIONS['tunnel']


def find_nearest_marmaray_station(lat: float, lon: float) -> Dict[str, Any]:
    """
    Find nearest Marmaray station to given coordinates
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dict with nearest station info and distance
    """
    import math
    
    def distance(lat1, lon1, lat2, lon2):
        """Calculate distance in meters using Haversine formula"""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    all_stations = get_all_stations()
    nearest = None
    min_distance = float('inf')
    
    for station in all_stations:
        dist = distance(lat, lon, station['lat'], station['lon'])
        if dist < min_distance:
            min_distance = dist
            nearest = station.copy()
            nearest['distance_meters'] = int(dist)
    
    return nearest


def crosses_bosphorus(origin_lon: float, dest_lon: float) -> bool:
    """
    Check if route crosses the Bosphorus
    
    Args:
        origin_lon: Origin longitude
        dest_lon: Destination longitude
        
    Returns:
        True if route crosses Bosphorus
    """
    BOSPHORUS_LON = 29.0  # Approximate longitude of Bosphorus
    
    # Check if origin and destination are on opposite sides
    return (origin_lon < BOSPHORUS_LON and dest_lon > BOSPHORUS_LON) or \
           (origin_lon > BOSPHORUS_LON and dest_lon < BOSPHORUS_LON)


def estimate_marmaray_time(origin_lat: float, origin_lon: float, 
                          dest_lat: float, dest_lon: float) -> int:
    """
    Estimate Marmaray travel time between two points
    
    Args:
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        dest_lat: Destination latitude
        dest_lon: Destination longitude
        
    Returns:
        Estimated time in minutes
    """
    # Find nearest stations
    origin_station = find_nearest_marmaray_station(origin_lat, origin_lon)
    dest_station = find_nearest_marmaray_station(dest_lat, dest_lon)
    
    # Check if we have a direct travel time
    time_key = (origin_station['name'], dest_station['name'])
    if time_key in MARMARAY_TRAVEL_TIMES:
        base_time = MARMARAY_TRAVEL_TIMES[time_key]
    else:
        # Estimate: ~2 minutes per station + undersea crossing bonus
        origin_idx = next((i for i, s in enumerate(get_all_stations()) 
                          if s['name'] == origin_station['name']), 0)
        dest_idx = next((i for i, s in enumerate(get_all_stations()) 
                        if s['name'] == dest_station['name']), 0)
        
        stations_between = abs(dest_idx - origin_idx)
        base_time = stations_between * 2  # ~2 min per station
        
        # Add time if crossing Bosphorus
        if crosses_bosphorus(origin_lon, dest_lon):
            base_time += 4  # Undersea tunnel time
    
    # Add walking time to/from stations (assume ~5 min each)
    walking_time = 10
    
    # Add waiting time (average half the frequency)
    waiting_time = MARMARAY_INFO['frequency_minutes']['peak'] // 2
    
    total_time = base_time + walking_time + waiting_time
    
    return total_time


def get_marmaray_recommendation(origin_lat: float, origin_lon: float,
                               dest_lat: float, dest_lon: float,
                               weather_conditions: str = None) -> Dict[str, Any]:
    """
    Get Marmaray recommendation with routing details
    
    Args:
        origin_lat: Origin latitude
        origin_lon: Origin longitude
        dest_lat: Destination latitude
        dest_lon: Destination longitude
        weather_conditions: Current weather (e.g., 'rainy', 'clear')
        
    Returns:
        Dict with recommendation details
    """
    crosses = crosses_bosphorus(origin_lon, dest_lon)
    
    if not crosses:
        return {
            'use_marmaray': False,
            'reason': 'Route does not cross Bosphorus'
        }
    
    origin_station = find_nearest_marmaray_station(origin_lat, origin_lon)
    dest_station = find_nearest_marmaray_station(dest_lat, dest_lon)
    travel_time = estimate_marmaray_time(origin_lat, origin_lon, dest_lat, dest_lon)
    
    # Determine strength of recommendation based on weather
    recommendation_strength = 'recommended'
    if weather_conditions:
        if 'rain' in weather_conditions.lower() or 'snow' in weather_conditions.lower():
            recommendation_strength = 'highly_recommended'
        elif 'clear' in weather_conditions.lower() or 'sunny' in weather_conditions.lower():
            recommendation_strength = 'alternative'  # Ferry might be nice in good weather
    
    return {
        'use_marmaray': True,
        'origin_station': origin_station,
        'dest_station': dest_station,
        'travel_time_minutes': travel_time,
        'undersea_crossing_time': 4,
        'recommendation_strength': recommendation_strength,
        'advantages': MARMARAY_INFO['advantages'],
        'transfer_info': {
            'origin': MARMARAY_TRANSFER_STATIONS.get(origin_station['name']),
            'destination': MARMARAY_TRANSFER_STATIONS.get(dest_station['name'])
        },
        'frequency_minutes': MARMARAY_INFO['frequency_minutes']['peak'],
        'weather_independent': True
    }


# Example usage
if __name__ == "__main__":
    print("🚇 Marmaray Station Database")
    print("=" * 60)
    
    print(f"\n📍 Total Stations: {len(get_all_stations())}")
    print(f"   European Side: {len(MARMARAY_STATIONS['european_side'])}")
    print(f"   Asian Side: {len(MARMARAY_STATIONS['asian_side'])}")
    print(f"   Undersea Tunnel: {len(MARMARAY_STATIONS['tunnel'])}")
    
    print(f"\n🔄 Transfer Stations: {len(MARMARAY_TRANSFER_STATIONS)}")
    for station, info in MARMARAY_TRANSFER_STATIONS.items():
        print(f"   {station}: {', '.join(info['connections'])}")
    
    print(f"\n📊 Route Info:")
    print(f"   Total Length: {MARMARAY_INFO['total_length_km']} km")
    print(f"   Undersea Section: {MARMARAY_INFO['undersea_section_km']} km")
    print(f"   Depth: {MARMARAY_INFO['depth_meters']} meters below sea level")
    
    print(f"\n⏱️  Sample Travel Times:")
    print(f"   Yenikapı → Üsküdar (undersea): 4 minutes")
    print(f"   Halkalı → Gebze (full line): 105 minutes")
    
    # Test: Find nearest station to Taksim
    print(f"\n🎯 Example: Nearest station to Taksim (41.0370, 28.9857)")
    nearest = find_nearest_marmaray_station(41.0370, 28.9857)
    print(f"   {nearest['name']} - {nearest['distance_meters']}m away")
    
    # Test: Taksim to Kadıköy
    print(f"\n🗺️  Example: Taksim → Kadıköy recommendation")
    recommendation = get_marmaray_recommendation(
        41.0370, 28.9857,  # Taksim
        40.9903, 29.0267,  # Kadıköy
        weather_conditions='rainy'
    )
    print(f"   Use Marmaray: {recommendation['use_marmaray']}")
    print(f"   Strength: {recommendation['recommendation_strength']}")
    print(f"   Estimated Time: {recommendation['travel_time_minutes']} minutes")
    print(f"   Route: {recommendation['origin_station']['name']} → {recommendation['dest_station']['name']}")
