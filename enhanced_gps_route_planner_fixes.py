#!/usr/bin/env python3
"""
Critical Missing Methods for EnhancedGPSRoutePlanner
These methods need to be added to the main file to make it functional.
"""

import math
from typing import Dict, List, Optional
from datetime import datetime
from enhanced_gps_route_planner import (
    GPSLocation, PersonalizedWaypoint, TransportMode
)


def _calculate_distance(loc1: GPSLocation, loc2: GPSLocation) -> float:
    """
    Calculate distance between two GPS points using Haversine formula
    
    Args:
        loc1: First GPS location
        loc2: Second GPS location
        
    Returns:
        Distance in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1 = math.radians(loc1.latitude)
    lon1 = math.radians(loc1.longitude)
    lat2 = math.radians(loc2.latitude)
    lon2 = math.radians(loc2.longitude)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    distance = R * c
    
    return distance


def _find_nearby_pois(
    self, 
    location: GPSLocation, 
    radius_km: float = 5.0,
    interests: List[str] = None
) -> List[Dict]:
    """
    Find points of interest within radius matching user interests
    
    Args:
        location: Current GPS location
        radius_km: Search radius in kilometers
        interests: List of user interests
        
    Returns:
        List of nearby POIs with distance and interest match scores
    """
    if interests is None:
        interests = []
    
    nearby_pois = []
    
    # Search through all POI categories
    for category_name, pois in self.poi_database.items():
        for poi in pois:
            # Calculate distance
            poi_location = poi.get('location')
            if not poi_location:
                continue
            
            distance = self._calculate_distance(location, poi_location)
            
            # Check if within radius
            if distance <= radius_km:
                # Calculate interest match
                interest_match = self._calculate_interest_match(poi, interests)
                
                # Create enhanced POI entry
                enhanced_poi = {
                    'name': poi.get('name'),
                    'location': poi_location,
                    'category': poi.get('category'),
                    'popularity': poi.get('popularity', 0.5),
                    'visit_duration_min': poi.get('visit_duration_min', 60),
                    'district': poi.get('district', ''),
                    'distance_km': distance,
                    'interest_match': interest_match,
                    'category_type': category_name
                }
                
                nearby_pois.append(enhanced_poi)
    
    # Sort by composite score (interest match and popularity, weighted by distance)
    nearby_pois.sort(
        key=lambda x: (x['interest_match'] * 0.6 + x['popularity'] * 0.4) * (1.0 / (1.0 + x['distance_km'])),
        reverse=True
    )
    
    return nearby_pois


def _calculate_interest_match(poi: Dict, interests: List[str]) -> float:
    """
    Calculate how well a POI matches user interests
    
    Args:
        poi: POI dictionary with category information
        interests: List of user interest keywords
        
    Returns:
        Match score from 0.0 to 1.0
    """
    if not interests:
        return 0.5  # Default neutral score
    
    poi_category = poi.get('category', '').lower()
    poi_name = poi.get('name', '').lower()
    
    # Check for exact category matches
    category_matches = sum(1 for interest in interests if interest.lower() in poi_category)
    name_matches = sum(1 for interest in interests if interest.lower() in poi_name)
    
    # Calculate match score
    max_possible_matches = len(interests)
    total_matches = category_matches + (name_matches * 0.5)  # Name matches count less
    
    match_score = min(total_matches / max_possible_matches, 1.0)
    
    # Boost score for popular POIs
    popularity_boost = poi.get('popularity', 0.5) * 0.2
    
    final_score = min(match_score + popularity_boost, 1.0)
    
    return final_score


def _score_pois_for_user(
    self,
    pois: List[Dict],
    user_profile: Dict,
    current_location: GPSLocation
) -> List[Dict]:
    """
    Score and rank POIs based on user preferences and context
    
    Args:
        pois: List of POIs to score
        user_profile: User preference profile
        current_location: Current location for distance weighting
        
    Returns:
        Scored and sorted list of POIs
    """
    scored_pois = []
    
    # Get user preferences
    preferred_categories = user_profile.get('interests', [])
    budget = user_profile.get('budget_preference', 'medium')
    activity_level = user_profile.get('activity_level', 'medium')
    
    # Activity level affects preferred distances
    max_comfortable_distance = {
        'low': 2.0,
        'medium': 5.0,
        'high': 10.0
    }.get(activity_level, 5.0)
    
    for poi in pois:
        # Base score components
        interest_score = poi.get('interest_match', 0.5)
        popularity_score = poi.get('popularity', 0.5)
        distance_km = poi.get('distance_km', 1.0)
        
        # Distance penalty (closer is better)
        distance_score = max(0, 1.0 - (distance_km / max_comfortable_distance))
        
        # Time of day suitability (museums better during day, restaurants during meal times)
        time_score = self._calculate_time_suitability(poi)
        
        # Calculate composite score with weights
        composite_score = (
            interest_score * 0.35 +      # Interest match is most important
            popularity_score * 0.20 +     # Popularity matters
            distance_score * 0.25 +       # Distance is significant
            time_score * 0.20             # Time suitability
        )
        
        # Apply budget constraints
        if budget == 'low':
            # Prefer free attractions
            if poi.get('category') in ['park', 'viewpoint', 'historic_site']:
                composite_score *= 1.2
        elif budget == 'high':
            # Boost premium experiences
            if poi.get('category') in ['palace', 'museum', 'restaurant']:
                composite_score *= 1.1
        
        poi['composite_score'] = min(composite_score, 1.0)
        scored_pois.append(poi)
    
    # Sort by composite score
    scored_pois.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return scored_pois


def _calculate_time_suitability(poi: Dict) -> float:
    """
    Calculate how suitable a POI is for current time of day
    
    Args:
        poi: POI dictionary with category information
        
    Returns:
        Time suitability score from 0.0 to 1.0
    """
    current_hour = datetime.now().hour
    category = poi.get('category', '')
    
    # Define optimal time ranges for different POI types
    time_preferences = {
        'museum': (9, 17),
        'palace': (9, 17),
        'park': (8, 20),
        'restaurant': (12, 14, 18, 22),  # Lunch and dinner
        'viewpoint': (8, 10, 17, 20),    # Morning and evening
        'market': (9, 18),
        'mosque': (8, 20)
    }
    
    if category not in time_preferences:
        return 0.7  # Default neutral score
    
    time_range = time_preferences[category]
    
    # Check if current time is within optimal range
    if len(time_range) == 2:
        start, end = time_range
        if start <= current_hour <= end:
            return 1.0
        else:
            # Calculate distance from optimal time
            distance_from_optimal = min(
                abs(current_hour - start),
                abs(current_hour - end)
            )
            return max(0.3, 1.0 - (distance_from_optimal / 12.0))
    else:
        # Multiple time ranges (e.g., lunch and dinner)
        for i in range(0, len(time_range), 2):
            start, end = time_range[i], time_range[i+1]
            if start <= current_hour <= end:
                return 1.0
        return 0.5
    
    return 0.7


def _select_optimal_waypoints(
    self,
    scored_pois: List[Dict],
    constraints: Dict = None
) -> List[PersonalizedWaypoint]:
    """
    Select optimal waypoints from scored POIs respecting constraints
    
    Args:
        scored_pois: POIs sorted by score
        constraints: Route constraints (time, distance, count)
        
    Returns:
        List of PersonalizedWaypoint objects
    """
    if constraints is None:
        constraints = {}
    
    # Extract constraints
    max_waypoints = constraints.get('max_waypoints', 5)
    max_time_minutes = constraints.get('max_time_minutes', 300)  # 5 hours default
    max_distance_km = constraints.get('max_distance_km', 10.0)
    
    selected_waypoints = []
    total_time = 0
    total_distance = 0
    
    # Track visited districts to ensure variety
    visited_districts = set()
    max_per_district = 2
    
    for poi in scored_pois:
        # Check if we've reached limits
        if len(selected_waypoints) >= max_waypoints:
            break
        
        visit_duration = poi.get('visit_duration_min', 60)
        distance_km = poi.get('distance_km', 0)
        
        # Check time constraint
        if total_time + visit_duration > max_time_minutes:
            continue
        
        # Check distance constraint
        if total_distance + distance_km > max_distance_km:
            continue
        
        # Check district variety (don't cluster too much in one district)
        district = poi.get('district', '')
        district_count = sum(1 for wp in selected_waypoints if wp.location.district == district)
        if district_count >= max_per_district:
            continue
        
        # Create PersonalizedWaypoint
        waypoint = PersonalizedWaypoint(
            name=poi['name'],
            location=poi['location'],
            category=poi['category'],
            interest_match=poi.get('interest_match', 0.5),
            popularity_score=poi.get('popularity', 0.5),
            weather_suitability=self._calculate_weather_suitability(poi),
            accessibility_score=self._calculate_accessibility_score(poi),
            estimated_duration=visit_duration,
            transport_modes=self._determine_transport_modes(poi, distance_km),
            personalization_reasons=self._generate_personalization_reasons(poi),
            real_time_updates={}
        )
        
        selected_waypoints.append(waypoint)
        total_time += visit_duration
        total_distance += distance_km
        visited_districts.add(district)
    
    return selected_waypoints


def _calculate_weather_suitability(poi: Dict) -> float:
    """
    Calculate how suitable a POI is for current weather
    (Placeholder - would integrate with weather API)
    
    Args:
        poi: POI dictionary
        
    Returns:
        Weather suitability score from 0.0 to 1.0
    """
    category = poi.get('category', '')
    
    # Indoor activities are weather-proof
    indoor_categories = ['museum', 'palace', 'restaurant', 'market']
    if category in indoor_categories:
        return 1.0
    
    # Outdoor activities (would check actual weather)
    # For now, return moderate score
    return 0.7


def _calculate_accessibility_score(poi: Dict) -> float:
    """
    Calculate accessibility score for POI
    (Placeholder - would use real accessibility data)
    
    Args:
        poi: POI dictionary
        
    Returns:
        Accessibility score from 0.0 to 1.0
    """
    # Major tourist attractions typically have good accessibility
    popularity = poi.get('popularity', 0.5)
    
    # Museums and palaces usually have ramps, elevators
    category = poi.get('category', '')
    if category in ['museum', 'palace']:
        return 0.9
    
    # Use popularity as proxy for accessibility
    return 0.7 + (popularity * 0.3)


def _determine_transport_modes(poi: Dict, distance_km: float) -> List[TransportMode]:
    """
    Determine suitable transport modes based on distance
    
    Args:
        poi: POI dictionary
        distance_km: Distance to POI
        
    Returns:
        List of suitable TransportMode enums
    """
    modes = []
    
    # Walking is suitable for short distances
    if distance_km <= 2.0:
        modes.append(TransportMode.WALKING)
    
    # Public transport for medium distances
    if 1.0 <= distance_km <= 10.0:
        modes.append(TransportMode.PUBLIC_TRANSPORT)
    
    # Metro for longer distances on metro lines
    if distance_km >= 3.0:
        modes.append(TransportMode.METRO)
    
    # Ferry if near water
    # (Would check actual location near Bosphorus)
    district = poi.get('district', '')
    if district in ['kadikoy', 'besiktas', 'eminonu']:
        modes.append(TransportMode.FERRY)
    
    # Default to walking if nothing else
    if not modes:
        modes.append(TransportMode.WALKING)
    
    return modes


def _generate_personalization_reasons(poi: Dict) -> List[str]:
    """
    Generate human-readable reasons why this POI was selected
    
    Args:
        poi: POI dictionary
        
    Returns:
        List of reason strings
    """
    reasons = []
    
    # Interest match
    interest_match = poi.get('interest_match', 0)
    if interest_match > 0.7:
        reasons.append(f"Strongly matches your interest in {poi.get('category')}")
    elif interest_match > 0.5:
        reasons.append(f"Matches your interest in {poi.get('category')}")
    
    # Popularity
    popularity = poi.get('popularity', 0)
    if popularity > 0.85:
        reasons.append("Highly popular attraction")
    
    # Distance
    distance = poi.get('distance_km', 0)
    if distance < 1.0:
        reasons.append("Very close to your location")
    elif distance < 2.0:
        reasons.append("Walking distance")
    
    # Time suitability
    time_score = poi.get('time_suitability', 0.5)
    if time_score > 0.8:
        reasons.append("Perfect timing for a visit")
    
    # District
    district = poi.get('district', '')
    if district:
        reasons.append(f"Located in {district.title()} district")
    
    return reasons if reasons else ["Recommended based on your preferences"]


def _get_district_from_location(location: GPSLocation) -> str:
    """
    Determine district name from GPS coordinates
    
    Args:
        location: GPS location
        
    Returns:
        District name or 'unknown'
    """
    # Use the existing method if available
    if hasattr(location, 'district') and location.district:
        return location.district
    
    # Calculate from coordinates using district centers
    min_distance = float('inf')
    closest_district = 'unknown'
    
    # District centers (from the planner's istanbul_districts)
    district_centers = {
        'sultanahmet': GPSLocation(41.0082, 28.9784),
        'beyoglu': GPSLocation(41.0369, 28.9744),
        'kadikoy': GPSLocation(40.9907, 29.0205),
        'besiktas': GPSLocation(41.0422, 29.0084)
    }
    
    for district_name, center in district_centers.items():
        distance = _calculate_distance(location, center)
        if distance < min_distance:
            min_distance = distance
            closest_district = district_name
    
    return closest_district


# Integration instructions:
"""
To integrate these methods into EnhancedGPSRoutePlanner:

1. Add these methods to the EnhancedGPSRoutePlanner class:
   - Copy each method (except _get_district_from_location as it's already there)
   - Ensure proper indentation
   
2. Update the __init__ method to initialize any new attributes:
   self.spatial_index = None  # For future optimization
   
3. Run tests:
   pytest tests/test_enhanced_gps_route_planner.py -v
   
4. Verify route creation works:
   python -c "
   from enhanced_gps_route_planner import EnhancedGPSRoutePlanner, GPSLocation
   planner = EnhancedGPSRoutePlanner()
   location = GPSLocation(41.0082, 28.9784)
   route = await planner.create_personalized_route(
       'test_user', location, {'interests': ['historical']}
   )
   print(f'Created route with {len(route.waypoints)} waypoints')
   "
"""
