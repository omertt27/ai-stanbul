#!/usr/bin/env python3
"""
Action-oriented features for Istanbul AI app
Direct integration with maps, booking, and navigation
"""

import re
from typing import Dict, List, Optional
import json

class ActionableResponseGenerator:
    """Generate responses with actionable buttons and integrations"""
    
    def __init__(self):
        self.google_maps_base = "https://maps.google.com/maps"
        self.booking_integrations = {
            "opentable": "https://www.opentable.com/istanbul",
            "tripadvisor": "https://www.tripadvisor.com/Istanbul",
            "viator": "https://www.viator.com/Istanbul"
        }
    
    def generate_map_link(self, location_name: str, district: str = "") -> str:
        """Generate Google Maps link for navigation"""
        query = f"{location_name}, {district}, Istanbul, Turkey"
        encoded_query = query.replace(" ", "+")
        return f"{self.google_maps_base}?q={encoded_query}"
    
    def generate_directions_link(self, from_location: str, to_location: str) -> str:
        """Generate Google Maps directions link"""
        from_query = f"{from_location}, Istanbul, Turkey".replace(" ", "+")
        to_query = f"{to_location}, Istanbul, Turkey".replace(" ", "+") 
        return f"{self.google_maps_base}?saddr={from_query}&daddr={to_query}"
    
    def add_action_buttons(self, response: str, location_data: List[Dict]) -> Dict:
        """Add actionable buttons to response"""
        actions = []
        
        for location in location_data:
            name = location.get('name', '')
            district = location.get('district', '')
            category = location.get('category', '').lower()
            
            # Navigation action
            map_link = self.generate_map_link(name, district)
            actions.append({
                "type": "navigation",
                "text": f"📍 Navigate to {name}",
                "url": map_link,
                "location": name
            })
            
            # Category-specific actions
            if 'restaurant' in category or 'cafe' in category:
                actions.append({
                    "type": "booking",
                    "text": f"🍽️ Reserve table at {name}",
                    "url": f"{self.booking_integrations['opentable']}?q={name.replace(' ', '+')}",
                    "location": name
                })
            
            elif 'museum' in category:
                actions.append({
                    "type": "tickets",
                    "text": f"🎫 Buy tickets for {name}",
                    "url": f"{self.booking_integrations['viator']}?q={name.replace(' ', '+')}",
                    "location": name
                })
            
            elif 'attraction' in category or 'historical' in category:
                actions.append({
                    "type": "tour",
                    "text": f"🗺️ Book tour of {name}",
                    "url": f"{self.booking_integrations['tripadvisor']}?q={name.replace(' ', '+')}+tour",
                    "location": name
                })
        
        return {
            "response": response,
            "actions": actions[:6],  # Limit to 6 actions to avoid clutter
            "has_actions": len(actions) > 0
        }
    
    def generate_transportation_actions(self, from_loc: str, to_loc: str, transport_options: List[Dict]) -> List[Dict]:
        """Generate transportation-specific actions"""
        actions = []
        
        # Google Maps directions
        directions_link = self.generate_directions_link(from_loc, to_loc)
        actions.append({
            "type": "directions",
            "text": f"🗺️ Get directions from {from_loc} to {to_loc}",
            "url": directions_link
        })
        
        # Ferry-specific actions
        ferry_options = [opt for opt in transport_options if opt.get('route_name', '').lower().find('ferry') != -1]
        if ferry_options:
            actions.append({
                "type": "schedule",
                "text": "⛴️ Check live ferry times",
                "url": "https://www.sehirhatlari.istanbul/en/timetables",
                "info": "Live ferry schedules and updates"
            })
        
        # Metro/public transport
        metro_options = [opt for opt in transport_options if opt.get('route_name', '').lower().find('metro') != -1]
        if metro_options:
            actions.append({
                "type": "metro",
                "text": "🚇 Metro route planner",
                "url": "https://www.metro.istanbul/en",
                "info": "Official Istanbul metro planner"
            })
        
        return actions
    
    def create_offline_package(self, user_profile: Dict, recommendations: List[Dict]) -> Dict:
        """Create offline data package for user"""
        offline_data = {
            "user_profile": user_profile,
            "cached_recommendations": recommendations,
            "essential_phrases": [],  # Will be populated from database
            "offline_maps": [],
            "emergency_contacts": {
                "police": "155",
                "ambulance": "112", 
                "fire": "110",
                "tourist_police": "+90 212 527 4503"
            },
            "currency_info": {
                "currency": "Turkish Lira (TL/₺)",
                "approximate_rates": {
                    "1_usd": "~30 TL",
                    "1_eur": "~33 TL",
                    "1_gbp": "~38 TL"
                },
                "last_updated": "Rates change daily - check current rates"
            },
            "basic_costs": {
                "metro_bus_tram": "5 TL with Istanbul Card",
                "taxi_start": "5 TL base fare",
                "coffee": "15-25 TL",
                "street_food": "10-30 TL",
                "restaurant_meal": "50-150 TL per person",
                "museum_entry": "20-50 TL"
            }
        }
        
        return offline_data

def enhance_response_with_actions(
    base_response: str,
    query: str, 
    location_data: Optional[List[Dict]] = None,
    transport_data: Optional[List[Dict]] = None,
    user_location: Optional[str] = None
) -> Dict:
    """Main function to enhance any response with actionable elements"""
    
    # Provide default empty list if None
    if location_data is None:
        location_data = []
    if transport_data is None:
        transport_data = []
    
    action_gen = ActionableResponseGenerator()
    
    # Add location-based actions
    enhanced_response = action_gen.add_action_buttons(base_response, location_data)
    
    # Add transportation actions if relevant
    if transport_data and user_location and location_data:
        for location in location_data[:2]:  # Limit to first 2 locations
            transport_actions = action_gen.generate_transportation_actions(
                user_location or "", 
                location.get('name', ''),
                transport_data
            )
            enhanced_response['actions'].extend(transport_actions)
    
    # Add context-aware suggestions
    query_lower = query.lower()
    if 'restaurant' in query_lower or 'food' in query_lower:
        enhanced_response['context_actions'] = [
            {
                "type": "context",
                "text": "🍽️ Check restaurant reviews",
                "url": "https://foursquare.com/explore?mode=url&near=Istanbul%2C%20Turkey&q=restaurant"
            },
            {
                "type": "context", 
                "text": "📱 Download Zomato for delivery",
                "info": "Popular food delivery app in Istanbul"
            }
        ]
    
    elif 'attraction' in query_lower or 'visit' in query_lower:
        enhanced_response['context_actions'] = [
            {
                "type": "context",
                "text": "🎫 Istanbul Museum Pass",
                "url": "https://muze.gov.tr/istanbul-museum-pass-en",
                "info": "Save money on multiple museum visits"
            },
            {
                "type": "context",
                "text": "🚌 Hop-on Hop-off Bus Tours", 
                "info": "Convenient way to see multiple attractions"
            }
        ]
    
    return enhanced_response

# Example usage functions that can be called from main.py
def get_actionable_restaurant_response(restaurants: List[Dict], user_location: Optional[str] = None) -> Dict:
    """Generate restaurant response with booking and navigation actions"""
    base_response = "Here are some great restaurant recommendations:\n\n"
    
    for i, restaurant in enumerate(restaurants[:4], 1):
        base_response += f"{i}. **{restaurant.get('name')}**\n"
        if restaurant.get('rating'):
            base_response += f"   ⭐ {restaurant['rating']}\n"
        if restaurant.get('location'):
            base_response += f"   📍 {restaurant['location']}\n"
        if restaurant.get('description'):
            base_response += f"   {restaurant['description'][:100]}...\n"
        base_response += "\n"
    
    return enhance_response_with_actions(
        base_response, 
        "restaurant recommendations", 
        [{"name": r.get('name'), "district": r.get('location', '').split(',')[0], "category": "restaurant"} 
         for r in restaurants],
        user_location=user_location
    )

def get_actionable_places_response(places: List[Dict], transport_info: Optional[List[Dict]] = None, user_location: Optional[str] = None) -> Dict:
    """Generate places response with navigation and tour booking actions"""
    base_response = "Here are some amazing places to visit:\n\n"
    
    location_data = []
    for i, place in enumerate(places[:5], 1):
        # Handle both dict and object formats
        if isinstance(place, dict):
            name = place.get('name', 'Unknown')
            category = place.get('category', 'Unknown')
            district = place.get('district', '')
        else:
            name = getattr(place, 'name', 'Unknown')
            category = getattr(place, 'category', 'Unknown')
            district = getattr(place, 'district', '')
        
        base_response += f"{i}. **{name}**\n"
        base_response += f"   Category: {category}\n"
        if district:
            base_response += f"   District: {district}\n"
        base_response += "\n"
        
        location_data.append({
            "name": name,
            "district": district,
            "category": category
        })
    
    return enhance_response_with_actions(
        base_response,
        "places to visit",
        location_data,
        transport_info,
        user_location
    )
