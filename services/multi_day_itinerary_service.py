#!/usr/bin/env python3
"""
Phase 6: Multi-Day Itinerary Planning Service
==============================================

Features:
- Plan 2-7 day Istanbul trips
- Daily route optimization with POIs
- Budget tracking across days
- Energy/fatigue management
- Accommodation-aware planning
- Day-to-day variety (avoid repetitive experiences)
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from enhanced_gps_route_planner import GPSLocation, get_enhanced_gps_planner

logger = logging.getLogger(__name__)


class TripPace(Enum):
    """Trip pace preferences"""
    RELAXED = "relaxed"      # 2-3 POIs per day, long visits
    MODERATE = "moderate"    # 4-5 POIs per day, balanced
    INTENSIVE = "intensive"  # 6+ POIs per day, quick visits


@dataclass
class DayPlan:
    """Single day itinerary"""
    day_number: int
    date: datetime
    accommodation_location: GPSLocation
    morning_route: Optional[Dict] = None
    afternoon_route: Optional[Dict] = None
    evening_activities: List[Dict] = field(default_factory=list)
    
    # Stats
    total_pois: int = 0
    total_distance_km: float = 0.0
    total_time_minutes: int = 0
    total_cost_usd: float = 0.0
    energy_level: float = 1.0  # 0.0 (exhausted) to 1.0 (full energy)
    
    # Visited categories (for variety)
    visited_categories: set = field(default_factory=set)
    visited_districts: set = field(default_factory=set)


@dataclass
class MultiDayItinerary:
    """Complete multi-day trip plan"""
    trip_id: str
    user_id: str
    start_date: datetime
    num_days: int
    accommodation_location: GPSLocation
    
    daily_plans: List[DayPlan] = field(default_factory=list)
    
    # Overall trip stats
    total_pois: int = 0
    total_distance_km: float = 0.0
    total_cost_usd: float = 0.0
    budget_remaining: float = 0.0
    
    # Preferences used
    pace: TripPace = TripPace.MODERATE
    interests: List[str] = field(default_factory=list)
    budget_usd: float = 0.0


class MultiDayItineraryService:
    """
    Multi-day trip planning with:
    - Daily route optimization
    - Budget management
    - Energy/fatigue tracking
    - Category diversity across days
    - Accommodation-centered planning
    """
    
    def __init__(self):
        self.route_planner = get_enhanced_gps_planner()
        
        # Energy costs by activity type
        self.energy_costs = {
            'walk': 0.1,      # Walking depletes energy
            'metro': 0.05,    # Light transport
            'tram': 0.05,
            'ferry': 0.03,    # Relaxing
            'visit_poi': 0.15, # POI visits are tiring
            'rest': -0.3      # Rest recovers energy
        }
        
        # Daily energy regeneration
        self.daily_energy_regen = 0.8  # Sleep recovers 80% energy
        
        # Pace configurations
        self.pace_configs = {
            TripPace.RELAXED: {
                'max_pois_per_half_day': 2,
                'max_time_per_half_day': 180,  # 3 hours
                'visit_duration_multiplier': 1.5,  # Longer visits
                'rest_breaks': 3
            },
            TripPace.MODERATE: {
                'max_pois_per_half_day': 3,
                'max_time_per_half_day': 240,  # 4 hours
                'visit_duration_multiplier': 1.0,
                'rest_breaks': 2
            },
            TripPace.INTENSIVE: {
                'max_pois_per_half_day': 4,
                'max_time_per_half_day': 300,  # 5 hours
                'visit_duration_multiplier': 0.7,  # Quick visits
                'rest_breaks': 1
            }
        }
    
    async def create_multi_day_itinerary(
        self,
        user_id: str,
        num_days: int,
        accommodation_location: GPSLocation,
        start_date: datetime,
        preferences: Dict,
        budget_usd: float = 1000.0
    ) -> MultiDayItinerary:
        """
        Create complete multi-day itinerary
        
        Args:
            user_id: User identifier
            num_days: Number of days (2-7)
            accommodation_location: Hotel/accommodation GPS location
            start_date: Trip start date
            preferences: User preferences (interests, pace, dietary, accessibility)
            budget_usd: Total trip budget in USD
        
        Returns:
            Complete multi-day itinerary
        """
        logger.info(f"Creating {num_days}-day itinerary for user {user_id}")
        
        # Validate input
        if num_days < 2 or num_days > 7:
            raise ValueError("Trip duration must be 2-7 days")
        
        # Parse preferences
        pace = TripPace(preferences.get('pace', 'moderate'))
        interests = preferences.get('interests', ['museum', 'history', 'culture'])
        
        # Initialize itinerary
        itinerary = MultiDayItinerary(
            trip_id=f"{user_id}_{start_date.strftime('%Y%m%d')}",
            user_id=user_id,
            start_date=start_date,
            num_days=num_days,
            accommodation_location=accommodation_location,
            pace=pace,
            interests=interests,
            budget_usd=budget_usd,
            budget_remaining=budget_usd
        )
        
        # Track visited categories and districts for variety
        visited_categories = set()
        visited_districts = set()
        
        # Plan each day
        for day_num in range(1, num_days + 1):
            day_date = start_date + timedelta(days=day_num - 1)
            
            day_plan = await self._plan_single_day(
                itinerary=itinerary,
                day_number=day_num,
                day_date=day_date,
                visited_categories=visited_categories,
                visited_districts=visited_districts,
                preferences=preferences
            )
            
            itinerary.daily_plans.append(day_plan)
            
            # Update global stats
            itinerary.total_pois += day_plan.total_pois
            itinerary.total_distance_km += day_plan.total_distance_km
            itinerary.total_cost_usd += day_plan.total_cost_usd
            itinerary.budget_remaining -= day_plan.total_cost_usd
            
            # Update visited tracking
            visited_categories.update(day_plan.visited_categories)
            visited_districts.update(day_plan.visited_districts)
            
            logger.info(f"Day {day_num} planned: {day_plan.total_pois} POIs, "
                       f"${day_plan.total_cost_usd:.0f}, "
                       f"Energy: {day_plan.energy_level:.1f}")
        
        logger.info(f"Itinerary complete: {itinerary.total_pois} POIs, "
                   f"${itinerary.total_cost_usd:.0f} spent, "
                   f"${itinerary.budget_remaining:.0f} remaining")
        
        return itinerary
    
    async def _plan_single_day(
        self,
        itinerary: MultiDayItinerary,
        day_number: int,
        day_date: datetime,
        visited_categories: set,
        visited_districts: set,
        preferences: Dict
    ) -> DayPlan:
        """Plan a single day's itinerary"""
        
        config = self.pace_configs[itinerary.pace]
        
        # Initialize day plan
        day_plan = DayPlan(
            day_number=day_number,
            date=day_date,
            accommodation_location=itinerary.accommodation_location
        )
        
        # Calculate available budget for this day
        remaining_days = itinerary.num_days - day_number + 1
        daily_budget = itinerary.budget_remaining / remaining_days if remaining_days > 0 else 0
        
        # Track visited POI IDs to avoid repeats
        visited_poi_ids = set()
        for prev_day in itinerary.daily_plans:
            for route_key in ['morning_route', 'afternoon_route']:
                route = getattr(prev_day, route_key, None)
                if route:
                    for poi in route.get('pois_included', []):
                        if isinstance(poi, dict):
                            visited_poi_ids.add(poi.get('id'))
        
        # Morning route (accommodation → explore → lunch spot)
        morning_interests = self._select_diverse_interests(
            itinerary.interests,
            visited_categories,
            prefer_new=True
        )
        
        morning_route = await self._plan_half_day_route(
            start_location=itinerary.accommodation_location,
            time_budget_minutes=config['max_time_per_half_day'],
            cost_budget_usd=daily_budget * 0.4,  # 40% of daily budget for morning
            interests=morning_interests,
            max_pois=config['max_pois_per_half_day'],
            user_id=itinerary.user_id,
            preferences=preferences,
            avoid_categories=visited_categories if day_number > 1 else set(),
            visited_poi_ids=visited_poi_ids
        )
        
        day_plan.morning_route = morning_route
        
        # Update day stats from morning
        self._update_day_stats_from_route(day_plan, morning_route)
        
        # Update visited POIs
        for poi in morning_route.get('pois_included', []):
            if isinstance(poi, dict):
                visited_poi_ids.add(poi.get('id'))
        
        # Afternoon route (lunch → explore → back to accommodation)
        afternoon_interests = self._select_diverse_interests(
            itinerary.interests,
            day_plan.visited_categories,  # Avoid repeating morning categories
            prefer_new=True
        )
        
        # Check if we have budget and energy for afternoon
        remaining_budget = daily_budget - day_plan.total_cost_usd
        if (day_plan.energy_level > 0.3 and remaining_budget > 10):
            
            # Start afternoon from last morning location or central point
            afternoon_start = self._get_last_location_from_route(morning_route) or \
                            itinerary.accommodation_location
            
            afternoon_route = await self._plan_half_day_route(
                start_location=afternoon_start,
                time_budget_minutes=config['max_time_per_half_day'],
                cost_budget_usd=remaining_budget * 0.8,  # Use 80% of remaining budget
                interests=afternoon_interests,
                max_pois=config['max_pois_per_half_day'],
                user_id=itinerary.user_id,
                preferences=preferences,
                avoid_categories=day_plan.visited_categories,
                return_to=itinerary.accommodation_location,  # End at hotel
                visited_poi_ids=visited_poi_ids
            )
            
            day_plan.afternoon_route = afternoon_route
            self._update_day_stats_from_route(day_plan, afternoon_route)
        
        # Evening activities (optional, low-energy)
        remaining_budget = daily_budget - day_plan.total_cost_usd
        if day_plan.energy_level > 0.4 and remaining_budget > 10:
            evening_suggestions = self._suggest_evening_activities(
                location=itinerary.accommodation_location,
                visited_districts=day_plan.visited_districts,
                budget_remaining=remaining_budget * 0.5
            )
            day_plan.evening_activities = evening_suggestions
        
        # Apply overnight recovery
        day_plan.energy_level = min(1.0, day_plan.energy_level + self.daily_energy_regen)
        
        return day_plan
    
    async def _plan_half_day_route(
        self,
        start_location: GPSLocation,
        time_budget_minutes: int,
        cost_budget_usd: float,
        interests: List[str],
        max_pois: int,
        user_id: str,
        preferences: Dict,
        avoid_categories: set = None,
        return_to: Optional[GPSLocation] = None,
        visited_poi_ids: set = None
    ) -> Dict:
        """Plan a half-day route (morning or afternoon)"""
        
        # Create constraints
        constraints = {
            'max_pois': max_pois,
            'max_detour_minutes': time_budget_minutes,
            'max_total_detour': time_budget_minutes
        }
        
        # Add cost constraint if budget provided
        if cost_budget_usd and cost_budget_usd > 0:
            constraints['max_cost_usd'] = cost_budget_usd
        
        # Enhanced preferences
        route_preferences = {
            **preferences,
            'interests': interests,
            'avoid_categories': list(avoid_categories) if avoid_categories else [],
            'exclude_poi_ids': list(visited_poi_ids) if visited_poi_ids else []
        }
        
        # Generate end location if not specified
        end_location = return_to or self._generate_exploration_endpoint(
            start_location,
            max_distance_km=3.0
        )
        
        try:
            route = await self.route_planner.create_poi_optimized_route(
                user_id=user_id,
                start_location=start_location,
                end_location=end_location,
                preferences=route_preferences,
                constraints=constraints
            )
            return route
        except Exception as e:
            logger.error(f"Error planning half-day route: {e}")
            # Return minimal fallback route
            return {
                'base_route': {'distance_km': 0, 'time_minutes': 0},
                'enhanced_route': {'distance_km': 0, 'time_minutes': 0},
                'pois_included': [],
                'total_cost_usd': 0
            }
    
    def _update_day_stats_from_route(self, day_plan: DayPlan, route: Dict):
        """Update day statistics from a route"""
        if not route:
            return
        
        enhanced = route.get('enhanced_route', {})
        pois = route.get('pois_included', [])
        
        # Update stats
        day_plan.total_pois += len(pois)
        day_plan.total_distance_km += enhanced.get('distance_km', 0)
        day_plan.total_time_minutes += enhanced.get('time_minutes', 0)
        
        # Calculate costs from POI dictionaries
        cost = route.get('total_cost_usd', 0)
        if cost == 0:  # Fallback calculation
            for poi in pois:
                # POIs are dictionaries in the route response
                if isinstance(poi, dict):
                    cost += poi.get('entrance_fee', 0)
                elif hasattr(poi, 'entrance_fee'):
                    cost += poi.entrance_fee
        
        day_plan.total_cost_usd += cost
        
        # Update energy (walking, visits deplete energy)
        distance_km = enhanced.get('distance_km', 0)
        energy_cost = (distance_km * 0.1) + (len(pois) * 0.15)
        day_plan.energy_level = max(0.0, day_plan.energy_level - energy_cost)
        
        # Track visited categories and districts
        for poi in pois:
            if isinstance(poi, dict):
                if 'category' in poi:
                    day_plan.visited_categories.add(poi['category'])
                if 'district' in poi:
                    day_plan.visited_districts.add(poi['district'])
            else:
                if hasattr(poi, 'category'):
                    day_plan.visited_categories.add(poi.category)
                if hasattr(poi, 'district'):
                    day_plan.visited_districts.add(poi.district)
    
    def _select_diverse_interests(
        self,
        interests: List[str],
        visited_categories: set,
        prefer_new: bool = True
    ) -> List[str]:
        """Select diverse interests, preferring unvisited categories"""
        
        if not prefer_new or not visited_categories:
            return interests[:3]  # Return top 3 interests
        
        # Prioritize interests not yet visited
        new_interests = [i for i in interests if i not in visited_categories]
        visited_interests = [i for i in interests if i in visited_categories]
        
        # Mix: 2 new + 1 visited
        selected = new_interests[:2] + visited_interests[:1]
        
        return selected if selected else interests[:3]
    
    def _get_last_location_from_route(self, route: Dict) -> Optional[GPSLocation]:
        """Extract last location from a route"""
        if not route:
            return None
        
        pois = route.get('pois_included', [])
        if pois:
            last_poi = pois[-1]
            if hasattr(last_poi, 'latitude') and hasattr(last_poi, 'longitude'):
                return GPSLocation(
                    last_poi.latitude,
                    last_poi.longitude,
                    district=getattr(last_poi, 'district', None)
                )
        
        return None
    
    def _generate_exploration_endpoint(
        self,
        start: GPSLocation,
        max_distance_km: float = 3.0
    ) -> GPSLocation:
        """Generate an exploration endpoint for circular routes"""
        
        # Simple offset (in reality, would choose interesting district)
        lat_offset = 0.02  # ~2km north
        lon_offset = 0.01  # ~1km east
        
        return GPSLocation(
            start.latitude + lat_offset,
            start.longitude + lon_offset,
            district='exploration_zone'
        )
    
    def _suggest_evening_activities(
        self,
        location: GPSLocation,
        visited_districts: set,
        budget_remaining: float
    ) -> List[Dict]:
        """Suggest low-energy evening activities"""
        
        suggestions = []
        
        # Dinner recommendations
        suggestions.append({
            'type': 'dinner',
            'name': 'Local Turkish Restaurant',
            'description': 'Traditional Turkish cuisine near accommodation',
            'estimated_cost_usd': 20,
            'duration_minutes': 90,
            'energy_cost': 0.0  # Eating recovers energy
        })
        
        # Evening walk/viewpoint
        if budget_remaining > 10:
            suggestions.append({
                'type': 'evening_activity',
                'name': 'Sunset Viewpoint or Bosphorus Walk',
                'description': 'Relaxing evening stroll with city views',
                'estimated_cost_usd': 0,
                'duration_minutes': 60,
                'energy_cost': 0.05
            })
        
        return suggestions
    
    def format_itinerary_summary(self, itinerary: MultiDayItinerary) -> str:
        """Format itinerary as readable text summary"""
        
        lines = [
            f"\n{'='*80}",
            f"🗓️  {itinerary.num_days}-DAY ISTANBUL ITINERARY",
            f"{'='*80}",
            f"\n📍 Accommodation: {itinerary.accommodation_location.district or 'Central Istanbul'}",
            f"📅 Dates: {itinerary.start_date.strftime('%B %d, %Y')} - "
            f"{(itinerary.start_date + timedelta(days=itinerary.num_days-1)).strftime('%B %d, %Y')}",
            f"⚡ Pace: {itinerary.pace.value.upper()}",
            f"💰 Budget: ${itinerary.budget_usd:.0f} (${itinerary.budget_remaining:.0f} remaining)",
            f"📊 Total: {itinerary.total_pois} POIs, {itinerary.total_distance_km:.1f}km",
            ""
        ]
        
        for day in itinerary.daily_plans:
            lines.extend([
                f"\n{'─'*80}",
                f"DAY {day.day_number}: {day.date.strftime('%A, %B %d')}",
                f"{'─'*80}",
                f"POIs: {day.total_pois} | Distance: {day.total_distance_km:.1f}km | "
                f"Cost: ${day.total_cost_usd:.0f} | Energy: {day.energy_level:.0%}",
                ""
            ])
            
            if day.morning_route:
                lines.append("🌅 MORNING:")
                pois = day.morning_route.get('pois_included', [])
                for poi in pois:
                    # Handle both dict and object POIs
                    if isinstance(poi, dict):
                        name = poi.get('name', 'Unknown POI')
                        category = poi.get('category', '')
                        cost = poi.get('entrance_fee', 0)
                        cost_str = f" (${cost})" if cost > 0 else " (Free)"
                        lines.append(f"  • {name} [{category}]{cost_str}")
                    else:
                        name = getattr(poi, 'name', 'Unknown POI')
                        lines.append(f"  • {name}")
            
            if day.afternoon_route:
                lines.append("\n🌆 AFTERNOON:")
                pois = day.afternoon_route.get('pois_included', [])
                for poi in pois:
                    if isinstance(poi, dict):
                        name = poi.get('name', 'Unknown POI')
                        category = poi.get('category', '')
                        cost = poi.get('entrance_fee', 0)
                        cost_str = f" (${cost})" if cost > 0 else " (Free)"
                        lines.append(f"  • {name} [{category}]{cost_str}")
                    else:
                        name = getattr(poi, 'name', 'Unknown POI')
                        lines.append(f"  • {name}")
            
            if day.evening_activities:
                lines.append("\n🌙 EVENING:")
                for activity in day.evening_activities:
                    lines.append(f"  • {activity['name']}")
        
        lines.extend([
            f"\n{'='*80}",
            "✨ Have a wonderful trip to Istanbul! ✨",
            f"{'='*80}\n"
        ])
        
        return "\n".join(lines)


# Singleton instance
_multi_day_service = None

def get_multi_day_service() -> MultiDayItineraryService:
    """Get singleton multi-day itinerary service"""
    global _multi_day_service
    if _multi_day_service is None:
        _multi_day_service = MultiDayItineraryService()
    return _multi_day_service
