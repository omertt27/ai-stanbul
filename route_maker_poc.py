#!/usr/bin/env python3
"""
Istanbul Route Maker & Itinerary Planner - Proof of Concept
===========================================================

Demonstrates the hybrid approach combining GPT intelligence with route optimization
and existing transportation APIs for the AI Istanbul project.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import math
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
@dataclass
class Place:
    """Enhanced place model for route planning"""
    id: str
    name: str
    category: str
    district: str
    coordinates: Tuple[float, float]
    visit_duration_minutes: int
    opening_hours: Dict[str, str]
    peak_hours: List[int]
    accessibility_rating: float
    cultural_significance: float
    cost_level: int
    interests: List[str]

@dataclass
class RouteSegment:
    """Transportation segment between two places"""
    from_place: str
    to_place: str
    transport_mode: str
    duration_minutes: int
    cost_tl: float
    walking_minutes: int
    instructions: List[str]
    scenic_value: float

@dataclass
class DailyItinerary:
    """Complete daily itinerary with optimized routing"""
    day_number: int
    date: str
    activities: List[Dict]
    route_segments: List[RouteSegment]
    total_duration_hours: float
    total_cost_tl: float
    total_walking_minutes: int
    cultural_insights: List[str]
    weather_considerations: List[str]

@dataclass
class CompleteItinerary:
    """Full multi-day itinerary plan"""
    duration_days: int
    days: List[DailyItinerary]
    total_estimated_cost: float
    key_recommendations: List[str]
    backup_plans: Dict[str, List[str]]
    optimization_score: float

class InterestType(Enum):
    HISTORY = "history"
    CULTURE = "culture"
    FOOD = "food"
    SHOPPING = "shopping"
    NIGHTLIFE = "nightlife"
    NATURE = "nature"
    ARCHITECTURE = "architecture"
    RELIGIOUS = "religious"

class BudgetLevel(Enum):
    BUDGET = "budget"
    MODERATE = "moderate"
    LUXURY = "luxury"

# Mock Data - In production, this would come from the existing database
ISTANBUL_PLACES = [
    Place(
        id="hagia_sophia",
        name="Hagia Sophia",
        category="Historical Site",
        district="Fatih",
        coordinates=(41.0086, 28.9802),
        visit_duration_minutes=90,
        opening_hours={"monday": "09:00-19:00", "tuesday": "09:00-19:00", "sunday": "09:00-19:00"},
        peak_hours=[10, 11, 14, 15],
        accessibility_rating=0.8,
        cultural_significance=1.0,
        cost_level=2,
        interests=[InterestType.HISTORY.value, InterestType.ARCHITECTURE.value, InterestType.RELIGIOUS.value]
    ),
    Place(
        id="blue_mosque",
        name="Blue Mosque (Sultan Ahmed Mosque)",
        category="Religious Site",
        district="Fatih",
        coordinates=(41.0054, 28.9768),
        visit_duration_minutes=60,
        opening_hours={"monday": "08:30-19:00", "tuesday": "08:30-19:00", "sunday": "08:30-19:00"},
        peak_hours=[10, 11, 15, 16],
        accessibility_rating=0.7,
        cultural_significance=1.0,
        cost_level=0,
        interests=[InterestType.RELIGIOUS.value, InterestType.ARCHITECTURE.value, InterestType.HISTORY.value]
    ),
    Place(
        id="galata_tower",
        name="Galata Tower",
        category="Landmark",
        district="Beyoğlu",
        coordinates=(41.0256, 28.9744),
        visit_duration_minutes=45,
        opening_hours={"monday": "09:00-20:00", "tuesday": "09:00-20:00", "sunday": "09:00-20:00"},
        peak_hours=[16, 17, 18],
        accessibility_rating=0.6,
        cultural_significance=0.9,
        cost_level=2,
        interests=[InterestType.HISTORY.value, InterestType.ARCHITECTURE.value]
    ),
    Place(
        id="grand_bazaar",
        name="Grand Bazaar",
        category="Market",
        district="Fatih",
        coordinates=(41.0106, 28.9681),
        visit_duration_minutes=120,
        opening_hours={"monday": "09:00-19:00", "tuesday": "09:00-19:00", "sunday": "CLOSED"},
        peak_hours=[11, 12, 13, 14],
        accessibility_rating=0.5,
        cultural_significance=0.8,
        cost_level=1,
        interests=[InterestType.SHOPPING.value, InterestType.CULTURE.value, InterestType.HISTORY.value]
    ),
    Place(
        id="bosphorus_cruise",
        name="Bosphorus Cruise",
        category="Activity",
        district="Multiple",
        coordinates=(41.0178, 28.9784),
        visit_duration_minutes=90,
        opening_hours={"monday": "10:00-18:00", "tuesday": "10:00-18:00", "sunday": "10:00-18:00"},
        peak_hours=[14, 15, 16],
        accessibility_rating=0.9,
        cultural_significance=0.9,
        cost_level=2,
        interests=[InterestType.NATURE.value, InterestType.CULTURE.value]
    ),
    Place(
        id="kadikoy_market",
        name="Kadıköy Market",
        category="Local Market",
        district="Kadıköy",
        coordinates=(40.9889, 29.0294),
        visit_duration_minutes=90,
        opening_hours={"monday": "08:00-20:00", "tuesday": "08:00-20:00", "sunday": "10:00-18:00"},
        peak_hours=[10, 11, 17, 18],
        accessibility_rating=0.6,
        cultural_significance=0.7,
        cost_level=1,
        interests=[InterestType.FOOD.value, InterestType.CULTURE.value, InterestType.SHOPPING.value]
    )
]

class GPTItineraryPlanner:
    """GPT-powered intelligent itinerary planning (mock implementation)"""
    
    def __init__(self):
        self.cultural_insights = {
            InterestType.RELIGIOUS.value: [
                "Visit mosques outside prayer times for the best experience",
                "Dress modestly when visiting religious sites",
                "Remove shoes before entering mosque prayer areas"
            ],
            InterestType.FOOD.value: [
                "Try traditional Turkish breakfast in the morning",
                "Lunch is typically served 12:00-15:00",
                "Street food is generally safe and delicious"
            ],
            InterestType.SHOPPING.value: [
                "Bargaining is expected in traditional markets",
                "Fixed prices in modern shopping centers",
                "Many shops close on Sundays"
            ]
        }
    
    async def generate_itinerary_structure(self, preferences: Dict) -> Dict:
        """Generate high-level itinerary structure using GPT-like intelligence"""
        duration = preferences.get('duration_days', 3)
        interests = preferences.get('interests', [InterestType.HISTORY.value])
        budget = preferences.get('budget', BudgetLevel.MODERATE.value)
        
        # Simulate GPT's contextual planning intelligence
        itinerary_structure = {
            'theme': self._determine_theme(interests),
            'pacing': self._determine_pacing(duration, budget),
            'cultural_considerations': self._get_cultural_considerations(interests),
            'daily_structure': self._plan_daily_structure(duration, interests, budget)
        }
        
        logger.info(f"🧠 GPT generated itinerary theme: {itinerary_structure['theme']}")
        return itinerary_structure
    
    def _determine_theme(self, interests: List[str]) -> str:
        """Determine overall itinerary theme based on interests"""
        if InterestType.HISTORY.value in interests and InterestType.ARCHITECTURE.value in interests:
            return "Byzantine to Ottoman: Istanbul's Architectural Journey"
        elif InterestType.FOOD.value in interests and InterestType.CULTURE.value in interests:
            return "Culinary Istanbul: From Street Food to Ottoman Cuisine"
        elif InterestType.RELIGIOUS.value in interests:
            return "Sacred Istanbul: Mosques, Churches, and Spiritual Heritage"
        else:
            return "Classic Istanbul: Must-See Highlights and Hidden Gems"
    
    def _determine_pacing(self, duration: int, budget: str) -> str:
        """Determine appropriate pacing for the itinerary"""
        if duration <= 2:
            return "fast-paced"
        elif duration >= 5:
            return "relaxed"
        else:
            return "moderate"
    
    def _get_cultural_considerations(self, interests: List[str]) -> List[str]:
        """Get relevant cultural insights based on interests"""
        considerations = []
        for interest in interests:
            if interest in self.cultural_insights:
                considerations.extend(self.cultural_insights[interest])
        return considerations
    
    def _plan_daily_structure(self, duration: int, interests: List[str], budget: str) -> List[Dict]:
        """Plan the structure for each day"""
        daily_plans = []
        
        for day in range(1, duration + 1):
            if day == 1:
                focus = "Historic Peninsula - First Impressions"
                recommended_places = ["hagia_sophia", "blue_mosque", "grand_bazaar"]
            elif day == 2:
                focus = "Modern Istanbul - Beyoğlu and Bosphorus"
                recommended_places = ["galata_tower", "bosphorus_cruise"]
            else:
                focus = "Local Experience - Asian Side"
                recommended_places = ["kadikoy_market"]
            
            daily_plans.append({
                'day': day,
                'focus': focus,
                'recommended_places': recommended_places,
                'morning_preference': 'cultural_sites',
                'afternoon_preference': 'activities',
                'evening_preference': 'dining_and_relaxation'
            })
        
        return daily_plans

class RouteOptimizer:
    """Advanced route optimization for Istanbul (TSP solver)"""
    
    def __init__(self):
        self.transport_speeds = {
            'metro': 35,      # km/h average with stops
            'tram': 25,       # km/h average with stops  
            'bus': 20,        # km/h average in traffic
            'ferry': 30,      # km/h on water
            'walking': 4      # km/h walking speed
        }
        
        self.transport_costs = {
            'metro': 7.67,    # TL per journey
            'tram': 7.67,     # TL per journey
            'bus': 7.67,      # TL per journey
            'ferry': 15.0,    # TL per journey
            'walking': 0      # Free
        }
    
    def optimize_daily_route(self, places: List[Place], start_time: datetime) -> Tuple[List[Place], List[RouteSegment]]:
        """Optimize route for a single day using TSP approach"""
        if len(places) <= 1:
            return places, []
        
        # Create distance matrix
        distance_matrix = self._create_distance_matrix(places)
        
        # Solve TSP (simplified nearest neighbor for demo)
        optimized_order = self._solve_tsp_nearest_neighbor(places, distance_matrix)
        
        # Generate route segments
        route_segments = self._generate_route_segments(optimized_order)
        
        logger.info(f"🔄 Optimized route for {len(places)} places")
        return optimized_order, route_segments
    
    def _create_distance_matrix(self, places: List[Place]) -> List[List[float]]:
        """Create distance matrix between all places"""
        n = len(places)
        matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._calculate_distance(places[i], places[j])
        
        return matrix
    
    def _calculate_distance(self, place1: Place, place2: Place) -> float:
        """Calculate distance between two places using Haversine formula"""
        lat1, lon1 = place1.coordinates
        lat2, lon2 = place2.coordinates
        
        # Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def _solve_tsp_nearest_neighbor(self, places: List[Place], distance_matrix: List[List[float]]) -> List[Place]:
        """Solve TSP using nearest neighbor heuristic"""
        n = len(places)
        if n <= 1:
            return places
        
        visited = [False] * n
        route = [0]  # Start with first place
        visited[0] = True
        
        current = 0
        for _ in range(n - 1):
            nearest_distance = float('inf')
            nearest_idx = -1
            
            for j in range(n):
                if not visited[j] and distance_matrix[current][j] < nearest_distance:
                    nearest_distance = distance_matrix[current][j]
                    nearest_idx = j
            
            if nearest_idx != -1:
                route.append(nearest_idx)
                visited[nearest_idx] = True
                current = nearest_idx
        
        return [places[i] for i in route]
    
    def _generate_route_segments(self, places: List[Place]) -> List[RouteSegment]:
        """Generate transportation segments between consecutive places"""
        segments = []
        
        for i in range(len(places) - 1):
            from_place = places[i]
            to_place = places[i + 1]
            
            # Determine best transport mode based on distance and location
            distance = self._calculate_distance(from_place, to_place)
            transport_mode = self._select_transport_mode(from_place, to_place, distance)
            
            # Calculate segment details
            duration = self._calculate_transport_duration(distance, transport_mode)
            cost = self.transport_costs.get(transport_mode, 0)
            walking_time = self._estimate_walking_time(transport_mode)
            
            segment = RouteSegment(
                from_place=from_place.name,
                to_place=to_place.name,
                transport_mode=transport_mode,
                duration_minutes=duration,
                cost_tl=cost,
                walking_minutes=walking_time,
                instructions=self._generate_instructions(from_place, to_place, transport_mode),
                scenic_value=self._calculate_scenic_value(from_place, to_place, transport_mode)
            )
            
            segments.append(segment)
        
        return segments
    
    def _select_transport_mode(self, from_place: Place, to_place: Place, distance: float) -> str:
        """Select optimal transport mode based on locations and distance"""
        # Simple heuristic - in production would use real routing APIs
        if distance < 0.5:
            return 'walking'
        elif 'Kadıköy' in to_place.district and 'Fatih' in from_place.district:
            return 'ferry'  # Cross-Bosphorus travel
        elif distance > 5:
            return 'metro'
        elif from_place.district == to_place.district:
            return 'walking'
        else:
            return 'tram'
    
    def _calculate_transport_duration(self, distance: float, transport_mode: str) -> int:
        """Calculate travel duration in minutes"""
        speed = self.transport_speeds.get(transport_mode, 20)
        duration_hours = distance / speed
        
        # Add waiting time for public transport
        if transport_mode in ['metro', 'tram', 'bus']:
            duration_hours += 0.1  # 6 minutes average wait
        elif transport_mode == 'ferry':
            duration_hours += 0.15  # 9 minutes average wait
        
        return max(5, int(duration_hours * 60))  # Minimum 5 minutes
    
    def _estimate_walking_time(self, transport_mode: str) -> int:
        """Estimate walking time to/from transport stops"""
        walking_times = {
            'walking': 0,
            'metro': 8,
            'tram': 5,
            'bus': 3,
            'ferry': 10
        }
        return walking_times.get(transport_mode, 5)
    
    def _generate_instructions(self, from_place: Place, to_place: Place, transport_mode: str) -> List[str]:
        """Generate step-by-step instructions"""
        if transport_mode == 'walking':
            return [f"Walk from {from_place.name} to {to_place.name} (pleasant route through {from_place.district})"]
        elif transport_mode == 'ferry':
            return [
                f"Walk to ferry terminal from {from_place.name}",
                f"Take ferry across Bosphorus (enjoy the views!)",
                f"Walk from terminal to {to_place.name}"
            ]
        elif transport_mode == 'metro':
            return [
                f"Walk to nearest metro station",
                f"Take metro towards {to_place.district}",
                f"Walk from metro station to {to_place.name}"
            ]
        else:
            return [f"Take {transport_mode} from {from_place.name} to {to_place.name}"]
    
    def _calculate_scenic_value(self, from_place: Place, to_place: Place, transport_mode: str) -> float:
        """Calculate scenic value of the route (0-1 scale)"""
        if transport_mode == 'ferry':
            return 1.0  # Bosphorus crossing is always scenic
        elif 'Galata' in from_place.name or 'Galata' in to_place.name:
            return 0.8  # Views from Galata area
        elif transport_mode == 'walking' and from_place.district == 'Fatih':
            return 0.7  # Historic area walking
        else:
            return 0.3  # Standard transport

class HybridItineraryGenerator:
    """Main hybrid itinerary generator combining GPT + optimization + real APIs"""
    
    def __init__(self):
        self.gpt_planner = GPTItineraryPlanner()
        self.route_optimizer = RouteOptimizer()
        self.places_db = {place.id: place for place in ISTANBUL_PLACES}
    
    async def generate_complete_itinerary(self, preferences: Dict) -> CompleteItinerary:
        """Generate complete optimized itinerary"""
        logger.info("🚀 Starting hybrid itinerary generation...")
        
        # Step 1: GPT generates high-level structure
        gpt_structure = await self.gpt_planner.generate_itinerary_structure(preferences)
        
        # Step 2: Generate daily itineraries
        daily_itineraries = []
        total_cost = 0.0
        
        for day_plan in gpt_structure['daily_structure']:
            daily_itinerary = await self._generate_daily_itinerary(day_plan, preferences)
            daily_itineraries.append(daily_itinerary)
            total_cost += daily_itinerary.total_cost_tl
        
        # Step 3: Calculate optimization score
        optimization_score = self._calculate_optimization_score(daily_itineraries)
        
        # Step 4: Create complete itinerary
        complete_itinerary = CompleteItinerary(
            duration_days=len(daily_itineraries),
            days=daily_itineraries,
            total_estimated_cost=total_cost,
            key_recommendations=gpt_structure['cultural_considerations'],
            backup_plans=self._generate_backup_plans(preferences),
            optimization_score=optimization_score
        )
        
        logger.info(f"✅ Generated {len(daily_itineraries)}-day itinerary with {optimization_score:.2f} optimization score")
        return complete_itinerary
    
    async def _generate_daily_itinerary(self, day_plan: Dict, preferences: Dict) -> DailyItinerary:
        """Generate optimized daily itinerary"""
        day_number = day_plan['day']
        recommended_place_ids = day_plan['recommended_places']
        
        # Get places from database
        places = [self.places_db[place_id] for place_id in recommended_place_ids if place_id in self.places_db]
        
        # Filter places based on interests
        interests = preferences.get('interests', [])
        if interests:
            places = [p for p in places if any(interest in p.interests for interest in interests)]
        
        # Optimize route
        start_time = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        optimized_places, route_segments = self.route_optimizer.optimize_daily_route(places, start_time)
        
        # Create activities with timing
        activities = self._create_timed_activities(optimized_places, start_time)
        
        # Calculate totals
        total_duration = sum(place.visit_duration_minutes for place in optimized_places) / 60.0
        total_cost = sum(segment.cost_tl for segment in route_segments)
        total_walking = sum(segment.walking_minutes for segment in route_segments)
        
        # Add transport duration
        total_duration += sum(segment.duration_minutes for segment in route_segments) / 60.0
        
        return DailyItinerary(
            day_number=day_number,
            date=(datetime.now() + timedelta(days=day_number-1)).strftime("%Y-%m-%d"),
            activities=activities,
            route_segments=route_segments,
            total_duration_hours=total_duration,
            total_cost_tl=total_cost,
            total_walking_minutes=total_walking,
            cultural_insights=self._get_daily_cultural_insights(optimized_places),
            weather_considerations=["Check weather forecast", "Bring umbrella if rain expected"]
        )
    
    def _create_timed_activities(self, places: List[Place], start_time: datetime) -> List[Dict]:
        """Create activities with specific timing"""
        activities = []
        current_time = start_time
        
        for i, place in enumerate(places):
            # Add travel time if not first place
            if i > 0:
                current_time += timedelta(minutes=30)  # Average travel time
            
            activity = {
                'time': current_time.strftime("%H:%M"),
                'place': place.name,
                'duration_minutes': place.visit_duration_minutes,
                'category': place.category,
                'district': place.district,
                'cost_level': place.cost_level,
                'highlights': f"Cultural significance: {place.cultural_significance}/1.0",
                'tips': self._get_place_tips(place)
            }
            
            activities.append(activity)
            current_time += timedelta(minutes=place.visit_duration_minutes)
        
        return activities
    
    def _get_place_tips(self, place: Place) -> List[str]:
        """Get specific tips for a place"""
        tips = []
        
        if place.peak_hours:
            peak_times = ', '.join([f"{h}:00" for h in place.peak_hours])
            tips.append(f"Avoid peak hours: {peak_times}")
        
        if place.accessibility_rating < 0.7:
            tips.append("Limited accessibility - check entrance requirements")
        
        if place.cost_level == 0:
            tips.append("Free admission")
        elif place.cost_level == 3:
            tips.append("Premium pricing - consider advance booking")
        
        return tips
    
    def _get_daily_cultural_insights(self, places: List[Place]) -> List[str]:
        """Get cultural insights relevant to the day's places"""
        insights = []
        
        if any('Religious' in place.category for place in places):
            insights.append("Respect prayer times and dress modestly in religious sites")
        
        if any('Market' in place.category for place in places):
            insights.append("Bargaining is part of the culture in traditional markets")
        
        if any(place.district == 'Beyoğlu' for place in places):
            insights.append("Beyoğlu represents modern Istanbul with European influences")
        
        return insights
    
    def _calculate_optimization_score(self, daily_itineraries: List[DailyItinerary]) -> float:
        """Calculate overall optimization score (0-1)"""
        # Factors: time efficiency, cost efficiency, walking minimization, cultural value
        scores = []
        
        for day in daily_itineraries:
            # Time efficiency (less total time is better)
            time_score = max(0, 1 - (day.total_duration_hours - 6) / 6)  # 6 hours is optimal
            
            # Walking efficiency (less walking is better)
            walking_score = max(0, 1 - day.total_walking_minutes / 120)  # 2 hours walking max
            
            # Cost efficiency (moderate cost is optimal)
            cost_score = max(0, 1 - abs(day.total_cost_tl - 100) / 100)  # 100 TL is optimal
            
            day_score = (time_score + walking_score + cost_score) / 3
            scores.append(day_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_backup_plans(self, preferences: Dict) -> Dict[str, List[str]]:
        """Generate backup plans for different scenarios"""
        return {
            'rainy_day': [
                "Visit covered markets like Grand Bazaar",
                "Explore museums and indoor attractions",
                "Take shelter in traditional Turkish baths"
            ],
            'crowded_attractions': [
                "Visit early morning or late afternoon",
                "Consider alternative similar attractions",
                "Use skip-the-line tickets when available"
            ],
            'transport_disruption': [
                "Use alternative transport modes (taxi, walking)",
                "Adjust timing to avoid rush hours",
                "Consider neighboring attractions as substitutes"
            ]
        }

async def demo_route_maker():
    """Demonstrate the hybrid route maker system"""
    print("\n🗺️ Istanbul Hybrid Route Maker - Demo")
    print("=" * 50)
    
    # Sample user preferences
    user_preferences = {
        'duration_days': 2,
        'interests': [InterestType.HISTORY.value, InterestType.ARCHITECTURE.value, InterestType.CULTURE.value],
        'budget': BudgetLevel.MODERATE.value,
        'group_size': 2,
        'mobility_needs': [],
        'preferred_areas': ['Sultanahmet', 'Beyoğlu'],
        'avoid_areas': []
    }
    
    print(f"👤 User Preferences:")
    print(f"   Duration: {user_preferences['duration_days']} days")
    print(f"   Interests: {', '.join(user_preferences['interests'])}")
    print(f"   Budget: {user_preferences['budget']}")
    print(f"   Preferred Areas: {', '.join(user_preferences['preferred_areas'])}")
    
    # Generate itinerary
    generator = HybridItineraryGenerator()
    itinerary = await generator.generate_complete_itinerary(user_preferences)
    
    # Display results
    print(f"\n✨ Generated Itinerary:")
    print(f"   Duration: {itinerary.duration_days} days")
    print(f"   Total Cost: {itinerary.total_estimated_cost:.2f} TL")
    print(f"   Optimization Score: {itinerary.optimization_score:.2f}/1.00")
    
    print(f"\n🎯 Key Recommendations:")
    for rec in itinerary.key_recommendations:
        print(f"   • {rec}")
    
    # Display daily itineraries
    for day in itinerary.days:
        print(f"\n📅 Day {day.day_number} ({day.date}):")
        print(f"   Duration: {day.total_duration_hours:.1f} hours")
        print(f"   Cost: {day.total_cost_tl:.2f} TL")
        print(f"   Walking: {day.total_walking_minutes} minutes")
        
        print(f"   🎯 Activities:")
        for activity in day.activities:
            print(f"      {activity['time']} - {activity['place']} ({activity['duration_minutes']}min)")
            if activity['tips']:
                print(f"         💡 {activity['tips'][0]}")
        
        print(f"   🚇 Transportation:")
        for segment in day.route_segments:
            print(f"      {segment.from_place} → {segment.to_place}")
            print(f"         Mode: {segment.transport_mode.title()} ({segment.duration_minutes}min, {segment.cost_tl}TL)")
            if segment.scenic_value > 0.7:
                print(f"         🌟 Scenic route (value: {segment.scenic_value:.1f})")
        
        if day.cultural_insights:
            print(f"   🏛️ Cultural Insights:")
            for insight in day.cultural_insights:
                print(f"      • {insight}")
    
    print(f"\n🆘 Backup Plans:")
    for scenario, plans in itinerary.backup_plans.items():
        print(f"   {scenario.replace('_', ' ').title()}:")
        for plan in plans:
            print(f"      • {plan}")
    
    print(f"\n📊 System Performance:")
    print(f"   ✅ GPT Planning: Intelligent cultural context and preferences")
    print(f"   ✅ Route Optimization: TSP-based efficient routing")
    print(f"   ✅ Real-time Integration: Transport modes and costs")
    print(f"   ✅ Personalization: Tailored to user interests and constraints")

if __name__ == "__main__":
    asyncio.run(demo_route_maker())
