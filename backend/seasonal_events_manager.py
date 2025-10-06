#!/usr/bin/env python3
"""
Seasonal Events & Updates Manager
================================

Automates daily updates for:
1. Seasonal events in Istanbul
2. Transport schedule changes
3. Restaurant/museum status updates
4. Weather-based recommendations
5. Local festivities and cultural events
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

class EventType(Enum):
    FESTIVAL = "festival"
    CULTURAL = "cultural"
    RELIGIOUS = "religious"
    SEASONAL = "seasonal"
    TRANSPORT = "transport"
    WEATHER = "weather"
    SPECIAL_OFFER = "special_offer"

class Season(Enum):
    SPRING = "spring"  # March-May
    SUMMER = "summer"  # June-August
    AUTUMN = "autumn"  # September-November
    WINTER = "winter"  # December-February

@dataclass
class SeasonalEvent:
    """A seasonal event or update"""
    event_id: str
    title: str
    description: str
    event_type: EventType
    season: Season
    start_date: datetime
    end_date: datetime
    location: str
    impact_level: int  # 1-5, how much it affects recommendations
    recommendations: List[str]
    related_attractions: List[str]
    active: bool = True

class SeasonalEventsManager:
    """Manages seasonal events and daily updates"""
    
    def __init__(self):
        self.events = {}  # event_id -> SeasonalEvent
        self.daily_updates = {}  # date -> List[updates]
        self._initialize_seasonal_events()
    
    def _initialize_seasonal_events(self):
        """Initialize seasonal events calendar"""
        
        # Current date for seasonal logic
        now = datetime.now()
        current_year = now.year
        
        seasonal_events = [
            # SPRING EVENTS
            SeasonalEvent(
                event_id="tulip_season",
                title="Istanbul Tulip Season",
                description="Millions of tulips bloom across Istanbul's parks and gardens",
                event_type=EventType.SEASONAL,
                season=Season.SPRING,
                start_date=datetime(current_year, 3, 15),
                end_date=datetime(current_year, 5, 15),
                location="Emirgan Park, Gülhane Park, Various parks",
                impact_level=4,
                recommendations=[
                    "Visit Emirgan Park for the famous Tulip Festival",
                    "Best photography opportunities in early morning",
                    "Combine with Bosphorus ferry ride for complete spring experience"
                ],
                related_attractions=["emirgan_park", "gulhane_park", "bosphorus_cruise"]
            ),
            
            SeasonalEvent(
                event_id="spring_weather",
                title="Perfect Spring Weather",
                description="Mild temperatures (15-20°C) ideal for walking and outdoor activities",
                event_type=EventType.WEATHER,
                season=Season.SPRING,
                start_date=datetime(current_year, 3, 20),
                end_date=datetime(current_year, 5, 30),
                location="City-wide",
                impact_level=3,
                recommendations=[
                    "Perfect time for walking tours of historic areas",
                    "Outdoor dining becomes comfortable",
                    "Great weather for Bosphorus activities"
                ],
                related_attractions=["walking_tours", "outdoor_restaurants", "bosphorus_activities"]
            ),
            
            # SUMMER EVENTS
            SeasonalEvent(
                event_id="summer_festivals",
                title="Istanbul Summer Festivals",
                description="Various music, arts, and cultural festivals throughout the city",
                event_type=EventType.FESTIVAL,
                season=Season.SUMMER,
                start_date=datetime(current_year, 6, 1),
                end_date=datetime(current_year, 8, 31),
                location="Various venues",
                impact_level=5,
                recommendations=[
                    "Book events in advance - popular festivals sell out",
                    "Evening events are more comfortable due to heat",
                    "Combine with rooftop dining for complete summer experience"
                ],
                related_attractions=["cultural_venues", "rooftop_restaurants", "evening_activities"]
            ),
            
            SeasonalEvent(
                event_id="ramadan_season",
                title="Ramadan Period",
                description="Holy month of fasting affecting restaurant hours and cultural activities",
                event_type=EventType.RELIGIOUS,
                season=Season.SPRING,  # Varies by year
                start_date=datetime(current_year, 3, 10),  # Approximate - varies yearly
                end_date=datetime(current_year, 4, 10),
                location="City-wide",
                impact_level=4,
                recommendations=[
                    "Many restaurants close during day, open for iftar (sunset)",
                    "Special Ramadan atmosphere in historic areas after sunset",
                    "Experience traditional iftar meals",
                    "Some attractions may have modified hours"
                ],
                related_attractions=["restaurants", "mosques", "historic_areas"]
            ),
            
            # AUTUMN EVENTS
            SeasonalEvent(
                event_id="autumn_colors",
                title="Autumn Foliage Season",
                description="Beautiful fall colors in Istanbul's parks and along the Bosphorus",
                event_type=EventType.SEASONAL,
                season=Season.AUTUMN,
                start_date=datetime(current_year, 10, 1),
                end_date=datetime(current_year, 11, 30),
                location="Parks, Bosphorus shores, Asian side",
                impact_level=3,
                recommendations=[
                    "Perfect time for photography in parks",
                    "Comfortable walking weather returns",
                    "Great season for ferry rides with colorful scenery"
                ],
                related_attractions=["parks", "bosphorus_ferry", "walking_areas"]
            ),
            
            # WINTER EVENTS
            SeasonalEvent(
                event_id="winter_indoor_season",
                title="Indoor Activities Season",
                description="Focus shifts to museums, shopping, and indoor cultural activities",
                event_type=EventType.SEASONAL,
                season=Season.WINTER,
                start_date=datetime(current_year, 12, 1),
                end_date=datetime(current_year + 1, 2, 28),
                location="Museums, malls, indoor venues",
                impact_level=4,
                recommendations=[
                    "Perfect museum weather - fewer crowds indoors",
                    "Turkish baths (hammams) especially appealing in cold weather",
                    "Cozy indoor dining experiences",
                    "Shopping in covered bazaars and malls"
                ],
                related_attractions=["museums", "hammams", "indoor_restaurants", "covered_bazaars"]
            ),
            
            # TRANSPORT UPDATES
            SeasonalEvent(
                event_id="summer_transport_schedule",
                title="Extended Summer Transport Hours",
                description="Metro and ferry services run later during summer months",
                event_type=EventType.TRANSPORT,
                season=Season.SUMMER,
                start_date=datetime(current_year, 6, 15),
                end_date=datetime(current_year, 9, 15),
                location="Metro and ferry systems",
                impact_level=2,
                recommendations=[
                    "Take advantage of later metro hours for evening activities",
                    "More frequent ferry services to Prince Islands",
                    "Extended hours make night photography easier"
                ],
                related_attractions=["metro_system", "ferry_system", "prince_islands"]
            )
        ]
        
        # Store events
        for event in seasonal_events:
            self.events[event.event_id] = event
        
        print(f"✅ Initialized {len(seasonal_events)} seasonal events")
    
    def get_current_events(self, date: Optional[datetime] = None) -> List[SeasonalEvent]:
        """Get events active on a specific date (default: today)"""
        
        if date is None:
            date = datetime.now()
        
        active_events = []
        
        for event in self.events.values():
            if event.active and event.start_date <= date <= event.end_date:
                active_events.append(event)
        
        # Sort by impact level (higher first)
        active_events.sort(key=lambda e: e.impact_level, reverse=True)
        
        return active_events
    
    def get_seasonal_recommendations(self, season: Optional[Season] = None) -> Dict[str, Any]:
        """Get recommendations based on current or specified season"""
        
        if season is None:
            season = self._get_current_season()
        
        season_events = [e for e in self.events.values() 
                        if e.season == season and e.active]
        
        # Collect all recommendations
        all_recommendations = []
        affected_attractions = set()
        
        for event in season_events:
            all_recommendations.extend(event.recommendations)
            affected_attractions.update(event.related_attractions)
        
        return {
            'season': season.value,
            'active_events': len(season_events),
            'recommendations': all_recommendations[:10],  # Top 10
            'affected_attractions': list(affected_attractions),
            'high_impact_events': [e for e in season_events if e.impact_level >= 4]
        }
    
    def get_weather_recommendations(self) -> Dict[str, Any]:
        """Get weather-based recommendations for current conditions"""
        
        current_month = datetime.now().month
        
        # Temperature-based recommendations
        if current_month in [12, 1, 2]:  # Winter
            return {
                'season': 'winter',
                'temperature_range': '5-15°C',
                'clothing': 'Warm jacket, umbrella recommended',
                'best_activities': [
                    'Museum visits - perfect indoor weather',
                    'Turkish bath (hammam) experiences',
                    'Shopping in Grand Bazaar and covered markets',
                    'Cozy restaurant experiences'
                ],
                'avoid': [
                    'Long outdoor walking tours',
                    'Ferry rides (can be cold and windy)',
                    'Outdoor dining'
                ],
                'insider_tips': [
                    'Museums are less crowded in winter',
                    'Hammams feel extra luxurious in cold weather',
                    'Perfect time for Turkish tea culture'
                ]
            }
        
        elif current_month in [6, 7, 8]:  # Summer
            return {
                'season': 'summer',
                'temperature_range': '25-35°C',
                'clothing': 'Light clothing, sun protection, comfortable shoes',
                'best_activities': [
                    'Early morning or evening sightseeing',
                    'Bosphorus ferry rides with sea breeze',
                    'Rooftop dining with views',
                    'Swimming at Prince Islands'
                ],
                'avoid': [
                    'Midday outdoor walking (11 AM - 4 PM)',
                    'Heavy museum days (air conditioning varies)',
                    'Crowded indoor markets during heat'
                ],
                'insider_tips': [
                    'Start sightseeing by 9 AM to beat heat',
                    'Take ferry rides for natural air conditioning',
                    'Evening is the best time for Bosphorus views'
                ]
            }
        
        else:  # Spring/Autumn
            return {
                'season': 'spring/autumn',
                'temperature_range': '15-25°C',
                'clothing': 'Layers recommended, light jacket for evenings',
                'best_activities': [
                    'Perfect weather for all outdoor activities',
                    'Walking tours of historic areas',
                    'Park visits and photography',
                    'Outdoor dining experiences'
                ],
                'avoid': [],
                'insider_tips': [
                    'Best overall weather for Istanbul exploration',
                    'Perfect temperature for walking between attractions',
                    'Great season for combining indoor and outdoor activities'
                ]
            }
    
    def get_daily_updates(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get daily updates and recommendations"""
        
        if date is None:
            date = datetime.now()
        
        current_events = self.get_current_events(date)
        seasonal_recs = self.get_seasonal_recommendations()
        weather_recs = self.get_weather_recommendations()
        
        # Generate daily insights
        insights = []
        
        if current_events:
            insights.append(f"📅 {len(current_events)} seasonal events active today")
        
        # Add seasonal insight
        season = self._get_current_season()
        if season == Season.SPRING:
            insights.append("🌸 Spring is perfect for outdoor exploration and park visits")
        elif season == Season.SUMMER:
            insights.append("☀️ Summer heat - plan early morning or evening activities")
        elif season == Season.AUTUMN:
            insights.append("🍂 Autumn offers ideal weather and beautiful colors")
        elif season == Season.WINTER:
            insights.append("❄️ Winter is perfect for museums and indoor cultural experiences")
        
        return {
            'date': date.strftime('%Y-%m-%d'),
            'season': season.value,
            'active_events': [
                {
                    'title': event.title,
                    'description': event.description,
                    'impact_level': event.impact_level,
                    'recommendations': event.recommendations[:3]
                } for event in current_events[:3]
            ],
            'weather_recommendations': weather_recs,
            'daily_insights': insights,
            'last_updated': datetime.now().isoformat()
        }
    
    def _get_current_season(self) -> Season:
        """Determine current season based on date"""
        
        month = datetime.now().month
        
        if month in [3, 4, 5]:
            return Season.SPRING
        elif month in [6, 7, 8]:
            return Season.SUMMER
        elif month in [9, 10, 11]:
            return Season.AUTUMN
        else:
            return Season.WINTER
    
    def add_custom_event(self, event: SeasonalEvent) -> bool:
        """Add a custom seasonal event"""
        
        if event.event_id in self.events:
            return False
        
        self.events[event.event_id] = event
        return True
    
    def update_event_status(self, event_id: str, active: bool) -> bool:
        """Update event active status"""
        
        if event_id not in self.events:
            return False
        
        self.events[event_id].active = active
        return True
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        
        total_events = len(self.events)
        active_events = len([e for e in self.events.values() if e.active])
        current_events = len(self.get_current_events())
        
        return {
            'total_events': total_events,
            'active_events': active_events,
            'current_events': current_events,
            'current_season': self._get_current_season().value,
            'last_update': datetime.now().isoformat(),
            'event_types': {
                event_type.value: len([e for e in self.events.values() 
                                     if e.event_type == event_type])
                for event_type in EventType
            }
        }

# Global instance
seasonal_events_manager = SeasonalEventsManager()

def get_current_seasonal_events() -> List[Dict[str, Any]]:
    """Get current seasonal events"""
    events = seasonal_events_manager.get_current_events()
    return [
        {
            'title': event.title,
            'description': event.description,
            'type': event.event_type.value,
            'season': event.season.value,
            'location': event.location,
            'recommendations': event.recommendations,
            'impact_level': event.impact_level
        } for event in events
    ]

def get_daily_istanbul_updates() -> Dict[str, Any]:
    """Get daily updates for Istanbul"""
    return seasonal_events_manager.get_daily_updates()

def get_weather_based_recommendations() -> Dict[str, Any]:
    """Get weather-based recommendations"""
    return seasonal_events_manager.get_weather_recommendations()
