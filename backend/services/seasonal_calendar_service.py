"""
Seasonal Calendar and Events Service
Integrates local festivals, events, and seasonal factors into recommendations
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import uuid

class EventCategory(Enum):
    CULTURAL = "cultural"
    SPORTS = "sports"
    RELIGIOUS = "religious"
    MUSIC = "music"
    FOOD = "food"
    ARTS = "arts"
    SEASONAL = "seasonal"
    TRANSPORTATION = "transportation"

class ImpactLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class SeasonalEvent:
    id: str
    name: str
    category: EventCategory
    start_date: datetime
    end_date: datetime
    description: str
    location: str
    impact_level: ImpactLevel
    related_attractions: List[int]
    ticket_info: str
    weather_dependency: bool = False
    crowd_impact: str = "medium"

class SeasonalCalendarService:
    """Service for managing seasonal events and their impact on recommendations"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.init_database()
        self._populate_default_events()
        
    def init_database(self):
        """Initialize the seasonal calendar database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Seasonal events table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS seasonal_events (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            description TEXT,
            location TEXT,
            impact_level TEXT NOT NULL,
            related_attractions TEXT,  -- JSON array of attraction IDs
            ticket_info TEXT,
            weather_dependency BOOLEAN DEFAULT FALSE,
            crowd_impact TEXT DEFAULT 'medium',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Event impact tracking
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS event_impacts (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            attraction_id INTEGER NOT NULL,
            impact_type TEXT NOT NULL,  -- crowd, pricing, availability
            impact_value REAL,
            impact_description TEXT,
            valid_from TEXT NOT NULL,
            valid_to TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES seasonal_events (id)
        )
        ''')
        
        # Seasonal patterns
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS seasonal_patterns (
            id TEXT PRIMARY KEY,
            attraction_id INTEGER NOT NULL,
            season TEXT NOT NULL,  -- spring, summer, autumn, winter
            month INTEGER NOT NULL,
            avg_crowd_level REAL,
            price_modifier REAL DEFAULT 1.0,
            weather_score REAL,
            recommendation_score REAL,
            best_times TEXT,  -- JSON array of best visit times
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Festival calendar
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS festival_calendar (
            id TEXT PRIMARY KEY,
            festival_name TEXT NOT NULL,
            festival_type TEXT NOT NULL,
            typical_dates TEXT,  -- JSON array of typical date ranges
            locations TEXT,  -- JSON array of locations
            tourist_appeal REAL,
            local_significance REAL,
            transportation_impact TEXT,
            accommodation_impact TEXT,
            description TEXT
        )
        ''')
        
        self.conn.commit()
    
    def _populate_default_events(self):
        """Populate database with key Istanbul events"""
        default_events = [
            {
                "name": "Istanbul Music Festival",
                "category": "CULTURAL",
                "start_date": "2024-06-01",
                "end_date": "2024-06-30",
                "description": "International classical music festival at historic venues",
                "location": "Various historic venues",
                "impact_level": "HIGH",
                "related_attractions": [1, 3, 4, 21],  # Historic venues
                "ticket_info": "Tickets 200-800 TL",
                "crowd_impact": "high"
            },
            {
                "name": "Ramadan and Eid Celebrations",
                "category": "RELIGIOUS",
                "start_date": "2024-03-10",
                "end_date": "2024-04-15",
                "description": "Holy month with special iftar meals and Eid celebrations",
                "location": "Mosques and community centers",
                "impact_level": "VERY_HIGH",
                "related_attractions": [1, 2, 3],  # Mosques
                "ticket_info": "Free community participation",
                "crowd_impact": "very_high"
            },
            {
                "name": "Bosphorus Cross-Continental Swimming",
                "category": "SPORTS",
                "start_date": "2024-07-21",
                "end_date": "2024-07-21",
                "description": "Annual swimming race across the Bosphorus",
                "location": "Bosphorus Strait",
                "impact_level": "VERY_HIGH",
                "related_attractions": [13, 14, 15],  # Bosphorus area
                "ticket_info": "Registration required for participants",
                "crowd_impact": "very_high"
            },
            {
                "name": "Istanbul Tulip Season",
                "category": "SEASONAL",
                "start_date": "2024-04-01",
                "end_date": "2024-05-15",
                "description": "City-wide tulip displays in parks and gardens",
                "location": "Emirgan Park, Gülhane Park, and city gardens",
                "impact_level": "MEDIUM",
                "related_attractions": [20, 21, 22],  # Parks and gardens
                "ticket_info": "Free to view",
                "weather_dependency": True,
                "crowd_impact": "medium"
            },
            {
                "name": "Istanbul Biennial",
                "category": "ARTS",
                "start_date": "2024-09-15",
                "end_date": "2024-11-15",
                "description": "International contemporary art exhibition",
                "location": "Various galleries and cultural spaces",
                "impact_level": "HIGH",
                "related_attractions": [16, 17, 18],  # Art venues
                "ticket_info": "Various pricing, some free venues",
                "crowd_impact": "high"
            },
            {
                "name": "Golden Horn Winter Festival",
                "category": "CULTURAL",
                "start_date": "2024-12-15",
                "end_date": "2024-12-31",
                "description": "Winter market and cultural events along Golden Horn",
                "location": "Golden Horn waterfront",
                "impact_level": "MEDIUM",
                "related_attractions": [11, 12],  # Golden Horn area
                "ticket_info": "Various pricing for events",
                "weather_dependency": True,
                "crowd_impact": "medium"
            }
        ]
        
        # Check if events already exist
        self.cursor.execute('SELECT COUNT(*) FROM seasonal_events')
        if self.cursor.fetchone()[0] > 0:
            return  # Already populated
        
        for event_data in default_events:
            event_id = str(uuid.uuid4())
            self.cursor.execute('''
            INSERT INTO seasonal_events (
                id, name, category, start_date, end_date, description,
                location, impact_level, related_attractions, ticket_info,
                weather_dependency, crowd_impact
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                event_data['name'],
                event_data['category'],
                event_data['start_date'],
                event_data['end_date'],
                event_data['description'],
                event_data['location'],
                event_data['impact_level'],
                json.dumps(event_data['related_attractions']),
                event_data['ticket_info'],
                event_data.get('weather_dependency', False),
                event_data['crowd_impact']
            ))
        
        self.conn.commit()
    
    def add_event(self, name: str, category: str, start_date: str, end_date: str,
                  description: str, location: str, impact_level: str,
                  related_attractions: List[int], ticket_info: str = "",
                  weather_dependency: bool = False, crowd_impact: str = "medium") -> str:
        """Add a new seasonal event"""
        event_id = str(uuid.uuid4())
        
        self.cursor.execute('''
        INSERT INTO seasonal_events (
            id, name, category, start_date, end_date, description,
            location, impact_level, related_attractions, ticket_info,
            weather_dependency, crowd_impact
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event_id, name, category, start_date, end_date, description,
            location, impact_level, json.dumps(related_attractions),
            ticket_info, weather_dependency, crowd_impact
        ))
        
        self.conn.commit()
        return event_id
    
    def get_current_events(self, date: datetime = None) -> List[Dict]:
        """Get events active on a specific date (or current date)"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        
        self.cursor.execute('''
        SELECT id, name, category, start_date, end_date, description,
               location, impact_level, related_attractions, ticket_info,
               crowd_impact
        FROM seasonal_events
        WHERE ? BETWEEN start_date AND end_date
        ORDER BY impact_level DESC, start_date
        ''', (date_str,))
        
        events = []
        for row in self.cursor.fetchall():
            related_attractions = json.loads(row[8]) if row[8] else []
            events.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'start_date': row[3],
                'end_date': row[4],
                'description': row[5],
                'location': row[6],
                'impact_level': row[7],
                'related_attractions': related_attractions,
                'ticket_info': row[9],
                'crowd_impact': row[10]
            })
        
        return events
    
    def get_events_affecting_attraction(self, attraction_id: int, 
                                      date_range: Tuple[datetime, datetime] = None) -> List[Dict]:
        """Get events that affect a specific attraction"""
        if date_range is None:
            start_date = datetime.now()
            end_date = start_date + timedelta(days=30)
        else:
            start_date, end_date = date_range
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        self.cursor.execute('''
        SELECT id, name, category, start_date, end_date, description,
               impact_level, related_attractions, crowd_impact
        FROM seasonal_events
        WHERE (start_date <= ? AND end_date >= ?)
        AND (related_attractions LIKE ? OR related_attractions LIKE ? OR related_attractions LIKE ?)
        ORDER BY impact_level DESC
        ''', (end_str, start_str, f'%[{attraction_id}]%', f'%[{attraction_id},%', f'%,{attraction_id}%'))
        
        events = []
        for row in self.cursor.fetchall():
            related_attractions = json.loads(row[7]) if row[7] else []
            if attraction_id in related_attractions:
                events.append({
                    'id': row[0],
                    'name': row[1],
                    'category': row[2],
                    'start_date': row[3],
                    'end_date': row[4],
                    'description': row[5],
                    'impact_level': row[6],
                    'crowd_impact': row[8]
                })
        
        return events
    
    def get_seasonal_recommendations(self, date: datetime = None, 
                                   category: str = None) -> Dict[str, Any]:
        """Get seasonal recommendations based on current events and patterns"""
        if date is None:
            date = datetime.now()
        
        current_events = self.get_current_events(date)
        month = date.month
        season = self._get_season(month)
        
        recommendations = {
            'season': season,
            'month': month,
            'active_events': current_events,
            'seasonal_highlights': [],
            'crowd_warnings': [],
            'special_opportunities': []
        }
        
        # Analyze current events for recommendations
        for event in current_events:
            if event['impact_level'] in ['HIGH', 'VERY_HIGH']:
                if event['crowd_impact'] in ['high', 'very_high']:
                    recommendations['crowd_warnings'].append({
                        'event': event['name'],
                        'affected_areas': event['location'],
                        'impact': event['crowd_impact'],
                        'advice': f"Expect crowds due to {event['name']}. Visit early morning or consider alternatives."
                    })
                
                recommendations['special_opportunities'].append({
                    'event': event['name'],
                    'category': event['category'],
                    'opportunity': event['description'],
                    'ticket_info': event['ticket_info']
                })
        
        # Add seasonal highlights based on month
        seasonal_highlights = self._get_seasonal_highlights(month)
        recommendations['seasonal_highlights'] = seasonal_highlights
        
        return recommendations
    
    def _get_season(self, month: int) -> str:
        """Get season name from month"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "autumn"
    
    def _get_seasonal_highlights(self, month: int) -> List[Dict]:
        """Get seasonal highlights for a specific month"""
        highlights = {
            1: [{"activity": "Winter warm indoor experiences", "reason": "Cold weather, cozy venues preferred"}],
            2: [{"activity": "Museum visits and indoor cultural sites", "reason": "Still cold, fewer tourists"}],
            3: [{"activity": "Early spring walks, blooming areas", "reason": "Weather improving, nature awakening"}],
            4: [{"activity": "Tulip gardens and parks", "reason": "Peak tulip season in Istanbul"}],
            5: [{"activity": "Outdoor dining and Bosphorus tours", "reason": "Perfect weather, spring blooms"}],
            6: [{"activity": "Music festivals and outdoor events", "reason": "Festival season begins, warm weather"}],
            7: [{"activity": "Swimming and water activities", "reason": "Hot summer, water activities popular"}],
            8: [{"activity": "Early morning and evening activities", "reason": "Peak summer heat, timing important"}],
            9: [{"activity": "Art exhibitions and cultural events", "reason": "Cultural season restarts, pleasant weather"}],
            10: [{"activity": "Walking tours and photography", "reason": "Beautiful autumn light, comfortable temperatures"}],
            11: [{"activity": "Cozy cafes and indoor markets", "reason": "Getting cooler, harvest season"}],
            12: [{"activity": "Winter festivals and warm experiences", "reason": "Winter holidays, festive atmosphere"}]
        }
        
        return highlights.get(month, [])
    
    def get_event_calendar(self, year: int = None) -> Dict[str, List]:
        """Get full event calendar for a year"""
        if year is None:
            year = datetime.now().year
        
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        self.cursor.execute('''
        SELECT name, category, start_date, end_date, impact_level, location
        FROM seasonal_events
        WHERE start_date <= ? AND end_date >= ?
        ORDER BY start_date
        ''', (end_date, start_date))
        
        calendar = {}
        for row in self.cursor.fetchall():
            start = datetime.strptime(row[2], '%Y-%m-%d')
            month_key = start.strftime('%Y-%m')
            
            if month_key not in calendar:
                calendar[month_key] = []
            
            calendar[month_key].append({
                'name': row[0],
                'category': row[1],
                'start_date': row[2],
                'end_date': row[3],
                'impact_level': row[4],
                'location': row[5]
            })
        
        return calendar
    
    def add_event_impact(self, event_id: str, attraction_id: int, impact_type: str,
                        impact_value: float, description: str, valid_from: str, valid_to: str) -> str:
        """Add specific impact tracking for an event"""
        impact_id = str(uuid.uuid4())
        
        self.cursor.execute('''
        INSERT INTO event_impacts (
            id, event_id, attraction_id, impact_type, impact_value,
            impact_description, valid_from, valid_to
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (impact_id, event_id, attraction_id, impact_type, impact_value,
              description, valid_from, valid_to))
        
        self.conn.commit()
        return impact_id
    
    def get_weather_dependent_events(self, date: datetime = None) -> List[Dict]:
        """Get events that depend on weather conditions"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        
        self.cursor.execute('''
        SELECT name, category, start_date, end_date, description, location
        FROM seasonal_events
        WHERE weather_dependency = TRUE
        AND ? BETWEEN start_date AND end_date
        ''', (date_str,))
        
        return [
            {
                'name': row[0],
                'category': row[1],
                'start_date': row[2],
                'end_date': row[3],
                'description': row[4],
                'location': row[5]
            }
            for row in self.cursor.fetchall()
        ]
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
