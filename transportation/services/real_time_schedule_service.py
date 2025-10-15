"""
Real-Time Schedule Service
=========================

Advanced real-time transportation schedule simulation and management.
Provides dynamic scheduling with traffic, weather, and event considerations.
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class RealTimeScheduleInfo:
    """Real-time schedule information for a transportation mode"""
    line_id: str
    line_name: str
    next_arrivals: List[int]  # Minutes from now
    frequency: str
    status: str  # 'normal', 'delayed', 'disrupted', 'maintenance'
    last_updated: datetime
    disruptions: List[str]
    estimated_delay: int  # Extra minutes due to conditions


class RealTimeScheduleService:
    """Enhanced real-time schedule service with dynamic conditions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache
        
        # Initialize base schedules
        self.base_schedules = self._initialize_base_schedules()
        
        # Current conditions affecting schedules
        self.current_conditions = {
            'weather': 'clear',  # clear, rain, snow, storm
            'traffic_level': 'normal',  # light, normal, heavy, extreme
            'events': [],  # List of ongoing events
            'maintenance': [],  # Planned maintenance
            'time_period': self._get_time_period()
        }
    
    def _initialize_base_schedules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize base schedules for all transportation modes"""
        return {
            # Metro Lines
            'M1A': {
                'name': 'Yenikapı-Atatürk Havalimanı',
                'type': 'metro',
                'base_frequency': {'peak': 3, 'normal': 5, 'night': 12},
                'operating_hours': {'start': '06:00', 'end': '00:30'},
                'stations': ['Yenikapı', 'Zeytinburnu', 'Bakırköy', 'Atatürk Havalimanı'],
                'reliability': 0.95
            },
            'M1B': {
                'name': 'Yenikapı-Kirazlı',
                'type': 'metro',
                'base_frequency': {'peak': 3, 'normal': 5, 'night': 12},
                'operating_hours': {'start': '06:00', 'end': '00:30'},
                'stations': ['Yenikapı', 'Aksaray', 'Esenler', 'Kirazlı'],
                'reliability': 0.95
            },
            'M2': {
                'name': 'Yenikapı-Hacıosman',
                'type': 'metro',
                'base_frequency': {'peak': 2, 'normal': 4, 'night': 10},
                'operating_hours': {'start': '06:00', 'end': '00:30'},
                'stations': ['Yenikapı', 'Şişhane', 'Osmanbey', 'Taksim', 'Şişli', 'Levent', 'Hacıosman'],
                'reliability': 0.97
            },
            'M3': {
                'name': 'Olympiakent-Başakşehir',
                'type': 'metro',
                'base_frequency': {'peak': 4, 'normal': 6, 'night': 15},
                'operating_hours': {'start': '06:00', 'end': '00:30'},
                'stations': ['Olympiakent', 'Başakşehir', 'İkitelli'],
                'reliability': 0.93
            },
            'M4': {
                'name': 'Kadıköy-Tavşantepe',
                'type': 'metro',
                'base_frequency': {'peak': 3, 'normal': 5, 'night': 12},
                'operating_hours': {'start': '06:00', 'end': '00:30'},
                'stations': ['Kadıköy', 'Bostancı', 'Kartal', 'Tavşantepe'],
                'reliability': 0.94
            },
            'M11': {
                'name': 'Gayrettepe-İstanbul Havalimanı',
                'type': 'metro',
                'base_frequency': {'peak': 8, 'normal': 12, 'night': 20},
                'operating_hours': {'start': '06:00', 'end': '01:00'},
                'stations': ['Gayrettepe', 'Kağıthane', 'İstanbul Havalimanı'],
                'reliability': 0.96
            },
            
            # Tram Lines
            'T1': {
                'name': 'Bağcılar-Kabataş',
                'type': 'tram',
                'base_frequency': {'peak': 4, 'normal': 6, 'night': 15},
                'operating_hours': {'start': '06:00', 'end': '00:30'},
                'stations': ['Zeytinburnu', 'Sultanahmet', 'Eminönü', 'Karaköy', 'Kabataş'],
                'reliability': 0.90
            },
            'T4': {
                'name': 'Topkapı-Mescid-i Selam',
                'type': 'tram',
                'base_frequency': {'peak': 5, 'normal': 8, 'night': 20},
                'operating_hours': {'start': '06:00', 'end': '00:30'},
                'stations': ['Topkapı', 'Yusufpaşa', 'Mescid-i Selam'],
                'reliability': 0.88
            },
            
            # Ferry Lines
            'BOSPHORUS_TOUR': {
                'name': 'Bosphorus Tour Ferry',
                'type': 'ferry',
                'base_frequency': {'peak': 30, 'normal': 45, 'night': 120},
                'operating_hours': {'start': '07:00', 'end': '19:00'},
                'stations': ['Eminönü', 'Karaköy', 'Beşiktaş', 'Üsküdar'],
                'reliability': 0.85,
                'weather_dependent': True
            },
            'KADIKOY_EMINONU': {
                'name': 'Kadıköy-Eminönü Ferry',
                'type': 'ferry',
                'base_frequency': {'peak': 20, 'normal': 30, 'night': 60},
                'operating_hours': {'start': '06:30', 'end': '23:30'},
                'stations': ['Kadıköy', 'Eminönü'],
                'reliability': 0.88,
                'weather_dependent': True
            },
            
            # Bus Routes (simplified major routes)
            'BUS_500T': {
                'name': 'Taksim-Beşiktaş Express',
                'type': 'bus',
                'base_frequency': {'peak': 8, 'normal': 12, 'night': 25},
                'operating_hours': {'start': '05:30', 'end': '01:00'},
                'stations': ['Taksim', 'Harbiye', 'Beşiktaş'],
                'reliability': 0.75,
                'traffic_dependent': True
            }
        }
    
    def get_real_time_schedule(self, line_id: str, station: str = None) -> RealTimeScheduleInfo:
        """Get real-time schedule for a specific line"""
        cache_key = f"{line_id}_{station or 'all'}_{datetime.now().minute}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        if line_id not in self.base_schedules:
            return self._create_unknown_line_info(line_id)
        
        schedule_info = self._generate_real_time_info(line_id, station)
        self.cache[cache_key] = schedule_info
        
        return schedule_info
    
    def _generate_real_time_info(self, line_id: str, station: str = None) -> RealTimeScheduleInfo:
        """Generate real-time schedule information with current conditions"""
        base_schedule = self.base_schedules[line_id]
        current_time = datetime.now()
        
        # Determine current frequency based on time period
        time_period = self._get_time_period()
        base_freq = base_schedule['base_frequency'][time_period]
        
        # Apply condition adjustments
        adjusted_freq, delay, disruptions = self._apply_condition_adjustments(
            base_freq, base_schedule, current_time
        )
        
        # Generate next arrivals
        next_arrivals = self._generate_next_arrivals(adjusted_freq, delay)
        
        # Determine status
        status = self._determine_line_status(delay, disruptions)
        
        # Create frequency description
        frequency_desc = self._create_frequency_description(adjusted_freq, time_period)
        
        return RealTimeScheduleInfo(
            line_id=line_id,
            line_name=base_schedule['name'],
            next_arrivals=next_arrivals,
            frequency=frequency_desc,
            status=status,
            last_updated=current_time,
            disruptions=disruptions,
            estimated_delay=delay
        )
    
    def _apply_condition_adjustments(
        self, 
        base_freq: int, 
        base_schedule: Dict[str, Any], 
        current_time: datetime
    ) -> Tuple[int, int, List[str]]:
        """Apply current conditions to base frequency"""
        adjusted_freq = base_freq
        delay = 0
        disruptions = []
        
        # Weather impact
        if self.current_conditions['weather'] == 'rain':
            if base_schedule.get('weather_dependent', False):
                adjusted_freq = int(base_freq * 1.3)
                delay += 3
                disruptions.append("Slight delays due to rain")
        elif self.current_conditions['weather'] == 'snow':
            if base_schedule.get('weather_dependent', False):
                adjusted_freq = int(base_freq * 1.8)
                delay += 8
                disruptions.append("Significant delays due to snow")
        elif self.current_conditions['weather'] == 'storm':
            if base_schedule.get('weather_dependent', False):
                adjusted_freq = int(base_freq * 2.5)
                delay += 15
                disruptions.append("Major delays due to storm conditions")
        
        # Traffic impact (mainly for buses)
        if base_schedule.get('traffic_dependent', False):
            if self.current_conditions['traffic_level'] == 'heavy':
                adjusted_freq = int(base_freq * 1.4)
                delay += 5
                disruptions.append("Traffic delays expected")
            elif self.current_conditions['traffic_level'] == 'extreme':
                adjusted_freq = int(base_freq * 2.0)
                delay += 12
                disruptions.append("Severe traffic delays")
        
        # Random reliability factor
        reliability = base_schedule.get('reliability', 0.9)
        if random.random() > reliability:
            delay += random.randint(2, 8)
            disruptions.append("Minor operational delay")
        
        # Special events (simplified)
        if current_time.weekday() == 6 and current_time.hour >= 14:  # Sunday afternoon
            adjusted_freq = int(base_freq * 1.2)
            disruptions.append("Weekend service adjustments")
        
        return adjusted_freq, delay, disruptions
    
    def _generate_next_arrivals(self, frequency: int, delay: int) -> List[int]:
        """Generate realistic next arrival times"""
        arrivals = []
        
        # First arrival (immediate next)
        first_arrival = frequency + delay + random.randint(-2, 3)
        first_arrival = max(1, first_arrival)  # At least 1 minute
        arrivals.append(first_arrival)
        
        # Next 4 arrivals
        current_arrival = first_arrival
        for _ in range(4):
            next_interval = frequency + random.randint(-1, 2)
            current_arrival += next_interval
            arrivals.append(current_arrival)
        
        return arrivals
    
    def _determine_line_status(self, delay: int, disruptions: List[str]) -> str:
        """Determine overall line status"""
        if delay >= 15:
            return 'disrupted'
        elif delay >= 8:
            return 'delayed'
        elif disruptions:
            return 'minor_delays'
        else:
            return 'normal'
    
    def _create_frequency_description(self, frequency: int, time_period: str) -> str:
        """Create human-readable frequency description"""
        if frequency <= 3:
            return f"Every {frequency} min ({time_period})"
        elif frequency <= 6:
            return f"Every {frequency}-{frequency+2} min ({time_period})"
        elif frequency <= 12:
            return f"Every {frequency}-{frequency+3} min ({time_period})"
        else:
            return f"Every {frequency}-{frequency+5} min ({time_period})"
    
    def _get_time_period(self) -> str:
        """Determine current time period"""
        current_hour = datetime.now().hour
        
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            return 'peak'
        elif 22 <= current_hour <= 6:
            return 'night'
        else:
            return 'normal'
    
    def _create_unknown_line_info(self, line_id: str) -> RealTimeScheduleInfo:
        """Create info for unknown line"""
        return RealTimeScheduleInfo(
            line_id=line_id,
            line_name=f"Line {line_id}",
            next_arrivals=[8, 16, 24, 32, 40],
            frequency="Every 8-10 min (estimated)",
            status='unknown',
            last_updated=datetime.now(),
            disruptions=["Limited real-time data available"],
            estimated_delay=0
        )
    
    def get_comprehensive_area_schedule(self, area: str) -> Dict[str, RealTimeScheduleInfo]:
        """Get schedules for all lines serving a specific area"""
        area_lines = self._get_lines_for_area(area)
        schedules = {}
        
        for line_id in area_lines:
            schedules[line_id] = self.get_real_time_schedule(line_id)
        
        return schedules
    
    def _get_lines_for_area(self, area: str) -> List[str]:
        """Get all transportation lines serving a specific area"""
        area_mapping = {
            'taksim': ['M2', 'BUS_500T'],
            'sultanahmet': ['T1'],
            'kadikoy': ['M4', 'KADIKOY_EMINONU'],
            'eminonu': ['T1', 'KADIKOY_EMINONU', 'BOSPHORUS_TOUR'],
            'airport': ['M11'],
            'karakoy': ['T1', 'BOSPHORUS_TOUR'],
            'besiktas': ['BUS_500T', 'BOSPHORUS_TOUR']
        }
        
        return area_mapping.get(area.lower(), ['M2', 'T1'])  # Default lines
    
    def update_conditions(self, **conditions):
        """Update current conditions affecting schedules"""
        self.current_conditions.update(conditions)
        self.cache.clear()  # Clear cache when conditions change
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_lines = len(self.base_schedules)
        disrupted = sum(1 for line_id in self.base_schedules 
                       if self.get_real_time_schedule(line_id).status in ['disrupted', 'delayed'])
        
        return {
            'total_lines': total_lines,
            'operational': total_lines - disrupted,
            'disrupted': disrupted,
            'system_health': 'good' if disrupted < 2 else 'moderate' if disrupted < 4 else 'poor',
            'current_conditions': self.current_conditions,
            'last_updated': datetime.now()
        }
