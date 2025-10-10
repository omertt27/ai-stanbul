#!/usr/bin/env python3
"""
Simplified Istanbul Real-Time Transportation Data
===============================================

Simplified version without external dependencies for demo purposes.
In production, this would integrate with real APIs like:
- İBB Açık Veri Portalı (İBB Open Data Portal) 
- CitySDK Istanbul
- İETT Bus Live GPS Data
- Metro İstanbul Real-Time Status
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class RealTimeTransportData:
    """Real-time transportation data structure"""
    timestamp: datetime
    data_source: str
    transport_type: str
    status: str
    delays: Optional[int] = None
    crowd_level: Optional[str] = None
    reliability_score: Optional[float] = None

@dataclass
class LiveBusData:
    """Live bus/tram GPS data"""
    vehicle_id: str
    route_name: str
    current_location: Tuple[float, float]
    next_stop: str
    estimated_arrival: datetime
    capacity_percentage: int
    delay_minutes: int

class SimplifiedTransportationDataAPI:
    """Simplified Istanbul transportation data for demo"""
    
    def __init__(self):
        # Data cache for API responses
        self.data_cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.last_api_call = {}
        
        logger.info("🌐 Simplified Istanbul Transportation Data API initialized")
    
    def get_live_metro_status(self) -> Dict[str, Any]:
        """📊 Get simulated real-time metro system status"""
        
        cache_key = 'metro_status'
        
        # Simulate real-time metro data (represents actual İBB Metro API data)
        metro_data = {
            'timestamp': datetime.now(),
            'data_source': 'İBB Metro İstanbul API (simulated)',
            'lines': {
                'M1A': {
                    'name': 'Yenikapı - Atatürk Havalimanı',
                    'status': 'operational',
                    'average_delay': 2,
                    'crowd_level': 'moderate',
                    'next_maintenance': None,
                    'disruptions': [],
                    'frequency': '5-8 minutes',
                    'operating_hours': '06:00 - 00:30'
                },
                'M1B': {
                    'name': 'Yenikapı - Kirazlı',
                    'status': 'operational',
                    'average_delay': 0,
                    'crowd_level': 'low',
                    'next_maintenance': None,
                    'disruptions': [],
                    'frequency': '5-8 minutes',
                    'operating_hours': '06:00 - 00:30'
                },
                'M2': {
                    'name': 'Yenikapı - Hacıosman',
                    'status': 'operational',
                    'average_delay': 1,
                    'crowd_level': 'high',
                    'next_maintenance': '2025-10-15 02:00',
                    'disruptions': ['Vezneciler station - elevator maintenance'],
                    'frequency': '3-5 minutes',
                    'operating_hours': '06:00 - 00:30'
                },
                'M3': {
                    'name': 'Kirazlı - Başakşehir/Kayaşehir',
                    'status': 'operational',
                    'average_delay': 0,
                    'crowd_level': 'low',
                    'next_maintenance': None,
                    'disruptions': [],
                    'frequency': '5-8 minutes',
                    'operating_hours': '06:00 - 00:30'
                },
                'M4': {
                    'name': 'Kadıköy - Tavşantepe',
                    'status': 'operational',
                    'average_delay': 3,
                    'crowd_level': 'very_high',
                    'next_maintenance': None,
                    'disruptions': ['Heavy crowds due to football match at Fenerbahçe'],
                    'frequency': '4-6 minutes',
                    'operating_hours': '06:00 - 00:30'
                },
                'M5': {
                    'name': 'Üsküdar - Çekmeköy',
                    'status': 'operational',
                    'average_delay': 1,
                    'crowd_level': 'moderate',
                    'next_maintenance': None,
                    'disruptions': [],
                    'frequency': '5-8 minutes',
                    'operating_hours': '06:00 - 00:30'
                },
                'M6': {
                    'name': 'Levent - Boğaziçi Üniversitesi',
                    'status': 'operational',
                    'average_delay': 0,
                    'crowd_level': 'low',
                    'next_maintenance': None,
                    'disruptions': [],
                    'frequency': '6-10 minutes',
                    'operating_hours': '06:00 - 00:30'
                },
                'M7': {
                    'name': 'Kabataş - Mecidiyeköy',
                    'status': 'operational',
                    'average_delay': 2,
                    'crowd_level': 'moderate',
                    'next_maintenance': None,
                    'disruptions': [],
                    'frequency': '4-7 minutes',
                    'operating_hours': '06:00 - 00:30'
                }
            },
            'system_announcements': [
                'Metro services running normally',
                'M4 line experiencing high passenger volume due to Fenerbahçe match',
                'All lines equipped with wheelchair accessibility'
            ]
        }
        
        # Cache the result
        self.data_cache[cache_key] = metro_data
        self.last_api_call[cache_key] = time.time()
        
        logger.info(f"📊 Metro status updated: {len(metro_data['lines'])} lines operational")
        return metro_data
    
    def get_live_bus_data(self, route_filter: Optional[str] = None) -> List[LiveBusData]:
        """🚌 Get simulated live bus/metrobus GPS data"""
        
        # Simulate live bus data (represents İETT Live Bus API)
        bus_data = [
            LiveBusData(
                vehicle_id='34-BZ-1234',
                route_name='Metrobüs',
                current_location=(41.0082, 28.9784),  # Near Sultanahmet
                next_stop='Kabataş',
                estimated_arrival=datetime.now() + timedelta(minutes=8),
                capacity_percentage=85,
                delay_minutes=3
            ),
            LiveBusData(
                vehicle_id='34-BZ-5678',
                route_name='Metrobüs',
                current_location=(41.0369, 28.9850),  # Taksim area
                next_stop='Mecidiyeköy',
                estimated_arrival=datetime.now() + timedelta(minutes=12),
                capacity_percentage=92,
                delay_minutes=5
            ),
            LiveBusData(
                vehicle_id='34-AB-9012',
                route_name='28T Tram',
                current_location=(41.0129, 28.9594),  # Vezneciler area
                next_stop='Eminönü',
                estimated_arrival=datetime.now() + timedelta(minutes=6),
                capacity_percentage=67,
                delay_minutes=2
            ),
            LiveBusData(
                vehicle_id='34-CD-3456',
                route_name='500T Bus',
                current_location=(41.0255, 28.9744),  # Galata area
                next_stop='Karaköy',
                estimated_arrival=datetime.now() + timedelta(minutes=4),
                capacity_percentage=43,
                delay_minutes=1
            )
        ]
        
        # Filter by route if specified
        if route_filter:
            bus_data = [bus for bus in bus_data if route_filter.lower() in bus.route_name.lower()]
        
        logger.info(f"🚌 Bus data updated: {len(bus_data)} vehicles tracked")
        return bus_data
    
    def get_ferry_schedules(self) -> Dict[str, Any]:
        """⛴️ Get simulated real-time ferry schedules and status"""
        
        current_time = datetime.now()
        
        # Simulate ferry data (represents İDO API)
        ferry_data = {
            'timestamp': current_time,
            'data_source': 'İDO Ferry API (simulated)',
            'routes': {
                'eminonu_kadikoy': {
                    'route_name': 'Eminönü ↔ Kadıköy',
                    'status': 'operational',
                    'frequency': '15 minutes',
                    'next_departures': [
                        (current_time + timedelta(minutes=7)).strftime('%H:%M'),
                        (current_time + timedelta(minutes=22)).strftime('%H:%M'),
                        (current_time + timedelta(minutes=37)).strftime('%H:%M')
                    ],
                    'weather_impact': 'none',
                    'capacity': 'normal',
                    'journey_time': '20 minutes',
                    'cost': '15 TL'
                },
                'besiktas_uskudar': {
                    'route_name': 'Beşiktaş ↔ Üsküdar',
                    'status': 'operational',
                    'frequency': '20 minutes',
                    'next_departures': [
                        (current_time + timedelta(minutes=12)).strftime('%H:%M'),
                        (current_time + timedelta(minutes=32)).strftime('%H:%M'),
                        (current_time + timedelta(minutes=52)).strftime('%H:%M')
                    ],
                    'weather_impact': 'none',
                    'capacity': 'busy',
                    'journey_time': '15 minutes',
                    'cost': '15 TL'
                },
                'kabatas_uskudar': {
                    'route_name': 'Kabataş ↔ Üsküdar',
                    'status': 'operational',
                    'frequency': '30 minutes',
                    'next_departures': [
                        (current_time + timedelta(minutes=18)).strftime('%H:%M'),
                        (current_time + timedelta(minutes=48)).strftime('%H:%M')
                    ],
                    'weather_impact': 'none',
                    'capacity': 'normal',
                    'journey_time': '25 minutes',
                    'cost': '15 TL'
                }
            },
            'weather_conditions': 'favorable_for_ferries',
            'bosphorus_conditions': 'calm_seas',
            'system_notes': [
                'All ferry services running on schedule',
                'Perfect weather for Bosphorus crossing',
                'Wheelchair accessible ferries available'
            ]
        }
        
        logger.info(f"⛴️ Ferry schedules updated: {len(ferry_data['routes'])} routes")
        return ferry_data
    
    def get_traffic_conditions(self) -> Dict[str, Any]:
        """🚗 Get simulated real-time traffic conditions"""
        
        # Simulate traffic data (represents İBB Traffic API)
        traffic_data = {
            'timestamp': datetime.now(),
            'data_source': 'İBB Traffic Management API (simulated)',
            'overall_status': 'moderate',
            'areas': {
                'bosphorus_bridge': {
                    'name': '15 Temmuz Şehitler Köprüsü (Bosphorus Bridge)',
                    'status': 'heavy',
                    'average_speed': 25,
                    'delay_factor': 2.3,
                    'alternative_suggested': 'Use M2 metro or ferry crossing',
                    'traffic_density': '78%'
                },
                'fatih_sultan_mehmet_bridge': {
                    'name': 'Fatih Sultan Mehmet Köprüsü',
                    'status': 'very_heavy',
                    'average_speed': 15,
                    'delay_factor': 3.1,
                    'alternative_suggested': 'Use M7 metro or ferry',
                    'traffic_density': '91%'
                },
                'city_center': {
                    'name': 'Historic Peninsula',
                    'status': 'moderate',
                    'average_speed': 35,
                    'delay_factor': 1.4,
                    'alternative_suggested': 'Walking or T1 tram recommended',
                    'traffic_density': '58%'
                },
                'asian_side': {
                    'name': 'Kadıköy - Üsküdar Area',
                    'status': 'light',
                    'average_speed': 45,
                    'delay_factor': 1.1,
                    'alternative_suggested': 'Normal traffic flow',
                    'traffic_density': '34%'
                },
                'tem_highway': {
                    'name': 'TEM Highway',
                    'status': 'heavy',
                    'average_speed': 30,
                    'delay_factor': 2.0,
                    'alternative_suggested': 'Use Metrobüs for parallel route',
                    'traffic_density': '82%'
                }
            },
            'incidents': [
                {
                    'type': 'minor_accident',
                    'location': 'E-5 Bakırköy direction',
                    'impact': 'moderate',
                    'estimated_clearance': '30 minutes',
                    'lanes_affected': '1 of 3'
                }
            ],
            'recommendations': [
                'Avoid bridge crossings during 17:00-19:00',
                'Metro and ferry are fastest for cross-Bosphorus travel',
                'Consider walking in historic areas during peak hours'
            ]
        }
        
        logger.info(f"🚗 Traffic conditions updated: {traffic_data['overall_status']}")
        return traffic_data
    
    def get_comprehensive_transport_status(self) -> Dict[str, Any]:
        """🌐 Get comprehensive transportation status"""
        
        try:
            metro_data = self.get_live_metro_status()
            bus_data = self.get_live_bus_data()
            ferry_data = self.get_ferry_schedules()
            traffic_data = self.get_traffic_conditions()
            
            comprehensive_status = {
                'timestamp': datetime.now(),
                'data_sources': ['İBB Metro API', 'İETT Bus API', 'İDO Ferry API', 'İBB Traffic API'],
                'metro': metro_data,
                'bus': bus_data,
                'ferry': ferry_data,
                'traffic': traffic_data,
                'system_recommendations': self._generate_system_recommendations(metro_data, ferry_data, traffic_data),
                'overall_status': 'operational_with_minor_delays',
                'best_transport_modes': ['metro', 'ferry', 'walking'],
                'avoid_now': ['car_bridges', 'metrobus_peak_hours']
            }
            
            logger.info("🌐 Comprehensive transport status updated successfully")
            return comprehensive_status
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive transport status: {e}")
            return self._get_fallback_comprehensive_data()
    
    def _generate_system_recommendations(self, metro_data, ferry_data, traffic_data) -> List[str]:
        """Generate intelligent system-wide transportation recommendations"""
        recommendations = []
        
        try:
            # Analyze metro delays
            if isinstance(metro_data, dict) and 'lines' in metro_data:
                high_delay_lines = [line for line, data in metro_data['lines'].items() 
                                   if data.get('average_delay', 0) > 2]
                if high_delay_lines:
                    recommendations.append(f"Metro delays on {', '.join(high_delay_lines)} - consider alternative routes")
            
            # Traffic-based recommendations
            if isinstance(traffic_data, dict) and 'areas' in traffic_data:
                heavy_traffic_areas = [area for area, data in traffic_data['areas'].items() 
                                      if data.get('status') in ['heavy', 'very_heavy']]
                if len(heavy_traffic_areas) >= 2:
                    recommendations.append("Heavy traffic on bridges - metro or ferry strongly recommended for Bosphorus crossing")
            
            # Weather and ferry recommendations
            if isinstance(ferry_data, dict) and ferry_data.get('weather_conditions') == 'favorable_for_ferries':
                recommendations.append("Perfect weather for scenic Bosphorus ferry rides - highly recommended experience")
            
            # Time-based recommendations
            current_hour = datetime.now().hour
            if 17 <= current_hour <= 19:
                recommendations.append("Peak traffic hours - public transport strongly recommended over private vehicles")
            elif 7 <= current_hour <= 9:
                recommendations.append("Morning commute peak - allow extra time and consider metro/ferry")
                
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
        
        # Always provide at least some recommendations
        if not recommendations:
            recommendations = [
                "Metro system is most reliable for tourist attractions",
                "Ferry crossings offer beautiful Bosphorus views",
                "Walking is perfect for exploring historic neighborhoods"
            ]
        
        return recommendations[:4]  # Limit to top 4 recommendations
    
    def _get_fallback_comprehensive_data(self) -> Dict[str, Any]:
        """Fallback data when API calls fail"""
        return {
            'timestamp': datetime.now(),
            'status': 'fallback_mode',
            'message': 'Real-time transportation data temporarily unavailable - using cached information',
            'metro': {'status': 'operational'},
            'ferry': {'status': 'operational'},
            'traffic': {'overall_status': 'moderate'},
            'system_recommendations': ['Use metro for reliable transportation', 'Ferry offers scenic routes']
        }

# Create global instance
istanbul_transport_api = SimplifiedTransportationDataAPI()

def get_real_time_transport_data() -> Dict[str, Any]:
    """Convenience function to get comprehensive real-time transport data"""
    return istanbul_transport_api.get_comprehensive_transport_status()

if __name__ == "__main__":
    # Test the simplified API
    print("🌐 Testing Simplified Istanbul Transportation Data API...")
    
    # Test individual components
    metro_status = istanbul_transport_api.get_live_metro_status()
    print(f"📊 Metro Status: {len(metro_status.get('lines', {}))} lines operational")
    
    bus_data = istanbul_transport_api.get_live_bus_data()
    print(f"🚌 Bus Data: {len(bus_data)} vehicles tracked")
    
    ferry_schedules = istanbul_transport_api.get_ferry_schedules()
    print(f"⛴️ Ferry Routes: {len(ferry_schedules.get('routes', {}))}")
    
    traffic_conditions = istanbul_transport_api.get_traffic_conditions()
    print(f"🚗 Traffic Status: {traffic_conditions.get('overall_status', 'unknown')}")
    
    # Test comprehensive status
    comprehensive = istanbul_transport_api.get_comprehensive_transport_status()
    print(f"🌐 Comprehensive Status: {len(comprehensive['system_recommendations'])} recommendations")
    
    print("✅ All simplified API tests completed successfully!")
