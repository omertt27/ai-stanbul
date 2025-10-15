"""
Transportation Service
====================

Main transportation service that coordinates all transportation modes and processors.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..models.transportation_models import MetroStation, WalkingRoute, GPSLocation


class EnhancedTransportationSystem:
    """Enhanced transportation system with accurate information"""
    
    def __init__(self):
        self.metro_stations = self._load_metro_stations()
        self.metro_lines = self._load_corrected_metro_lines()
        self.walking_routes = self._load_walking_routes()
        self.timing_data = self._load_timing_data()
        self.cost_data = self._load_cost_data()
        self.accessibility_info = self._load_accessibility_info()
        self.logger = logging.getLogger(__name__)
    
    def _load_metro_stations(self) -> Dict[str, MetroStation]:
        """Load detailed metro station information with coordinates"""
        return {
            # M2 Line stations
            'taksim': MetroStation(
                name='Taksim',
                turkish_name='Taksim',
                line='M2',
                coordinates=(41.0369, 28.9850),
                connections=['Funicular to Kabataş'],
                accessibility=True,
                nearby_attractions=['İstiklal Avenue', 'Galata Tower (15min walk)'],
                exit_info=['Exit A: İstiklal Avenue', 'Exit B: Taksim Square', 'Exit C: Hotels area']
            ),
            'vezneciler': MetroStation(
                name='Vezneciler',
                turkish_name='Vezneciler',
                line='M2',
                coordinates=(41.0129, 28.9594),
                connections=['T1 tram connection via 10min walk to Beyazıt-Kapalıçarşı'],
                accessibility=True,
                nearby_attractions=['Sultanahmet (10min walk)', 'Grand Bazaar (5min walk)'],
                exit_info=['Exit A: University', 'Exit B: Toward Sultanahmet', 'Exit C: Grand Bazaar']
            ),
            'sisane': MetroStation(
                name='Şişhane',
                turkish_name='Şişhane',
                line='M2',
                coordinates=(41.0254, 28.9744),
                connections=['Galata Tower area'],
                accessibility=True,
                nearby_attractions=['Galata Tower (5min walk)', 'Galata Bridge (10min walk)'],
                exit_info=['Exit A: Galata Tower', 'Exit B: Galata district']
            ),
            # T1 Tram stations (KEY for tourists)
            'sultanahmet': MetroStation(
                name='Sultanahmet',
                turkish_name='Sultanahmet',
                line='T1',
                coordinates=(41.0054, 28.9768),
                connections=['T1 tram to Kabataş/Bağcılar'],
                accessibility=True,
                nearby_attractions=['Blue Mosque', 'Hagia Sophia', 'Topkapi Palace'],
                exit_info=['Exit: Historic center, major attractions within 200m']
            ),
            'eminonu': MetroStation(
                name='Eminönü',
                turkish_name='Eminönü',
                line='T1',
                coordinates=(41.0166, 28.9737),
                connections=['Ferry terminals', 'Galata Bridge'],
                accessibility=True,
                nearby_attractions=['Spice Bazaar', 'Ferry terminals', 'Galata Bridge'],
                exit_info=['Exit A: Spice Bazaar', 'Exit B: Ferry terminal', 'Exit C: Galata Bridge']
            ),
            'karakoy': MetroStation(
                name='Karaköy',
                turkish_name='Karaköy',
                line='T1',
                coordinates=(41.0256, 28.9741),
                connections=['Tünel historic subway'],
                accessibility=True,
                nearby_attractions=['Galata Tower (8min walk)', 'Modern art museums'],
                exit_info=['Exit A: Galata Tower direction', 'Exit B: Waterfront']
            ),
            # M1 Line key stations
            'yenikapi': MetroStation(
                name='Yenikapı',
                turkish_name='Yenikapı',
                line='M1A/M1B/M2',
                coordinates=(41.0043, 28.9515),
                connections=['M1A to airport', 'M1B to Kirazlı', 'M2 to Taksim', 'Marmaray'],
                accessibility=True,
                nearby_attractions=['Ferry terminal', 'Marmaray connection'],
                exit_info=['Exit A: Ferry', 'Exit B: Marmaray', 'Exit C: City center']
            )
        }
    
    def _load_corrected_metro_lines(self) -> Dict[str, Dict[str, Any]]:
        """CORRECTED metro line information - fixes Sultanahmet access error"""
        return {
            'M2': {
                'name': 'M2 Vezneciler - Hacıosman',
                'route': 'Vezneciler → Haliç → Şişhane → Taksim → Osmanbey → Şişli-Mecidiyeköy → Gayrettepe → Levent → 4.Levent → İTÜ-Ayazağa → Atatürk Oto Sanayi → Hacıosman',
                'stations': ['Vezneciler', 'Haliç', 'Şişhane', 'Taksim', 'Osmanbey', 'Şişli-Mecidiyeköy', 'Gayrettepe', 'Levent', '4.Levent', 'İTÜ-Ayazağa', 'Atatürk Oto Sanayi', 'Hacıosman'],
                'operation_hours': {
                    'weekdays': '06:00-00:30',
                    'weekends': '06:00-00:30',
                    'frequency_peak': '2-3 minutes',
                    'frequency_off_peak': '4-5 minutes'
                },
                'accessibility': 'Fully wheelchair accessible',
                'key_connections': {
                    'Vezneciler': 'Walking connection to T1 tram at Beyazıt-Kapalıçarşı (10min)',
                    'Şişhane': 'Galata Tower area',
                    'Taksim': 'İstiklal Avenue, Funicular to Kabataş',
                    'Gayrettepe': 'M11 connection to Istanbul Airport'
                },
                'sultanahmet_access': '🚇 M2 to Vezneciler → 🚶‍♂️ 10min walk → 🎯 Sultanahmet'
            },
            'T1': {
                'name': 'T1 Tram Bağcılar - Kabataş',
                'route': 'Bağcılar → Zeytinburnu → Aksaray → Eminönü → Karaköy → Kabataş',
                'stations': ['Bağcılar', 'Zeytinburnu', 'Aksaray', 'Beyazıt-Kapalıçarşı', 'Eminönü', 'Sultanahmet', 'Gülhane', 'Karaköy', 'Kabataş'],
                'operation_hours': {
                    'weekdays': '06:00-23:30',
                    'weekends': '06:00-23:30',
                    'frequency_peak': '3-5 minutes',
                    'frequency_off_peak': '6-8 minutes'
                },
                'accessibility': 'Fully wheelchair accessible',
                'key_connections': {
                    'Zeytinburnu': 'M1A metro connection',
                    'Aksaray': 'M1A metro connection',
                    'Eminönü': 'Ferry terminals',
                    'Sultanahmet': 'Historic attractions - Blue Mosque, Hagia Sophia',
                    'Karaköy': 'Tünel, M2 via walk',
                    'Kabataş': 'Funicular to Taksim, Dolmabahçe Palace'
                },
                'tourist_importance': 'PRIMARY route for historic peninsula access'
            },
            'M11': {
                'name': 'M11 Gayrettepe - Istanbul Airport',
                'route': 'Gayrettepe → Seyrantepe → Kağıthane → Çağlayan → Kemerburgaz → Göktürk → İstanbul Havalimanı',
                'stations': ['Gayrettepe', 'Seyrantepe', 'Kağıthane', 'Çağlayan', 'Kemerburgaz', 'Göktürk', 'İstanbul Havalimanı'],
                'operation_hours': {
                    'weekdays': '06:00-01:00',
                    'weekends': '06:00-01:00',
                    'frequency_peak': '10-12 minutes',
                    'frequency_off_peak': '15-20 minutes'
                },
                'accessibility': 'Fully wheelchair accessible',
                'key_connections': {
                    'Gayrettepe': 'M2 metro connection',
                    'İstanbul Havalimanı': 'Istanbul Airport Terminal'
                },
                'tourist_importance': 'Direct airport connection'
            }
        }
    
    def _load_walking_routes(self) -> Dict[str, WalkingRoute]:
        """Load walking route information"""
        return {
            'taksim_galata_tower': WalkingRoute(
                start='Taksim Square',
                end='Galata Tower',
                distance_km=1.2,
                duration_minutes=15,
                elevation_gain=-50,
                difficulty='easy',
                landmarks=['İstiklal Avenue', 'Galata Bridge view', 'Historic streets'],
                terrain_notes='Mostly downhill, paved streets',
                family_friendly=True,
                accessibility_notes='Steep sections may be challenging for wheelchairs'
            ),
            'sultanahmet_topkapi': WalkingRoute(
                start='Sultanahmet Square',
                end='Topkapi Palace',
                distance_km=0.8,
                duration_minutes=10,
                elevation_gain=30,
                difficulty='easy',
                landmarks=['Blue Mosque', 'Hagia Sophia', 'Sultanahmet Park'],
                terrain_notes='Historic cobblestone areas',
                family_friendly=True,
                accessibility_notes='Some cobblestone sections, mostly accessible'
            )
        }
    
    def _load_timing_data(self) -> Dict[str, Dict[str, Any]]:
        """Load timing and schedule data"""
        return {
            'peak_hours': {
                'morning': '07:30-09:30',
                'evening': '17:30-19:30'
            },
            'service_frequency': {
                'metro_peak': '2-3 minutes',
                'metro_offpeak': '4-6 minutes',
                'tram_peak': '3-5 minutes',
                'tram_offpeak': '6-8 minutes',
                'bus_peak': '5-10 minutes',
                'bus_offpeak': '10-20 minutes'
            },
            'operating_hours': {
                'metro': '06:00-00:30',
                'tram': '06:00-23:30',
                'bus': '05:30-00:30',
                'ferry': '07:00-21:00'
            }
        }
    
    def _load_cost_data(self) -> Dict[str, Any]:
        """Load cost and payment information"""
        return {
            'istanbulkart_prices': {
                'full_fare': 7.67,
                'student': 2.05,
                'senior': 3.84,
                'transfer_discount': True
            },
            'single_use_tickets': {
                'metro_tram': 15.0,
                'bus': 15.0,
                'ferry': 25.0
            },
            'daily_passes': {
                '1_day': 50.0,
                '3_day': 130.0,
                '5_day': 200.0
            },
            'payment_methods': [
                'İstanbulkart (recommended)',
                'Contactless credit/debit card',
                'Mobile payment apps',
                'Single-use tickets'
            ]
        }
    
    def _load_accessibility_info(self) -> Dict[str, Any]:
        """Load accessibility information"""
        return {
            'wheelchair_accessible': {
                'metro_lines': ['M1A', 'M1B', 'M2', 'M11'],
                'tram_lines': ['T1'],
                'stations_with_elevators': ['All metro stations', 'Most tram stations'],
                'accessible_vehicles': '100% of metro/tram fleet'
            },
            'visual_impairment': {
                'tactile_guidance': 'Available at all metro stations',
                'audio_announcements': 'Available in Turkish and English',
                'braille_signage': 'Available at major stations'
            },
            'mobility_assistance': {
                'staff_assistance': 'Available on request',
                'priority_seating': 'Available in all vehicles',
                'platform_gap_assistance': 'Staff assistance available'
            }
        }
    
    def get_station_info(self, station_name: str) -> Optional[MetroStation]:
        """Get detailed information about a metro station"""
        station_key = station_name.lower().replace(' ', '_').replace('ş', 's').replace('ı', 'i').replace('ç', 'c').replace('ğ', 'g').replace('ü', 'u').replace('ö', 'o')
        return self.metro_stations.get(station_key)
    
    def find_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """Find the best route between two locations"""
        # Simplified route finding logic
        origin_station = self.get_station_info(origin)
        destination_station = self.get_station_info(destination)
        
        if origin_station and destination_station:
            return {
                'origin': origin_station,
                'destination': destination_station,
                'route_suggestions': self._calculate_route_suggestions(origin_station, destination_station)
            }
        
        return {
            'error': 'Station information not found',
            'suggestion': 'Please check station names or use nearby major stations'
        }
    
    def _calculate_route_suggestions(self, origin: MetroStation, destination: MetroStation) -> List[Dict[str, Any]]:
        """Calculate route suggestions between stations"""
        suggestions = []
        
        # Same line direct route
        if origin.line == destination.line:
            suggestions.append({
                'type': 'direct',
                'line': origin.line,
                'duration_minutes': abs(hash(origin.name) - hash(destination.name)) % 30 + 10,
                'transfers': 0,
                'cost_tl': 7.67,
                'accessibility': 'Fully accessible'
            })
        else:
            # Transfer route (simplified)
            suggestions.append({
                'type': 'transfer',
                'route': [origin.line, destination.line],
                'duration_minutes': abs(hash(origin.name) - hash(destination.name)) % 45 + 20,
                'transfers': 1,
                'cost_tl': 7.67,  # Same price with İstanbulkart
                'accessibility': 'Fully accessible'
            })
        
        return suggestions
