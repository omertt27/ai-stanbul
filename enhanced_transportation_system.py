#!/usr/bin/env python3
"""
Enhanced Istanbul Transportation System
=======================================

Comprehensive transportation knowledge enhancement to address test findings:
- Metro system accuracy (Sultanahmet access issue)
- Detailed line information and timing
- Walking navigation with distance estimation
- Cost analysis and practical information
- Accessibility details for disabled travelers
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class MetroStation:
    """Detailed metro station information"""
    name: str
    turkish_name: str
    line: str
    coordinates: Tuple[float, float]  # lat, lng for distance calculations
    connections: List[str]
    accessibility: bool
    nearby_attractions: List[str]
    exit_info: List[str]

@dataclass
class WalkingRoute:
    """Detailed walking route information"""
    start: str
    end: str
    distance_km: float
    duration_minutes: int
    elevation_gain: int
    difficulty: str  # easy, moderate, challenging
    landmarks: List[str]
    terrain_notes: str
    family_friendly: bool
    accessibility_notes: str

class EnhancedTransportationSystem:
    """Enhanced transportation system with accurate information"""
    
    def __init__(self):
        self.metro_stations = self._load_metro_stations()
        self.metro_lines = self._load_corrected_metro_lines()
        self.walking_routes = self._load_walking_routes()
        self.timing_data = self._load_timing_data()
        self.cost_data = self._load_cost_data()
        self.accessibility_info = self._load_accessibility_info()
    
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
            'sisli': MetroStation(
                name='Şişli',
                turkish_name='Şişli',
                line='M2',
                coordinates=(41.0458, 28.9867),
                connections=[],
                accessibility=True,
                nearby_attractions=['Cevahir Mall', 'Business district'],
                exit_info=['Exit A: Mall', 'Exit B: Business area']
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
            ),
            'zeytinburnu': MetroStation(
                name='Zeytinburnu',
                turkish_name='Zeytinburnu',
                line='M1A',
                coordinates=(40.9929, 28.9034),
                connections=['T1 tram connection'],
                accessibility=True,
                nearby_attractions=['Connection point'],
                exit_info=['Exit A: T1 tram', 'Exit B: Bus connections']
            )
        }
    
    def _load_corrected_metro_lines(self) -> Dict[str, Dict[str, Any]]:
        """CORRECTED metro line information - fixes Sultanahmet access error"""
        return {
            'M1A': {
                'name': 'M1A Yenikapı - Atatürk Airport',
                'route': 'Yenikapı → Zeytinburnu → Bakırköy → Atatürk Airport',
                'stations': ['Yenikapı', 'Zeytinburnu', 'Esenler', 'Bakırköy', 'Atatürk Airport'],
                'operation_hours': {
                    'weekdays': '06:00-24:00',
                    'weekends': '06:00-24:00',
                    'frequency_peak': '3-5 minutes',
                    'frequency_off_peak': '6-8 minutes'
                },
                'journey_times': {
                    'yenikapi_to_airport': '35 minutes',
                    'zeytinburnu_to_airport': '25 minutes'
                },
                'key_connections': {
                    'Yenikapı': ['M2', 'M1B', 'Marmaray', 'Ferry'],
                    'Zeytinburnu': ['T1 tram - walk 2 minutes']
                },
                'tourist_relevance': 'Airport access, T1 connection at Zeytinburnu'
            },
            'M1B': {
                'name': 'M1B Yenikapı - Kirazlı',
                'route': 'Yenikapı → Esenler → Kirazlı',
                'stations': ['Yenikapı', 'Esenler', 'Başakşehir Şehir Hastanesi', 'Kirazlı'],
                'operation_hours': {
                    'weekdays': '06:00-24:00',
                    'weekends': '06:00-24:00',
                    'frequency_peak': '3-5 minutes',
                    'frequency_off_peak': '6-8 minutes'
                },
                'journey_times': {
                    'yenikapi_to_kirazli': '25 minutes'
                },
                'key_connections': {
                    'Yenikapı': ['M2', 'M1A', 'Marmaray', 'Ferry'],
                    'Kirazlı': ['M3 connection']
                },
                'tourist_relevance': 'Connects to M3 for western suburbs'
            },
            'M2': {
                'name': 'M2 Yenikapı - Hacıosman',
                'route': 'Yenikapı → Vezneciler → Şişhane → Taksim → Şişli → Hacıosman',
                'stations': ['Yenikapı', 'Haliç', 'Vezneciler', 'Üniversite', 'Beyazıt-Kapalıçarşı', 'Şişhane', 'Taksim', 'Osmanbey', 'Şişli', 'Mecidiyeköy', 'Gayrettepe', 'Levent', 'Hacıosman'],
                'operation_hours': {
                    'weekdays': '06:00-24:00',
                    'weekends': '06:00-24:00',
                    'frequency_peak': '2-4 minutes',
                    'frequency_off_peak': '4-6 minutes'
                },
                'journey_times': {
                    'taksim_to_vezneciler': '12 minutes',
                    'vezneciler_to_yenikapi': '8 minutes',
                    'sisane_to_taksim': '5 minutes',
                    'full_line': '35 minutes'
                },
                'key_connections': {
                    'Yenikapı': ['M1A', 'M1B', 'Marmaray', 'Ferry'],
                    'Vezneciler': ['Walk 10min to T1 Beyazıt-Kapalıçarşı', 'Sultanahmet access'],
                    'Şişhane': ['Galata Tower area', 'Walk 8min to T1 Karaköy'],
                    'Taksim': ['Funicular to Kabataş', 'İstiklal Avenue']
                },
                'tourist_relevance': 'MAIN TOURIST LINE - connects all major areas',
                'sultanahmet_access': 'NO DIRECT ACCESS - Use Vezneciler station + 10min walk OR take T1 tram from Zeytinburnu'
            },
            'T1': {
                'name': 'T1 Bağcılar - Kabataş',
                'route': 'Bağcılar → Zeytinburnu → Sultanahmet → Eminönü → Karaköy → Kabataş',
                'stations': ['Bağcılar', 'Zeytinburnu', 'Aksaray', 'Laleli', 'Beyazıt-Kapalıçarşı', 'Eminönü', 'Sultanahmet', 'Gülhane', 'Karaköy', 'Tophane', 'Kabataş'],
                'operation_hours': {
                    'weekdays': '06:00-24:00',
                    'weekends': '06:00-24:00',
                    'frequency_peak': '4-7 minutes',
                    'frequency_off_peak': '8-12 minutes'
                },
                'journey_times': {
                    'sultanahmet_to_taksim_via_kabatas': '20 minutes (T1 + Funicular)',
                    'sultanahmet_to_eminonu': '3 minutes',
                    'karakoy_to_sultanahmet': '8 minutes',
                    'full_line': '55 minutes'
                },
                'key_connections': {
                    'Zeytinburnu': ['M1A connection - walk 2min'],
                    'Beyazıt-Kapalıçarşı': ['Walk 10min to M2 Vezneciler', 'Grand Bazaar access'],
                    'Sultanahmet': ['DIRECT ACCESS to Blue Mosque, Hagia Sophia, Topkapi'],
                    'Eminönü': ['Ferry terminals', 'Spice Bazaar'],
                    'Karaköy': ['Galata Tower 8min walk', 'Tünel historic subway'],
                    'Kabataş': ['Funicular to Taksim', 'Ferry to Princes Islands']
                },
                'tourist_relevance': 'CRITICAL TOURIST LINE - all historic attractions accessible'
            }
        }
    
    def _load_walking_routes(self) -> Dict[str, WalkingRoute]:
        """Detailed walking routes with distance and terrain information"""
        return {
            'galata_bridge_to_spice_bazaar': WalkingRoute(
                start='Galata Bridge',
                end='Spice Bazaar',
                distance_km=0.4,
                duration_minutes=5,
                elevation_gain=0,
                difficulty='easy',
                landmarks=['New Mosque', 'Ferry terminal', 'Rüstem Pasha Mosque'],
                terrain_notes='Flat, paved walkway',
                family_friendly=True,
                accessibility_notes='Wheelchair accessible, smooth pavement'
            ),
            'blue_mosque_to_hagia_sophia': WalkingRoute(
                start='Blue Mosque',
                end='Hagia Sophia',
                distance_km=0.2,
                duration_minutes=3,
                elevation_gain=5,
                difficulty='easy',
                landmarks=['Sultanahmet Square', 'German Fountain'],
                terrain_notes='Historic square, stone pavement',
                family_friendly=True,
                accessibility_notes='Some uneven stones, manageable for wheelchairs'
            ),
            'taksim_to_galata_tower': WalkingRoute(
                start='Taksim Square',
                end='Galata Tower',
                distance_km=1.2,
                duration_minutes=15,
                elevation_gain=-50,
                difficulty='easy',
                landmarks=['İstiklal Avenue', 'Galatasaray High School', 'Fish Market'],
                terrain_notes='Downhill walk, pedestrian street',
                family_friendly=True,
                accessibility_notes='Good for wheelchairs, smooth pedestrian area'
            ),
            'karakoy_to_sultanahmet_via_bridge': WalkingRoute(
                start='Karaköy',
                end='Sultanahmet',
                distance_km=1.8,
                duration_minutes=25,
                elevation_gain=30,
                difficulty='moderate',
                landmarks=['Galata Bridge', 'Eminönü', 'New Mosque', 'Grand Bazaar area'],
                terrain_notes='Bridge crossing, some hills in Sultanahmet',
                family_friendly=True,
                accessibility_notes='Bridge has sidewalk, final approach has some steps'
            ),
            'sultanahmet_walking_tour': WalkingRoute(
                start='Sultanahmet Tram Stop',
                end='Topkapi Palace',
                distance_km=0.8,
                duration_minutes=12,
                elevation_gain=25,
                difficulty='easy',
                landmarks=['Blue Mosque', 'Hagia Sophia', 'Sultanahmet Park', 'Topkapi Palace walls'],
                terrain_notes='Historic area, mixed pavement and stone',
                family_friendly=True,
                accessibility_notes='Some steps near attractions, main paths accessible'
            ),
            'bosphorus_waterfront_walk': WalkingRoute(
                start='Beşiktaş',
                end='Ortaköy',
                distance_km=2.1,
                duration_minutes=30,
                elevation_gain=15,
                difficulty='easy',
                landmarks=['Dolmabahçe Palace', 'Naval Museum', 'Bosphorus Bridge view'],
                terrain_notes='Flat waterfront promenade',
                family_friendly=True,
                accessibility_notes='Excellent wheelchair access, dedicated walkway'
            ),
            'steep_hill_avoidance_sultanahmet': WalkingRoute(
                start='Sultanahmet',
                end='Grand Bazaar',
                distance_km=0.6,
                duration_minutes=8,
                elevation_gain=10,
                difficulty='easy',
                landmarks=['Basilica Cistern', 'Yerebatan Street', 'Nuruosmaniye Mosque'],
                terrain_notes='Gentle slope, avoid steep Divanyolu',
                family_friendly=True,
                accessibility_notes='Wheelchair friendly route avoiding major hills'
            )
        }
    
    def _load_timing_data(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive timing information including rush hour variations"""
        return {
            'metro_timing': {
                'rush_hour_periods': {
                    'morning': '07:30-09:30',
                    'evening': '17:30-19:30'
                },
                'frequency_variations': {
                    'M2_peak': '2-3 minutes',
                    'M2_off_peak': '4-6 minutes',
                    'M2_late_night': '8-10 minutes',
                    'T1_peak': '4-6 minutes',
                    'T1_off_peak': '8-10 minutes',
                    'T1_late_night': '12-15 minutes'
                },
                'journey_times_realistic': {
                    'taksim_to_sultanahmet_total': {
                        'description': 'M2 Taksim → Vezneciler + 10min walk',
                        'peak_time': '25 minutes',
                        'off_peak': '22 minutes',
                        'alternative': 'M2 Taksim → Şişhane + T1 from Karaköy = 30min'
                    },
                    'airport_to_sultanahmet': {
                        'description': 'M1A to Zeytinburnu + T1 to Sultanahmet',
                        'total_time': '50-60 minutes',
                        'cost': '7.67 TL with İstanbulkart',
                        'alternative': 'Havaist bus to Taksim + M2 = 45-75min depending on traffic'
                    },
                    'kadikoy_to_galata_tower': {
                        'description': 'Ferry to Eminönü + walk to Karaköy + walk to tower',
                        'total_time': '35-40 minutes',
                        'ferry_time': '20 minutes',
                        'walking_time': '15-20 minutes',
                        'scenic_bonus': 'Beautiful Bosphorus views during ferry ride'
                    }
                }
            },
            'walking_timing': {
                'average_walking_speed': '4 km/h on flat, 3 km/h uphill, 5 km/h downhill',
                'elderly_adjustment': 'Add 50% more time',
                'family_with_children': 'Add 30% more time',
                'tourist_with_stops': 'Add 100% more time for photo stops',
                'rush_hour_walking': 'Crowded areas may slow walking by 20%'
            }
        }
    
    def _load_cost_data(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive cost information for all transportation modes"""
        return {
            'istanbulkart_costs': {
                'card_purchase': '13 TL (6 TL refundable)',
                'single_ride_metro': '7.67 TL',
                'single_ride_bus': '7.67 TL',
                'single_ride_ferry': '7.67 TL',
                'transfers': {
                    'metro_to_metro': 'Free within 2 hours',
                    'metro_to_bus': '1.40 TL discount',
                    'bus_to_ferry': '1.40 TL discount',
                    'any_to_any': 'Maximum 2 transfers with discounts'
                },
                'daily_caps': 'No daily cap, pay per ride with transfer discounts',
                'student_discount': '50% off all rides with valid student ID'
            },
            'alternative_costs': {
                'single_journey_tokens': {
                    'metro': '15 TL (no transfer discounts)',
                    'bus': '15 TL (no transfer discounts)',
                    'tram': '15 TL (no transfer discounts)',
                    'recommendation': 'Only for single rides, İstanbulkart much cheaper'
                },
                'taxi_estimates': {
                    'airport_to_sultanahmet': '150-200 TL',
                    'taksim_to_sultanahmet': '40-60 TL',
                    'galata_to_sultanahmet': '35-50 TL',
                    'short_rides_city': '25-40 TL minimum'
                },
                'tourist_transport': {
                    'hop_on_hop_off_bus': '120-150 TL per day',
                    'bosphorus_cruise_tour': '80-120 TL',
                    'private_guide_transport': '300-500 TL per day'
                }
            },
            'budget_strategies': {
                'daily_budget_transport': '50-70 TL covers all public transport needs',
                'weekly_transport': '200-300 TL for comprehensive access',
                'money_saving_tips': [
                    'Always use İstanbulkart - saves 50% vs single tickets',
                    'Plan routes to maximize transfer discounts',
                    'Walk short distances rather than take transport',
                    'Use ferries instead of tourist cruises for Bosphorus views',
                    'Student ID gives 50% discount on everything'
                ]
            }
        }
    
    def _load_accessibility_info(self) -> Dict[str, Dict[str, Any]]:
        """Comprehensive accessibility information for disabled travelers"""
        return {
            'metro_accessibility': {
                'wheelchair_accessible_stations': [
                    'All M2 stations (Taksim, Şişli, Vezneciler, Şişhane)',
                    'All T1 stations (Sultanahmet, Eminönü, Karaköy)',
                    'Most M1 stations (Yenikapı, Zeytinburnu)'
                ],
                'elevator_availability': 'All major stations have working elevators',
                'platform_gaps': 'Minimal gaps, suitable for wheelchairs',
                'audio_announcements': 'Available in Turkish and English',
                'tactile_guidance': 'Available at major stations',
                'assistance_available': 'Staff assistance available 06:00-24:00'
            },
            'ferry_accessibility': {
                'wheelchair_boarding': 'All major ferry terminals have ramps',
                'onboard_facilities': 'Designated wheelchair areas on all ferries',
                'accessible_restrooms': 'Available on longer routes',
                'assistance': 'Ferry staff trained to assist disabled passengers'
            },
            'walking_accessibility': {
                'wheelchair_friendly_routes': [
                    'Galata Bridge to Spice Bazaar',
                    'Bosphorus waterfront walks',
                    'İstiklal Avenue (pedestrian only)',
                    'Sultanahmet main square area'
                ],
                'routes_to_avoid': [
                    'Steep hills in Beyoğlu',
                    'Old city back streets (cobblestones)',
                    'Direct routes up to Galata Tower'
                ],
                'rest_points': 'Benches and cafes every 200-300m on main routes',
                'accessible_restrooms': 'Available at all major attractions and transport hubs'
            },
            'general_accessibility_tips': [
                'Download İstanbul Metropolitan Municipality accessibility app',
                'Major attractions have wheelchair access',
                'Most hotels have accessible rooms',
                'Taxi apps allow requesting wheelchair-accessible vehicles',
                'Tourist information centers provide accessibility maps'
            ]
        }
    
    def get_route_to_sultanahmet(self, from_location: str) -> Dict[str, Any]:
        """CORRECTED routing to Sultanahmet - addresses major test failure"""
        routes = {
            'taksim': {
                'recommended_route': 'M2 Taksim → Vezneciler + 10-minute walk to Sultanahmet',
                'total_time': '20-25 minutes',
                'cost': '7.67 TL with İstanbulkart',
                'steps': [
                    '1. Take M2 metro from Taksim towards Yenikapı',
                    '2. Get off at Vezneciler station (3 stops, 12 minutes)',
                    '3. Exit station via Exit B (toward Sultanahmet)',
                    '4. Walk 10 minutes downhill to Sultanahmet Square',
                    '5. Follow signs to Blue Mosque/Hagia Sophia'
                ],
                'alternative_route': 'M2 Taksim → Şişhane → walk to Karaköy → T1 to Sultanahmet',
                'alternative_time': '30-35 minutes',
                'walking_directions': 'From Vezneciler: Head south on İstanbul Üniversitesi Merkez Kampüsü, turn right on Ordu Caddesi, continue to Divanyolu, arrive at Sultanahmet Square',
                'landmarks_on_walk': ['Istanbul University', 'Beyazıt Tower', 'Firuz Ağa Mosque'],
                'accessibility': 'Wheelchair accessible with elevator at Vezneciler'
            },
            'airport_ist': {
                'recommended_route': 'M11 metro to Gayrettepe → M2 to Vezneciler → walk to Sultanahmet',
                'total_time': '60-75 minutes',
                'cost': '15.34 TL with İstanbulkart',
                'steps': [
                    '1. Take M11 from IST Airport to Gayrettepe (35 minutes)',
                    '2. Transfer to M2 at Gayrettepe',
                    '3. Take M2 to Vezneciler (15 minutes)',
                    '4. Walk 10 minutes to Sultanahmet'
                ],
                'alternative_route': 'Havaist bus to Taksim → M2 to Vezneciler → walk',
                'alternative_time': '45-90 minutes (traffic dependent)'
            },
            'kadikoy': {
                'recommended_route': 'Ferry to Eminönü → T1 tram to Sultanahmet',
                'total_time': '35-40 minutes',
                'cost': '15.34 TL with İstanbulkart',
                'steps': [
                    '1. Take ferry from Kadıköy to Eminönü (20 minutes)',
                    '2. Walk 3 minutes to T1 tram stop at Eminönü',
                    '3. Take T1 tram to Sultanahmet (3 minutes)',
                    '4. Exit directly at historic attractions'
                ],
                'scenic_bonus': 'Beautiful Bosphorus and Golden Horn views during ferry ride',
                'accessibility': 'Fully wheelchair accessible'
            },
            'galata_tower': {
                'recommended_route': 'Walk to Karaköy → T1 tram to Sultanahmet',
                'total_time': '20-25 minutes',
                'cost': '7.67 TL with İstanbulkart',
                'steps': [
                    '1. Walk downhill from Galata Tower to Karaköy (8 minutes)',
                    '2. Take T1 tram from Karaköy to Sultanahmet (8 minutes)',
                    '3. Exit directly at attractions'
                ],
                'walking_alternative': 'Walk entire route via Galata Bridge (25 minutes)',
                'walking_highlights': 'Cross historic Galata Bridge, pass Spice Bazaar'
            }
        }
        
        return routes.get(from_location.lower(), {
            'error': f'Route from {from_location} not found',
            'general_advice': 'Use T1 tram line for direct access to Sultanahmet, or M2 to Vezneciler + 10min walk'
        })
    
    def get_walking_distance_estimate(self, start: str, end: str) -> Dict[str, Any]:
        """Estimate walking distance and time between locations"""
        route_key = f"{start.lower()}_to_{end.lower()}"
        
        if route_key in self.walking_routes:
            route = self.walking_routes[route_key]
            return {
                'distance_km': route.distance_km,
                'duration_minutes': route.duration_minutes,
                'difficulty': route.difficulty,
                'elevation_gain': route.elevation_gain,
                'landmarks': route.landmarks,
                'terrain_notes': route.terrain_notes,
                'family_friendly': route.family_friendly,
                'accessibility_notes': route.accessibility_notes
            }
        
        # Use coordinate-based estimation for unknown routes
        return self._estimate_walking_route(start, end)
    
    def _estimate_walking_route(self, start: str, end: str) -> Dict[str, Any]:
        """Estimate walking route using coordinate-based calculation"""
        # Simplified estimation - in real implementation would use Google Maps API
        return {
            'distance_km': 'unknown',
            'duration_minutes': 'unknown',
            'recommendation': f'Use public transport or check specific route from {start} to {end}',
            'general_walking_speed': '4 km/h on flat terrain, 3 km/h uphill',
            'suggestion': 'Consider using metro/tram for distances over 1km'
        }
    
    def get_comprehensive_transportation_guide(self) -> Dict[str, Any]:
        """Get complete transportation guide addressing all test issues"""
        return {
            'critical_corrections': {
                'sultanahmet_access': 'NO DIRECT METRO to Sultanahmet. Use T1 tram OR M2 to Vezneciler + walk',
                'key_connections': 'M2 Vezneciler connects to Sultanahmet area via 10-minute walk',
                'tourist_line': 'T1 tram is the MAIN tourist line - connects all historic attractions'
            },
            'metro_system': self.metro_lines,
            'walking_routes': {route_key: {
                'distance': route.distance_km,
                'time': route.duration_minutes,
                'difficulty': route.difficulty,
                'family_friendly': route.family_friendly
            } for route_key, route in self.walking_routes.items()},
            'timing_information': self.timing_data,
            'cost_breakdown': self.cost_data,
            'accessibility_guide': self.accessibility_info,
            'practical_tips': [
                'Always use İstanbulkart for significant savings',
                'T1 tram provides direct access to all major historic attractions',
                'M2 metro serves modern city areas and connects to historic center via Vezneciler',
                'Walking between close metro/tram stations often faster than transfers',
                'Ferry rides offer scenic transport and are part of public transport system'
            ]
        }

# Integration function for the main knowledge database
def get_enhanced_transportation_system() -> EnhancedTransportationSystem:
    """Get the enhanced transportation system instance"""
    return EnhancedTransportationSystem()

if __name__ == "__main__":
    # Test the enhanced system
    transport = EnhancedTransportationSystem()
    
    # Test Sultanahmet routing - the major issue from tests
    print("🚇 TESTING SULTANAHMET ACCESS (Major Test Issue)")
    print("=" * 60)
    
    from_taksim = transport.get_route_to_sultanahmet('taksim')
    print(f"From Taksim: {from_taksim['recommended_route']}")
    print(f"Time: {from_taksim['total_time']}")
    print(f"Cost: {from_taksim['cost']}")
    print()
    
    # Test walking route estimation
    print("🚶‍♂️ TESTING WALKING ROUTE ESTIMATION")
    print("=" * 60)
    
    walking_route = transport.get_walking_distance_estimate('galata_bridge', 'spice_bazaar')
    print(f"Galata Bridge to Spice Bazaar:")
    print(f"Distance: {walking_route['distance_km']} km")
    print(f"Time: {walking_route['duration_minutes']} minutes")
    print(f"Difficulty: {walking_route['difficulty']}")
    print(f"Family friendly: {walking_route['family_friendly']}")
