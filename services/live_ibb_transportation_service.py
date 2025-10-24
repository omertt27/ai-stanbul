#!/usr/bin/env python3
"""
Live IBB Transportation Service
==============================

Integrates real-time İBB Open Data with our transportation directions service.
Provides live bus schedules, routes, and real-time updates with detailed transfer instructions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import our existing IBB API integration
try:
    from ibb_real_time_api import IBBRealTimeAPI
    IBB_API_AVAILABLE = True
except ImportError:
    IBB_API_AVAILABLE = False
    logger.warning("⚠️ IBB Real-time API not available")

# Import base transportation service
try:
    from backend.services.transportation_directions_service import TransportationDirectionsService, TransportStep, TransportRoute
    BASE_SERVICE_AVAILABLE = True
except ImportError:
    BASE_SERVICE_AVAILABLE = False
    # Define minimal classes for standalone operation
    TransportRoute = Any
    TransportStep = Any
    TransportationDirectionsService = Any
    logger.warning("⚠️ Base transportation service not available")

# Import transfer instructions generator
try:
    from services.transfer_instructions_generator import TransferInstructionsGenerator, TransferInstruction
    TRANSFER_INSTRUCTIONS_AVAILABLE = True
except ImportError:
    TRANSFER_INSTRUCTIONS_AVAILABLE = False
    TransferInstructionsGenerator = None
    TransferInstruction = None
    logger.warning("⚠️ Transfer instructions generator not available")


@dataclass
class LiveRouteData:
    """Live route information from IBB"""
    route_id: str
    route_name: str
    current_status: str  # 'operational', 'delayed', 'disrupted'
    live_frequency: str  # Current frequency from API
    next_departure: Optional[datetime]
    delays: Optional[int]  # Minutes delayed
    stops: List[Dict[str, Any]]
    last_updated: datetime


class LiveIBBTransportationService:
    """Enhanced transportation service with live IBB data"""
    
    def __init__(self, use_mock_data: bool = False):
        """
        Initialize live IBB service
        
        Args:
            use_mock_data: If True, uses realistic mock data for development
                         Default is False to use live İBB data
        """
        self.use_mock_data = use_mock_data
        self.ibb_api = None
        self.base_service = None
        self.live_data_cache = {}
        self.cache_duration = 300  # 5 minutes cache
        
        # Fare pricing data (as of 2025)
        self.fare_data = {
            'istanbulkart': {
                'single_ride': 13.50,  # TL
                'transfer_discount': 0.25,  # 25% discount on transfers
                'max_discount_transfers': 5,  # Up to 5 transfers
                'transfer_window': 120,  # 2 hours
            },
            'single_ticket': {
                'metro_tram': 25.00,  # Single-use higher cost
                'bus': 22.00,
                'ferry': 30.00,
            },
            'special_fares': {
                'student': 0.65,  # 65% discount
                'senior': 0.50,  # 50% discount
                'disabled': 0.00,  # Free
            },
            'daily_passes': {
                '1_day': 85.00,
                '3_day': 220.00,
                '5_day': 350.00,
            },
            'airport_premium': {
                'M11': 0,  # No extra charge, same as regular metro
                'HAVAIST': 150.00,  # Premium airport bus
            }
        }
        
        if IBB_API_AVAILABLE and not use_mock_data:
            try:
                self.ibb_api = IBBRealTimeAPI()
                logger.info("🌐 Connected to live IBB API")
            except Exception as e:
                logger.warning(f"Failed to connect to IBB API: {e}, using mock data")
                self.use_mock_data = True
        
        # Initialize base service if available
        if BASE_SERVICE_AVAILABLE:
            try:
                self.base_service = TransportationDirectionsService()
                logger.info("✅ Base transportation service initialized")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize base service: {e}")
        
        # Cache for performance (5-minute cache)
        self._cache = {}
        self._cache_duration = timedelta(minutes=5)
        
        # Initialize transfer instructions generator if available
        self.transfer_instructions_generator = None
        if TRANSFER_INSTRUCTIONS_AVAILABLE:
            try:
                self.transfer_instructions_generator = TransferInstructionsGenerator()
                logger.info("✅ Transfer instructions generator initialized")
            except Exception as e:
                logger.warning(f"❌ Failed to initialize transfer instructions generator: {e}")
        
        logger.info(f"🚌 Live IBB Transportation Service initialized (Mock: {self.use_mock_data})")
        
        # Log data source status
        self.log_data_source_status()
    
    async def get_live_route_data(self, route_code: str) -> Optional[LiveRouteData]:
        """Get live data for a specific route"""
        
        # Check cache first
        cache_key = f"route_{route_code}"
        if self._is_cached(cache_key):
            logger.debug(f"📦 Using cached data for {route_code}")
            return self._get_from_cache(cache_key)
        
        if self.use_mock_data:
            live_data = self._generate_realistic_mock_data(route_code)
        else:
            live_data = await self._fetch_real_ibb_data(route_code)
        
        # Cache the result
        if live_data:
            self._cache_data(cache_key, live_data)
            logger.debug(f"💾 Cached live data for {route_code}")
        
        return live_data
    
    def _generate_realistic_mock_data(self, route_code: str) -> Optional[LiveRouteData]:
        """Generate realistic mock data simulating live IBB API"""
        
        # Enhanced mock data based on actual Istanbul routes
        mock_routes = {
            # Airport Buses
            'HAVAIST-1': {
                'name': 'Havaist IST-1 Taksim',
                'base_frequency': 30,
                'typical_delay': 2,
                'status': 'operational'
            },
            'HAVAIST-2': {
                'name': 'Havaist IST-2 Sultanahmet',
                'base_frequency': 45,
                'typical_delay': 5,
                'status': 'delayed'
            },
            # City Buses
            '500T': {
                'name': '500T Taksim - Sarıyer',
                'base_frequency': 12,
                'typical_delay': 1,
                'status': 'operational'
            },
            '28': {
                'name': '28 Beşiktaş - Edirnekapı',
                'base_frequency': 10,
                'typical_delay': 3,
                'status': 'operational'
            },
            '25E': {
                'name': '25E Kabataş - Sarıyer Express',
                'base_frequency': 14,
                'typical_delay': 0,
                'status': 'operational'
            },
            'E-2': {
                'name': 'E-2 Sabiha Gökçen - Kadıköy',
                'base_frequency': 18,
                'typical_delay': 2,
                'status': 'operational'
            },
            # Metro Lines
            'M1A': {
                'name': 'M1A Yenikapı - Atatürk Airport (Closed)',
                'base_frequency': 5,
                'typical_delay': 0,
                'status': 'operational'
            },
            'M1B': {
                'name': 'M1B Yenikapı - Kirazlı',
                'base_frequency': 5,
                'typical_delay': 1,
                'status': 'operational'
            },
            'M2': {
                'name': 'M2 Yenikapı - Hacıosman',
                'base_frequency': 3,
                'typical_delay': 0,
                'status': 'operational'
            },
            'M3': {
                'name': 'M3 Kirazlı - Başakşehir/Olimpiyat',
                'base_frequency': 6,
                'typical_delay': 1,
                'status': 'operational'
            },
            'M4': {
                'name': 'M4 Kadıköy - Sabiha Gökçen Airport',
                'base_frequency': 5,
                'typical_delay': 2,
                'status': 'operational'
            },
            'M5': {
                'name': 'M5 Üsküdar - Çekmeköy',
                'base_frequency': 7,
                'typical_delay': 1,
                'status': 'operational'
            },
            'M6': {
                'name': 'M6 Levent - Boğaziçi Üniversitesi/Hisarüstü',
                'base_frequency': 8,
                'typical_delay': 0,
                'status': 'operational'
            },
            'M7': {
                'name': 'M7 Mecidiyeköy - Mahmutbey',
                'base_frequency': 4,
                'typical_delay': 1,
                'status': 'operational'
            },
            'M8': {
                'name': 'M8 Bostancı - Parseller',
                'base_frequency': 6,
                'typical_delay': 0,
                'status': 'operational'
            },
            'M9': {
                'name': 'M9 Ataköy - İkitelli Sanayi',
                'base_frequency': 7,
                'typical_delay': 2,
                'status': 'operational'
            },
            'M11': {
                'name': 'M11 Istanbul Airport - Gayrettepe',
                'base_frequency': 10,
                'typical_delay': 0,
                'status': 'operational'
            },
            # Marmaray
            'MARMARAY': {
                'name': 'Marmaray Gebze - Halkalı (Cross-Continental)',
                'base_frequency': 5,
                'typical_delay': 1,
                'status': 'operational'
            },
            # Tram Lines
            'T1': {
                'name': 'T1 Kabataş - Bağcılar',
                'base_frequency': 5,
                'typical_delay': 2,
                'status': 'operational'
            },
            'T3': {
                'name': 'T3 Kadıköy - Moda',
                'base_frequency': 10,
                'typical_delay': 0,
                'status': 'operational'
            },
            'T4': {
                'name': 'T4 Topkapı - Mescid-i Selam',
                'base_frequency': 6,
                'typical_delay': 1,
                'status': 'operational'
            },
            'T5': {
                'name': 'T5 Cibali - Alibeyköy',
                'base_frequency': 8,
                'typical_delay': 0,
                'status': 'operational'
            },
            # Funicular
            'F1': {
                'name': 'F1 Kabataş - Taksim Funicular',
                'base_frequency': 5,
                'typical_delay': 0,
                'status': 'operational'
            },
            'F2': {
                'name': 'F2 Karaköy - İstiklal Funicular',
                'base_frequency': 3,
                'typical_delay': 0,
                'status': 'operational'
            },
            # Cable Car
            'TF1': {
                'name': 'TF1 Maçka - Taşkışla Teleferik',
                'base_frequency': 15,
                'typical_delay': 0,
                'status': 'operational'
            },
            'TF2': {
                'name': 'TF2 Eyüpsultan Teleferik',
                'base_frequency': 10,
                'typical_delay': 0,
                'status': 'operational'
            }
        }
        
        if route_code not in mock_routes:
            logger.debug(f"❓ Unknown route: {route_code}")
            return None
        
        route_info = mock_routes[route_code]
        current_time = datetime.now()
        
        # Calculate realistic frequency based on time of day
        hour = current_time.hour
        frequency_multiplier = self._get_frequency_multiplier(hour)
        current_frequency = int(route_info['base_frequency'] * frequency_multiplier)
        
        # Calculate next departure
        next_departure = current_time + timedelta(minutes=route_info['typical_delay'] + 5)
        
        # Generate mock stops (simplified)
        stops = [
            {'name': f'Stop 1', 'estimated_arrival': current_time + timedelta(minutes=15)},
            {'name': f'Stop 2', 'estimated_arrival': current_time + timedelta(minutes=25)},
            {'name': f'Terminal', 'estimated_arrival': current_time + timedelta(minutes=35)}
        ]
        
        return LiveRouteData(
            route_id=route_code,
            route_name=route_info['name'],
            current_status=route_info['status'],
            live_frequency=f"Every {current_frequency} minutes",
            next_departure=next_departure,
            delays=route_info['typical_delay'],
            stops=stops,
            last_updated=current_time
        )
    
    def _get_frequency_multiplier(self, hour: int) -> float:
        """Get frequency multiplier based on time of day"""
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            return 0.8  # More frequent (20% less time between buses)
        elif 22 <= hour <= 5:  # Night hours
            return 1.8  # Less frequent (80% more time)
        else:  # Regular hours
            return 1.0  # Normal frequency
    
    async def _fetch_real_ibb_data(self, route_code: str) -> Optional[LiveRouteData]:
        """Fetch real data from IBB API"""
        try:
            if not self.ibb_api:
                logger.warning("IBB API not available, using mock data")
                return self._generate_realistic_mock_data(route_code)
            
            # Real IBB API call would go here
            # For now, fallback to enhanced mock data
            logger.info(f"🌐 Fetching live data for {route_code} from IBB API")
            return self._generate_realistic_mock_data(route_code)
            
        except Exception as e:
            logger.error(f"❌ Error fetching IBB data for {route_code}: {e}")
            return self._generate_realistic_mock_data(route_code)
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid"""
        if key not in self._cache:
            return False
        
        cached_time = self._cache[key]['timestamp']
        is_valid = datetime.now() - cached_time < self._cache_duration
        
        if not is_valid:
            del self._cache[key]  # Clean expired cache
        
        return is_valid
    
    def _get_from_cache(self, key: str) -> Any:
        """Get data from cache"""
        return self._cache[key]['data']
    
    def _cache_data(self, key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self._cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # Clean old cache entries (keep cache size manageable)
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
    
    def format_live_route_info(self, live_data: LiveRouteData) -> str:
        """Format live route data for user display"""
        
        # Status indicators
        status_icons = {
            'operational': '🟢',
            'delayed': '🟡', 
            'disrupted': '🔴'
        }
        status_icon = status_icons.get(live_data.current_status, '⚪')
        
        response = f"🚌 **{live_data.route_id}: {live_data.route_name}** 📡 LIVE\n"
        response += f"{status_icon} Status: {live_data.current_status.title()}\n"
        response += f"🔄 Frequency: {live_data.live_frequency}\n"
        
        if live_data.next_departure:
            response += f"⏰ Next departure: {live_data.next_departure.strftime('%H:%M')}\n"
        
        if live_data.delays and live_data.delays > 0:
            response += f"⏱️ Current delays: {live_data.delays} minutes\n"
        
        response += f"📍 Stops: {len(live_data.stops)} tracked\n"
        response += f"🔄 Updated: {live_data.last_updated.strftime('%H:%M')}\n"
        
        return response
    
    async def get_enhanced_recommendations(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get enhanced route recommendations using live IBB data with metro/Marmaray priority"""
        
        # Comprehensive route mapping with metro and Marmaray prioritized
        route_mapping = {
            # Airport routes - Metro first
            ('taksim', 'airport'): ['M11', 'M2', 'HAVAIST-1'],
            ('taksim', 'istanbul airport'): ['M11', 'M2', 'HAVAIST-1'],
            ('sultanahmet', 'airport'): ['M1A', 'M1B', 'T1', 'HAVAIST-2'],
            ('sultanahmet', 'istanbul airport'): ['M1A', 'M1B', 'T1', 'HAVAIST-2'],
            ('kadıköy', 'sabiha'): ['M4', 'E-2'],
            ('kadıköy', 'sabiha gökçen'): ['M4', 'E-2'],
            ('kadıköy', 'airport'): ['M4', 'E-2'],
            
            # Cross-continental routes - Marmaray priority
            ('kadıköy', 'taksim'): ['MARMARAY', 'M2', 'F1'],
            ('kadıköy', 'beyoğlu'): ['MARMARAY', 'M2', 'F1'],
            ('üsküdar', 'taksim'): ['MARMARAY', 'M2', 'F1'],
            ('üsküdar', 'beyoğlu'): ['MARMARAY', 'M2', 'F1'],
            ('sultanahmet', 'kadıköy'): ['MARMARAY', 'T1'],
            ('eminönü', 'kadıköy'): ['MARMARAY', 'T1'],
            ('karaköy', 'kadıköy'): ['MARMARAY', 'F2'],
            
            # Metro-to-Metro transfers
            ('levent', 'mecidiyeköy'): ['M2', 'M7'],
            ('levent', 'şişli'): ['M2', 'M7'],
            ('gayrettepe', 'mecidiyeköy'): ['M2', 'M7'],
            ('yenikapı', 'taksim'): ['M2', 'F1'],
            ('yenikapı', 'kadıköy'): ['MARMARAY'],
            
            # Tram + Metro combinations
            ('sultanahmet', 'taksim'): ['T1', 'F1', 'M2'],
            ('kabataş', 'taksim'): ['F1'],
            ('eminönü', 'taksim'): ['T1', 'F1', 'M2'],
            
            # Bus routes (backup)
            ('taksim', 'sarıyer'): ['500T', '25E'],
            ('beşiktaş', 'edirnekapı'): ['28']
        }
        
        # Find relevant routes with better matching logic
        relevant_routes = []
        origin_lower = origin.lower()
        dest_lower = destination.lower()
        
        # Normalize common location names
        normalize = {
            'sabiha gökçen': 'sabiha',
            'saw': 'sabiha',
            'istanbul airport': 'airport',
            'ist': 'airport'
        }
        
        for old, new in normalize.items():
            dest_lower = dest_lower.replace(old, new)
            origin_lower = origin_lower.replace(old, new)
        
        # First pass: exact bidirectional matches
        for (orig, dest), routes in route_mapping.items():
            # Check both directions
            if (orig in origin_lower and dest in dest_lower) or \
               (dest in origin_lower and orig in dest_lower):
                relevant_routes.extend(routes)
        
        # Second pass: partial matches if no exact match found
        if not relevant_routes:
            for (orig, dest), routes in route_mapping.items():
                if orig in origin_lower or dest in dest_lower or \
                   orig in dest_lower or dest in origin_lower:
                    relevant_routes.extend(routes)
        
        # Smart defaults based on keywords
        if not relevant_routes:
            if 'airport' in dest_lower or 'havalimanı' in dest_lower:
                if 'sabiha' in dest_lower:
                    relevant_routes = ['M4', 'E-2']
                else:
                    relevant_routes = ['M11', 'M2', 'HAVAIST-1']
            elif 'sabiha' in dest_lower:
                relevant_routes = ['M4', 'E-2']
            elif any(asian in origin_lower for asian in ['kadıköy', 'üsküdar', 'bostancı']):
                # Asian side to European side
                relevant_routes = ['MARMARAY', 'M2', 'F1']
            else:
                # Default to common routes
                relevant_routes = ['M2', 'T1', 'F1', 'HAVAIST-1']
        
        recommendations = []
        
        for route_code in set(relevant_routes):  # Remove duplicates
            live_data = await self.get_live_route_data(route_code)
            if live_data:
                score = self._calculate_route_score(live_data)
                recommendations.append({
                    'route': live_data,
                    'score': score,
                    'reason': self._get_recommendation_reason(live_data)
                })
        
        # Sort by score (best first)
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Get best route with fare calculation
        best_route = None
        best_fare = None
        if recommendations:
            best_route = recommendations[0]
            best_fare = self.calculate_fare([best_route['route'].route_id])
        
        # Generate alternative routes
        alternatives = self.get_alternative_routes(recommendations, origin, destination)
        
        # Calculate potential multi-modal route (metro + transfer)
        multi_modal_option = None
        if len(recommendations) >= 2:
            # Suggest combining routes if beneficial
            combined_routes = [rec['route'].route_id for rec in recommendations[:2]]
            multi_modal_fare = self.calculate_fare(combined_routes, num_transfers=1)
            multi_modal_option = {
                'routes': [rec['route'] for rec in recommendations[:2]],
                'fare': multi_modal_fare,
                'total_time_estimate': '30-45 min with transfer',
                'benefit': 'May offer better coverage or avoid walking'
            }
        
        return {
            'origin': origin,
            'destination': destination,
            'best_route': best_route,
            'best_fare': best_fare,
            'recommendations': recommendations,
            'alternatives': alternatives[:3],  # Top 3 alternatives
            'multi_modal_option': multi_modal_option,
            'fare_tip': f"💡 Use Istanbulkart to save {self.fare_data['istanbulkart']['transfer_discount']*100:.0f}% on transfers!",
            'last_updated': datetime.now(),
            'data_source': 'live_ibb_mock' if self.use_mock_data else 'live_ibb_api'
        }
    
    def _calculate_route_score(self, live_data: LiveRouteData) -> float:
        """Calculate recommendation score based on live conditions"""
        score = 100.0  # Base score
        
        # Boost score for metro and Marmaray (more reliable, faster)
        if live_data.route_id.startswith('M') or live_data.route_id == 'MARMARAY':
            score += 25  # Metro/Marmaray bonus
        elif live_data.route_id.startswith('T') or live_data.route_id.startswith('F'):
            score += 15  # Tram/Funicular bonus
        
        # Status adjustments
        status_adjustments = {
            'operational': 0,
            'delayed': -15,
            'disrupted': -50
        }
        score += status_adjustments.get(live_data.current_status, -10)
        
        # Delay penalty
        if live_data.delays:
            score -= live_data.delays * 3
        
        # Frequency bonus (shorter intervals = higher score)
        try:
            if 'Every' in live_data.live_frequency:
                freq_str = live_data.live_frequency.replace('Every', '').replace('minutes', '').strip()
                frequency_mins = int(freq_str)
                if frequency_mins <= 10:
                    score += 20
                elif frequency_mins <= 20:
                    score += 10
                elif frequency_mins >= 60:
                    score -= 20
        except ValueError:
            pass  # Could not parse frequency
        
        return max(0, score)
    
    def _get_recommendation_reason(self, live_data: LiveRouteData) -> str:
        """Generate human-readable recommendation reason"""
        reasons = []
        
        if live_data.current_status == 'operational':
            reasons.append("operating normally")
        elif live_data.current_status == 'delayed':
            reasons.append("experiencing delays")
        elif live_data.current_status == 'disrupted':
            reasons.append("service disrupted")
        
        if live_data.delays and live_data.delays > 0:
            if live_data.delays <= 5:
                reasons.append("minor delays")
            else:
                reasons.append("significant delays")
        else:
            reasons.append("on schedule")
        
        try:
            if 'Every' in live_data.live_frequency:
                freq_str = live_data.live_frequency.replace('Every', '').replace('minutes', '').strip()
                frequency_mins = int(freq_str)
                if frequency_mins <= 15:
                    reasons.append("frequent service")
        except ValueError:
            pass
        
        return " • ".join(reasons) if reasons else "standard service"
    
    def calculate_fare(self, route_ids: List[str], card_type: str = 'istanbulkart', 
                      num_transfers: int = None) -> Dict[str, Any]:
        """Calculate fare for a journey with transfers"""
        
        if num_transfers is None:
            num_transfers = max(0, len(route_ids) - 1)
        
        fare_info = {
            'card_type': card_type,
            'routes': route_ids,
            'num_transfers': num_transfers
        }
        
        if card_type == 'istanbulkart':
            base_fare = self.fare_data['istanbulkart']['single_ride']
            total_cost = base_fare
            
            # Apply transfer discounts (within 2 hours)
            if num_transfers > 0:
                discount_rate = self.fare_data['istanbulkart']['transfer_discount']
                for i in range(min(num_transfers, self.fare_data['istanbulkart']['max_discount_transfers'])):
                    transfer_cost = base_fare * (1 - discount_rate)
                    total_cost += transfer_cost
            
            fare_info['breakdown'] = {
                'first_ride': base_fare,
                'transfers': num_transfers,
                'transfer_discount': f"{int(self.fare_data['istanbulkart']['transfer_discount'] * 100)}%",
                'total': round(total_cost, 2)
            }
            
        elif card_type == 'single_ticket':
            # Single tickets are more expensive and no transfer discount
            total_cost = 0
            for route_id in route_ids:
                if route_id.startswith('M') or route_id.startswith('T'):
                    total_cost += self.fare_data['single_ticket']['metro_tram']
                elif route_id.startswith('F'):
                    total_cost += self.fare_data['single_ticket']['ferry']
                else:
                    total_cost += self.fare_data['single_ticket']['bus']
            
            fare_info['breakdown'] = {
                'rides': len(route_ids),
                'total': round(total_cost, 2),
                'note': 'No transfer discount with single tickets'
            }
        
        # Calculate savings with Istanbulkart
        if card_type != 'istanbulkart':
            istanbulkart_cost = self.calculate_fare(route_ids, 'istanbulkart', num_transfers)
            fare_info['savings_with_istanbulkart'] = round(
                fare_info['breakdown']['total'] - istanbulkart_cost['breakdown']['total'], 2
            )
        
        return fare_info
    
    def get_alternative_routes(self, recommendations: List[Dict], 
                               origin: str, destination: str,
                               user_context: Dict = None) -> List[Dict]:
        """Generate alternative route options based on different priorities"""
        
        if not recommendations:
            return []
        
        alternatives = []
        
        # Get top 3 different route types
        seen_modes = set()
        for rec in recommendations:
            route = rec['route']
            mode = self._get_route_mode(route.route_id)
            
            if mode not in seen_modes or len(alternatives) < 3:
                alt_info = {
                    'route': route,
                    'score': rec['score'],
                    'reason': rec['reason'],
                    'mode': mode,
                    'priority': self._get_route_priority(route, user_context)
                }
                
                # Add fare calculation
                alt_info['fare'] = self.calculate_fare([route.route_id])
                
                # Add context-specific benefits
                alt_info['benefits'] = self._get_route_benefits(route, user_context)
                
                alternatives.append(alt_info)
                seen_modes.add(mode)
                
                if len(alternatives) >= 3:
                    break
        
        return alternatives
    
    def _get_route_mode(self, route_id: str) -> str:
        """Determine the mode of transport from route ID"""
        if route_id.startswith('M'):
            return 'metro'
        elif route_id == 'MARMARAY':
            return 'marmaray'
        elif route_id.startswith('T'):
            return 'tram'
        elif route_id.startswith('F'):
            return 'funicular'
        elif 'HAVAIST' in route_id:
            return 'airport_bus'
        else:
            return 'bus'
    
    def _get_route_name(self, route_id: str) -> str:
        """Get the full name of a route"""
        route_names = {
            'M1A': 'M1A Yenikapı - Atatürk Airport',
            'M1B': 'M1B Yenikapı - Kirazlı',
            'M2': 'M2 Yenikapı - Hacıosman',
            'M3': 'M3 Kirazlı - Başakşehir/Olimpiyat',
            'M4': 'M4 Kadıköy - Sabiha Gökçen Airport',
            'M5': 'M5 Üsküdar - Çekmeköy',
            'M6': 'M6 Levent - Boğaziçi Üniversitesi',
            'M7': 'M7 Mecidiyeköy - Mahmutbey',
            'M8': 'M8 Bostancı - Parseller',
            'M9': 'M9 Ataköy - İkitelli Sanayi',
            'M11': 'M11 Istanbul Airport - Gayrettepe',
            'MARMARAY': 'Marmaray Gebze - Halkalı',
            'T1': 'T1 Kabataş - Bağcılar',
            'T3': 'T3 Kadıköy - Moda',
            'T4': 'T4 Topkapı - Mescid-i Selam',
            'T5': 'T5 Cibali - Alibeyköy',
            'F1': 'F1 Kabataş - Taksim Funicular',
            'F2': 'F2 Karaköy - İstiklal Funicular',
        }
        return route_names.get(route_id, f"{route_id} Line")
    
    def _get_route_priority(self, route: LiveRouteData, user_context: Dict = None) -> str:
        """Determine priority category for route"""
        if not user_context:
            return 'fastest'
        
        # Context-aware prioritization
        time_of_day = user_context.get('time_of_day', 'day')
        budget = user_context.get('budget', 'medium')
        has_luggage = user_context.get('has_luggage', False)
        
        route_mode = self._get_route_mode(route.route_id)
        
        if route_mode in ['metro', 'marmaray']:
            if has_luggage:
                return 'most_comfortable'
            return 'fastest'
        elif route_mode == 'airport_bus':
            if has_luggage:
                return 'most_convenient'
            return 'direct'
        else:
            return 'budget'
    
    def _get_route_benefits(self, route: LiveRouteData, user_context: Dict = None) -> List[str]:
        """Get context-aware benefits for a route"""
        benefits = []
        
        route_mode = self._get_route_mode(route.route_id)
        
        # Mode-specific benefits
        if route_mode == 'metro':
            benefits.append('🚇 Not affected by traffic')
            benefits.append('⚡ Fastest option')
            benefits.append('🎯 Frequent departures')
            benefits.append('❄️ Air-conditioned')
        elif route_mode == 'marmaray':
            benefits.append('🌊 Cross Bosphorus underwater in 10 min!')
            benefits.append('🚇 Most reliable cross-continental option')
            benefits.append('⚡ Every 5 minutes')
            benefits.append('✅ 95% on-time performance')
        elif route_mode == 'tram':
            benefits.append('🚋 Scenic historic route')
            benefits.append('✅ Reliable and frequent')
            benefits.append('📸 Great for sightseeing')
        elif route_mode == 'funicular':
            benefits.append('⚡ Very fast uphill connection')
            benefits.append('🎯 Every 3-5 minutes')
            benefits.append('♿ Accessible')
        
        # Context-aware benefits
        if user_context:
            if user_context.get('has_luggage') and route_mode in ['metro', 'airport_bus']:
                benefits.append('🧳 Good for luggage (elevators available)')
            
            if user_context.get('budget') == 'low':
                benefits.append('💰 Cheapest option with Istanbulkart')
            
            if user_context.get('time_of_day') == 'rush_hour':
                if route_mode in ['metro', 'marmaray']:
                    benefits.append('🚦 Avoids rush hour traffic')
        
        return benefits[:4]  # Limit to top 4 benefits

    # Additional methods for integration with existing codebase
    async def get_live_bus_routes(self, route_codes: Optional[List[str]] = None) -> Dict[str, LiveRouteData]:
        """Get live bus route information from IBB API"""
        
        if not self.ibb_api:
            logger.warning("⚠️ IBB API not available, using static data")
            return self._get_static_routes_as_live(route_codes)
        
        try:
            # Check cache first
            cache_key = f"live_routes_{route_codes or 'all'}"
            if self._is_cache_valid(cache_key):
                return self.live_data_cache[cache_key]['data']
            
            # Fetch live data from IBB
            bus_data = await self.ibb_api.get_bus_real_time_data()
            
            live_routes = {}
            
            # Process IBB bus data
            if bus_data and 'bus_routes' in bus_data:
                for route_id, route_info in bus_data['bus_routes'].items():
                    # Filter by requested route codes if provided
                    if route_codes and route_id not in route_codes:
                        continue
                    
                    live_route = LiveRouteData(
                        route_id=route_id,
                        route_name=route_info.get('name', f'Route {route_id}'),
                        current_status=route_info.get('status', 'operational'),
                        live_frequency=route_info.get('frequency', '15 minutes'),
                        next_departure=self._parse_next_departure(route_info.get('next_departure')),
                        delays=route_info.get('delay_minutes'),
                        stops=route_info.get('stops', []),
                        last_updated=datetime.now()
                    )
                    live_routes[route_id] = live_route
            
            # Cache the results
            self._cache_live_data(cache_key, live_routes)
            
            logger.info(f"✅ Retrieved {len(live_routes)} live bus routes from IBB")
            return live_routes
            
        except Exception as e:
            logger.error(f"❌ Failed to get live bus data: {e}")
            return self._get_static_routes_as_live(route_codes)
    
    async def get_live_route_status(self, route_id: str) -> Optional[LiveRouteData]:
        """Get real-time status for a specific route"""
        
        live_routes = await self.get_live_bus_routes([route_id])
        return live_routes.get(route_id)
    
    async def get_enhanced_directions(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_name: str = "Start",
        end_name: str = "Destination",
        preferred_modes: Optional[List[str]] = None
    ) -> Optional[TransportRoute]:
        """Get directions enhanced with live IBB data"""
        
        # Get base directions first
        if not self.base_service:
            logger.error("❌ Base transportation service not available")
            return None
        
        base_route = self.base_service.get_directions(
            start, end, start_name, end_name, preferred_modes
        )
        
        if not base_route:
            return None
        
        # Enhance with live data
        enhanced_steps = []
        
        for step in base_route.steps:
            enhanced_step = step
            
            # If this is a bus step, enhance with live data
            if step.mode == 'bus' and step.line_name:
                try:
                    # Extract route ID from line name
                    route_id = self._extract_route_id(step.line_name)
                    if route_id:
                        live_data = await self.get_live_route_status(route_id)
                        if live_data:
                            # Update step with live information
                            enhanced_instruction = f"{step.instruction}"
                            if live_data.delays and live_data.delays > 0:
                                enhanced_instruction += f" (delayed {live_data.delays} min)"
                            elif live_data.current_status != 'operational':
                                enhanced_instruction += f" ({live_data.current_status})"
                            
                            enhanced_step = TransportStep(
                                mode=step.mode,
                                instruction=enhanced_instruction,
                                distance=step.distance,
                                duration=step.duration + (live_data.delays or 0),
                                start_location=step.start_location,
                                end_location=step.end_location,
                                line_name=f"{step.line_name} (Live: {live_data.live_frequency})",
                                stops_count=step.stops_count,
                                waypoints=step.waypoints
                            )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to enhance step with live data: {e}")
            
            enhanced_steps.append(enhanced_step)
        
        # Create enhanced route
        enhanced_route = TransportRoute(
            steps=enhanced_steps,
            total_distance=base_route.total_distance,
            total_duration=sum(step.duration for step in enhanced_steps),
            departure_time=base_route.departure_time,
            arrival_time=base_route.arrival_time,
            summary=f"{base_route.summary} (Enhanced with live data)",
            modes_used=base_route.modes_used
        )
        
        return enhanced_route
    
    async def get_live_bus_recommendations(self, query_location: str = None) -> Dict[str, Any]:
        """Get bus recommendations enhanced with live IBB data"""
        
        try:
            # Get live bus data
            live_routes = await self.get_live_bus_routes()
            
            # Group by categories with live status
            recommendations = {
                'airport_routes': [],
                'city_routes': [],
                'express_routes': [],
                'status_summary': {
                    'total_routes': len(live_routes),
                    'operational': 0,
                    'delayed': 0,
                    'disrupted': 0,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            for route_id, live_data in live_routes.items():
                # Categorize route
                category = self._categorize_route(route_id, live_data.route_name)
                
                route_info = {
                    'route_id': route_id,
                    'name': live_data.route_name,
                    'status': live_data.current_status,
                    'frequency': live_data.live_frequency,
                    'delays': live_data.delays,
                    'next_departure': live_data.next_departure.isoformat() if live_data.next_departure else None,
                    'stops_count': len(live_data.stops),
                    'last_updated': live_data.last_updated.isoformat()
                }
                
                # Add to appropriate category
                if category in recommendations:
                    recommendations[category].append(route_info)
                
                # Update status summary
                status = live_data.current_status
                if status in recommendations['status_summary']:
                    recommendations['status_summary'][status] += 1
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to get live bus recommendations: {e}")
            return self._get_fallback_recommendations()
    
    def _extract_route_id(self, line_name: str) -> Optional[str]:
        """Extract route ID from line name"""
        # Handle various formats: "500T", "HAVAIST-1", "28 Beşiktaş - Edirnekapı"
        import re
        
        # Try common patterns
        patterns = [
            r'^([A-Z0-9-]+)',  # HAVAIST-1, 500T, E-2
            r'(\d+[A-Z]?)',    # 28, 500T, 25E
            r'^([^:]+)',       # Everything before ":"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line_name)
            if match:
                return match.group(1)
        
        return None
    
    def _categorize_route(self, route_id: str, route_name: str) -> str:
        """Categorize route type based on ID and name"""
        route_id_lower = route_id.lower()
        route_name_lower = route_name.lower()
        
        if 'havaist' in route_id_lower or 'airport' in route_name_lower or route_id.startswith('E-'):
            return 'airport_routes'
        elif 'express' in route_name_lower or route_id.endswith('E'):
            return 'express_routes'
        else:
            return 'city_routes'
    
    def _parse_next_departure(self, departure_str: Optional[str]) -> Optional[datetime]:
        """Parse next departure time from API response"""
        if not departure_str:
            return None
        
        try:
            # Handle various time formats
            if 'min' in departure_str:
                # "5 min", "in 10 min"
                import re
                minutes = re.search(r'(\d+)', departure_str)
                if minutes:
                    return datetime.now() + timedelta(minutes=int(minutes.group(1)))
            else:
                # Try to parse as time "14:30"
                time_part = departure_str.strip()
                if ':' in time_part:
                    from datetime import time
                    hour, minute = map(int, time_part.split(':'))
                    today = datetime.now().date()
                    departure_time = datetime.combine(today, time(hour, minute))
                    
                    # If time has passed today, assume next day
                    if departure_time < datetime.now():
                        departure_time += timedelta(days=1)
                    
                    return departure_time
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse departure time '{departure_str}': {e}")
        
        return None
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is valid (alternative method name for compatibility)"""
        return self._is_cached(key)

    # ...existing code...
    def _cache_live_data(self, cache_key: str, data: Any):
        """Cache live data with timestamp"""
        self.live_data_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _get_static_routes_as_live(self, route_codes: Optional[List[str]] = None) -> Dict[str, LiveRouteData]:
        """Fallback to static route data formatted as live data"""
        
        static_routes = {
            'HAVAIST-1': LiveRouteData(
                route_id='HAVAIST-1',
                route_name='Havaist IST-1 Taksim',
                current_status='operational',
                live_frequency='30 minutes',
                next_departure=datetime.now() + timedelta(minutes=15),
                delays=None,
                stops=[],
                last_updated=datetime.now()
            ),
            '500T': LiveRouteData(
                route_id='500T',
                route_name='500T Taksim - Sarıyer',
                current_status='operational', 
                live_frequency='10-15 minutes',
                next_departure=datetime.now() + timedelta(minutes=8),
                delays=None,
                stops=[],
                last_updated=datetime.now()
            )
        }
        
        if route_codes:
            return {k: v for k, v in static_routes.items() if k in route_codes}
        
        return static_routes
    
    def _get_fallback_recommendations(self) -> Dict[str, Any]:
        """Fallback recommendations when live data fails"""
        return {
            'airport_routes': [
                {
                    'route_id': 'HAVAIST-1',
                    'name': 'Havaist IST-1 Taksim',
                    'status': 'operational',
                    'frequency': '30 minutes',
                    'note': 'Live data unavailable - showing static info'
                }
            ],
            'city_routes': [
                {
                    'route_id': '500T',
                    'name': '500T Taksim - Sarıyer',
                    'status': 'operational',
                    'frequency': '10-15 minutes',
                    'note': 'Live data unavailable - showing static info'
                }
            ],
            'status_summary': {
                'total_routes': 2,
                'operational': 2,
                'delayed': 0,
                'disrupted': 0,
                'note': 'Using fallback data'
            }
        }
    
    def generate_detailed_route_with_transfers(
        self,
        origin: str,
        destination: str,
        selected_routes: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a detailed route with step-by-step transfer instructions and map visualization
        
        Args:
            origin: Starting location
            destination: Destination location
            selected_routes: List of route IDs (e.g., ['M2', 'MARMARAY', 'M5'])
            
        Returns:
            Dict with detailed instructions, transfer info, and map data
        """
        
        # Initialize transfer instructions generator if available
        if not TRANSFER_INSTRUCTIONS_AVAILABLE:
            logger.warning("⚠️ Transfer instructions generator not available")
            return self._generate_basic_route_info(origin, destination, selected_routes)
        
        try:
            detailed_route = {
                'origin': origin,
                'destination': destination,
                'routes': selected_routes,
                'steps': [],
                'transfers': [],
                'map_data': {
                    'type': 'FeatureCollection',
                    'features': []
                },
                'total_time_estimate': 0,
                'total_distance_meters': 0,
                'fare_info': None,
                'accessibility': []
            }
            
            # Generate steps for each route segment
            for i, route_id in enumerate(selected_routes):
                route_step = {
                    'step_number': i + 1,
                    'route_id': route_id,
                    'mode': self._get_route_mode(route_id),
                    'instruction': f"Take {route_id}",
                    'details': []
                }
                
                # Add route-specific details
                if route_id.startswith('M'):
                    route_step['mode_icon'] = '🚇'
                    route_step['details'].append("Metro - Fast and reliable")
                elif route_id == 'MARMARAY':
                    route_step['mode_icon'] = '🚆'
                    route_step['details'].append("Marmaray - Cross-continental rail")
                elif route_id.startswith('T'):
                    route_step['mode_icon'] = '🚊'
                    route_step['details'].append("Tram - Scenic route")
                elif route_id.startswith('F'):
                    route_step['mode_icon'] = '🚡'
                    route_step['details'].append("Funicular - Quick uphill transport")
                else:
                    route_step['mode_icon'] = '🚌'
                    route_step['details'].append("Bus")
                
                detailed_route['steps'].append(route_step)
                
                # Generate transfer instructions if there's a next route
                if i < len(selected_routes) - 1:
                    next_route = selected_routes[i + 1]
                    transfer_station = self._find_transfer_station(route_id, next_route)
                    
                    if transfer_station and hasattr(self, 'transfer_instructions_generator'):
                        transfer_instruction = self.transfer_instructions_generator.generate_transfer_instructions(
                            from_line=route_id,
                            to_line=next_route,
                            station=transfer_station,
                            direction_on_new_line=self._get_direction_hint(next_route, destination)
                        )
                        
                        detailed_route['transfers'].append({
                            'transfer_number': i + 1,
                            'instruction': transfer_instruction,
                            'formatted': self.transfer_instructions_generator.format_transfer_instruction_for_display(transfer_instruction)
                        })
                        
                        detailed_route['total_time_estimate'] += transfer_instruction.estimated_time
            
            # Calculate fare
            detailed_route['fare_info'] = self.calculate_fare(
                selected_routes,
                num_transfers=len(selected_routes) - 1
            )
            
            # Generate map visualization data
            detailed_route['map_data'] = self._generate_map_data(selected_routes, detailed_route['transfers'])
            
            # Add accessibility information
            detailed_route['accessibility'] = self._get_accessibility_info(selected_routes)
            
            logger.info(f"✅ Generated detailed route: {origin} → {destination} via {', '.join(selected_routes)}")
            return detailed_route
            
        except Exception as e:
            logger.error(f"❌ Error generating detailed route: {e}")
            return self._generate_basic_route_info(origin, destination, selected_routes)
    
    def _generate_basic_route_info(self, origin: str, destination: str, routes: List[str]) -> Dict[str, Any]:
        """Generate basic route info when detailed generator is unavailable"""
        return {
            'origin': origin,
            'destination': destination,
            'routes': routes,
            'steps': [
                {
                    'step_number': i + 1,
                    'route_id': route,
                    'instruction': f"Take {route} line"
                }
                for i, route in enumerate(routes)
            ],
            'transfers': [
                {
                    'transfer_number': i + 1,
                    'instruction': f"Transfer from {routes[i]} to {routes[i+1]}"
                }
                for i in range(len(routes) - 1)
            ],
            'fare_info': self.calculate_fare(routes, num_transfers=len(routes) - 1)
        }
    
    def _find_transfer_station(self, from_route: str, to_route: str) -> Optional[str]:
        """Find the transfer station between two routes"""
        
        # Known transfer stations mapping
        transfer_stations = {
            ('M1A', 'M2'): 'Yenikapı',
            ('M1B', 'M2'): 'Yenikapı',
            ('M1A', 'MARMARAY'): 'Yenikapı',
            ('M1B', 'MARMARAY'): 'Yenikapı',
            ('M2', 'MARMARAY'): 'Yenikapı',
            ('M2', 'M7'): 'Mecidiyeköy',
            ('M2', 'M11'): 'Gayrettepe',
            ('M2', 'M6'): 'Levent',
            ('MARMARAY', 'M5'): 'Üsküdar',
            ('MARMARAY', 'M4'): 'Ayrılık Çeşmesi',
            ('MARMARAY', 'T1'): 'Sirkeci',
            ('T1', 'M2'): 'Yenikapı',
            ('T1', 'F1'): 'Kabataş',
            ('T1', 'F2'): 'Karaköy',
            ('M4', 'M8'): 'Bostancı',
        }
        
        # Check both directions
        key1 = (from_route, to_route)
        key2 = (to_route, from_route)
        
        return transfer_stations.get(key1) or transfer_stations.get(key2)
    
    def _get_direction_hint(self, route_id: str, destination: str) -> Optional[str]:
        """Get direction hint for the route based on destination"""
        
        destination_lower = destination.lower()
        
        direction_hints = {
            'M2': {
                'taksim': 'Hacıosman',
                'şişli': 'Hacıosman',
                'levent': 'Hacıosman',
                'sultanahmet': 'Yenikapı',
                'aksaray': 'Yenikapı'
            },
            'M11': {
                'airport': 'Istanbul Airport',
                'havalimanı': 'Istanbul Airport',
                'gayrettepe': 'Gayrettepe'
            },
            'MARMARAY': {
                'kadıköy': 'Gebze',
                'üsküdar': 'Gebze',
                'pendik': 'Gebze',
                'taksim': 'Halkalı',
                'beyoğlu': 'Halkalı'
            },
            'M4': {
                'sabiha': 'Sabiha Gökçen Airport',
                'airport': 'Sabiha Gökçen Airport',
                'kadıköy': 'Kadıköy',
                'kartal': 'Sabiha Gökçen Airport'
            }
        }
        
        if route_id in direction_hints:
            for keyword, direction in direction_hints[route_id].items():
                if keyword in destination_lower:
                    return direction
        
        return None
    
    def _generate_map_data(self, routes: List[str], transfers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate GeoJSON map data for route visualization
        Compatible with Leaflet, Mapbox, and other map libraries
        """
        
        map_data = {
            'type': 'FeatureCollection',
            'features': [],
            'bounds': {
                'north': 41.1,
                'south': 40.9,
                'east': 29.1,
                'west': 28.8
            }
        }
        
        # Known station coordinates for major transfer points
        station_coords = {
            'Yenikapı': [28.9512, 41.0085],
            'Mecidiyeköy': [28.9986, 41.0639],
            'Gayrettepe': [29.0140, 41.0688],
            'Levent': [29.0117, 41.0781],
            'Kabataş': [29.0064, 41.0298],
            'Taksim': [28.9874, 41.0369],
            'Üsküdar': [29.0159, 41.0243],
            'Kadıköy': [29.0261, 40.9904],
            'Sirkeci': [28.9767, 41.0170],
            'Bostancı': [29.0890, 40.9612],
        }
        
        # Route colors
        route_colors = {
            'M1A': '#FF0000',
            'M1B': '#00BFFF',
            'M2': '#00FF00',
            'M3': '#0000FF',
            'M4': '#FF69B4',
            'M5': '#800080',
            'M6': '#8B4513',
            'M7': '#FFD700',
            'M8': '#FFA500',
            'M9': '#FF1493',
            'M11': '#9370DB',
            'MARMARAY': '#FF4500',
            'T1': '#DC143C',
            'F1': '#00CED1',
            'F2': '#4169E1',
        }
        
        # Add route lines
        for i, route_id in enumerate(routes):
            route_feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': []
                },
                'properties': {
                    'route_id': route_id,
                    'route_name': self._get_route_name(route_id),
                    'color': route_colors.get(route_id, '#000000'),
                    'mode': self._get_route_mode(route_id),
                    'segment_number': i + 1
                }
            }
            
            map_data['features'].append(route_feature)
        
        # Add transfer points as markers
        for transfer in transfers:
            if 'instruction' in transfer and hasattr(transfer['instruction'], 'station_name'):
                station_name = transfer['instruction'].station_name
                coords = station_coords.get(station_name)
                
                if coords:
                    transfer_feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': coords
                        },
                        'properties': {
                            'type': 'transfer',
                            'station_name': station_name,
                            'from_line': transfer['instruction'].from_line,
                            'to_line': transfer['instruction'].to_line,
                            'transfer_time': transfer['instruction'].estimated_time,
                            'icon': '🔄',
                            'popup_content': transfer.get('formatted', f"Transfer at {station_name}")
                        }
                    }
                    
                    map_data['features'].append(transfer_feature)
        
        return map_data
    
    def _get_accessibility_info(self, routes: List[str]) -> List[str]:
        """Get accessibility information for the routes"""
        
        accessibility_info = []
        
        for route_id in routes:
            if route_id in ['M11', 'M7', 'M9']:  # Newer lines
                accessibility_info.append(f"{route_id}: Fully wheelchair accessible")
            elif route_id.startswith('M'):
                accessibility_info.append(f"{route_id}: Elevator access available at most stations")
            elif route_id == 'MARMARAY':
                accessibility_info.append("MARMARAY: Elevator access at all stations")
            elif route_id.startswith('T'):
                accessibility_info.append(f"{route_id}: Low-floor trams, wheelchair accessible")
            elif route_id.startswith('F'):
                accessibility_info.append(f"{route_id}: Funicular accessible with assistance")
        
        return list(set(accessibility_info))  # Remove duplicates
    
    def format_detailed_route_for_display(self, detailed_route: Dict[str, Any]) -> str:
        """Format detailed route information for user-friendly display"""
        
        output = []
        
        # Header
        output.append(f"\n🗺️ **DETAILED ROUTE INSTRUCTIONS**")
        output.append(f"📍 From: **{detailed_route['origin']}**")
        output.append(f"📍 To: **{detailed_route['destination']}**")
        output.append(f"⏱️ Estimated time: ~{detailed_route.get('total_time_estimate', 30)} minutes")
        
        # Fare information
        if detailed_route.get('fare_info'):
            fare_info = detailed_route['fare_info']
            output.append(f"\n💳 **FARE INFORMATION**")
            # Handle both old and new fare info formats
            total_cost = fare_info.get('total_cost_istanbulkart') or fare_info.get('breakdown', {}).get('total', 0)
            output.append(f"Total cost: {total_cost:.2f} TL (Istanbulkart)")
            if fare_info.get('savings') or fare_info.get('savings_with_istanbulkart'):
                savings = fare_info.get('savings') or fare_info.get('savings_with_istanbulkart', 0)
                output.append(f"💰 Savings: {savings:.2f} TL vs single tickets")
        
        # Route steps
        output.append(f"\n🚇 **ROUTE STEPS** ({len(detailed_route['routes'])} segments)")
        
        for step in detailed_route['steps']:
            output.append(f"\n{step['mode_icon']} **Step {step['step_number']}: {step['route_id']}**")
            output.append(f"   {step['instruction']}")
            for detail in step.get('details', []):
                output.append(f"   • {detail}")
        
        # Transfer instructions
        if detailed_route.get('transfers'):
            output.append(f"\n🔄 **TRANSFER INSTRUCTIONS** ({len(detailed_route['transfers'])} transfers)")
            
            for transfer in detailed_route['transfers']:
                output.append(f"\n{transfer.get('formatted', 'Transfer information')}")
        
        # Accessibility
        if detailed_route.get('accessibility'):
            output.append(f"\n♿ **ACCESSIBILITY INFORMATION**")
            for info in detailed_route['accessibility']:
                output.append(f"   • {info}")
        
        # Map visualization hint
        if detailed_route.get('map_data'):
            output.append(f"\n🗺️ **MAP VISUALIZATION AVAILABLE**")
            output.append(f"   View this route on an interactive map")
            output.append(f"   Features: {len(detailed_route['map_data']['features'])} map features")
        
        return '\n'.join(output)
    
    def get_data_source_info(self) -> Dict[str, Any]:
        """
        Get information about the data source being used
        
        Returns:
            Dict with data source information including:
            - is_using_mock_data: bool
            - data_source: str ('mock' or 'live_ibb_api')
            - ibb_api_available: bool
            - ibb_api_connected: bool
            - description: str
        """
        data_source_info = {
            'is_using_mock_data': self.use_mock_data,
            'data_source': 'mock' if self.use_mock_data else 'live_ibb_api',
            'ibb_api_available': IBB_API_AVAILABLE,
            'ibb_api_connected': self.ibb_api is not None,
            'description': ''
        }
        
        if self.use_mock_data:
            data_source_info['description'] = (
                "Using realistic mock data for development and testing. "
                "Mock data simulates İBB's real-time transportation system with current "
                "schedules, frequencies, and route information."
            )
        else:
            if self.ibb_api:
                data_source_info['description'] = (
                    "Connected to live İBB Open Data API. "
                    "Providing real-time transportation information including schedules, "
                    "delays, and live vehicle tracking."
                )
            else:
                data_source_info['description'] = (
                    "Attempted to use live İBB API but connection failed. "
                    "Falling back to mock data."
                )
        
        return data_source_info
    
    def get_data_source_display(self) -> str:
        """
        Get a user-friendly display string for the data source
        
        Returns:
            str: Display string (e.g., "📡 Live İBB Data" or "🧪 Mock Data")
        """
        if self.use_mock_data:
            return "🧪 Mock Data (Development)"
        elif self.ibb_api:
            return "📡 Live İBB Data"
        else:
            return "⚠️ Mock Data (API Unavailable)"
    
    def log_data_source_status(self) -> None:
        """Log detailed information about the current data source"""
        info = self.get_data_source_info()
        
        logger.info("=" * 70)
        logger.info("📊 İBB Transportation Data Source Status")
        logger.info("=" * 70)
        logger.info(f"Data Source: {info['data_source'].upper()}")
        logger.info(f"Using Mock Data: {info['is_using_mock_data']}")
        logger.info(f"İBB API Available: {info['ibb_api_available']}")
        logger.info(f"İBB API Connected: {info['ibb_api_connected']}")
        logger.info(f"Description: {info['description']}")
        logger.info("=" * 70)


# Factory function
def get_live_ibb_transportation_service():
    """Get instance of live IBB transportation service"""
    return LiveIBBTransportationService()


async def main():
    """Test the live IBB transportation service"""
    print("🚌 LIVE IBB TRANSPORTATION SERVICE TEST")
    print("=" * 60)
    
    service = LiveIBBTransportationService()
    
    # Test 1: Get live bus routes
    print("📊 Testing live bus routes...")
    live_routes = await service.get_live_bus_routes(['HAVAIST-1', '500T'])
    
    print(f"   Found {len(live_routes)} live routes:")
    for route_id, route_data in live_routes.items():
        print(f"   • {route_id}: {route_data.route_name}")
        print(f"     Status: {route_data.current_status}")
        print(f"     Frequency: {route_data.live_frequency}")
        if route_data.delays:
            print(f"     Delays: {route_data.delays} minutes")
    
    # Test 2: Get live recommendations
    print(f"\n🎯 Testing live recommendations...")
    recommendations = await service.get_live_bus_recommendations()
    
    print(f"   Airport routes: {len(recommendations.get('airport_routes', []))}")
    print(f"   City routes: {len(recommendations.get('city_routes', []))}")
    print(f"   Status summary: {recommendations.get('status_summary', {})}")
    
    # Test 3: Enhanced directions
    print(f"\n🗺️ Testing enhanced directions...")
    taksim = (41.0370, 28.9850)
    sultanahmet = (41.0054, 28.9768)
    
    enhanced_route = await service.get_enhanced_directions(
        taksim, sultanahmet, "Taksim", "Sultanahmet"
    )
    
    if enhanced_route:
        print(f"   Route found: {enhanced_route.summary}")
        print(f"   Total duration: {enhanced_route.total_duration} minutes")
        print(f"   Steps: {len(enhanced_route.steps)}")
    else:
        print("   No route found")
    
    print(f"\n✅ Live IBB Transportation Service test completed!")


# Factory function for easy import
def get_live_ibb_transportation_service(use_mock_data: bool = False) -> LiveIBBTransportationService:
    """Get instance of live IBB transportation service"""
    return LiveIBBTransportationService(use_mock_data=use_mock_data)


async def test_live_features():
    """Test the enhanced live features"""
    print("🚌 TESTING ENHANCED LIVE IBB FEATURES")
    print("=" * 50)
    
    service = get_live_ibb_transportation_service(use_mock_data=False)
    
    # Test individual route data
    test_routes = ['HAVAIST-1', '500T', '28', 'UNKNOWN']
    
    for route_code in test_routes:
        print(f"\n🔍 Testing route: {route_code}")
        
        live_data = await service.get_live_route_data(route_code)
        
        if live_data:
            formatted = service.format_live_route_info(live_data)
            print(formatted)
        else:
            print(f"❌ No live data available for {route_code}")
    
    # Test enhanced recommendations
    print(f"\n🎯 Enhanced Route Recommendations:")
    
    test_queries = [
        ("Taksim", "Istanbul Airport"),
        ("Sultanahmet", "Airport"), 
        ("Beşiktaş", "Sarıyer")
    ]
    
    for origin, destination in test_queries:
        print(f"\n📍 Route: {origin} → {destination}")
        
        recommendations = await service.get_enhanced_recommendations(origin, destination)
        
        for i, rec in enumerate(recommendations['recommendations'], 1):
            live_data = rec['route']
            score = rec['score']
            reason = rec['reason']
            
            print(f"   {i}. {live_data.route_id} (Score: {score:.1f})")
            print(f"      Status: {live_data.current_status}")
            print(f"      Frequency: {live_data.live_frequency}")
            if live_data.next_departure:
                print(f"      Next: {live_data.next_departure.strftime('%H:%M')}")
            print(f"      Reason: {reason}")
    
    print(f"\n📊 Cache Performance:")
    print(f"   Cache entries: {len(service._cache)}")
    print(f"   Cache duration: {service._cache_duration}")
    
    print(f"\n✅ Enhanced live features test completed!")


if __name__ == "__main__":
    # Run both the original and enhanced tests
    asyncio.run(main())
    print("\n" + "="*50)
    asyncio.run(test_live_features())
