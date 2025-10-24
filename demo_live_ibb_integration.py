#!/usr/bin/env python3
"""
Enhanced Transportation Service with Live IBB Data Integration
=============================================================

Shows how to integrate real İBB Open Data for live bus information.
This is the approach we should use instead of static route data.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class LiveIBBDataProvider:
    """Provider for live İBB transportation data"""
    
    def __init__(self):
        """Initialize live data provider"""
        self.api_available = False
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Simulate API availability check
        try:
            # In real implementation, this would test IBB API connectivity
            self.api_available = self._test_ibb_api_connection()
            if self.api_available:
                logger.info("✅ Live İBB API connection established")
            else:
                logger.info("⚠️ Using simulated live data (API not configured)")
        except Exception as e:
            logger.warning(f"⚠️ IBB API not available: {e}")
    
    def _test_ibb_api_connection(self) -> bool:
        """Test if IBB API is accessible"""
        # In real implementation, this would make a test API call
        # For demo purposes, return False to show fallback behavior
        return False
    
    async def get_live_bus_data(self, route_codes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get live bus data from İBB API or simulated data"""
        
        if self.api_available:
            return await self._fetch_live_ibb_data(route_codes)
        else:
            return self._get_simulated_live_data(route_codes)
    
    async def _fetch_live_ibb_data(self, route_codes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fetch real live data from İBB Open Data Portal
        
        This would use actual IBB API endpoints:
        - https://data.ibb.gov.tr/api/3/action/datastore_search
        - Dataset IDs for bus routes, schedules, real-time positions
        """
        
        # Example of what real API integration would look like:
        live_data = {
            'timestamp': datetime.now().isoformat(),
            'source': 'ibb_open_data_portal',
            'routes': {},
            'service_alerts': [],
            'system_status': 'operational'
        }
        
        # Real implementation would:
        # 1. Query IBB bus routes dataset
        # 2. Get real-time vehicle positions
        # 3. Fetch service alerts and disruptions
        # 4. Calculate live frequencies and delays
        
        api_routes = {
            'HAVAIST-1': {
                'name': 'Havaist IST-1 Taksim',
                'status': 'operational',
                'live_frequency': '25 minutes',  # From real-time calculation
                'next_departures': ['14:45', '15:10', '15:35'],
                'current_delays': 5,  # 5 minutes delayed
                'vehicle_count': 12,
                'stops': [
                    {'name': 'Istanbul Airport Terminal', 'arrival_estimate': '14:40'},
                    {'name': 'Mecidiyeköy', 'arrival_estimate': '15:25'},
                    {'name': 'Taksim Square', 'arrival_estimate': '15:45'}
                ],
                'last_position': {'lat': 41.1500, 'lng': 28.8500},
                'api_updated': datetime.now().isoformat()
            },
            '500T': {
                'name': '500T Taksim - Sarıyer',
                'status': 'operational',
                'live_frequency': '8 minutes',  # Calculated from vehicle positions
                'next_departures': ['14:42', '14:50', '14:58'],
                'current_delays': 0,
                'vehicle_count': 15,
                'stops': [
                    {'name': 'Taksim Square', 'arrival_estimate': '14:42'},
                    {'name': 'Beşiktaş', 'arrival_estimate': '14:52'},
                    {'name': 'Ortaköy', 'arrival_estimate': '15:05'}
                ],
                'last_position': {'lat': 41.0400, 'lng': 28.9900},
                'api_updated': datetime.now().isoformat()
            }
        }
        
        # Filter by requested routes
        if route_codes:
            live_data['routes'] = {k: v for k, v in api_routes.items() if k in route_codes}
        else:
            live_data['routes'] = api_routes
        
        return live_data
    
    def _get_simulated_live_data(self, route_codes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Simulated live data showing what real IBB integration would provide"""
        
        current_time = datetime.now()
        
        simulated_data = {
            'timestamp': current_time.isoformat(),
            'source': 'simulated_live_data',
            'routes': {},
            'service_alerts': [
                {
                    'route': 'M2',
                    'type': 'delay',
                    'message': 'Minor delays due to signal problems at Taksim station',
                    'severity': 'low',
                    'start_time': (current_time - timedelta(minutes=30)).isoformat()
                }
            ],
            'system_status': 'operational_with_delays'
        }
        
        # Simulate live route data with realistic variations
        simulated_routes = {
            'HAVAIST-1': {
                'name': 'Havaist IST-1 Taksim',
                'status': 'operational',
                'live_frequency': '32 minutes',  # Slightly different from scheduled 30
                'next_departures': [
                    (current_time + timedelta(minutes=8)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=40)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=72)).strftime('%H:%M')
                ],
                'current_delays': 2,  # 2 minutes delayed from schedule
                'vehicle_count': 8,   # Live vehicle count
                'occupancy': 'medium',  # Live occupancy data
                'last_updated': current_time.isoformat(),
                'price': '18 TL (cash only)',
                'payment_methods': ['cash'],
                'accessibility': 'wheelchair_accessible'
            },
            '500T': {
                'name': '500T Taksim - Sarıyer',
                'status': 'operational',
                'live_frequency': '12 minutes',  # Live calculation from vehicle tracking
                'next_departures': [
                    (current_time + timedelta(minutes=3)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=15)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=27)).strftime('%H:%M')
                ],
                'current_delays': 0,
                'vehicle_count': 12,
                'occupancy': 'high',  # Rush hour
                'last_updated': current_time.isoformat(),
                'price': '13.5 TL (Istanbulkart)',
                'payment_methods': ['istanbulkart', 'contactless'],
                'accessibility': 'wheelchair_accessible'
            },
            'E-2': {
                'name': 'E-2 Sabiha Gökçen - Kadıköy',
                'status': 'operational',
                'live_frequency': '18 minutes',  # Real frequency varies
                'next_departures': [
                    (current_time + timedelta(minutes=6)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=24)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=42)).strftime('%H:%M')
                ],
                'current_delays': 3,  # Airport routes often have delays
                'vehicle_count': 6,
                'occupancy': 'low',
                'last_updated': current_time.isoformat(),
                'price': '13.5 TL (Istanbulkart)',
                'payment_methods': ['istanbulkart', 'contactless'],
                'accessibility': 'wheelchair_accessible'
            },
            '28': {
                'name': '28 Beşiktaş - Edirnekapı',
                'status': 'operational',
                'live_frequency': '9 minutes',  # Frequent city route
                'next_departures': [
                    (current_time + timedelta(minutes=2)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=11)).strftime('%H:%M'),
                    (current_time + timedelta(minutes=20)).strftime('%H:%M')
                ],
                'current_delays': 1,
                'vehicle_count': 18,
                'occupancy': 'high',
                'last_updated': current_time.isoformat(),
                'price': '13.5 TL (Istanbulkart)',
                'payment_methods': ['istanbulkart', 'contactless'],
                'accessibility': 'wheelchair_accessible'
            }
        }
        
        # Filter by requested routes
        if route_codes:
            simulated_data['routes'] = {k: v for k, v in simulated_routes.items() if k in route_codes}
        else:
            simulated_data['routes'] = simulated_routes
        
        return simulated_data
    
    def format_live_route_info(self, route_data: Dict[str, Any]) -> str:
        """Format live route information for display"""
        
        route_info = []
        route_info.append(f"🚌 **{route_data['name']}**")
        route_info.append(f"📍 Status: {route_data['status'].replace('_', ' ').title()}")
        route_info.append(f"🔄 Live frequency: {route_data['live_frequency']}")
        
        # Show delays if any
        if route_data.get('current_delays', 0) > 0:
            route_info.append(f"⏰ Current delays: {route_data['current_delays']} minutes")
        
        # Next departures
        if route_data.get('next_departures'):
            departures = ', '.join(route_data['next_departures'][:3])
            route_info.append(f"🕐 Next departures: {departures}")
        
        # Live vehicle info
        if route_data.get('vehicle_count'):
            route_info.append(f"🚐 Active vehicles: {route_data['vehicle_count']}")
        
        if route_data.get('occupancy'):
            occupancy_emoji = {
                'low': '🟢',
                'medium': '🟡', 
                'high': '🔴'
            }.get(route_data['occupancy'], '⚪')
            route_info.append(f"👥 Occupancy: {occupancy_emoji} {route_data['occupancy'].title()}")
        
        # Payment and accessibility
        route_info.append(f"💰 Price: {route_data.get('price', 'Standard fare')}")
        
        if route_data.get('accessibility') == 'wheelchair_accessible':
            route_info.append("♿ Wheelchair accessible")
        
        route_info.append(f"🕐 Last updated: {route_data.get('last_updated', 'Unknown')}")
        
        return '\n'.join(route_info)


async def demonstrate_live_vs_static():
    """Demonstrate the difference between live and static data"""
    
    print("🚌 LIVE İBB DATA vs STATIC DATA COMPARISON")
    print("=" * 60)
    
    provider = LiveIBBDataProvider()
    
    # Get live data for major routes
    print("📊 Fetching live bus data...")
    live_data = await provider.get_live_bus_data(['HAVAIST-1', '500T', 'E-2'])
    
    print(f"\n🌟 **LIVE DATA ADVANTAGES:**")
    print(f"✅ Real-time frequencies (not scheduled)")
    print(f"✅ Current delays and disruptions")
    print(f"✅ Live vehicle counts and positions")
    print(f"✅ Real occupancy levels")
    print(f"✅ Service alerts and status updates")
    print(f"✅ Accurate next departure times")
    
    print(f"\n📱 **EXAMPLE LIVE ROUTE DATA:**")
    print(f"Source: {live_data['source']}")
    print(f"Timestamp: {live_data['timestamp']}")
    print(f"System status: {live_data['system_status']}")
    
    # Show service alerts
    if live_data.get('service_alerts'):
        print(f"\n🚨 **LIVE SERVICE ALERTS:**")
        for alert in live_data['service_alerts']:
            print(f"   • {alert['route']}: {alert['message']}")
    
    # Show detailed route information
    for route_id, route_data in live_data['routes'].items():
        print(f"\n" + provider.format_live_route_info(route_data))
    
    print(f"\n" + "="*60)
    print(f"🎯 **WHY USE LIVE İBB DATA:**")
    print(f"1. 📍 **Accuracy**: Real frequencies vs estimated schedules")
    print(f"2. ⚡ **Real-time**: Current delays, not historical averages")
    print(f"3. 🚌 **Live tracking**: Actual vehicle positions and counts") 
    print(f"4. 👥 **Occupancy**: Help users choose less crowded buses")
    print(f"5. 🚨 **Alerts**: Immediate notification of service disruptions")
    print(f"6. 💰 **Payment**: Current fare information and payment methods")
    print(f"7. ♿ **Accessibility**: Real-time accessibility status")
    
    print(f"\n🔗 **INTEGRATION POINTS:**")
    print(f"• İBB Open Data Portal: https://data.ibb.gov.tr/")
    print(f"• Real-time bus positions API")
    print(f"• Service alerts and disruptions feed")
    print(f"• Live schedule and frequency calculations")
    print(f"• Vehicle occupancy sensors data")


async def main():
    """Main demonstration"""
    await demonstrate_live_vs_static()


if __name__ == "__main__":
    asyncio.run(main())
