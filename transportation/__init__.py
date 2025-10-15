"""
Istanbul Transportation System
=============================

A comprehensive, modular transportation system for Istanbul with:
- Route planning and navigation
- Real-time schedule integration  
- GPS location processing
- Multi-modal journey suggestions
- Accessibility information

Main Components:
- models: Data structures (MetroStation, WalkingRoute, GPSLocation)
- services: Core transportation service
- processors: Query processors for different functionality
"""

from .models import MetroStation, WalkingRoute, GPSLocation
from .services import (
    EnhancedTransportationSystem,
    RealTimeScheduleService, 
    MultiModalJourneyPlanner
)
from .processors import (
    ComprehensiveTransportProcessor,
    GPSLocationProcessor,
    GPSTransportationQueryProcessor
)

__version__ = "2.0.0"
__all__ = [
    # Models
    'MetroStation',
    'WalkingRoute', 
    'GPSLocation',
    
    # Services
    'EnhancedTransportationSystem',
    'RealTimeScheduleService',
    'MultiModalJourneyPlanner',
    
    # Processors
    'ComprehensiveTransportProcessor',
    'GPSLocationProcessor',
    'GPSTransportationQueryProcessor'
]


def create_transportation_system():
    """Factory function to create a fully configured transportation system"""
    return {
        'enhanced_system': EnhancedTransportationSystem(),
        'real_time_service': RealTimeScheduleService(),
        'journey_planner': MultiModalJourneyPlanner(),
        'comprehensive_processor': ComprehensiveTransportProcessor(),
        'gps_processor': GPSLocationProcessor(),
        'gps_query_processor': GPSTransportationQueryProcessor()
    }
