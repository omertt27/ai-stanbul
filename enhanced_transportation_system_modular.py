#!/usr/bin/env python3
"""
Istanbul Transportation System - Main Interface
==============================================

This is the main interface that maintains backward compatibility with the original
enhanced_transportation_system.py while using the new modular architecture.

For new code, import directly from the transportation package:
    from transportation import ComprehensiveTransportProcessor, GPSTransportationQueryProcessor
"""

# Backward compatibility imports
from transportation import (
    MetroStation,
    WalkingRoute, 
    GPSLocation,
    EnhancedTransportationSystem,
    ComprehensiveTransportProcessor,
    GPSLocationProcessor,
    GPSTransportationQueryProcessor,
    create_transportation_system
)

# Re-export everything for backward compatibility
__all__ = [
    'MetroStation',
    'WalkingRoute',
    'GPSLocation', 
    'EnhancedTransportationSystem',
    'ComprehensiveTransportProcessor',
    'GPSLocationProcessor',
    'GPSTransportationQueryProcessor',
    'create_transportation_system'
]

# Create default instances for backward compatibility
def get_default_system():
    """Get default transportation system instances"""
    return create_transportation_system()

# Legacy function for tests
def main():
    """Main function for testing the transportation system"""
    print("🚇 Istanbul Transportation System v2.0 - Modular Architecture")
    print("=" * 60)
    
    # Create system
    system = create_transportation_system()
    
    print(f"✅ Enhanced Transportation System: {type(system['enhanced_system']).__name__}")
    print(f"✅ Comprehensive Processor: {type(system['comprehensive_processor']).__name__}")
    print(f"✅ GPS Processor: {type(system['gps_processor']).__name__}")
    print(f"✅ GPS Query Processor: {type(system['gps_query_processor']).__name__}")
    
    print("\n📍 Testing station lookup...")
    enhanced_system = system['enhanced_system']
    taksim = enhanced_system.get_station_info('Taksim')
    if taksim:
        print(f"   Taksim Station: {taksim.name} on {taksim.line} line")
        print(f"   Coordinates: {taksim.coordinates}")
        print(f"   Accessibility: {'Yes' if taksim.accessibility else 'No'}")
    
    print("\n🗺️ Testing route planning...")
    route_info = enhanced_system.find_route('Taksim', 'Sultanahmet')
    if 'error' not in route_info:
        print(f"   Route from {route_info['origin'].name} to {route_info['destination'].name}")
        print(f"   Found {len(route_info['route_suggestions'])} route options")
    
    print("\n✨ System ready for use!")
    print("\n📚 Usage Examples:")
    print("   from enhanced_transportation_system_modular import ComprehensiveTransportProcessor")
    print("   processor = ComprehensiveTransportProcessor()")
    print("   response = await processor.process_transportation_query('How to get from Taksim to Sultanahmet')")


if __name__ == "__main__":
    main()
