#!/usr/bin/env python3
"""
Quick Demo - Transfer Instructions & Map Visualization
======================================================

Run this script for a quick demonstration of all features.
"""

import asyncio
import json
from services.live_ibb_transportation_service import LiveIBBTransportationService
from services.transfer_instructions_generator import TransferInstructionsGenerator


async def main():
    print("=" * 80)
    print("  🗺️  ISTANBUL TRANSPORTATION - TRANSFER & MAP VISUALIZATION DEMO")
    print("=" * 80)
    
    service = LiveIBBTransportationService(use_mock_data=False)
    generator = TransferInstructionsGenerator()
    
    # Demo 1: Quick Transfer Instruction
    print("\n📍 DEMO 1: Transfer Instructions (Yenikapı)")
    print("-" * 80)
    
    instruction = generator.generate_transfer_instructions(
        from_line='M2',
        to_line='MARMARAY',
        station='Yenikapı',
        direction_on_new_line='Kadıköy'
    )
    
    print(generator.format_transfer_instruction_for_display(instruction))
    
    # Demo 2: Complete Route with Map
    print("\n" + "=" * 80)
    print("📍 DEMO 2: Complete Route - Taksim to Kadıköy")
    print("=" * 80)
    
    detailed_route = service.generate_detailed_route_with_transfers(
        origin='Taksim',
        destination='Kadıköy',
        selected_routes=['M2', 'MARMARAY']
    )
    
    print(service.format_detailed_route_for_display(detailed_route))
    
    # Save map data
    with open('quick_demo_map.geojson', 'w', encoding='utf-8') as f:
        json.dump(detailed_route['map_data'], f, indent=2, ensure_ascii=False)
    
    print("\n💾 Map data saved to: quick_demo_map.geojson")
    
    # Demo 3: Station Information
    print("\n" + "=" * 80)
    print("📍 DEMO 3: Transfer Station Information")
    print("=" * 80)
    
    stations = ['Yenikapı', 'Gayrettepe', 'Mecidiyeköy']
    
    for station_name in stations:
        station_info = generator.get_station_info(station_name)
        if station_info:
            print(f"\n🚇 {station_info.station_name}")
            print(f"   Lines: {', '.join(station_info.transfer_available)}")
            print(f"   Layout: {station_info.platform_layout}")
            print(f"   Accessibility: {', '.join(station_info.accessibility[:2])}")
    
    # Demo 4: Multiple Route Options
    print("\n" + "=" * 80)
    print("📍 DEMO 4: Route Recommendations")
    print("=" * 80)
    
    recommendations = await service.get_enhanced_recommendations(
        'Taksim',
        'Istanbul Airport'
    )
    
    print(f"\n🎯 Best Route to Airport:")
    if recommendations.get('best_route'):
        best = recommendations['best_route']
        print(f"   Route: {best['route'].route_id}")
        print(f"   Score: {best['score']:.1f}")
        print(f"   Reason: {best['reason']}")
    
    print(f"\n💡 Alternative Options:")
    for i, rec in enumerate(recommendations.get('recommendations', [])[:3], 1):
        print(f"   {i}. {rec['route'].route_id} - {rec['reason']}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE!")
    print("=" * 80)
    print("\n📋 What you just saw:")
    print("   ✓ Detailed transfer instructions with visual cues")
    print("   ✓ Complete route generation with map data")
    print("   ✓ Station information database")
    print("   ✓ Multiple route recommendations")
    print("\n🗺️ Next Steps:")
    print("   1. Open map_visualization_demo.html in your browser")
    print("   2. View quick_demo_map.geojson at https://geojson.io")
    print("   3. Run full tests: python3 test_transfer_and_map_visualization.py")
    print("\n🎉 All features are production-ready!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
