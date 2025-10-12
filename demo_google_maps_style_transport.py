#!/usr/bin/env python3
"""
Google Maps-Style Transportation Demo for A/ISTANBUL Route Planner
Shows detailed, weather-aware transportation advice similar to Google Maps
"""

from dotenv import load_dotenv
load_dotenv()

from services.route_cache import get_transportation_advice_for_weather

class WeatherScenario:
    def __init__(self, name, condition, temp, wind_speed=5, rainfall=0):
        self.name = name
        self.condition = condition
        self.current_temp = temp
        self.wind_speed = wind_speed
        self.rainfall_1h = rainfall

def demo_transportation_advice():
    """Demo Google Maps-style transportation advice for different weather scenarios"""
    
    print("🧭 A/ISTANBUL ROUTE PLANNER - Google Maps Style Transportation")
    print("=" * 80)
    print("Detailed, step-by-step transportation advice with weather awareness")
    print("=" * 80)
    print()
    
    # Different weather scenarios
    scenarios = [
        WeatherScenario("Rainy Day", "Rain", 16, 12, 3.0),
        WeatherScenario("Hot Summer Day", "Clear", 34, 8, 0),
        WeatherScenario("Windy Autumn", "Partly Cloudy", 20, 32, 0),
        WeatherScenario("Perfect Spring", "Clear", 22, 6, 0),
        WeatherScenario("Cold Winter", "Overcast", 3, 15, 0)
    ]
    
    routes = [
        ("Taksim", "Sultanahmet"),
        ("Kadıköy", "Beyoğlu"),
        ("Beşiktaş", "Eminönü")
    ]
    
    for i, (scenario, (start, end)) in enumerate(zip(scenarios, routes)):
        print(f"📋 SCENARIO {i+1}: {scenario.name.upper()}")
        print(f"🌤️ Weather: {scenario.condition}, {scenario.current_temp}°C, Wind: {scenario.wind_speed}m/s")
        if scenario.rainfall_1h > 0:
            print(f"🌧️ Rainfall: {scenario.rainfall_1h}mm/hour")
        print(f"📍 Route: {start} → {end}")
        print("-" * 60)
        
        # Get detailed transportation advice
        advice = get_transportation_advice_for_weather(scenario, start, end)
        
        print(f"🌤️ {advice['weather_impact']}")
        print()
        
        # Show recommended route
        if advice['recommended_routes']:
            route = advice['recommended_routes'][0]
            print(f"🎯 RECOMMENDED: {route['route_name']}")
            print(f"⏱️ Duration: {route['duration']} | 💰 Cost: {route['cost']}")
            print(f"🌟 {route['weather_rating']}")
            print(f"🎭 Comfort Level: {route['comfort_level']}")
            print()
            
            print("📋 STEP-BY-STEP DIRECTIONS:")
            for j, step in enumerate(route['steps'], 1):
                instruction = step['instruction']
                duration = step['duration']
                print(f"{j}. {instruction} ({duration})")
                
                if 'distance' in step:
                    print(f"   📏 Distance: {step['distance']}")
                if 'weather_tip' in step:
                    print(f"   💡 Weather Tip: {step['weather_tip']}")
                if 'stations' in step:
                    print(f"   🚇 Stations: {step['stations']}")
            print()
        
        # Show alternative route if available
        if advice['alternative_routes']:
            alt_route = advice['alternative_routes'][0]
            print(f"🔄 ALTERNATIVE: {alt_route['route_name']}")
            print(f"⏱️ {alt_route['duration']} | 💰 {alt_route['cost']} | {alt_route.get('weather_rating', '')}")
            if 'scenic_value' in alt_route:
                print(f"🏞️ Scenic Value: {alt_route['scenic_value']}")
            print()
        
        # Show real-time alerts
        if advice['real_time_alerts']:
            print("⚠️ REAL-TIME ALERTS:")
            for alert in advice['real_time_alerts']:
                print(f"• {alert}")
            print()
        
        # Show cost breakdown for first scenario
        if i == 0:
            print("💰 TRANSPORTATION COSTS (Istanbulkart prices):")
            for transport, cost in advice['cost_breakdown'].items():
                print(f"• {transport.replace('_', ' ').title()}: {cost}")
            print()
        
        # Show accessibility info for first scenario
        if i == 0:
            print("♿ ACCESSIBILITY INFORMATION:")
            for info in advice['accessibility_info']:
                print(f"• {info}")
            print()
        
        print("=" * 80)
        print()
    
    print("🎉 DEMO COMPLETE!")
    print()
    print("✅ FEATURES DEMONSTRATED:")
    print("• Google Maps-style step-by-step directions")
    print("• Weather-aware route optimization")
    print("• Real-time alerts and warnings")
    print("• Multiple transportation options")
    print("• Cost breakdown and accessibility info")
    print("• Detailed walking instructions with weather tips")
    print("• Alternative routes for different preferences")
    print()
    print("🚀 A/ISTANBUL Route Planner is ready for production!")

if __name__ == "__main__":
    demo_transportation_advice()
