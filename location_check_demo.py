#!/usr/bin/env python3
"""
Istanbul AI - Location Check Demo
Demonstrates how the system checks and works with user location
"""

from istanbul_daily_talk_system import IstanbulDailyTalkAI
import json

def main():
    # Initialize the Istanbul AI system
    print("🏛️ Initializing Istanbul Daily Talk AI...")
    system = IstanbulDailyTalkAI()
    print("✅ System ready!")
    print()
    
    print("🗺️ LOCATION CHECK DEMO")
    print("=" * 50)
    print()
    
    # Demo user
    user_id = "demo_user"
    
    print("📱 AVAILABLE LOCATION METHODS:")
    print("1. 🌐 Web Browser GPS (HTML5 Geolocation)")
    print("2. 📱 Mobile App GPS (React Native)")
    print("3. ✋ Manual Input (Tell me where you are)")
    print("4. 📍 Simulate GPS coordinates")
    print()
    
    while True:
        print("💬 CHOOSE AN OPTION:")
        print("A. Check location without GPS")
        print("B. Set GPS location and check")
        print("C. Manual location input")
        print("D. Show location collection methods")
        print("Q. Quit")
        print()
        
        choice = input("Your choice (A/B/C/D/Q): ").upper().strip()
        print()
        
        if choice == 'Q':
            print("👋 Goodbye! Thanks for testing the location system!")
            break
            
        elif choice == 'A':
            print("📍 CHECKING LOCATION WITHOUT GPS...")
            print("-" * 30)
            
            # Process location query without GPS
            response = system.process_message("Where am I?", user_id)
            print("🤖 AI Response:")
            print(response)
            print()
            
        elif choice == 'B':
            print("📍 SETTING GPS LOCATION...")
            print("-" * 30)
            
            # Show some Istanbul locations to choose from
            locations = {
                "1": {"name": "Taksim Square", "lat": 41.0367, "lng": 28.9850},
                "2": {"name": "Sultanahmet (Hagia Sophia)", "lat": 41.0086, "lng": 28.9802},
                "3": {"name": "Galata Tower", "lat": 41.0256, "lng": 28.9744},
                "4": {"name": "Kadıköy Ferry Terminal", "lat": 40.9928, "lng": 29.0253},
                "5": {"name": "Beşiktaş", "lat": 41.0422, "lng": 29.0069}
            }
            
            print("📍 Choose a location:")
            for key, loc in locations.items():
                print(f"{key}. {loc['name']}")
            
            loc_choice = input("\nEnter location number (1-5): ").strip()
            
            if loc_choice in locations:
                location = locations[loc_choice]
                
                # Set GPS location
                gps_data = {
                    'latitude': location['lat'],
                    'longitude': location['lng'],
                    'accuracy': 5,
                    'method': 'simulated_gps'
                }
                
                result = system.update_user_location(user_id, gps_data)
                
                print(f"\n✅ GPS Location Set: {location['name']}")
                print(f"📍 Coordinates: ({location['lat']:.4f}, {location['lng']:.4f})")
                print(f"🏛️ In Istanbul: {result['is_in_istanbul']}")
                
                if result['location_info']:
                    info = result['location_info']
                    if info.get('neighborhood'):
                        print(f"🏘️ Neighborhood: {info['neighborhood'].title()}")
                    if info.get('district'):
                        print(f"🏛️ District: {info['district']}")
                
                # Now check location with GPS
                print("\n🤖 AI Response with GPS location:")
                response = system.process_message("Where am I now?", user_id)
                print(response)
                print()
            else:
                print("❌ Invalid choice!")
                print()
                
        elif choice == 'C':
            print("✋ MANUAL LOCATION INPUT...")
            print("-" * 30)
            
            manual_input = system.collect_user_gps_location(user_id, 'manual_input')
            print("📍 Manual Location Prompt:")
            print(manual_input['prompt'])
            
            # Let user enter a location manually
            user_location = input("\n👤 Tell me where you are: ").strip()
            
            if user_location:
                print(f"\n📍 Processing: '{user_location}'")
                
                # Check if it's a recognized neighborhood
                neighborhood = system._extract_neighborhood_from_message(user_location)
                if neighborhood:
                    print(f"✅ Recognized neighborhood: {neighborhood.title()}")
                
                # Check if it's a landmark
                landmark = system._extract_landmark_with_coordinates(user_location)
                if landmark:
                    print(f"✅ Recognized landmark: {landmark['name']}")
                    print(f"📍 Coordinates: ({landmark['coordinates']['latitude']:.4f}, {landmark['coordinates']['longitude']:.4f})")
                
                # Generate response
                response = system.process_message(user_id, f"I am at {user_location}")
                print("\n🤖 AI Response:")
                print(response)
                print()
            else:
                print("❌ No location entered!")
                print()
                
        elif choice == 'D':
            print("📱 LOCATION COLLECTION METHODS...")
            print("-" * 30)
            
            methods = ['web_browser', 'mobile_app', 'manual_input', 'request']
            
            for method in methods:
                result = system.collect_user_gps_location(f"demo_{method}", method)
                
                print(f"\n🔧 {method.replace('_', ' ').title()}:")
                if method == 'web_browser':
                    print("   ✅ HTML5 Geolocation JavaScript generated")
                    print("   📱 Works in all modern browsers")
                elif method == 'mobile_app':
                    print("   ✅ React Native GPS code generated")
                    print("   📱 Native iOS/Android integration")
                elif method == 'manual_input':
                    print("   ✅ User-friendly input prompts")
                    print("   🗺️ Accepts landmarks, neighborhoods, addresses")
                elif method == 'request':
                    print("   ✅ Permission request dialog")
                    print("   🔐 Privacy-focused approach")
            
            print()
            
        else:
            print("❌ Invalid choice! Please try again.")
            print()

if __name__ == "__main__":
    main()
