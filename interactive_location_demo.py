#!/usr/bin/env python3
"""
Istanbul AI - Interactive Location Check Demo
Comprehensive demonstration of GPS location collection and processing
"""

import sys
import json
from datetime import datetime
from istanbul_daily_talk_system import IstanbulDailyTalkAI

class LocationDemo:
    def __init__(self):
        print("🏛️ Initializing Istanbul Daily Talk AI System...")
        try:
            self.system = IstanbulDailyTalkAI()
            print("✅ System initialized successfully!")
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            sys.exit(1)
        
        self.demo_user_id = "location_demo_user"
        print()
    
    def display_banner(self):
        """Display the demo banner"""
        print("=" * 60)
        print("🗺️  ISTANBUL AI - INTERACTIVE LOCATION DEMO  🗺️")
        print("=" * 60)
        print("Discover how our AI system collects and uses your location")
        print("for personalized Istanbul recommendations!")
        print("=" * 60)
        print()
    
    def show_location_collection_methods(self):
        """Show all available location collection methods"""
        print("📍 LOCATION COLLECTION METHODS")
        print("-" * 40)
        print()
        
        methods = {
            'web_browser': {
                'title': '🌐 Web Browser GPS',
                'description': 'HTML5 Geolocation API for web apps',
                'features': ['High accuracy GPS', 'Works in all modern browsers', 'Real-time location updates']
            },
            'mobile_app': {
                'title': '📱 Mobile App GPS',
                'description': 'React Native GPS for iOS/Android',
                'features': ['Native device GPS', 'Background location tracking', 'Battery optimized']
            },
            'manual_input': {
                'title': '✋ Manual Location Input',
                'description': 'User-friendly text input fallback',
                'features': ['Landmark recognition', 'Neighborhood detection', 'Address parsing']
            },
            'request': {
                'title': '🔐 Permission Request',
                'description': 'Privacy-focused location request',
                'features': ['Clear privacy policy', 'User consent required', 'Graceful fallbacks']
            }
        }
        
        for method_id, info in methods.items():
            print(f"{info['title']}")
            print(f"   {info['description']}")
            for feature in info['features']:
                print(f"   • {feature}")
            print()
    
    def demonstrate_gps_collection(self):
        """Demonstrate GPS location collection"""
        print("🧪 GPS COLLECTION DEMONSTRATION")
        print("-" * 40)
        
        print("1. Web Browser JavaScript Code Generation:")
        web_result = self.system.collect_user_gps_location(self.demo_user_id, 'web_browser')
        print("   ✅ HTML5 Geolocation JavaScript generated")
        print("   📋 Ready for web page integration")
        print()
        
        print("2. Mobile App Code Generation:")
        mobile_result = self.system.collect_user_gps_location(self.demo_user_id, 'mobile_app')
        print("   ✅ React Native GPS code generated")
        print("   📱 Ready for iOS/Android deployment")
        print()
        
        print("3. Manual Input Prompt:")
        manual_result = self.system.collect_user_gps_location(self.demo_user_id, 'manual_input')
        print("   ✅ User-friendly location input ready")
        print()
        
        print("4. Permission Request Message:")
        permission_result = self.system.collect_user_gps_location(self.demo_user_id, 'request')
        print("   ✅ Privacy-compliant permission dialog ready")
        print()
    
    def test_location_scenarios(self):
        """Test different location input scenarios"""
        print("🌍 LOCATION DETECTION SCENARIOS")
        print("-" * 40)
        
        test_scenarios = [
            {
                'input': 'I am at Taksim Square',
                'type': 'Landmark Detection',
                'expected': 'Should detect Taksim neighborhood and landmark'
            },
            {
                'input': 'Near Blue Mosque',
                'type': 'Landmark Proximity',
                'expected': 'Should detect Sultanahmet area and Blue Mosque'
            },
            {
                'input': 'Galata Tower area',
                'type': 'Area Reference',
                'expected': 'Should detect Galata neighborhood'
            },
            {
                'input': 'Istiklal Caddesi 45',
                'type': 'Street Address',
                'expected': 'Should detect Beyoğlu area and street'
            },
            {
                'input': 'Kadıköy ferry terminal',
                'type': 'Transportation Hub',
                'expected': 'Should detect Kadıköy and transport connection'
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"{i}. {scenario['type']}")
            print(f"   Input: \"{scenario['input']}\"")
            
            # Test neighborhood detection
            neighborhood = self.system._extract_neighborhood_from_message(scenario['input'])
            if neighborhood:
                print(f"   ✅ Neighborhood: {neighborhood.title()}")
            
            # Test landmark detection
            landmark = self.system._extract_landmark_with_coordinates(scenario['input'])
            if landmark:
                print(f"   ✅ Landmark: {landmark['name']}")
                coords = landmark['coordinates']
                print(f"   📍 Coordinates: ({coords['latitude']:.4f}, {coords['longitude']:.4f})")
            
            print(f"   Expected: {scenario['expected']}")
            print()
    
    def simulate_gps_locations(self):
        """Simulate GPS location updates"""
        print("📡 GPS LOCATION SIMULATION")
        print("-" * 40)
        
        istanbul_locations = [
            {
                'name': 'Sultanahmet Historic Area',
                'lat': 41.0082,
                'lng': 28.9784,
                'description': 'Heart of Ottoman Istanbul'
            },
            {
                'name': 'Taksim Square',
                'lat': 41.0367,
                'lng': 28.9850,
                'description': 'Modern city center'
            },
            {
                'name': 'Galata Tower',
                'lat': 41.0256,
                'lng': 28.9744,
                'description': 'Medieval tower with panoramic views'
            },
            {
                'name': 'Kadıköy Waterfront',
                'lat': 40.9928,
                'lng': 29.0253,
                'description': 'Asian side cultural district'
            },
            {
                'name': 'Beşiktaş District',
                'lat': 41.0422,
                'lng': 29.0069,
                'description': 'Upscale neighborhood by the Bosphorus'
            }
        ]
        
        print("Choose a location to simulate GPS coordinates:")
        for i, location in enumerate(istanbul_locations, 1):
            print(f"{i}. {location['name']} - {location['description']}")
        
        try:
            choice = input("\nEnter location number (1-5) or 'skip': ").strip()
            
            if choice.lower() == 'skip':
                print("⏭️  Skipping GPS simulation...")
                return None
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(istanbul_locations):
                location = istanbul_locations[choice_idx]
                
                # Update GPS location
                gps_data = {
                    'latitude': location['lat'],
                    'longitude': location['lng'],
                    'accuracy': 5,
                    'method': 'simulated_gps'
                }
                
                result = self.system.update_user_location(self.demo_user_id, gps_data)
                
                print(f"\n✅ GPS Location Set: {location['name']}")
                print(f"📍 Coordinates: ({location['lat']:.4f}, {location['lng']:.4f})")
                print(f"🏛️ In Istanbul: {result['is_in_istanbul']}")
                
                if result.get('location_info'):
                    info = result['location_info']
                    if info.get('neighborhood'):
                        print(f"🏘️ Neighborhood: {info['neighborhood'].title()}")
                    if info.get('district'):
                        print(f"🏛️ District: {info['district']}")
                
                return location
            else:
                print("❌ Invalid choice!")
                return None
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input or cancelled!")
            return None
    
    def demonstrate_location_queries(self, location=None):
        """Demonstrate location-based queries"""
        print("\n💬 LOCATION-BASED QUERY DEMONSTRATION")
        print("-" * 40)
        
        queries = [
            "Where am I?",
            "Check my location",
            "What neighborhood am I in?",
            "Find restaurants near me",
            "How do I get to Blue Mosque?",
            "What attractions are nearby?"
        ]
        
        if location:
            print(f"📍 Current simulated location: {location['name']}")
            print()
        
        print("Testing common location queries:")
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: \"{query}\"")
            try:
                response = self.system.process_message(self.demo_user_id, query)
                # Show first 150 characters of response
                short_response = response[:150] + "..." if len(response) > 150 else response
                print(f"   🤖 Response: {short_response}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    def show_location_benefits(self):
        """Show benefits of location-based features"""
        print("\n🌟 LOCATION-BASED BENEFITS")
        print("-" * 40)
        
        benefits = {
            '🍽️ Restaurant Recommendations': [
                'Walking distances to restaurants',
                'Neighborhood-specific cuisine suggestions',
                'Real-time availability and ratings',
                'Hidden local gems within 500m radius'
            ],
            '🚇 Transportation Guidance': [
                'Turn-by-turn directions to stations',
                'Real-time metro, bus, and ferry updates',
                'Alternative routes during disruptions',
                'Walking times to transport hubs'
            ],
            '🏛️ Attraction Discovery': [
                'Nearby historical sites and museums',
                'Optimal visiting routes and timing',
                'Cultural context for your location',
                'Photography spots and viewpoints'
            ],
            '🌍 Cultural Insights': [
                'Local customs and etiquette tips',
                'Seasonal recommendations',
                'Weather-appropriate suggestions',
                'Authentic local experiences'
            ]
        }
        
        for category, items in benefits.items():
            print(f"{category}")
            for item in items:
                print(f"   • {item}")
            print()
    
    def interactive_menu(self):
        """Interactive menu for the demo"""
        while True:
            print("\n🎯 INTERACTIVE MENU")
            print("-" * 30)
            print("1. Show Location Collection Methods")
            print("2. Demonstrate GPS Collection")
            print("3. Test Location Detection Scenarios")
            print("4. Simulate GPS Location")
            print("5. Try Location-Based Queries")
            print("6. Show Location Benefits")
            print("7. Complete Demo Walkthrough")
            print("Q. Quit Demo")
            
            choice = input("\nYour choice: ").strip().upper()
            
            if choice == 'Q':
                print("\n👋 Thank you for exploring the Istanbul AI Location System!")
                print("🚀 Ready to experience personalized Istanbul recommendations!")
                break
            elif choice == '1':
                self.show_location_collection_methods()
            elif choice == '2':
                self.demonstrate_gps_collection()
            elif choice == '3':
                self.test_location_scenarios()
            elif choice == '4':
                location = self.simulate_gps_locations()
                if location:
                    self.demonstrate_location_queries(location)
            elif choice == '5':
                self.demonstrate_location_queries()
            elif choice == '6':
                self.show_location_benefits()
            elif choice == '7':
                self.run_complete_demo()
            else:
                print("❌ Invalid choice! Please try again.")
    
    def run_complete_demo(self):
        """Run the complete demo walkthrough"""
        print("\n🎬 COMPLETE DEMO WALKTHROUGH")
        print("=" * 50)
        
        # Step 1: Show methods
        print("\n📋 STEP 1: Location Collection Methods")
        self.show_location_collection_methods()
        input("Press Enter to continue...")
        
        # Step 2: Demonstrate collection
        print("\n📋 STEP 2: GPS Collection Demo")
        self.demonstrate_gps_collection()
        input("Press Enter to continue...")
        
        # Step 3: Test scenarios
        print("\n📋 STEP 3: Location Detection Test")
        self.test_location_scenarios()
        input("Press Enter to continue...")
        
        # Step 4: Simulate GPS
        print("\n📋 STEP 4: GPS Location Simulation")
        location = self.simulate_gps_locations()
        input("Press Enter to continue...")
        
        # Step 5: Query demonstration
        print("\n📋 STEP 5: Location-Based Queries")
        self.demonstrate_location_queries(location)
        input("Press Enter to continue...")
        
        # Step 6: Show benefits
        print("\n📋 STEP 6: System Benefits")
        self.show_location_benefits()
        
        print("\n✅ COMPLETE DEMO FINISHED!")
        print("🎉 Istanbul AI Location System is ready for production!")
    
    def run(self):
        """Run the interactive location demo"""
        self.display_banner()
        
        print("Welcome to the Istanbul AI Location System Demo!")
        print("This interactive demonstration shows how our AI system")
        print("collects, processes, and uses location data to provide")
        print("personalized Istanbul recommendations.")
        print()
        
        # Ask for demo mode
        print("Choose demo mode:")
        print("1. Interactive Menu (explore features step by step)")
        print("2. Complete Walkthrough (full automated demo)")
        
        try:
            mode = input("Enter choice (1 or 2): ").strip()
            
            if mode == '1':
                self.interactive_menu()
            elif mode == '2':
                self.run_complete_demo()
            else:
                print("Invalid choice, starting interactive menu...")
                self.interactive_menu()
                
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Thanks for exploring!")
        except Exception as e:
            print(f"\n❌ Demo error: {e}")

def main():
    """Main function to run the demo"""
    try:
        demo = LocationDemo()
        demo.run()
    except Exception as e:
        print(f"❌ Failed to start demo: {e}")
        print("Please ensure the Istanbul AI system is properly installed.")

if __name__ == "__main__":
    main()
