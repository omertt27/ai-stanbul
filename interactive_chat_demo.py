#!/usr/bin/env python3
"""
Interactive Chat Demo with Transfer Instructions & Map Visualization
====================================================================

Interactive demo of the Istanbul AI chat with integrated transfer instructions
and map visualization features.
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Print welcome banner"""
    print("\n" + "🗺️" + "="*78 + "🗺️")
    print("  ISTANBUL AI CHAT - Transfer Instructions & Map Visualization")
    print("  Interactive Demo")
    print("🗺️" + "="*78 + "🗺️\n")

def print_help():
    """Print help message"""
    print("\n📋 Available Commands:")
    print("  • Type any transportation query (e.g., 'How do I get from Taksim to Kadıköy?')")
    print("  • 'help' - Show this help message")
    print("  • 'examples' - Show example queries")
    print("  • 'quit' or 'exit' - Exit the demo")
    print()

def print_examples():
    """Print example queries"""
    print("\n💡 Example Queries:")
    print("  1. How do I get from Taksim to Kadıköy?")
    print("  2. What's the best route from Sultanahmet to Istanbul Airport?")
    print("  3. Route from Beşiktaş to Üsküdar")
    print("  4. How to reach Kadıköy from Taksim by metro?")
    print("  5. Directions to the airport from Sultanahmet")
    print("  6. Tell me about Istanbul metro")
    print()

def format_response(result):
    """Format and display the response"""
    if isinstance(result, dict):
        response = result.get('response', '')
        map_data = result.get('map_data', {})
        intent = result.get('intent', 'unknown')
        
        print(f"\n{'─'*80}")
        print(response)
        print(f"{'─'*80}")
        
        # Show metadata
        print("\n📊 Response Metadata:")
        print(f"   Intent: {intent}")
        print(f"   Has Map Visualization: {bool(map_data)}")
        
        if map_data:
            features = map_data.get('features', [])
            print(f"   Map Features: {len(features)}")
            
            # Ask if user wants to save map
            save = input("\n💾 Save map data to file? (y/n): ").strip().lower()
            if save == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"route_map_{timestamp}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(map_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Map saved to: {filename}")
                print(f"   You can visualize this in map_visualization_demo.html")
        
    else:
        # String response
        print(f"\n{'─'*80}")
        print(result)
        print(f"{'─'*80}")

def run_interactive_demo():
    """Run the interactive demo"""
    try:
        from istanbul_ai.main_system import IstanbulDailyTalkAI
        
        print("🚀 Initializing Istanbul AI system...")
        ai = IstanbulDailyTalkAI()
        
        # Check if transportation chat is available
        if hasattr(ai, 'transportation_chat') and ai.transportation_chat:
            print("✅ Transfer Instructions & Map Visualization: Available")
        else:
            print("⚠️ Transfer Instructions & Map Visualization: Not Available")
        
        print("✅ System ready!\n")
        
        # Show help
        print_help()
        
        # User ID for this session
        user_id = f"demo_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Main loop
        while True:
            try:
                # Get user input
                query = input("💬 You: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() in ['quit', 'exit']:
                    print("\n👋 Thank you for using Istanbul AI! Goodbye!\n")
                    break
                
                if query.lower() == 'help':
                    print_help()
                    continue
                
                if query.lower() == 'examples':
                    print_examples()
                    continue
                
                # Process query
                print("\n🤔 Processing...")
                
                # Get structured response with map data
                result = ai.process_message(query, user_id, return_structured=True)
                
                # Display response
                print("\n🤖 Istanbul AI:")
                format_response(result)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Make sure all required modules are installed.")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def run_automated_demo():
    """Run an automated demo with preset queries"""
    print("\n📺 Running Automated Demo...\n")
    
    try:
        from istanbul_ai.main_system import IstanbulDailyTalkAI
        
        ai = IstanbulDailyTalkAI()
        user_id = "automated_demo_user"
        
        # Demo queries
        demo_queries = [
            "How do I get from Taksim to Kadıköy?",
            "What's the route from Sultanahmet to Istanbul Airport?",
            "Tell me about Istanbul metro"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{'='*80}")
            print(f"Demo Query {i}/{len(demo_queries)}")
            print(f"{'='*80}")
            print(f"\n💬 User: {query}\n")
            
            # Process
            result = ai.process_message(query, user_id, return_structured=True)
            
            # Display
            print("🤖 Istanbul AI:")
            format_response(result)
            
            if i < len(demo_queries):
                input("\n⏸️  Press Enter to continue...")
        
        print(f"\n{'='*80}")
        print("✅ Automated demo complete!")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print_banner()
    
    print("Choose demo mode:")
    print("  1. Interactive Chat (recommended)")
    print("  2. Automated Demo")
    print("  3. Quit")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        run_interactive_demo()
    elif choice == '2':
        run_automated_demo()
    elif choice == '3':
        print("\n👋 Goodbye!\n")
    else:
        print("\n❌ Invalid choice. Please run the script again.\n")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
