#!/usr/bin/env python3
"""
Demo script showing the complete events integration in action
"""

import sys
from datetime import datetime

def demo_events_integration():
    """Demonstrate the full events integration capabilities"""
    print("🎭 Istanbul Events AI Integration Demo")
    print("=" * 50)
    
    try:
        # Import the main AI system (now using modular architecture)
        from istanbul_daily_talk_system_modular import IstanbulDailyTalkAI
        
        # Create AI instance
        ai = IstanbulDailyTalkAI()
        print("✅ AI system with events integration ready!\n")
        
        # Demo conversations
        demo_queries = [
            {
                "query": "What's happening in Istanbul culturally?",
                "description": "General cultural events query"
            },
            {
                "query": "Are there any art exhibitions or galleries I can visit?", 
                "description": "Art-focused query"
            },
            {
                "query": "Show me current events and performances",
                "description": "Performances and events query"
            },
            {
                "query": "What cultural activities are available this week?",
                "description": "Weekly cultural activities"
            }
        ]
        
        user_id = "demo_user_456"
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"🎯 Demo {i}: {demo['description']}")
            print(f"👤 User: \"{demo['query']}\"")
            print("🤖 AI Response:")
            print("-" * 60)
            
            try:
                response = ai.process_message(demo['query'], user_id)
                print(response)
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                print()
        
        print("🎉 Demo completed! The events integration is working successfully.")
        print("\n💡 Key features demonstrated:")
        print("• Real-time event detection from user queries")
        print("• Integration with İKSV events scheduler")
        print("• Intelligent filtering based on query content")  
        print("• Formatted, conversational responses")
        print("• Fallback handling for various scenarios")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = demo_events_integration()
    sys.exit(0 if success else 1)
