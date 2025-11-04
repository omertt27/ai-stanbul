#!/usr/bin/env python3
"""
Simple Demo: Map-Focused LLM Output
Shows how LLM provides brief context while map shows details
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ml_systems.llm_service_wrapper import LLMServiceWrapper

def simple_prompt(route, weather):
    """Create ultra-simple prompt for brief context"""
    return f"""Route: {route['from']} to {route['to']} ({route['duration']} min via {route['mode']})
Weather: {weather['condition']}, {weather['temp']}°C

In ONE sentence (max 20 words): Why is this route good right now?

Tip:"""

def main():
    print("=" * 70)
    print("MAP-FOCUSED LLM OUTPUT DEMO")
    print("=" * 70)
    print("\n🎯 Goal: LLM gives brief context, map shows full details\n")
    
    # Initialize LLM
    print("📦 Loading LLM...")
    llm = LLMServiceWrapper()
    info = llm.get_info()
    print(f"✅ Loaded {info['model_name']} on {info['device']}\n")
    
    # Test Case 1: Marmaray in rain
    print("-" * 70)
    print("TEST 1: Marmaray route (rainy weather)")
    print("-" * 70)
    
    route1 = {'from': 'Taksim', 'to': 'Kadıköy', 'duration': 35, 'mode': 'Marmaray'}
    weather1 = {'condition': 'Rainy', 'temp': 14}
    
    prompt1 = simple_prompt(route1, weather1)
    print(f"\n📍 Route: {route1['from']} → {route1['to']}")
    print(f"🌧️ Weather: {weather1['condition']}, {weather1['temp']}°C")
    print(f"\n🤖 Generating tip...")
    
    try:
        response1 = llm.generate(
            prompt1,
            max_tokens=30,  # Very brief
            temperature=0.7
        )
        if response1:
            # Count tokens (approximate)
            tokens = len(response1.split())
            print(f"\n💡 LLM Tip: \"{response1}\"")
            print(f"   (~{tokens} words)")
        else:
            print(f"\n❌ Generation failed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Test Case 2: Ferry in sunshine
    print("\n" + "-" * 70)
    print("TEST 2: Ferry route (sunny weather)")
    print("-" * 70)
    
    route2 = {'from': 'Beşiktaş', 'to': 'Kadıköy', 'duration': 25, 'mode': 'Ferry'}
    weather2 = {'condition': 'Sunny', 'temp': 22}
    
    prompt2 = simple_prompt(route2, weather2)
    print(f"\n📍 Route: {route2['from']} → {route2['to']}")
    print(f"☀️ Weather: {weather2['condition']}, {weather2['temp']}°C")
    print(f"\n🤖 Generating tip...")
    
    try:
        response2 = llm.generate(
            prompt2,
            max_tokens=30,
            temperature=0.7
        )
        if response2:
            tokens = len(response2.split())
            print(f"\n💡 LLM Tip: \"{response2}\"")
            print(f"   (~{tokens} words)")
        else:
            print(f"\n❌ Generation failed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Show example UI
    print("\n" + "=" * 70)
    print("EXAMPLE UI RENDERING")
    print("=" * 70)
    print("""
    ┌────────────────────────────────┐
    │   🗺️ [Interactive Route Map]   │
    │   Taksim → Kadıköy             │
    │   35 min | 18 km | Marmaray   │
    └────────────────────────────────┘
    
    ┌────────────────────────────────┐
    │ 💡 Local Tip                   │
    │ Perfect for rainy days -       │
    │ completely underground!        │
    └────────────────────────────────┘
    """)
    
    print("\n✅ Key Benefits:")
    print("   • LLM: Brief context (1-2 sentences)")
    print("   • Map: Full route visualization")
    print("   • Fast: < 30 tokens, ~1-2 seconds")
    print("   • Clear: Separation of concerns\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
