"""
Service-Enhanced LLM Integration Example
Demonstrates how to use LLM with real-time service data
"""

import asyncio
import logging
from typing import Dict, Any

from services.runpod_llm_client import get_llm_client
from services.llm_context_builder import get_context_builder
from services.llm_service_registry import get_service_registry

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_service_enhanced_llm():
    """Demonstrate service-enhanced LLM responses"""
    
    print("\n" + "="*70)
    print("🚀 SERVICE-ENHANCED LLM DEMO")
    print("="*70 + "\n")
    
    # Initialize components
    llm_client = get_llm_client()
    context_builder = get_context_builder()
    service_registry = get_service_registry()
    
    # Check LLM health
    print("📊 Checking LLM health...")
    health = await llm_client.health_check()
    print(f"   Status: {health.get('status')}")
    print(f"   Endpoint: {health.get('endpoint')}\n")
    
    if health.get("status") != "healthy":
        print("❌ LLM not available. Make sure RunPod vLLM is running.")
        print("   See: RUNPOD_VLLM_SETUP.md")
        return
    
    # List available services
    print("📋 Available Services:")
    services = service_registry.list_all_services()
    for service in services:
        print(f"   • {service['name']} - {service['description'][:60]}...")
    print()
    
    # Test cases
    test_cases = [
        {
            "query": "Best kebab restaurants in Sultanahmet?",
            "intent": "restaurant_recommendation",
            "entities": {
                "cuisine": "kebab",
                "district": "Sultanahmet"
            },
            "user_location": {"lat": 41.0082, "lon": 28.9784}  # Sultanahmet
        },
        {
            "query": "How do I get from Taksim to Kadıköy?",
            "intent": "route_planning",
            "entities": {
                "from_location": "Taksim",
                "to_location": "Kadıköy"
            },
            "user_location": {"lat": 41.0369, "lon": 28.9857}  # Taksim
        },
        {
            "query": "What's the weather like in Istanbul?",
            "intent": "weather",
            "entities": {},
            "user_location": None
        },
        {
            "query": "What museums should I visit?",
            "intent": "attractions",
            "entities": {
                "category": "museum"
            },
            "user_location": None
        }
    ]
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST {idx}: {test['query']}")
        print(f"{'='*70}\n")
        
        # Step 1: Build context from services
        print("🔍 Step 1: Building context from services...")
        context = await context_builder.build_context(
            query=test["query"],
            intent=test["intent"],
            entities=test["entities"],
            user_location=test["user_location"]
        )
        
        print(f"   Intent: {context['intent']}")
        print(f"   Services called: {list(context['service_data'].keys())}")
        
        # Step 2: Format context for LLM
        print("\n📝 Step 2: Formatting context for LLM...")
        formatted_context = context_builder.format_context_for_llm(context)
        
        if formatted_context:
            print(f"   Context preview: {formatted_context[:200]}...")
        else:
            print("   No service data available")
        
        # Step 3: Generate LLM response with service context
        print("\n🤖 Step 3: Generating service-enhanced response...")
        response = await llm_client.generate_with_service_context(
            query=test["query"],
            intent=test["intent"],
            entities=test["entities"],
            service_context=context
        )
        
        # Display results
        print("\n" + "─"*70)
        print("📤 RESPONSE:")
        print("─"*70)
        
        if response:
            print(f"\n{response}\n")
        else:
            print("\n❌ No response generated\n")
        
        print("─"*70)
        
        # Wait before next test
        await asyncio.sleep(2)
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70 + "\n")


async def test_without_services():
    """Test LLM without service context (for comparison)"""
    
    print("\n" + "="*70)
    print("🔵 LLM WITHOUT SERVICE CONTEXT (Baseline)")
    print("="*70 + "\n")
    
    llm_client = get_llm_client()
    
    query = "Best kebab restaurants in Sultanahmet?"
    
    print(f"Query: {query}\n")
    print("🤖 Generating generic response...\n")
    
    response = await llm_client.generate_istanbul_response(query)
    
    print("─"*70)
    print("📤 RESPONSE (No Service Data):")
    print("─"*70)
    print(f"\n{response}\n")
    print("─"*70)
    
    print("\nNotice: Generic response without specific restaurant names/details")


async def test_with_services():
    """Test LLM with service context"""
    
    print("\n" + "="*70)
    print("🟢 LLM WITH SERVICE CONTEXT (Enhanced)")
    print("="*70 + "\n")
    
    llm_client = get_llm_client()
    context_builder = get_context_builder()
    
    query = "Best kebab restaurants in Sultanahmet?"
    
    print(f"Query: {query}\n")
    print("🔍 Fetching real restaurant data...\n")
    
    context = await context_builder.build_context(
        query=query,
        intent="restaurant_recommendation",
        entities={"cuisine": "kebab", "district": "Sultanahmet"},
        user_location={"lat": 41.0082, "lon": 28.9784}
    )
    
    print("🤖 Generating service-enhanced response...\n")
    
    response = await llm_client.generate_with_service_context(
        query=query,
        intent="restaurant_recommendation",
        service_context=context
    )
    
    print("─"*70)
    print("📤 RESPONSE (With Service Data):")
    print("─"*70)
    print(f"\n{response}\n")
    print("─"*70)
    
    print("\nNotice: Specific restaurant names, ratings, and details!")


async def run_comparison():
    """Run side-by-side comparison"""
    print("\n" + "="*70)
    print("⚖️  SERVICE INTEGRATION COMPARISON")
    print("="*70)
    
    await test_without_services()
    await asyncio.sleep(2)
    await test_with_services()
    
    print("\n" + "="*70)
    print("🎯 COMPARISON COMPLETE")
    print("="*70)
    print("\nKey Benefits of Service Integration:")
    print("✓ Specific restaurant names and ratings")
    print("✓ Real-time data (opening hours, prices)")
    print("✓ Accurate locations and directions")
    print("✓ Current weather and transportation info")
    print("✓ Personalized recommendations based on user location")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "comparison":
            asyncio.run(run_comparison())
        elif mode == "demo":
            asyncio.run(demo_service_enhanced_llm())
        else:
            print("Usage: python llm_service_integration_demo.py [demo|comparison]")
    else:
        # Run full demo by default
        asyncio.run(demo_service_enhanced_llm())
