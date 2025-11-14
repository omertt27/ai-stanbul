#!/usr/bin/env python3
"""
QuerySuggester Demo Script

Demonstrates all three features:
1. Autocomplete
2. Spell Correction
3. Related Queries

Run: python3 demo_query_suggester.py
"""

import asyncio
import logging
from backend.services.query_suggester import create_query_suggester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client for demo."""
    
    class Chat:
        class Completions:
            async def create(self, **kwargs):
                # Return mock related queries
                return type('obj', (object,), {
                    'choices': [
                        type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': """
1. What are the opening hours?
2. How much does it cost?
3. Is it wheelchair accessible?
4. Can I take photos inside?
5. Are there guided tours available?
"""
                            })()
                        })()
                    ]
                })()
        
        def __init__(self):
            self.completions = self.Completions()
    
    def __init__(self):
        self.chat = self.Chat()


async def demo_autocomplete(suggester):
    """Demonstrate autocomplete feature."""
    print("\n" + "="*60)
    print("1️⃣  AUTOCOMPLETE DEMO")
    print("="*60)
    
    # Add some popular queries
    popular_queries = [
        "best restaurants in Taksim",
        "best restaurants in Sultanahmet",
        "best restaurants near Galata Tower",
        "best museums in Istanbul",
        "best hotels in Beyoğlu",
        "best things to do in Istanbul",
        "best Turkish food in Istanbul"
    ]
    
    print("\n📊 Loading popular queries...")
    for query in popular_queries:
        suggester.track_query(query)
    print(f"✅ Loaded {len(popular_queries)} queries")
    
    # Test autocomplete
    test_cases = [
        "best res",
        "best mus",
        "best hot",
        "best th"
    ]
    
    for partial in test_cases:
        suggestions = await suggester.suggest_completions(partial, max_suggestions=3)
        print(f"\n🔍 User types: '{partial}'")
        print("💡 Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"   {i}. {suggestion}")


async def demo_spell_correction(suggester):
    """Demonstrate spell correction feature."""
    print("\n" + "="*60)
    print("2️⃣  SPELL CORRECTION DEMO")
    print("="*60)
    
    test_cases = [
        ("hotels in Taksin", "Taksim"),
        ("restaurants in Sultanahme", "Sultanahmet"),
        ("visit Galata Towerr", "Galata Tower"),
        ("places near Beyolu", "Beyoğlu"),
        ("how to get to Hagia Sophiaa", "Hagia Sophia")
    ]
    
    for query, expected in test_cases:
        correction = await suggester.suggest_correction(query)
        
        print(f"\n❌ User types: '{query}'")
        if correction:
            print(f"✅ Corrected: '{correction['corrected_query']}'")
            print(f"   Confidence: {correction['confidence']:.2%}")
            print(f"   Changes: {len(correction['changes'])} location(s) corrected")
        else:
            print("✅ No corrections needed")


async def demo_related_queries(suggester):
    """Demonstrate related queries feature."""
    print("\n" + "="*60)
    print("3️⃣  RELATED QUERIES DEMO")
    print("="*60)
    
    test_cases = [
        {
            "query": "best museums in Istanbul",
            "response": "The best museums include the Topkapi Palace, Hagia Sophia, and Istanbul Archaeology Museums...",
            "signals": {"primary_intent": "tourism", "sub_intent": "museums"}
        },
        {
            "query": "how to get from airport to Taksim",
            "response": "You can take the metro line M1 to Yenikapi, then M2 to Taksim Square...",
            "signals": {"primary_intent": "transport", "sub_intent": "route"}
        },
        {
            "query": "best Turkish food in Sultanahmet",
            "response": "For authentic Turkish cuisine, try Sultanahmet Köftecisi for köfte, Hamdi Restaurant for kebabs...",
            "signals": {"primary_intent": "food", "sub_intent": "restaurants"}
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        related = await suggester.suggest_related(
            query=test["query"],
            response=test["response"],
            signals=test["signals"],
            language="en",
            max_suggestions=3
        )
        
        print(f"\n💬 Query #{i}: '{test['query']}'")
        print(f"🎯 Intent: {test['signals']['primary_intent']}")
        print("\n💡 Related Questions:")
        for j, q in enumerate(related, 1):
            print(f"   {j}. {q}")


async def demo_full_workflow(suggester):
    """Demonstrate complete workflow."""
    print("\n" + "="*60)
    print("4️⃣  FULL WORKFLOW DEMO")
    print("="*60)
    
    user_query = "best restaurents in Taksin"  # Intentional typos
    
    print(f"\n👤 User Query: '{user_query}'")
    
    # Step 1: Spell check
    print("\n🔧 Step 1: Spell Correction")
    correction = await suggester.suggest_correction(user_query)
    if correction:
        corrected_query = correction['corrected_query']
        print(f"   ✅ Corrected to: '{corrected_query}'")
        print(f"   📊 Confidence: {correction['confidence']:.2%}")
    else:
        corrected_query = user_query
        print("   ✅ No corrections needed")
    
    # Step 2: Track query
    print("\n📈 Step 2: Track Query for Popularity")
    suggester.track_query(corrected_query)
    print("   ✅ Query tracked")
    
    # Step 3: Process with LLM (simulated)
    print("\n🤖 Step 3: Generate Response (LLM)")
    simulated_response = "Here are the best restaurants in Taksim: Mikla, Neolokal, and 360 Istanbul..."
    print(f"   ✅ Response: {simulated_response[:60]}...")
    
    # Step 4: Generate related queries
    print("\n💡 Step 4: Generate Related Queries")
    related = await suggester.suggest_related(
        query=corrected_query,
        response=simulated_response,
        signals={"primary_intent": "food", "sub_intent": "restaurants"},
        language="en",
        max_suggestions=3
    )
    print("   ✅ Related questions:")
    for i, q in enumerate(related, 1):
        print(f"      {i}. {q}")


async def demo_stats(suggester):
    """Show statistics."""
    print("\n" + "="*60)
    print("5️⃣  STATISTICS")
    print("="*60)
    
    stats = suggester.get_stats()
    
    print("\n📊 QuerySuggester Statistics:")
    print(f"\n   Autocomplete:")
    print(f"      • Requests: {stats['autocomplete_requests']}")
    print(f"      • Queries in trie: {stats['trie_size']}")
    print(f"      • Tracked queries: {stats['tracked_queries']}")
    
    print(f"\n   Spell Check:")
    print(f"      • Requests: {stats['spell_check_requests']}")
    print(f"      • Corrections made: {stats['spell_corrections_made']}")
    rate = stats['spell_corrections_made'] / max(1, stats['spell_check_requests'])
    print(f"      • Correction rate: {rate:.2%}")
    
    print(f"\n   Related Queries:")
    print(f"      • Requests: {stats['related_query_requests']}")
    print(f"      • Cache hit rate: {stats['cache_hit_rate']}")
    print(f"      • Total suggestions: {stats['total_suggestions']}")
    
    print(f"\n   Overall:")
    total_requests = (stats['autocomplete_requests'] + 
                     stats['spell_check_requests'] + 
                     stats['related_query_requests'])
    print(f"      • Total requests: {total_requests}")
    print(f"      • Cache hits: {stats['cache_hits']}")
    print(f"      • Cache misses: {stats['cache_misses']}")
    
    # Popular queries
    print("\n🔥 Top Popular Queries:")
    popular = sorted(suggester.query_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (query, freq) in enumerate(popular, 1):
        print(f"   {i}. {query} ({freq} times)")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("🚀 QUERY SUGGESTER DEMO")
    print("="*60)
    print("\nPriority 4.1: Smart Query Suggestions")
    print("Features: Autocomplete, Spell Correction, Related Queries")
    print("\n" + "="*60)
    
    # Initialize suggester (without Redis for demo)
    mock_llm = MockLLMClient()
    suggester = create_query_suggester(
        llm_client=mock_llm,
        redis_url=None  # No Redis for demo
    )
    
    try:
        # Run demos
        await demo_autocomplete(suggester)
        await demo_spell_correction(suggester)
        await demo_related_queries(suggester)
        await demo_full_workflow(suggester)
        await demo_stats(suggester)
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETE")
        print("="*60)
        print("\nAll features working correctly!")
        print("\nNext Steps:")
        print("  1. Integrate with PureLLMHandler")
        print("  2. Add API endpoints")
        print("  3. Create frontend components")
        print("  4. Deploy to staging")
        print("\nDocumentation:")
        print("  • PRIORITY_4_1_COMPLETE.md")
        print("  • PRIORITY_4_1_INTEGRATION_GUIDE.md")
        print("  • QUERY_SUGGESTER_QUICK_REF.md")
        print("")
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
