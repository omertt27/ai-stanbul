"""
Advanced features example for Pure LLM Handler.

This script demonstrates advanced features including:
- Multi-intent detection
- Query enhancement
- A/B testing
- Analytics tracking
- Context building
"""
import asyncio
from backend.services.llm.core import PureLLMHandler


async def main():
    """Run advanced features example."""
    # Initialize handler with advanced configuration
    config = {
        'openai': {
            'api_key': 'your-api-key-here',
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 500
        },
        'signals': {
            'enable_multi_intent': True,
            'confidence_threshold': 0.6
        },
        'query_enhancement': {
            'enable_spell_check': True,
            'enable_expansion': True,
            'enable_translation': True
        },
        'caching': {
            'enable_cache': True,
            'l1_ttl': 300,
            'l2_ttl': 3600
        },
        'analytics': {
            'enable_analytics': True,
            'track_performance': True
        },
        'experimentation': {
            'enable_ab_testing': True
        }
    }
    
    handler = PureLLMHandler(config)
    
    print("=" * 60)
    print("Pure LLM Handler - Advanced Features Example")
    print("=" * 60)
    
    # Example 1: Multi-intent detection
    print("\n1. Multi-Intent Detection")
    print("-" * 60)
    query1 = "I need a hotel near the Blue Mosque and some good restaurants nearby"
    context1 = {'user_id': 'advanced_user_001'}
    
    print(f"Query: {query1}")
    response1 = await handler.process_query(query1, context1)
    print(f"Response: {response1['response']}")
    print(f"Intents Detected: {response1.get('detected_intents', [])}")
    
    # Example 2: Query enhancement with misspellings
    print("\n2. Query Enhancement (Spell Correction)")
    print("-" * 60)
    query2 = "Show me resturants with gud kebap in sultanahmt"
    context2 = {'user_id': 'advanced_user_001'}
    
    print(f"Original Query: {query2}")
    response2 = await handler.process_query(query2, context2)
    print(f"Enhanced Query: {response2.get('enhanced_query', query2)}")
    print(f"Response: {response2['response']}")
    
    # Example 3: Multilingual query with translation
    print("\n3. Multilingual Query (Turkish)")
    print("-" * 60)
    query3 = "Beyoğlu'nda iyi bir meyhane önerir misiniz?"
    context3 = {'user_id': 'advanced_user_001'}
    
    print(f"Query (Turkish): {query3}")
    response3 = await handler.process_query(query3, context3)
    print(f"Detected Language: {response3.get('detected_language', 'unknown')}")
    print(f"Response: {response3['response']}")
    
    # Example 4: Context-aware recommendation
    print("\n4. Context-Aware Recommendation")
    print("-" * 60)
    query4 = "What should I visit today?"
    context4 = {
        'user_id': 'advanced_user_001',
        'user_location': {
            'lat': 41.0082,
            'lng': 28.9784,
            'area': 'Sultanahmet'
        },
        'user_preferences': {
            'interests': ['history', 'architecture'],
            'pace': 'relaxed'
        },
        'time_context': {
            'time_of_day': 'morning',
            'day_of_week': 'saturday'
        }
    }
    
    print(f"Query: {query4}")
    print(f"Location: {context4['user_location']['area']}")
    print(f"Interests: {context4['user_preferences']['interests']}")
    response4 = await handler.process_query(query4, context4)
    print(f"Response: {response4['response']}")
    
    # Example 5: A/B testing different prompts
    print("\n5. A/B Testing Example")
    print("-" * 60)
    
    # Create an experiment
    experiment_config = {
        'name': 'prompt_style_test',
        'variants': ['friendly', 'professional'],
        'traffic_split': {'friendly': 0.5, 'professional': 0.5}
    }
    
    query5 = "Tell me about Istanbul's nightlife"
    
    for i in range(2):
        context5 = {
            'user_id': f'ab_test_user_{i}',
            'experiment': experiment_config
        }
        print(f"\nUser {i + 1}:")
        print(f"Query: {query5}")
        response5 = await handler.process_query(query5, context5)
        print(f"Assigned Variant: {response5.get('experiment_variant', 'none')}")
        print(f"Response: {response5['response'][:100]}...")
    
    # Example 6: Conversation with context compression
    print("\n6. Long Conversation with Context Compression")
    print("-" * 60)
    
    session_id = 'long_conversation_001'
    conversation_queries = [
        "What are the must-see attractions in Istanbul?",
        "Tell me more about the Hagia Sophia",
        "What's the best time to visit?",
        "Are there any special events?",
        "What about nearby restaurants?"
    ]
    
    for idx, query in enumerate(conversation_queries, 1):
        context6 = {
            'user_id': 'advanced_user_002',
            'session_id': session_id
        }
        print(f"\nTurn {idx}: {query}")
        response6 = await handler.process_query(query, context6)
        print(f"Response: {response6['response'][:150]}...")
        print(f"Conversation Length: {len(response6.get('conversation_history', []))} messages")
    
    # Example 7: Performance monitoring
    print("\n7. Performance Metrics")
    print("-" * 60)
    
    analytics = handler.get_analytics()
    if analytics:
        print(f"Total Queries Processed: {analytics.get('total_queries', 0)}")
        print(f"Average Response Time: {analytics.get('avg_response_time', 0):.3f}s")
        print(f"Cache Hit Rate: {analytics.get('cache_hit_rate', 0):.2%}")
        print(f"Error Rate: {analytics.get('error_rate', 0):.2%}")
    
    # Example 8: Signal detection
    print("\n8. Signal Detection (User Intent Signals)")
    print("-" * 60)
    query8 = "URGENT: Need a hotel near airport NOW! Budget is tight."
    context8 = {'user_id': 'advanced_user_003'}
    
    print(f"Query: {query8}")
    response8 = await handler.process_query(query8, context8)
    signals = response8.get('detected_signals', {})
    print("Detected Signals:")
    if signals:
        for signal_name, signal_value in signals.items():
            print(f"  - {signal_name}: {signal_value}")
    print(f"Response: {response8['response']}")
    
    print("\n" + "=" * 60)
    print("Advanced example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
