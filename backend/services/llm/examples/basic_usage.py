"""
Basic usage example for Pure LLM Handler.

This script demonstrates the simplest way to use the Pure LLM Handler
for processing user queries.
"""
import asyncio
from backend.services.llm.core import PureLLMHandler


async def main():
    """Run basic example."""
    # Initialize handler with minimal configuration
    config = {
        'openai': {
            'api_key': 'your-api-key-here',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7
        },
        'caching': {
            'enable_cache': True
        }
    }
    
    handler = PureLLMHandler(config)
    
    print("=" * 60)
    print("Pure LLM Handler - Basic Usage Example")
    print("=" * 60)
    
    # Example 1: Simple restaurant query
    print("\n1. Simple Restaurant Query")
    print("-" * 60)
    query1 = "Can you recommend a good Turkish restaurant in Sultanahmet?"
    context1 = {
        'user_id': 'user_demo_001',
        'session_id': 'session_001'
    }
    
    print(f"Query: {query1}")
    response1 = await handler.process_query(query1, context1)
    print(f"Response: {response1['response']}")
    print(f"Cache Hit: {response1.get('cache_hit', False)}")
    print(f"Response Time: {response1.get('response_time', 0):.3f}s")
    
    # Example 2: Hotel query
    print("\n2. Hotel Query")
    print("-" * 60)
    query2 = "What are some affordable hotels near Taksim Square?"
    context2 = {
        'user_id': 'user_demo_001',
        'session_id': 'session_001'
    }
    
    print(f"Query: {query2}")
    response2 = await handler.process_query(query2, context2)
    print(f"Response: {response2['response']}")
    print(f"Intent Detected: {response2.get('intent_type', 'unknown')}")
    
    # Example 3: Follow-up query (uses conversation history)
    print("\n3. Follow-up Query")
    print("-" * 60)
    query3 = "What about ones with sea view?"
    context3 = {
        'user_id': 'user_demo_001',
        'session_id': 'session_001'
    }
    
    print(f"Query: {query3}")
    response3 = await handler.process_query(query3, context3)
    print(f"Response: {response3['response']}")
    print(f"Conversation Context Used: {len(response3.get('conversation_history', []))} messages")
    
    # Example 4: Query with explicit preferences
    print("\n4. Query with User Preferences")
    print("-" * 60)
    query4 = "Show me vegetarian restaurants"
    context4 = {
        'user_id': 'user_demo_002',
        'user_preferences': {
            'dietary': 'vegetarian',
            'price_range': 'medium'
        }
    }
    
    print(f"Query: {query4}")
    print(f"Preferences: {context4['user_preferences']}")
    response4 = await handler.process_query(query4, context4)
    print(f"Response: {response4['response']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
