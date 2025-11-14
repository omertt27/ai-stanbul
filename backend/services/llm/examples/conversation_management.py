"""
Conversation management example for Pure LLM Handler.

This script demonstrates conversation management features including:
- Multi-turn conversations
- Session management
- Context preservation
- Topic tracking
"""
import asyncio
from backend.services.llm.core import PureLLMHandler


async def main():
    """Run conversation management example."""
    config = {
        'openai': {
            'api_key': 'your-api-key-here',
            'model': 'gpt-3.5-turbo'
        },
        'conversation': {
            'max_history_length': 10,
            'session_timeout': 1800,
            'enable_context_compression': True
        }
    }
    
    handler = PureLLMHandler(config)
    
    print("=" * 60)
    print("Pure LLM Handler - Conversation Management Example")
    print("=" * 60)
    
    # Scenario: Planning a day trip in Istanbul
    user_id = 'conversation_demo_user'
    session_id = 'day_trip_planning'
    
    conversation = [
        "I'm planning to visit Istanbul next week",
        "What are the must-see historical sites?",
        "Tell me more about the Blue Mosque",
        "What's the best time to visit it?",
        "Are there any good restaurants nearby?",
        "I prefer Turkish cuisine, not too expensive",
        "Great! What about after lunch?",
        "I'm interested in shopping for souvenirs",
        "Where is the Grand Bazaar exactly?",
        "How do I get there from the Blue Mosque?"
    ]
    
    print("\n" + "=" * 60)
    print("Starting Multi-Turn Conversation")
    print("=" * 60)
    
    for turn_num, query in enumerate(conversation, 1):
        print(f"\n{'='*60}")
        print(f"Turn {turn_num}")
        print(f"{'='*60}")
        print(f"User: {query}")
        
        context = {
            'user_id': user_id,
            'session_id': session_id
        }
        
        response = await handler.process_query(query, context)
        
        print(f"\nAssistant: {response['response']}")
        
        # Show conversation metadata
        metadata = response.get('metadata', {})
        if metadata:
            print(f"\n--- Conversation Metadata ---")
            if 'conversation_length' in metadata:
                print(f"Messages in history: {metadata['conversation_length']}")
            if 'detected_topic' in metadata:
                print(f"Current topic: {metadata['detected_topic']}")
            if 'topic_changed' in metadata:
                print(f"Topic changed: {metadata['topic_changed']}")
        
        # Small delay between turns
        await asyncio.sleep(0.5)
    
    # Show conversation summary
    print("\n" + "=" * 60)
    print("Conversation Summary")
    print("=" * 60)
    
    summary = handler.get_conversation_summary(session_id)
    if summary:
        print(f"Total turns: {summary.get('turn_count', 0)}")
        print(f"Topics discussed: {', '.join(summary.get('topics', []))}")
        print(f"User preferences identified:")
        for key, value in summary.get('preferences', {}).items():
            print(f"  - {key}: {value}")
    
    # Example of starting a new topic in same session
    print("\n" + "=" * 60)
    print("Topic Change Detection")
    print("=" * 60)
    
    new_topic_query = "Actually, I also need to find a hotel for my stay"
    print(f"User: {new_topic_query}")
    
    response = await handler.process_query(
        new_topic_query,
        {'user_id': user_id, 'session_id': session_id}
    )
    
    print(f"\nAssistant: {response['response']}")
    if response.get('metadata', {}).get('topic_changed'):
        print("\n✓ Topic change detected: Restaurants/Tourism → Hotels")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
