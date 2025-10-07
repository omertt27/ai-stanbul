#!/usr/bin/env python3
"""
Main API Integration for Non-LLM System
=======================================

Integrates the ultra-specialized rule-based Istanbul assistant 
with the main API endpoints, replacing GPT/LLM dependency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Optional, Any
from non_llm_istanbul_assistant import get_response

def integrate_with_main_api():
    """
    Integration function that can replace GPT calls in main.py
    """
    return NonLLMAPIIntegration()

class NonLLMAPIIntegration:
    """Integration layer for the main API"""
    
    def __init__(self):
        self.system_name = "Ultra-Specialized Istanbul Assistant"
        self.version = "1.0.0"
    
    async def generate_response(self, user_input: str, session_id: str = None, 
                              user_ip: str = None, context: Dict = None) -> Dict[str, Any]:
        """
        Main response generation method that replaces GPT calls
        """
        try:
            # Generate response using rule-based system
            response = get_response(user_input, context)
            
            # Determine response category
            category = self._determine_category(user_input)
            
            return {
                'success': True,
                'response': response,
                'session_id': session_id or 'rule_based_session',
                'category': category,
                'system': 'rule_based',
                'has_context': bool(context),
                'conversation_turns': context.get('conversation_turns', 0) if context else 0
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'system': 'rule_based'
            }
    
    def _determine_category(self, user_input: str) -> str:
        """Determine the category of the user query"""
        query_lower = user_input.lower()
        
        if any(word in query_lower for word in ['restaurant', 'food', 'eat', 'dining', 'coffee']):
            return 'restaurant'
        elif any(word in query_lower for word in ['tell me about', 'describe', 'neighborhood', 'district']):
            return 'district'
        elif any(word in query_lower for word in ['attractions', 'things to see', 'places to visit', 'sightseeing']):
            return 'attraction'
        elif any(word in query_lower for word in ['transport', 'metro', 'how to get', 'getting to']):
            return 'transportation'
        else:
            return 'general'

# For backward compatibility with existing code
async def get_unified_ai_response(user_input: str, session_id: str = None, 
                                user_ip: str = None, context: Dict = None) -> Dict[str, Any]:
    """
    Drop-in replacement for unified AI system calls
    """
    integration = NonLLMAPIIntegration()
    return await integration.generate_response(user_input, session_id, user_ip, context)

# Main interface function
def get_non_llm_response(user_input: str, session_id: str = None, context: Dict = None) -> str:
    """
    Simple interface that returns just the response text
    """
    response = get_response(user_input, context)
    return response

# Test the integration
if __name__ == "__main__":
    import asyncio
    
    async def test_integration():
        """Test the API integration"""
        print("🧪 TESTING API INTEGRATION")
        print("=" * 50)
        
        test_queries = [
            "restaurants in Sultanahmet",
            "tell me about Beyoğlu",
            "things to see in Istanbul",
            "how to get from Taksim to Sultanahmet"
        ]
        
        integration = NonLLMAPIIntegration()
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            result = await integration.generate_response(query)
            
            print(f"✅ Success: {result['success']}")
            print(f"📂 Category: {result['category']}")
            print(f"💬 Response: {result['response'][:100]}...")
            print("-" * 30)
    
    # Run the test
    asyncio.run(test_integration())
