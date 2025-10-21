"""
Example: Integrating Intelligent Route System into Chat Backend
================================================================

This shows how to integrate the intelligent route system into your
existing chat backend (backend/main.py or similar).
"""

import logging
from typing import Dict, Any, Optional
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add backend services to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'services'))

# Import the chat route handler
try:
    from ai_chat_route_integration import process_chat_route_request, get_chat_route_handler
    ROUTE_HANDLER_AVAILABLE = True
    logger.info("✅ Route handler available")
except ImportError as e:
    ROUTE_HANDLER_AVAILABLE = False
    logger.warning(f"⚠️ Route handler not available: {e}")


class EnhancedChatHandler:
    """
    Enhanced Chat Handler with Intelligent Route Integration
    
    This is an example of how to integrate route handling into your
    existing chat system.
    """
    
    def __init__(self):
        """Initialize chat handler with route integration"""
        self.route_handler = None
        
        if ROUTE_HANDLER_AVAILABLE:
            self.route_handler = get_chat_route_handler()
            logger.info("✅ Chat handler initialized with route integration")
        else:
            logger.warning("⚠️ Route integration not available")
    
    def process_message(
        self,
        message: str,
        user_id: Optional[str] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process chat message with route intelligence
        
        Args:
            message: User's chat message
            user_id: Optional user ID
            user_context: Optional user context (location, preferences, etc.)
            
        Returns:
            Response dict with message and optional route data
        """
        # 1. Check if this is a route request
        if self.route_handler:
            route_response = self.route_handler.handle_route_request(
                message, user_context
            )
            
            if route_response:
                # This is a route request - return route response
                logger.info(f"🗺️ Route request processed for user {user_id}")
                return {
                    **route_response,
                    'source': 'route_integration',
                    'user_id': user_id
                }
        
        # 2. Not a route request - process as normal chat
        return self._process_normal_chat(message, user_id, user_context)
    
    def _process_normal_chat(
        self,
        message: str,
        user_id: Optional[str],
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process normal chat message (your existing chat logic)
        
        Replace this with your actual chat processing logic
        """
        # This is where your existing chat processing would go
        # For example:
        # - Call GPT/LLM
        # - Query knowledge base
        # - Process restaurant requests
        # - Handle general questions
        
        return {
            'type': 'chat',
            'message': f"Echo: {message}",
            'source': 'normal_chat',
            'user_id': user_id
        }


# Example usage in FastAPI endpoint (for backend/main.py)
def example_fastapi_integration():
    """
    Example of how to integrate into FastAPI chat endpoint
    """
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    app = FastAPI()
    chat_handler = EnhancedChatHandler()
    
    class ChatRequest(BaseModel):
        message: str
        user_id: Optional[str] = None
        user_context: Optional[Dict[str, Any]] = None
    
    @app.post("/api/chat")
    async def chat_endpoint(request: ChatRequest):
        """
        Chat endpoint with route integration
        """
        response = chat_handler.process_message(
            message=request.message,
            user_id=request.user_id,
            user_context=request.user_context
        )
        
        return response
    
    return app


# Example usage in Flask (if you're using Flask)
def example_flask_integration():
    """
    Example of how to integrate into Flask chat endpoint
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    chat_handler = EnhancedChatHandler()
    
    @app.route('/api/chat', methods=['POST'])
    def chat_endpoint():
        """
        Chat endpoint with route integration
        """
        data = request.get_json()
        
        response = chat_handler.process_message(
            message=data.get('message'),
            user_id=data.get('user_id'),
            user_context=data.get('user_context')
        )
        
        return jsonify(response)
    
    return app


# Example for Istanbul AI main system integration
def example_istanbul_ai_integration():
    """
    Example of how to integrate into Istanbul AI main system
    """
    # Add to istanbul_ai/main_system.py
    
    code_example = """
    # In istanbul_ai/main_system.py
    
    from backend.services.ai_chat_route_integration import get_chat_route_handler
    
    class IstanbulAI:
        def __init__(self):
            # ...existing initialization...
            
            # Add route handler
            try:
                self.route_handler = get_chat_route_handler()
                logger.info("✅ Route handler integrated")
            except Exception as e:
                self.route_handler = None
                logger.warning(f"⚠️ Route handler not available: {e}")
        
        def process_message(self, message: str, user_context: dict = None):
            # Check for route request FIRST
            if self.route_handler:
                route_response = self.route_handler.handle_route_request(
                    message, user_context
                )
                if route_response:
                    return route_response
            
            # Continue with existing message processing
            # ...your existing code...
    """
    
    print(code_example)


# Testing
if __name__ == "__main__":
    print("🧪 Testing Chat Integration...\n")
    
    # Initialize handler
    handler = EnhancedChatHandler()
    
    # Test messages
    test_cases = [
        {
            'message': "Show me route from Sultanahmet to Galata Tower",
            'user_context': {'preferences': {'prefer_walking': True}}
        },
        {
            'message': "What's the weather like?",
            'user_context': {}
        },
        {
            'message': "Plan route visiting Taksim, Grand Bazaar, and Blue Mosque",
            'user_context': {}
        },
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['message']}")
        print('='*60)
        
        response = handler.process_message(
            message=test['message'],
            user_id=f"test_user_{i}",
            user_context=test['user_context']
        )
        
        print(f"\nType: {response['type']}")
        print(f"Source: {response['source']}")
        
        if response['type'] == 'route':
            print(f"\n{response['message'][:200]}...")
            print(f"\nRoute data available: {bool(response.get('route_data'))}")
        else:
            print(f"\nMessage: {response['message']}")
    
    print("\n" + "="*60)
    print("✅ Integration test complete!")
    print("\nNext: Add this to your backend/main.py")
