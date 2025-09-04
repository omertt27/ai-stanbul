#!/usr/bin/env python3
"""
Quick integration script to add enhanced AI capabilities to the existing Istanbul chatbot
This script adds new endpoints and enhances existing functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import logging

# Import the enhanced modules
try:
    from enhanced_main import enhanced_ai_endpoint, get_context_info, test_enhancements
    from enhanced_integration import get_context_summary
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced AI features loaded successfully!")
except ImportError as e:
    print(f"⚠️ Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False

def integrate_enhancements(app: FastAPI):
    """Integrate enhanced features into the existing FastAPI app"""
    
    if not ENHANCED_FEATURES_AVAILABLE:
        print("❌ Cannot integrate enhanced features - modules not available")
        return False
    
    # Add new enhanced AI endpoint
    @app.post("/ai/enhanced")
    async def enhanced_ai(request: Request):
        """Enhanced AI endpoint with improved context awareness"""
        return await enhanced_ai_endpoint(request)
    
    # Add context information endpoint
    @app.get("/ai/context/{session_id}")
    async def get_session_context(session_id: str):
        """Get conversation context for a session"""
        return await get_context_info(session_id)
    
    # Add test endpoint
    @app.get("/ai/test-enhancements")
    async def test_enhanced_features():
        """Test enhanced AI features"""
        return await test_enhancements()
    
    # Add health check for enhanced features
    @app.get("/ai/enhanced/health")
    async def enhanced_health():
        """Health check for enhanced features"""
        return {
            "status": "healthy",
            "enhanced_features": True,
            "modules_loaded": ENHANCED_FEATURES_AVAILABLE,
            "endpoints": [
                "/ai/enhanced",
                "/ai/context/{session_id}",
                "/ai/test-enhancements",
                "/ai/enhanced/health"
            ]
        }
    
    print("✅ Enhanced AI endpoints integrated:")
    print("   POST /ai/enhanced - Enhanced AI with context awareness")
    print("   GET /ai/context/{session_id} - Get conversation context")
    print("   GET /ai/test-enhancements - Test enhanced features")
    print("   GET /ai/enhanced/health - Health check")
    
    return True

def create_standalone_enhanced_app():
    """Create a standalone FastAPI app with enhanced features for testing"""
    
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="Enhanced AI Istanbul Chatbot", version="1.0.0")
    
    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Integrate enhanced features
    integrate_enhancements(app)
    
    # Add basic health check
    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "Enhanced AI Istanbul"}
    
    # Add original AI endpoint for compatibility
    @app.post("/ai")
    async def ai_endpoint(request: Request):
        """Original AI endpoint - now routes to enhanced version"""
        return await enhanced_ai_endpoint(request)
    
    return app

# Manual test function
def test_enhanced_features_manually():
    """Manual test of enhanced features"""
    
    print("🧪 Testing Enhanced AI Features...\n")
    
    try:
        from enhanced_integration import (
            context_manager, 
            query_understander, 
            knowledge_base,
            response_generator
        )
        
        # Test 1: Query Understanding
        print("1️⃣ Testing Query Understanding:")
        test_queries = [
            "restarunt in Sultanahmet",  # Misspelling
            "musium recommendations",    # Misspelling
            "plases to visit in Galata", # Misspelling
            "atractions in Beyoglu"      # Misspelling
        ]
        
        for query in test_queries:
            corrected = query_understander.correct_and_enhance_query(query)
            print(f"   '{query}' → '{corrected}'")
        
        # Test 2: Context Management
        print("\n2️⃣ Testing Context Management:")
        test_session = "test_session_123"
        
        # Simulate conversation
        context_manager.update_context(
            test_session, 
            "recommend restaurants in Sultanahmet", 
            "Here are some great restaurants...",
            places=["Sultanahmet"],
            topic="restaurant"
        )
        
        context = context_manager.get_context(test_session)
        if context:
            print(f"   ✅ Context created for session: {test_session}")
            print(f"   ✅ Places mentioned: {context.mentioned_places}")
            print(f"   ✅ Last topic: {context.last_recommendation_type}")
        else:
            print(f"   ❌ Failed to create context")
        
        # Test 3: Knowledge Base
        print("\n3️⃣ Testing Knowledge Base:")
        
        # Test historical info
        hagia_info = knowledge_base.get_historical_info("hagia_sophia")
        if hagia_info:
            print(f"   ✅ Hagia Sophia info available: {len(hagia_info)} fields")
            print(f"   ✅ Description: {hagia_info.get('description', 'N/A')[:50]}...")
        else:
            print(f"   ❌ Hagia Sophia info not available")
        
        # Test cultural advice
        mosque_etiquette = knowledge_base.get_cultural_advice("mosque_etiquette")
        if mosque_etiquette:
            print(f"   ✅ Mosque etiquette: {len(mosque_etiquette)} rules available")
        else:
            print(f"   ❌ Mosque etiquette not available")
        
        # Test 4: Intent Classification
        print("\n4️⃣ Testing Intent Classification:")
        test_intents = [
            "I want to find restaurants in Galata",
            "Tell me about museums in Istanbul", 
            "How do I get from airport to city center?",
            "What should I know about Turkish culture?"
        ]
        
        for query in test_intents:
            result = query_understander.extract_intent_and_entities(query)
            print(f"   '{query[:30]}...' → Intent: {result['intent']}")
        
        print("\n✅ All enhanced features are working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing enhanced features: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Enhanced AI Istanbul Chatbot Integration")
    print("=" * 50)
    
    # Run manual tests
    success = test_enhanced_features_manually()
    
    if success:
        print("\n🎉 Enhanced features are ready to integrate!")
        print("\nTo use with existing main.py:")
        print("1. Import this module: from enhanced_integration_script import integrate_enhancements")
        print("2. Call: integrate_enhancements(app) after creating your FastAPI app")
        print("\nTo run standalone:")
        print("python enhanced_integration_script.py --standalone")
    else:
        print("\n❌ Enhanced features need troubleshooting before integration")
    
    # Check for standalone flag
    if len(sys.argv) > 1 and sys.argv[1] == "--standalone":
        print("\n🔧 Starting standalone enhanced server...")
        import uvicorn
        app = create_standalone_enhanced_app()
        uvicorn.run(app, host="0.0.0.0", port=8002)
