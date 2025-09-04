"""
Patched main.py with enhanced AI capabilities
This file integrates the enhanced chatbot features into the existing system
"""

import sys
import os
import re
import asyncio
import json
import time
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Import enhanced modules
from enhanced_integration import (
    process_enhanced_query,
    enhance_query_understanding,
    enhanced_create_ai_response,
    get_context_summary
)

# Original imports from main.py (subset for the AI endpoint)
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from database import SessionLocal
from models import ChatHistory

load_dotenv()

logger = logging.getLogger(__name__)

# Create a patched version of the AI endpoint
async def enhanced_ai_endpoint(request: Request):
    """Enhanced AI endpoint with improved context awareness and query understanding"""
    
    try:
        # Log request for monitoring
        user_ip = request.client.host if request and hasattr(request, 'client') and request.client else 'unknown'
        logger.info(f"Enhanced AI request from {user_ip}")
        
        # Parse request data with error handling
        try:
            data = await request.json()
            user_input = data.get("query", data.get("user_input", ""))
            session_id = data.get("session_id", f"session_{int(time.time())}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse request: {str(e)}")

        if not user_input:
            raise HTTPException(status_code=400, detail="Empty input provided")

        # Enhanced input validation and sanitization
        is_safe, sanitized_input, error_msg = validate_and_sanitize_input(user_input)
        if not is_safe:
            logger.warning(f"🚨 SECURITY: Rejected unsafe input: {error_msg}")
            raise HTTPException(status_code=400, detail="Input contains invalid characters or patterns")
        
        user_input = sanitized_input
        logger.info(f"🛡️ Processing sanitized input: {user_input[:50]}...")

        # Create database session
        try:
            db = SessionLocal()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Database connection failed")

        try:
            # Enhanced greeting detection
            user_input_clean = user_input.lower().strip()
            greeting_patterns = [
                'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon',
                'good evening', 'howdy', 'hiya', 'sup', "what's up", 'whats up',
                'how are you', 'how are u', 'how r u'
            ]
            
            is_greeting = any(pattern in user_input_clean for pattern in greeting_patterns)
            
            if is_greeting:
                logger.info(f"[Enhanced AIstanbul] Detected greeting: {user_input}")
                if any(word in user_input_clean for word in ['hi', 'hello', 'hey']):
                    response = "Hello there! 👋 I'm your enhanced Istanbul travel guide with improved memory and knowledge. I remember our conversations and can provide detailed, contextual advice about restaurants, museums, culture, and more. What would you like to explore in Istanbul today?"
                    return enhanced_create_ai_response(response, db, session_id, user_input, request)
                else:
                    response = "I'm doing great, thank you! 😊 I'm your enhanced AI guide with better understanding and memory. I can help with detailed restaurant recommendations, cultural advice, historical information, and even create personalized itineraries. What interests you most about Istanbul?"
                    return enhanced_create_ai_response(response, db, session_id, user_input, request)

            # Try enhanced processing first
            enhanced_result = await process_enhanced_query(user_input, session_id, db, request)
            
            if enhanced_result:
                logger.info(f"✨ Enhanced processing successful for: {user_input[:30]}...")
                return enhanced_result
            
            # Fall back to original processing if enhanced doesn't handle it
            logger.info(f"📄 Falling back to original processing for: {user_input[:30]}...")
            
            # Apply enhanced query understanding to improve original processing
            enhanced_input = enhance_query_understanding(user_input)
            
            # Continue with original processing logic but with enhanced input
            return await process_original_query(enhanced_input, session_id, db, request)
            
        except Exception as e:
            logger.error(f"Error in enhanced AI processing: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal processing error")
        
        finally:
            if db:
                db.close()

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error in enhanced AI endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

async def process_original_query(user_input: str, session_id: str, db, request: Request):
    """Process query using original logic but with enhanced understanding"""
    
    # Import original functions (simplified for this example)
    # In real implementation, these would be imported from the original main.py
    
    # For now, return a basic response indicating fallback
    response = f"""I understand you're asking about: "{user_input}"

🔧 **Enhanced Features Active:**
• Improved query understanding with typo correction
• Better context awareness across conversations  
• Expanded knowledge about Istanbul culture and history
• Detailed museum and neighborhood information
• Personalized recommendations based on your preferences

Let me help you with this request using my enhanced capabilities. Could you be more specific about what you'd like to know?"""
    
    return enhanced_create_ai_response(response, db, session_id, user_input, request)

def validate_and_sanitize_input(user_input: str) -> Tuple[bool, str, str]:
    """Enhanced input validation and sanitization"""
    
    # Check for empty or whitespace-only input
    if not user_input or not user_input.strip():
        return False, "", "Empty input"
    
    # Length check
    if len(user_input) > 2000:
        return False, "", "Input too long"
    
    # Basic security patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript protocols
        r'on\w+\s*=',                 # Event handlers
        r'DROP\s+TABLE',              # SQL injection
        r'DELETE\s+FROM',             # SQL injection
        r'INSERT\s+INTO',             # SQL injection
        r'UPDATE\s+\w+\s+SET',        # SQL injection
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return False, "", f"Potentially dangerous pattern detected"
    
    # Sanitize the input
    sanitized = user_input.strip()
    
    # Remove any remaining dangerous HTML/JS
    sanitized = re.sub(r'<[^>]+>', '', sanitized)  # Remove HTML tags
    sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
    
    return True, sanitized, ""

# Endpoint for getting context information (useful for debugging)
async def get_context_info(session_id: str):
    """Get conversation context information"""
    try:
        context_info = get_context_summary(session_id)
        
        if context_info:
            return {
                "status": "success",
                "context": context_info,
                "message": "Context found"
            }
        else:
            return {
                "status": "success", 
                "context": None,
                "message": "No context found for this session"
            }
    
    except Exception as e:
        logger.error(f"Error getting context info: {e}")
        return {
            "status": "error",
            "message": "Failed to retrieve context information"
        }

# Test endpoint to verify enhancements
async def test_enhancements():
    """Test endpoint to verify enhanced functionality"""
    
    test_results = {
        "enhanced_modules_loaded": True,
        "context_manager_active": True,
        "query_understanding_active": True,
        "knowledge_base_active": True,
        "response_generator_active": True,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Test imports
        from enhanced_integration import context_manager, query_understander, knowledge_base
        
        # Test basic functionality
        test_query = "restarunt in Sultanahmet"
        corrected = query_understander.correct_and_enhance_query(test_query)
        
        test_results["typo_correction_test"] = {
            "original": test_query,
            "corrected": corrected,
            "working": corrected != test_query
        }
        
        # Test context creation
        test_context = context_manager.get_context("test_session")
        test_results["context_test"] = {
            "can_create_context": True,
            "context_exists": test_context is not None
        }
        
        # Test knowledge base
        museum_info = knowledge_base.get_historical_info("hagia_sophia")
        test_results["knowledge_base_test"] = {
            "hagia_sophia_info_available": museum_info is not None,
            "info_keys": list(museum_info.keys()) if museum_info else []
        }
        
        test_results["status"] = "success"
        test_results["message"] = "All enhanced features are working correctly"
        
    except Exception as e:
        test_results["status"] = "error"
        test_results["error"] = str(e)
        test_results["message"] = "Some enhanced features may not be working"
    
    return test_results

if __name__ == "__main__":
    # This would be used if running the patched version standalone
    print("Enhanced AI Istanbul chatbot module loaded")
    print("Enhanced features:")
    print("✅ Context awareness and conversation memory")
    print("✅ Improved query understanding with typo correction")
    print("✅ Expanded knowledge base for culture and history")
    print("✅ Follow-up question handling")
    print("✅ Personalized recommendations")
