#!/usr/bin/env python3
"""
Phase 5A: Backend API Integration - Quick Start Script

This script guides you through integrating UnifiedLLMService into the FastAPI backend.

Execution time: ~2 hours
Impact: HIGH - Makes entire backend use UnifiedLLMService

Steps:
1. Initialize UnifiedLLMService in FastAPI startup
2. Update /api/chat endpoint
3. Add /health/llm endpoint
4. Add /metrics endpoint (optional)
5. Test integration

Author: AI Istanbul Team
Date: January 18, 2026
"""

import os
import sys
from pathlib import Path

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_step(number, title):
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}Step {number}: {title}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'─' * 70}{Colors.ENDC}")


def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_info(text):
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def check_file_exists(filepath):
    """Check if a file exists"""
    if Path(filepath).exists():
        print_success(f"Found: {filepath}")
        return True
    else:
        print_error(f"Not found: {filepath}")
        return False


def main():
    """Main execution flow"""
    print_header("PHASE 5A: BACKEND API INTEGRATION")
    
    print(f"{Colors.BOLD}Goal:{Colors.ENDC} Integrate UnifiedLLMService into FastAPI backend")
    print(f"{Colors.BOLD}Time:{Colors.ENDC} ~2 hours")
    print(f"{Colors.BOLD}Impact:{Colors.ENDC} HIGH - Complete backend integration\n")
    
    # Step 0: Check prerequisites
    print_step(0, "Prerequisites Check")
    
    required_files = [
        "backend/main_modular.py",
        "unified_system/services/unified_llm_service.py",
        "backend/api/chat.py",
        "backend/api/health.py",
    ]
    
    all_found = True
    for filepath in required_files:
        if not check_file_exists(filepath):
            all_found = False
    
    if not all_found:
        print_error("\nMissing required files. Please check your project structure.")
        return 1
    
    print_success("\nAll prerequisites met!")
    
    # Step 1: Guide for main_modular.py
    print_step(1, "Initialize UnifiedLLMService in FastAPI Startup")
    
    print_info("File to edit: backend/main_modular.py")
    print("\nAdd this code to the startup section (inside lifespan function):\n")
    
    print(f"{Colors.OKCYAN}")
    print("""```python
# After other imports
from unified_system.services.unified_llm_service import get_unified_llm

# Inside lifespan() startup, after startup_manager.initialize()
try:
    # Initialize UnifiedLLMService singleton
    unified_llm = get_unified_llm()
    app.state.unified_llm = unified_llm
    logger.info("✅ UnifiedLLMService initialized and ready")
    
    # Log configuration
    logger.info(f"📡 vLLM endpoint: {unified_llm.vllm_endpoint}")
    logger.info(f"🔄 Fallback: Groq API enabled")
    logger.info(f"💾 Cache size: {len(unified_llm.cache)}/{unified_llm.cache_max_size}")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize UnifiedLLMService: {e}")
    app.state.unified_llm = None
```""")
    print(f"{Colors.ENDC}")
    
    input(f"\n{Colors.WARNING}Press Enter when you've added this code...{Colors.ENDC}")
    print_success("Step 1 complete!")
    
    # Step 2: Guide for chat.py
    print_step(2, "Update /api/chat Endpoint")
    
    print_info("File to edit: backend/api/chat.py")
    print("\nAdd this dependency injection function:\n")
    
    print(f"{Colors.OKCYAN}")
    print("""```python
from fastapi import Depends, Request

async def get_unified_llm(request: Request):
    \"\"\"Dependency to inject UnifiedLLMService\"\"\"
    if not hasattr(request.app.state, 'unified_llm'):
        raise HTTPException(500, "UnifiedLLMService not initialized")
    return request.app.state.unified_llm
```""")
    print(f"{Colors.ENDC}")
    
    print("\nThen update your chat endpoint to use it:\n")
    
    print(f"{Colors.OKCYAN}")
    print("""```python
@router.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest,
    unified_llm = Depends(get_unified_llm)
):
    \"\"\"
    Chat endpoint using UnifiedLLMService
    Features: caching, circuit breaker, metrics, fallback
    \"\"\"
    try:
        # Use UnifiedLLMService for LLM generation
        response_text = await unified_llm.complete(
            prompt=request.message,
            component="api.chat",
            max_tokens=500,
            temperature=0.7
        )
        
        # Get metadata
        metrics = unified_llm.get_metrics()
        
        return {
            "response": response_text,
            "session_id": request.session_id,
            "cached": metrics.get("cache_hits", 0) > 0,
            "backend": "vllm" if not unified_llm.circuit_breaker_open else "groq",
            "latency_ms": metrics.get("avg_latency_ms", 0),
            "component": "api.chat"
        }
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(500, f"Chat failed: {str(e)}")
```""")
    print(f"{Colors.ENDC}")
    
    input(f"\n{Colors.WARNING}Press Enter when you've updated the endpoint...{Colors.ENDC}")
    print_success("Step 2 complete!")
    
    # Step 3: Guide for health check
    print_step(3, "Add /health/llm Endpoint")
    
    print_info("File to edit: backend/api/health.py")
    print("\nAdd this new health check endpoint:\n")
    
    print(f"{Colors.OKCYAN}")
    print("""```python
@router.get("/health/llm")
async def llm_health_check(request: Request):
    \"\"\"
    UnifiedLLMService health check
    
    Returns:
    - Service availability
    - Circuit breaker state
    - Cache statistics
    - Backend status
    \"\"\"
    if not hasattr(request.app.state, 'unified_llm'):
        return {
            "status": "unavailable",
            "error": "UnifiedLLMService not initialized"
        }
    
    unified_llm = request.app.state.unified_llm
    metrics = unified_llm.get_metrics()
    
    return {
        "status": "healthy",
        "service": "UnifiedLLMService",
        "backend": {
            "vllm_endpoint": unified_llm.vllm_endpoint,
            "fallback": "groq"
        },
        "circuit_breaker": {
            "state": "open" if unified_llm.circuit_breaker_open else "closed",
            "failure_count": unified_llm.circuit_breaker_failures,
            "failure_threshold": unified_llm.circuit_breaker_threshold
        },
        "cache": {
            "size": len(unified_llm.cache),
            "max_size": unified_llm.cache_max_size,
            "hit_rate": metrics.get("cache_hit_rate", 0)
        },
        "metrics": metrics
    }
```""")
    print(f"{Colors.ENDC}")
    
    input(f"\n{Colors.WARNING}Press Enter when you've added the health check...{Colors.ENDC}")
    print_success("Step 3 complete!")
    
    # Step 4: Test
    print_step(4, "Test Integration")
    
    print_info("Start the backend server:")
    print(f"\n  {Colors.BOLD}cd backend && python main.py{Colors.ENDC}\n")
    
    print_info("Then test the endpoints:")
    
    print(f"\n{Colors.OKCYAN}# Test health check{Colors.ENDC}")
    print("  curl http://localhost:8000/health/llm | jq\n")
    
    print(f"{Colors.OKCYAN}# Test chat endpoint{Colors.ENDC}")
    print("""  curl -X POST http://localhost:8000/api/chat \\
    -H "Content-Type: application/json" \\
    -d '{"message": "Hello Istanbul!", "session_id": "test123"}' | jq\n""")
    
    print_info("Expected response should include:")
    print("  - response: (AI response text)")
    print("  - cached: true/false")
    print("  - backend: 'vllm' or 'groq'")
    print("  - latency_ms: (response time)")
    
    print_success("\n✨ Backend API Integration Complete!")
    
    # Summary
    print_header("NEXT STEPS")
    
    print(f"{Colors.BOLD}What you've accomplished:{Colors.ENDC}")
    print("  ✅ UnifiedLLMService initialized in FastAPI")
    print("  ✅ /api/chat endpoint uses UnifiedLLMService")
    print("  ✅ /health/llm endpoint for monitoring")
    print("  ✅ Automatic caching and circuit breaker protection")
    
    print(f"\n{Colors.BOLD}What's next (Phase 5B):{Colors.ENDC}")
    print("  🔄 Update frontend chatService.js")
    print("  🔄 Add streaming support")
    print("  🔄 Update UI to show cache/backend badges")
    
    print(f"\n{Colors.BOLD}Documentation:{Colors.ENDC}")
    print("  📖 See UNIFIED_LLM_BACKEND_FRONTEND_INTEGRATION.md for details")
    print("  📖 See UNIFIED_LLM_PRODUCTION_ROADMAP.md for full plan")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 Great work! Backend is now fully integrated with UnifiedLLMService!{Colors.ENDC}\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}⚠️  Integration interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}❌ Error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
