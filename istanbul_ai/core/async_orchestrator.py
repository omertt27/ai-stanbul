"""
Async Orchestrator for Istanbul AI
Handles concurrent user requests efficiently with thread/process pools
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)

class AsyncOrchestrator:
    """
    High-performance async orchestrator for Istanbul AI
    Handles concurrent user requests efficiently
    """
    
    def __init__(self, max_workers: int = 50):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Semaphore to limit concurrent GPU operations
        self.gpu_semaphore = asyncio.Semaphore(10)
        
        logger.info(f"✅ AsyncOrchestrator initialized (max_workers={max_workers})")
    
    async def initialize(self):
        """Initialize async resources"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        logger.info("✅ Async HTTP session initialized")
    
    async def shutdown(self):
        """Cleanup async resources"""
        if self.session:
            await self.session.close()
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        logger.info("✅ AsyncOrchestrator shutdown complete")
    
    async def process_user_query(self, user_id: str, query: str, 
                                 context: Dict) -> Dict:
        """
        Process user query with async operations
        Handles parallel processing for multiple components
        """
        try:
            # Import here to avoid circular imports
            from ..core.main_system import IstanbulDailyTalkAI
            from ..core.user_profile import UserProfile, UserType
            from ..core.conversation_context import ConversationContext
            
            # For now, use synchronous processing
            # TODO: Implement full async pipeline
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._process_sync,
                user_id,
                query,
                context
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._fallback_response(query)
    
    def _process_sync(self, user_id: str, query: str, context: Dict) -> Dict:
        """Synchronous processing in thread pool"""
        try:
            # Get the AI system instance from production server
            # Note: This is imported at module level now
            import sys
            ai_system = None
            
            # Try to get AI system from production_server module
            if 'production_server' in sys.modules:
                ai_system = sys.modules['production_server'].ai_system
            
            if ai_system is None:
                logger.warning("AI system not available, returning placeholder")
                return {
                    "response": f"I understand you're asking about: {query}. The AI system is initializing. Please try again in a moment.",
                    "language": "en",
                    "intent": "system_initializing"
                }
            
            # Process with actual AI system
            response = ai_system.process_query(query, user_id)
            
            return {
                "response": response.get("response", "I apologize, but I couldn't generate a response."),
                "language": response.get("language", "en"),
                "intent": response.get("intent", "unknown"),
                "confidence": response.get("confidence")
            }
            
        except Exception as e:
            logger.error(f"Error in sync processing: {e}", exc_info=True)
            return self._fallback_response(query)
    
    def _fallback_response(self, query: str) -> Dict:
        """Generate fallback response for errors"""
        return {
            "response": "I apologize, but I'm having trouble processing your request. Please try again.",
            "language": "en",
            "intent": "error"
        }


# Global orchestrator instance
_orchestrator: Optional[AsyncOrchestrator] = None

async def get_orchestrator() -> AsyncOrchestrator:
    """Get or create global orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AsyncOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator
