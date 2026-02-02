"""
RunPod Pod Warmer Service

Keeps RunPod serverless pods warm to prevent cold starts.
Sends periodic health checks and lightweight requests to maintain GPU memory.

Author: AI Istanbul Team
Date: February 2025
"""

import asyncio
import logging
import time
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RunPodWarmer:
    """Service to keep RunPod pods warm and prevent cold starts"""
    
    def __init__(self, warmup_interval: int = 30):
        """
        Initialize RunPod warmer
        
        Args:
            warmup_interval: Seconds between warmup requests (default: 30s)
        """
        self.warmup_interval = warmup_interval
        self.is_running = False
        self.last_warmup = None
        self.warmup_task = None
        self.llm_client = None
        
    async def start(self):
        """Start the warming service"""
        if self.is_running:
            logger.warning("⚠️ RunPod warmer already running")
            return
            
        try:
            from services.runpod_llm_client import RunPodLLMClient
            self.llm_client = RunPodLLMClient()
            
            if not self.llm_client.enabled:
                logger.warning("⚠️ LLM client not enabled - skipping RunPod warming")
                return
                
            self.is_running = True
            self.warmup_task = asyncio.create_task(self._warmup_loop())
            logger.info(f"🔥 RunPod warmer started (interval: {self.warmup_interval}s)")
            
        except Exception as e:
            logger.error(f"❌ Failed to start RunPod warmer: {e}")
            self.is_running = False
    
    async def stop(self):
        """Stop the warming service"""
        if not self.is_running:
            return
            
        self.is_running = False
        if self.warmup_task:
            self.warmup_task.cancel()
            try:
                await self.warmup_task
            except asyncio.CancelledError:
                pass
            
        logger.info("🔥 RunPod warmer stopped")
    
    async def _warmup_loop(self):
        """Main warming loop"""
        while self.is_running:
            try:
                await self._perform_warmup()
                await asyncio.sleep(self.warmup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Warmup error: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _perform_warmup(self):
        """Perform a warmup request"""
        if not self.llm_client:
            return
            
        try:
            # Start timing
            start_time = time.time()
            
            # Try health check first (lightweight)
            health_result = await self.llm_client.health_check()
            
            if health_result.get("status") == "healthy":
                # Health check passed - now do a minimal generation to keep GPU warm
                warmup_prompt = "Hi"  # Very short prompt for speed
                
                try:
                    # Use minimal settings for warmup
                    response = await self.llm_client.generate_chat_response(
                        prompt=warmup_prompt,
                        max_tokens=10,  # Very small response
                        temperature=0.1  # Low creativity for consistency
                    )
                    
                    duration = time.time() - start_time
                    
                    if response and response.get("generated_text"):
                        logger.debug(f"🔥 RunPod warmed successfully ({duration:.2f}s)")
                        self.last_warmup = datetime.now()
                    else:
                        logger.warning("⚠️ Warmup generation failed - no text generated")
                        
                except Exception as gen_error:
                    # If generation fails, at least the health check worked
                    duration = time.time() - start_time
                    logger.warning(f"⚠️ Warmup generation failed ({duration:.2f}s): {gen_error}")
                    self.last_warmup = datetime.now()  # Still count as warmup
                    
            else:
                logger.warning(f"⚠️ RunPod health check failed: {health_result}")
                
        except Exception as e:
            logger.error(f"❌ RunPod warmup failed: {e}")
    
    def get_status(self) -> dict:
        """Get warmer status"""
        return {
            "running": self.is_running,
            "warmup_interval": self.warmup_interval,
            "last_warmup": self.last_warmup.isoformat() if self.last_warmup else None,
            "time_since_warmup": (
                (datetime.now() - self.last_warmup).total_seconds() 
                if self.last_warmup else None
            )
        }
    
    async def manual_warmup(self) -> dict:
        """Manually trigger a warmup (for testing)"""
        if not self.is_running:
            return {"error": "Warmer not running"}
            
        start_time = time.time()
        await self._perform_warmup()
        duration = time.time() - start_time
        
        return {
            "success": True,
            "duration": round(duration, 2),
            "last_warmup": self.last_warmup.isoformat() if self.last_warmup else None
        }


# Global warmer instance
_runpod_warmer: Optional[RunPodWarmer] = None


def get_runpod_warmer() -> RunPodWarmer:
    """Get the global RunPod warmer instance"""
    global _runpod_warmer
    if _runpod_warmer is None:
        # Use 30-second interval (good balance between effectiveness and resource usage)
        _runpod_warmer = RunPodWarmer(warmup_interval=30)
    return _runpod_warmer


async def start_runpod_warming():
    """Start the global RunPod warming service"""
    warmer = get_runpod_warmer()
    await warmer.start()


async def stop_runpod_warming():
    """Stop the global RunPod warming service"""
    warmer = get_runpod_warmer()
    await warmer.stop()


def is_runpod_warm() -> bool:
    """Check if RunPod was recently warmed (within last 60 seconds)"""
    warmer = get_runpod_warmer()
    if not warmer.last_warmup:
        return False
        
    time_since = datetime.now() - warmer.last_warmup
    return time_since.total_seconds() < 60  # Consider warm if warmed within 1 minute
