"""
ML Answering Service Client
Integrates with standalone ML API service (ml_api_service.py)
Provides graceful degradation and fallback logic
"""

import os
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any
import httpx

logger = logging.getLogger(__name__)


class MLServiceCircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failures = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.is_open = False
    
    def record_success(self):
        """Record successful request"""
        self.failures = 0
        self.is_open = False
    
    def record_failure(self):
        """Record failed request"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.is_open = True
            logger.warning(f"🔴 Circuit breaker OPEN - {self.failures} consecutive failures")
    
    def can_attempt(self) -> bool:
        """Check if request can be attempted"""
        if not self.is_open:
            return True
        
        # Check if timeout has passed (half-open state)
        if time.time() - self.last_failure_time > self.timeout:
            logger.info("🟡 Circuit breaker attempting half-open state")
            self.is_open = False
            self.failures = 0
            return True
        
        return False


class MLServiceClient:
    """Client for ML Answering Service with caching and fallback"""
    
    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: float = 30.0,
        enabled: bool = True,
        cache_ttl: int = 300
    ):
        """
        Initialize ML service client
        
        Args:
            service_url: ML service URL (default: env ML_SERVICE_URL or localhost:8000)
            timeout: Request timeout in seconds
            enabled: Whether ML service is enabled
            cache_ttl: Cache time-to-live in seconds
        """
        self.service_url = service_url or os.getenv("ML_SERVICE_URL", "http://localhost:8000")
        self.timeout = timeout
        self.enabled = enabled
        self.cache_ttl = cache_ttl
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = MLServiceCircuitBreaker()
        
        # Response cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🤖 ML Service Client initialized")
        logger.info(f"   URL: {self.service_url}")
        logger.info(f"   Enabled: {self.enabled}")
        logger.info(f"   Timeout: {self.timeout}s")
        logger.info(f"   Cache TTL: {self.cache_ttl}s")
    
    def _cache_key(self, query: str, intent: str, use_llm: bool) -> str:
        """Generate cache key for query"""
        key_str = f"{query}:{intent}:{use_llm}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached response if valid"""
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                logger.debug(f"💾 Cache hit: {key}")
                return cached['response']
            else:
                # Remove expired cache
                del self.cache[key]
        return None
    
    def _set_cache(self, key: str, response: Dict):
        """Cache response"""
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    async def get_answer(
        self,
        query: str,
        intent: str = "general",
        user_location: Optional[Dict[str, float]] = None,
        use_llm: bool = False,
        language: str = "en"
    ) -> Optional[Dict]:
        """
        Get answer from ML service with fallback
        
        Args:
            query: User query
            intent: Detected intent
            user_location: User location (lat, lon)
            use_llm: Whether to use LLM generation (slower)
            language: Response language
            
        Returns:
            ML response dict or None if unavailable
        """
        if not self.enabled:
            logger.debug("ML service disabled")
            return None
        
        # Check circuit breaker
        if not self.circuit_breaker.can_attempt():
            logger.warning("🔴 Circuit breaker OPEN - skipping ML service")
            return None
        
        # Check cache
        cache_key = self._cache_key(query, intent, use_llm)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        # Make request
        try:
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.service_url}/api/ml/chat",
                    json={
                        "query": query,
                        "intent": intent,
                        "user_location": user_location,
                        "use_llm": use_llm,
                        "language": language
                    }
                )
                
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Record success
                    self.circuit_breaker.record_success()
                    
                    # Cache result
                    self._set_cache(cache_key, result)
                    
                    logger.info(
                        f"✅ ML service response: "
                        f"{result.get('generation_method', 'unknown')} "
                        f"({elapsed:.2f}s)"
                    )
                    
                    return result
                else:
                    logger.warning(f"ML service returned {response.status_code}")
                    self.circuit_breaker.record_failure()
                    return None
        
        except httpx.TimeoutException:
            logger.warning(f"⏱️ ML service timeout (>{self.timeout}s)")
            self.circuit_breaker.record_failure()
            return None
        
        except httpx.ConnectError:
            logger.warning("🔌 ML service unavailable (connection refused)")
            self.circuit_breaker.record_failure()
            return None
        
        except Exception as e:
            logger.error(f"❌ ML service error: {e}")
            self.circuit_breaker.record_failure()
            return None
    
    async def check_health(self) -> Dict[str, Any]:
        """
        Check ML service health
        
        Returns:
            Health status dict
        """
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "healthy": False
            }
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.service_url}/health")
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "enabled": True,
                        "status": "healthy",
                        "healthy": True,
                        "details": data
                    }
                else:
                    return {
                        "enabled": True,
                        "status": "unhealthy",
                        "healthy": False,
                        "error": f"Status code: {response.status_code}"
                    }
        
        except Exception as e:
            return {
                "enabled": True,
                "status": "unavailable",
                "healthy": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive ML service status
        
        Returns:
            Status dict with health, circuit breaker, cache info
        """
        health = await self.check_health()
        
        return {
            "ml_service": {
                "url": self.service_url,
                "enabled": self.enabled,
                "healthy": health["healthy"],
                "status": health["status"]
            },
            "circuit_breaker": {
                "is_open": self.circuit_breaker.is_open,
                "failures": self.circuit_breaker.failures,
                "threshold": self.circuit_breaker.failure_threshold
            },
            "cache": {
                "size": len(self.cache),
                "ttl": self.cache_ttl
            }
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self.cache.clear()
        logger.info("🧹 Cache cleared")


# Global instance (singleton pattern)
_ml_client: Optional[MLServiceClient] = None


def get_ml_client() -> MLServiceClient:
    """Get or create ML service client singleton"""
    global _ml_client
    
    if _ml_client is None:
        _ml_client = MLServiceClient(
            service_url=os.getenv("ML_SERVICE_URL", "http://localhost:8000"),
            timeout=float(os.getenv("ML_SERVICE_TIMEOUT", "30.0")),
            enabled=os.getenv("ML_SERVICE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("ML_CACHE_TTL", "300"))
        )
    
    return _ml_client


# Convenience functions

async def get_ml_answer(
    query: str,
    intent: str = "general",
    user_location: Optional[Dict[str, float]] = None,
    use_llm: bool = False,
    language: str = "en"
) -> Optional[Dict]:
    """Convenience function to get ML answer"""
    client = get_ml_client()
    return await client.get_answer(query, intent, user_location, use_llm, language)


async def check_ml_health() -> Dict[str, Any]:
    """Convenience function to check ML service health"""
    client = get_ml_client()
    return await client.check_health()


async def get_ml_status() -> Dict[str, Any]:
    """Convenience function to get ML service status"""
    client = get_ml_client()
    return await client.get_status()
