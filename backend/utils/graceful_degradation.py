"""
Graceful Degradation for cascading fallbacks and error recovery
Provides intelligent fallback strategies when services fail
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class FallbackStrategy(Enum):
    """Fallback strategy types"""
    CACHE = "cache"              # Use cached response
    SIMPLIFIED = "simplified"    # Use simplified/basic version
    DEFAULT = "default"          # Use default response
    RETRY = "retry"              # Retry with backoff
    SKIP = "skip"                # Skip feature entirely


class ServiceStatus(Enum):
    """Service status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    DOWN = "down"


class GracefulDegradation:
    """Intelligent error recovery and fallback management"""
    
    def __init__(self):
        """Initialize graceful degradation system"""
        self.service_status = {}
        self.failure_counts = {}
        self.last_failure_time = {}
        self.fallback_usage = {}
        
        # Circuit breaker settings
        self.failure_threshold = 5
        self.recovery_timeout = 300  # 5 minutes
        self.half_open_attempts = 3
        
        logger.info("✅ GracefulDegradation initialized")
    
    def execute_with_fallback(
        self,
        primary_func: Callable,
        fallbacks: List[Tuple[FallbackStrategy, Callable]],
        service_name: str,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute function with cascading fallbacks
        
        Args:
            primary_func: Primary function to execute
            fallbacks: List of (strategy, fallback_func) tuples
            service_name: Name of the service
            *args, **kwargs: Arguments for the functions
            
        Returns:
            (result, metadata) tuple with result and execution info
        """
        metadata = {
            "service": service_name,
            "strategy_used": "primary",
            "fallback_level": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check circuit breaker
        if self._is_circuit_open(service_name):
            logger.warning(f"⚠️ Circuit breaker OPEN for {service_name}, skipping to fallbacks")
            return self._execute_fallbacks(fallbacks, service_name, metadata, *args, **kwargs)
        
        # Try primary function
        try:
            result = primary_func(*args, **kwargs)
            self._record_success(service_name)
            return result, metadata
            
        except Exception as e:
            logger.error(f"❌ Primary function failed for {service_name}: {e}")
            self._record_failure(service_name)
            
            metadata["primary_error"] = str(e)
            
            # Execute fallbacks
            return self._execute_fallbacks(fallbacks, service_name, metadata, *args, **kwargs)
    
    def _execute_fallbacks(
        self,
        fallbacks: List[Tuple[FallbackStrategy, Callable]],
        service_name: str,
        metadata: Dict[str, Any],
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute fallback strategies in order
        
        Args:
            fallbacks: List of (strategy, fallback_func) tuples
            service_name: Service name
            metadata: Metadata to update
            *args, **kwargs: Function arguments
            
        Returns:
            (result, metadata) tuple
        """
        for level, (strategy, fallback_func) in enumerate(fallbacks, 1):
            try:
                logger.info(f"🔄 Attempting fallback level {level}: {strategy.value}")
                
                result = fallback_func(*args, **kwargs)
                
                metadata["strategy_used"] = strategy.value
                metadata["fallback_level"] = level
                
                self._record_fallback_usage(service_name, strategy)
                
                logger.info(f"✅ Fallback succeeded: {strategy.value}")
                return result, metadata
                
            except Exception as e:
                logger.error(f"❌ Fallback {strategy.value} failed: {e}")
                metadata[f"fallback_{level}_error"] = str(e)
                continue
        
        # All fallbacks failed
        logger.error(f"❌ All fallbacks exhausted for {service_name}")
        metadata["strategy_used"] = "none"
        metadata["all_failed"] = True
        
        return None, metadata
    
    def _is_circuit_open(self, service_name: str) -> bool:
        """
        Check if circuit breaker is open for service
        
        Args:
            service_name: Service name
            
        Returns:
            True if circuit is open (service blocked)
        """
        status = self.service_status.get(service_name)
        
        if status != ServiceStatus.DOWN:
            return False
        
        # Check if recovery timeout has passed (half-open state)
        last_failure = self.last_failure_time.get(service_name)
        if last_failure:
            time_since_failure = (datetime.now() - last_failure).total_seconds()
            if time_since_failure > self.recovery_timeout:
                logger.info(f"🔄 Circuit breaker HALF-OPEN for {service_name}")
                self.service_status[service_name] = ServiceStatus.DEGRADED
                return False
        
        return True
    
    def _record_success(self, service_name: str):
        """Record successful service call"""
        # Reset failure count
        if service_name in self.failure_counts:
            self.failure_counts[service_name] = 0
        
        # Update status to healthy
        old_status = self.service_status.get(service_name)
        self.service_status[service_name] = ServiceStatus.HEALTHY
        
        if old_status in [ServiceStatus.DEGRADED, ServiceStatus.DOWN]:
            logger.info(f"✅ Service {service_name} recovered: {old_status.value} -> HEALTHY")
    
    def _record_failure(self, service_name: str):
        """Record service failure and update status"""
        # Increment failure count
        self.failure_counts[service_name] = self.failure_counts.get(service_name, 0) + 1
        self.last_failure_time[service_name] = datetime.now()
        
        failures = self.failure_counts[service_name]
        
        # Update service status based on failure count
        old_status = self.service_status.get(service_name, ServiceStatus.HEALTHY)
        
        if failures >= self.failure_threshold:
            self.service_status[service_name] = ServiceStatus.DOWN
            if old_status != ServiceStatus.DOWN:
                logger.error(f"🔴 Service {service_name} marked as DOWN (failures: {failures})")
        elif failures >= self.failure_threshold / 2:
            self.service_status[service_name] = ServiceStatus.FAILING
            if old_status not in [ServiceStatus.FAILING, ServiceStatus.DOWN]:
                logger.warning(f"⚠️ Service {service_name} FAILING (failures: {failures})")
        else:
            self.service_status[service_name] = ServiceStatus.DEGRADED
            if old_status == ServiceStatus.HEALTHY:
                logger.warning(f"⚠️ Service {service_name} DEGRADED (failures: {failures})")
    
    def _record_fallback_usage(self, service_name: str, strategy: FallbackStrategy):
        """Record fallback strategy usage"""
        key = f"{service_name}:{strategy.value}"
        self.fallback_usage[key] = self.fallback_usage.get(key, 0) + 1
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """
        Get status information for a service
        
        Args:
            service_name: Service name
            
        Returns:
            Dictionary with service status details
        """
        status = self.service_status.get(service_name, ServiceStatus.HEALTHY)
        failures = self.failure_counts.get(service_name, 0)
        last_failure = self.last_failure_time.get(service_name)
        
        return {
            "service": service_name,
            "status": status.value,
            "failure_count": failures,
            "last_failure": last_failure.isoformat() if last_failure else None,
            "circuit_open": self._is_circuit_open(service_name)
        }
    
    def get_all_services_status(self) -> Dict[str, Any]:
        """
        Get status for all tracked services
        
        Returns:
            Dictionary with all services status
        """
        services = set(list(self.service_status.keys()) + list(self.failure_counts.keys()))
        
        return {
            service: self.get_service_status(service)
            for service in services
        }
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """
        Get fallback usage statistics
        
        Returns:
            Dictionary with fallback usage stats
        """
        total_fallbacks = sum(self.fallback_usage.values())
        
        return {
            "total_fallbacks": total_fallbacks,
            "by_strategy": self.fallback_usage.copy()
        }
    
    def reset_service(self, service_name: str):
        """
        Manually reset a service status
        
        Args:
            service_name: Service to reset
        """
        self.service_status[service_name] = ServiceStatus.HEALTHY
        self.failure_counts[service_name] = 0
        if service_name in self.last_failure_time:
            del self.last_failure_time[service_name]
        
        logger.info(f"🔄 Service {service_name} manually reset to HEALTHY")
    
    def configure_circuit_breaker(
        self,
        failure_threshold: int = None,
        recovery_timeout: int = None,
        half_open_attempts: int = None
    ):
        """
        Configure circuit breaker settings
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_attempts: Number of test attempts in half-open state
        """
        if failure_threshold is not None:
            self.failure_threshold = failure_threshold
        if recovery_timeout is not None:
            self.recovery_timeout = recovery_timeout
        if half_open_attempts is not None:
            self.half_open_attempts = half_open_attempts
        
        logger.info(f"⚙️ Circuit breaker configured: threshold={self.failure_threshold}, "
                   f"timeout={self.recovery_timeout}s, attempts={self.half_open_attempts}")


# Common fallback implementations
def cache_fallback(cache, query: str, intent: str = "") -> Optional[Dict[str, Any]]:
    """
    Fallback to cached response
    
    Args:
        cache: Cache instance
        query: User query
        intent: Intent type
        
    Returns:
        Cached response or None
    """
    cached = cache.get(query, intent)
    if cached:
        logger.info(f"✅ Using cached response for fallback")
        return cached
    raise Exception("No cached response available")


def simplified_fallback(query: str) -> Dict[str, Any]:
    """
    Fallback to simplified response
    
    Args:
        query: User query
        
    Returns:
        Simplified response
    """
    return {
        "response": "I apologize, but I'm having trouble processing your request right now. "
                   "Please try again in a moment or rephrase your question.",
        "intent": "error_recovery",
        "confidence": 0.5,
        "simplified": True
    }


def default_fallback() -> Dict[str, Any]:
    """
    Default fallback response
    
    Returns:
        Default error response
    """
    return {
        "response": "I'm currently experiencing technical difficulties. "
                   "Please try again later.",
        "intent": "system_error",
        "confidence": 0.0,
        "default": True
    }


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing GracefulDegradation...")
    
    degradation = GracefulDegradation()
    
    # Test successful primary function
    def primary_success():
        return {"data": "success"}
    
    result, metadata = degradation.execute_with_fallback(
        primary_success,
        [],
        "test_service"
    )
    assert result == {"data": "success"}, "Should return primary result"
    assert metadata["strategy_used"] == "primary", "Should use primary"
    print("✅ Primary success test passed")
    
    # Test primary failure with fallback
    def primary_fail():
        raise Exception("Primary failed")
    
    def fallback_success():
        return {"data": "fallback"}
    
    result, metadata = degradation.execute_with_fallback(
        primary_fail,
        [(FallbackStrategy.CACHE, fallback_success)],
        "failing_service"
    )
    assert result == {"data": "fallback"}, "Should return fallback result"
    assert metadata["strategy_used"] == "cache", "Should use cache fallback"
    print("✅ Fallback success test passed")
    
    # Test circuit breaker
    for i in range(6):  # Trigger circuit breaker
        try:
            degradation.execute_with_fallback(
                primary_fail,
                [],
                "circuit_test"
            )
        except:
            pass
    
    status = degradation.get_service_status("circuit_test")
    assert status["status"] == "down", "Service should be down"
    assert status["circuit_open"], "Circuit should be open"
    print("✅ Circuit breaker test passed")
    
    # Test service status
    all_status = degradation.get_all_services_status()
    assert "test_service" in all_status, "Should track test_service"
    print("✅ Service status test passed")
    
    # Test fallback stats
    stats = degradation.get_fallback_stats()
    assert stats["total_fallbacks"] > 0, "Should have fallback usage"
    print("✅ Fallback stats test passed")
    print(f"📊 {stats}")
    
    # Test service reset
    degradation.reset_service("circuit_test")
    status = degradation.get_service_status("circuit_test")
    assert status["status"] == "healthy", "Service should be healthy after reset"
    print("✅ Service reset test passed")
    
    print("\n✅ All GracefulDegradation tests passed!")
