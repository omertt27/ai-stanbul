"""
Resilience Module for Production-Grade LLM System

This module provides:
1. Circuit Breaker pattern for service protection
2. Retry Strategy with exponential backoff
3. Timeout Management for all operations
4. Graceful degradation patterns

Author: AI Istanbul Team
Date: January 2025
"""

import asyncio
import time
import logging
from typing import Callable, Optional, Any, List, Type
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes to close from half-open
    timeout: float = 60.0  # Seconds to wait before half-open
    expected_exception: Type[Exception] = Exception


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: List[dict] = field(default_factory=list)
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation
    
    Protects external services from cascading failures by:
    1. Tracking failure rates
    2. Opening circuit after threshold failures
    3. Testing recovery in half-open state
    4. Closing circuit when service recovers
    
    Usage:
        cb = CircuitBreaker('MyService', failure_threshold=3)
        result = await cb.call(my_async_function, arg1, arg2)
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.name = name
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=expected_exception
        )
        self.metrics = CircuitBreakerMetrics()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        
    def _record_success(self):
        """Record successful request"""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.metrics.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"[{self.name}] Success in HALF_OPEN state ({self.success_count}/{self.config.success_threshold})")
            
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset failure count on success
            
    def _record_failure(self, exception: Exception):
        """Record failed request"""
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.metrics.last_failure_time = datetime.now()
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"[{self.name}] Failure in HALF_OPEN state, reopening circuit")
            self._transition_to_open()
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            logger.warning(f"[{self.name}] Failure recorded ({self.failure_count}/{self.config.failure_threshold}): {exception}")
            
            if self.failure_count >= self.config.failure_threshold:
                self._transition_to_open()
                
    def _transition_to_open(self):
        """Transition to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.failure_count = 0
        self.success_count = 0
        self.metrics.state_changes.append({
            'from': old_state.value,
            'to': CircuitState.OPEN.value,
            'timestamp': datetime.now().isoformat(),
            'reason': 'failure_threshold_exceeded'
        })
        logger.error(f"[{self.name}] Circuit breaker OPENED - rejecting requests for {self.config.timeout}s")
        
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.metrics.state_changes.append({
            'from': old_state.value,
            'to': CircuitState.HALF_OPEN.value,
            'timestamp': datetime.now().isoformat(),
            'reason': 'timeout_elapsed'
        })
        logger.info(f"[{self.name}] Circuit breaker HALF_OPEN - testing service recovery")
        
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.metrics.state_changes.append({
            'from': old_state.value,
            'to': CircuitState.CLOSED.value,
            'timestamp': datetime.now().isoformat(),
            'reason': 'service_recovered'
        })
        logger.info(f"[{self.name}] Circuit breaker CLOSED - normal operation resumed")
        
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open state"""
        if self.state != CircuitState.OPEN:
            return False
            
        if self.last_failure_time is None:
            return False
            
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.config.timeout
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from func
        """
        # Check if we should attempt reset
        if self._should_attempt_reset():
            self._transition_to_half_open()
            
        # Reject if circuit is open
        if self.state == CircuitState.OPEN:
            self.metrics.rejected_requests += 1
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN for {self.name}. "
                f"Service unavailable. Try again in {self.config.timeout}s"
            )
            
        # Attempt to execute
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
                
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure(e)
            raise
            
    def get_state(self) -> dict:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'rejected_requests': self.metrics.rejected_requests,
                'success_rate': (
                    self.metrics.successful_requests / self.metrics.total_requests * 100
                    if self.metrics.total_requests > 0 else 0
                ),
                'last_failure': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                'last_success': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }
        
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        logger.info(f"[{self.name}] Manual reset to CLOSED state")
        self._transition_to_closed()


class RetryStrategy:
    """
    Retry strategy with exponential backoff and jitter
    
    Features:
    1. Exponential backoff to prevent thundering herd
    2. Jitter to distribute retries
    3. Configurable max retries and delays
    4. Selective retry based on exception type
    
    Usage:
        retry = RetryStrategy(max_retries=3)
        result = await retry.execute(
            my_function,
            retryable_exceptions=[TimeoutError, ConnectionError]
        )
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter (0-25% of delay)
            delay = delay * (0.75 + random.random() * 0.25)
            
        return delay
        
    async def execute(
        self,
        func: Callable,
        *args,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Async function to execute
            *args: Positional arguments for func
            retryable_exceptions: List of exception types to retry on
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Exception: Last exception if all retries exhausted
        """
        if retryable_exceptions is None:
            retryable_exceptions = [Exception]
            
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                    
                if attempt > 0:
                    logger.info(f"Retry succeeded on attempt {attempt + 1}")
                    
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                is_retryable = any(
                    isinstance(e, exc_type) 
                    for exc_type in retryable_exceptions
                )
                
                if not is_retryable:
                    logger.warning(f"Non-retryable exception: {type(e).__name__}")
                    raise
                    
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    
        raise last_exception


class TimeoutManager:
    """
    Timeout management for operations
    
    Features:
    1. Per-operation timeout configuration
    2. Graceful timeout handling
    3. Timeout metrics tracking
    
    Usage:
        timeout_mgr = TimeoutManager()
        result = await timeout_mgr.execute(
            'llm_generation',
            my_function,
            args,
            timeout=15.0
        )
    """
    
    def __init__(self):
        self.default_timeouts = {
            'query_enhancement': 2.0,
            'cache_lookup': 0.5,
            'signal_detection': 1.0,
            'multi_intent_detection': 5.0,  # Fast timeout for multi-intent
            'context_building': 5.0,
            'database_query': 3.0,
            'rag_search': 4.0,
            'weather_api': 2.0,
            'events_api': 2.0,
            'llm_generation': 30.0,  # Increased from 15s to 30s
            'cache_storage': 1.0
        }
        self.metrics = {
            'total_operations': 0,
            'timed_out_operations': 0,
            'timeout_by_stage': {}
        }
        
    async def execute(
        self,
        operation_name: str,
        func: Callable,
        *args,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with timeout
        
        Args:
            operation_name: Name of operation (for metrics)
            func: Async function to execute
            *args: Positional arguments for func
            timeout: Timeout in seconds (uses default if None)
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            asyncio.TimeoutError: If operation times out
        """
        if timeout is None:
            timeout = self.default_timeouts.get(operation_name, 10.0)
            
        self.metrics['total_operations'] += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout
                )
            else:
                result = func(*args, **kwargs)
                
            return result
            
        except asyncio.TimeoutError:
            self.metrics['timed_out_operations'] += 1
            
            if operation_name not in self.metrics['timeout_by_stage']:
                self.metrics['timeout_by_stage'][operation_name] = 0
            self.metrics['timeout_by_stage'][operation_name] += 1
            
            logger.error(
                f"Operation '{operation_name}' timed out after {timeout}s"
            )
            raise
            
    def get_metrics(self) -> dict:
        """Get timeout metrics"""
        return {
            'total_operations': self.metrics['total_operations'],
            'timed_out_operations': self.metrics['timed_out_operations'],
            'timeout_rate': (
                self.metrics['timed_out_operations'] / self.metrics['total_operations'] * 100
                if self.metrics['total_operations'] > 0 else 0
            ),
            'timeouts_by_stage': self.metrics['timeout_by_stage'],
            'configured_timeouts': self.default_timeouts
        }
        
    def update_timeout(self, operation_name: str, timeout: float):
        """Update default timeout for an operation"""
        old_timeout = self.default_timeouts.get(operation_name, 'not set')
        self.default_timeouts[operation_name] = timeout
        logger.info(
            f"Updated timeout for '{operation_name}': {old_timeout} -> {timeout}s"
        )


class GracefulDegradation:
    """
    Graceful degradation patterns for when services fail
    
    Provides fallback strategies when primary services are unavailable
    """
    
    @staticmethod
    def get_fallback_context(service_name: str) -> dict:
        """Get fallback context when service fails"""
        fallbacks = {
            'weather': {
                'weather_info': 'Weather information temporarily unavailable. Please check weather.com for current conditions.',
                'temperature': None,
                'condition': 'unknown'
            },
            'events': {
                'events': [],
                'message': 'Event information temporarily unavailable. Please check local event calendars.'
            },
            'rag': {
                'documents': [],
                'message': 'Additional context temporarily unavailable. Providing response based on available data.'
            }
        }
        
        return fallbacks.get(service_name, {
            'message': f'{service_name} temporarily unavailable'
        })
        
    @staticmethod
    def create_degraded_response(
        original_query: str,
        available_data: dict,
        unavailable_services: List[str]
    ) -> dict:
        """Create response when operating in degraded mode"""
        return {
            'response': None,  # Will be filled by LLM
            'metadata': {
                'degraded_mode': True,
                'unavailable_services': unavailable_services,
                'available_data': list(available_data.keys()),
                'notice': (
                    f"Note: Some services are temporarily unavailable "
                    f"({', '.join(unavailable_services)}). "
                    f"Response may be limited."
                )
            }
        }


# Export all classes
__all__ = [
    'CircuitBreaker',
    'CircuitState',
    'CircuitBreakerError',
    'RetryStrategy',
    'TimeoutManager',
    'GracefulDegradation'
]
