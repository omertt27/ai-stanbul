"""
API Resilience Module
====================

Provides robust error handling, retries, and fallback mechanisms for all transport API calls.
Ensures the system remains operational even when external APIs fail.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
from functools import wraps
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)

T = TypeVar('T')


class APIError(Exception):
    """Base exception for API errors"""
    pass


class APIConnectionError(APIError):
    """Raised when API connection fails"""
    pass


class APITimeoutError(APIError):
    """Raised when API request times out"""
    pass


class APIResponseError(APIError):
    """Raised when API returns invalid response"""
    pass


class CircuitBreaker:
    """Circuit breaker pattern for API calls"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half_open
        
    def is_open(self) -> bool:
        """Check if circuit is open"""
        if self.state == 'open':
            # Check if we should transition to half-open
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).seconds
                if time_since_failure >= self.reset_timeout:
                    self.state = 'half_open'
                    logger.info(f"Circuit breaker transitioning to half-open state")
                    return False
            return True
        return False
    
    def record_success(self):
        """Record successful API call"""
        if self.state == 'half_open':
            logger.info(f"Circuit breaker closing after successful call")
        self.state = 'closed'
        self.failure_count = 0
        self.last_failure_time = None
        
    def record_failure(self):
        """Record failed API call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry in {self.reset_timeout} seconds"
            )


class APIResilientClient:
    """Resilient API client with comprehensive error handling"""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 10.0,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Initialize resilient API client
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            timeout: Request timeout in seconds
            circuit_breaker: Optional circuit breaker instance
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.logger = logging.getLogger(__name__)
        
    async def resilient_call(
        self,
        api_func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        fallback_args: Optional[tuple] = None,
        **kwargs
    ) -> T:
        """
        Execute API call with retry logic and fallback
        
        Args:
            api_func: Async function to call
            *args: Positional arguments for api_func
            fallback: Optional fallback function if all retries fail
            fallback_args: Arguments for fallback function
            **kwargs: Keyword arguments for api_func
            
        Returns:
            Result from api_func or fallback
            
        Raises:
            APIError: If all retries fail and no fallback provided
        """
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            self.logger.warning(f"Circuit breaker is open for {api_func.__name__}, using fallback")
            if fallback:
                return await self._execute_fallback(fallback, fallback_args or ())
            raise APIError(f"Circuit breaker is open for {api_func.__name__}")
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Execute API call with timeout
                result = await asyncio.wait_for(
                    api_func(*args, **kwargs),
                    timeout=self.timeout
                )
                
                # Validate result
                if result is None:
                    raise APIResponseError(f"{api_func.__name__} returned None")
                
                # Success - record and return
                self.circuit_breaker.record_success()
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = APITimeoutError(
                    f"Timeout calling {api_func.__name__} (attempt {attempt + 1}/{self.max_retries})"
                )
                self.logger.warning(f"{last_exception}")
                
            except (ConnectionError, OSError) as e:
                last_exception = APIConnectionError(
                    f"Connection error calling {api_func.__name__}: {e} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                self.logger.warning(f"{last_exception}")
                
            except Exception as e:
                last_exception = APIError(
                    f"Error calling {api_func.__name__}: {e} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                self.logger.error(f"{last_exception}\n{traceback.format_exc()}")
            
            # Record failure
            self.circuit_breaker.record_failure()
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                self.logger.info(f"Retrying in {delay:.1f} seconds...")
                await asyncio.sleep(delay)
        
        # All retries failed - use fallback if available
        self.logger.error(
            f"All {self.max_retries} attempts failed for {api_func.__name__}. "
            f"Last error: {last_exception}"
        )
        
        if fallback:
            self.logger.info(f"Using fallback for {api_func.__name__}")
            try:
                return await self._execute_fallback(fallback, fallback_args or ())
            except Exception as e:
                self.logger.error(f"Fallback also failed: {e}")
                raise last_exception
        
        raise last_exception
    
    async def _execute_fallback(self, fallback: Callable[..., T], args: tuple) -> T:
        """Execute fallback function (async or sync)"""
        if asyncio.iscoroutinefunction(fallback):
            return await fallback(*args)
        else:
            return fallback(*args)


def validate_response(
    response: Optional[Dict[str, Any]],
    required_fields: Optional[List[str]] = None,
    source_name: str = "API"
) -> Dict[str, Any]:
    """
    Validate API response structure
    
    Args:
        response: API response to validate
        required_fields: List of required field names
        source_name: Name of the API for logging
        
    Returns:
        Validated response
        
    Raises:
        APIResponseError: If response is invalid
    """
    if response is None:
        raise APIResponseError(f"{source_name} returned None")
    
    if not isinstance(response, dict):
        raise APIResponseError(
            f"{source_name} returned invalid type: {type(response).__name__}"
        )
    
    if required_fields:
        missing_fields = [field for field in required_fields if field not in response]
        if missing_fields:
            raise APIResponseError(
                f"{source_name} missing required fields: {', '.join(missing_fields)}"
            )
    
    return response


def safe_get(
    data: Optional[Dict[str, Any]],
    *keys: str,
    default: Any = None,
    expected_type: Optional[type] = None
) -> Any:
    """
    Safely extract nested dictionary values with type checking
    
    Args:
        data: Dictionary to extract from
        *keys: Path of keys to traverse
        default: Default value if key not found or type mismatch
        expected_type: Expected type of the final value
        
    Returns:
        Extracted value or default
    """
    if data is None:
        return default
    
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    # Type checking
    if expected_type is not None and not isinstance(current, expected_type):
        logger.warning(
            f"Type mismatch for {'.'.join(keys)}: "
            f"expected {expected_type.__name__}, got {type(current).__name__}"
        )
        return default
    
    return current


def safe_list_get(
    data: Optional[List[Any]],
    index: int,
    default: Any = None,
    expected_type: Optional[type] = None
) -> Any:
    """
    Safely extract list element with bounds checking and type validation
    
    Args:
        data: List to extract from
        index: Index to extract
        default: Default value if index out of bounds or type mismatch
        expected_type: Expected type of the element
        
    Returns:
        Extracted value or default
    """
    if not isinstance(data, list) or index >= len(data) or index < -len(data):
        return default
    
    value = data[index]
    
    if expected_type is not None and not isinstance(value, expected_type):
        logger.warning(
            f"Type mismatch at index {index}: "
            f"expected {expected_type.__name__}, got {type(value).__name__}"
        )
        return default
    
    return value


def create_fallback_response(
    response_type: str,
    message: str = "Using cached/fallback data",
    timestamp: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Create a standardized fallback response
    
    Args:
        response_type: Type of response (metro, bus, ferry, weather)
        message: Fallback message
        timestamp: Response timestamp
        
    Returns:
        Fallback response dictionary
    """
    return {
        'type': response_type,
        'source': 'fallback',
        'message': message,
        'timestamp': (timestamp or datetime.now()).isoformat(),
        'is_fallback': True
    }


async def batch_resilient_calls(
    client: APIResilientClient,
    calls: List[tuple[Callable, tuple, dict]],
    return_exceptions: bool = True
) -> List[Any]:
    """
    Execute multiple API calls concurrently with resilience
    
    Args:
        client: Resilient API client
        calls: List of (function, args, kwargs) tuples
        return_exceptions: If True, exceptions are returned instead of raised
        
    Returns:
        List of results (or exceptions if return_exceptions=True)
    """
    tasks = []
    for func, args, kwargs in calls:
        fallback = kwargs.pop('fallback', None)
        fallback_args = kwargs.pop('fallback_args', None)
        
        task = client.resilient_call(
            func,
            *args,
            fallback=fallback,
            fallback_args=fallback_args,
            **kwargs
        )
        tasks.append(task)
    
    if return_exceptions:
        return await asyncio.gather(*tasks, return_exceptions=True)
    else:
        return await asyncio.gather(*tasks)


# Decorator for automatic resilience
def resilient_api_call(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: float = 10.0,
    fallback: Optional[Callable] = None
):
    """
    Decorator to make any async function resilient with automatic retries
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries
        timeout: Request timeout in seconds
        fallback: Optional fallback function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            client = APIResilientClient(
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout
            )
            
            return await client.resilient_call(
                func,
                *args,
                fallback=fallback,
                **kwargs
            )
        
        return wrapper
    return decorator


# Health check utilities
class APIHealthMonitor:
    """Monitor API health and availability"""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize health monitor
        
        Args:
            window_size: Number of recent calls to track
        """
        self.window_size = window_size
        self.call_history = []
        self.stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0
        }
        
    def record_call(self, success: bool, response_time: float):
        """Record API call result"""
        self.call_history.append({
            'success': success,
            'response_time': response_time,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(self.call_history) > self.window_size:
            self.call_history.pop(0)
        
        # Update stats
        self.stats['total_calls'] += 1
        if success:
            self.stats['successful_calls'] += 1
        else:
            self.stats['failed_calls'] += 1
        
        self._update_average_response_time()
    
    def _update_average_response_time(self):
        """Update average response time"""
        if self.call_history:
            total_time = sum(call['response_time'] for call in self.call_history)
            self.stats['average_response_time'] = total_time / len(self.call_history)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        if not self.call_history:
            return {
                'status': 'unknown',
                'success_rate': 0.0,
                'average_response_time': 0.0
            }
        
        recent_success = sum(1 for call in self.call_history if call['success'])
        success_rate = recent_success / len(self.call_history)
        
        # Determine status
        if success_rate >= 0.9:
            status = 'healthy'
        elif success_rate >= 0.7:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'success_rate': success_rate,
            'average_response_time': self.stats['average_response_time'],
            'recent_calls': len(self.call_history),
            'total_stats': self.stats
        }


if __name__ == "__main__":
    # Test the resilience module
    async def test_resilience():
        """Test resilience features"""
        
        # Simulate failing API
        call_count = 0
        
        async def failing_api():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated connection error")
            return {"status": "success", "data": "test"}
        
        async def fallback_func():
            return {"status": "fallback", "data": "fallback_data"}
        
        # Test with resilient client
        client = APIResilientClient(max_retries=3, retry_delay=0.5)
        
        try:
            result = await client.resilient_call(
                failing_api,
                fallback=fallback_func
            )
            print(f"✅ Result: {result}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test health monitor
        monitor = APIHealthMonitor()
        monitor.record_call(True, 0.5)
        monitor.record_call(True, 0.3)
        monitor.record_call(False, 2.0)
        
        health = monitor.get_health_status()
        print(f"📊 Health Status: {health}")
    
    asyncio.run(test_resilience())
