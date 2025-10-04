"""
Industry-Level Error Handling and Resilience System
=================================================

Enterprise-grade error handling, circuit breakers, retry logic, and
system resilience for the AI Istanbul system.
"""

import asyncio
import time
import logging
import traceback
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps
import json
import threading
from collections import defaultdict, deque
import random
import hashlib

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class RetryStrategy(Enum):
    """Retry strategies"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    RANDOM = "random"

@dataclass
class ErrorContext:
    """Error context information"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: datetime
    function_name: str
    module_name: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    monitor_window_seconds: int = 300

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True

class CircuitBreaker:
    """Circuit breaker implementation for service resilience"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.failures = deque(maxlen=100)  # Store recent failures
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"🔄 Circuit breaker {self.name} entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (time.time() - self.last_failure_time) >= self.config.timeout_seconds
    
    def _on_success(self):
        """Handle successful call"""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"✅ Circuit breaker {self.name} CLOSED (recovered)")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0  # Reset failure count on success
    
    def _on_failure(self, error: Exception):
        """Handle failed call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            self.failures.append({
                "timestamp": datetime.now().isoformat(),
                "error": str(error),
                "error_type": type(error).__name__
            })
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"🔴 Circuit breaker {self.name} OPEN (half-open test failed)")
            elif self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(f"🔴 Circuit breaker {self.name} OPEN (threshold reached)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "recent_failures": len(self.failures),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds
            }
        }

class RetryHandler:
    """Advanced retry handler with multiple strategies"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def execute(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"🔄 Retry attempt {attempt + 1}/{self.config.max_attempts} "
                                 f"after {delay:.2f}s: {str(e)}")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ All retry attempts failed: {str(e)}")
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay based on retry strategy"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay_seconds
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay_seconds * (attempt + 1)
        elif self.config.strategy == RetryStrategy.RANDOM:
            delay = random.uniform(0.1, self.config.base_delay_seconds * 2)
        else:
            delay = self.config.base_delay_seconds
        
        # Apply jitter to avoid thundering herd
        if self.config.jitter:
            jitter = random.uniform(0.8, 1.2)
            delay *= jitter
        
        # Respect maximum delay
        return min(delay, self.config.max_delay_seconds)

class IndustryErrorHandler:
    """
    Enterprise error handling system with:
    - Structured error logging
    - Circuit breakers for resilience
    - Intelligent retry mechanisms
    - Error classification and routing
    - Recovery strategies
    - Performance monitoring
    """
    
    def __init__(self):
        # Error storage and tracking
        self.errors = deque(maxlen=10000)
        self.error_counts = defaultdict(int)
        self.error_patterns = defaultdict(list)
        
        # Circuit breakers
        self.circuit_breakers = {}
        
        # Retry handlers
        self.retry_configs = {
            "database": RetryConfig(max_attempts=3, strategy=RetryStrategy.EXPONENTIAL),
            "api_call": RetryConfig(max_attempts=5, strategy=RetryStrategy.EXPONENTIAL, base_delay_seconds=0.5),
            "file_operation": RetryConfig(max_attempts=2, strategy=RetryStrategy.FIXED),
            "network": RetryConfig(max_attempts=4, strategy=RetryStrategy.EXPONENTIAL, base_delay_seconds=1.0)
        }
        
        # Error classification rules
        self.classification_rules = self._load_classification_rules()
        
        # Recovery strategies
        self.recovery_strategies = {}
        
        # Performance tracking
        self.performance_metrics = defaultdict(list)
        
        # Initialize monitoring
        try:
            from .industry_monitoring import get_monitoring_system
            self.monitoring = get_monitoring_system()
            self.monitoring_enabled = True
        except ImportError:
            self.monitoring = None
            self.monitoring_enabled = False
        
        logger.info("🛠️ Industry Error Handler initialized")
    
    def _load_classification_rules(self) -> List[Dict[str, Any]]:
        """Load error classification rules"""
        return [
            {
                "pattern": r"connection.*timeout|timeout.*connection",
                "category": "network_timeout",
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": "network",
                "recovery_action": "retry_with_backoff"
            },
            {
                "pattern": r"connection.*refused|refused.*connection",
                "category": "connection_refused",
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": "network",
                "recovery_action": "circuit_breaker"
            },
            {
                "pattern": r"database.*lock|lock.*database",
                "category": "database_lock",
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": "database",
                "recovery_action": "retry_with_delay"
            },
            {
                "pattern": r"permission.*denied|access.*denied",
                "category": "permission_error",
                "severity": ErrorSeverity.HIGH,
                "retry_strategy": None,
                "recovery_action": "escalate"
            },
            {
                "pattern": r"memory|out.*of.*memory|memoryerror",
                "category": "memory_error",
                "severity": ErrorSeverity.CRITICAL,
                "retry_strategy": None,
                "recovery_action": "immediate_alert"
            },
            {
                "pattern": r"disk.*full|no.*space|space.*left",
                "category": "disk_space",
                "severity": ErrorSeverity.CRITICAL,
                "retry_strategy": None,
                "recovery_action": "immediate_alert"
            },
            {
                "pattern": r"rate.*limit|too.*many.*requests",
                "category": "rate_limited",
                "severity": ErrorSeverity.MEDIUM,
                "retry_strategy": "api_call",
                "recovery_action": "exponential_backoff"
            },
            {
                "pattern": r"validation.*error|invalid.*input",
                "category": "validation_error",
                "severity": ErrorSeverity.LOW,
                "retry_strategy": None,
                "recovery_action": "user_feedback"
            }
        ]
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorContext:
        """
        Handle an error with comprehensive processing
        
        Args:
            error: The exception to handle
            context: Additional context information
            
        Returns:
            ErrorContext with processed error information
        """
        if context is None:
            context = {}
        
        # Create error context
        error_context = ErrorContext(
            error_id=self._generate_error_id(error, context),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=self._classify_error_severity(error),
            timestamp=datetime.now(),
            function_name=context.get("function_name", "unknown"),
            module_name=context.get("module_name", "unknown"),
            user_id=context.get("user_id"),
            request_id=context.get("request_id"),
            stack_trace=traceback.format_exc(),
            metadata=context
        )
        
        # Store error
        self.errors.append(error_context)
        self.error_counts[error_context.error_type] += 1
        
        # Classify and route error
        classification = self._classify_error(error_context)
        error_context.metadata.update(classification)
        
        # Log structured error
        self._log_structured_error(error_context)
        
        # Send to monitoring system
        if self.monitoring_enabled and self.monitoring:
            self.monitoring.create_alert(
                f"Error: {error_context.error_type}",
                self._severity_to_alert_severity(error_context.severity),
                error_context.error_message,
                "error_handler",
                {
                    "error_id": error_context.error_id,
                    "function": error_context.function_name,
                    "module": error_context.module_name,
                    "classification": classification
                }
            )
        
        # Execute recovery strategy
        self._execute_recovery_strategy(error_context, classification)
        
        return error_context
    
    def _generate_error_id(self, error: Exception, context: Dict[str, Any]) -> str:
        """Generate unique error ID"""
        error_string = f"{type(error).__name__}:{str(error)}:{context.get('function_name', '')}"
        return hashlib.md5(error_string.encode()).hexdigest()[:12]
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and content"""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ["memory", "disk", "critical", "fatal"]):
            return ErrorSeverity.CRITICAL
        if any(error_type.startswith(prefix) for prefix in ["memory", "system", "os"]):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_message for keyword in ["permission", "access", "security", "authentication"]):
            return ErrorSeverity.HIGH
        if any(error_type.startswith(prefix) for prefix in ["permission", "security", "auth"]):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_message for keyword in ["timeout", "connection", "network", "database"]):
            return ErrorSeverity.MEDIUM
        if any(error_type.startswith(prefix) for prefix in ["connection", "timeout", "network"]):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _classify_error(self, error_context: ErrorContext) -> Dict[str, Any]:
        """Classify error using rules"""
        error_text = f"{error_context.error_message} {error_context.stack_trace or ''}"
        
        for rule in self.classification_rules:
            if re.search(rule["pattern"], error_text, re.IGNORECASE):
                return {
                    "category": rule["category"],
                    "matched_rule": rule["pattern"],
                    "suggested_retry_strategy": rule["retry_strategy"],
                    "recovery_action": rule["recovery_action"]
                }
        
        # Default classification
        return {
            "category": "unclassified",
            "matched_rule": None,
            "suggested_retry_strategy": None,
            "recovery_action": "log_only"
        }
    
    def _log_structured_error(self, error_context: ErrorContext):
        """Log error with structured format"""
        log_data = {
            "error_id": error_context.error_id,
            "error_type": error_context.error_type,
            "severity": error_context.severity.value,
            "timestamp": error_context.timestamp.isoformat(),
            "function": error_context.function_name,
            "module": error_context.module_name,
            "message": error_context.error_message,
            "user_id": error_context.user_id,
            "request_id": error_context.request_id,
            "metadata": error_context.metadata
        }
        
        # Log at appropriate level
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🚨 CRITICAL ERROR: {json.dumps(log_data, indent=2)}")
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(f"❌ HIGH ERROR: {json.dumps(log_data, indent=2)}")
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"⚠️ MEDIUM ERROR: {json.dumps(log_data, indent=2)}")
        else:
            logger.info(f"ℹ️ LOW ERROR: {json.dumps(log_data, indent=2)}")
    
    def _execute_recovery_strategy(self, error_context: ErrorContext, classification: Dict[str, Any]):
        """Execute recovery strategy based on error classification"""
        recovery_action = classification.get("recovery_action", "log_only")
        
        if recovery_action == "immediate_alert":
            self._send_immediate_alert(error_context)
        elif recovery_action == "circuit_breaker":
            self._activate_circuit_breaker(error_context)
        elif recovery_action == "escalate":
            self._escalate_error(error_context)
        elif recovery_action == "user_feedback":
            self._prepare_user_feedback(error_context)
        # "log_only" and others just log (already done)
    
    def _send_immediate_alert(self, error_context: ErrorContext):
        """Send immediate alert for critical errors"""
        logger.critical(f"🚨 IMMEDIATE ALERT: {error_context.error_type} - {error_context.error_message}")
        # Here you would integrate with alerting systems like PagerDuty, email, SMS, etc.
    
    def _activate_circuit_breaker(self, error_context: ErrorContext):
        """Activate circuit breaker for service"""
        service_name = error_context.module_name or "unknown"
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                service_name, 
                CircuitBreakerConfig()
            )
        # Circuit breaker will handle future calls automatically
    
    def _escalate_error(self, error_context: ErrorContext):
        """Escalate error to higher support level"""
        logger.error(f"🆙 ESCALATED: {error_context.error_type} requires manual intervention")
        # Here you would integrate with ticketing systems, notifications, etc.
    
    def _prepare_user_feedback(self, error_context: ErrorContext):
        """Prepare user-friendly error message"""
        user_message = self._get_user_friendly_message(error_context)
        error_context.metadata["user_message"] = user_message
    
    def _get_user_friendly_message(self, error_context: ErrorContext) -> str:
        """Generate user-friendly error message"""
        category = error_context.metadata.get("category", "unknown")
        
        messages = {
            "validation_error": "Please check your input and try again.",
            "network_timeout": "Service is temporarily unavailable. Please try again in a few moments.",
            "connection_refused": "Service is currently down. Please try again later.",
            "rate_limited": "Too many requests. Please wait a moment before trying again.",
            "permission_error": "You don't have permission to perform this action.",
            "database_lock": "System is busy. Please try again in a moment.",
            "memory_error": "System is experiencing high load. Please try again later.",
            "disk_space": "System maintenance required. Please try again later."
        }
        
        return messages.get(category, "An unexpected error occurred. Please try again or contact support.")
    
    def _severity_to_alert_severity(self, severity: ErrorSeverity):
        """Convert error severity to monitoring alert severity"""
        # This would import from monitoring system
        from .industry_monitoring import AlertSeverity
        
        mapping = {
            ErrorSeverity.LOW: AlertSeverity.LOW,
            ErrorSeverity.MEDIUM: AlertSeverity.MEDIUM,
            ErrorSeverity.HIGH: AlertSeverity.HIGH,
            ErrorSeverity.CRITICAL: AlertSeverity.CRITICAL
        }
        return mapping.get(severity, AlertSeverity.MEDIUM)
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(
                name, 
                config or CircuitBreakerConfig()
            )
        return self.circuit_breakers[name]
    
    def with_retry(self, category: str = "default"):
        """Decorator for automatic retry handling"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                config = self.retry_configs.get(category, RetryConfig())
                retry_handler = RetryHandler(config)
                
                try:
                    return retry_handler.execute(func, *args, **kwargs)
                except Exception as e:
                    # Handle error through error handler
                    context = {
                        "function_name": func.__name__,
                        "module_name": func.__module__,
                        "retry_category": category
                    }
                    self.handle_error(e, context)
                    raise
            
            return wrapper
        return decorator
    
    def with_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Decorator for circuit breaker protection"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                circuit_breaker = self.get_circuit_breaker(name, config)
                
                try:
                    return circuit_breaker.call(func, *args, **kwargs)
                except Exception as e:
                    # Handle error through error handler
                    context = {
                        "function_name": func.__name__,
                        "module_name": func.__module__,
                        "circuit_breaker": name
                    }
                    self.handle_error(e, context)
                    raise
            
            return wrapper
        return decorator
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.errors if e.timestamp > cutoff_time]
        
        # Count by type and severity
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for error in recent_errors:
            error_types[error.error_type] += 1
            severity_counts[error.severity.value] += 1
            category = error.metadata.get("category", "unclassified")
            category_counts[category] += 1
        
        # Circuit breaker stats
        circuit_stats = {name: cb.get_stats() for name, cb in self.circuit_breakers.items()}
        
        return {
            "time_period_hours": hours,
            "total_errors": len(recent_errors),
            "error_types": dict(error_types),
            "severity_distribution": dict(severity_counts),
            "category_distribution": dict(category_counts),
            "circuit_breakers": circuit_stats,
            "top_error_types": sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10],
            "error_rate_per_hour": len(recent_errors) / hours if hours > 0 else 0
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get system health report based on errors"""
        recent_errors = [e for e in self.errors if e.timestamp > datetime.now() - timedelta(hours=1)]
        
        critical_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL])
        high_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.HIGH])
        
        # Determine overall health
        if critical_errors > 0:
            health_status = "critical"
        elif high_errors > 5:
            health_status = "degraded"
        elif len(recent_errors) > 50:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "overall_health": health_status,
            "timestamp": datetime.now().isoformat(),
            "recent_errors_count": len(recent_errors),
            "critical_errors_count": critical_errors,
            "high_errors_count": high_errors,
            "active_circuit_breakers": len([cb for cb in self.circuit_breakers.values() 
                                          if cb.state != CircuitState.CLOSED]),
            "recommendations": self._get_health_recommendations(health_status, recent_errors)
        }
    
    def _get_health_recommendations(self, health_status: str, recent_errors: List[ErrorContext]) -> List[str]:
        """Get health improvement recommendations"""
        recommendations = []
        
        if health_status == "critical":
            recommendations.append("Immediate investigation required for critical errors")
            recommendations.append("Consider activating emergency procedures")
        
        if health_status in ["critical", "degraded"]:
            recommendations.append("Review error patterns and implement fixes")
            recommendations.append("Check system resources and capacity")
        
        # Analyze error patterns
        error_categories = defaultdict(int)
        for error in recent_errors:
            category = error.metadata.get("category", "unclassified")
            error_categories[category] += 1
        
        top_category = max(error_categories.items(), key=lambda x: x[1])[0] if error_categories else None
        
        if top_category == "network_timeout":
            recommendations.append("Check network connectivity and service availability")
        elif top_category == "database_lock":
            recommendations.append("Review database performance and optimize queries")
        elif top_category == "memory_error":
            recommendations.append("Investigate memory usage and potential leaks")
        
        return recommendations

# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass

class RetryExhaustedException(Exception):
    """Raised when all retry attempts are exhausted"""
    pass

# Global error handler instance
_error_handler_instance = None

def get_error_handler() -> IndustryErrorHandler:
    """Get the global error handler instance"""
    global _error_handler_instance
    if _error_handler_instance is None:
        _error_handler_instance = IndustryErrorHandler()
    return _error_handler_instance

# Convenience decorators
def handle_errors(func):
    """Decorator to automatically handle errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler = get_error_handler()
            context = {
                "function_name": func.__name__,
                "module_name": func.__module__
            }
            error_context = error_handler.handle_error(e, context)
            
            # Return user-friendly message if available
            user_message = error_context.metadata.get("user_message")
            if user_message:
                raise UserFriendlyError(user_message) from e
            else:
                raise
    
    return wrapper

def with_retry(category: str = "default"):
    """Convenience retry decorator"""
    return get_error_handler().with_retry(category)

def with_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Convenience circuit breaker decorator"""
    return get_error_handler().with_circuit_breaker(name, config)

class UserFriendlyError(Exception):
    """Exception with user-friendly message"""
    pass

# Initialize error handler when module is imported
if __name__ != "__main__":
    try:
        _error_handler_instance = IndustryErrorHandler()
        logger.info("🛠️ Industry Error Handler auto-initialized")
    except Exception as e:
        logger.error(f"❌ Failed to auto-initialize error handler: {e}")

# Import guard fix
import re
