"""
Structured logging system for the Istanbul AI chatbot.
Provides JSON-formatted logs with context tracking and performance metrics.
"""

import logging
import json
import time
import traceback
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from functools import wraps

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None
    STRUCTLOG_AVAILABLE = False
    print("[WARNING] structlog not available, using standard logging")

class StructuredLogger:
    """
    Advanced structured logging with JSON output and context tracking.
    Falls back to standard logging if structlog is not available.
    """
    
    def __init__(self, name: str = "istanbul_ai", level: str = "INFO"):
        self.name = name
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.context_stack = []
        
        if STRUCTLOG_AVAILABLE:
            self._setup_structlog()
        else:
            self._setup_standard_logging()
    
    def _setup_structlog(self):
        """Setup structlog with JSON formatting."""
        if not structlog:
            raise ImportError("structlog not available")
            
        # Configure structlog processors
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        # Add renderer based on environment
        if os.getenv("ENVIRONMENT") == "development":
            processors.append(structlog.dev.ConsoleRenderer())
        else:
            processors.append(structlog.processors.JSONRenderer())
        
        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(self.name)
        self.is_structlog = True
    
    def _setup_standard_logging(self):
        """Setup standard logging with JSON formatting."""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        
        # Create JSON formatter
        formatter = JSONFormatter()
        
        # Create console handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.is_structlog = False
    
    def _format_message(self, level: str, message: str, **kwargs) -> Dict[str, Any]:
        """Format log message with context."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.upper(),
            "logger": self.name,
            "message": message,
            **kwargs
        }
        
        # Add context from stack
        if self.context_stack:
            log_entry["context"] = dict(self.context_stack[-1]) if self.context_stack else {}
        
        return log_entry
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if self.is_structlog:
            self.logger.debug(message, **kwargs)
        else:
            log_data = self._format_message("debug", message, **kwargs)
            self.logger.debug(json.dumps(log_data))
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        if self.is_structlog:
            self.logger.info(message, **kwargs)
        else:
            log_data = self._format_message("info", message, **kwargs)
            self.logger.info(json.dumps(log_data))
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if self.is_structlog:
            self.logger.warning(message, **kwargs)
        else:
            log_data = self._format_message("warning", message, **kwargs)
            self.logger.warning(json.dumps(log_data))
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        if self.is_structlog:
            self.logger.error(message, **kwargs)
        else:
            log_data = self._format_message("error", message, **kwargs)
            self.logger.error(json.dumps(log_data))
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        if self.is_structlog:
            self.logger.critical(message, **kwargs)
        else:
            log_data = self._format_message("critical", message, **kwargs)
            self.logger.critical(json.dumps(log_data))
    
    def log_request(self, request_id: str, method: str, path: str, **kwargs):
        """Log HTTP request."""
        self.info(
            "HTTP request received",
            request_id=request_id,
            method=method,
            path=path,
            **kwargs
        )
    
    def log_response(self, request_id: str, status_code: int, duration_ms: float, **kwargs):
        """Log HTTP response."""
        self.info(
            "HTTP response sent",
            request_id=request_id,
            status_code=status_code,
            duration_ms=round(duration_ms, 2),
            **kwargs
        )
    
    def log_ai_query(self, query: str, query_type: str, session_id: str, **kwargs):
        """Log AI query processing."""
        self.info(
            "AI query processed",
            query=query[:200] + "..." if len(query) > 200 else query,
            query_type=query_type,
            session_id=session_id,
            query_length=len(query),
            **kwargs
        )
    
    def log_cache_hit(self, cache_key: str, cache_type: str = "query"):
        """Log cache hit."""
        self.debug(
            "Cache hit",
            cache_key=cache_key,
            cache_type=cache_type
        )
    
    def log_cache_miss(self, cache_key: str, cache_type: str = "query"):
        """Log cache miss."""
        self.debug(
            "Cache miss",
            cache_key=cache_key,
            cache_type=cache_type
        )
    
    def log_rate_limit(self, identifier: str, endpoint: str, action: str):
        """Log rate limiting events."""
        self.warning(
            "Rate limit event",
            identifier=identifier,
            endpoint=endpoint,
            action=action
        )
    
    def log_error_with_traceback(self, message: str, exception: Exception, **kwargs):
        """Log error with full traceback."""
        self.error(
            message,
            error_type=type(exception).__name__,
            error_message=str(exception),
            traceback=traceback.format_exc(),
            **kwargs
        )
    
    @contextmanager
    def context(self, **kwargs):
        """Add context to subsequent log messages."""
        self.context_stack.append(kwargs)
        try:
            yield
        finally:
            if self.context_stack:
                self.context_stack.pop()
    
    def performance_timer(self, operation: str, **context_kwargs):
        """Decorator for timing operations."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    self.info(
                        f"Operation completed: {operation}",
                        operation=operation,
                        duration_ms=round(duration_ms, 2),
                        success=True,
                        **context_kwargs
                    )
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    self.error(
                        f"Operation failed: {operation}",
                        operation=operation,
                        duration_ms=round(duration_ms, 2),
                        success=False,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        **context_kwargs
                    )
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    self.info(
                        f"Operation completed: {operation}",
                        operation=operation,
                        duration_ms=round(duration_ms, 2),
                        success=True,
                        **context_kwargs
                    )
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    self.error(
                        f"Operation failed: {operation}",
                        operation=operation,
                        duration_ms=round(duration_ms, 2),
                        success=False,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        **context_kwargs
                    )
                    raise
            
            # Return appropriate wrapper based on function type
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for standard logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["traceback"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "exc_info", "exc_text", "stack_info",
                          "lineno", "funcName", "created", "msecs", "relativeCreated",
                          "thread", "threadName", "processName", "process", "getMessage"):
                log_entry[key] = value
        
        return json.dumps(log_entry)


class LoggerManager:
    """Singleton manager for structured loggers."""
    
    _instance = None
    _loggers = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_logger(self, name: str = "istanbul_ai", level: Optional[str] = None) -> StructuredLogger:
        """Get or create a structured logger."""
        if name not in self._loggers:
            log_level = level or os.getenv("LOG_LEVEL", "INFO")
            self._loggers[name] = StructuredLogger(name, log_level)
        return self._loggers[name]
    
    def set_global_level(self, level: str):
        """Set log level for all existing loggers."""
        for logger in self._loggers.values():
            logger.level = getattr(logging, level.upper(), logging.INFO)
            if hasattr(logger.logger, 'setLevel'):
                logger.logger.setLevel(logger.level)


# Global logger instance
logger_manager = LoggerManager()

def get_logger(name: str = "istanbul_ai") -> StructuredLogger:
    """Get the main application logger."""
    return logger_manager.get_logger(name)

# Create main application logger
logger = get_logger()

# Performance monitoring decorators
def log_performance(operation: str, **kwargs):
    """Decorator to log operation performance."""
    return logger.performance_timer(operation, **kwargs)

def log_api_call(endpoint: str):
    """Decorator to log API call performance."""
    return logger.performance_timer(f"API call: {endpoint}", endpoint=endpoint)

def log_db_operation(operation: str, table: Optional[str] = None):
    """Decorator to log database operation performance."""
    context = {"operation_type": "database"}
    if table:
        context["table"] = table
    return logger.performance_timer(f"DB operation: {operation}", **context)

def log_ai_operation(operation: str, model: Optional[str] = None):
    """Decorator to log AI operation performance."""
    context = {"operation_type": "ai"}
    if model:
        context["model"] = model
    return logger.performance_timer(f"AI operation: {operation}", **context)
