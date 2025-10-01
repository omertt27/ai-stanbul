#!/usr/bin/env python3
"""
Intelligent Error Handling System
================================

Provides intelligent error categorization with user-friendly responses
and graceful degradation for the AI Istanbul system.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Categories of errors with different handling strategies"""
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    API_RATE_LIMIT = "api_rate_limit"
    INVALID_INPUT = "invalid_input"
    AUTHENTICATION_ERROR = "authentication_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    DATA_NOT_FOUND = "data_not_found"
    PROCESSING_ERROR = "processing_error"
    CONFIGURATION_ERROR = "configuration_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, system can continue
    MEDIUM = "medium"     # Some functionality affected
    HIGH = "high"         # Major functionality impacted
    CRITICAL = "critical" # System-wide issues

@dataclass
class ErrorContext:
    """Context information for error handling"""
    category: ErrorCategory
    severity: ErrorSeverity
    user_message: str
    technical_message: str
    suggested_actions: list
    retry_possible: bool
    fallback_available: bool
    error_code: str

class IntelligentErrorHandler:
    """Intelligent error handler with categorization and user-friendly responses"""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.fallback_responses = self._initialize_fallback_responses()
        self.error_statistics = {}
        
    def _initialize_error_patterns(self) -> Dict[str, ErrorCategory]:
        """Initialize error pattern matching"""
        return {
            # Network errors
            'connection': ErrorCategory.NETWORK_ERROR,
            'timeout': ErrorCategory.NETWORK_ERROR,
            'network': ErrorCategory.NETWORK_ERROR,
            'unreachable': ErrorCategory.NETWORK_ERROR,
            'connection refused': ErrorCategory.NETWORK_ERROR,
            
            # Database errors
            'database': ErrorCategory.DATABASE_ERROR,
            'sql': ErrorCategory.DATABASE_ERROR,
            'table': ErrorCategory.DATABASE_ERROR,
            'column': ErrorCategory.DATABASE_ERROR,
            'constraint': ErrorCategory.DATABASE_ERROR,
            'postgresql': ErrorCategory.DATABASE_ERROR,
            'psycopg2': ErrorCategory.DATABASE_ERROR,
            
            # API rate limiting
            'rate limit': ErrorCategory.API_RATE_LIMIT,
            'quota exceeded': ErrorCategory.API_RATE_LIMIT,
            'too many requests': ErrorCategory.API_RATE_LIMIT,
            'rate_limit_exceeded': ErrorCategory.API_RATE_LIMIT,
            
            # Authentication
            'authentication': ErrorCategory.AUTHENTICATION_ERROR,
            'unauthorized': ErrorCategory.AUTHENTICATION_ERROR,
            'api key': ErrorCategory.AUTHENTICATION_ERROR,
            'forbidden': ErrorCategory.AUTHENTICATION_ERROR,
            
            # Service unavailable
            'service unavailable': ErrorCategory.SERVICE_UNAVAILABLE,
            'internal server error': ErrorCategory.SERVICE_UNAVAILABLE,
            'bad gateway': ErrorCategory.SERVICE_UNAVAILABLE,
            'gateway timeout': ErrorCategory.SERVICE_UNAVAILABLE,
            
            # Data issues
            'not found': ErrorCategory.DATA_NOT_FOUND,
            'no results': ErrorCategory.DATA_NOT_FOUND,
            'empty': ErrorCategory.DATA_NOT_FOUND,
            
            # Configuration
            'import': ErrorCategory.CONFIGURATION_ERROR,
            'module': ErrorCategory.CONFIGURATION_ERROR,
            'config': ErrorCategory.CONFIGURATION_ERROR,
        }
    
    def _initialize_fallback_responses(self) -> Dict[ErrorCategory, Dict[str, Any]]:
        """Initialize fallback responses for each error category"""
        return {
            ErrorCategory.NETWORK_ERROR: {
                'user_message': "I'm having trouble connecting to external services right now. Let me provide you with some general information instead.",
                'fallback_action': 'use_cached_data',
                'retry_delay': 30
            },
            ErrorCategory.DATABASE_ERROR: {
                'user_message': "I'm experiencing some database issues, but I can still help you with general Istanbul information.",
                'fallback_action': 'use_static_data',
                'retry_delay': 60
            },
            ErrorCategory.API_RATE_LIMIT: {
                'user_message': "I've reached my limit for external data sources temporarily. Let me share what I know from my local knowledge.",
                'fallback_action': 'use_local_knowledge',
                'retry_delay': 300  # 5 minutes
            },
            ErrorCategory.INVALID_INPUT: {
                'user_message': "I didn't quite understand that request. Could you please rephrase your question about Istanbul?",
                'fallback_action': 'request_clarification',
                'retry_delay': 0
            },
            ErrorCategory.AUTHENTICATION_ERROR: {
                'user_message': "There's a temporary issue with my access to external services. I'll use my built-in knowledge to help you.",
                'fallback_action': 'use_builtin_knowledge',
                'retry_delay': 120
            },
            ErrorCategory.SERVICE_UNAVAILABLE: {
                'user_message': "Some of my services are temporarily unavailable, but I can still provide helpful information about Istanbul.",
                'fallback_action': 'degraded_service',
                'retry_delay': 180
            },
            ErrorCategory.DATA_NOT_FOUND: {
                'user_message': "I couldn't find specific information about that, but let me suggest some popular alternatives in Istanbul.",
                'fallback_action': 'suggest_alternatives',
                'retry_delay': 0
            },
            ErrorCategory.PROCESSING_ERROR: {
                'user_message': "I encountered an issue processing your request. Let me try a different approach to help you.",
                'fallback_action': 'alternative_processing',
                'retry_delay': 10
            },
            ErrorCategory.CONFIGURATION_ERROR: {
                'user_message': "There's a technical configuration issue, but I can still provide general assistance about Istanbul.",
                'fallback_action': 'basic_service',
                'retry_delay': 300
            },
            ErrorCategory.UNKNOWN_ERROR: {
                'user_message': "Something unexpected happened, but don't worry - I'm still here to help you explore Istanbul!",
                'fallback_action': 'safe_fallback',
                'retry_delay': 60
            }
        }
    
    def categorize_error(self, error: Exception, context: Optional[str] = None) -> ErrorContext:
        """Categorize an error and determine appropriate handling"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Determine category
        category = ErrorCategory.UNKNOWN_ERROR
        for pattern, cat in self.error_patterns.items():
            if pattern in error_str or pattern in error_type:
                category = cat
                break
        
        # Determine severity
        severity = self._determine_severity(category, error, context)
        
        # Get fallback information
        fallback_info = self.fallback_responses.get(category, self.fallback_responses[ErrorCategory.UNKNOWN_ERROR])
        
        # Generate error code
        error_code = f"{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create error context
        error_context = ErrorContext(
            category=category,
            severity=severity,
            user_message=fallback_info['user_message'],
            technical_message=str(error),
            suggested_actions=self._get_suggested_actions(category),
            retry_possible=category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.SERVICE_UNAVAILABLE, ErrorCategory.PROCESSING_ERROR],
            fallback_available=True,
            error_code=error_code
        )
        
        # Log error with context
        self._log_error(error_context, error, context)
        
        # Update statistics
        self._update_error_statistics(category)
        
        return error_context
    
    def _determine_severity(self, category: ErrorCategory, error: Exception, context: Optional[str] = None) -> ErrorSeverity:
        """Determine error severity based on category and context"""
        severity_mapping = {
            ErrorCategory.NETWORK_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.DATABASE_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.API_RATE_LIMIT: ErrorSeverity.LOW,
            ErrorCategory.INVALID_INPUT: ErrorSeverity.LOW,
            ErrorCategory.AUTHENTICATION_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.SERVICE_UNAVAILABLE: ErrorSeverity.HIGH,
            ErrorCategory.DATA_NOT_FOUND: ErrorSeverity.LOW,
            ErrorCategory.PROCESSING_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.CONFIGURATION_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.UNKNOWN_ERROR: ErrorSeverity.MEDIUM
        }
        
        base_severity = severity_mapping.get(category, ErrorSeverity.MEDIUM)
        
        # Escalate severity if this is a critical service context
        if context and any(critical in context.lower() for critical in ['chat', 'core', 'main']):
            if base_severity == ErrorSeverity.LOW:
                return ErrorSeverity.MEDIUM
            elif base_severity == ErrorSeverity.MEDIUM:
                return ErrorSeverity.HIGH
        
        return base_severity
    
    def _get_suggested_actions(self, category: ErrorCategory) -> list:
        """Get suggested actions for error recovery"""
        action_map = {
            ErrorCategory.NETWORK_ERROR: [
                "Check your internet connection",
                "Try again in a few moments",
                "Use offline information if available"
            ],
            ErrorCategory.DATABASE_ERROR: [
                "Try rephrasing your query",
                "Use general search instead of specific filters",
                "Contact support if issue persists"
            ],
            ErrorCategory.API_RATE_LIMIT: [
                "Wait a few minutes before trying again",
                "Use general information instead",
                "Try a different type of query"
            ],
            ErrorCategory.INVALID_INPUT: [
                "Rephrase your question",
                "Be more specific about what you're looking for",
                "Use simpler terms"
            ],
            ErrorCategory.DATA_NOT_FOUND: [
                "Try broader search terms",
                "Check spelling of locations",
                "Ask for general recommendations instead"
            ]
        }
        
        return action_map.get(category, ["Try again later", "Contact support if issue persists"])
    
    def _log_error(self, error_context: ErrorContext, error: Exception, context: Optional[str]):
        """Log error with appropriate level and context"""
        log_message = f"[{error_context.error_code}] {error_context.category.value}: {error_context.technical_message}"
        
        if context:
            log_message += f" | Context: {context}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, exc_info=True)
        elif error_context.severity == ErrorSeverity.HIGH:
            logger.error(log_message, exc_info=True)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _update_error_statistics(self, category: ErrorCategory):
        """Update error statistics for monitoring"""
        if category.value not in self.error_statistics:
            self.error_statistics[category.value] = 0
        self.error_statistics[category.value] += 1
    
    def handle_error_with_fallback(self, error: Exception, context: Optional[str] = None, 
                                  user_query: Optional[str] = None) -> Dict[str, Any]:
        """Handle error with intelligent fallback response"""
        error_context = self.categorize_error(error, context)
        
        # Generate appropriate fallback response
        fallback_response = self._generate_fallback_response(error_context, user_query)
        
        return {
            'success': False,
            'error_handled': True,
            'error_code': error_context.error_code,
            'category': error_context.category.value,
            'severity': error_context.severity.value,
            'user_message': error_context.user_message,
            'fallback_response': fallback_response,
            'retry_possible': error_context.retry_possible,
            'retry_delay': self.fallback_responses.get(error_context.category, {}).get('retry_delay', 60),
            'suggested_actions': error_context.suggested_actions
        }
    
    def _generate_fallback_response(self, error_context: ErrorContext, user_query: Optional[str] = None) -> str:
        """Generate contextual fallback response"""
        base_message = error_context.user_message
        
        # Add contextual information based on query
        if user_query:
            query_lower = user_query.lower()
            
            if any(word in query_lower for word in ['restaurant', 'food', 'eat']):
                contextual_help = "\n\nHere are some popular restaurant areas in Istanbul:\n• Sultanahmet - Traditional Turkish cuisine\n• Beyoğlu - International and modern restaurants\n• Kadıköy - Local eateries and cafes\n• Galata - Fine dining and rooftop restaurants"
            elif any(word in query_lower for word in ['museum', 'attraction', 'visit']):
                contextual_help = "\n\nPopular attractions in Istanbul include:\n• Hagia Sophia - Historic basilica and mosque\n• Topkapi Palace - Ottoman imperial palace\n• Blue Mosque - Iconic 6-minaret mosque\n• Grand Bazaar - Historic covered market"
            elif any(word in query_lower for word in ['transport', 'metro', 'bus', 'get']):
                contextual_help = "\n\nGeneral transportation tips:\n• Metro and tram are efficient for longer distances\n• Buses connect all areas of the city\n• Taxis and ride-sharing are widely available\n• Walking is great for exploring historic areas"
            elif any(word in query_lower for word in ['weather', 'climate']):
                contextual_help = "\n\nIstanbul has a temperate climate:\n• Mild, wet winters (Dec-Feb)\n• Warm, dry summers (Jun-Aug)\n• Pleasant spring and fall seasons\n• Always check current weather before visiting"
            else:
                contextual_help = "\n\nI'm still here to help with general information about Istanbul's attractions, restaurants, transportation, and culture!"
            
            return base_message + contextual_help
        
        return base_message + "\n\nFeel free to ask me anything about Istanbul - I'm here to help!"
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        total_errors = sum(self.error_statistics.values())
        
        return {
            'total_errors': total_errors,
            'error_breakdown': self.error_statistics.copy(),
            'most_common_error': max(self.error_statistics.items(), key=lambda x: x[1])[0] if self.error_statistics else None,
            'error_rate_by_category': {
                category: (count / total_errors * 100) if total_errors > 0 else 0
                for category, count in self.error_statistics.items()
            }
        }

# Global error handler instance
_error_handler = None

def get_error_handler() -> IntelligentErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = IntelligentErrorHandler()
    return _error_handler

def handle_error_intelligently(error: Exception, context: Optional[str] = None, 
                              user_query: Optional[str] = None) -> Dict[str, Any]:
    """Convenient function to handle errors intelligently"""
    error_handler = get_error_handler()
    return error_handler.handle_error_with_fallback(error, context, user_query)

if __name__ == "__main__":
    # Test the error handler
    handler = IntelligentErrorHandler()
    
    print("🧪 Testing Intelligent Error Handler...")
    
    # Test different error types
    test_errors = [
        (ConnectionError("Connection timeout"), "network_test", "Where can I eat in Sultanahmet?"),
        (Exception("column restaurants.source does not exist"), "database_query", "Find restaurants"),
        (Exception("Rate limit exceeded"), "api_call", "What's the weather?"),
        (ValueError("Invalid input provided"), "user_input", "sjdkfjskdj"),
    ]
    
    for error, context, query in test_errors:
        print(f"\n--- Testing {type(error).__name__}: {str(error)[:50]}... ---")
        result = handler.handle_error_with_fallback(error, context, query)
        print(f"Category: {result['category']}")
        print(f"Severity: {result['severity']}")
        print(f"User Message: {result['user_message'][:100]}...")
        print(f"Retry Possible: {result['retry_possible']}")
    
    # Show statistics
    print(f"\n📊 Error Statistics:")
    stats = handler.get_error_statistics()
    for category, count in stats['error_breakdown'].items():
        print(f"   {category}: {count}")
    
    print("✅ Intelligent Error Handler test complete!")
