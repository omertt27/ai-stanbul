"""
LLM Module - Modular Pure LLM Handler System

This module provides a clean, modular architecture for the Pure LLM Handler:
- core.py: Central orchestration layer
- signals.py: Multi-intent signal detection
- context.py: Smart context building
- prompts.py: Prompt engineering
- analytics.py: Analytics and monitoring
- query_enhancement.py: Query enhancement (spell check, rewrite, validate)
- conversation.py: Conversation management
- caching.py: Semantic and exact-match caching
- experimentation.py: A/B testing and threshold learning

Author: AI Istanbul Team
Date: November 2025
"""

# Import core orchestrator
from .core import PureLLMCore

# Import all individual modules for direct access
from .signals import SignalDetector
from .context import ContextBuilder
from .prompts import PromptBuilder
from .analytics import AnalyticsManager
from .caching import CacheManager
from .query_enhancement import QueryEnhancer
from .conversation import ConversationManager
from .experimentation import ExperimentationManager


def create_pure_llm_core(
    db,
    rag_service=None,
    redis_client=None,
    weather_service=None,
    events_service=None,
    map_service=None,
    routing_service=None,
    enable_cache=True,
    enable_analytics=True,
    enable_experimentation=False,
    enable_conversation=True,
    enable_query_enhancement=True
):
    """
    Factory function to create a Pure LLM Core instance.
    
    Args:
        db: Database session
        rag_service: Optional RAG service
        redis_client: Optional Redis client
        weather_service: Optional weather service
        events_service: Optional events service
        map_service: Optional map visualization service
        routing_service: Optional OSRM routing service
        enable_cache: Enable caching
        enable_analytics: Enable analytics
        enable_experimentation: Enable A/B testing
        enable_conversation: Enable conversation management
        enable_query_enhancement: Enable query enhancement
    
    Returns:
        PureLLMCore instance
    """
    from services.runpod_llm_client import get_llm_client
    
    # Get LLM client
    llm_client = get_llm_client()
    
    # Build configuration
    config = {
        'rag_service': rag_service,
        'redis_client': redis_client,
        'weather_service': weather_service,
        'events_service': events_service,
        'map_service': map_service,
        'routing_service': routing_service,
        'enable_cache': enable_cache,
        'enable_analytics': enable_analytics,
        'enable_experimentation': enable_experimentation,
        'enable_conversation': enable_conversation,
        'enable_query_enhancement': enable_query_enhancement,
    }
    
    # Create and return Pure LLM Core
    return PureLLMCore(
        llm_client=llm_client,
        db_connection=db,
        config=config
    )


__all__ = [
    # Core
    'PureLLMCore',
    'create_pure_llm_core',
    # Individual modules
    'SignalDetector',
    'ContextBuilder',
    'PromptBuilder',
    'AnalyticsManager',
    'CacheManager',
    'QueryEnhancer',
    'ConversationManager',
    'ExperimentationManager',
]

__version__ = '2.0.0'
