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

__all__ = [
    # Core
    'PureLLMCore',
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
