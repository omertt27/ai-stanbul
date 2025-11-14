"""
LLM Handler Package
Modular architecture for Pure LLM Query Processing

This package contains the modularized components of the Pure LLM Handler,
split from the original monolithic pure_llm_handler.py for better
maintainability, testability, and scalability.

Architecture:
- core.py: Main PureLLMHandler coordinator class
- analytics.py: Performance metrics and monitoring
- signal_detection.py: Multi-intent signal detection
- context_builder.py: Database and RAG context building
- cache_manager.py: Redis caching strategies
- service_integrations.py: External service integrations
- prompt_builder.py: Prompt construction and templates
- response_handler.py: Response validation and formatting
- threshold_manager.py: Dynamic threshold learning
- ab_testing_manager.py: A/B testing framework

Author: Istanbul AI Team
Date: November 14, 2025
Version: 2.0.0 (Modularized)
"""

from .core import PureLLMHandler

__all__ = ['PureLLMHandler']
__version__ = '2.0.0'
