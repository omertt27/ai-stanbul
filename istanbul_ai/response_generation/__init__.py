"""
Response Generation Layer - Generate and format AI responses

This package contains modules for generating, formatting, and managing AI responses
with bilingual support (English/Turkish).

Week 7-8 Refactoring: Extracted from main_system.py
"""

from .language_handler import LanguageHandler
from .context_builder import ContextBuilder
from .response_formatter import ResponseFormatter
from .bilingual_responder import BilingualResponder
from .response_orchestrator import ResponseOrchestrator

__all__ = [
    'LanguageHandler',
    'ContextBuilder',
    'ResponseFormatter',
    'BilingualResponder',
    'ResponseOrchestrator'
]
