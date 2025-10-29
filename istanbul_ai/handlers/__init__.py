"""
Handler Layer - Response handlers for different query types

This package contains specialized handlers for processing different types of user queries.
Each handler focuses on a specific domain (restaurants, attractions, transportation, etc.).

Week 5-6 Refactoring: Extracted from main_system.py
"""

from .base_handler import BaseHandler

__all__ = ['BaseHandler']
