"""
API Router Index

Registers all API routers
"""

from . import health, auth, chat, llm

__all__ = ['health', 'auth', 'chat', 'llm']
