"""
API Router Index

Registers all API routers
"""

from . import health, auth, chat, llm, aws_test, monitoring_routes

__all__ = ['health', 'auth', 'chat', 'llm', 'aws_test', 'monitoring_routes']
