"""
Database package - SQLAlchemy Base
Note: models_registry is NOT imported here to avoid circular imports.
Import models directly from 'models' module instead.
"""
from .base import Base

__all__ = ["Base"]
