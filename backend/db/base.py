"""
Central SQLAlchemy Base - Single source of truth for all models
This module ensures we only have ONE declarative_base() instance
"""
from sqlalchemy.orm import declarative_base

# Single Base instance for the entire application
Base = declarative_base()

__all__ = ["Base"]
