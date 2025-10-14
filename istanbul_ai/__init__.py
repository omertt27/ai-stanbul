"""
Istanbul AI System - Modular Architecture
A comprehensive AI assistant for Istanbul tourism and local recommendations.
Refactored from the monolithic istanbul_daily_talk_system.py for better maintainability.
"""

# Import main components from new modular structure
from .main_system import IstanbulDailyTalkAI
from .core.models import UserProfile, ConversationContext, ConversationTone, UserType
from .core.entity_recognition import IstanbulEntityRecognizer

__version__ = "2.0.0"
__all__ = [
    "IstanbulDailyTalkAI", 
    "UserProfile", 
    "ConversationContext", 
    "ConversationTone", 
    "UserType",
    "IstanbulEntityRecognizer"
]

__version__ = "2.0.0"
__author__ = "Istanbul AI Team"
