"""
Istanbul AI System - Modular Architecture
Refactored from the monolithic istanbul_daily_talk_system.py for better maintainability
"""

# Maintain backward compatibility by importing the main class
from .core.main_system import IstanbulDailyTalkAI
from .core.user_profile import UserProfile
from .core.conversation_context import ConversationContext
from .core.entity_recognizer import IstanbulEntityRecognizer
from .utils.constants import UserType, ConversationTone

# Export main classes for backward compatibility
__all__ = [
    'IstanbulDailyTalkAI',
    'UserProfile', 
    'UserType',
    'ConversationContext',
    'ConversationTone',
    'IstanbulEntityRecognizer'
]

__version__ = "2.0.0"
__author__ = "Istanbul AI Team"
