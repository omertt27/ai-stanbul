#!/usr/bin/env python3
"""
Istanbul Daily Talk AI System - Backward Compatibility Layer
This file maintains backward compatibility while using the new modular architecture

REFACTORED: The original 2915-line monolithic file has been broken down into:
- istanbul_ai/core/main_system.py (Main AI class)
- istanbul_ai/core/user_profile.py (User profiling)
- istanbul_ai/core/conversation_context.py (Conversation management)
- istanbul_ai/core/entity_recognizer.py (Entity recognition)
- istanbul_ai/utils/constants.py (Constants and enums)

This significantly improves:
✅ Code maintainability
✅ Testing capabilities
✅ Module reusability
✅ Development speed
✅ Bug isolation
"""

# Import everything from the modular system for backward compatibility
from istanbul_ai import (
    IstanbulDailyTalkAI,
    UserProfile,
    UserType, 
    ConversationContext,
    ConversationTone,
    IstanbulEntityRecognizer
)

# Also import constants for backward compatibility
from istanbul_ai.utils.constants import (
    ISTANBUL_DISTRICTS,
    CUISINE_TYPES,
    TRANSPORT_MODES,
    DEFAULT_RESPONSES
)

# Version information
__version__ = "2.0.0-modular"
__refactored_from__ = "2915-line monolithic file"
__architecture__ = "modular"

# Export everything that was in the original file
__all__ = [
    'IstanbulDailyTalkAI',
    'UserProfile',
    'UserType',
    'ConversationContext', 
    'ConversationTone',
    'IstanbulEntityRecognizer',
    'ISTANBUL_DISTRICTS',
    'CUISINE_TYPES',
    'TRANSPORT_MODES',
    'DEFAULT_RESPONSES'
]

print("🚀 Istanbul Daily Talk AI System v2.0 - Modular Architecture Loaded")
print("✅ Refactored from 2915-line monolithic file into maintainable modules")
