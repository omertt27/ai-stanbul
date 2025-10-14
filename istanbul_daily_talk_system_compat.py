# Istanbul Daily Talk System - Compatibility Layer
# This file provides backward compatibility while redirecting to the new modular system

import warnings
from istanbul_ai import IstanbulDailyTalkAI

# Show deprecation warning
warnings.warn(
    "istanbul_daily_talk_system.py is deprecated. Please use 'from istanbul_ai import IstanbulDailyTalkAI' instead.",
    DeprecationWarning,
    stacklevel=2
)

# For backward compatibility, expose the new system with the old class name
class IstanbulDailyTalkAI_Old:
    """Backward compatibility wrapper for the old system"""
    
    def __init__(self):
        # Use the new modular system internally
        self._new_system = IstanbulDailyTalkAI()
    
    def __getattr__(self, name):
        # Delegate all method calls to the new system
        return getattr(self._new_system, name)

# Export the main class for compatibility
IstanbulDailyTalkAI = IstanbulDailyTalkAI_Old
