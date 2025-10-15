#!/usr/bin/env python3
"""
Enhanced Transportation System - Main System Integration
=======================================================

This module provides the exact interface expected by the main AI system
while routing to the new modular transportation system.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional

# Import from the integration wrapper
from transportation_integration_wrapper import (
    TransportationQueryProcessor,
    GPSLocation,
    MODULAR_SYSTEM_AVAILABLE,
    create_ml_enhanced_transportation_system
)

# Re-export everything the main system expects
__all__ = [
    'TransportationQueryProcessor',
    'GPSLocation', 
    'create_ml_enhanced_transportation_system'
]

# Initialize logging
logger = logging.getLogger(__name__)

if MODULAR_SYSTEM_AVAILABLE:
    logger.info("🚇 Enhanced transportation system initialized with NEW modular architecture")
else:
    logger.warning("⚠️ Enhanced transportation system using legacy fallback")


def get_enhanced_transportation_system():
    """Legacy compatibility function"""
    processor = TransportationQueryProcessor()
    return processor.system.get('enhanced_system') if hasattr(processor, 'system') else None
