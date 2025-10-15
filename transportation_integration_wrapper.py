#!/usr/bin/env python3
"""
Transportation System Integration Wrapper
=========================================

This module provides a compatibility wrapper to integrate the new modular
transportation system with the existing main AI system.

It maintains backward compatibility while leveraging the new modular architecture.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional

# Import the new modular transportation system
try:
    from transportation import (
        ComprehensiveTransportProcessor,
        GPSTransportationQueryProcessor, 
        GPSLocation,
        create_transportation_system
    )
    MODULAR_SYSTEM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"⚠️ New modular transportation system not available: {e}")
    MODULAR_SYSTEM_AVAILABLE = False

# Fallback to legacy system if modular not available
LEGACY_SYSTEM_AVAILABLE = False
if not MODULAR_SYSTEM_AVAILABLE:
    try:
        from enhanced_transportation_system import (
            ComprehensiveTransportProcessor,
            GPSTransportationQueryProcessor,
            GPSLocation
        )
        LEGACY_SYSTEM_AVAILABLE = True
    except ImportError:
        LEGACY_SYSTEM_AVAILABLE = False


class TransportationQueryProcessor:
    """
    Compatibility wrapper for the main AI system.
    
    This class maintains the expected interface while using the new modular
    transportation system underneath.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if MODULAR_SYSTEM_AVAILABLE:
            self.logger.info("🚇 Using NEW modular transportation system")
            self.system = create_transportation_system()
            self.comprehensive_processor = self.system['comprehensive_processor']
            self.gps_processor = self.system['gps_query_processor']
            self.use_modular = True
        elif LEGACY_SYSTEM_AVAILABLE:
            self.logger.info("🚇 Using LEGACY transportation system")
            self.comprehensive_processor = ComprehensiveTransportProcessor()
            self.gps_processor = GPSTransportationQueryProcessor()
            self.use_modular = False
        else:
            self.logger.error("❌ No transportation system available!")
            self.comprehensive_processor = None
            self.gps_processor = None
            self.use_modular = False
    
    def process_transportation_query_sync(
        self,
        user_input: str,
        entities: Dict[str, Any] = None,
        user_profile: Any = None,
        user_gps: Optional[GPSLocation] = None
    ) -> str:
        """
        Synchronous wrapper for transportation query processing.
        
        This method maintains compatibility with the main system's expectations
        while routing to the appropriate processor.
        """
        
        if not self.comprehensive_processor:
            return "⚠️ Transportation system not available. Please try again later."
        
        try:
            # Route to GPS processor if GPS data is provided
            if user_gps and self.gps_processor:
                return self._run_async_method(
                    self.gps_processor.process_gps_transportation_query,
                    user_input=user_input,
                    user_gps=user_gps,
                    entities=entities,
                    user_profile=user_profile
                )
            
            # Otherwise use comprehensive processor
            return self._run_async_method(
                self.comprehensive_processor.process_transportation_query,
                user_input=user_input,
                entities=entities,
                user_profile=user_profile
            )
            
        except Exception as e:
            self.logger.error(f"Error in transportation query processing: {e}")
            return f"⚠️ Error processing transportation query: {str(e)}"
    
    def _run_async_method(self, async_method, **kwargs) -> str:
        """Run an async method synchronously"""
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, async_method(**kwargs))
                    return future.result(timeout=30)  # 30 second timeout
            else:
                # If no loop is running, we can run directly
                return loop.run_until_complete(async_method(**kwargs))
                
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(async_method(**kwargs))
        except Exception as e:
            self.logger.error(f"Error running async method: {e}")
            return f"⚠️ Error processing request: {str(e)}"
    
    async def process_transportation_query_async(
        self,
        user_input: str,
        entities: Dict[str, Any] = None,
        user_profile: Any = None,
        user_gps: Optional[GPSLocation] = None
    ) -> str:
        """
        Async version for better performance when called from async contexts
        """
        
        if not self.comprehensive_processor:
            return "⚠️ Transportation system not available. Please try again later."
        
        try:
            # Route to GPS processor if GPS data is provided
            if user_gps and self.gps_processor:
                return await self.gps_processor.process_gps_transportation_query(
                    user_input=user_input,
                    user_gps=user_gps,
                    entities=entities,
                    user_profile=user_profile
                )
            
            # Otherwise use comprehensive processor
            return await self.comprehensive_processor.process_transportation_query(
                user_input=user_input,
                entities=entities,
                user_profile=user_profile
            )
            
        except Exception as e:
            self.logger.error(f"Error in async transportation query processing: {e}")
            return f"⚠️ Error processing transportation query: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current transportation system"""
        return {
            'modular_system_available': MODULAR_SYSTEM_AVAILABLE,
            'legacy_system_available': LEGACY_SYSTEM_AVAILABLE,
            'using_modular': self.use_modular,
            'processors_available': {
                'comprehensive': self.comprehensive_processor is not None,
                'gps': self.gps_processor is not None
            }
        }


# Re-export for backward compatibility
__all__ = [
    'TransportationQueryProcessor',
    'GPSLocation',
    'MODULAR_SYSTEM_AVAILABLE'
]


def create_ml_enhanced_transportation_system():
    """
    Legacy compatibility function - returns the new modular system
    """
    if MODULAR_SYSTEM_AVAILABLE:
        return create_transportation_system()
    else:
        # Return a basic fallback
        return {
            'processor': TransportationQueryProcessor(),
            'available': MODULAR_SYSTEM_AVAILABLE or LEGACY_SYSTEM_AVAILABLE
        }
