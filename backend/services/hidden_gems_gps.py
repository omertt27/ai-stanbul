"""
Hidden Gems GPS Handler - Stub Module

This is a placeholder module to prevent import errors.
The actual hidden gems functionality is handled by other services.
"""

import logging

logger = logging.getLogger(__name__)


class HiddenGemsHandler:
    """Stub handler for hidden gems GPS functionality."""
    
    def __init__(self):
        logger.info("✅ Hidden Gems GPS Handler initialized (stub)")
    
    def handle_chat_message(
        self,
        message: str,
        user_location: dict = None,
        session_id: str = None
    ) -> dict:
        """
        Handle a chat message for hidden gems discovery.
        
        Returns a dict with type='not_handled' to indicate this handler
        doesn't process the message (fallback to other handlers).
        """
        return {'type': 'not_handled'}
    
    def get_nearby_gems(self, lat: float, lon: float, radius_km: float = 2.0) -> list:
        """Get nearby hidden gems - stub returns empty list."""
        return []


# Singleton instance
hidden_gems_handler = HiddenGemsHandler()
