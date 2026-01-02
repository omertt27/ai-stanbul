"""
AI Chat Route Integration - Simplified Wrapper
Uses Transportation RAG System for routing (production-ready Istanbul transit)
Only handles NLP detection and chat formatting
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Import Transportation RAG (the production routing system)
try:
    from services.transportation_rag_system import get_transportation_rag, TransitRoute
    TRANSPORTATION_RAG_AVAILABLE = True
except ImportError:
    logger.error("❌ Transportation RAG System not available!")
    TRANSPORTATION_RAG_AVAILABLE = False


class AIChatRouteHandlerSimplified:
    """
    Simplified route handler that uses Transportation RAG.
    
    Responsibilities:
    1. Detect if a chat message is asking for route directions (NLP)
    2. Extract origin and destination from the message
    3. Delegate routing to Transportation RAG (production system)
    4. Format the route response for chat display
    
    Does NOT:
    - Implement its own routing algorithm (uses Transportation RAG)
    - Duplicate caching logic (Transportation RAG has it)
    - Maintain location database (Transportation RAG has 264 stations)
    """
    
    def __init__(self):
        """Initialize with Transportation RAG"""
        if not TRANSPORTATION_RAG_AVAILABLE:
            raise ImportError("Transportation RAG System is required")
        
        self.transport_rag = get_transportation_rag()
        logger.info(f"✅ AI Chat Route Handler initialized with Transportation RAG ({len(self.transport_rag.station_graph)} stations)")
    
    async def handle_route_request(
        self,
        message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Handle route request from chat
        
        Args:
            message: User's chat message
            user_context: Optional user context (language, preferences, etc.)
            
        Returns:
            Route response dict if it's a route request, None otherwise
        """
        # Step 1: NLP Detection - Is this a route request?
        if not self._is_route_request(message):
            logger.debug(f"Not a route request: {message}")
            return None
        
        logger.info(f"🗺️ Route request detected: {message}")
        
        # Step 2: Extract locations from message
        try:
            origin, destination = self._extract_locations(message, user_context)
        except Exception as e:
            logger.error(f"Failed to extract locations: {e}")
            return {
                'type': 'error',
                'message': "I couldn't understand the locations in your request. Please specify origin and destination clearly.",
                'error_code': 'LOCATION_EXTRACTION_ERROR'
            }
        
        if not origin or not destination:
            return {
                'type': 'error',
                'message': "Please specify both origin and destination. For example: 'How do I get from Taksim to Sultanahmet?'",
                'error_code': 'MISSING_LOCATIONS'
            }
        
        logger.info(f"📍 Extracted: {origin} → {destination}")
        
        # Step 3: Use Transportation RAG for routing (its job!)
        try:
            max_transfers = user_context.get('max_transfers', 3) if user_context else 3
            
            route = self.transport_rag.find_route(
                origin=origin,
                destination=destination,
                max_transfers=max_transfers
            )
            
            if not route:
                return {
                    'type': 'error',
                    'message': f"Sorry, I couldn't find a route from {origin} to {destination}. Please check the location names and try again.",
                    'error_code': 'NO_ROUTE_FOUND'
                }
            
            logger.info(f"✅ Route found: {route.total_time} min, {route.transfers} transfers")
            
        except Exception as e:
            logger.error(f"Transportation RAG routing failed: {e}", exc_info=True)
            return {
                'type': 'error',
                'message': f"Route planning failed: {str(e)}",
                'error_code': 'ROUTING_ERROR'
            }
        
        # Step 4: Format for chat display (our other job!)
        return self._format_chat_response(route, user_context)
    
    def _is_route_request(self, message: str) -> bool:
        """
        Detect if user is asking for route directions using NLP patterns
        
        Args:
            message: User's chat message
            
        Returns:
            True if this is a route/directions request
        """
        message_lower = message.lower()
        
        # Route request patterns
        route_patterns = [
            # Direct route requests
            r'\b(route|way|path|directions?)\b.*(to|from)',
            r'\b(show|find|plan|get|give)\b.*(route|directions?|way)',
            
            # "How do I" patterns
            r'\bhow\b.*(get|go|travel|reach|come).*(to|from)',
            r'\bwhere\b.*(get|go|travel|reach).*(to|from)',
            
            # Navigation patterns
            r'\b(navigate|guide)\b.*(to|from)',
            r'\b(take me|bring me|lead me)\b',
            
            # "From X to Y" patterns
            r'\bfrom\b.*\bto\b',
            r'\bto\b.*\bfrom\b',
            
            # Turkish patterns
            r'\bnasıl\b.*(gider|gidilir|ulaş)',
            r'\byol\b.*(tarif|göster)',
        ]
        
        for pattern in route_patterns:
            if re.search(pattern, message_lower):
                return True
        
        return False
    
    def _extract_locations(
        self,
        message: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract origin and destination from message
        
        Args:
            message: User's chat message
            user_context: Optional user context
            
        Returns:
            Tuple of (origin, destination)
        """
        message_lower = message.lower()
        
        # Pattern 1: "from X to Y"
        match = re.search(r'\bfrom\s+([^to]+?)\s+to\s+(.+?)(?:\?|$)', message_lower, re.IGNORECASE)
        if match:
            origin = match.group(1).strip()
            destination = match.group(2).strip()
            return self._clean_location_name(origin), self._clean_location_name(destination)
        
        # Pattern 2: "to Y from X"
        match = re.search(r'\bto\s+([^from]+?)\s+from\s+(.+?)(?:\?|$)', message_lower, re.IGNORECASE)
        if match:
            destination = match.group(1).strip()
            origin = match.group(2).strip()
            return self._clean_location_name(origin), self._clean_location_name(destination)
        
        # Pattern 3: "X to Y" (simple) - but be more careful with extraction
        # Look for the pattern but skip common routing words
        words = message_lower.split()
        common_words = {'how', 'get', 'go', 'route', 'way', 'show', 'me', 'the', 'a', 'an', 'do', 'i', 'from', 'to'}
        
        # Find "to" and extract words before and after
        if 'to' in words:
            to_idx = words.index('to')
            # Get word before 'to' (origin)
            origin_candidates = []
            for i in range(to_idx - 1, -1, -1):
                if words[i] not in common_words and len(words[i]) > 2:
                    origin_candidates.insert(0, words[i])
                elif origin_candidates:  # Stop at first common word after finding candidates
                    break
            
            # Get word after 'to' (destination)
            dest_candidates = []
            for i in range(to_idx + 1, len(words)):
                word = words[i].rstrip('?.!')  # Remove punctuation
                if word not in common_words and len(word) > 2:
                    dest_candidates.append(word)
                elif dest_candidates:  # Stop at first common word after finding candidates
                    break
            
            if origin_candidates and dest_candidates:
                origin = ' '.join(origin_candidates)
                destination = ' '.join(dest_candidates)
                return self._clean_location_name(origin), self._clean_location_name(destination)
        
        # If we have user context with current location, use it as origin
        if user_context and user_context.get('current_location'):
            # Try to find destination in message
            # Remove common words and extract location name
            words = message_lower.split()
            location_candidates = [w for w in words if len(w) > 3 and w not in 
                                 {'from', 'show', 'find', 'route', 'directions', 'how', 'get'}]
            if location_candidates:
                destination = location_candidates[-1]  # Last significant word is likely destination
                return user_context['current_location'], self._clean_location_name(destination)
        
        return None, None
    
    def _clean_location_name(self, location: str) -> str:
        """Clean and normalize location name"""
        if not location:
            return location
        
        # Remove punctuation
        location = re.sub(r'[?.!,]', '', location)
        
        # Capitalize first letter of each word (Turkish standard)
        location = location.title()
        
        return location.strip()
    
    def _format_chat_response(
        self,
        route: TransitRoute,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format Transportation RAG route for chat display
        
        Args:
            route: TransitRoute object from Transportation RAG
            user_context: Optional user context
            
        Returns:
            Formatted response dict for chat API
        """
        # Determine language
        language = user_context.get('language', 'en') if user_context else 'en'
        
        # Create natural language message
        message = self._create_natural_message(route, language)
        
        # Format steps for display
        formatted_steps = []
        for i, step in enumerate(route.steps, 1):
            if isinstance(step, dict):
                step_text = step.get('instruction', str(step))
            else:
                step_text = str(step)
            formatted_steps.append(f"{i}. {step_text}")
        
        # Build response
        return {
            'type': 'route',
            'message': message,
            'route_data': {
                'origin': route.origin,
                'destination': route.destination,
                'total_time': route.total_time,  # minutes
                'total_distance': route.total_distance,  # km
                'transfers': route.transfers,
                'lines_used': route.lines_used,
                'steps': formatted_steps,
                'confidence': route.time_confidence
            },
            'metadata': {
                'routing_method': 'transportation_rag',
                'routing_engine': 'Istanbul Transportation RAG System',
                'timestamp': datetime.now().isoformat(),
                'cache_enabled': True
            }
        }
    
    def _create_natural_message(self, route: TransitRoute, language: str = 'en') -> str:
        """Create natural language route description"""
        
        if language == 'tr':
            # Turkish message
            if route.transfers == 0:
                msg = f"🚇 {route.origin}'dan {route.destination}'a direkt {route.lines_used[0]} hattıyla gidebilirsiniz.\n\n"
            else:
                transfer_text = "aktarma" if route.transfers == 1 else "aktarma"
                msg = f"🚇 {route.origin}'dan {route.destination}'a {route.transfers} {transfer_text} ile gidebilirsiniz.\n\n"
            
            msg += f"⏱️ Süre: {route.total_time} dakika\n"
            msg += f"📏 Mesafe: {route.total_distance:.1f} km\n"
            msg += f"🚊 Hatlar: {', '.join(route.lines_used)}\n\n"
            msg += "📋 Adım adım:\n"
        else:
            # English message
            if route.transfers == 0:
                msg = f"🚇 You can go directly from {route.origin} to {route.destination} via {route.lines_used[0]} line.\n\n"
            else:
                transfer_text = "transfer" if route.transfers == 1 else "transfers"
                msg = f"🚇 Route from {route.origin} to {route.destination} with {route.transfers} {transfer_text}.\n\n"
            
            msg += f"⏱️ Duration: {route.total_time} minutes\n"
            msg += f"📏 Distance: {route.total_distance:.1f} km\n"
            msg += f"🚊 Lines: {', '.join(route.lines_used)}\n\n"
            msg += "📋 Step by step:\n"
        
        # Add steps
        for i, step in enumerate(route.steps, 1):
            if isinstance(step, dict):
                step_text = step.get('instruction', str(step))
            else:
                step_text = str(step)
            msg += f"{i}. {step_text}\n"
        
        return msg


# Singleton instance
_handler_instance = None

def get_ai_chat_route_handler() -> AIChatRouteHandlerSimplified:
    """Get singleton instance of the simplified route handler"""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = AIChatRouteHandlerSimplified()
    return _handler_instance
