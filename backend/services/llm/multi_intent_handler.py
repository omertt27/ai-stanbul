"""
Multi-Intent Handler for AI-stanbul
Handles queries with multiple intentions (e.g., "route to Hagia Sophia and show restaurants nearby")

Phase 4.3 of LLM Enhancement Proposal
"""

from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Single intent extracted from query"""
    type: str  # route, restaurant, info, etc.
    priority: int  # 1 = primary, 2 = secondary, etc.
    entities: Dict[str, Any]  # origin, destination, preferences, etc.
    confidence: float
    raw_text: str  # Part of query related to this intent


@dataclass
class MultiIntentResult:
    """Result from multi-intent handling"""
    intents: List[Intent]
    responses: Dict[str, Any]  # intent_type -> response
    synthesized_response: str
    metadata: Dict[str, Any]


class MultiIntentHandler:
    """
    Detects and handles multiple intents in a single query.
    
    Examples:
    - "route to Hagia Sophia and show restaurants nearby"
      → Intent 1: route (Hagia Sophia)
      → Intent 2: restaurants (near Hagia Sophia)
    
    - "how do I get to Taksim, is there a museum there?"
      → Intent 1: route (Taksim)
      → Intent 2: info (museums in Taksim)
    
    - "show me blue mosque and closest metro station"
      → Intent 1: info (Blue Mosque)
      → Intent 2: transport (metro near Blue Mosque)
    """
    
    def __init__(self, llm_service, route_handler, restaurant_handler, info_handler):
        self.llm_service = llm_service
        self.route_handler = route_handler
        self.restaurant_handler = restaurant_handler
        self.info_handler = info_handler
        
        # Handler registry
        self.handlers = {
            'route': self.route_handler,
            'restaurant': self.restaurant_handler,
            'info': self.info_handler,
            'museum': self.info_handler,
            'transport': self.route_handler,
        }
        
        # Map response integrator for visualization
        self.map_integrator = None
        self._init_map_integrator()
    
    def _init_map_integrator(self):
        """Initialize map response integrator"""
        try:
            from services.llm.map_response_integrator import get_map_response_integrator
            self.map_integrator = get_map_response_integrator()
            logger.info("✅ Multi-Intent Handler initialized with Map Integrator")
        except Exception as e:
            logger.warning(f"⚠️ Map integrator not available: {e}")
            self.map_integrator = None
    
    async def detect_multiple_intents(
        self,
        query: str,
        user_context: Dict[str, Any],
        gps: Optional[Dict] = None
    ) -> List[Intent]:
        """
        Use LLM to detect multiple intents in a query.
        
        Args:
            query: User query
            user_context: User context (history, preferences)
            gps: User GPS coordinates
            
        Returns:
            List of Intent objects, ordered by priority
        """
        
        prompt = f"""
Analyze this Istanbul travel query for MULTIPLE intents:

Query: "{query}"
User GPS: {gps}
User Context: {user_context.get('recent_queries', [])}

Detect ALL distinct intentions in the query. Common patterns:
- Route + nearby search: "route to X and show Y nearby"
- Info + transport: "tell me about X and how to get there"
- Multiple locations: "show me X and Y"
- Chained requests: "route to X, then show restaurants"

For EACH intent, extract:
1. Intent type: [route, info, restaurant, museum, transport, event, weather]
2. Priority: 1 (primary) or 2 (secondary)
3. Entities:
   - origin: (if route intent)
   - destination: (if route intent)
   - location: (for info/restaurant/etc)
   - preferences: (cheap, wheelchair accessible, etc)
   - radius: (nearby, close, within X km)
4. Confidence: 0-1
5. Raw text: The part of query related to this intent

Return JSON array of intents, ordered by priority.

Example:
Query: "route to Hagia Sophia and show restaurants nearby"
Output:
{{
  "intents": [
    {{
      "type": "route",
      "priority": 1,
      "entities": {{
        "destination": "Hagia Sophia",
        "origin": null
      }},
      "confidence": 0.95,
      "raw_text": "route to Hagia Sophia"
    }},
    {{
      "type": "restaurant",
      "priority": 2,
      "entities": {{
        "location": "Hagia Sophia",
        "radius": "nearby",
        "preferences": []
      }},
      "confidence": 0.90,
      "raw_text": "show restaurants nearby"
    }}
  ]
}}

Now analyze: "{query}"
"""
        
        try:
            response = await self.llm_service.structured_call(
                prompt=prompt,
                response_format="json"
            )
            
            intents_data = response.get('intents', [])
            
            intents = []
            for intent_data in intents_data:
                intent = Intent(
                    type=intent_data['type'],
                    priority=intent_data['priority'],
                    entities=intent_data.get('entities', {}),
                    confidence=intent_data.get('confidence', 0.8),
                    raw_text=intent_data.get('raw_text', query)
                )
                intents.append(intent)
            
            # Sort by priority
            intents.sort(key=lambda x: x.priority)
            
            logger.info(f"Detected {len(intents)} intents in query: {query}")
            for intent in intents:
                logger.debug(f"  - Intent {intent.priority}: {intent.type} (confidence: {intent.confidence})")
            
            return intents
            
        except Exception as e:
            logger.error(f"Error detecting multiple intents: {e}")
            # Fallback: treat as single intent
            return [Intent(
                type='general',
                priority=1,
                entities={},
                confidence=0.5,
                raw_text=query
            )]
    
    async def handle_intent(
        self,
        intent: Intent,
        user_context: Dict[str, Any],
        gps: Optional[Dict] = None,
        previous_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle a single intent using the appropriate handler.
        
        Args:
            intent: Intent to handle
            user_context: User context
            gps: User GPS
            previous_results: Results from previous intents (for context)
            
        Returns:
            Handler response
        """
        
        handler = self.handlers.get(intent.type)
        
        if not handler:
            logger.warning(f"No handler for intent type: {intent.type}")
            return {
                'error': f"Cannot handle intent type: {intent.type}",
                'intent': intent.type
            }
        
        try:
            # Build handler request with context from previous intents
            request_data = self._build_handler_request(
                intent,
                user_context,
                gps,
                previous_results
            )
            
            logger.info(f"Handling intent: {intent.type} with priority {intent.priority}")
            
            # Call the appropriate handler
            if intent.type == 'route':
                result = await handler.handle_route(**request_data)
            elif intent.type == 'restaurant':
                result = await handler.handle_restaurant(**request_data)
            elif intent.type in ['info', 'museum']:
                result = await handler.handle_info(**request_data)
            else:
                result = await handler.handle(**request_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling intent {intent.type}: {e}")
            return {
                'error': str(e),
                'intent': intent.type
            }
    
    def _build_handler_request(
        self,
        intent: Intent,
        user_context: Dict[str, Any],
        gps: Optional[Dict] = None,
        previous_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build handler request with context from previous intents.
        
        Example:
        - Intent 1: Route to Hagia Sophia → result includes destination coords
        - Intent 2: Restaurants nearby → use destination coords from Intent 1
        """
        
        request = {
            'user_context': user_context,
            'gps': gps,
            **intent.entities
        }
        
        # Apply context from previous results
        if previous_results and intent.priority > 1:
            # For secondary intents, infer missing data from primary intent
            
            if intent.type == 'restaurant' and 'location' not in intent.entities:
                # If asking for restaurants after a route, use destination
                if 'route' in previous_results:
                    route_result = previous_results['route']
                    if 'destination' in route_result:
                        request['location'] = route_result['destination']
                        logger.debug(f"Inferred restaurant location from route: {route_result['destination']}")
            
            if intent.type == 'info' and 'location' not in intent.entities:
                # If asking for info after a route, use destination
                if 'route' in previous_results:
                    route_result = previous_results['route']
                    if 'destination' in route_result:
                        request['location'] = route_result['destination']
                        logger.debug(f"Inferred info location from route: {route_result['destination']}")
        
        return request
    
    async def synthesize_responses(
        self,
        query: str,
        intents: List[Intent],
        responses: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> str:
        """
        Use LLM to synthesize multiple responses into a coherent answer.
        
        Args:
            query: Original user query
            intents: List of detected intents
            responses: Dict of intent_type -> response
            user_context: User context
            
        Returns:
            Synthesized natural language response
        """
        
        # Check if we have map data to mention
        has_map = any(
            response.get('map_data') or response.get('route_data')
            for response in responses.values()
        )
        
        prompt = f"""
User asked: "{query}"

We detected {len(intents)} intentions and got these results:

"""
        
        for i, intent in enumerate(intents, 1):
            response = responses.get(intent.type, {})
            prompt += f"\nIntent {i} ({intent.type}):\n"
            
            if intent.type == 'route':
                route_data = response.get('route', {}) or response.get('route_data', {})
                prompt += f"- Duration: {route_data.get('duration', 'unknown')}\n"
                prompt += f"- Distance: {route_data.get('distance', 'unknown')}\n"
                prompt += f"- Steps: {len(route_data.get('steps', []))} steps\n"
                if route_data.get('modes'):
                    prompt += f"- Transport modes: {', '.join(route_data['modes'])}\n"
            
            elif intent.type == 'restaurant':
                restaurants = response.get('restaurants', [])
                prompt += f"- Found {len(restaurants)} restaurants\n"
                if restaurants:
                    prompt += f"- Top pick: {restaurants[0].get('name', 'N/A')}\n"
            
            elif intent.type == 'info':
                info = response.get('info', {})
                prompt += f"- Name: {info.get('name', 'N/A')}\n"
                prompt += f"- Description: {info.get('description', 'N/A')[:100]}...\n"
        
        if has_map:
            prompt += "\n📍 Note: Interactive map visualization is available for this response.\n"
        
        prompt += f"""

Synthesize these results into a natural, conversational response that:
1. Addresses ALL the user's intentions in order of priority
2. Connects the information logically
3. Is concise but complete (3-5 sentences)
4. Sounds natural and friendly
5. Provides actionable information
{f'6. Mention that an interactive map is available to visualize the route/locations' if has_map else ''}

Example good synthesis:
"I've found a route to Hagia Sophia that takes about 25 minutes by tram. Along the way, there are 3 great restaurants nearby: Sultanahmet Köftecisi is the most popular, known for their traditional meatballs. Check the map below to see the full route and restaurant locations. Would you like detailed directions?"

Now synthesize for the user query: "{query}"
"""
        
        try:
            synthesis = await self.llm_service.generate(prompt)
            return synthesis.strip()
            
        except Exception as e:
            logger.error(f"Error synthesizing responses: {e}")
            # Fallback: simple concatenation
            return self._simple_synthesis(intents, responses)
    
    def _simple_synthesis(self, intents: List[Intent], responses: Dict[str, Any]) -> str:
        """Fallback synthesis without LLM"""
        parts = []
        
        for intent in intents:
            response = responses.get(intent.type, {})
            
            if intent.type == 'route':
                route = response.get('route', {})
                duration = route.get('duration', 'unknown')
                parts.append(f"Route: {duration} to {intent.entities.get('destination', 'destination')}")
            
            elif intent.type == 'restaurant':
                restaurants = response.get('restaurants', [])
                parts.append(f"Found {len(restaurants)} restaurants nearby")
            
            elif intent.type == 'info':
                info = response.get('info', {})
                parts.append(f"Info about {info.get('name', 'location')}")
        
        return ". ".join(parts)
    
    async def handle_multi_intent_query(
        self,
        query: str,
        user_context: Dict[str, Any],
        gps: Optional[Dict] = None
    ) -> MultiIntentResult:
        """
        Main entry point: detect and handle multiple intents in a query.
        
        Args:
            query: User query
            user_context: User context
            gps: User GPS
            
        Returns:
            MultiIntentResult with all responses and synthesis (including map data)
        """
        
        # Step 1: Detect intents
        intents = await self.detect_multiple_intents(query, user_context, gps)
        
        if len(intents) == 1:
            logger.info("Single intent detected, using standard handlers")
            # Could fall through to regular single-intent handling
            # But let's handle it here for consistency
        else:
            logger.info(f"Multi-intent query detected: {len(intents)} intents")
        
        # Step 2: Handle each intent (in order of priority)
        responses = {}
        previous_results = {}
        
        # Handle intents sequentially (primary first, then secondary)
        # This allows secondary intents to use context from primary
        for intent in intents:
            response = await self.handle_intent(
                intent,
                user_context,
                gps,
                previous_results
            )
            
            responses[intent.type] = response
            previous_results[intent.type] = response
        
        # Step 3: Generate map data for route/location intents
        map_data = None
        if self.map_integrator:
            try:
                logger.info("🗺️ Generating map visualization for intents...")
                
                # Convert Intent objects to dicts for map integrator
                intent_dicts = []
                for intent in intents:
                    intent_dict = {
                        'type': intent.type,
                        'priority': intent.priority,
                        'entities': intent.entities,
                        'raw_text': intent.raw_text,
                        'confidence': intent.confidence
                    }
                    intent_dicts.append(intent_dict)
                
                # Generate aggregated map
                map_data = await self.map_integrator.aggregate_maps_for_multi_intent(
                    intents=intent_dicts,
                    intent_responses=responses,
                    user_location=gps,
                    language=user_context.get('language', 'en')
                )
                
                if map_data:
                    logger.info(f"✅ Generated map visualization: type={map_data.get('type')}")
                    # Format for response
                    map_data = self.map_integrator.format_map_for_response(map_data)
                else:
                    logger.debug("No map data generated for this query")
                    
            except Exception as e:
                logger.error(f"Error generating map data: {e}")
                map_data = None
        
        # Step 4: Synthesize responses
        synthesized = await self.synthesize_responses(
            query,
            intents,
            responses,
            user_context
        )
        
        # Step 5: Return result with map data
        return MultiIntentResult(
            intents=intents,
            responses=responses,
            synthesized_response=synthesized,
            metadata={
                'intent_count': len(intents),
                'primary_intent': intents[0].type if intents else None,
                'query': query,
                'has_map': map_data is not None,
                'map_data': map_data  # Include map data in metadata
            }
        )
