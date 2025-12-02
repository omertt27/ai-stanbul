"""
Phase 4.4: Suggestion Generator

LLM-powered generation of proactive suggestions based on conversation context.
This module gives the LLM 95% control over creating relevant, helpful suggestions.

The generator handles:
1. Generating contextually relevant suggestions
2. Ranking suggestions by usefulness
3. Ensuring diversity across suggestion types
4. Formatting suggestions naturally

Author: AI Istanbul Team
Date: December 3, 2025
"""

import json
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    SuggestionContext,
    ProactiveSuggestion,
    ProactiveSuggestionResponse
)

logger = logging.getLogger(__name__)


class SuggestionGenerator:
    """
    LLM-powered suggestion generator.
    
    Generates relevant, actionable suggestions based on conversation context.
    The LLM creates natural, contextually appropriate suggestions that feel
    like helpful recommendations from a knowledgeable travel guide.
    
    LLM Responsibility: 95%
    - Suggestion generation
    - Relevance scoring
    - Natural language phrasing
    - Context adaptation
    
    Fallback: Template-based suggestions for common scenarios
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the suggestion generator.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'timeout_seconds': 4,
            'temperature': 0.8,  # Higher for creative suggestions
            'max_tokens': 800,
            'max_suggestions': 5,
            'min_diversity_score': 0.6,
            'fallback_enabled': True,
            **(config or {})
        }
        
        logger.info("SuggestionGenerator initialized")
    
    async def generate_suggestions(
        self,
        context: SuggestionContext,
        max_suggestions: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> List[ProactiveSuggestion]:
        """
        Generate proactive suggestions based on context.
        
        Args:
            context: Analyzed conversation context
            max_suggestions: Maximum number of suggestions to generate
            categories: Preferred suggestion categories (if any)
            
        Returns:
            List of ProactiveSuggestion objects, ranked by relevance
        """
        start_time = time.time()
        max_count = max_suggestions or self.config['max_suggestions']
        
        try:
            # Generate suggestions using LLM
            suggestions_data = await self._generate_suggestions_llm(
                context, max_count, categories
            )
            
            # Convert to ProactiveSuggestion objects
            suggestions = []
            for i, data in enumerate(suggestions_data):
                suggestion = ProactiveSuggestion(
                    suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                    suggestion_text=data['text'],
                    suggestion_type=data['type'],
                    intent_type=data['intent'],
                    entities=data.get('entities', {}),
                    relevance_score=data.get('relevance', 0.8),
                    priority=max_count - i,  # Higher priority = shown first
                    reasoning=data.get('reasoning'),
                    icon=data.get('icon'),
                    action_type=data.get('action_type', 'query')
                )
                suggestions.append(suggestion)
            
            # Rank and filter
            suggestions = await self.rank_suggestions(suggestions, context)
            suggestions = suggestions[:max_count]
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Generated {len(suggestions)} suggestions "
                f"({processing_time:.0f}ms)"
            )
            
            return suggestions
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}, using template fallback")
            return self._template_suggestions(context, max_count, start_time)
    
    async def rank_suggestions(
        self,
        suggestions: List[ProactiveSuggestion],
        context: SuggestionContext
    ) -> List[ProactiveSuggestion]:
        """
        Rank suggestions by relevance and diversity.
        
        Args:
            suggestions: List of suggestions to rank
            context: Conversation context
            
        Returns:
            Ranked list of suggestions
        """
        if not suggestions:
            return []
        
        try:
            # Use LLM for intelligent ranking
            ranked_data = await self._rank_suggestions_llm(suggestions, context)
            
            # Update priorities based on LLM ranking
            for i, sugg_id in enumerate(ranked_data['ranked_ids']):
                for suggestion in suggestions:
                    if suggestion.suggestion_id == sugg_id:
                        suggestion.priority = len(suggestions) - i
                        break
            
            # Sort by priority
            suggestions.sort(key=lambda s: s.priority, reverse=True)
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"LLM ranking failed: {e}, using relevance scores")
            # Fallback: sort by relevance score
            suggestions.sort(key=lambda s: s.relevance_score, reverse=True)
            return suggestions
    
    async def generate_with_response(
        self,
        context: SuggestionContext,
        max_suggestions: Optional[int] = None
    ) -> ProactiveSuggestionResponse:
        """
        Generate suggestions and return complete response.
        
        Args:
            context: Conversation context
            max_suggestions: Maximum suggestions
            
        Returns:
            ProactiveSuggestionResponse with metadata
        """
        start_time = time.time()
        
        suggestions = await self.generate_suggestions(context, max_suggestions)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate diversity score
        diversity = self._calculate_diversity(suggestions)
        
        return ProactiveSuggestionResponse(
            suggestions=suggestions,
            context=context,
            generation_method="llm" if suggestions and suggestions[0].reasoning else "template",
            generation_time_ms=processing_time,
            total_suggestions_considered=len(suggestions) * 2,  # Estimate
            confidence=context.trigger_confidence,
            diversity_score=diversity,
            timestamp=datetime.now(),
            llm_used=bool(suggestions and suggestions[0].reasoning)
        )
    
    async def _generate_suggestions_llm(
        self,
        context: SuggestionContext,
        max_count: int,
        categories: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to generate suggestions.
        
        Args:
            context: Conversation context
            max_count: Maximum suggestions
            categories: Preferred categories
            
        Returns:
            List of suggestion data dictionaries
        """
        prompt = self._build_generation_prompt(context, max_count, categories)
        llm_output = await self._call_llm(prompt)
        
        # Parse JSON response
        data = json.loads(llm_output)
        return data.get('suggestions', [])
    
    async def _rank_suggestions_llm(
        self,
        suggestions: List[ProactiveSuggestion],
        context: SuggestionContext
    ) -> Dict[str, Any]:
        """
        Use LLM to rank suggestions.
        
        Args:
            suggestions: Suggestions to rank
            context: Conversation context
            
        Returns:
            Ranking data with ordered IDs
        """
        prompt = self._build_ranking_prompt(suggestions, context)
        llm_output = await self._call_llm(prompt)
        
        return json.loads(llm_output)
    
    def _build_generation_prompt(
        self,
        context: SuggestionContext,
        max_count: int,
        categories: Optional[List[str]] = None
    ) -> str:
        """Build prompt for suggestion generation."""
        
        entities_text = ""
        if context.extracted_entities:
            entities_text = f"\nExtracted Entities:\n{json.dumps(context.extracted_entities, indent=2)}"
        
        categories_text = ""
        if categories:
            categories_text = f"\nPrefer these categories: {', '.join(categories)}"
        
        return f"""Generate {max_count} proactive, helpful suggestions for a travel chatbot user:

Current Query: "{context.current_query}"
Response Given: "{context.current_response[:400]}..."
Response Type: {context.response_type}
Detected Intents: {', '.join(context.detected_intents) if context.detected_intents else 'none'}
{entities_text}
{categories_text}

{self._get_generation_instructions()}

Return ONLY valid JSON with exactly {max_count} suggestions."""
    
    def _build_ranking_prompt(
        self,
        suggestions: List[ProactiveSuggestion],
        context: SuggestionContext
    ) -> str:
        """Build prompt for suggestion ranking."""
        
        suggestions_text = "\n".join([
            f"{i+1}. [{s.suggestion_id}] {s.suggestion_text} ({s.suggestion_type})"
            for i, s in enumerate(suggestions)
        ])
        
        return f"""Rank these suggestions by relevance and usefulness:

Query: "{context.current_query}"
Response Type: {context.response_type}

Suggestions:
{suggestions_text}

Rank them from most to least relevant considering:
1. Natural next step for user
2. Relevance to current context
3. Practical value
4. Diversity (don't rank all same type highly)

Respond with JSON:
{{
  "ranked_ids": ["sugg_abc123", "sugg_def456", ...],
  "reasoning": "brief explanation of ranking"
}}"""
    
    def _get_generation_instructions(self) -> str:
        """Get instructions for suggestion generation."""
        return """You are a helpful Istanbul travel guide. Generate suggestions that feel natural and helpful.

SUGGESTION TYPES:
- exploration: "Discover hidden gems nearby", "See other attractions in this area"
- practical: "Get directions", "Check the weather", "Find transportation options"
- cultural: "See cultural events tonight", "Learn about local customs"
- dining: "Find restaurants nearby", "Try local specialties"
- refinement: "Filter by price range", "Show only outdoor activities"

SUGGESTION GUIDELINES:
1. Be specific and actionable (not vague like "learn more")
2. Use natural language ("Get directions to these restaurants" not "directions_query")
3. Include relevant emojis (🗺️ 🍽️ 🎭 🌤️ 💎 🏛️)
4. Ensure diversity - don't repeat similar suggestions
5. Make each suggestion feel valuable
6. Consider what a helpful human guide would suggest

INTENT MAPPING:
Map each suggestion to an intent:
- "Get directions to X" → intent: "get_directions"
- "Find restaurants near X" → intent: "find_restaurant"
- "Check weather for X" → intent: "get_weather"
- "See attractions in X" → intent: "find_attraction"
- "Discover hidden gems in X" → intent: "find_hidden_gems"
- "See events in X" → intent: "find_events"

Respond with JSON:
{
  "suggestions": [
    {
      "text": "Get directions to these restaurants",
      "type": "practical",
      "intent": "get_directions",
      "relevance": 0.95,
      "reasoning": "User asked about restaurants, likely wants to visit them",
      "icon": "🗺️",
      "action_type": "query",
      "entities": {"destination": "Sultanahmet"}
    },
    ...
  ]
}

IMPORTANT:
- Generate exactly the requested number of suggestions
- Ensure high diversity across types
- Make suggestions feel personal and contextual
- Each suggestion should be genuinely helpful"""
    
    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM via existing client.
        
        Args:
            prompt: Prompt to send
            
        Returns:
            LLM response text
        """
        try:
            # Check if client has OpenAI-style interface
            if hasattr(self.llm_client, 'chat') and hasattr(self.llm_client.chat, 'completions'):
                # OpenAI-style client
                response = await self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert Istanbul travel guide creating helpful suggestions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.config['temperature'],
                    timeout=self.config['timeout_seconds']
                )
                return response.choices[0].message.content.strip()
            else:
                # Generic LLM client with generate method
                return await self.llm_client.generate(
                    prompt=prompt,
                    max_tokens=self.config['max_tokens'],
                    temperature=self.config['temperature']
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _template_suggestions(
        self,
        context: SuggestionContext,
        max_count: int,
        start_time: float
    ) -> List[ProactiveSuggestion]:
        """
        Fallback template-based suggestions.
        
        Uses predefined templates based on response type.
        
        Args:
            context: Conversation context
            max_count: Maximum suggestions
            start_time: Start time for metrics
            
        Returns:
            List of template-based suggestions
        """
        templates = self._get_suggestion_templates(context.response_type)
        
        suggestions = []
        for i, template in enumerate(templates[:max_count]):
            # Fill template with context entities
            text = template['text']
            entities = {}
            
            # Extract location from entities
            if context.extracted_entities:
                location = (
                    context.extracted_entities.get('destination') or
                    context.extracted_entities.get('location') or
                    context.extracted_entities.get('origin')
                )
                if location:
                    text = text.replace("{location}", location)
                    entities['location'] = location
            
            suggestion = ProactiveSuggestion(
                suggestion_id=f"sugg_{uuid.uuid4().hex[:8]}",
                suggestion_text=text,
                suggestion_type=template['type'],
                intent_type=template['intent'],
                entities=entities,
                relevance_score=0.7 - (i * 0.05),  # Decreasing relevance
                priority=max_count - i,
                reasoning=None,  # No reasoning for templates
                icon=template.get('icon'),
                action_type='query'
            )
            suggestions.append(suggestion)
        
        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Generated {len(suggestions)} template suggestions ({processing_time:.0f}ms)")
        
        return suggestions
    
    def _get_suggestion_templates(self, response_type: str) -> List[Dict[str, Any]]:
        """
        Get template suggestions for a response type.
        
        Args:
            response_type: Type of response
            
        Returns:
            List of suggestion templates
        """
        templates_map = {
            'restaurant': [
                {'text': 'Get directions to these restaurants', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Check the weather forecast', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Find attractions nearby', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Discover hidden gems in {location}', 'type': 'exploration', 'intent': 'find_hidden_gems', 'icon': '💎'},
                {'text': 'See cultural events tonight', 'type': 'cultural', 'intent': 'find_events', 'icon': '🎭'},
            ],
            'attraction': [
                {'text': 'Get directions to this attraction', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Find restaurants nearby', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See other attractions in the area', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Check opening hours and prices', 'type': 'practical', 'intent': 'get_information', 'icon': '🎫'},
                {'text': 'Discover hidden gems nearby', 'type': 'exploration', 'intent': 'find_hidden_gems', 'icon': '💎'},
            ],
            'route': [
                {'text': 'Find restaurants along the route', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See attractions near your destination', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Check the weather for your journey', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Find alternative transportation', 'type': 'practical', 'intent': 'get_transport', 'icon': '🚇'},
                {'text': 'Save this route for later', 'type': 'refinement', 'intent': 'save_route', 'icon': '💾'},
            ],
            'museum': [
                {'text': 'Get directions to this museum', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Check current exhibitions', 'type': 'cultural', 'intent': 'get_information', 'icon': '🎨'},
                {'text': 'Find restaurants nearby', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See other museums in the area', 'type': 'exploration', 'intent': 'find_museum', 'icon': '🏛️'},
                {'text': 'Plan a full day museum tour', 'type': 'exploration', 'intent': 'plan_tour', 'icon': '📍'},
            ],
            'event': [
                {'text': 'Get directions to this event', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Find restaurants near the venue', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See other events tonight', 'type': 'cultural', 'intent': 'find_events', 'icon': '🎭'},
                {'text': 'Check the weather forecast', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Find nearby parking', 'type': 'practical', 'intent': 'find_parking', 'icon': '🅿️'},
            ],
            'hidden_gems': [
                {'text': 'Get directions to these places', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
                {'text': 'Find restaurants in the area', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
                {'text': 'See nearby attractions', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
                {'text': 'Check the weather', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
                {'text': 'Plan a walking tour', 'type': 'exploration', 'intent': 'plan_tour', 'icon': '🚶'},
            ],
        }
        
        # Default suggestions for unknown types
        default = [
            {'text': 'Find restaurants nearby', 'type': 'dining', 'intent': 'find_restaurant', 'icon': '🍽️'},
            {'text': 'See nearby attractions', 'type': 'exploration', 'intent': 'find_attraction', 'icon': '🏛️'},
            {'text': 'Check the weather', 'type': 'practical', 'intent': 'get_weather', 'icon': '🌤️'},
            {'text': 'Get directions', 'type': 'practical', 'intent': 'get_directions', 'icon': '🗺️'},
            {'text': 'Discover hidden gems', 'type': 'exploration', 'intent': 'find_hidden_gems', 'icon': '💎'},
        ]
        
        return templates_map.get(response_type, default)
    
    def _calculate_diversity(self, suggestions: List[ProactiveSuggestion]) -> float:
        """
        Calculate diversity score for suggestions.
        
        Args:
            suggestions: List of suggestions
            
        Returns:
            Diversity score (0-1)
        """
        if not suggestions:
            return 0.0
        
        # Count unique types
        types = set(s.suggestion_type for s in suggestions)
        intents = set(s.intent_type for s in suggestions)
        
        # Diversity based on unique types and intents
        type_diversity = len(types) / len(suggestions)
        intent_diversity = len(intents) / len(suggestions)
        
        return (type_diversity + intent_diversity) / 2


# Singleton instance
_generator_instance: Optional[SuggestionGenerator] = None


def get_suggestion_generator(llm_client, config: Optional[Dict[str, Any]] = None) -> SuggestionGenerator:
    """
    Get or create the singleton SuggestionGenerator instance.
    
    Args:
        llm_client: LLM API client
        config: Optional configuration
        
    Returns:
        SuggestionGenerator instance
    """
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = SuggestionGenerator(llm_client=llm_client, config=config)
    return _generator_instance


async def generate_proactive_suggestions(
    context: SuggestionContext,
    max_suggestions: Optional[int] = None,
    llm_client=None
) -> ProactiveSuggestionResponse:
    """
    Convenience function to generate suggestions.
    
    Args:
        context: Conversation context
        max_suggestions: Maximum suggestions
        llm_client: LLM client (uses singleton if provided once)
        
    Returns:
        ProactiveSuggestionResponse with suggestions
    """
    if llm_client:
        generator = get_suggestion_generator(llm_client)
    else:
        generator = _generator_instance
        if generator is None:
            raise ValueError("SuggestionGenerator not initialized. Provide llm_client.")
    
    return await generator.generate_with_response(context, max_suggestions)
