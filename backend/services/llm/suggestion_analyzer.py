"""
Phase 4.4: Suggestion Analyzer

Analyzes conversation context to determine if proactive suggestions should be shown.
This module gives the LLM 85% control over understanding when suggestions add value.

The analyzer handles:
1. Context analysis of query and response
2. Entity extraction and enrichment
3. Opportunity identification
4. Trigger confidence scoring

Author: AI Istanbul Team
Date: December 3, 2025
"""

import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from .models import (
    SuggestionContext,
    SuggestionAnalysis
)
from .llm_response_parser import parse_llm_json_response

logger = logging.getLogger(__name__)


class SuggestionAnalyzer:
    """
    LLM-powered suggestion analyzer.
    
    Determines when and why to show proactive suggestions based on
    conversation context, response quality, and user behavior patterns.
    
    LLM Responsibility: 85%
    - Context understanding
    - Entity extraction
    - Opportunity identification
    - Trigger reasoning
    
    Fallback: Rule-based heuristics for basic triggers
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the suggestion analyzer.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'use_llm': True,  # NEW: Enable/disable LLM  
            'timeout_seconds': 3,
            'temperature': 0.3,  # Lower for more consistent analysis
            'max_tokens': 500,
            'min_confidence': 0.6,  # Minimum confidence to show suggestions
            'fallback_enabled': True,
            **(config or {})
        }
        
        logger.info(f"SuggestionAnalyzer initialized (LLM: {self.config['use_llm']})")
    
    async def analyze_context(
        self,
        query: str,
        response: str,
        conversation_history: Optional[List[Dict]] = None,
        detected_intents: Optional[List[str]] = None,
        entities: Optional[Dict[str, Any]] = None,
        response_type: Optional[str] = None,
        user_location: Optional[str] = None
    ) -> SuggestionContext:
        """
        Analyze conversation context for suggestion generation.
        
        Args:
            query: User's current query
            response: Response we just provided
            conversation_history: Recent conversation turns
            detected_intents: Detected intents from query
            entities: Extracted entities
            response_type: Type of response (restaurant, route, etc.)
            user_location: User's location if known
            
        Returns:
            SuggestionContext with analyzed information
        """
        start_time = time.time()
        
        try:
            # Build context
            context = SuggestionContext(
                current_query=query,
                current_response=response[:500],  # Limit response length
                detected_intents=detected_intents or [],
                extracted_entities=entities or {},
                conversation_history=conversation_history[-5:] if conversation_history else [],
                user_location=user_location,
                response_type=response_type or "unknown",
                response_success=not any(err in response.lower() for err in ["error", "sorry", "couldn't", "unable"]),
                trigger_confidence=0.0
            )
            
            # Analyze if we should trigger suggestions
            if self.config.get('use_llm', True):
                try:
                    should_suggest, confidence, reason = await asyncio.wait_for(
                        self._should_suggest_llm(context),
                        timeout=self.config['timeout_seconds']
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"LLM trigger analysis timed out after {self.config['timeout_seconds']}s, using heuristics")
                    should_suggest, confidence, reason = self._heuristic_trigger(context)
            else:
                should_suggest, confidence, reason = self._heuristic_trigger(context)
                logger.debug("Using heuristic trigger (LLM disabled)")
            
            context.trigger_confidence = confidence
            context.trigger_reason = reason
            
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Context analyzed: should_suggest={should_suggest}, "
                f"confidence={confidence:.2f} ({processing_time:.0f}ms)"
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}, using minimal context")
            # Return minimal context on error
            return SuggestionContext(
                current_query=query,
                current_response=response[:500],
                detected_intents=detected_intents or [],
                extracted_entities=entities or {},
                conversation_history=[],
                response_type=response_type or "unknown",
                response_success=True,
                trigger_confidence=0.5  # Moderate confidence on fallback
            )
    
    async def should_suggest(
        self,
        context: SuggestionContext,
        min_confidence: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Determine if suggestions should be shown.
        
        Args:
            context: Analyzed conversation context
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (should_suggest, confidence)
        """
        threshold = min_confidence or self.config['min_confidence']
        
        # Quick checks
        if not context.response_success:
            return False, 0.0
        
        if len(context.current_response) < 50:
            return False, 0.0  # Response too short
        
        # Use stored confidence from context analysis
        should_show = context.trigger_confidence >= threshold
        
        return should_show, context.trigger_confidence
    
    async def analyze_should_suggest(
        self,
        context: SuggestionContext
    ) -> SuggestionAnalysis:
        """
        Detailed analysis of whether suggestions should be shown.
        
        Args:
            context: Conversation context
            
        Returns:
            SuggestionAnalysis with detailed reasoning
        """
        start_time = time.time()
        
        try:
            # Use LLM for analysis
            analysis_data = await self._analyze_suggestion_need_llm(context)
            processing_time = (time.time() - start_time) * 1000
            
            return SuggestionAnalysis(
                should_suggest=analysis_data['should_suggest'],
                confidence=analysis_data['confidence'],
                reasoning=analysis_data['reasoning'],
                context_summary=analysis_data['context_summary'],
                suggested_categories=analysis_data.get('suggested_categories', []),
                analysis_method="llm",
                analysis_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}, using heuristic fallback")
            return self._heuristic_analysis(context, start_time)
    
    async def _should_suggest_llm(
        self,
        context: SuggestionContext
    ) -> Tuple[bool, float, str]:
        """
        Use LLM to determine if suggestions should be shown.
        
        Args:
            context: Conversation context
            
        Returns:
            Tuple of (should_suggest, confidence, reason)
        """
        try:
            prompt = self._build_trigger_prompt(context)
            llm_output = await self._call_llm(prompt)
            
            # Parse JSON response (handles both dict and string responses)
            data = parse_llm_json_response(llm_output)
            
            if data is None:
                logger.error("LLM trigger analysis failed: unable to parse response")
                return self._heuristic_trigger(context)
            
            return (
                data.get('should_suggest', True),
                data.get('confidence', 0.7),
                data.get('reasoning', 'LLM analysis')
            )
            
        except Exception as e:
            logger.error(f"LLM trigger analysis failed: {e}")
            # Fallback to heuristic
            return self._heuristic_trigger(context)
    
    async def _analyze_suggestion_need_llm(
        self,
        context: SuggestionContext
    ) -> Dict[str, Any]:
        """
        Detailed LLM analysis of suggestion need.
        
        Args:
            context: Conversation context
            
        Returns:
            Analysis data dictionary
        """
        prompt = self._build_analysis_prompt(context)
        llm_output = await self._call_llm(prompt)
        
        # Parse JSON response (handles both dict and string responses)
        data = parse_llm_json_response(llm_output)
        return data if data is not None else {}
    
    def _build_trigger_prompt(self, context: SuggestionContext) -> str:
        """Build prompt for quick trigger decision."""
        return f"""Analyze if we should show proactive suggestions for this conversation:

Query: "{context.current_query}"
Response: "{context.current_response[:300]}..."
Response Type: {context.response_type}
Success: {context.response_success}
Intents: {', '.join(context.detected_intents) if context.detected_intents else 'none'}

Quick Analysis:
1. Is the response complete and successful?
2. Are there logical next steps the user might want?
3. Is this a natural conversation break point?
4. Would suggestions enhance user experience?

{self._get_trigger_instructions()}

Return ONLY valid JSON."""
    
    def _build_analysis_prompt(self, context: SuggestionContext) -> str:
        """Build prompt for detailed analysis."""
        history_text = ""
        if context.conversation_history:
            history_text = "\nRecent History:\n"
            for turn in context.conversation_history[-3:]:
                history_text += f"  Q: {turn.get('query', 'N/A')}\n"
        
        return f"""Provide detailed analysis of suggestion opportunity:

Current Query: "{context.current_query}"
Current Response: "{context.current_response[:400]}..."
Response Type: {context.response_type}
Detected Intents: {context.detected_intents}
Entities: {json.dumps(context.extracted_entities, indent=2) if context.extracted_entities else 'none'}
{history_text}

Analyze:
1. Should we show suggestions? Why or why not?
2. What is the conversation context?
3. What categories of suggestions would be most relevant?
4. What is your confidence level?

{self._get_analysis_instructions()}

Return ONLY valid JSON."""
    
    def _get_trigger_instructions(self) -> str:
        """Get instructions for trigger decision."""
        return """Respond with JSON:
{
  "should_suggest": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Show suggestions when:
- Response is successful and complete
- User likely has follow-up questions
- Natural conversation break
- Suggestions would add value

Don't show suggestions when:
- Response contains errors
- User asked a very specific follow-up
- Response is incomplete
- Too many suggestions shown recently"""
    
    def _get_analysis_instructions(self) -> str:
        """Get instructions for detailed analysis."""
        return """Respond with JSON:
{
  "should_suggest": true/false,
  "confidence": 0.85,
  "reasoning": "User asked about restaurants, will likely want directions or nearby attractions",
  "context_summary": "User exploring dining options in Sultanahmet",
  "suggested_categories": ["practical", "exploration", "dining"]
}

Categories:
- exploration: Discover new places/things
- practical: Directions, weather, transportation
- cultural: Events, customs, activities
- dining: Food and restaurants
- refinement: Filter/refine current results"""
    
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
                        {"role": "system", "content": "You are an expert at analyzing conversation context."},
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
    
    def _heuristic_trigger(self, context: SuggestionContext) -> Tuple[bool, float, str]:
        """
        Fallback heuristic trigger decision.
        
        Args:
            context: Conversation context
            
        Returns:
            Tuple of (should_suggest, confidence, reason)
        """
        # Basic heuristics
        if not context.response_success:
            return False, 0.0, "Response was not successful"
        
        if len(context.current_response) < 50:
            return False, 0.0, "Response too short"
        
        # Response types that benefit from suggestions
        good_types = ['restaurant', 'attraction', 'route', 'museum', 'event', 'hidden_gems']
        if context.response_type in good_types:
            return True, 0.7, f"Response type '{context.response_type}' typically benefits from suggestions"
        
        # Default to showing suggestions with moderate confidence
        return True, 0.6, "Default heuristic trigger"
    
    def _heuristic_analysis(
        self,
        context: SuggestionContext,
        start_time: float
    ) -> SuggestionAnalysis:
        """
        Fallback heuristic analysis.
        
        Args:
            context: Conversation context
            start_time: Start time for metrics
            
        Returns:
            SuggestionAnalysis with heuristic reasoning
        """
        should_suggest, confidence, reasoning = self._heuristic_trigger(context)
        processing_time = (time.time() - start_time) * 1000
        
        # Determine suggested categories based on response type
        category_map = {
            'restaurant': ['practical', 'exploration', 'cultural'],
            'attraction': ['practical', 'cultural', 'dining'],
            'route': ['dining', 'exploration', 'cultural'],
            'museum': ['practical', 'cultural', 'dining'],
            'event': ['practical', 'exploration', 'dining']
        }
        
        suggested_categories = category_map.get(context.response_type, ['practical', 'exploration'])
        
        return SuggestionAnalysis(
            should_suggest=should_suggest,
            confidence=confidence,
            reasoning=reasoning,
            context_summary=f"User asked about {context.response_type}",
            suggested_categories=suggested_categories,
            analysis_method="heuristic",
            analysis_time_ms=processing_time
        )


# Singleton instance
_analyzer_instance: Optional[SuggestionAnalyzer] = None


def get_suggestion_analyzer(llm_client, config: Optional[Dict[str, Any]] = None) -> SuggestionAnalyzer:
    """
    Get or create the singleton SuggestionAnalyzer instance.
    
    Args:
        llm_client: LLM API client
        config: Optional configuration
        
    Returns:
        SuggestionAnalyzer instance
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SuggestionAnalyzer(llm_client=llm_client, config=config)
    return _analyzer_instance


async def analyze_suggestion_context(
    query: str,
    response: str,
    conversation_history: Optional[List[Dict]] = None,
    detected_intents: Optional[List[str]] = None,
    entities: Optional[Dict[str, Any]] = None,
    response_type: Optional[str] = None,
    user_location: Optional[str] = None,
    llm_client=None
) -> SuggestionContext:
    """
    Convenience function to analyze suggestion context.
    
    Args:
        query: User's query
        response: Bot's response
        conversation_history: Recent conversation
        detected_intents: Detected intents
        entities: Extracted entities
        response_type: Type of response
        user_location: User's location
        llm_client: LLM client (uses singleton if provided once)
        
    Returns:
        SuggestionContext with analysis
    """
    if llm_client:
        analyzer = get_suggestion_analyzer(llm_client)
    else:
        analyzer = _analyzer_instance
        if analyzer is None:
            raise ValueError("SuggestionAnalyzer not initialized. Provide llm_client.")
    
    return await analyzer.analyze_context(
        query, response, conversation_history,
        detected_intents, entities, response_type, user_location
    )
