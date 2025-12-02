"""
Phase 4.3: Response Synthesizer

LLM-powered synthesis of multiple intent responses into coherent answers.
This module gives the LLM 100% control over combining responses from multiple intents
into a natural, flowing conversation response.

The synthesizer handles:
1. Combining multiple responses naturally
2. Creating logical flow and transitions
3. Maintaining conversation context
4. Formatting for user presentation

Author: AI Istanbul Team
Date: December 2, 2025
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional

from .models import (
    IntentResult,
    MultiIntentResponse,
    MultiIntentDetection
)

logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """
    LLM-powered response synthesizer.
    
    Combines multiple intent responses into coherent, natural answers.
    This is critical for user experience - the LLM creates flowing narratives
    from separate responses.
    
    LLM Responsibility: 100%
    - Response combination
    - Narrative flow creation
    - Transition generation
    - Context maintenance
    - Format optimization
    
    No Fallback: Response quality is paramount
    """
    
    def __init__(self, llm_client, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the response synthesizer.
        
        Args:
            llm_client: LLM API client (RunPod, OpenAI, etc.)
            config: Optional configuration overrides
        """
        self.llm_client = llm_client
        
        # Configuration
        self.config = {
            'timeout_seconds': 5,
            'temperature': 0.7,  # Higher for natural language
            'max_tokens': 1000,
            'fallback_enabled': True,
            **(config or {})
        }
        
        logger.info(f"ResponseSynthesizer initialized")
    
    async def synthesize_responses(
        self,
        query: str,
        intent_results: List[IntentResult],
        detection: Optional[MultiIntentDetection] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> MultiIntentResponse:
        """
        Synthesize multiple intent responses into one coherent response.
        
        Args:
            query: Original user query
            intent_results: Results from each intent execution
            detection: Original multi-intent detection (optional)
            context: Optional conversation context
            
        Returns:
            MultiIntentResponse with synthesized answer
        """
        start_time = time.time()
        
        try:
            # Prepare response information for LLM
            responses_info = []
            for result in intent_results:
                info = {
                    "intent_type": result.intent_type,
                    "success": result.success,
                    "response": result.response if result.success else f"Error: {result.error}"
                }
                if result.data:
                    info["data_summary"] = self._summarize_data(result.data)
                responses_info.append(info)
            
            # Determine suggested structure
            suggested_structure = self._suggest_structure(query, intent_results, detection)
            
            # Prepare context info
            context_info = ""
            if context:
                if context.get("user_name"):
                    context_info += f"\nUser name: {context['user_name']}"
                if context.get("conversation_history"):
                    context_info += f"\nRecent context: {context['conversation_history'][-2:]}"
            
            # Build prompt
            prompt = f"""Synthesize responses for this query:

Original Query: "{query}"

Intent Responses:
{json.dumps(responses_info, indent=2)}

Suggested Structure: {suggested_structure}
{context_info}

{self._get_synthesis_instructions()}

Return ONLY a valid JSON object with the structure specified above."""
            
            logger.info(f"Synthesizing {len(intent_results)} responses...")
            
            # Call LLM via existing client
            llm_output = await self._call_llm(prompt)
            
            # Parse JSON response
            data = json.loads(llm_output)

            
            # Parse JSON response
            data = json.loads(llm_output)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate total processing time (including intent execution)
            total_time = processing_time + sum(
                r.execution_time_ms or 0 for r in intent_results
            )
            
            result = MultiIntentResponse(
                original_query=query,
                intent_results=intent_results,
                synthesized_response=data["synthesized_response"],
                response_structure=data["response_structure"],
                coherence_score=data.get("coherence_score", 0.9),
                completeness=data.get("completeness", 0.9),
                synthesis_method="llm",
                synthesis_time_ms=processing_time,
                total_processing_time_ms=total_time
            )
            
            logger.info(
                f"Synthesized response: structure={result.response_structure}, "
                f"coherence={result.coherence_score:.2f}, "
                f"completeness={result.completeness:.2f} "
                f"({processing_time:.0f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}, using template fallback")
            return self._template_synthesis(query, intent_results, start_time)
    
    def _get_synthesis_instructions(self) -> str:
        """Get synthesis instructions for LLM."""
        return """You are an expert at creating natural, flowing responses by combining multiple pieces of information.

Your task is to synthesize multiple intent responses into a single, coherent answer that:
1. Flows naturally like a conversation
2. Includes all important information from each response
3. Uses smooth transitions between topics
4. Maintains context and relevance
5. Is concise but complete
6. Sounds helpful and friendly

RESPONSE STRUCTURES:
- narrative: Flowing story combining all responses
- structured: Clear sections for each intent (use for 3+ intents)
- comparison: Side-by-side comparison (for comparison queries)
- conditional: "If X, then Y" format (for conditional queries)

STYLE GUIDELINES:
- Be conversational and helpful
- Use "I found...", "Here's...", "Also..." for flow
- Don't repeat information unnecessarily
- Group related information together
- End with a helpful note or suggestion if appropriate
- Keep emojis minimal and appropriate for travel context

RESPOND IN JSON:
{
  "synthesized_response": "<complete natural language response>",
  "response_structure": "narrative|structured|comparison|conditional",
  "coherence_score": 0.95,
  "completeness": 0.95,
  "reasoning": "Combined route directions with restaurant recommendations using location flow"
}

IMPORTANT:
- Don't lose critical information from any response
- Maintain the helpful, travel-guide tone
- Create smooth transitions between topics
- Be concise but informative"""
    
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
                        {"role": "system", "content": "You are an expert at creating natural responses."},
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
    
    def _suggest_structure(
        self,
        query: str,
        intent_results: List[IntentResult],
        detection: Optional[MultiIntentDetection]
    ) -> str:
        """
        Suggest response structure based on query and results.
        
        Args:
            query: Original query
            intent_results: Intent results
            detection: Optional multi-intent detection
            
        Returns:
            Suggested structure type
        """
        query_lower = query.lower()
        
        # Comparison queries
        if any(word in query_lower for word in ["compare", "versus", "vs", "difference between"]):
            return "comparison"
        
        # Conditional queries
        if any(word in query_lower for word in ["if", "when", "unless", "otherwise"]):
            return "conditional"
        
        # Many intents → structured
        if len(intent_results) >= 3:
            return "structured"
        
        # Default to narrative flow
        return "narrative"
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """
        Create brief summary of structured data.
        
        Args:
            data: Structured data from intent execution
            
        Returns:
            Brief summary string
        """
        if not data:
            return "No additional data"
        
        # Summarize based on data type
        summary_parts = []
        
        if "route" in data:
            route = data["route"]
            if isinstance(route, dict):
                if "duration" in route:
                    summary_parts.append(f"Duration: {route['duration']}")
                if "distance" in route:
                    summary_parts.append(f"Distance: {route['distance']}")
        
        if "count" in data:
            summary_parts.append(f"Count: {data['count']}")
        
        if "items" in data and isinstance(data["items"], list):
            summary_parts.append(f"{len(data['items'])} items")
        
        if summary_parts:
            return ", ".join(summary_parts)
        
        return f"{len(data)} data fields"
    
    def _template_synthesis(
        self,
        query: str,
        intent_results: List[IntentResult],
        start_time: float
    ) -> MultiIntentResponse:
        """
        Fallback template-based synthesis (when LLM fails).
        
        Uses simple templates to combine responses.
        Only used as emergency fallback - quality is not as good as LLM.
        
        Args:
            query: Original query
            intent_results: Intent results
            start_time: Start time for metrics
            
        Returns:
            MultiIntentResponse with template-based synthesis
        """
        # Simple concatenation with basic transitions
        parts = []
        
        for i, result in enumerate(intent_results):
            if not result.success:
                parts.append(f"⚠️ {result.intent_type}: {result.error}")
                continue
            
            # Add transition
            if i == 0:
                parts.append(result.response)
            elif i == len(intent_results) - 1:
                parts.append(f"\nAlso, {result.response}")
            else:
                parts.append(f"\nAdditionally, {result.response}")
        
        synthesized = "\n".join(parts)
        
        processing_time = (time.time() - start_time) * 1000
        total_time = processing_time + sum(r.execution_time_ms or 0 for r in intent_results)
        
        return MultiIntentResponse(
            original_query=query,
            intent_results=intent_results,
            synthesized_response=synthesized,
            response_structure="structured",
            coherence_score=0.6,
            completeness=0.8,
            synthesis_method="template",
            synthesis_time_ms=processing_time,
            total_processing_time_ms=total_time
        )


# Singleton instance
_synthesizer_instance: Optional[ResponseSynthesizer] = None


def get_response_synthesizer(llm_client, config: Optional[Dict[str, Any]] = None) -> ResponseSynthesizer:
    """
    Get or create the singleton ResponseSynthesizer instance.
    
    Args:
        llm_client: LLM API client
        config: Optional configuration
        
    Returns:
        ResponseSynthesizer instance
    """
    global _synthesizer_instance
    if _synthesizer_instance is None:
        _synthesizer_instance = ResponseSynthesizer(llm_client=llm_client, config=config)
    return _synthesizer_instance


async def synthesize_multi_intent_response(
    query: str,
    intent_results: List[IntentResult],
    detection: Optional[MultiIntentDetection] = None,
    context: Optional[Dict[str, Any]] = None,
    llm_client=None
) -> MultiIntentResponse:
    """
    Convenience function to synthesize responses.
    
    Args:
        query: Original user query
        intent_results: Results from intent execution
        detection: Optional multi-intent detection
        context: Optional conversation context
        llm_client: LLM client (uses singleton if provided once)
        
    Returns:
        MultiIntentResponse with synthesized answer
    """
    if llm_client:
        synthesizer = get_response_synthesizer(llm_client)
    else:
        synthesizer = _synthesizer_instance
        if synthesizer is None:
            raise ValueError("ResponseSynthesizer not initialized. Provide llm_client.")
    
    return await synthesizer.synthesize_responses(
        query, intent_results, detection, context
    )
