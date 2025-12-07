"""
advanced_prompts.py - Advanced Prompt Engineering for Low-Signal Scenarios

Phase 4 Priority 3: Intelligent prompt adaptation based on signal confidence,
multi-intent handling, and clarifying question strategies.

Features:
- Low-signal scenario prompting
- Multi-intent query handling
- Clarifying question strategies
- Adaptive prompt engineering

Author: AI Istanbul Team
Date: December 7, 2025
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AdvancedPromptEngineer:
    """
    Advanced prompt engineering system for handling complex query scenarios.
    
    Adapts prompts based on:
    - Signal confidence levels
    - Number of detected intents
    - Query ambiguity
    - User context availability
    """
    
    def __init__(self):
        """Initialize advanced prompt engineer."""
        self.stats = {
            'low_signal_prompts': 0,
            'multi_intent_prompts': 0,
            'clarifying_prompts': 0,
            'standard_prompts': 0
        }
        logger.info("✅ Advanced Prompt Engineer initialized")
    
    def enhance_prompt_for_signals(
        self,
        base_prompt: str,
        query: str,
        signals: Dict[str, Any],
        user_location: Optional[Dict[str, float]] = None,
        language: str = 'en'
    ) -> str:
        """
        Enhance prompt based on signal detection results.
        
        Args:
            base_prompt: Base system prompt
            query: User query
            signals: Signal detection results with confidence scores
            user_location: User's GPS location
            language: Query language
            
        Returns:
            Enhanced prompt string
        """
        # Extract signal metadata
        detected_signals = signals.get('signals', {})
        confidence_scores = signals.get('confidence_scores', {})
        overall_confidence = signals.get('overall_confidence', 0.5)
        active_count = signals.get('active_count', 0)
        
        # Determine prompt enhancement strategy
        if overall_confidence < 0.5:
            # Low confidence: Add explicit guidance
            enhanced = self._add_low_signal_guidance(
                base_prompt, query, detected_signals, confidence_scores, 
                user_location, language
            )
            self.stats['low_signal_prompts'] += 1
            logger.debug(f"🔍 Low-signal prompt (confidence: {overall_confidence:.2f})")
            
        elif active_count > 2:
            # Multi-intent: Add multi-intent handling
            enhanced = self._add_multi_intent_guidance(
                base_prompt, query, detected_signals, user_location, language
            )
            self.stats['multi_intent_prompts'] += 1
            logger.debug(f"🎯 Multi-intent prompt ({active_count} intents)")
            
        elif len(query.split()) < 3 and overall_confidence < 0.7:
            # Very short + uncertain: Add clarifying strategy
            enhanced = self._add_clarifying_strategy(
                base_prompt, query, detected_signals, user_location, language
            )
            self.stats['clarifying_prompts'] += 1
            logger.debug("💬 Clarifying question strategy enabled")
            
        else:
            # Standard: Return base prompt with minor enhancements
            enhanced = self._add_standard_enhancements(
                base_prompt, query, overall_confidence
            )
            self.stats['standard_prompts'] += 1
        
        return enhanced
    
    def _add_low_signal_guidance(
        self,
        base_prompt: str,
        query: str,
        detected_signals: Dict[str, bool],
        confidence_scores: Dict[str, float],
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> str:
        """Add guidance for low-confidence signal detection."""
        
        # Build signal summary
        active_signals = [
            f"{name} ({confidence_scores.get(name, 0.0):.2f})"
            for name, detected in detected_signals.items()
            if detected
        ]
        
        guidance = f"""

---

🚨 **LOW CONFIDENCE SIGNAL DETECTION**

Query: "{query}"
Language: {language}
Detected Intents (LOW CONFIDENCE): {', '.join(active_signals) if active_signals else 'None detected'}
GPS Available: {'Yes' if user_location else 'No'}

**IMPORTANT INSTRUCTIONS:**

1. **Intent Inference**: The automatic intent detection has low confidence. Use your natural language understanding to infer the user's actual intent from:
   - Query text and keywords
   - User's GPS location (if available)
   - Provided context (database results, articles, etc.)
   - Common query patterns

2. **Context Analysis**: Examine ALL provided context carefully:
   - Restaurant data → User may want dining recommendations
   - Attraction data → User may want sightseeing info
   - Transportation data → User may want directions
   - Neighborhood data → User may want area information

3. **Response Strategy**:
   - If you can confidently infer intent: Provide a helpful, complete answer
   - If multiple interpretations are possible: Address the most likely 1-2 interpretations
   - If truly ambiguous: Ask ONE clarifying question, then provide your best answer

4. **Avoid**:
   - Don't say "I'm not sure what you're asking"
   - Don't list all possible interpretations without attempting to help
   - Don't require more information if you can reasonably infer intent

**Your task:** Analyze the query and context, infer the user's intent, and provide a helpful response.

---
"""
        
        return base_prompt + guidance
    
    def _add_multi_intent_guidance(
        self,
        base_prompt: str,
        query: str,
        detected_signals: Dict[str, bool],
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> str:
        """Add guidance for multi-intent queries."""
        
        # List detected intents
        active_intents = [name for name, detected in detected_signals.items() if detected]
        
        guidance = f"""

---

🎯 **MULTI-INTENT QUERY DETECTED**

Query: "{query}"
Detected Intents: {', '.join(active_intents)}
GPS Available: {'Yes' if user_location else 'No'}

**MULTI-INTENT RESPONSE STRATEGY:**

1. **Address ALL Detected Intents**: This query has multiple aspects. Structure your response to address each one:
   {self._format_intent_bullets(active_intents)}

2. **Logical Flow**: Organize information in a natural, helpful order:
   - Start with the primary intent (usually the first or most specific)
   - Connect related intents smoothly
   - End with any additional helpful context

3. **Completeness**: Ensure each intent is fully addressed:
   - Provide specific recommendations/information for each
   - Include relevant details (addresses, hours, costs, etc.)
   - Add practical tips for combining multiple activities

4. **Conciseness**: Be comprehensive but avoid redundancy:
   - Merge overlapping information
   - Use clear section breaks if needed
   - Prioritize actionable information

**Example Structure:**
"[Address primary intent with specific details]
[Connect to secondary intent]
[Provide additional context/tips]
[Close with practical advice]"

---
"""
        
        return base_prompt + guidance
    
    def _add_clarifying_strategy(
        self,
        base_prompt: str,
        query: str,
        detected_signals: Dict[str, bool],
        user_location: Optional[Dict[str, float]],
        language: str
    ) -> str:
        """Add strategy for asking clarifying questions when needed."""
        
        guidance = f"""

---

💬 **AMBIGUOUS QUERY - CLARIFYING STRATEGY**

Query: "{query}" (very short, {len(query.split())} words)
GPS Available: {'Yes' if user_location else 'No'}

**CLARIFYING QUESTION STRATEGY:**

1. **Attempt Inference First**: Before asking for clarification:
   - Check if GPS location provides strong context
   - Examine what context/data is available
   - Consider common interpretations of the query
   
2. **Smart Clarification**: If you must ask for clarification:
   - Ask ONE specific question (not multiple choices)
   - Offer 2-3 likely options based on context
   - Provide a helpful answer to the most likely interpretation
   
3. **Response Format** (if clarification needed):
   ```
   [Provide answer to most likely interpretation]
   
   If you're looking for [alternative interpretation], I can also help with that.
   Just let me know what specific information you need!
   ```

4. **Avoid**:
   - Don't just ask "What are you looking for?"
   - Don't list all possibilities without attempting to help
   - Don't wait for clarification before providing useful information

**Your task:** Provide the best answer you can based on available context, with gentle clarification if needed.

---
"""
        
        return base_prompt + guidance
    
    def _add_standard_enhancements(
        self,
        base_prompt: str,
        query: str,
        confidence: float
    ) -> str:
        """Add standard enhancements for well-detected queries."""
        
        enhancement = f"""

---

✅ **QUERY CONTEXT**

Confidence: {confidence:.2f} (Good detection)
Query: "{query}"

Proceed with standard response generation. Intent is clear.

---
"""
        
        return base_prompt + enhancement
    
    def _format_intent_bullets(self, intents: List[str]) -> str:
        """Format detected intents as helpful bullets."""
        intent_descriptions = {
            'needs_restaurant': '• Restaurant: Provide specific recommendations with details',
            'needs_attraction': '• Attractions: Suggest relevant sites with context',
            'needs_transportation': '• Directions: Give clear routing information',
            'needs_neighborhood': '• Area info: Describe the neighborhood characteristics',
            'needs_events': '• Events: List current or upcoming activities',
            'needs_weather': '• Weather: Include weather-appropriate suggestions',
            'needs_hidden_gems': '• Local spots: Highlight authentic, less-touristy options',
            'needs_shopping': '• Shopping: Recommend shopping areas or stores',
            'needs_nightlife': '• Nightlife: Suggest bars, clubs, or evening entertainment',
            'needs_family_friendly': '• Family activities: Provide kid-friendly options'
        }
        
        bullets = []
        for intent in intents:
            if intent in intent_descriptions:
                bullets.append(intent_descriptions[intent])
        
        return '\n   '.join(bullets) if bullets else '• Address all aspects of the query'
    
    def get_stats(self) -> Dict[str, int]:
        """Get prompt engineering statistics."""
        return dict(self.stats)


# Singleton instance
_prompt_engineer = None


def get_prompt_engineer() -> AdvancedPromptEngineer:
    """Get singleton prompt engineer instance."""
    global _prompt_engineer
    if _prompt_engineer is None:
        _prompt_engineer = AdvancedPromptEngineer()
    return _prompt_engineer


# Utility functions for common prompt enhancements

def add_context_awareness_prompt(
    base_prompt: str,
    context_types_available: List[str]
) -> str:
    """
    Add awareness of what context is available.
    
    Args:
        base_prompt: Base prompt
        context_types_available: List of context types (e.g., ['database', 'rag', 'weather'])
        
    Returns:
        Enhanced prompt
    """
    context_descriptions = {
        'database': 'POI database results (restaurants, attractions, etc.)',
        'rag': 'Knowledge base articles about Istanbul',
        'weather': 'Current weather information',
        'events': 'Upcoming events and activities',
        'hidden_gems': 'Local recommendations and hidden gems',
        'user_location': 'User\'s GPS coordinates'
    }
    
    available_context = [
        f"- {context_descriptions.get(ctx, ctx)}"
        for ctx in context_types_available
    ]
    
    enhancement = f"""

**AVAILABLE CONTEXT:**
{chr(10).join(available_context)}

Use this context to provide accurate, specific recommendations.
"""
    
    return base_prompt + enhancement


def add_language_awareness(
    base_prompt: str,
    detected_language: str,
    query_language: str
) -> str:
    """
    Add language awareness to prompt.
    
    Args:
        base_prompt: Base prompt
        detected_language: Detected user language
        query_language: Language query was processed in
        
    Returns:
        Enhanced prompt
    """
    if detected_language != query_language:
        enhancement = f"""

**LANGUAGE NOTE:**
- Query processed in: {query_language}
- User's language: {detected_language}
- Respond in: {detected_language}

Make sure your response is in the user's language ({detected_language}).
"""
        return base_prompt + enhancement
    
    return base_prompt


def add_gps_context_prompt(
    base_prompt: str,
    user_location: Dict[str, float],
    location_name: Optional[str] = None
) -> str:
    """
    Add GPS-specific guidance to prompt.
    
    Args:
        base_prompt: Base prompt
        user_location: User's GPS coordinates
        location_name: Human-readable location name (if available)
        
    Returns:
        Enhanced prompt
    """
    lat, lon = user_location.get('lat'), user_location.get('lon')
    
    enhancement = f"""

**GPS CONTEXT:**
- User's coordinates: {lat:.6f}, {lon:.6f}
"""
    
    if location_name:
        enhancement += f"- Approximate area: {location_name}\n"
    
    enhancement += """
- Use this location for "nearby" recommendations
- Provide walking/transit directions from this location
- Mention distances/travel times from here
"""
    
    return base_prompt + enhancement
