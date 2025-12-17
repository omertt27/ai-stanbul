"""
LLM Clarification Handler

Handles medium-confidence queries by asking clarifying questions.
This improves intent detection accuracy for ambiguous transportation queries.

Author: AI Istanbul Team
Date: December 17, 2025
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class LLMClarificationHandler:
    """
    Generates clarifying questions for ambiguous queries.
    
    When intent confidence is medium (0.45-0.64), this handler helps
    resolve ambiguity by asking targeted questions.
    
    Examples:
    - "I need directions" → "Where would you like to go?"
    - "How to get there?" → "Where is 'there'? Could you specify?"
    - "transportation" → "Are you looking for route directions?"
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the clarification handler.
        
        Args:
            llm_client: Optional LLM client for generating questions
        """
        self.llm = llm_client
        logger.info("🔍 LLM Clarification Handler initialized")
    
    async def generate_clarification(
        self,
        query: str,
        signals: Dict[str, bool],
        confidence: float,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate a clarifying question for an ambiguous query.
        
        Args:
            query: The user's original query
            signals: Detected signals from intent classifier
            confidence: Overall confidence score (should be 0.45-0.64)
            language: Response language
            
        Returns:
            Dict with:
            - needs_clarification: bool
            - question: str (the clarifying question)
            - suggestions: List[str] (suggested refinements)
            - reason: str (why clarification is needed)
        """
        logger.info(f"🔍 Analyzing query for clarification need: '{query}' (confidence: {confidence:.2f})")
        
        # Check if this is a clear case that needs clarification
        query_lower = query.lower().strip()
        
        # Pattern 1: Very vague queries
        vague_patterns = [
            'directions', 'route', 'how to go', 'how to get',
            'i need to go', 'transport', 'transportation',
            'yol tarifi', 'nasıl giderim', 'nasıl gidilir'
        ]
        
        is_vague = any(pattern in query_lower for pattern in vague_patterns)
        
        # Pattern 2: Incomplete location info
        has_origin = any(word in query_lower for word in ['from', 'dan', 'den', 'başlangıç'])
        has_destination = any(word in query_lower for word in ['to', 'a', 'e', 'hedef', 'toward'])
        
        # Extract signal with highest confidence (if available)
        active_signals = [k for k, v in signals.items() if v]
        primary_intent = active_signals[0] if active_signals else None
        
        # Rule-based clarification for common cases
        if is_vague and not (has_origin and has_destination):
            return self._generate_location_clarification(query, language)
        
        # Use LLM for more nuanced clarification
        if self.llm and confidence < 0.55:
            try:
                llm_clarification = await self._generate_llm_clarification(
                    query, signals, confidence, language
                )
                if llm_clarification:
                    return llm_clarification
            except Exception as e:
                logger.warning(f"LLM clarification generation failed: {e}")
        
        # Fallback: No clarification needed
        return {
            'needs_clarification': False,
            'question': None,
            'suggestions': [],
            'reason': 'query_is_clear_enough'
        }
    
    def _generate_location_clarification(
        self,
        query: str,
        language: str
    ) -> Dict[str, Any]:
        """Generate clarification for missing location information"""
        
        if language == 'tr':
            question = (
                "Size yardımcı olmaktan mutluluk duyarım! "
                "Lütfen şunları belirtir misiniz:\n\n"
                "• Nereden başlayacaksınız?\n"
                "• Nereye gitmek istiyorsunuz?"
            )
            suggestions = [
                "Kadıköy'den Taksim'e nasıl gidilir?",
                "Sultanahmet'ten Galata'ya yol tarifi",
                "Beşiktaş'tan Üsküdar'a ulaşım"
            ]
        else:
            question = (
                "I'd be happy to help with directions! "
                "Could you please specify:\n\n"
                "• Where are you starting from?\n"
                "• Where would you like to go?"
            )
            suggestions = [
                "How do I get from Kadıköy to Taksim?",
                "Directions from Sultanahmet to Galata",
                "Route from Beşiktaş to Üsküdar"
            ]
        
        return {
            'needs_clarification': True,
            'question': question,
            'suggestions': suggestions,
            'reason': 'missing_location_info'
        }
    
    async def _generate_llm_clarification(
        self,
        query: str,
        signals: Dict[str, bool],
        confidence: float,
        language: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to generate a clarifying question.
        
        This is more nuanced than rule-based clarification.
        """
        
        prompt = f"""You are helping a user who asked an ambiguous question about Istanbul travel.

User query: "{query}"

The query is ambiguous (confidence: {confidence:.2f}). Generate a brief, friendly clarifying question.

Requirements:
- Keep it SHORT (max 2 sentences)
- Be specific about what information is missing
- Language: {language}

Clarifying question:"""
        
        try:
            result = await self.llm.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.3
            )
            
            clarification_text = result.get('generated_text', '').strip()
            
            # Validate
            if len(clarification_text) > 200 or len(clarification_text) < 10:
                logger.warning("LLM clarification invalid length, using fallback")
                return None
            
            return {
                'needs_clarification': True,
                'question': clarification_text,
                'suggestions': self._get_example_suggestions(language),
                'reason': 'llm_generated_clarification'
            }
            
        except Exception as e:
            logger.error(f"LLM clarification generation error: {e}")
            return None
    
    def _get_example_suggestions(self, language: str) -> List[str]:
        """Get example query suggestions"""
        if language == 'tr':
            return [
                "Kadıköy'den Taksim'e nasıl gidilir?",
                "Sultanahmet'te gezilecek yerler",
                "Beşiktaş civarında restoranlar"
            ]
        else:
            return [
                "How do I get from Kadıköy to Taksim?",
                "Things to do in Sultanahmet",
                "Restaurants near Beşiktaş"
            ]


# Singleton instance
_clarification_handler = None

def get_clarification_handler(llm_client=None) -> LLMClarificationHandler:
    """Get or create the clarification handler singleton"""
    global _clarification_handler
    if _clarification_handler is None:
        _clarification_handler = LLMClarificationHandler(llm_client)
    return _clarification_handler
