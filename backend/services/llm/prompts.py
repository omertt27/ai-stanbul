"""
prompts.py - Prompt Engineering System

Advanced prompt construction for optimal LLM performance.

Features:
- Intent-specific prompts
- Dynamic context injection
- Conversation history formatting
- Multi-language support
- Token optimization
- Few-shot examples

Author: AI Istanbul Team
Date: November 2025
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Advanced prompt engineering system.
    
    Builds optimized prompts based on:
    - Detected signals/intents
    - Available context (database, RAG, services)
    - Conversation history
    - Language preferences
    """
    
    def __init__(
        self,
        system_prompts: Optional[Dict[str, str]] = None,
        intent_prompts: Optional[Dict[str, str]] = None
    ):
        """
        Initialize prompt builder.
        
        Args:
            system_prompts: Custom system prompts
            intent_prompts: Custom intent-specific prompts
        """
        self.system_prompts = system_prompts or self._default_system_prompts()
        self.intent_prompts = intent_prompts or self._default_intent_prompts()
        
        logger.info("✅ Prompt Builder initialized")
    
    def _default_system_prompts(self) -> Dict[str, str]:
        """Default system prompts for each language."""
        return {
            'en': """You are Istanbul AI, an expert travel assistant for Istanbul, Turkey.

Your role:
- Provide accurate, helpful information about Istanbul
- Use provided database and context information
- Be conversational and friendly
- Give specific recommendations with details
- Include practical information (prices, hours, directions)
- Respect cultural sensitivities

Guidelines:
- ALWAYS use information from the provided context
- Do NOT make up information
- If you don't know, say so honestly
- Keep responses concise but informative
- Use natural, conversational language""",

            'tr': """Istanbul AI'sınız, İstanbul için uzman bir seyahat asistanısınız.

Rolünüz:
- İstanbul hakkında doğru ve yararlı bilgiler sağlayın
- Sağlanan veritabanı ve bağlam bilgilerini kullanın
- Samimi ve dostane olun
- Detaylı öneriler verin
- Pratik bilgiler ekleyin (fiyatlar, saatler, yol tarifleri)
- Kültürel hassasiyetlere saygı gösterin

Kurallar:
- HER ZAMAN sağlanan bağlamı kullanın
- Bilgi uydurmayın
- Bilmiyorsanız, dürüstçe söyleyin
- Yanıtları kısa ama bilgilendirici tutun
- Doğal, konuşma dili kullanın"""
        }
    
    def _default_intent_prompts(self) -> Dict[str, str]:
        """Default intent-specific prompt additions."""
        return {
            'needs_restaurant': """
Focus on restaurant recommendations from the provided database.
Include: name, cuisine type, location/district, price range, rating.
Mention dietary options if relevant (vegetarian, halal, seafood, etc.).
Provide 2-3 specific recommendations.""",

            'needs_attraction': """
Focus on attractions and cultural sites from the provided context.
Include: name, location, description, opening hours, ticket prices.
Prioritize based on user interests and location.
Mention historical significance where relevant.""",

            'needs_transportation': """
Provide clear, step-by-step transportation directions.
Include: metro lines, bus numbers, ferry routes, tram lines.
Mention transfer points and approximate travel times.
Reference the map if one is provided.""",

            'needs_neighborhood': """
Describe the neighborhood's character and atmosphere.
Include: vibe, best times to visit, what it's known for.
Mention nearby attractions, dining, and shopping.
Give practical tips for visitors.""",

            'needs_events': """
Focus on current and upcoming events and activities.
Include: event name, date/time, location, price if applicable.
Prioritize cultural experiences and authentic local events.
Mention booking requirements if needed.""",

            'needs_weather': """
Provide weather-aware recommendations.
Include current conditions in your advice.
Suggest indoor alternatives for bad weather.
Recommend outdoor activities for good weather.
Mention what to wear/bring.""",

            'needs_hidden_gems': """
Focus on authentic, off-the-beaten-path locations.
Include lesser-known spots away from tourist crowds.
Mention what makes each place special.
Provide tips on best times to visit and how to get there.""",

            'needs_map': """
Reference the provided map visualization in your response.
Guide the user on how to use the map.
Mention key landmarks visible on the map.""",

            'needs_gps_routing': """
Provide turn-by-turn navigation guidance.
Start from the user's current location.
Include estimated walking/transit time.
Reference the map for visual guidance.""",

            'needs_translation': """
Provide accurate translations with pronunciation guides.
Include cultural context where relevant.
Explain when/how to use phrases appropriately."""
        }
    
    def build_prompt(
        self,
        query: str,
        signals: Dict[str, bool],
        context: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> str:
        """
        Build complete optimized prompt.
        
        Args:
            query: User query
            signals: Detected signals
            context: Built context (database, RAG, services)
            conversation_context: Conversation history
            language: Response language
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # 1. System prompt
        system_prompt = self.system_prompts.get(language, self.system_prompts['en'])
        prompt_parts.append(system_prompt)
        
        # 2. Intent-specific instructions
        active_signals = [k for k, v in signals.items() if v]
        if active_signals:
            intent_instructions = self._build_intent_instructions(active_signals)
            if intent_instructions:
                prompt_parts.append("\n## Special Instructions:")
                prompt_parts.append(intent_instructions)
        
        # 3. Conversation context (if available)
        if conversation_context:
            conv_formatted = self._format_conversation_context(conversation_context)
            if conv_formatted:
                prompt_parts.append("\n## Previous Conversation:")
                prompt_parts.append(conv_formatted)
        
        # 4. Database context
        if context.get('database'):
            prompt_parts.append("\n## Database Information:")
            prompt_parts.append(context['database'])
        
        # 5. RAG context
        if context.get('rag'):
            prompt_parts.append("\n## Additional Context:")
            prompt_parts.append(context['rag'])
        
        # 6. Service context (weather, events, hidden gems)
        service_context = self._format_service_context(context.get('services', {}))
        if service_context:
            prompt_parts.append("\n## Real-Time Information:")
            prompt_parts.append(service_context)
        
        # 7. Map reference (if available)
        if context.get('map_data'):
            prompt_parts.append("\n## Map Visualization:")
            prompt_parts.append("A visual map has been generated and will be shown to the user.")
            prompt_parts.append("Reference this map in your response to help guide the user.")
        
        # 8. User query
        prompt_parts.append(f"\n## User Question:\n{query}")
        
        # 9. Response instructions
        response_instructions = self._get_response_instructions(language, signals)
        prompt_parts.append(f"\n## Response:\n{response_instructions}")
        
        # Join all parts
        full_prompt = "\n".join(prompt_parts)
        
        logger.debug(f"Built prompt: {len(full_prompt)} chars")
        
        return full_prompt
    
    def _build_intent_instructions(self, active_signals: List[str]) -> str:
        """Build intent-specific instructions."""
        instructions = []
        
        for signal in active_signals:
            if signal in self.intent_prompts:
                instructions.append(self.intent_prompts[signal])
        
        return "\n".join(instructions) if instructions else ""
    
    def _format_conversation_context(
        self,
        conversation_context: Dict[str, Any]
    ) -> str:
        """Format conversation history for prompt."""
        if not conversation_context or not conversation_context.get('history'):
            return ""
        
        formatted = []
        history = conversation_context['history']
        
        for turn in history[-3:]:  # Last 3 turns
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            
            if role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        return "\n".join(formatted) if formatted else ""
    
    def _format_service_context(self, services: Dict[str, Any]) -> str:
        """Format service context (weather, events, etc.)."""
        if not services:
            return ""
        
        formatted = []
        
        # Weather
        if 'weather' in services:
            formatted.append(f"Weather: {services['weather']}")
        
        # Events
        if 'events' in services:
            formatted.append(f"Events:\n{services['events']}")
        
        # Hidden Gems
        if 'hidden_gems' in services:
            formatted.append(f"Hidden Gems:\n{services['hidden_gems']}")
        
        return "\n\n".join(formatted) if formatted else ""
    
    def _get_response_instructions(
        self,
        language: str,
        signals: Dict[str, bool]
    ) -> str:
        """Get response format instructions."""
        if language == 'tr':
            base = "Lütfen Türkçe olarak yanıt verin."
        else:
            base = "Please respond in English."
        
        # Add signal-specific instructions
        if signals.get('needs_map') or signals.get('needs_gps_routing'):
            base += " Reference the provided map to help guide the user."
        
        if signals.get('needs_transportation'):
            base += " Provide step-by-step directions."
        
        if signals.get('needs_restaurant'):
            base += " Recommend 2-3 specific restaurants with details."
        
        return base
    
    def build_few_shot_prompt(
        self,
        query: str,
        examples: List[Dict[str, str]],
        context: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """
        Build prompt with few-shot examples.
        
        Args:
            query: User query
            examples: List of {'query': ..., 'response': ...}
            context: Optional context
            language: Language code
            
        Returns:
            Few-shot prompt
        """
        prompt_parts = []
        
        # System prompt
        system_prompt = self.system_prompts.get(language, self.system_prompts['en'])
        prompt_parts.append(system_prompt)
        
        # Few-shot examples
        if examples:
            prompt_parts.append("\n## Examples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"User: {example['query']}")
                prompt_parts.append(f"Assistant: {example['response']}")
        
        # Context
        if context:
            prompt_parts.append(f"\n## Context:\n{context}")
        
        # User query
        prompt_parts.append(f"\n## User Question:\n{query}")
        prompt_parts.append("\n## Response:")
        
        return "\n".join(prompt_parts)
    
    def build_chain_of_thought_prompt(
        self,
        query: str,
        context: Optional[str] = None,
        language: str = "en"
    ) -> str:
        """
        Build prompt for chain-of-thought reasoning.
        
        Args:
            query: User query
            context: Optional context
            language: Language code
            
        Returns:
            Chain-of-thought prompt
        """
        if language == 'tr':
            thinking_instruction = "Önce adım adım düşünün, sonra yanıt verin."
        else:
            thinking_instruction = "Let's think step by step, then provide your answer."
        
        prompt_parts = [
            self.system_prompts.get(language, self.system_prompts['en']),
            f"\n## Approach:\n{thinking_instruction}"
        ]
        
        if context:
            prompt_parts.append(f"\n## Context:\n{context}")
        
        prompt_parts.append(f"\n## Question:\n{query}")
        prompt_parts.append("\n## Reasoning:")
        
        return "\n".join(prompt_parts)
    
    def optimize_prompt_length(
        self,
        prompt: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Optimize prompt length to fit within token limits.
        
        Args:
            prompt: Original prompt
            max_tokens: Maximum allowed tokens
            
        Returns:
            Optimized prompt
        """
        # Simple character-based approximation (1 token ≈ 4 chars)
        max_chars = max_tokens * 4
        
        if len(prompt) <= max_chars:
            return prompt
        
        # Truncate context sections intelligently
        # TODO: Implement smarter truncation (preserve system prompt, truncate context)
        logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {max_chars}")
        
        return prompt[:max_chars] + "\n\n[Context truncated for length]"
    
    def add_safety_guidelines(self, prompt: str, language: str = "en") -> str:
        """
        Add safety and ethical guidelines to prompt.
        
        Args:
            prompt: Base prompt
            language: Language code
            
        Returns:
            Prompt with safety guidelines
        """
        if language == 'tr':
            safety = """
## Güvenlik Kuralları:
- Zararlı, yasadışı veya uygunsuz içerik sağlamayın
- Kültürel hassasiyetlere saygı gösterin
- Kişisel bilgi istemeyin veya paylaşmayın
- Tıbbi, hukuki veya finansal tavsiye vermeyin"""
        else:
            safety = """
## Safety Guidelines:
- Do not provide harmful, illegal, or inappropriate content
- Respect cultural sensitivities
- Do not request or share personal information
- Do not provide medical, legal, or financial advice"""
        
        return f"{prompt}\n{safety}"
