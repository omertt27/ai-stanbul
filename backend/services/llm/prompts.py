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
- Advanced prompt engineering for low-signal scenarios (Phase 4 Priority 3)

Author: AI Istanbul Team
Date: November 2025
"""

import logging
from typing import Dict, Any, Optional, List

# Import Phase 4 Priority 3: Advanced Prompt Engineering
try:
    from .advanced_prompts import get_prompt_engineer
    ADVANCED_PROMPTS_AVAILABLE = True
except ImportError:
    ADVANCED_PROMPTS_AVAILABLE = False
    logger.warning("⚠️ Advanced prompt engineering module not available")

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
        """Simplified system prompts optimized for Llama 3.1 8B."""
        
        # SIMPLIFIED PROMPT - Clear structure, no confusing instructions
        simplified_prompt = """You are KAM, a knowledgeable and friendly Istanbul tour guide.

IMPORTANT: Start your answer immediately. Do NOT write "Assistant:", "KAM:", or any label before your response.

YOUR ROLE:
- You're a local Istanbul expert helping visitors explore the city
- You know all about transportation (metro, tram, ferry), restaurants, attractions, and neighborhoods
- You're warm, helpful, and give specific, actionable advice

RESPONSE RULES:
1. Always respond in the SAME language as the user's question (English, Turkish, etc.)
2. Use information from the CONTEXT below when available - it's factual database data
3. Be specific: Give exact metro lines (M1, M2, T1, F1), restaurant names, addresses
4. For directions: Provide step-by-step routes with real transit lines
5. Keep responses focused and helpful - answer the question directly

TRANSPORTATION LINES (Use only these REAL lines):
- Metro: M1, M2, M3, M4, M5, M6, M7, M9, M11
- Tram: T1, T4, T5
- Funicular: F1 (Taksim-Kabataş), F2 (Karaköy-Tünel)
- Marmaray: Underground rail (Kazlıçeşme to Ayrılık Çeşmesi)
- Ferries: Kadıköy-Karaköy, Kadıköy-Eminönü, Üsküdar-Eminönü, etc.

EXAMPLE GOOD RESPONSES:

Question: "How do I get from Taksim to Sultanahmet?"
Answer: "To get from Taksim to Sultanahmet:

Take the F1 Funicular from Taksim down to Kabataş (2 minutes), then transfer to the T1 Tram and ride it to Sultanahmet stop (about 20 minutes).

Total time: ~25 minutes
Cost: One Istanbulkart tap covers the whole journey

The tram runs along the coast with nice views of the Bosphorus!"

Question: "Best kebab restaurant in Sultanahmet?"
Answer: "I'd recommend Tarihi Sultanahmet Köftecisi - it's a local institution! They serve delicious köfte (meatballs) and kebabs right in the heart of Sultanahmet. It's on Divan Yolu street, very close to the Blue Mosque. Expect to pay around $-$$ (budget-moderate). Get there early to avoid crowds!"

---
CONTEXT FROM DATABASE:
---
CONTEXT FROM DATABASE:
{context}

---
USER'S QUESTION:
{query}

---
YOUR ANSWER (start immediately, no labels):"""
        
        # Use the same simplified prompt for all languages
        return {
            'en': simplified_prompt,
            'tr': simplified_prompt,
            'fr': simplified_prompt,
            'ru': simplified_prompt,
            'de': simplified_prompt,
            'ar': simplified_prompt
        }
    
    def _default_intent_prompts(self) -> Dict[str, str]:
        """Intent-specific prompts - NOT USED with Llama 3.1 8B (LLM handles intent detection)."""
        # Keeping this empty - Llama 3.1 8B is smart enough to understand user intent
        # without explicit signal-based instructions
        return {}
    
    def build_prompt(
        self,
        query: str,
        signals: Dict[str, bool],
        context: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None,
        language: str = "en",
        user_location: Optional[Dict[str, float]] = None,
        enable_intent_classification: bool = False,
        signal_confidence: float = 1.0
    ) -> str:
        """
        Build complete optimized prompt.
        
        Let Llama 3.1 8B handle intent detection naturally from the query.
        We just provide context and let the LLM figure out what the user needs.
        
        Args:
            query: User query
            signals: Detected signals (kept for backwards compatibility, but not heavily used)
            context: Built context (database, RAG, services)
            conversation_context: Conversation history
            language: Response language
            user_location: User's GPS coordinates (if available)
            enable_intent_classification: Enable LLM intent classification (Priority 2)
            signal_confidence: Overall signal detection confidence (Priority 3)
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # 1. System prompt (contains all the intelligence)
        system_prompt = self.system_prompts.get(language, self.system_prompts['en'])
        
        # ADD GPS CONTEXT if available - for ANY location-based query
        if user_location:
            # Check if this is any location-based query (routing, nearby, recommendations, etc.)
            is_location_query = any([
                signals.get('needs_gps_routing'),
                signals.get('needs_directions'),
                signals.get('needs_transportation'),
                signals.get('needs_restaurant'),
                signals.get('needs_attraction'),
                signals.get('needs_hidden_gems'),
                signals.get('needs_shopping'),
                signals.get('needs_nightlife'),
                signals.get('needs_events'),
                signals.get('needs_daily_life'),
                'nearby' in query.lower(),
                'near me' in query.lower(),
                'close to me' in query.lower(),
                'around me' in query.lower(),
                'around here' in query.lower(),
                'how' in query.lower() and ('get' in query.lower() or 'go' in query.lower())
            ])
            
            if is_location_query:
                system_prompt += f"\n\n🌍 **GPS STATUS**: User's current location is AVAILABLE at coordinates ({user_location['lat']}, {user_location['lon']})."
                system_prompt += "\n✅ IMPORTANT: The user HAS GPS enabled. Use their current location for recommendations and directions."
                system_prompt += "\n🚨 DO NOT ask the user to enable GPS or share location - it's already available!"
                system_prompt += "\n📍 When recommending places or giving directions, reference their current location."
        
        prompt_parts.append(system_prompt)
        
        # 2. Conversation context (if available)
        if conversation_context:
            conv_formatted = self._format_conversation_context(conversation_context)
            if conv_formatted:
                prompt_parts.append("\n## Previous Conversation:")
                prompt_parts.append(conv_formatted)
        
        # 3. Database context
        if context.get('database'):
            prompt_parts.append("\n## Database Information:")
            prompt_parts.append(context['database'])
        
        # 4. RAG context
        if context.get('rag'):
            prompt_parts.append("\n## Additional Context:")
            prompt_parts.append(context['rag'])
        
        # 5. Service context (weather, events, hidden gems)
        service_context = self._format_service_context(context.get('services', {}))
        if service_context:
            prompt_parts.append("\n## Real-Time Information:")
            prompt_parts.append(service_context)
        
        # 6. Map reference (if available)
        if context.get('map_data'):
            map_data = context['map_data']
            has_origin = map_data.get('has_origin', False)
            has_destination = map_data.get('has_destination', False)
            origin_name = map_data.get('origin_name')
            destination_name = map_data.get('destination_name')
            
            prompt_parts.append("\n## Map Visualization:")
            prompt_parts.append("A visual map has been generated and will be shown to the user.")
            
            if has_origin and has_destination:
                # Both locations are known - NO GPS NEEDED
                prompt_parts.append(f"\n🚨 CRITICAL INSTRUCTION - MUST FOLLOW:")
                prompt_parts.append(f"Both origin ({origin_name}) and destination ({destination_name}) are EXPLICITLY PROVIDED by the user.")
                prompt_parts.append("✅ The route CAN be shown WITHOUT GPS")
                prompt_parts.append("❌ DO NOT mention GPS, location services, or ask user to enable anything")
                prompt_parts.append("❌ DO NOT say 'I need your current location'")
                prompt_parts.append("✅ INSTEAD: Directly provide the route directions from {origin_name} to {destination_name}")
                prompt_parts.append("The user already told you where they want to go FROM and TO.")
            elif has_destination and not has_origin:
                # Only destination is known - might need GPS
                prompt_parts.append(f"Destination ({destination_name}) is known, but origin is not specified.")
                if 'gps' not in str(context).lower():
                    prompt_parts.append("Consider asking the user for their starting location or to enable GPS.")
            
            prompt_parts.append("Reference this map in your response to help guide the user.")
        
        # 6.5. Add intent classification request (PRIORITY 2) - NEW
        if enable_intent_classification:
            intent_classification_prompt = """

---

🎯 INTENT CLASSIFICATION (Required - Mark first, then answer):
Before answering, identify the user's intents by marking with [X]:

Transportation/Directions: [ ] (how to get somewhere, routes, transit info)
Restaurant Recommendation: [ ] (places to eat, cuisine, dining)
Attraction Information: [ ] (museums, sites, historical places)
Neighborhood/Area Info: [ ] (districts, areas, local info)
Event/Activity Query: [ ] (concerts, festivals, things to do)
Shopping: [ ] (shopping areas, malls, markets)
Nightlife: [ ] (bars, clubs, entertainment)
General Question: [ ] (other queries about Istanbul)

Example:
Query: "how do I get to a good kebab place near Taksim"
Intents: [X] Transportation/Directions [X] Restaurant Recommendation [ ] Attraction Information [ ] Neighborhood/Area Info [ ] Event/Activity Query [ ] Shopping [ ] Nightlife [ ] General Question"""

            prompt_parts.append(intent_classification_prompt)
        
        # 6.6. Add low-confidence signal instructions (PRIORITY 3) - NEW
        if signal_confidence < 0.6:
            low_confidence_prompt = f"""

---

🚨 UNCERTAIN INTENT DETECTED (Confidence: {signal_confidence:.2f})

The user's query may be ambiguous or unclear. Here's what we know:
- Query: "{query}"
- Detected intents: {[k for k, v in signals.items() if v]} (LOW CONFIDENCE)
- User location: {"Available" if user_location else "Not available"}

Please:
1. Carefully analyze the query to infer the user's actual intent
2. Use ALL the provided context below (it may contain relevant information)
3. If truly ambiguous, ask ONE clarifying question (see strategies below)
4. Be helpful even with limited information

The context below may include:
- Restaurants nearby
- Attractions and museums
- Transportation options
- Neighborhood information
- Events and activities
- General Istanbul information

Use whichever context is most relevant to answer the query.

📋 CLARIFYING QUESTION STRATEGIES:
If the query is truly ambiguous, use ONE of these approaches:
- Option-based: "Are you looking for [option A] or [option B]?"
- Specific detail: "What type of [category] are you interested in?"
- Context-seeking: "Could you tell me more about [missing detail]?"
- Location-based: "Which neighborhood/area are you interested in?"

Example: Query "what's around" → "Are you looking for restaurants, attractions, or something else nearby?"""

            prompt_parts.append(low_confidence_prompt)
        
        # 6.7. Add multi-intent query handling (PRIORITY 3) - NEW
        active_signal_count = sum(1 for v in signals.values() if v)
        if active_signal_count >= 2:
            multi_intent_prompt = f"""

---

🎯 MULTI-INTENT QUERY DETECTED ({active_signal_count} intents)

This query involves multiple needs. Active intents: {[k for k, v in signals.items() if v]}

HANDLING STRATEGY:
1. **Identify Primary Intent**: What's the user's MAIN need?
2. **Address Secondary Intents**: Incorporate related information naturally
3. **Structured Response**: Organize your answer into clear sections
4. **Smooth Integration**: Connect different aspects logically

Example structures:
- Restaurant + Transportation: Recommend places THEN explain how to get there
- Attraction + Neighborhood: Describe attraction THEN provide area context
- Shopping + Dining: Suggest shopping areas THEN mention nearby food options

Be comprehensive but concise - address all intents without overwhelming the user."""

            prompt_parts.append(multi_intent_prompt)
        
        # 7. User query - simplified format to prevent template generation
        prompt_parts.append(f"\n---\n\n🚨 REMEMBER: Answer ONLY this user's question directly. Do NOT include example dialogues.\n\nCurrent User Question: {query}\n\nYour Direct Answer:")

        
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
        # Language-specific response instructions
        language_instructions = {
            'en': "Please respond in English.",
            'tr': "Lütfen Türkçe olarak yanıt verin.",
            'fr': "Veuillez répondre en français.",
            'ru': "Пожалуйста, отвечайте на русском языке.",
            'de': "Bitte antworten Sie auf Deutsch.",
            'ar': "يرجى الرد باللغة العربية."
        }
        
        base = language_instructions.get(language, language_instructions['en'])
        
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
        thinking_instructions = {
            'en': "Let's think step by step, then provide your answer.",
            'tr': "Önce adım adım düşünün, sonra yanıt verin.",
            'fr': "Réfléchissons étape par étape, puis fournissez votre réponse.",
            'ru': "Давайте подумаем шаг за шагом, а затем дадим ответ.",
            'de': "Lassen Sie uns Schritt für Schritt denken und dann Ihre Antwort geben.",
            'ar': "دعنا نفكر خطوة بخطوة، ثم قدم إجابتك."
        }
        
        thinking_instruction = thinking_instructions.get(language, thinking_instructions['en'])
        
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
        safety_guidelines = {
            'en': """
## Safety Guidelines:
- Do not provide harmful, illegal, or inappropriate content
- Respect cultural sensitivities
- Do not request or share personal information
- Do not provide medical, legal, or financial advice""",

            'tr': """
## Güvenlik Kuralları:
- Zararlı, yasadışı veya uygunsuz içerik sağlamayın
- Kültürel hassasiyetlere saygı gösterin
- Kişisel bilgi istemeyin veya paylaşmayın
- Tıbbi, hukuki veya finansal tavsiye vermeyin""",

            'fr': """
## Directives de sécurité:
- Ne fournissez pas de contenu nuisible, illégal ou inapproprié
- Respectez les sensibilités culturelles
- Ne demandez pas et ne partagez pas d'informations personnelles
- Ne fournissez pas de conseils médicaux, juridiques ou financiers""",

            'ru': """
## Правила безопасности:
- Не предоставляйте вредный, незаконный или неуместный контент
- Уважайте культурные особенности
- Не запрашивайте и не делитесь личной информацией
- Не давайте медицинских, юридических или финансовых советов""",

            'de': """
## Sicherheitsrichtlinien:
- Bieten Sie keine schädlichen, illegalen oder unangemessenen Inhalte an
- Respektieren Sie kulturelle Empfindlichkeiten
- Fordern Sie keine persönlichen Informationen an und geben Sie keine weiter
- Geben Sie keine medizinischen, rechtlichen oder finanziellen Ratschläge""",

            'ar': """
## إرشادات السلامة:
- لا تقدم محتوى ضار أو غير قانوني أو غير لائق
- احترم الحساسيات الثقافية
- لا تطلب أو تشارك معلومات شخصية
- لا تقدم نصائح طبية أو قانونية أو مالية"""
        }
        
        safety = safety_guidelines.get(language, safety_guidelines['en'])
        
        return f"{prompt}\n{safety}"
