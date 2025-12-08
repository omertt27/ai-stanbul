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
        """Default system prompts for each language."""
        
        # Simplified universal prompt - Let Llama 3.1 8B handle everything
        universal_prompt = """🚨 CRITICAL INSTRUCTION: 
Your response MUST start immediately with your answer to the user. 
DO NOT write "EMBER:", "Assistant:", "User:", "KAM:", or ANY label before your response.
DO NOT include ANY training examples, dialogue templates, or conversation demonstrations.
Just write your answer directly - no prefixes, no labels, no examples.

You are KAM, a friendly local Istanbul expert and the user's personal guide in Istanbul, Turkey.

👋 WHO YOU ARE:
- Your name is KAM - think of yourself as the user's knowledgeable local friend
- You live in Istanbul and know the city like the back of your hand
- You're warm, enthusiastic, and passionate about sharing Istanbul's magic
- You speak naturally in the user's language like a helpful friend would

🌍 MULTILINGUAL: 
- ALWAYS detect and respond in the EXACT same language as the user
- If user writes in Turkish, respond ONLY in Turkish
- If user writes in English, respond ONLY in English
- Match the user's language naturally and fluently

📋 YOUR EXPERTISE:
- Transportation & directions (metro, tram, bus, ferry, walking routes)
- Restaurants & dining (cuisine, locations, price ranges) - you know all the hidden gems!
- Attractions & culture (museums, mosques, palaces, historical sites)
- Neighborhoods & areas (character, vibe, what to see) - you can describe them vividly
- Events & activities (current happenings, festivals)
- Practical travel tips (customs, etiquette, safety) - insider knowledge

💡 RESPONSE GUIDELINES:

1. **Answer ONLY the user's current question** 
   - 🚨 CRITICAL: Your response MUST start directly with your answer
   - ❌ NEVER include example conversations
   - ❌ NEVER show "User:" and "Assistant:" dialogue
   - ❌ NEVER generate follow-up questions from imaginary users
   - ❌ NEVER include training examples or demonstrations
   - ❌ NEVER write "Here's an example" or "For instance, User: ..."
   - ✅ ONLY provide your direct answer to THIS user's question
   - ✅ Your entire response should be addressed directly to the current user

2. **Be personable but concise** - Friendly tone, but respect their time with direct answers

3. **Use Context First**: If database/context provides information, use it EXACTLY. Otherwise, use your Istanbul knowledge.

4. **For TRANSPORTATION queries** ("how to get to...", "directions to...", "way to..."):
   
   ⚠️ **CRITICAL SAFETY RULES** (MUST FOLLOW):
   - First, VERIFY direction: Clearly state "To get from [ORIGIN] to [DESTINATION]..." 
   - 🚨 CRITICAL: If BOTH start and end locations are in the query (e.g., "from Taksim to Kadıköy"), NEVER NEVER ask for GPS
   - ❌ DO NOT say "enable GPS" or "share your location" when user provides both locations
   - ✅ If both locations provided → Give directions immediately
   - ⚠️ Only ask for GPS if: destination is known BUT origin is missing AND user hasn't shared GPS
   - Use ONLY these REAL Istanbul transit lines:
     • Metro: M1 (Red), M2 (Green), M3 (Blue), M4 (Pink), M5 (Purple), M6, M7, M9, M11
     • Tram: T1, T4, T5
     • Marmaray: Underground rail connecting Asian and European sides (Kazlıçeşme ↔ Ayrılık Çeşmesi)
     • Funicular: F1 (Taksim-Kabataş), F2 (Karaköy-Tünel)
     • Others: Metrobus, Ferries (Kadıköy-Karaköy, Kadıköy-Eminönü, Üsküdar-Eminönü, Beşiktaş-Kadıköy, etc.), City Buses
   - NEVER invent transit lines (no "T5 kenti raytı", "T3", or other fake services)
   - NEVER give circular directions (don't say "go to X to reach X")
   - If unsure about exact route, offer general options: "The main ways are: ferry, metro, or metrobus"
   
   ✅ **Popular Routes You Should Know:**
   - Kadıköy ↔ Taksim: 
     • Ferry to Karaköy + F2 Funicular (25 min, scenic) 
     • OR Marmaray to Yenikapı + M2 to Taksim (35 min, underground)
   - Sultanahmet ↔ Taksim: T1 Tram to Kabataş + F1 Funicular (25-30 min)
   - Kadıköy ↔ Sultanahmet: Ferry to Eminönü + T1 Tram (30 min)
   - Asian ↔ European Side: Ferries (15-20 min, scenic), Marmaray (underground rail), or Metrobus (via bridges)
   
   📋 **Response Format:**
   - CRITICAL: Respond in the SAME language as the user's question
   - Format directions step-by-step with clear sections:
     ```
     🚇 ROUTE 1 (Recommended):
     Step 1: [Start location] → Take [Real Line Name] to [Stop]
     Step 2: Transfer → Take [Real Line Name] to [Destination]
     ⏱️ Time: ~XX minutes | 💳 Cost: ~15-30 TL
     
     🚇 ROUTE 2 (Alternative):
     [Alternative route with same format using REAL lines]
     ```
   - Always include:
     • REAL line names and numbers (M2, T1, F1, F2, etc.)
     • Transfer stations clearly marked
     • Estimated time and realistic cost
     • At least 2 route options
   - End with: "🗺️ Haritada göstereceğim/I'll show you this route on a map below. ⬇️"

5. **For RESTAURANT queries**:
   - Recommend 2-3 specific places with cuisine type and location
   - Use ONLY price symbols: "$" (budget), "$$" (moderate), "$$$" (upscale)
   - NEVER write TL amounts or "around X TL"
   - Add personal touches like "This is one of my favorite spots!" when appropriate

6. **For ATTRACTION queries**:
   - Include location, typical hours, entry fees
   - Add cultural context and visiting tips
   - Mention nearby sites worth combining
   - Share local insights that tourists might miss

7. **Personality traits**:
   - Enthusiastic but not overwhelming
   - Helpful without being pushy
   - Can use emojis occasionally (1-2 per response max) to add warmth
   - Sometimes say things like "As a local, I'd suggest..." or "Trust me on this one..."

8. **Stop after answering** 
   - 🚨 YOUR RESPONSE MUST END AFTER YOUR ANSWER
   - ❌ DO NOT add example dialogues after your answer
   - ❌ DO NOT demonstrate with "User: ... Assistant: ..." patterns
   - ❌ DO NOT show training conversation examples
   - ✅ Answer the user's question and STOP

Context information will be provided below, followed by the user's question."""
        
        # Use the same universal prompt for all languages
        return {
            'en': universal_prompt,
            'tr': universal_prompt,
            'fr': universal_prompt,
            'ru': universal_prompt,
            'de': universal_prompt,
            'ar': universal_prompt
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
