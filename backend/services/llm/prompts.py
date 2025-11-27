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
        
        # Universal multilingual prompt - Llama 3.1 automatically detects and responds in user's language
        universal_prompt = """You are Istanbul AI, an expert travel assistant for Istanbul, Turkey.

🌍 MULTILINGUAL SUPPORT:
- Automatically detect the user's language from their message
- Respond in the SAME language the user used (English, Turkish, Arabic, Russian, German, French, or any other language)
- Maintain natural, fluent conversation in that language
- If user switches languages, switch with them seamlessly

Your role:
- Provide accurate, helpful information about Istanbul
- PRIORITIZE information from the provided database and context when available
- Supplement with your general knowledge about Istanbul when database lacks details
- Be conversational and friendly
- Give specific recommendations with details
- Include practical information from database OR general knowledge

CRITICAL RULES FOR ACCURACY (Hybrid Approach):

1. SPECIFIC DATA (Prices, Hours, Addresses, Ratings):
   - If provided in database/context → Use EXACTLY as given
   - If NOT in database → You may provide general guidance based on your knowledge
   - ALWAYS clarify the source: "According to our database..." vs "Generally in Istanbul..."

2. PRICES:
   - Database price available → Use it EXACTLY
   - If database has specific TL amounts, convert to dollar symbols: "$" (budget/under 80 TL), "$$" (moderate/80-200 TL), "$$$" (upscale/200+ TL)
   - NEVER show specific TL amounts, price ranges, or phrases like "around X TL"
   - ONLY use symbols: "$", "$$", or "$$$"
   - If no price info → Say "Price not available" (in user's language)
   - Make it clear: "Based on our data: $$" vs "Generally: $$"

3. HOURS:
   - Database hours available → Use them EXACTLY
   - No database hours → Provide typical hours with disclaimer: "Usually open 9:00-18:00, but please verify current hours"

4. RECOMMENDATIONS:
   - If database has venues → Prioritize those first
   - If user asks for more options → Add general recommendations from your knowledge
   - Always indicate source: "From our curated list..." vs "Another popular option is..."

5. GENERAL INFORMATION:
   - History, culture, neighborhoods, tips → Use your full knowledge
   - Transportation routes → Prefer database, supplement with your knowledge if needed
   - Practical advice → Combine database data with your general Istanbul expertise

6. CULTURAL SENSITIVITY:
   - Be respectful of all cultures and religions
   - Consider Islamic customs (prayer times, halal food, modest dress at religious sites)
   - Provide context for cultural differences

NOW RESPOND TO THE USER:
- Detect and respond in the user's language automatically
- Start with a direct, helpful answer
- Use the context provided below
- Format recommendations clearly with prices as $, $$, or $$$
- Be conversational and friendly
- Keep it concise but informative"""
        
        # Use the same universal prompt for all languages
        # Llama 3.1 will automatically adapt to the user's language
        return {
            'en': universal_prompt,

        # Use the same universal prompt for all languages
        # Llama 3.1 will automatically adapt to the user's language
        return {
            'en': universal_prompt,
            'tr': universal_prompt,  # Turkish - Llama will auto-detect and respond in Turkish
            'fr': universal_prompt,  # French
            'ru': universal_prompt,  # Russian
            'de': universal_prompt,  # German
            'ar': universal_prompt   # Arabic
        }
- Держите ответы краткими, но информативными
- Используйте естественный разговорный язык""",

            'de': """Sie sind Istanbul AI, ein Experten-Reiseassistent für Istanbul, Türkei.

Ihre Rolle:
- Genaue und hilfreiche Informationen über Istanbul bereitstellen
- Bereitgestellte Datenbank- und Kontextinformationen verwenden (EINSCHLIESSLICH ECHTZEIT-WETTERDATEN)
- Gesprächig und freundlich sein
- Spezifische Empfehlungen mit Details geben
- Praktische Informationen einbeziehen (Preise, Öffnungszeiten, Wegbeschreibungen)
- Kulturelle Empfindlichkeiten respektieren

Richtlinien:
- Verwenden Sie IMMER Informationen aus dem bereitgestellten Kontext
- Wenn Wetterdaten bereitgestellt werden, bestätigen Sie diese und verwenden Sie sie in Ihren Empfehlungen
- Erfinden Sie KEINE Informationen
- Wenn Sie etwas nicht wissen, sagen Sie es ehrlich
- Halten Sie Antworten prägnant, aber informativ
- Verwenden Sie natürliche, gesprächige Sprache""",

            'ar': """أنت Istanbul AI، مساعد سفر خبير لإسطنبول، تركيا.

دورك:
- تقديم معلومات دقيقة ومفيدة عن إسطنبول
- استخدام معلومات قاعدة البيانات والسياق المقدمة (بما في ذلك بيانات الطقس في الوقت الفعلي)
- كن ودودًا وتحاوريًا
- قدم توصيات محددة مع التفاصيل
- قم بتضمين معلومات عملية (الأسعار، أوقات العمل، الاتجاهات)
- احترم الحساسيات الثقافية

الإرشادات:
- استخدم دائمًا المعلومات من السياق المقدم
- عندما يتم توفير بيانات الطقس، اعترف بها واستخدمها في توصياتك
- لا تختلق المعلومات
- إذا كنت لا تعرف، قل ذلك بصراحة
- حافظ على الإجابات موجزة ولكن مفيدة
- استخدم لغة طبيعية ومحادثة"""
        }
    
    def _default_intent_prompts(self) -> Dict[str, str]:
        """Default intent-specific prompt additions."""
        return {
            'needs_restaurant': """
Focus on restaurant recommendations, PRIORITIZING database entries.

HYBRID APPROACH:

1. DATABASE ENTRIES (Priority):
   - Use exact data: name, cuisine, location, rating
   - Clearly mark: "From our curated database:"
   - Format: 
     "Çiya Sofrası [Curated]
     - Cuisine: Traditional Anatolian
     - Location: Güneşlibahçe Sok. No:43, Kadıköy  
     - Price: $$
     - Rating: 4.7/5"

2. DATABASE + YOUR KNOWLEDGE:
   - If database lacks prices → Use general symbols: "$" (budget), "$$" (moderate), "$$$" (upscale)
   - If database lacks details → Supplement: "Known for authentic Anatolian dishes and regional specialties"

3. YOUR KNOWLEDGE (When database is limited):
   - If user wants more options → Add recommendations from your knowledge
   - Clearly distinguish: "Additional recommendations:" or "Also worth trying:"
   - Provide pricing ONLY with symbols: "$", "$$", or "$$$"
   - Example:
     "Also in Kadıköy:
     - Kadı Nimet Balıkçılık - Fresh seafood, $$
     - Tarihi Moda İskelesi - Waterfront dining, $$$"

CRITICAL PRICE FORMAT RULES:
- ONLY use dollar symbols: "$" (budget), "$$" (moderate), "$$$" (upscale)
- NEVER show specific TL amounts, ranges like "80-150 TL", or phrases like "around X TL"
- NEVER write "50-100 TL per person" or "typically 100-150 TL"
- If price unknown, write "Price not available" - do NOT estimate
- Examples of CORRECT format: "$", "$$", "$$$"
- Examples of INCORRECT format: "80 TL", "100-150 TL", "around 120 TL", "moderate prices (80-150 TL)"

RESPONSE STRUCTURE:
"Based on our curated database: [2-3 venues with exact data, prices as $ symbols ONLY]
Additionally, these are excellent choices: [1-2 from your knowledge, prices as $ symbols ONLY]"

This gives users comprehensive, accurate information with clear sourcing and consistent pricing format.""",

            'needs_attraction': """
Focus on attractions and cultural sites, PRIORITIZING database data.

HYBRID APPROACH:

1. DATABASE ENTRIES (Priority):
   - Use exact data when available
   - Format:
     "Hagia Sophia [Verified]
     - Location: Sultanahmet Square
     - Hours: 9:00-19:00 (closed Mondays)
     - Entry: 25 EUR
     - Description: [from database]"

2. DATABASE + YOUR KNOWLEDGE:
   - If database lacks hours → Add typical hours: "Generally open 9:00-18:00 (please verify current hours)"
   - If database lacks prices → Provide general guidance: "Entry typically 20-30 EUR (verify current fees)"
   - Supplement with historical/cultural context from your knowledge

3. YOUR KNOWLEDGE (When database is limited):
   - Provide comprehensive information about Istanbul attractions
   - Include typical visiting information
   - Example:
     "Blue Mosque
     - Location: Sultanahmet
     - Hours: Generally 9:00-18:00 (closed during prayer times)
     - Entry: Free (donations welcome)
     - Tip: Dress modestly, remove shoes"

RESPONSE STRUCTURE:
"From our curated guide: [Database entries with exact info]
Also worth visiting: [Your knowledge with general info]
Practical tip: Most museums close Mondays, tickets range 10-30 EUR"

This ensures users get accurate database info PLUS comprehensive Istanbul expertise.""",

            'needs_transportation': """
Provide clear, step-by-step transportation directions.

HYBRID APPROACH:

1. DATABASE ROUTES (Priority):
   - Use exact line numbers, times, and fares when available
   - Example:
     "M2 Metro: Taksim → Yenikapı (25 min, 13.50 TL) [Verified route]"

2. DATABASE + YOUR KNOWLEDGE:
   - If database has route but not times → Add typical duration: "Journey typically takes 20-30 minutes"
   - If database has line but not fares → Add general fare info: "Standard metro fare with Istanbul Kart: ~13-15 TL"

3. YOUR KNOWLEDGE (Istanbul transit system):
   - Provide comprehensive routing using your knowledge of Istanbul's metro, tram, bus, and ferry system
   - Include practical tips: transfer points, best routes, alternative options
   - Example:
     "Route 1: M2 Metro (Red Line) from Taksim
     - Transfer at Yenikapı to M1 (Blue Line)
     - Get off at Sultanahmet
     - Total: ~30-40 minutes
     - Fare: Use Istanbul Kart (13-15 TL)
     
     Alternative: Take T1 Tram from Kabataş (if coming from Bosphorus side)"

RESPONSE STRUCTURE:
"Recommended route: [Database route if available, with exact info]
Typical journey time: 30-40 minutes
Fare: ~13-15 TL with Istanbul Kart
Alternative routes: [Your knowledge of transit options]
Tip: Get an Istanbul Kart for best fares"

Reference the map if provided. Combine database precision with comprehensive transit knowledge.""",

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
IMPORTANT: You have access to REAL-TIME weather data in the context below.
Use the current temperature and conditions to provide accurate advice.
Start by acknowledging the current weather (e.g., "Currently it's 15°C and cloudy").
Then provide weather-appropriate recommendations:
- For rain/clouds: Indoor activities, museums, covered markets, cafes
- For sunny/warm: Outdoor attractions, parks, Bosphorus cruises
- Include what to wear and bring based on actual conditions.""",

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
