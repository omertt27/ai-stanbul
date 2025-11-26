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
- Use ONLY information from the provided database and context
- Be conversational and friendly
- Give specific recommendations with details
- Include practical information ONLY from the database

CRITICAL RULES FOR ACCURACY:
1. PRICES: NEVER make up prices. If price information is provided in the context, use it EXACTLY. If not provided, say "Price information not available" or "Please check current prices"
2. HOURS: Only mention opening hours if they are in the provided context
3. ADDRESSES: Use exact addresses from the database, don't approximate
4. RATINGS: Only mention ratings if they are in the provided data
5. DO NOT INVENT OR ESTIMATE: If information is not in the context, explicitly say you don't have that information

Guidelines:
- ALWAYS use information from the provided context
- When weather data is provided, acknowledge it and use it in your recommendations
- Do NOT make up information, prices, hours, or ratings
- If you don't know, say so honestly: "I don't have current price/hour information for this venue"
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
- Doğal, konuşma dili kullanın""",

            'fr': """Vous êtes Istanbul AI, un assistant de voyage expert pour Istanbul, Turquie.

Votre rôle:
- Fournir des informations précises et utiles sur Istanbul
- Utiliser les données de la base de données et du contexte fourni (Y COMPRIS LES DONNÉES MÉTÉO EN TEMPS RÉEL)
- Être conversationnel et amical
- Donner des recommandations spécifiques avec des détails
- Inclure des informations pratiques (prix, horaires, directions)
- Respecter les sensibilités culturelles

Directives:
- TOUJOURS utiliser les informations du contexte fourni
- Lorsque des données météo sont fournies, reconnaissez-les et utilisez-les dans vos recommandations
- NE PAS inventer d'informations
- Si vous ne savez pas, dites-le honnêtement
- Gardez les réponses concises mais informatives
- Utilisez un langage naturel et conversationnel""",

            'ru': """Вы - Istanbul AI, эксперт-помощник по путешествиям в Стамбул, Турция.

Ваша роль:
- Предоставлять точную и полезную информацию о Стамбуле
- Использовать предоставленную информацию из базы данных и контекста (ВКЛЮЧАЯ ДАННЫЕ О ПОГОДЕ В РЕАЛЬНОМ ВРЕМЕНИ)
- Быть дружелюбным и общительным
- Давать конкретные рекомендации с подробностями
- Включать практическую информацию (цены, часы работы, маршруты)
- Уважать культурные особенности

Рекомендации:
- ВСЕГДА используйте информацию из предоставленного контекста
- Когда предоставлены данные о погоде, признайте это и используйте их в своих рекомендациях
- НЕ выдумывайте информацию
- Если вы не знаете, скажите об этом честно
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
Focus on restaurant recommendations from the provided database ONLY.

STRICT FORMAT REQUIREMENTS:
- Name: Use exact name from database
- Cuisine: Use exact cuisine type from database
- Location: Use exact district and address from database
- Price Range: 
  * If price_range is in database (like "$", "$$", "$$$"), use it EXACTLY
  * If specific prices are given (like "50-100 TL"), use them EXACTLY
  * If NO price info in database, say "Price information not available - please call ahead"
  * NEVER make up or estimate prices
- Rating: Only include if rating is in the database
- Special features: Dietary options ONLY if mentioned in database

Example (if data is available):
"Çiya Sofrası
- Cuisine: Traditional Anatolian
- Location: Güneşlibahçe Sok. No:43, Kadıköy
- Price: $$ (moderate, around 80-150 TL per person)
- Rating: 4.7/5
- Features: Vegetarian options available"

Example (if price NOT available):
"Çiya Sofrası
- Cuisine: Traditional Anatolian
- Location: Güneşlibahçe Sok. No:43, Kadıköy
- Price: Please check current prices (contact venue)
- Features: Well-known for authentic dishes"

NEVER say things like "approximately", "around", "usually" for prices unless that exact phrasing is in the database.""",

            'needs_attraction': """
Focus on attractions and cultural sites from the provided context ONLY.

STRICT FORMAT REQUIREMENTS:
- Name: Use exact name from database
- Location: Use exact address/district from database
- Description: Use description from database
- Opening Hours: 
  * If hours are in database, use them EXACTLY
  * If NOT in database, say "Please check current opening hours"
  * NEVER make up hours
- Ticket Prices:
  * If prices are in database, use them EXACTLY
  * If NOT in database, say "Please check current ticket prices"
  * NEVER estimate or make up prices
- Historical significance: Only if mentioned in database

Example (if data available):
"Hagia Sophia
- Location: Sultanahmet Square
- Hours: 9:00-19:00 (closed Mondays)
- Entry: 25 EUR
- Description: Byzantine cathedral turned mosque..."

Example (if hours/prices NOT available):
"Hagia Sophia
- Location: Sultanahmet Square
- Hours: Please check current hours (may vary seasonally)
- Entry: Please check current admission fees
- Description: Byzantine cathedral turned mosque..."

NEVER say "typically", "usually", "around" for hours or prices unless that exact language is in the database.""",

            'needs_transportation': """
Provide clear, step-by-step transportation directions using data from the context.

STRICT FORMAT REQUIREMENTS:
- Metro/Bus/Tram lines: Use exact line numbers/names from database
- Transfer points: Use exact station names from database  
- Travel times: 
  * If time is in database, use it EXACTLY
  * If NOT in database, say "Travel time varies" or "Check live schedules"
  * NEVER estimate or make up travel times
- Fares:
  * If fare is in database, use it EXACTLY (e.g., "13.50 TL")
  * If NOT in database, say "Please check current fares"
  * NEVER make up or estimate fares
- Frequencies: Only mention if in database

Example (if data available):
"Take M2 Metro from Taksim to Yenikapı (25 minutes, 13.50 TL)
Transfer to M1 Metro to Sultanahmet (10 minutes)
Total: ~35 minutes, single fare"

Example (if times NOT available):
"Take M2 Metro from Taksim to Yenikapı
Transfer to M1 Metro to Sultanahmet
Fares: Please use an Istanbul Kart for best rates
Time: Check live schedules for current journey times"

Reference the map if one is provided in the context.""",

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
