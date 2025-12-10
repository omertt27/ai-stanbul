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
        
        # ENGLISH PROMPT
        english_prompt = """You are KAM, an expert Istanbul tour guide.

⚠️ CRITICAL LANGUAGE RULE: You MUST answer in ENGLISH ONLY. Never use French, Turkish, or any other language.

GUIDELINES:
- Use the information provided in the CONTEXT below
- Be specific with names, metro lines (M1, M2, T1, F1), and locations
- For directions: Give step-by-step transit instructions
- Keep answers focused and practical
- Write ONLY in English - this is mandatory

ISTANBUL TRANSPORTATION:
Metro: M1, M2, M3, M4, M5, M6, M7, M9, M11
Tram: T1, T4, T5
Funicular: F1 (Taksim-Kabataş), F2 (Karaköy-Tünel)
Marmaray: Underground rail crossing Bosphorus
Ferries: Kadıköy-Karaköy, Kadıköy-Eminönü, Üsküdar-Eminönü

Start your answer immediately in ENGLISH without repeating these instructions."""
        
        # TURKISH PROMPT
        turkish_prompt = """Sen KAM, uzman bir İstanbul tur rehberisin.

ÖNEMLİ: Sadece TÜRKÇE cevap ver.

KURALLAR:
- Aşağıdaki BAĞLAM bilgilerini kullan
- Metro hatları (M1, M2, T1, F1) ve yer isimleri belirt
- Yol tarifi için: Adım adım ulaşım talimatları ver
- Cevapları odaklı ve pratik tut

İSTANBUL ULAŞIM:
Metro: M1, M2, M3, M4, M5, M6, M7, M9, M11
Tramvay: T1, T4, T5
Füniküler: F1 (Taksim-Kabataş), F2 (Karaköy-Tünel)
Marmaray: Boğaz'ı geçen yeraltı treni
Vapur: Kadıköy-Karaköy, Kadıköy-Eminönü, Üsküdar-Eminönü

Bu talimatları tekrarlama, cevabını hemen başlat."""
        
        # RUSSIAN PROMPT
        russian_prompt = """Вы KAM, эксперт по Стамбулу.

⚠️ КРИТИЧЕСКОЕ ПРАВИЛО: Вы ДОЛЖНЫ отвечать ТОЛЬКО на РУССКОМ языке. Никогда не используйте английский, турецкий или другие языки.

ПРАВИЛА:
- Используйте информацию из КОНТЕКСТА ниже
- Указывайте конкретные названия, линии метро (M1, M2, T1, F1) и места
- Для маршрутов: Давайте пошаговые инструкции по транспорту
- Держите ответы сфокусированными и практичными
- Пишите ТОЛЬКО на русском - это обязательно

СТАМБУЛЬСКИЙ ТРАНСПОРТ:
Метро: M1, M2, M3, M4, M5, M6, M7, M9, M11
Трамвай: T1, T4, T5
Фуникулер: F1 (Таксим-Кабаташ), F2 (Каракёй-Тюнель)
Мармарай: Подземная железная дорога через Босфор
Паромы: Кадыкёй-Каракёй, Кадыкёй-Эминёню, Ускюдар-Эминёню

Начните свой ответ сразу на РУССКОМ языке, не повторяя эти инструкции."""

        # GERMAN PROMPT
        german_prompt = """Sie sind KAM, ein Istanbul-Experte.

⚠️ KRITISCHE SPRACHREGEL: Sie MÜSSEN NUR auf DEUTSCH antworten. Verwenden Sie niemals Englisch, Türkisch oder andere Sprachen.

RICHTLINIEN:
- Verwenden Sie die Informationen aus dem KONTEXT unten
- Seien Sie spezifisch mit Namen, Metrolinien (M1, M2, T1, F1) und Orten
- Für Wegbeschreibungen: Geben Sie schrittweise Verkehrsanweisungen
- Halten Sie Antworten fokussiert und praktisch
- Schreiben Sie NUR auf Deutsch - dies ist obligatorisch

ISTANBULER VERKEHR:
Metro: M1, M2, M3, M4, M5, M6, M7, M9, M11
Straßenbahn: T1, T4, T5
Seilbahn: F1 (Taksim-Kabataş), F2 (Karaköy-Tünel)
Marmaray: Unterirdische Bahn über den Bosporus
Fähren: Kadıköy-Karaköy, Kadıköy-Eminönü, Üsküdar-Eminönü

Beginnen Sie Ihre Antwort sofort auf DEUTSCH, ohne diese Anweisungen zu wiederholen."""

        # ARABIC PROMPT
        arabic_prompt = """أنت KAM، خبير في إسطنبول.

⚠️ قاعدة لغوية حاسمة: يجب أن تجيب باللغة العربية فقط. لا تستخدم أبداً الإنجليزية أو التركية أو أي لغة أخرى.

إرشادات:
- استخدم المعلومات المقدمة في السياق أدناه
- كن محدداً مع الأسماء وخطوط المترو (M1، M2، T1، F1) والمواقع
- للاتجاهات: قدم تعليمات النقل خطوة بخطوة
- اجعل الإجابات مركزة وعملية
- اكتب بالعربية فقط - هذا إلزامي

النقل في إسطنبول:
مترو: M1، M2، M3، M4، M5، M6، M7، M9، M11
ترام: T1، T4، T5
قطار جبلي مائل: F1 (تقسيم-كاباتاش)، F2 (كاراكوي-تونيل)
مرمراي: قطار تحت الأرض يعبر البوسفور
عبارات: كاديكوي-كاراكوي، كاديكوي-إمينونو، أوسكودار-إمينونو

ابدأ إجابتك فوراً بالعربية دون تكرار هذه التعليمات."""
        
        # REMOVED: French language support (causes confusion with LLM)
        # We only support: English, Turkish, Russian, German, Arabic
        return {
            'en': english_prompt,
            'tr': turkish_prompt,
            'ru': russian_prompt,
            'de': german_prompt,
            'ar': arabic_prompt
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
        if user_location and isinstance(user_location, dict) and 'lat' in user_location and 'lon' in user_location:
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
                try:
                    lat = float(user_location['lat'])
                    lon = float(user_location['lon'])
                    
                    # Try to identify if user is on Asian or European side
                    side = "Asian" if lon > 29.05 else "European"
                    
                    # ROUTING QUERIES GET EXTRA DETAILED GPS CONTEXT
                    is_routing = signals.get('needs_gps_routing') or signals.get('needs_directions') or ('how' in query.lower() and any(w in query.lower() for w in ['get', 'go', 'reach']))
                    
                    if is_routing:
                        system_prompt += f"\n\nGPS ROUTING REQUEST:"
                        system_prompt += f"\nUser starting location: {lat:.5f}, {lon:.5f} ({side} side of Istanbul)"
                        system_prompt += f"\nGive specific step-by-step transit directions from this GPS point."
                    else:
                        system_prompt += f"\n\nUser GPS location: {lat}, {lon} ({side} side)"
                        system_prompt += f"\nUse for nearby recommendations."
                except (ValueError, TypeError):
                    # Invalid coordinates - skip GPS status
                    pass
        
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
            
            prompt_parts.append("\n## Map:")
            prompt_parts.append("A map will be shown to the user.")
            
            if has_origin and has_destination:
                prompt_parts.append(f"\nRoute: {origin_name} to {destination_name}")
                prompt_parts.append(f"Provide step-by-step transit directions with specific metro/tram lines.")
            elif has_destination and not has_origin:
                prompt_parts.append(f"Destination: {destination_name}")
            
            prompt_parts.append("Mention the map in your response.")
        
        # DISABLED: Intent classification, low-confidence, and multi-intent prompts cause template artifacts
        # These features are currently disabled to keep responses clean and focused
        
        # 7. Language reminder + User query
        # Add STRONG explicit language reminder right before the answer section
        # REMOVED French language support - it was causing LLM confusion
        lang_name_map = {
            'en': 'ENGLISH',
            'tr': 'TURKISH (Türkçe)',
            'ru': 'RUSSIAN (Русский)',
            'de': 'GERMAN (Deutsch)',
            'ar': 'ARABIC (العربية)'
        }
        lang_name = lang_name_map.get(language, 'ENGLISH')
        
        # Add multiple language reminders for maximum enforcement
        prompt_parts.append(f"\n---\n\n⚠️ CRITICAL: Your response MUST be written ONLY in {lang_name}.")
        prompt_parts.append(f"❌ DO NOT use any other language. Write in {lang_name} only.")
        prompt_parts.append(f"\nUser Question: {query}\n\n{lang_name} Answer:")

        
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
        # Language-specific response instructions (REMOVED: French)
        language_instructions = {
            'en': "Please respond in English.",
            'tr': "Lütfen Türkçe olarak yanıt verin.",
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
        # REMOVED: French safety guidelines (language support removed)
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
