"""
LLM Response Sanitizer

Removes system prompt artifacts and cleans up LLM outputs for production use.
"""

import re
from typing import Optional


class ResponseSanitizer:
    """
    Production-grade response sanitizer for LLM outputs.
    
    Handles:
    - System prompt leakage removal
    - Language consistency enforcement
    - Artifact cleanup
    - Output validation
    """
    
    def __init__(self):
        """Initialize sanitizer with cleanup patterns"""
        self.system_prompt_patterns = [
            # System instructions - catch all variations
            r"Never use Turkish[,\s]*(?:French[,\s]*)?(?:or any other language)?\.?\s*\|?\s*",
            r"Never use (?:Turkish|French|any other language).*?\.?\s*",
            r"Do: Not: use other languages\.?\s*",
            r"NO EXCEPTIONS!.*?Please respond.*?\n",
            r"No English\. No French\. No other language\..*?\n",
            r"NO other languages!.*?\n",
            r"Translate any context or question to English if necessary\.?\s*",
            r"No Turkish or French!.*?Answer in the context.*?\n",
            r"0_0\s*\|?\s*",
            
            # Meta-commands
            r"^ANSWER:\s*",
            r"^Please respond!?\s*",
            r"^You are KAM,.*?\n\n",
            
            # Repeated artifacts at end
            r"\s*0_0\)+\s*$",
            r"\s*\|\s*$",
            
            # Turkish language slip warnings
            r"Bütün yanıtlarını Türkçe olarak verin\..*?\n",
            r"İçinde bulunduğunuz mahalleyi.*?\n",
            
            # NEW: Conversation history leakage patterns
            r"---\s*User:.*?(?=---|$)",  # --- User: ... pattern
            r"Response:.*?(?=---|$)",  # Response: ... pattern  
            r"Turn \d+:.*?(?=Turn \d+:|$)",  # Turn numbering
            r"\n  User:.*?(?=\n|$)",  # Indented User: labels
            r"\n  Bot:.*?(?=\n|$)",  # Indented Bot: labels
            r"Intent:.*?(?=\n|$)",  # Intent: labels
            r"Locations:.*?(?=\n|$)",  # Locations: labels
            r"Session Context:.*?(?=\n\n|\Z)",  # Session context
            r"Last Mentioned.*?(?=\n|$)",  # Last Mentioned metadata
            r"User's GPS Location.*?(?=\n|$)",  # GPS metadata
            r"Active Task:.*?(?=\n|$)",  # Task tracking
            r"User Preferences:.*?(?=\n|$)",  # Preference data
            r"Conversation Age:.*?(?=\n|$)",  # Conversation stats
            r"CONVERSATION HISTORY:.*?(?=\n\n|\Z)",  # History section
            r"CURRENT QUERY:.*?(?=\n\n|\Z)",  # Query markers
            r"YOUR TASK:.*?(?=\n\n|\Z)",  # Task instructions
            r"RETURN FORMAT.*?(?=\n\n|\Z)",  # Format instructions
            r'"has_references".*?(?=\n|$)',  # JSON analysis
            r'"resolved_references".*?(?=\n|$)',  # Reference resolution
            r'"implicit_context".*?(?=\n|$)',  # Context analysis
            r'"needs_clarification".*?(?=\n|$)',  # Clarification flags
        ]
    
    def sanitize(
        self,
        response: str,
        expected_language: str = "en",
        strict_language_check: bool = False
    ) -> str:
        """
        Clean and sanitize LLM response.
        
        Args:
            response: Raw LLM output
            expected_language: Expected language code (en/tr)
            strict_language_check: Whether to enforce language strictly
            
        Returns:
            Cleaned response text
        """
        if not response:
            return response
        
        # Apply all cleanup patterns
        cleaned = response
        for pattern in self.system_prompt_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove markdown bold formatting (**text**)
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        
        # Remove markdown italic formatting (*text* or _text_)
        cleaned = re.sub(r'(?<!\*)\*(?!\*)([^*]+)\*(?!\*)', r'\1', cleaned)
        cleaned = re.sub(r'_([^_]+)_', r'\1', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Single spaces only
        
        # Trim
        cleaned = cleaned.strip()
        
        # Language consistency check (fix mixed language issues)
        if strict_language_check:
            original_length = len(cleaned)
            cleaned = self._enforce_language_consistency(cleaned, expected_language)
            if len(cleaned) != original_length:
                # Log language fixes applied
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"🌍 Language consistency enforced: {expected_language} ({original_length} → {len(cleaned)} chars)")
        
        return cleaned
    
    def _validate_language(self, text: str, expected_lang: str) -> tuple[bool, Optional[str]]:
        """Validate response language matches expected"""
        turkish_chars = set('çğıİöşüÇĞÖŞÜ')
        has_turkish = any(c in turkish_chars for c in text[:200])
        has_english = any(c.isalpha() and c.lower() in 'abcdefghijklmnopqrstuvwxyz' for c in text[:200])
        
        if expected_lang == "en":
            if has_turkish and not has_english:
                return False, "Response in Turkish when English expected"
        elif expected_lang == "tr":
            if has_english and not has_turkish:
                return False, "Response in English when Turkish expected"
        
        return True, None
    
    def validate(self, response: str, min_length: int = 20) -> tuple[bool, Optional[str]]:
        """
        Validate that response is usable and not corrupted.
        
        Args:
            response: Cleaned response text
            min_length: Minimum acceptable length
            
        Returns:
            (is_valid, error_reason)
        """
        if not response or len(response.strip()) < min_length:
            return False, "Response too short or empty"
        
        # Check for remaining artifacts
        artifact_patterns = [
            r"^(Do:|Never|ANSWER:|Please respond)",
            r"0_0{5,}",  # Too many artifacts
            r"^\s*\|\s*$",  # Empty pipe
        ]
        
        for pattern in artifact_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False, f"Artifact detected: {pattern}"
        
        return True, None
    
    def _enforce_language_consistency(self, text: str, expected_lang: str) -> str:
        """
        Enforce language consistency by translating common mixed-language phrases.
        Supports 5 languages: English (en), Turkish (tr), Russian (ru), German (de), Arabic (ar)
        
        Args:
            text: Text to check
            expected_lang: Expected language code (en/tr/ru/de/ar)
            
        Returns:
            Text with consistent language
        """
        # Base translations (English as pivot language)
        translations = {
            # Time units
            'min': {
                'en': 'min', 'tr': 'dk', 'ru': 'мин', 'de': 'Min', 'ar': 'د'
            },
            'minutes': {
                'en': 'minutes', 'tr': 'dakika', 'ru': 'минут', 'de': 'Minuten', 'ar': 'دقائق'
            },
            'hours': {
                'en': 'hours', 'tr': 'saat', 'ru': 'часов', 'de': 'Stunden', 'ar': 'ساعات'
            },
            # Labels
            'Duration:': {
                'en': 'Duration:', 'tr': 'Süre:', 'ru': 'Время:', 'de': 'Dauer:', 'ar': 'المدة:'
            },
            'Distance:': {
                'en': 'Distance:', 'tr': 'Mesafe:', 'ru': 'Расстояние:', 'de': 'Entfernung:', 'ar': 'المسافة:'
            },
            'Transfers:': {
                'en': 'Transfers:', 'tr': 'Aktarma:', 'ru': 'Пересадки:', 'de': 'Umstiege:', 'ar': 'التحويلات:'
            },
            'Lines:': {
                'en': 'Lines:', 'tr': 'Hatlar:', 'ru': 'Линии:', 'de': 'Linien:', 'ar': 'الخطوط:'
            },
            'Step by Step:': {
                'en': 'Step by Step:', 'tr': 'Adım Adım:', 'ru': 'Пошагово:', 'de': 'Schritt für Schritt:', 'ar': 'خطوة بخطوة:'
            },
            'Route:': {
                'en': 'Route:', 'tr': 'Güzergah:', 'ru': 'Маршрут:', 'de': 'Route:', 'ar': 'المسار:'
            },
            'Route': {
                'en': 'Route', 'tr': 'Güzergah', 'ru': 'Маршрут', 'de': 'Route', 'ar': 'المسار'
            },
            # Common phrases
            'transfer': {
                'en': 'transfer', 'tr': 'aktarma', 'ru': 'пересадка', 'de': 'Umstieg', 'ar': 'تحويل'
            },
            'transfers': {
                'en': 'transfers', 'tr': 'aktarma', 'ru': 'пересадок', 'de': 'Umstiege', 'ar': 'تحويلات'
            },
            'This route is verified from Istanbul transportation database': {
                'en': 'This route is verified from Istanbul transportation database',
                'tr': 'Bu güzergah İstanbul ulaşım veritabanından doğrulanmıştır',
                'ru': 'Этот маршрут проверен по базе данных транспорта Стамбула',
                'de': 'Diese Route wurde aus der Istanbuler Verkehrsdatenbank verifiziert',
                'ar': 'تم التحقق من هذا المسار من قاعدة بيانات النقل في إسطنبول'
            },
            # Action words
            'Take': {
                'en': 'Take', 'tr': 'Binin', 'ru': 'Сядьте на', 'de': 'Nehmen Sie', 'ar': 'خذ'
            },
            'Walk to': {
                'en': 'Walk to', 'tr': 'Yürüyün', 'ru': 'Идите к', 'de': 'Gehen Sie zu', 'ar': 'امشِ إلى'
            },
            'Transfer to': {
                'en': 'Transfer to', 'tr': 'Aktarma yapın', 'ru': 'Пересядьте на', 'de': 'Umsteigen auf', 'ar': 'انتقل إلى'
            },
            'from': {
                'en': 'from', 'tr': "'dan", 'ru': 'от', 'de': 'von', 'ar': 'من'
            },
            'to': {
                'en': 'to', 'tr': "'a", 'ru': 'до', 'de': 'nach', 'ar': 'إلى'
            },
            'at': {
                'en': 'at', 'tr': "'da", 'ru': 'на', 'de': 'bei', 'ar': 'عند'
            },
        }
        
        # Language-specific patterns to detect
        lang_patterns = {
            'tr': [' dk', 'dakika', 'Süre:', 'Mesafe:', 'Aktarma:', 'Hatlar:', 'Adım Adım:', 'Güzergah'],
            'ru': [' мин', 'минут', 'Время:', 'Расстояние:', 'Пересадки:', 'Линии:', 'Пошагово:', 'Маршрут'],
            'de': [' Min', 'Minuten', 'Dauer:', 'Entfernung:', 'Umstiege:', 'Linien:', 'Schritt für Schritt:'],
            'ar': [' د', 'دقائق', 'المدة:', 'المسافة:', 'التحويلات:', 'الخطوط:', 'خطوة بخطوة:', 'المسار'],
            'en': [' min', 'minutes', 'Duration:', 'Distance:', 'Transfers:', 'Lines:', 'Step by Step:', 'Route:'],
        }
        
        # Build reverse lookup: for each phrase in any language, map to the expected language version
        # IMPORTANT: Use word-boundary matching to avoid corrupting Turkish/other words
        # e.g., don't replace "at" inside "Hayatın" or "katmak"
        for key, lang_map in translations.items():
            for source_lang, source_phrase in lang_map.items():
                if source_lang != expected_lang and source_phrase in text:
                    target_phrase = lang_map.get(expected_lang, lang_map['en'])
                    # For short words (2-4 chars), use word boundary matching to avoid
                    # corrupting words that contain these substrings
                    if len(source_phrase) <= 4 and source_phrase.isalpha():
                        # Use word boundaries (\b) to only match standalone words
                        pattern = r'\b' + re.escape(source_phrase) + r'\b'
                        text = re.sub(pattern, target_phrase, text, flags=re.IGNORECASE)
                    else:
                        # For longer phrases (like "Duration:"), simple replace is safe
                        text = text.replace(source_phrase, target_phrase)
        
        # Regex-based time unit conversions
        if expected_lang == "en":
            # Convert Turkish dk to min
            text = re.sub(r'\((\d+)\s*dk\)', r'(\1 min)', text)
            text = re.sub(r'(\d+)\s*dk([,.\s\n])', r'\1 min\2', text)
            text = re.sub(r'(\d+)\s*dk$', r'\1 min', text)
            # Convert Russian мин to min
            text = re.sub(r'\((\d+)\s*мин\)', r'(\1 min)', text)
            text = re.sub(r'(\d+)\s*мин([,.\s\n])', r'\1 min\2', text)
            # Convert German Min to min (case sensitive)
            text = re.sub(r'\((\d+)\s*Min\)', r'(\1 min)', text)
            text = re.sub(r'(\d+)\s*Min([,.\s\n])', r'\1 min\2', text)
            # Convert Arabic د to min
            text = re.sub(r'\((\d+)\s*د\)', r'(\1 min)', text)
            text = re.sub(r'(\d+)\s*د([,.\s\n])', r'\1 min\2', text)
            
        elif expected_lang == "tr":
            # Convert English min to dk
            text = re.sub(r'\((\d+)\s*min\)', r'(\1 dk)', text)
            text = re.sub(r'(\d+)\s*min([,.\s\n])', r'\1 dk\2', text)
            text = re.sub(r'(\d+)\s*min$', r'\1 dk', text)
            # Convert Russian мин to dk
            text = re.sub(r'\((\d+)\s*мин\)', r'(\1 dk)', text)
            text = re.sub(r'(\d+)\s*мин([,.\s\n])', r'\1 dk\2', text)
            
        elif expected_lang == "ru":
            # Convert English min to мин
            text = re.sub(r'\((\d+)\s*min\)', r'(\1 мин)', text)
            text = re.sub(r'(\d+)\s*min([,.\s\n])', r'\1 мин\2', text)
            text = re.sub(r'(\d+)\s*min$', r'\1 мин', text)
            # Convert Turkish dk to мин
            text = re.sub(r'\((\d+)\s*dk\)', r'(\1 мин)', text)
            text = re.sub(r'(\d+)\s*dk([,.\s\n])', r'\1 мин\2', text)
            
        elif expected_lang == "de":
            # Convert English min to Min
            text = re.sub(r'\((\d+)\s*min\)', r'(\1 Min)', text)
            text = re.sub(r'(\d+)\s*min([,.\s\n])', r'\1 Min\2', text)
            text = re.sub(r'(\d+)\s*min$', r'\1 Min', text)
            # Convert Turkish dk to Min
            text = re.sub(r'\((\d+)\s*dk\)', r'(\1 Min)', text)
            text = re.sub(r'(\d+)\s*dk([,.\s\n])', r'\1 Min\2', text)
            
        elif expected_lang == "ar":
            # Convert English min to د
            text = re.sub(r'\((\d+)\s*min\)', r'(\1 د)', text)
            text = re.sub(r'(\d+)\s*min([,.\s\n])', r'\1 د\2', text)
            text = re.sub(r'(\d+)\s*min$', r'\1 د', text)
            # Convert Turkish dk to د
            text = re.sub(r'\((\d+)\s*dk\)', r'(\1 د)', text)
            text = re.sub(r'(\d+)\s*dk([,.\s\n])', r'\1 د\2', text)
        
        return text
    
# Legacy function interface for backward compatibility
def sanitize_llm_response(response: str) -> str:
    """
    Remove system prompt leakage and artifacts from LLM responses.
    
    Common issues:
    - System instructions appearing in output ("Never use Turkish...")
    - Meta-commands ("Do: Not: use other languages")
    - Repeated artifacts (0_0, symbols)
    - Unnecessary prefixes ("ANSWER:", "Please respond")
    
    Args:
        response: Raw LLM output
        
    Returns:
        Cleaned response text
    """
    if not response:
        return response
    
    # Patterns to remove (order matters - more specific first)
    patterns_to_remove = [
        # System instructions
        r"Never use Turkish,?\s*French,?\s*or any other language\.?\s*\|?\s*",
        r"Do: Not: use other languages\.?\s*",
        r"NO EXCEPTIONS!.*?Please respond.*?\n",
        r"No English\. No French\. No other language\..*?\n",
        r"NO other languages!.*?\n",
        r"Translate any context or question to English if necessary\.?\s*",
        r"No Turkish or French!.*?Answer in the context.*?\n",
        r"0_0\s*\|?\s*",
        
        # Meta-commands
        r"^ANSWER:\s*",
        r"^Please respond!?\s*",
        r"^You are KAM,.*?\n\n",
        
        # Repeated artifacts at end
        r"\s*0_0\)+\s*$",
        r"\s*\|\s*$",
        
        # Turkish language slip warnings
        r"Bütün yanıtlarını Türkçe olarak verin\..*?\n",
        r"İçinde bulunduğunuz mahalleyi.*?\n",
    ]
    
    # Apply all cleanup patterns
    cleaned = response
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove excessive whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Single spaces only
    
    # Trim
    cleaned = cleaned.strip()
    
    return cleaned


def validate_response_quality(response: str, min_length: int = 20) -> tuple[bool, Optional[str]]:
    """
    Validate that response is usable and not corrupted.
    
    Args:
        response: Cleaned response text
        min_length: Minimum acceptable length
        
    Returns:
        (is_valid, error_reason)
    """
    if not response or len(response.strip()) < min_length:
        return False, "Response too short or empty"
    
    # Check for remaining artifacts
    artifact_patterns = [
        r"^(Do:|Never|ANSWER:|Please respond)",
        r"0_0{5,}",  # Too many artifacts
        r"^\s*\|\s*$",  # Empty pipe
    ]
    
    for pattern in artifact_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return False, f"Artifact detected: {pattern}"
    
    # Check language consistency (basic heuristic)
    # If response starts with Turkish characters but should be English
    turkish_chars = set('çğıİöşüÇĞÖŞÜ')
    has_turkish = any(c in turkish_chars for c in response[:100])
    has_english = any(c.isalpha() and c.lower() in 'abcdefghijklmnopqrstuvwxyz' for c in response[:100])
    
    if has_turkish and not has_english:
        return False, "Response appears to be in wrong language (Turkish when English expected)"
    
    return True, None


def sanitize_and_validate(response: str, min_length: int = 20) -> tuple[str, bool, Optional[str]]:
    """
    Clean and validate LLM response in one call.
    
    Args:
        response: Raw LLM output
        min_length: Minimum acceptable length
        
    Returns:
        (cleaned_response, is_valid, error_reason)
    """
    cleaned = sanitize_llm_response(response)
    is_valid, error = validate_response_quality(cleaned, min_length)
    return cleaned, is_valid, error


# Example usage
if __name__ == "__main__":
    # Test cases
    test_responses = [
        "Never use Turkish or any other language. |\n\nANSWER: \nTo take the metro from Kadıköy to Taksim...",
        "0_0\n\nTo get from Kadıköy to Sultanahmet, take the ferry... 0_0) 0_0) 0_0)",
        "Do: Not: use other languages.\n\nThe best way is to take the ferry...",
    ]
    
    for i, test in enumerate(test_responses, 1):
        print(f"\n=== Test {i} ===")
        print("BEFORE:", test[:100])
        cleaned, valid, error = sanitize_and_validate(test)
        print("AFTER:", cleaned[:100])
        print("VALID:", valid, f"({error})" if error else "")
