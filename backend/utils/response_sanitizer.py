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
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Single spaces only
        
        # Trim
        cleaned = cleaned.strip()
        
        # Language validation (if strict)
        if strict_language_check:
            is_valid, _ = self._validate_language(cleaned, expected_language)
            if not is_valid and expected_language == "en":
                # If response is in wrong language, prepend a note
                cleaned = f"[Note: Response provided in English]\n\n{cleaned}"
        
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
