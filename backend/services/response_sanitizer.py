"""
Response Sanitizer - Prevent Data Leakage
==========================================

Filters and cleans LLM responses to prevent internal conversation
context, prompts, and metadata from leaking to users.

Created: December 29, 2025
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseSanitizer:
    """Sanitize responses to prevent data leakage"""
    
    # Patterns that indicate internal conversation structure
    LEAKAGE_PATTERNS = [
        r'---\s*User:',  # Conversation history markers
        r'Response:.*?---',  # Response markers
        r'Turn \d+:',  # Turn numbering
        r'Bot:.*?\n',  # Bot response labels
        r'Intent:.*?\n',  # Intent labels
        r'Locations:.*?\n',  # Location lists
        r'Session Context:',  # Session context headers
        r'Last Mentioned',  # Context metadata
        r'User\'s GPS Location',  # GPS metadata
        r'Active Task:',  # Task tracking
        r'User Preferences:',  # Preference data
        r'Conversation Age:',  # Conversation stats
        r'\*\*[A-Z][A-Za-z\s]+\*\*.*?-.*?Resolve',  # Prompt instructions
        r'YOUR TASK:',  # Prompt headers
        r'RETURN FORMAT',  # Format instructions
        r'EXAMPLES:',  # Example sections in prompts
        r'\{\s*"has_references"',  # JSON analysis output
        r'"resolved_references"',  # Reference resolution output
        r'"implicit_context"',  # Context analysis
        r'"needs_clarification"',  # Clarification flags
        r'CONVERSATION HISTORY:',  # History section
        r'CURRENT QUERY:',  # Query markers
        r'---\s*\n.*?User:',  # Separator + User pattern
    ]
    
    @staticmethod
    def sanitize(response: str) -> str:
        """
        Remove internal conversation context and metadata from response
        
        Args:
            response: Raw response text that may contain internal data
            
        Returns:
            Cleaned response safe to return to user
        """
        if not response or not isinstance(response, str):
            return response
        
        original_length = len(response)
        cleaned = response
        
        # Check for conversation history leakage (the most common issue)
        # Pattern: "Hi! Welcome to Istanbul... --- User: question Response: answer --- User:..."
        if '--- User:' in cleaned or 'Response:' in cleaned:
            logger.warning("⚠️ Detected conversation history leakage in response!")
            
            # Strategy 1: If it starts with valid content before history, extract that
            first_history_marker = cleaned.find('--- User:')
            if first_history_marker == -1:
                first_history_marker = cleaned.find('---\nUser:')
            
            if first_history_marker > 50:  # There's substantial content before history
                cleaned = cleaned[:first_history_marker].strip()
                logger.info(f"✅ Extracted content before history marker ({len(cleaned)} chars)")
            else:
                # Strategy 2: Find the LAST "Response:" and get text after it
                # This handles cases where history is at the start
                last_response_idx = cleaned.rfind('Response:')
                if last_response_idx != -1:
                    after_response = cleaned[last_response_idx + len('Response:'):].strip()
                    
                    # Clean up any trailing markers
                    if '---' in after_response:
                        after_response = after_response.split('---')[0].strip()
                    
                    if len(after_response) > 20:
                        cleaned = after_response
                        logger.info(f"✅ Extracted final response after 'Response:' ({len(cleaned)} chars)")
                    else:
                        # Strategy 3: Take everything before the first marker
                        first_marker = min(
                            cleaned.find('--- User:') if '--- User:' in cleaned else len(cleaned),
                            cleaned.find('---\n') if '---\n' in cleaned else len(cleaned),
                            cleaned.find('Turn 1:') if 'Turn 1:' in cleaned else len(cleaned)
                        )
                        if first_marker > 20:
                            cleaned = cleaned[:first_marker].strip()
                            logger.info(f"✅ Extracted content before first marker ({len(cleaned)} chars)")
        
        # Remove any remaining leakage patterns
        for pattern in ResponseSanitizer.LEAKAGE_PATTERNS:
            if re.search(pattern, cleaned, re.IGNORECASE):
                logger.warning(f"⚠️ Found leakage pattern: {pattern[:30]}...")
                # Remove the matched pattern and surrounding context
                cleaned = re.sub(pattern + r'.*?(\n\n|\n|$)', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove JSON artifacts (sometimes LLM returns analysis JSON)
        if cleaned.strip().startswith('{') and '"has_references"' in cleaned:
            logger.warning("⚠️ Response contains JSON analysis, extracting text...")
            try:
                import json
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    # Prefer resolved_query, fallback to reasoning
                    cleaned = data.get('resolved_query', data.get('reasoning', cleaned))
                    logger.info("✅ Extracted text from JSON analysis")
            except:
                pass  # If JSON parsing fails, continue with regex cleanup
        
        # Remove metadata sections
        metadata_markers = [
            'Session Context:',
            'CURRENT QUERY:',
            'CONVERSATION HISTORY:',
            'YOUR TASK:',
            'RETURN FORMAT',
        ]
        for marker in metadata_markers:
            if marker in cleaned:
                # Remove everything from the marker to the next double newline or end
                cleaned = re.sub(
                    re.escape(marker) + r'.*?(?:\n\n|\Z)',
                    '',
                    cleaned,
                    flags=re.DOTALL
                )
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 newlines
        cleaned = cleaned.strip()
        
        # Validate we didn't remove everything
        if not cleaned or len(cleaned) < 10:
            logger.error("❌ Sanitization removed too much content, returning safe default")
            return "I apologize, but I encountered an issue generating a proper response. Could you please rephrase your question?"
        
        if len(cleaned) < original_length:
            logger.info(f"✅ Sanitized response: removed {original_length - len(cleaned)} characters of internal data")
        
        return cleaned
    
    @staticmethod
    def is_safe(response: str) -> bool:
        """
        Check if a response is safe to return (doesn't contain leakage)
        
        Args:
            response: Response text to check
            
        Returns:
            True if safe, False if contains leakage patterns
        """
        if not response:
            return True
        
        # Check for obvious leakage patterns
        dangerous_patterns = [
            '--- User:',
            'Response:',
            'Turn 1:',
            'Session Context:',
            '"has_references"',
            'YOUR TASK:',
            'RETURN FORMAT',
            'CONVERSATION HISTORY:',
            'CURRENT QUERY:',
        ]
        
        for pattern in dangerous_patterns:
            if pattern in response:
                logger.warning(f"⚠️ Response contains unsafe pattern: {pattern}")
                return False
        
        return True


# Convenience function for quick sanitization
def sanitize_response(response: str) -> str:
    """Quick sanitization function"""
    return ResponseSanitizer.sanitize(response)
