"""
LLM Response Parser Utility

Provides common utilities for parsing LLM responses across all LLM modules.
Handles both dict and string responses from different LLM clients.

Author: AI Istanbul Team
Date: December 2024
"""

import json
import logging
import re
from typing import Any, Dict, Union, Optional

logger = logging.getLogger(__name__)


def clean_response_formatting(text: str) -> str:
    """
    Remove formatting artifacts from LLM response.
    
    Cleans:
    - Checkbox patterns: [ ], [X], [x]
    - Duplicate emoji sequences
    - Incomplete meta-questions at the end
    - Intent classification artifacts
    - Extra instruction text
    
    Args:
        text: LLM response text
        
    Returns:
        Cleaned text without artifacts
    """
    if not text or not isinstance(text, str):
        return text
    
    original_length = len(text)
    
    # Remove checkbox patterns with labels (e.g., "[ ] Transportation/Directions")
    text = re.sub(r'\[\s*[Xx]?\s*\]\s*[A-Z][a-zA-Z/\s-]+', '', text)
    
    # Remove standalone checkboxes
    text = re.sub(r'\[\s*[Xx]?\s*\]', '', text)
    
    # Remove duplicate emoji sequences (e.g., "🎉\n\n🎉")
    text = re.sub(r'([\U0001F300-\U0001F9FF])\s*\n+\s*\1', r'\1', text)
    
    # Remove incomplete meta-questions at end (e.g., "👋 How can I help? 🤔")
    text = re.sub(r'\n+👋.*?(\?|🤔).*?$', '', text, flags=re.DOTALL)
    
    # Remove "ANSWER ONLY:" artifacts
    text = re.sub(r'\n*ANSWER ONLY:\s*', '\n', text)
    
    # Remove intent classification sections that might leak through
    text = re.sub(r'\n+\*\*Intent Classification:\*\*.*?(?=\n\n[A-Z]|\n\n\w|$)', '', text, flags=re.DOTALL)
    
    # Remove trailing emojis followed by arrows or similar
    text = re.sub(r'\s+[🚗👉⬇️📍🗺️]+\s*$', '', text)
    
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading emojis at the very start if they're just decorative
    text = re.sub(r'^[\U0001F300-\U0001F9FF\s]+\n+', '', text)
    
    cleaned = text.strip()
    
    if len(cleaned) != original_length:
        logger.info(f"🧹 Formatting cleanup: {original_length} → {len(cleaned)} chars (removed {original_length - len(cleaned)} chars)")
    
    return cleaned


def extract_generated_text(llm_response: Union[Dict, str, None]) -> Optional[str]:
    """
    Extract generated text from LLM response (handles both dict and string formats).
    
    Args:
        llm_response: Response from LLM client (can be dict with 'generated_text' key or string)
        
    Returns:
        Extracted text string or None if extraction fails
    """
    if llm_response is None:
        return None
    
    # If it's already a string, clean and return
    if isinstance(llm_response, str):
        return clean_training_data_leakage(llm_response)
    
    # If it's a dict, extract the generated text
    if isinstance(llm_response, dict):
        # Try different keys that might contain the generated text
        for key in ['generated_text', 'text', 'response', 'output', 'content']:
            if key in llm_response:
                text = llm_response[key]
                if isinstance(text, str):
                    return clean_training_data_leakage(text)
        
        # If no standard key found, try to get the first string value
        for value in llm_response.values():
            if isinstance(value, str) and len(value) > 0:
                return clean_training_data_leakage(value)
        
        # Last resort: convert dict to string
        logger.warning(f"Could not extract text from dict response, keys: {llm_response.keys()}")
        return str(llm_response)
    
    # For any other type, convert to string
    return str(llm_response)


def parse_llm_json_response(
    llm_response: Union[Dict, str, None],
    fallback_value: Optional[Dict] = None
) -> Optional[Dict]:
    """
    Parse LLM response as JSON, handling both dict and string formats.
    
    Args:
        llm_response: Response from LLM client
        fallback_value: Value to return if parsing fails
        
    Returns:
        Parsed JSON dict or fallback_value if parsing fails
    """
    if llm_response is None:
        return fallback_value
    
    # If it's already a dict, check if it's the LLM wrapper or the actual JSON
    if isinstance(llm_response, dict):
        # If it has 'generated_text' key, extract and parse that
        if 'generated_text' in llm_response:
            text = llm_response['generated_text']
            if isinstance(text, str):
                return parse_json_string(text, fallback_value)
            elif isinstance(text, dict):
                return text
        
        # If it has 'text' key, extract and parse that
        if 'text' in llm_response:
            text = llm_response['text']
            if isinstance(text, str):
                return parse_json_string(text, fallback_value)
            elif isinstance(text, dict):
                return text
        
        # Otherwise assume the dict itself is the JSON response
        return llm_response
    
    # If it's a string, parse it as JSON
    if isinstance(llm_response, str):
        return parse_json_string(llm_response, fallback_value)
    
    return fallback_value


def parse_json_string(json_str: str, fallback_value: Optional[Dict] = None) -> Optional[Dict]:
    """
    Parse a JSON string, cleaning it first if needed.
    
    Args:
        json_str: JSON string to parse
        fallback_value: Value to return if parsing fails
        
    Returns:
        Parsed JSON dict or fallback_value if parsing fails
    """
    try:
        # Clean the string
        cleaned = clean_json_string(json_str)
        
        if not cleaned:
            logger.warning("Empty JSON string after cleaning")
            return fallback_value
        
        # Try to parse
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Try to extract JSON object if there's extra text
        try:
            # Find first { and last }
            start = cleaned.find('{')
            end = cleaned.rfind('}')
            
            if start >= 0 and end > start:
                json_part = cleaned[start:end+1]
                result = json.loads(json_part)
                logger.info("Successfully extracted JSON from text with extra content")
                return result
        except:
            pass
        
        logger.warning(f"Failed to parse JSON: {e}")
        logger.debug(f"JSON string (first 200 chars): {json_str[:200]}")
        return fallback_value
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return fallback_value


def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string by removing markdown code blocks and extra whitespace.
    
    Args:
        json_str: JSON string to clean
        
    Returns:
        Cleaned JSON string
    """
    if not json_str:
        return json_str
    
    # Remove leading/trailing whitespace
    cleaned = json_str.strip()
    
    # Remove markdown code blocks
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    # Remove any remaining whitespace
    cleaned = cleaned.strip()
    
    # If there's text after the JSON, try to extract just the JSON part
    # Look for the first { and the matching closing }
    if '{' in cleaned and '}' in cleaned:
        start_idx = cleaned.find('{')
        # Count braces to find the matching closing brace
        brace_count = 0
        end_idx = -1
        
        for i in range(start_idx, len(cleaned)):
            if cleaned[i] == '{':
                brace_count += 1
            elif cleaned[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx+1]
    
    return cleaned


def _detect_response_language(text: str) -> str:
    """
    Detect the language of the response text.
    
    Args:
        text: Response text
        
    Returns:
        Language name: 'Russian', 'Arabic', 'Turkish', 'German', 'French', or 'English'
    """
    if not text:
        return 'English'
    
    # Check for Cyrillic (Russian)
    cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
    if cyrillic_count > 5:
        return 'Russian'
    
    # Check for Arabic
    arabic_count = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
    if arabic_count > 5:
        return 'Arabic'
    
    # Check for Turkish special chars
    if any(char in text for char in ['ı', 'ş', 'ğ', 'ü', 'ö', 'ç', 'İ', 'Ş', 'Ğ']):
        return 'Turkish'
    
    # Check for German umlauts
    if any(char in text for char in ['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü']):
        return 'German'
    
    # Check for French accents
    if any(char in text for char in ['é', 'è', 'ê', 'à', 'ù', 'û', 'ô', 'î', 'ç']):
        return 'French'
    
    return 'English'


def clean_training_data_leakage(text: str, prompt: Optional[str] = None) -> str:
    """
    Remove training data/example conversations that the LLM might have leaked.
    
    LANGUAGE-AWARE: Applies different cleaning strategies based on detected language.
    Non-Latin scripts (Russian, Arabic) get gentler cleaning to avoid over-removal.
    
    Strategy:
    1. Detect response language
    2. Check if LLM is echoing the prompt (common issue)
    3. Look for training format markers (EMBER:, Assistant:, etc.)
    4. Apply language-appropriate cleaning patterns
    5. If found at the start, try to extract the actual response after the marker
    6. If no valid content after marker, return empty string
    7. Otherwise, truncate at the first leak pattern
    
    Args:
        text: Generated text from LLM
        prompt: Optional original prompt (to detect echo)
        
    Returns:
        Cleaned text with training examples removed
    """
    if not text or not isinstance(text, str):
        return text
    
    original_text = text
    original_length = len(text)
    
    # Detect language for appropriate cleaning
    detected_lang = _detect_response_language(text)
    logger.debug(f"Detected response language: {detected_lang}")
    
    # NEW: Check if LLM is echoing the prompt (common failure mode)
    if prompt and len(prompt) > 100:
        # Check if response contains large chunks of the prompt
        prompt_fragments = [
            "🚨 CRITICAL INSTRUCTION",
            "⚠️ CRITICAL:",
            "CRITICAL LANGUAGE RULE",
            "DO NOT write \"EMBER",
            "Your response MUST start",
            "Follow the response guidelines",
            "Multi-Intent Query Handling",
            "Intent Classification:",
            "User's Question:",
            "Current User Question:",
            "The user writes in",
            "unnecessary information",
            "Provide a clear, concise response",
            "🌍 REMEMBER: Answer in",
            "REMEMBER: Answer in",
            "⚠️ CRITICAL: Your response MUST",
            "❌ DO NOT use any other language",
        ]
        
        # If response contains multiple prompt fragments, it's likely an echo
        fragment_count = sum(1 for frag in prompt_fragments if frag.lower() in text.lower())
        if fragment_count >= 2:
            logger.error(f"🚨 LLM is echoing the prompt! Found {fragment_count} prompt fragments in response")
            return ""  # Trigger fallback
        
        # Also check if response starts with a fragment from middle of prompt
        # These indicate the LLM started generating from partway through the prompt
        mid_prompt_fragments = [
            "ues or unnecessary",
            "or unnecessary information",
            "Provide a clear, concise",
            "The user writes in",
            "Follow the response",
            "unnecessary information. Provide",
        ]
        
        for fragment in mid_prompt_fragments:
            if text.strip().startswith(fragment):
                logger.error(f"🚨 Response starts with mid-prompt fragment: '{fragment[:30]}...'")
                return ""  # Trigger fallback
    
    # First, check if the response starts with a training format marker
    start_markers = [
        "EMBER:",
        "Assistant:",
        "KAM:",
        "KAM here",
        "User:",
        "A:",
        "Q:",
        "Hi, I'm",
    ]
    
    for marker in start_markers:
        if text.strip().startswith(marker):
            # Found a marker at the start - try to extract content after it
            logger.warning(f"🧹 Response starts with training marker '{marker}' - attempting to extract content")
            
            # Remove the marker and get what's after
            after_marker = text[len(marker):].strip()
            
            # Check if there's substantial content after the marker (at least 20 chars)
            if len(after_marker) >= 20:
                # Use the content after the marker
                text = after_marker
                logger.info(f"✅ Extracted {len(text)} chars of content after '{marker}'")
                break
            else:
                # Not enough content - this is likely just a training example
                logger.warning(f"⚠️  Insufficient content after '{marker}' - treating as training data")
                # Continue to check for other patterns
    
    # Patterns that indicate training data leakage or debug output
    # For non-Latin scripts (Russian, Arabic), use ONLY critical patterns
    # For Latin scripts, use full pattern list
    
    if detected_lang in ['Russian', 'Arabic']:
        # Minimal patterns for non-Latin scripts to avoid over-cleaning
        leak_patterns = [
            "\n🎯 INTENT CLASSIFICATION",
            "\n🚨 UNCERTAIN INTENT",
            "\n---\n\n🎯",
            "\n---\n\n🚨",
            "\nEMBER:",
            "\nAssistant:",
            "\nKAM here",
            "\nHi, I'm",
            "\n\n**Context information available:**",
            "\n**Identify Primary Intent",
            "🚨 CRITICAL:",
            "⚠️ IMPORTANT:",
        ]
        logger.debug(f"Using minimal leak patterns for {detected_lang}")
    else:
        # Full leak pattern list for Latin scripts
        leak_patterns = [
        "\nKAM here",
        "\nHi, I'm excited",
        "\nOf course, my friend",
        "\n🎯 INTENT CLASSIFICATION",
        "\n🚨 UNCERTAIN INTENT",
        "\n🎯 MULTI-INTENT",
        "\n---\n\n🎯",
        "\n---\n\n🚨",
        "\nIntents:",
        "\nTransportation/Directions:",
        "\nRestaurant Recommendation:",
        "\nAttraction Information:",
        "\n### USER QUESTION",
        "\n### User's question",
        "\n### User Question",
        "\n## User's Question",
        "\n## User Question",
        "\n## Context:",
        "\n### Context:",
        "\nUSER QUESTION:",
        "\nUser's question:",
        "\nUser Question:",
        "\nUser:",
        "\nAssistant:",
        "\nA:",
        "\nQ:",
        "\nExample:",
        "\nFor instance,",
        "\nHere's an example",
        "\n### EXAMPLE",
        "\nEXAMPLE:",
        # REFINED: Only catch step patterns if they look like meta-instructions
        # Valid directions like "**Step 1: Take metro M1..." are allowed
        # Training data like "**Step 1: Analyze the query" is caught
        "\n**Step 1: Analyze",
        "\n**Step 1: Identify",
        "\n**Step 1: Consider",
        "\n**Step 1: Determine",
        "\n**Step 1: Think",
        "\n**Step 1: Check",
        "\n**Step 2: Analyze",
        "\n**Step 2: Identify",
        "\n**Step 2: Consider",
        "\n**Step 3: Analyze",
        "\n**Step 3: Identify",
        "\n## Step 1: Analyze",
        "\n## Step 1: Identify",
        "\n## Step 2: Analyze",
        "\n### Example",
        "\nNow, let's get started!",
        "\nThe user has asked:",
        "\n**Your response should",
        "\nYour response should",
        "\nPlease respond with",
        "\n(Hello KAM!",
        "\nMerhaba KAM!",
        "\n### What's your answer",
        "\nWhat's your answer",
        # Meta-instructions that LLM echoes back
        "\n---\n\nWhat's your response",
        "\nWhat's your response",
        "\nType your answer directly",
        "\nType your response now",
        "\nPlease use the correct format",
        "\nDO NOT write \"EMBER",
        "\nDO NOT write \"Assistant",
        "\nYour response should start with",
        "\nLet's chat about",
        "\nHow can I help",
        "\n📝 Type your",
        "\n👋 Let's",
        "\n💬 How can",
        "\n🌍 REMEMBER:",
        "\nREMEMBER: Answer in",
        "\n⚠️ CRITICAL:",
        "\n❌ DO NOT use any other",
        "\nENGLISH Answer:",
        "\nTURKISH Answer:",
        "\nFRENCH Answer:",
        "\n\n Wait... How did I do",
        "\nWait... How did I do",
        "\n**Identify Primary Intent",
        "\n**Address Secondary Intents",
        "\n- User's MAIN need",
        "\n- Since the user",
        # EMBER system instruction patterns (in the middle of response)
        "\nEMBER:",
        "\n- Your response should",
        "\n- Only respond with",
        "\n- Start with a direct",
        "\n- NOT include example",
        # Catch critical markers anywhere in text
        "🚨 CRITICAL:",
        "CRITICAL:",
        "⚠️ IMPORTANT:",
        "IMPORTANT:",
        "\n**Context information available:**",
        ]
        logger.debug(f"Using full leak patterns for {detected_lang}")
    
    # Find the first occurrence of any leak pattern
    first_leak_pos = len(text)
    found_pattern = None
    
    for pattern in leak_patterns:
        pos = text.find(pattern)
        if pos != -1 and pos < first_leak_pos:
            first_leak_pos = pos
            found_pattern = pattern
    
    # If we found a leak pattern, truncate the text before it
    if found_pattern:
        cleaned = text[:first_leak_pos].strip()
        
        # Final validation: make sure we have substantial content
        if len(cleaned) < 10:
            logger.error(f"❌ After removing training data, only {len(cleaned)} chars remain - likely all training data")
            logger.debug(f"Original text (first 200 chars): {original_text[:200]}")
            # Return empty to trigger fallback
            return ""
        
        logger.warning(f"🧹 Removed training data leakage starting with '{found_pattern.strip()}' at position {first_leak_pos}")
        logger.info(f"📏 Original: {original_length} chars → Cleaned: {len(cleaned)} chars (removed {original_length - len(cleaned)} chars)")
        return cleaned
    
    # No leak patterns found - return the text as is (possibly already cleaned from start marker)
    if len(text) != original_length:
        logger.info(f"📏 Cleaned start marker: {original_length} chars → {len(text)} chars")
    
    return text
