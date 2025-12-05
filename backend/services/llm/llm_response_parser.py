"""
LLM Response Parser Utility

Provides common utilities for parsing LLM responses across all LLM modules.
Handles both dict and string responses from different LLM clients.

Author: AI Istanbul Team
Date: December 2024
"""

import json
import logging
from typing import Any, Dict, Union, Optional

logger = logging.getLogger(__name__)


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
    
    # If it's already a string, return as-is
    if isinstance(llm_response, str):
        return llm_response
    
    # If it's a dict, extract the generated text
    if isinstance(llm_response, dict):
        # Try different keys that might contain the generated text
        for key in ['generated_text', 'text', 'response', 'output', 'content']:
            if key in llm_response:
                text = llm_response[key]
                if isinstance(text, str):
                    return text
        
        # If no standard key found, try to get the first string value
        for value in llm_response.values():
            if isinstance(value, str) and len(value) > 0:
                return value
        
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
