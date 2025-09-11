import os
import json
import logging

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable name, not the actual key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.warning("OPENAI_API_KEY not found in environment variables")
    client = None
else:
    client = OpenAI(api_key=api_key)

def parse_user_input(user_input: str) -> dict:
    """Parse user input and extract intent and entities"""
    if not client:
        # Fallback when OpenAI is not available
        return {
            "intent": "fallback",
            "entities": {
                "location": "Istanbul",
                "query": user_input
            }
        }
    
    prompt = f"""
    You are a travel assistant for Istanbul only. All queries are about Istanbul, unless the user specifies a district or place within Istanbul.
    Extract the intent and entities from this user input: \"{user_input}\"
    If the user does not specify a location, set the location entity to \"Istanbul\" by default.
    Return ONLY a valid JSON object like: {{
      \"intent\": \"\",
      \"entities\": {{}}
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from OpenAI")
        content = content.strip()
        
        # Try to parse as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If JSON parsing fails, extract JSON from text
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
                
    except Exception as e:
        logging.error(f"Error parsing user input: {e}")
        # Return fallback response
        return {
            "intent": "fallback",
            "entities": {
                "location": "Istanbul",
                "query": user_input,
                "error": str(e)
            }
        }
