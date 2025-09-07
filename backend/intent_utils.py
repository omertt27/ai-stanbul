import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment variable - NEVER hardcode API keys!
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)

def parse_user_input(user_input: str) -> dict:
    prompt = f"""
    You are a travel assistant for Istanbul only. All queries are about Istanbul, unless the user specifies a district or place within Istanbul.
    Extract the intent and entities from this user input: \"{user_input}\"
    If the user does not specify a location, set the location entity to \"Istanbul\" by default.
    Return a JSON like: {{
      \"intent\": \"\",
      \"entities\": {{}}
    }}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
