import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("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"))

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
