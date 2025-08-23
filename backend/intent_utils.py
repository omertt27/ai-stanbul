import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def parse_user_input(user_input: str) -> dict:
    prompt = f"""
    You are a travel assistant. 
    Extract the intent and entities from this user input: \"{user_input}\"
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
