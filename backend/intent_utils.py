import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_user_input(user_input: str) -> dict:
    prompt = f"""
    You are a travel assistant. 
    Extract the intent and entities from this user input: \"{user_input}\"
    Return a JSON like: {{
      \"intent\": \"\",
      \"entities\": {{}}
    }}
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message["content"]
