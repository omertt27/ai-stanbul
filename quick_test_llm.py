#!/usr/bin/env python3
"""
Quick test of LLM intent classifier with few-shot prompt
"""

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize LLM
from ml_systems.llm_service_wrapper import LLMServiceWrapper
llm = LLMServiceWrapper()

test_messages = [
    "What's the weather like?",
    "Where can I find good restaurants?",
    "How do I get to Taksim?",
    "Tell me about Hagia Sophia",
]

print("\n" + "="*80)
print("Testing Few-Shot Prompt with TinyLlama")
print("="*80 + "\n")

for msg in test_messages:
    prompt = f"""Classify user intent. Examples:

Q: "What's the weather today?"
A: {{"primary_intent": "weather", "confidence": 0.95, "all_intents": ["weather"]}}

Q: "Where can I eat kebab?"
A: {{"primary_intent": "restaurant", "confidence": 0.95, "all_intents": ["restaurant"]}}

Q: "How do I get to Taksim?"
A: {{"primary_intent": "transportation", "confidence": 0.95, "all_intents": ["transportation"]}}

Q: "Show me Hagia Sophia"
A: {{"primary_intent": "attraction", "confidence": 0.95, "all_intents": ["attraction"]}}

Now classify this:
Q: "{msg}"
A:"""
    
    response = llm.generate(prompt=prompt, max_tokens=100, temperature=0.3)
    
    print(f"Message: {msg}")
    print(f"Response: {response}")
    
    # Try to extract JSON
    if '{' in response:
        start = response.find('{')
        end = response.rfind('}') + 1
        json_part = response[start:end]
        print(f"JSON part: {json_part}")
        
        # Try to parse
        import json
        try:
            parsed = json.loads(json_part)
            intent = parsed.get('primary_intent') or parsed.get('primaryIntent')
            print(f"✅ Detected intent: {intent}")
        except Exception as e:
            print(f"❌ Parse error: {e}")
    print("-" * 80 + "\n")
