#!/usr/bin/env python3
"""
Mock LLM Client for Testing
Simulates RunPod LLM responses for integration testing
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM client that returns realistic responses for testing"""
    
    def __init__(self):
        self.enabled = True
        self.api_url = "mock://localhost"
        self.model_name = "mock-llama-3.1-8b"
        self.api_type = "mock"
    
    async def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, **kwargs):
        """Generate mock response based on prompt"""
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Detect intent classification requests
        if "intent classification" in prompt.lower() or '"primary_intent"' in prompt:
            # Extract the user message from the prompt
            if 'Classify: "' in prompt:
                message = prompt.split('Classify: "')[1].split('"')[0].lower()
            else:
                message = ""
            
            # Simple intent detection
            intent_data = self._mock_classify_intent(message)
            generated_text = json.dumps(intent_data)
        else:
            # Regular response generation
            generated_text = self._mock_generate_response(prompt)
        
        return {
            "generated_text": generated_text,
            "raw": {
                "text": generated_text,
                "usage": {
                    "prompt_tokens": len(prompt) // 4,
                    "completion_tokens": len(generated_text) // 4,
                    "total_tokens": (len(prompt) + len(generated_text)) // 4
                }
            },
            "request_id": "mock-123"
        }
    
    def _mock_classify_intent(self, message: str) -> dict:
        """Mock intent classification"""
        message = message.lower()
        
        # Simple keyword matching
        if any(word in message for word in ['hello', 'hi', 'hey', 'merhaba', 'selam']):
            return {
                "primary_intent": "greeting",
                "confidence": 0.95,
                "all_intents": ["greeting"]
            }
        elif any(word in message for word in ['restaurant', 'food', 'eat', 'dining', 'lokanta']):
            return {
                "primary_intent": "restaurant",
                "confidence": 0.92,
                "all_intents": ["restaurant"]
            }
        elif any(word in message for word in ['weather', 'temperature', 'hava', 'sıcaklık']):
            return {
                "primary_intent": "weather",
                "confidence": 0.90,
                "all_intents": ["weather"]
            }
        elif any(word in message for word in ['how to get', 'route', 'metro', 'bus', 'transportation', 'nasıl giderim']):
            return {
                "primary_intent": "transportation",
                "confidence": 0.93,
                "all_intents": ["transportation"]
            }
        elif any(word in message for word in ['attraction', 'visit', 'see', 'tourist', 'görülecek']):
            return {
                "primary_intent": "attraction",
                "confidence": 0.91,
                "all_intents": ["attraction"]
            }
        else:
            return {
                "primary_intent": "general",
                "confidence": 0.75,
                "all_intents": ["general"]
            }
    
    def _mock_generate_response(self, prompt: str) -> str:
        """Mock text generation"""
        prompt_lower = prompt.lower()
        
        if 'restaurant' in prompt_lower:
            return "I'd recommend visiting **Mikla** for fine dining with a view, **Çiya Sofrası** for authentic Turkish cuisine, or **Karaköy Lokantası** for a modern take on traditional dishes."
        elif 'weather' in prompt_lower:
            return "The current weather in Istanbul is partly cloudy with a temperature of 18°C. It's a pleasant day with a light breeze from the north."
        elif 'transportation' in prompt_lower or 'taksim' in prompt_lower:
            return "To get to Taksim, you can take the M2 metro line. The journey takes about 20 minutes from most central locations. Alternatively, you can take a taxi or use the nostalgic tram (T2)."
        else:
            return "I'm here to help you discover Istanbul! Feel free to ask about restaurants, attractions, transportation, or anything else about this beautiful city."


def get_mock_llm_client():
    """Get mock LLM client instance"""
    return MockLLMClient()
