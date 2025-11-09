"""
Integration wrapper - connects LLM API to existing AI Istanbul code
Provides a simple client interface for Google Cloud VM LLM API
"""
import requests
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)

class GoogleCloudLLMClient:
    """Client for Google Cloud LLM API"""
    
    def __init__(self, endpoint: str = None):
        """
        Initialize LLM client
        
        Args:
            endpoint: LLM API endpoint URL. If None, uses environment variable or default
        """
        if endpoint is None:
            endpoint = os.getenv('GOOGLE_CLOUD_LLM_ENDPOINT', 'http://35.210.251.24:8000')
        
        self.endpoint = endpoint.rstrip('/')
        self.timeout = 120  # 120 seconds timeout for CPU inference
        logger.info(f"🌐 GoogleCloudLLMClient initialized: {self.endpoint}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check if LLM API is healthy"""
        try:
            response = requests.get(
                f"{self.endpoint}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """
        Generate response using LLM
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Generated text response
        """
        try:
            logger.info(f"🔄 Sending request to LLM: {prompt[:50]}...")
            
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("generated_text", "")
            
            logger.info(f"✅ Received response: {len(generated_text)} characters")
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("⏱️ LLM request timed out")
            return "I apologize, but the response is taking too long. Please try asking in a different way."
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ LLM request failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return "I apologize, but something went wrong. Please try again."
    
    def chat(self, message: str, max_tokens: int = 150, context: Optional[str] = None) -> str:
        """
        Chat interface with optional context
        
        Args:
            message: User message
            max_tokens: Maximum tokens to generate
            context: Optional context to prepend to message
            
        Returns:
            Generated response
        """
        try:
            # Build prompt with context
            if context:
                prompt = f"{context}\n\nUser: {message}\nAssistant:"
            else:
                prompt = f"You are a helpful AI assistant for Istanbul tourism. Provide accurate, friendly information about Istanbul attractions, restaurants, and transportation.\n\nUser: {message}\nAssistant:"
            
            logger.info(f"💬 Chat request: {message[:50]}...")
            
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("generated_text", "")
            
            logger.info(f"✅ Chat response: {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Chat request failed: {e}")
            return "I apologize, but I'm having trouble responding right now. Please try again."
    
    def generate_istanbul_response(self, query: str, intent: Optional[Dict] = None) -> str:
        """
        Generate Istanbul-specific tourism response
        
        Args:
            query: User query about Istanbul
            intent: Optional detected intent dictionary
            
        Returns:
            Generated tourism response
        """
        # Build Istanbul-specific context
        context = """You are an expert AI assistant for Istanbul tourism. 
You provide helpful, accurate, and friendly information about:
- Historical sites (Hagia Sophia, Blue Mosque, Topkapi Palace, etc.)
- Neighborhoods (Sultanahmet, Beyoglu, Kadikoy, etc.)
- Turkish cuisine and restaurants
- Public transportation (metro, tram, ferry, bus)
- Cultural experiences and local tips
- Shopping and bazaars

Always be concise, practical, and enthusiastic about Istanbul."""
        
        # Add intent-specific guidance if available
        if intent:
            intent_type = intent.get('type', '')
            if intent_type == 'attraction':
                context += "\n\nFocus on historical and cultural attractions with practical visiting tips."
            elif intent_type == 'restaurant':
                context += "\n\nFocus on authentic Turkish cuisine and restaurant recommendations."
            elif intent_type == 'transportation':
                context += "\n\nFocus on practical transportation advice and route information."
        
        return self.chat(query, max_tokens=200, context=context)


# Singleton instance
_llm_client = None

def get_llm_client(endpoint: str = None) -> GoogleCloudLLMClient:
    """
    Get or create singleton LLM client instance
    
    Args:
        endpoint: Optional custom endpoint URL
        
    Returns:
        GoogleCloudLLMClient instance
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = GoogleCloudLLMClient(endpoint=endpoint)
    return _llm_client


# Test the client
if __name__ == "__main__":
    import sys
    
    # Setup logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("🧪 Testing Google Cloud LLM Client")
    print("=" * 60)
    
    # Initialize client
    client = get_llm_client()
    
    # Test 1: Health check
    print("\n1️⃣ Testing health check...")
    health = client.health_check()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Model: {health.get('model_name', 'unknown')}")
    
    if health.get('status') != 'healthy':
        print("❌ LLM API is not healthy. Please check the server.")
        sys.exit(1)
    
    # Test 2: Simple generation
    print("\n2️⃣ Testing simple generation...")
    prompt = "What are the top 3 places to visit in Istanbul?"
    response = client.generate(prompt, max_tokens=100)
    print(f"   Prompt: {prompt}")
    print(f"   Response: {response[:200]}...")
    
    # Test 3: Chat interface
    print("\n3️⃣ Testing chat interface...")
    message = "I want to visit museums and try Turkish food"
    response = client.chat(message, max_tokens=150)
    print(f"   Message: {message}")
    print(f"   Response: {response[:200]}...")
    
    # Test 4: Istanbul-specific response
    print("\n4️⃣ Testing Istanbul-specific generation...")
    query = "What should I see in Sultanahmet?"
    intent = {"type": "attraction", "location": "Sultanahmet"}
    response = client.generate_istanbul_response(query, intent=intent)
    print(f"   Query: {query}")
    print(f"   Intent: {intent}")
    print(f"   Response: {response[:200]}...")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
