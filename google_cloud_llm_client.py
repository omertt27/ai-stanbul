"""
Google Cloud LLM Client for Llama 3.1 8B
Centralized client for all LLM requests in the AI Istanbul system

This client ensures ALL prompts and handlers use the Google Cloud deployed Llama 3.1 8B model.

Usage:
    from google_cloud_llm_client import get_llm_client
    
    client = get_llm_client()
    response = client.generate("What are the best places in Istanbul?", max_tokens=150)
"""

import os
import logging
import requests
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class GoogleCloudLLMClient:
    """
    Client for Google Cloud Llama 3.1 8B API
    
    This client communicates with the API server running on the Google Cloud VM.
    All LLM requests in the AI Istanbul system should route through this client.
    """
    
    def __init__(self, endpoint: Optional[str] = None, timeout: int = 30):
        """
        Initialize Google Cloud LLM client
        
        Args:
            endpoint: API endpoint URL (default: uses environment variable or fallback)
            timeout: Request timeout in seconds (default: 30)
        """
        self.endpoint = endpoint or os.getenv(
            'GOOGLE_CLOUD_LLM_ENDPOINT',
            'http://35.210.251.24:8000'
        )
        self.timeout = timeout
        
        # Remove trailing slash
        self.endpoint = self.endpoint.rstrip('/')
        
        logger.info(f"🌐 Google Cloud LLM Client initialized")
        logger.info(f"📍 Endpoint: {self.endpoint}")
        logger.info(f"⏱️  Timeout: {timeout}s")
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify connection to the API server"""
        try:
            response = requests.get(
                f"{self.endpoint}/health",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Connected to Google Cloud LLM API")
                logger.info(f"📊 Model loaded: {data.get('model_loaded', 'unknown')}")
                logger.info(f"🖥️  Device: {data.get('device', 'unknown')}")
            else:
                logger.warning(f"⚠️ API health check returned status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to API: {str(e)}")
            logger.error(f"⚠️ Make sure the API server is running at {self.endpoint}")
    
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 150,
                 temperature: float = 0.7,
                 **kwargs) -> str:
        """
        Generate text completion
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (default: 150)
            temperature: Sampling temperature (default: 0.7)
            **kwargs: Additional parameters
        
        Returns:
            Generated text
        """
        try:
            start_time = datetime.now()
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                f"{self.endpoint}/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                elapsed = (datetime.now() - start_time).total_seconds()
                
                logger.debug(f"✅ Generation successful ({elapsed:.2f}s)")
                logger.debug(f"📝 Tokens generated: {data.get('tokens_generated', 'unknown')}")
                
                return data.get('response', '')
            else:
                logger.error(f"❌ API error: {response.status_code} - {response.text}")
                return f"Error: API returned status {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Request timeout after {self.timeout}s")
            return "Error: Request timeout"
        except Exception as e:
            logger.error(f"❌ Generation failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def chat(self,
             prompt: str,
             max_tokens: int = 200,
             temperature: float = 0.7,
             **kwargs) -> str:
        """
        Chat completion (conversation-optimized)
        
        Args:
            prompt: Chat prompt
            max_tokens: Maximum tokens to generate (default: 200)
            temperature: Sampling temperature (default: 0.7)
            **kwargs: Additional parameters
        
        Returns:
            Chat response
        """
        try:
            start_time = datetime.now()
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                f"{self.endpoint}/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                elapsed = (datetime.now() - start_time).total_seconds()
                
                logger.debug(f"✅ Chat successful ({elapsed:.2f}s)")
                logger.debug(f"📝 Tokens generated: {data.get('tokens_generated', 'unknown')}")
                
                return data.get('response', '')
            else:
                logger.error(f"❌ API error: {response.status_code} - {response.text}")
                return f"Error: API returned status {response.status_code}"
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Request timeout after {self.timeout}s")
            return "Error: Request timeout"
        except Exception as e:
            logger.error(f"❌ Chat failed: {str(e)}")
            return f"Error: {str(e)}"
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status
        
        Returns:
            Health status dictionary
        """
        try:
            response = requests.get(
                f"{self.endpoint}/health",
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Status code: {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


# Singleton instance
_client_instance: Optional[GoogleCloudLLMClient] = None


def get_llm_client(endpoint: Optional[str] = None) -> GoogleCloudLLMClient:
    """
    Get or create the singleton LLM client instance
    
    This ensures all parts of the system use the same client instance.
    
    Args:
        endpoint: Optional custom endpoint (uses default if not provided)
    
    Returns:
        GoogleCloudLLMClient instance
    """
    global _client_instance
    
    if _client_instance is None:
        _client_instance = GoogleCloudLLMClient(endpoint=endpoint)
        logger.info("🚀 Created new Google Cloud LLM client instance")
    
    return _client_instance


def test_client():
    """Test the Google Cloud LLM client"""
    print("🧪 Testing Google Cloud LLM Client")
    print("=" * 60)
    
    # Create client
    client = get_llm_client()
    
    # Test health check
    print("\n1️⃣ Health Check:")
    health = client.health_check()
    print(f"   Status: {health.get('status', 'unknown')}")
    print(f"   Model Loaded: {health.get('model_loaded', 'unknown')}")
    print(f"   Device: {health.get('device', 'unknown')}")
    
    # Test generation
    print("\n2️⃣ Simple Generation:")
    prompt = "Istanbul is"
    print(f"   Prompt: {prompt}")
    response = client.generate(prompt, max_tokens=50)
    print(f"   Response: {response[:100]}...")
    
    # Test chat
    print("\n3️⃣ Chat:")
    prompt = "What are the top 3 places to visit in Istanbul?"
    print(f"   Prompt: {prompt}")
    response = client.chat(prompt, max_tokens=200)
    print(f"   Response: {response[:200]}...")
    
    print("\n" + "=" * 60)
    print("✅ All tests complete!")


if __name__ == "__main__":
    # Run tests
    test_client()
