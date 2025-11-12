"""
RunPod LLM Client
Integrates with RunPod-hosted Llama 3.1 8B LLM for text generation
"""

import os
import logging
import httpx
from typing import Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RunPodLLMClient:
    """Client for RunPod-hosted LLM API"""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 250
    ):
        """
        Initialize RunPod LLM client
        
        Args:
            api_url: RunPod LLM API URL (default: from LLM_API_URL env)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
        """
        self.api_url = api_url or os.getenv("LLM_API_URL")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.enabled = bool(self.api_url)
        
        if self.enabled:
            logger.info("🚀 RunPod LLM Client initialized")
            logger.info(f"   URL: {self.api_url}")
            logger.info(f"   Timeout: {self.timeout}s")
            logger.info(f"   Max Tokens: {self.max_tokens}")
        else:
            logger.warning("⚠️ RunPod LLM Client disabled (no LLM_API_URL)")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if RunPod LLM service is healthy
        
        Returns:
            Health status dict
        """
        if not self.enabled:
            return {"status": "disabled", "message": "LLM_API_URL not configured"}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.api_url}/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"❌ RunPod LLM health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text using RunPod LLM
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response dict or None if failed
        """
        if not self.enabled:
            logger.warning("RunPod LLM disabled - skipping generation")
            return None
        
        try:
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens or self.max_tokens
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/generate",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"✅ RunPod LLM generated {len(result.get('generated_text', ''))} chars")
                return result
                
        except httpx.TimeoutException:
            logger.error(f"⏱️ RunPod LLM timeout after {self.timeout}s")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ RunPod LLM HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"❌ RunPod LLM generation failed: {e}")
            return None
    
    async def generate_istanbul_response(
        self,
        query: str,
        context: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate Istanbul-specific response using LLM
        
        Args:
            query: User query
            context: Optional context from search/data
            intent: Detected intent type
            
        Returns:
            Generated response text or None
        """
        # Build Istanbul-focused prompt
        system_context = """You are an AI assistant specialized in Istanbul tourism.
Provide helpful, accurate, and friendly information about Istanbul's attractions,
restaurants, neighborhoods, transportation, and local culture.
Keep responses concise and actionable."""
        
        if context:
            prompt = f"""{system_context}

Context: {context}

User Question: {query}

Provide a helpful response:"""
        else:
            prompt = f"""{system_context}

User Question: {query}

Provide a helpful response about Istanbul:"""
        
        result = await self.generate(prompt=prompt, max_tokens=200)
        
        if result and 'generated_text' in result:
            # Extract just the response part (after the prompt)
            full_text = result['generated_text']
            # Try to extract response after "Provide a helpful response:"
            if "Provide a helpful response" in full_text:
                response = full_text.split("Provide a helpful response", 1)[-1]
                response = response.lstrip(":").strip()
                return response
            return full_text
        
        return None


# Global instance
_llm_client: Optional[RunPodLLMClient] = None


def get_llm_client() -> RunPodLLMClient:
    """Get or create global LLM client instance"""
    global _llm_client
    if _llm_client is None:
        _llm_client = RunPodLLMClient()
    return _llm_client


async def generate_llm_response(
    query: str,
    context: Optional[str] = None,
    intent: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to generate LLM response
    
    Args:
        query: User query
        context: Optional context
        intent: Optional intent
        
    Returns:
        Generated response or None
    """
    client = get_llm_client()
    return await client.generate_istanbul_response(query, context, intent)
