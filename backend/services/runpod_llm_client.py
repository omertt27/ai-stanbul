"""
LLM Client for RunPod and Hugging Face
Supports:
- RunPod-hosted models (vLLM, TGI)
- Hugging Face Inference API
- OpenAI-compatible endpoints
"""

import os
import logging
import httpx
from typing import Dict, Optional, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class RunPodLLMClient:
    """Client for LLM APIs (RunPod, Hugging Face, OpenAI-compatible)"""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 250
    ):
        """
        Initialize LLM client
        
        Args:
            api_url: API URL (from LLM_API_URL env)
                - RunPod: https://YOUR-POD-ID-8000.proxy.runpod.net/v1
                - Hugging Face: https://api-inference.huggingface.co/models/MODEL_NAME
                - OpenAI-compatible: http://localhost:8000/v1
            api_key: API key (from RUNPOD_API_KEY or HUGGING_FACE_API_KEY env)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens to generate
        """
        self.api_url = api_url or os.getenv("LLM_API_URL")
        self.model_name = os.getenv("LLM_MODEL_NAME", "/workspace/llama-3.1-8b")
        
        # Check for API keys (support both RunPod and Hugging Face)
        self.api_key = (
            api_key or 
            os.getenv("RUNPOD_API_KEY") or 
            os.getenv("HUGGING_FACE_API_KEY") or
            os.getenv("HF_TOKEN")
        )
        
        self.timeout = float(os.getenv("LLM_TIMEOUT", timeout))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", max_tokens))
        self.enabled = bool(self.api_url)
        
        # Detect API type
        self.api_type = self._detect_api_type()
        
        if self.enabled:
            logger.info("🚀 LLM Client initialized")
            logger.info(f"   Type: {self.api_type}")
            logger.info(f"   URL: {self.api_url}")
            logger.info(f"   API Key: {'***' + self.api_key[-4:] if self.api_key else 'None'}")
            logger.info(f"   Timeout: {self.timeout}s")
            logger.info(f"   Max Tokens: {self.max_tokens}")
        else:
            logger.warning("⚠️ LLM Client disabled (no LLM_API_URL)")
    
    def _detect_api_type(self) -> str:
        """Detect API type from URL"""
        if not self.api_url:
            return "unknown"
        
        url_lower = self.api_url.lower()
        
        if "huggingface.co" in url_lower or "hf.co" in url_lower:
            return "huggingface"
        elif "runpod" in url_lower:
            return "runpod"
        elif "/v1" in url_lower or "openai" in url_lower:
            return "openai-compatible"
        else:
            return "generic"
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if RunPod LLM service is healthy
        
        Returns:
            Health status dict
        """
        if not self.enabled:
            return {"status": "disabled", "message": "LLM_API_URL not configured"}
        
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try common health check endpoints
                for endpoint in ["/health", "/v1/models", "/"]:
                    try:
                        response = await client.get(
                            f"{self.api_url}{endpoint}",
                            headers=headers
                        )
                        if response.status_code == 200:
                            logger.info(f"✅ RunPod health check OK via {endpoint}")
                            return {
                                "status": "healthy",
                                "endpoint": self.api_url,
                                "response": response.json() if response.headers.get("content-type", "").startswith("application/json") else "OK"
                            }
                    except:
                        continue
                
                return {"status": "unknown", "message": "No valid health endpoint found"}
                
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
        Generate text using LLM (supports multiple API formats)
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response dict or None if failed
        """
        if not self.enabled:
            logger.warning("LLM disabled - skipping generation")
            return None
        
        try:
            if self.api_type == "huggingface":
                return await self._generate_huggingface(prompt, max_tokens, temperature)
            else:
                return await self._generate_openai_compatible(prompt, max_tokens, temperature, top_p)
                
        except httpx.TimeoutException:
            logger.error(f"⏱️ LLM timeout after {self.timeout}s")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ LLM HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return None
    
    async def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float
    ) -> Optional[Dict[str, Any]]:
        """Generate using Hugging Face Inference API format"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens or self.max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle Hugging Face response format
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                logger.info(f"✅ Hugging Face generated {len(generated_text)} chars")
                return {"generated_text": generated_text, "raw": result}
            elif isinstance(result, dict) and 'generated_text' in result:
                generated_text = result['generated_text']
                logger.info(f"✅ Hugging Face generated {len(generated_text)} chars")
                return {"generated_text": generated_text, "raw": result}
            else:
                logger.error("❌ Invalid response format from Hugging Face")
                return None
    
    async def _generate_openai_compatible(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float
    ) -> Optional[Dict[str, Any]]:
        """Generate using OpenAI-compatible API format (vLLM, RunPod, etc.)"""
        
        # For Instruct models, use chat completions format
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use chat/completions endpoint for Instruct models
        url = self.api_url
        if not url.endswith('/chat/completions'):
            if '/v1' in url:
                url = url.rstrip('/') + '/chat/completions'
            else:
                url = url.rstrip('/') + '/v1/chat/completions'
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            
            # Extract text from OpenAI chat format
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                generated_text = message.get('content', '')
                logger.info(f"✅ LLM generated {len(generated_text)} chars")
                return {"generated_text": generated_text, "raw": result}
            else:
                logger.error("❌ Invalid response format from LLM")
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
