"""
LLM Client for RunPod and Hugging Face
Supports:
- RunPod-hosted models (vLLM, TGI)
- Hugging Face Inference API
- OpenAI-compatible endpoints

Updated: December 2024
IMPORTANT: This client is for API communication ONLY.
Prompt building is handled by services/llm/prompts.py (PromptBuilder)
"""

import os
import logging
import httpx
import uuid
from typing import Dict, Optional, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMClientConfig:
    """Configuration for LLM client"""
    connect_timeout: float = 10.0  # Connection timeout
    read_timeout: float = 120.0    # Read timeout for generation (matches LLM_TIMEOUT env default)
    max_retries: int = 2
    max_connections: int = 10      # Connection pool size
    max_keepalive: int = 5         # Keep-alive connections


class RunPodLLMClient:
    """Client for LLM APIs (RunPod, Hugging Face, OpenAI-compatible)
    
    Features:
    - Persistent connection pool for better performance
    - Request correlation IDs for tracing
    - Separate connect/read timeouts
    - Automatic resource cleanup
    """
    
    # Class-level shared client for connection pooling
    _shared_client: Optional[httpx.AsyncClient] = None
    _client_lock: Optional[Any] = None
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 1024,
        config: Optional[LLMClientConfig] = None
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
            max_tokens: Maximum tokens to generate (default: 1024 for full responses)
            config: Optional configuration for client (timeouts, retries, etc.)
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
        
        # Connection pool configuration
        self.config = config or LLMClientConfig()
        
        # Detect API type
        self.api_type = self._detect_api_type()
        
        # Use max_tokens from environment variable (respects .env configuration)
        # Default is 768 for balanced speed and completeness
        # Can be overridden per request if needed for specific use cases
        
        if self.enabled:
            logger.info("🚀 LLM Client initialized")
            logger.info(f"   Type: {self.api_type}")
            logger.info(f"   URL: {self.api_url}")
            logger.info(f"   API Key: {'***' + self.api_key[-4:] if self.api_key else 'None'}")
            logger.info(f"   Timeout: {self.timeout}s")
            logger.info(f"   Max Tokens: {self.max_tokens}")
            logger.info(f"   Connection Pool: {self.config.max_connections} max, {self.config.max_keepalive} keepalive")
        else:
            logger.warning("⚠️ LLM Client disabled (no LLM_API_URL)")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create shared async HTTP client with connection pooling."""
        if RunPodLLMClient._shared_client is None or RunPodLLMClient._shared_client.is_closed:
            # Use the larger of config.read_timeout or self.timeout (from LLM_TIMEOUT env)
            read_timeout = max(self.config.read_timeout, self.timeout)
            timeout = httpx.Timeout(
                connect=self.config.connect_timeout,
                read=read_timeout,
                write=10.0,
                pool=5.0
            )
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive
            )
            
            # Check if h2 is available for HTTP/2 support
            try:
                import h2
                use_http2 = True
            except ImportError:
                use_http2 = False
                logger.info("ℹ️ HTTP/2 not available (h2 package not installed), using HTTP/1.1")
            
            RunPodLLMClient._shared_client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=use_http2
            )
            logger.info(f"🔌 Created shared HTTP client with connection pool (HTTP/2: {use_http2})")
        return RunPodLLMClient._shared_client
    
    async def close(self):
        """Close the shared HTTP client and release resources."""
        if RunPodLLMClient._shared_client is not None:
            await RunPodLLMClient._shared_client.aclose()
            RunPodLLMClient._shared_client = None
            logger.info("🔌 Closed shared HTTP client")
    
    def _detect_api_type(self) -> str:
        """Detect API type from URL"""
        if not self.api_url:
            return "unknown"
        
        url_lower = self.api_url.lower()
        
        if "huggingface.co" in url_lower or "hf.co" in url_lower:
            return "huggingface"
        elif "/v1" in url_lower or "openai" in url_lower:
            return "openai-compatible"
        elif "runpod" in url_lower:
            return "runpod"
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
                # Determine base URL and endpoints to try
                base_url = self.api_url.rstrip('/')
                
                # If URL ends with /v1, try models endpoint directly
                if base_url.endswith('/v1'):
                    endpoints = ["/models", "/health", ""]
                else:
                    endpoints = ["/v1/models", "/health", "/v1", ""]
                
                for endpoint in endpoints:
                    try:
                        url = f"{base_url}{endpoint}"
                        logger.debug(f"Trying health check at: {url}")
                        response = await client.get(url, headers=headers)
                        if response.status_code == 200:
                            logger.info(f"✅ LLM health check OK via {url}")
                            
                            # Try to parse JSON response
                            try:
                                response_data = response.json()
                            except:
                                response_data = "OK"
                            
                            return {
                                "status": "healthy",
                                "llm_available": True,
                                "endpoint": self.api_url,
                                "model": self.model_name,
                                "api_type": self.api_type,
                                "response": response_data
                            }
                    except Exception as e:
                        logger.debug(f"Health check failed at {endpoint}: {e}")
                        continue
                
                return {
                    "status": "unknown",
                    "message": f"No valid health endpoint found at {base_url}",
                    "endpoint": self.api_url
                }
                
        except Exception as e:
            logger.error(f"❌ LLM health check failed: {e}")
            return {"status": "unhealthy", "error": str(e), "endpoint": self.api_url}
    
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        request_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text using LLM (supports multiple API formats)
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling parameter
            request_id: Optional correlation ID for logging
            
        Returns:
            Generated response dict or None if failed
        """
        if not self.enabled:
            logger.warning("LLM disabled - skipping generation")
            return None
        
        # Generate correlation ID if not provided
        req_id = request_id or str(uuid.uuid4())[:8]
        
        # Get shared client with connection pool
        client = await self._get_client()
        
        try:
            if self.api_type == "huggingface":
                return await self._generate_huggingface(prompt, max_tokens, temperature, client, req_id)
            elif self.api_type == "runpod":
                return await self._generate_runpod_custom(prompt, max_tokens, temperature, top_p, client, req_id)
            else:
                return await self._generate_openai_compatible(prompt, max_tokens, temperature, top_p, client, req_id)
                
        except httpx.TimeoutException:
            logger.error(f"[{req_id}] ⏱️ LLM timeout after {self.config.read_timeout}s")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"[{req_id}] ❌ LLM HTTP error: {e.response.status_code}")
            logger.error(f"[{req_id}]    Response: {e.response.text if hasattr(e.response, 'text') else 'N/A'}")
            return None
        except Exception as e:
            import traceback
            logger.error(f"[{req_id}] ❌ LLM generation failed: {e}")
            logger.error(f"[{req_id}] ❌ Traceback: {traceback.format_exc()}")
            return None
    
    async def _generate_huggingface(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        client: httpx.AsyncClient,
        req_id: str
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
            logger.info(f"[{req_id}] ✅ Hugging Face generated {len(generated_text)} chars")
            return {"generated_text": generated_text, "raw": result}
        elif isinstance(result, dict) and 'generated_text' in result:
            generated_text = result['generated_text']
            logger.info(f"[{req_id}] ✅ Hugging Face generated {len(generated_text)} chars")
            return {"generated_text": generated_text, "raw": result}
        else:
            logger.error(f"[{req_id}] ❌ Invalid response format from Hugging Face")
            return None
    
    async def _generate_runpod_custom(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        client: httpx.AsyncClient,
        req_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate using RunPod custom /generate endpoint format.
        
        IMPORTANT: This method now passes the prompt directly to the API.
        Prompt building is handled upstream by PromptBuilder (prompts.py).
        """
        
        # Use the prompt as-is - it's already built by PromptBuilder
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use /generate endpoint (RunPod custom)
        url = self.api_url.rstrip('/')
        if not url.endswith('/generate'):
            url = url + '/generate'
        
        logger.info(f"[{req_id}] 🔄 Calling RunPod LLM at: {url}")
        logger.debug(f"[{req_id}]    Prompt length: {len(prompt)} chars")
        
        response = await client.post(
            url,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        
        # Handle RunPod custom response format
        # Support both 'response' and 'text' fields (different RunPod versions)
        generated_text = None
        
        if 'response' in result:
            generated_text = result['response']
            logger.info(f"[{req_id}] ✅ RunPod generated {len(generated_text)} chars in {result.get('generation_time', 0):.2f}s")
            logger.info(f"[{req_id}]    Tokens: {result.get('tokens_generated', 0)}, Speed: {result.get('tokens_per_second', 0):.1f} t/s")
        elif 'text' in result:
            generated_text = result['text']
            usage = result.get('usage', {})
            logger.info(f"[{req_id}] ✅ RunPod generated {len(generated_text)} chars")
            logger.info(f"[{req_id}]    Tokens: {usage.get('completion_tokens', 0)} completion, {usage.get('total_tokens', 0)} total")
        else:
            logger.error(f"[{req_id}] ❌ Invalid response format from RunPod: {result.keys() if isinstance(result, dict) else type(result)}")
            return None
        
        # Return result even if text is empty (let caller handle it)
        # Empty text might indicate model/prompt issue
        if generated_text is not None:
            # Hard limit: truncate responses that are too long
            MAX_RESPONSE_LENGTH = 2048
            if len(generated_text) > MAX_RESPONSE_LENGTH:
                logger.warning(f"[{req_id}] ⚠️ Response too long ({len(generated_text)} chars), truncating")
                generated_text = generated_text[:MAX_RESPONSE_LENGTH].rsplit('.', 1)[0] + '.'
            
            if not generated_text:
                logger.warning(f"[{req_id}] ⚠️ Model returned empty text (tokens: {result.get('usage', {}).get('completion_tokens', 0)})")
            
            return {"generated_text": generated_text, "raw": result, "request_id": req_id}
        else:
            logger.error(f"[{req_id}] ❌ No 'text' or 'response' field in result")
            return None
    
    async def _generate_openai_compatible(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        client: httpx.AsyncClient,
        req_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Generate using OpenAI-compatible API format (vLLM, RunPod, etc.)
        
        IMPORTANT: This method now passes the prompt directly to the API.
        Prompt building is handled upstream by PromptBuilder (prompts.py).
        """
        
        # Use chat/completions format for vLLM (OpenAI-compatible)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["<|eot_id|>", "\n\nUser:", "\n\n---"]
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use /chat/completions endpoint (vLLM OpenAI-compatible)
        url = self.api_url
        if not url.endswith('/chat/completions'):
            if '/v1' in url:
                url = url.rstrip('/') + '/chat/completions'
            else:
                url = url.rstrip('/') + '/v1/chat/completions'
        
        logger.info(f"[{req_id}] 🔄 Calling LLM at: {url}")
        logger.debug(f"[{req_id}]    Prompt length: {len(prompt)} chars")
        
        response = await client.post(
            url,
            json=payload,
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        
        # Handle OpenAI chat/completions response format
        generated_text = None
        
        if 'choices' in result:
            choices = result['choices']
            if isinstance(choices, list) and len(choices) > 0:
                # Chat completion format: choices[0].message.content
                if 'message' in choices[0]:
                    generated_text = choices[0]['message'].get('content', '')
                # Fallback to text format (legacy)
                elif 'text' in choices[0]:
                    generated_text = choices[0].get('text', '')
        elif 'text' in result:
            generated_text = result['text']
        elif 'generated_text' in result:
            generated_text = result['generated_text']
        
        if generated_text:
            MAX_RESPONSE_LENGTH = 2048
            if len(generated_text) > MAX_RESPONSE_LENGTH:
                logger.warning(f"[{req_id}] ⚠️ Response too long, truncating")
                generated_text = generated_text[:MAX_RESPONSE_LENGTH].rsplit('.', 1)[0] + '.'
            
            logger.info(f"[{req_id}] ✅ LLM generated {len(generated_text)} chars")
            return {"generated_text": generated_text, "raw": result, "request_id": req_id}
        else:
            logger.error(f"[{req_id}] ❌ Invalid response format from LLM")
            return None
    
    async def generate_istanbul_response(
        self,
        query: str,
        context: Optional[str] = None,
        intent: Optional[str] = None,
        language: str = "en"
    ) -> Optional[str]:
        """
        Generate Istanbul-specific response using LLM.
        
        NOTE: This method now uses the unified PromptBuilder from prompts.py
        for consistent prompt construction across all LLM calls.
        
        Args:
            query: User query
            context: Optional context from search/data
            intent: Detected intent type
            language: Response language (default: en)
            
        Returns:
            Generated response text or None
        """
        from services.llm.prompts import PromptBuilder
        
        prompt_builder = PromptBuilder()
        
        # Build signals from intent
        signals = {}
        if intent:
            if intent in ['transportation', 'directions', 'route']:
                signals['needs_transportation'] = True
            elif intent in ['restaurant', 'food', 'dining']:
                signals['needs_restaurant'] = True
            elif intent in ['attraction', 'museum', 'place']:
                signals['needs_attraction'] = True
            elif intent == 'weather':
                signals['needs_weather'] = True
        
        # Build context dict
        prompt_context = {
            'database': context or '',
            'rag': context or '',
            'services': {}
        }
        
        # Build prompt using unified PromptBuilder
        prompt = prompt_builder.build_prompt(
            query=query,
            signals=signals,
            context=prompt_context,
            language=language
        )
        
        result = await self.generate(prompt=prompt, max_tokens=500)
        
        if result and 'generated_text' in result:
            return result['generated_text']
        
        return None
    
    async def generate_with_service_context(
        self,
        query: str,
        intent: Optional[str] = None,
        entities: Optional[Dict] = None,
        service_context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> Optional[str]:
        """
        Generate response using service context data.
        
        NOTE: This method now uses the unified PromptBuilder from prompts.py
        for consistent prompt construction across all LLM calls.
        
        Args:
            query: User query
            intent: Detected intent
            entities: Extracted entities
            service_context: Context from LLMContextBuilder with service data
            language: Response language (default: en)
            
        Returns:
            Generated response text or None
        """
        from services.llm.prompts import PromptBuilder
        
        prompt_builder = PromptBuilder()
        
        # Build signals from intent
        signals = {}
        if intent:
            if intent in ['transportation', 'directions', 'route']:
                signals['needs_transportation'] = True
            elif intent in ['restaurant', 'food', 'dining']:
                signals['needs_restaurant'] = True
            elif intent in ['attraction', 'museum', 'place']:
                signals['needs_attraction'] = True
            elif intent == 'weather':
                signals['needs_weather'] = True
        
        # Build context dict from service context
        prompt_context = {
            'database': '',
            'rag': '',
            'services': service_context.get('service_data', {}) if service_context else {}
        }
        
        # Format service context into database/rag if available
        if service_context and service_context.get('service_data'):
            from services.llm_context_builder import get_context_builder
            context_builder = get_context_builder()
            formatted = context_builder.format_context_for_llm(service_context)
            if formatted:
                prompt_context['database'] = formatted
        
        # Build prompt using unified PromptBuilder
        prompt = prompt_builder.build_prompt(
            query=query,
            signals=signals,
            context=prompt_context,
            language=language
        )
        
        logger.info(f"🤖 Generating service-enhanced response for intent: {intent}")
        
        result = await self.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            top_p=0.9
        )
        
        if result and 'generated_text' in result:
            return result['generated_text']
        
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
