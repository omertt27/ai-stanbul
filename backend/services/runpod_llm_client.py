"""
LLM Client for RunPod and Hugging Face
Supports:
- RunPod-hosted models (vLLM, TGI)
- Hugging Face Inference API
- OpenAI-compatible endpoints

Updated: December 2024 - Using improved standardized prompt templates
"""

import os
import logging
import httpx
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Import improved prompt templates
from IMPROVED_PROMPT_TEMPLATES import (
    IMPROVED_BASE_PROMPT,
    INTENT_PROMPTS,
    CONTEXT_FORMAT_TEMPLATE
)

load_dotenv()

logger = logging.getLogger(__name__)


class RunPodLLMClient:
    """Client for LLM APIs (RunPod, Hugging Face, OpenAI-compatible)"""
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_tokens: int = 1024
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
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the language of the input text
        
        Supports: English, Turkish, Arabic, Russian, French, German
        
        Args:
            text: Input text to detect language
            
        Returns:
            Language name in English
        """
        text_lower = text.lower()
        
        # Turkish detection (ı, ş, ğ, ü, ö, ç characters)
        turkish_chars = ['ı', 'ş', 'ğ', 'ü', 'ö', 'ç']
        turkish_words = ['nerede', 'nasıl', 'ne', 'var', 'mı', 'mi', 'mu', 'mü', 
                        'için', 'ile', 'gitmek', 'yemek', 'restoran', 'nereye',
                        'İstanbul', 'Taksim', 'Beyoğlu', 'neresi', 'hangi']
        if any(char in text_lower for char in turkish_chars) or \
           any(word in text_lower for word in turkish_words):
            return "Turkish (Türkçe)"
        
        # Arabic detection (Arabic script)
        arabic_chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 
                       'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 
                       'ل', 'م', 'ن', 'ه', 'و', 'ي']
        if any(char in text for char in arabic_chars):
            return "Arabic (العربية)"
        
        # Russian detection (Cyrillic script)
        russian_chars = ['а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 
                        'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 
                        'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
        if any(char in text_lower for char in russian_chars):
            return "Russian (Русский)"
        
        # French detection (common words and accents)
        french_words = ['où', 'comment', 'quel', 'quelle', 'est', 'sont', 'pour',
                       'avec', 'dans', 'sur', 'une', 'des', 'les', 'très',
                       'restaurant', 'musée', 'près', 'château']
        french_chars = ['é', 'è', 'ê', 'à', 'ù', 'û', 'ô', 'î', 'ï', 'ç', 'œ']
        if any(word in text_lower for word in french_words) or \
           any(char in text_lower for char in french_chars):
            return "French (Français)"
        
        # German detection (common words and characters)
        german_words = ['wo', 'wie', 'was', 'welche', 'welcher', 'ist', 'sind', 
                       'für', 'mit', 'von', 'zu', 'nach', 'über', 'unter',
                       'restaurant', 'essen', 'museum', 'schloss']
        german_chars = ['ä', 'ö', 'ü', 'ß']
        if any(word in text_lower.split() for word in german_words) or \
           any(char in text_lower for char in german_chars):
            return "German (Deutsch)"
        
        # Default to English
        return "English"
    
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
    
    def _format_llama_chat_prompt(self, prompt: str) -> str:
        """
        Format prompt using Llama 3.1 chat template.
        
        Llama 3.1 requires special tokens for proper instruction following:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            Formatted prompt with Llama 3.1 chat tokens
        """
        # Split prompt into system instructions and user query
        # Look for patterns like "Current User Question:" to separate
        if "Current User Question:" in prompt:
            parts = prompt.split("Current User Question:")
            system_part = parts[0].strip()
            user_part = parts[1].replace("Your Direct Answer:", "").strip()
        elif "User Question:" in prompt:
            parts = prompt.split("User Question:")
            system_part = parts[0].strip()
            user_part = parts[1].strip()
        else:
            # If no clear separation, treat first 80% as system, last 20% as user
            split_point = int(len(prompt) * 0.8)
            system_part = prompt[:split_point].strip()
            user_part = prompt[split_point:].strip()
        
        # Build Llama 3.1 chat format
        formatted = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_part}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_part}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        logger.debug(f"🔄 Llama chat template applied:")
        logger.debug(f"   System part length: {len(system_part)} chars")
        logger.debug(f"   User part length: {len(user_part)} chars")
        logger.debug(f"   User part preview: {user_part[:100]}...")
        
        return formatted

    async def _generate_openai_compatible(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float
    ) -> Optional[Dict[str, Any]]:
        """Generate using OpenAI-compatible API format (vLLM, RunPod, etc.)"""
        
        # Format prompt for Llama 3.1 chat template
        # This fixes the echo issue where LLM was returning prompt fragments
        formatted_prompt = self._format_llama_chat_prompt(prompt)
        logger.debug(f"Formatted prompt length: {len(formatted_prompt)} chars")
        
        # Use standard completions format for vLLM
        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt,  # Use formatted prompt with chat template
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["<|eot_id|>", "\n\nUser:", "\n\n---"]  # Stop at end-of-turn token
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Use /completions endpoint (vLLM standard)
        url = self.api_url
        if not url.endswith('/completions'):
            if '/v1' in url:
                url = url.rstrip('/') + '/completions'
            else:
                url = url.rstrip('/') + '/v1/completions'
        
        logger.info(f"🔄 Calling LLM at: {url}")
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle multiple response formats from different vLLM versions
            generated_text = None
            
            # Format 1: Standard vLLM with choices array
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0].get('text', '')
            
            # Format 2: Direct text field (RunPod custom format)
            elif 'text' in result:
                generated_text = result['text']
            
            # Format 3: Generated_text field
            elif 'generated_text' in result:
                generated_text = result['generated_text']
            
            if generated_text:
                logger.info(f"✅ LLM generated {len(generated_text)} chars")
                return {"generated_text": generated_text, "raw": result}
            else:
                logger.error(f"❌ Invalid response format from LLM: {result.keys() if isinstance(result, dict) else type(result)}")
                return None
    
    async def generate_istanbul_response(
        self,
        query: str,
        context: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate Istanbul-specific response using LLM with improved prompts
        
        Args:
            query: User query
            context: Optional context from search/data
            intent: Detected intent type
            
        Returns:
            Generated response text or None
        """
        # Detect query language
        detected_language = self._detect_language(query)
        
        # Build prompt using improved template
        system_prompt = IMPROVED_BASE_PROMPT.format(detected_language=detected_language)
        
        # Add intent-specific guidance if available
        if intent and intent in INTENT_PROMPTS:
            system_prompt += f"\n\n{INTENT_PROMPTS[intent]}"
        
        if context:
            prompt = f"""{system_prompt}

CONTEXT DATA:
{context}

USER QUESTION: {query}

YOUR RESPONSE (in {detected_language}):"""
        else:
            prompt = f"""{system_prompt}

USER QUESTION: {query}

YOUR RESPONSE (in {detected_language}):"""
        
        result = await self.generate(prompt=prompt, max_tokens=200)
        
        if result and 'generated_text' in result:
            # Extract just the response part (after the prompt)
            full_text = result['generated_text']
            # Try to extract response after "YOUR RESPONSE"
            if "YOUR RESPONSE" in full_text:
                response = full_text.split("YOUR RESPONSE", 1)[-1]
                response = response.lstrip(":").lstrip("(in").split("):")[1] if "):" in response else response.lstrip(":")
                response = response.strip()
                return response
            return full_text
        
        return None
    
    async def generate_with_service_context(
        self,
        query: str,
        intent: Optional[str] = None,
        entities: Optional[Dict] = None,
        service_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate response using service context data with improved prompts
        
        Args:
            query: User query
            intent: Detected intent
            entities: Extracted entities
            service_context: Context from LLMContextBuilder with service data
            
        Returns:
            Generated response text or None
        """
        if not service_context or not service_context.get("service_data"):
            # No service data, fall back to basic generation
            return await self.generate_istanbul_response(query, intent=intent)
        
        # Import here to avoid circular dependency
        from services.llm_context_builder import get_context_builder
        
        context_builder = get_context_builder()
        formatted_context = context_builder.format_context_for_llm(service_context)
        
        # Detect query language
        detected_language = self._detect_language(query)
        
        # Build enhanced prompt using improved templates
        system_prompt = IMPROVED_BASE_PROMPT.format(detected_language=detected_language)
        
        # Add intent-specific guidance if available
        if intent and intent in INTENT_PROMPTS:
            system_prompt += f"\n\n{INTENT_PROMPTS[intent]}"
            logger.info(f"🎯 Added intent-specific prompt for: {intent}")
        
        if formatted_context:
            # Use improved context formatting
            from datetime import datetime
            context_section = f"""
---CONTEXT DATA PROVIDED---

{formatted_context}

**Data Sources**: {', '.join(service_context.get('service_data', {}).keys()) if service_context.get('service_data') else 'Multiple sources'}
**Data Status**: Real-time

---END OF CONTEXT---

**REMEMBER**: Base your response on this CONTEXT data. Include specific details (names, locations, ratings, prices) from above.
"""
            
            prompt = f"""{system_prompt}

{context_section}

USER QUESTION: {query}

YOUR RESPONSE (in {detected_language}):"""
        else:
            prompt = f"""{system_prompt}

USER QUESTION: {query}

YOUR RESPONSE (in {detected_language}):"""
        
        logger.info(f"🤖 Generating service-enhanced response for intent: {intent}")
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
        result = await self.generate(
            prompt=prompt,
            max_tokens=300,  # Allow longer responses with service data
            temperature=0.7,
            top_p=0.9
        )
        
        if result and 'generated_text' in result:
            full_text = result['generated_text']
            
            # Extract response after "YOUR RESPONSE"
            if "YOUR RESPONSE" in full_text:
                response = full_text.split("YOUR RESPONSE", 1)[-1]
                response = response.lstrip(":").lstrip("(in").split("):")[1] if "):" in response else response.lstrip(":")
                response = response.strip()
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
