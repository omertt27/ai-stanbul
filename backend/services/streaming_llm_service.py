"""
Streaming LLM Service

Provides real-time streaming responses for the chat interface.
Supports both Server-Sent Events (SSE) and WebSocket connections.

Features:
- Token-by-token streaming from LLM
- Graceful fallback to non-streaming
- Connection management
- Error handling with partial response recovery

Author: AI Istanbul Team
Date: December 2024
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional, Dict, Any, Callable
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming responses"""
    chunk_delay_ms: int = 30  # Delay between chunks for smooth UX
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    timeout_seconds: float = 60.0


class StreamingLLMService:
    """
    Service for streaming LLM responses.
    
    Provides real-time token streaming for better user experience.
    """
    
    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[StreamingConfig] = None
    ):
        import os
        self.api_url = api_url or os.getenv("LLM_API_URL")
        self.api_key = (
            api_key or 
            os.getenv("RUNPOD_API_KEY") or 
            os.getenv("HUGGING_FACE_API_KEY") or
            os.getenv("HF_TOKEN")
        )
        self.config = config or StreamingConfig()
        self.enabled = bool(self.api_url)
        
        if self.enabled:
            logger.info("🚀 Streaming LLM Service initialized")
            logger.info(f"   URL: {self.api_url}")
        else:
            logger.warning("⚠️ Streaming LLM Service disabled (no LLM_API_URL)")
    
    async def stream_response(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response token by token.
        
        Args:
            prompt: The formatted prompt for the LLM
            on_token: Optional callback for each token
            session_id: Session ID for logging
            
        Yields:
            Individual tokens/chunks as they're generated
        """
        if not self.enabled:
            logger.warning("Streaming disabled - yielding empty response")
            yield ""
            return
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            # Try streaming endpoint first
            async for chunk in self._stream_from_api(prompt):
                total_tokens += 1
                if on_token:
                    on_token(chunk)
                yield chunk
                
                # Small delay for smooth streaming UX
                await asyncio.sleep(self.config.chunk_delay_ms / 1000)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Streamed {total_tokens} tokens in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            # Fallback to non-streaming
            async for chunk in self._fallback_generate(prompt):
                yield chunk
    
    async def _stream_from_api(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream from LLM API with SSE support.
        
        Handles different streaming formats:
        - OpenAI-compatible streaming
        - Hugging Face TGI streaming
        - Custom RunPod streaming
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Build streaming request
        payload = {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True  # Enable streaming
        }
        
        url = self.api_url.rstrip('/')
        
        # Try different streaming endpoints
        streaming_endpoints = [
            f"{url}/v1/completions",
            f"{url}/generate_stream",
            f"{url}/v1/chat/completions"
        ]
        
        for endpoint in streaming_endpoints:
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                    async with client.stream(
                        "POST",
                        endpoint,
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status_code != 200:
                            continue
                        
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            
                            # Handle SSE format
                            if line.startswith("data: "):
                                data = line[6:]
                                if data == "[DONE]":
                                    return
                                
                                try:
                                    chunk_data = json.loads(data)
                                    
                                    # OpenAI format
                                    if "choices" in chunk_data:
                                        delta = chunk_data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        if content:
                                            yield content
                                    
                                    # Simple format
                                    elif "token" in chunk_data:
                                        yield chunk_data["token"]
                                    elif "text" in chunk_data:
                                        yield chunk_data["text"]
                                    elif "response" in chunk_data:
                                        yield chunk_data["response"]
                                        
                                except json.JSONDecodeError:
                                    # Raw text chunk
                                    yield data
                            else:
                                # Non-SSE format - raw text
                                yield line
                        
                        # Successfully streamed, exit
                        return
                        
            except Exception as e:
                logger.debug(f"Streaming endpoint {endpoint} failed: {e}")
                continue
        
        # If no streaming endpoint worked, fall back
        raise Exception("No streaming endpoint available")
    
    async def _fallback_generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Fallback to non-streaming generation with simulated streaming.
        
        Gets full response and yields it word-by-word for streaming UX.
        """
        logger.info("📝 Using fallback generation with simulated streaming")
        
        try:
            # Import the existing LLM client
            from services.runpod_llm_client import RunPodLLMClient
            
            client = RunPodLLMClient()
            result = await client.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            if result and "generated_text" in result:
                text = result["generated_text"]
                
                # Simulate streaming by yielding words
                words = text.split()
                for i, word in enumerate(words):
                    # Add space before word (except first)
                    if i > 0:
                        yield " "
                    yield word
                    await asyncio.sleep(self.config.chunk_delay_ms / 1000)
            else:
                yield "I apologize, but I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
    
    async def stream_chat_response(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream a complete chat response with metadata.
        
        Yields structured chunks with content and metadata.
        
        Args:
            message: User message
            context: Additional context (location, preferences, etc.)
            language: Response language
            
        Yields:
            Dict with 'type' and 'content' keys
        """
        # Build the prompt
        prompt = self._build_chat_prompt(message, context, language)
        
        # Yield start event
        yield {
            "type": "start",
            "content": "",
            "timestamp": time.time()
        }
        
        # Stream tokens
        full_response = ""
        async for token in self.stream_response(prompt):
            full_response += token
            yield {
                "type": "token",
                "content": token,
                "timestamp": time.time()
            }
        
        # Yield completion event
        yield {
            "type": "complete",
            "content": full_response,
            "timestamp": time.time()
        }
    
    def _build_chat_prompt(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> str:
        """Build a formatted prompt for chat."""
        
        # Language instruction
        lang_instruction = {
            "en": "Respond in English.",
            "tr": "Türkçe yanıt ver.",
            "ar": "الرد بالعربية.",
            "de": "Antworte auf Deutsch.",
            "fr": "Réponds en français.",
            "ru": "Отвечай на русском."
        }.get(language, "Respond in the same language as the question.")
        
        prompt = f"""You are KAM, a friendly and knowledgeable Istanbul tour guide assistant.

{lang_instruction}

"""
        
        # Add context if available
        if context:
            if context.get("location"):
                loc = context["location"]
                prompt += f"User's current location: ({loc.get('lat')}, {loc.get('lon')})\n"
            
            if context.get("preferences"):
                prompt += f"User preferences: {json.dumps(context['preferences'])}\n"
            
            if context.get("conversation_history"):
                prompt += "\nRecent conversation:\n"
                for turn in context["conversation_history"][-3:]:
                    prompt += f"User: {turn.get('query', '')}\n"
                    prompt += f"Assistant: {turn.get('response', '')}\n"
                prompt += "\n"
        
        prompt += f"User's question: {message}\n\nYour helpful response:"
        
        return prompt


# Singleton instance
_streaming_service: Optional[StreamingLLMService] = None


def get_streaming_llm_service() -> StreamingLLMService:
    """Get or create the streaming LLM service singleton."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingLLMService()
    return _streaming_service
