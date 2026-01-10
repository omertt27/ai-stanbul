"""
Streaming LLM Service

Provides real-time streaming responses for the chat interface.
Supports both Server-Sent Events (SSE) and WebSocket connections.

Features:
- Token-by-token streaming from LLM
- Graceful fallback to non-streaming
- Connection management
- Error handling with partial response recovery
- Semantic caching for instant responses (embedding-based)
- Circuit breaker for resilience
- Retry logic with exponential backoff
- Request deduplication
- Metrics tracking
- Context Assembly Layer integration
- Response validation with hallucination detection
- Streaming quality monitoring with early abort

Author: AI Istanbul Team
Date: December 2024
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import AsyncGenerator, Optional, Dict, Any, Callable, Set, List
from dataclasses import dataclass, field
from collections import defaultdict
import httpx

logger = logging.getLogger(__name__)

# Import new components
try:
    from services.llm.context_assembly import get_context_assembler, ContextType
    CONTEXT_ASSEMBLY_AVAILABLE = True
except ImportError:
    CONTEXT_ASSEMBLY_AVAILABLE = False
    logger.warning("⚠️ Context Assembly Layer not available")

try:
    from services.llm.response_validator import get_response_validator
    RESPONSE_VALIDATOR_AVAILABLE = True
except ImportError:
    RESPONSE_VALIDATOR_AVAILABLE = False
    logger.warning("⚠️ Response Validator not available")

try:
    from services.llm.semantic_cache import get_semantic_cache
    SEMANTIC_CACHE_AVAILABLE = True
except ImportError:
    SEMANTIC_CACHE_AVAILABLE = False
    logger.warning("⚠️ Semantic Cache not available")


def clean_streaming_response(text: str) -> str:
    """
    Clean prompt leakage and artifacts from streaming LLM response.
    
    This removes common patterns where the LLM echoes parts of the prompt
    or instruction text back in its response.
    """
    if not text:
        return text
    
    original_text = text
    
    # Patterns that indicate prompt leakage (should not appear in response)
    # Check from the END of the response for these patterns
    leakage_patterns = [
        "⚠️ CRITICAL",
        "❌ DO NOT",
        "CRITICAL: Your response",
        "Your response MUST",
        "[Respond in ",
        "**Map:** A map",
        "A map will be shown",
        "User Question:",
        "Answer:",
        "---\n\n⚠️",
        "CRITICAL LANGUAGE",
        "\n\nUser:",
        "\n\nAssistant:",
    ]
    
    # Find the earliest leakage pattern and truncate from there
    earliest_idx = len(text)
    found_pattern = None
    
    for pattern in leakage_patterns:
        idx = text.find(pattern)
        if idx != -1 and idx < earliest_idx and idx > 30:  # Must have some content before
            earliest_idx = idx
            found_pattern = pattern
    
    if found_pattern:
        text = text[:earliest_idx].rstrip(' \n-')
        logger.warning(f"🧹 Cleaned prompt leakage: '{found_pattern}' at position {earliest_idx}")
    
    # Additional cleanup: remove incomplete sentences at the end that might be cut off
    # Look for trailing fragments after a clean sentence
    if text and not text[-1] in '.!?؟':
        # Find the last complete sentence
        last_sentence_end = max(
            text.rfind('. '),
            text.rfind('! '),
            text.rfind('? '),
            text.rfind('.\n'),
            text.rfind('!\n'),
            text.rfind('?\n'),
        )
        # Only truncate if there's substantial content and the trailing part looks like garbage
        if last_sentence_end > len(text) * 0.7:
            trailing = text[last_sentence_end + 1:].strip()
            # Check if trailing looks like prompt leakage (contains instruction markers)
            if any(marker in trailing for marker in ['[', '⚠️', '❌', '---', 'CRITICAL']):
                text = text[:last_sentence_end + 1].rstrip()
                logger.warning(f"🧹 Removed trailing garbage after sentence end")
    
    return text


@dataclass
class StreamingConfig:
    """Configuration for streaming responses"""
    chunk_delay_ms: int = 30  # Delay between chunks for smooth UX
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    timeout_seconds: float = 60.0
    # Retry settings
    max_retries: int = 2
    retry_delay_base: float = 0.5  # Exponential backoff base
    # Circuit breaker settings
    circuit_breaker_threshold: int = 5  # Failures before opening
    circuit_breaker_timeout: float = 30.0  # Seconds before half-open
    # Cache settings
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.85


@dataclass
class StreamingMetrics:
    """Metrics for streaming service"""
    total_requests: int = 0
    successful_streams: int = 0
    failed_streams: int = 0
    fallback_used: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_streamed: int = 0
    total_latency_ms: float = 0.0
    retry_count: int = 0
    circuit_breaker_trips: int = 0
    
    def get_stats(self) -> Dict[str, Any]:
        avg_latency = self.total_latency_ms / max(self.total_requests, 1)
        cache_hit_rate = self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        success_rate = self.successful_streams / max(self.total_requests, 1)
        return {
            'total_requests': self.total_requests,
            'success_rate': round(success_rate * 100, 1),
            'avg_latency_ms': round(avg_latency, 1),
            'cache_hit_rate': round(cache_hit_rate * 100, 1),
            'fallback_rate': round(self.fallback_used / max(self.total_requests, 1) * 100, 1),
            'total_tokens': self.total_tokens_streamed,
            'retry_count': self.retry_count,
            'circuit_breaker_trips': self.circuit_breaker_trips
        }


class CircuitBreaker:
    """Simple circuit breaker for streaming service"""
    
    def __init__(self, threshold: int = 5, timeout: float = 30.0):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
    
    def can_proceed(self) -> bool:
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if timeout has passed
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.state = "open"
            logger.warning(f"🔴 Circuit breaker OPEN after {self.failures} failures")


class StreamingLLMService:
    """
    Service for streaming LLM responses.
    
    Provides real-time token streaming for better user experience.
    
    Architecture:
    - Context Assembly Layer: Intelligent context selection and grounding
    - Semantic Cache: Embedding-based similarity caching
    - Response Validator: Post-generation hallucination detection
    - Streaming Quality Monitor: Early abort for template responses
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
        
        # Initialize core components
        self.metrics = StreamingMetrics()
        self.circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout
        )
        self._in_flight_requests: Set[str] = set()  # For deduplication
        self._request_cache: Dict[str, str] = {}  # Simple in-memory cache (fallback)
        self._cache_timestamps: Dict[str, float] = {}
        
        # Initialize new architecture components
        self._context_assembler = None
        self._response_validator = None
        self._semantic_cache = None
        
        if CONTEXT_ASSEMBLY_AVAILABLE:
            self._context_assembler = get_context_assembler()
            logger.info("✅ Context Assembly Layer enabled")
        
        if RESPONSE_VALIDATOR_AVAILABLE:
            self._response_validator = get_response_validator()
            logger.info("✅ Response Validator enabled")
        
        if SEMANTIC_CACHE_AVAILABLE:
            self._semantic_cache = get_semantic_cache()
            logger.info("✅ Semantic Cache enabled")
        self._cache_ttl = 300  # 5 minutes
        
        if self.enabled:
            logger.info("🚀 Streaming LLM Service initialized")
            logger.info(f"   URL: {self.api_url}")
            logger.info(f"   Retry: {self.config.max_retries}x, Circuit Breaker: {self.config.circuit_breaker_threshold} failures")
        else:
            logger.warning("⚠️ Streaming LLM Service disabled (no LLM_API_URL)")
    
    def _get_request_hash(self, prompt: str) -> str:
        """Generate a hash for request deduplication and caching."""
        return hashlib.md5(prompt.encode()).hexdigest()[:16]
    
    def _check_cache(
        self,
        prompt: str,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ) -> Optional[str]:
        """
        Check if we have a cached response.
        
        Uses semantic cache if available, falls back to exact MD5 match.
        """
        if not self.config.cache_enabled:
            return None
        
        # Try semantic cache first (uses embedding similarity)
        if self._semantic_cache and query:
            cached = self._semantic_cache.get(query, context, language)
            if cached:
                self.metrics.cache_hits += 1
                return cached
        
        # Fallback to exact MD5 match
        request_hash = self._get_request_hash(prompt)
        
        if request_hash in self._request_cache:
            timestamp = self._cache_timestamps.get(request_hash, 0)
            if time.time() - timestamp < self._cache_ttl:
                self.metrics.cache_hits += 1
                logger.info(f"✅ Cache HIT (MD5) for request {request_hash[:8]}")
                return self._request_cache[request_hash]
            else:
                # Expired, remove
                del self._request_cache[request_hash]
                del self._cache_timestamps[request_hash]
        
        self.metrics.cache_misses += 1
        return None
    
    def _cache_response(
        self,
        prompt: str,
        response: str,
        query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en"
    ):
        """Cache a response for future use."""
        if not self.config.cache_enabled or not response:
            return
        
        # Store in semantic cache
        if self._semantic_cache and query:
            self._semantic_cache.put(query, response, context, language)
        
        # Also store in MD5 cache (fallback)
        request_hash = self._get_request_hash(prompt)
        self._request_cache[request_hash] = response
        self._cache_timestamps[request_hash] = time.time()
        
        # Limit cache size (simple LRU-like cleanup)
        if len(self._request_cache) > 1000:
            # Remove oldest 100 entries
            sorted_keys = sorted(self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k])
            for key in sorted_keys[:100]:
                self._request_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
    
    async def stream_response(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        session_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response token by token with caching, retry, and circuit breaker.
        
        Args:
            prompt: The formatted prompt for the LLM
            on_token: Optional callback for each token
            session_id: Session ID for logging
            
        Yields:
            Individual tokens/chunks as they're generated
        """
        self.metrics.total_requests += 1
        start_time = time.time()
        request_hash = self._get_request_hash(prompt)
        
        if not self.enabled:
            logger.warning("Streaming disabled - yielding empty response")
            yield ""
            return
        
        # Check cache first
        cached_response = self._check_cache(prompt)
        if cached_response:
            # Stream cached response word by word
            words = cached_response.split()
            for i, word in enumerate(words):
                if i > 0:
                    yield " "
                yield word
                await asyncio.sleep(self.config.chunk_delay_ms / 1000)
            self.metrics.successful_streams += 1
            self.metrics.total_latency_ms += (time.time() - start_time) * 1000
            return
        
        # Check circuit breaker
        if not self.circuit_breaker.can_proceed():
            self.metrics.circuit_breaker_trips += 1
            logger.warning("🔴 Circuit breaker OPEN - using fallback")
            async for chunk in self._fallback_generate(prompt):
                yield chunk
            return
        
        # Check for duplicate in-flight request
        if request_hash in self._in_flight_requests:
            logger.info(f"⏳ Duplicate request {request_hash[:8]} - waiting...")
            # Wait for the other request to complete and cache
            for _ in range(50):  # Max 5 seconds wait
                await asyncio.sleep(0.1)
                cached = self._check_cache(prompt)
                if cached:
                    for word in cached.split():
                        yield word + " "
                    return
            # Timeout - proceed anyway
        
        # Mark request as in-flight
        self._in_flight_requests.add(request_hash)
        total_tokens = 0
        full_response = ""
        
        try:
            # Try streaming with retry logic
            last_error = None
            for attempt in range(self.config.max_retries + 1):
                try:
                    async for chunk in self._stream_from_api(prompt):
                        total_tokens += 1
                        full_response += chunk
                        if on_token:
                            on_token(chunk)
                        yield chunk
                        await asyncio.sleep(self.config.chunk_delay_ms / 1000)
                    
                    # Success
                    self.circuit_breaker.record_success()
                    self.metrics.successful_streams += 1
                    self.metrics.total_tokens_streamed += total_tokens
                    
                    # Cache the response
                    self._cache_response(prompt, full_response)
                    
                    elapsed = time.time() - start_time
                    self.metrics.total_latency_ms += elapsed * 1000
                    logger.info(f"✅ Streamed {total_tokens} tokens in {elapsed:.2f}s")
                    return
                    
                except Exception as e:
                    last_error = e
                    if attempt < self.config.max_retries:
                        self.metrics.retry_count += 1
                        delay = self.config.retry_delay_base * (2 ** attempt)
                        logger.warning(f"⚠️ Stream attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                    else:
                        raise
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.metrics.failed_streams += 1
            logger.error(f"❌ Streaming error after {self.config.max_retries + 1} attempts: {e}")
            
            # Fallback to non-streaming
            self.metrics.fallback_used += 1
            async for chunk in self._fallback_generate(prompt):
                yield chunk
                
        finally:
            self._in_flight_requests.discard(request_hash)
            elapsed = time.time() - start_time
            self.metrics.total_latency_ms += elapsed * 1000
    
    async def _stream_from_api(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream from LLM API - uses non-streaming endpoint and simulates streaming.
        
        RunPod serverless endpoints don't support true streaming, so we:
        1. Call the /generate endpoint
        2. Get the full response
        3. Yield it word-by-word to simulate streaming
        """
        # Use RunPodLLMClient for generation
        from services.runpod_llm_client import RunPodLLMClient
        
        logger.info("📝 Using RunPod /generate endpoint with simulated streaming")
        
        try:
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
            logger.error(f"Generation failed: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
    
    async def _stream_from_api_OLD_BROKEN(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        OLD BROKEN CODE - Kept for reference
        
        This tried to use streaming endpoints that don't exist on RunPod serverless.
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
        
        # All streaming endpoints failed - this code should never run now
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
        
        Includes:
        - Context assembly with grounding
        - Semantic caching
        - Streaming quality monitoring
        - Post-generation validation
        
        Args:
            message: User message
            context: Additional context (location, preferences, etc.)
            language: Response language
            
        Yields:
            Dict with 'type' and 'content' keys
        """
        # Check semantic cache first
        cached_response = self._check_cache(
            prompt="",  # Not needed for semantic cache
            query=message,
            context=context,
            language=language
        )
        
        if cached_response:
            # Stream cached response word by word for consistent UX
            yield {"type": "start", "content": "", "timestamp": time.time(), "cached": True}
            
            words = cached_response.split()
            for i, word in enumerate(words):
                if i > 0:
                    yield {"type": "token", "content": " ", "timestamp": time.time()}
                yield {"type": "token", "content": word, "timestamp": time.time()}
                await asyncio.sleep(self.config.chunk_delay_ms / 1000)
            
            yield {"type": "complete", "content": cached_response, "timestamp": time.time(), "cached": True}
            return
        
        # Assemble context using the Context Assembly Layer
        assembled_context = None
        grounding_contract = None
        
        if self._context_assembler and context:
            try:
                assembled_context = self._context_assembler.quick_assemble(
                    rag_context=context.get('rag_context'),
                    database_context=context.get('database_context'),
                    service_data=context.get('service_data'),
                    query_intent=context.get('intent'),
                    signals=context.get('signals', {})
                )
                grounding_contract = {
                    'primary_context': assembled_context.primary_context[:200] if assembled_context.primary_context else "",
                    'grounding_instructions': assembled_context.grounding_instructions,
                    'warnings': assembled_context.warnings
                }
                logger.info(f"📦 Context assembled: {assembled_context.total_tokens_estimate} tokens, {len(assembled_context.warnings)} warnings")
            except Exception as e:
                logger.warning(f"Context assembly failed: {e}")
        
        # Build the prompt (with or without assembled context)
        prompt = self._build_chat_prompt(message, context, language, assembled_context)
        
        # Yield start event
        yield {
            "type": "start",
            "content": "",
            "timestamp": time.time(),
            "grounding": grounding_contract
        }
        
        # Stream tokens with quality monitoring
        full_response = ""
        token_count = 0
        abort_streaming = False
        abort_reason = None
        
        async for token in self.stream_response(prompt):
            full_response += token
            token_count += 1
            
            # Quality monitoring: Check for template responses early
            if self._response_validator and token_count == 20:  # Check after ~20 tokens
                should_continue, reason = self._response_validator.check_streaming_quality(full_response)
                if not should_continue:
                    abort_streaming = True
                    abort_reason = reason
                    logger.warning(f"⚠️ Streaming quality issue: {reason}")
                    yield {
                        "type": "warning",
                        "content": "Regenerating response...",
                        "reason": reason,
                        "timestamp": time.time()
                    }
                    break
            
            yield {
                "type": "token",
                "content": token,
                "timestamp": time.time()
            }
        
        # If aborted due to quality issues, try regeneration with adjusted prompt
        if abort_streaming:
            logger.info("🔄 Attempting response regeneration due to quality issue")
            # Add anti-template instruction and regenerate
            adjusted_prompt = prompt + "\n\n⚠️ DO NOT start with phrases like 'I'd be happy to help' or 'As an AI'. Answer the question directly."
            full_response = ""
            
            async for token in self.stream_response(adjusted_prompt):
                full_response += token
                yield {
                    "type": "token",
                    "content": token,
                    "timestamp": time.time()
                }
        
        # Post-generation validation
        validation_result = None
        
        # Clean any prompt leakage from the response
        full_response = clean_streaming_response(full_response)
        
        if self._response_validator and full_response:
            try:
                context_sources = []
                if context and context.get('rag_context'):
                    context_sources.append({
                        'source': 'rag',
                        'content': context['rag_context'],
                        'confidence': 0.75
                    })
                
                validation_result = self._response_validator.validate(
                    response=full_response,
                    context_sources=context_sources,
                    expected_language=language,
                    route_data=context.get('route_data') if context else None,
                    grounding_contract=grounding_contract
                )
                
                if not validation_result.is_valid:
                    logger.warning(f"⚠️ Response validation issues: {len(validation_result.issues)} issues")
                    for issue in validation_result.issues:
                        logger.debug(f"  - {issue['type'].value}: {issue['description']}")
            except Exception as e:
                logger.error(f"Response validation failed: {e}")
        
        # Cache the response (if valid)
        if full_response and (validation_result is None or validation_result.confidence_score > 0.5):
            self._cache_response(
                prompt=prompt,
                response=full_response,
                query=message,
                context=context,
                language=language
            )
        
        # Yield completion event with validation metadata
        yield {
            "type": "complete",
            "content": full_response,
            "timestamp": time.time(),
            "validation": {
                "is_valid": validation_result.is_valid if validation_result else True,
                "confidence": validation_result.confidence_score if validation_result else 1.0,
                "issues_count": len(validation_result.issues) if validation_result else 0,
                "suggested_action": validation_result.suggested_action if validation_result else "none"
            } if validation_result else None
        }
    
    def _build_chat_prompt(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        language: str = "en",
        assembled_context: Optional[Any] = None  # AssembledContext from context_assembly
    ) -> str:
        """
        Build a formatted prompt using the main prompt builder from prompts.py.
        
        If assembled_context is provided (from Context Assembly Layer), it uses
        the curated, ranked, and grounded context instead of raw concatenation.
        """
        from services.llm.prompts import PromptBuilder
        prompt_builder = PromptBuilder()
        
        # Build context dict for prompt builder
        prompt_context = {
            'database': '',
            'rag': '',
            'services': {},
        }
        
        # Use assembled context if available (preferred)
        if assembled_context:
            # Use the curated primary context from Context Assembly Layer
            prompt_context['database'] = assembled_context.primary_context
            prompt_context['rag'] = assembled_context.supporting_context
            
            # Add grounding instructions to services
            if assembled_context.grounding_instructions:
                prompt_context['services']['grounding'] = assembled_context.grounding_instructions
            
            logger.info(f"📦 Using assembled context: {assembled_context.total_tokens_estimate} tokens")
        else:
            # Fallback to raw context (backwards compatibility)
            prompt_context['database'] = context.get('rag_context', '') if context else ''
            prompt_context['rag'] = context.get('rag_context', '') if context else ''
        
        # Add map data if available
        if context and context.get('map_data'):
            prompt_context['map_data'] = context['map_data']
        
        # Add route data if available (for hybrid architecture)
        if context and context.get('route_data'):
            prompt_context['route_data'] = context['route_data']
        
        # Get user location
        user_location = None
        if context and context.get('location'):
            loc = context['location']
            user_location = {
                'lat': loc.get('lat') or loc.get('latitude'),
                'lon': loc.get('lon') or loc.get('longitude')
            }
        
        # Get conversation history
        conversation_context = None
        if context and context.get('conversation_history'):
            conversation_context = {'messages': context['conversation_history']}
        
        # Build signals from intent
        signals = {}
        if context and context.get('intent'):
            intent = context['intent']
            if intent in ['transportation', 'directions', 'route', 'navigate']:
                signals['needs_transportation'] = True
                signals['needs_directions'] = True
            elif intent in ['restaurant', 'food', 'dining']:
                signals['needs_restaurant'] = True
            elif intent in ['attraction', 'museum', 'place', 'sightseeing']:
                signals['needs_attraction'] = True
            elif intent == 'weather':
                signals['needs_weather'] = True
        
        # Add trip planning signal if present in context
        if context and context.get('needs_trip_planning'):
            signals['needs_trip_planning'] = True
        
        # Build prompt using the SINGLE source of truth
        prompt = prompt_builder.build_prompt(
            query=message,
            signals=signals,
            context=prompt_context,
            conversation_context=conversation_context,
            language=language,
            user_location=user_location
        )
        
        logger.info(f"✅ Built prompt (lang={language}, has_rag={bool(context and context.get('rag_context'))}, assembled={assembled_context is not None})")
        return prompt
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming service metrics including new components."""
        metrics = {
            'service': 'streaming_llm',
            'enabled': self.enabled,
            'circuit_breaker_state': self.circuit_breaker.state,
            'cache_size': len(self._request_cache),
            'in_flight_requests': len(self._in_flight_requests),
            'components': {
                'context_assembly': CONTEXT_ASSEMBLY_AVAILABLE,
                'response_validator': RESPONSE_VALIDATOR_AVAILABLE,
                'semantic_cache': SEMANTIC_CACHE_AVAILABLE,
            },
            **self.metrics.get_stats()
        }
        
        # Add semantic cache stats if available
        if self._semantic_cache:
            metrics['semantic_cache_stats'] = self._semantic_cache.get_stats()
        
        return metrics
    
    def clear_cache(self):
        """Clear all caches (MD5 and semantic)."""
        self._request_cache.clear()
        self._cache_timestamps.clear()
        
        if self._semantic_cache:
            self._semantic_cache.clear()
        
        logger.info("🗑️ All caches cleared")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check streaming service health."""
        return {
            'status': 'healthy' if self.enabled else 'disabled',
            'api_url': self.api_url[:50] + '...' if self.api_url and len(self.api_url) > 50 else self.api_url,
            'circuit_breaker': self.circuit_breaker.state,
            'components': {
                'context_assembly': CONTEXT_ASSEMBLY_AVAILABLE,
                'response_validator': RESPONSE_VALIDATOR_AVAILABLE,
                'semantic_cache': SEMANTIC_CACHE_AVAILABLE,
            },
            'metrics': self.metrics.get_stats()
        }


# Singleton instance
_streaming_service: Optional[StreamingLLMService] = None


def get_streaming_llm_service() -> StreamingLLMService:
    """Get or create the streaming LLM service singleton."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingLLMService()
    return _streaming_service


def get_streaming_metrics() -> Dict[str, Any]:
    """Get metrics from the streaming service."""
    service = get_streaming_llm_service()
    return service.get_metrics()
