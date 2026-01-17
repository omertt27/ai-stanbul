"""
LLM Configuration Manager
Central configuration for ALL LLM requests in the AI Istanbul system

This module ensures that ALL prompts, handlers, and ML components use the
Google Cloud deployed Llama 3.1 8B model via the centralized client.

Key Features:
- ✅ Single source of truth for LLM configuration
- ✅ Environment-based configuration (development/production)
- ✅ Automatic fallback handling
- ✅ Centralized timeout and retry configuration
- ✅ Easy integration for all system components

Usage:
    from llm_config import get_configured_llm
    
    llm = get_configured_llm()
    response = llm.generate("What's the weather in Istanbul?")
"""

import os
import logging
from typing import Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class LLMMode(Enum):
    """LLM deployment modes"""
    RUNPOD = "runpod"              # Production: RunPod vLLM with Llama 3.1 8B
    GOOGLE_CLOUD = "google_cloud"  # Legacy: Google Cloud VM with Llama 3.1 8B
    LOCAL = "local"                # Development: Local TinyLlama or Llama models
    MOCK = "mock"                  # Testing: Mock responses


class LLMConfig:
    """
    Central LLM configuration
    
    This class manages all LLM settings and ensures consistent behavior
    across the entire AI Istanbul system.
    """
    
    # Default configuration
    DEFAULT_MODE = LLMMode.RUNPOD  # Use RunPod as default
    DEFAULT_ENDPOINT = "http://35.210.251.24:8000"  # Legacy Google Cloud (fallback)
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_TOKENS = 200
    DEFAULT_TEMPERATURE = 0.7
    
    # Environment variable names
    ENV_MODE = "AI_ISTANBUL_LLM_MODE"
    ENV_ENDPOINT = "GOOGLE_CLOUD_LLM_ENDPOINT"
    ENV_TIMEOUT = "LLM_TIMEOUT"
    
    @classmethod
    def get_mode(cls) -> LLMMode:
        """Get current LLM mode from environment or default"""
        mode_str = os.getenv(cls.ENV_MODE, cls.DEFAULT_MODE.value).lower()
        
        try:
            return LLMMode(mode_str)
        except ValueError:
            logger.warning(f"⚠️ Invalid LLM mode: {mode_str}, using default")
            return cls.DEFAULT_MODE
    
    @classmethod
    def get_endpoint(cls) -> str:
        """Get Google Cloud LLM endpoint"""
        return os.getenv(cls.ENV_ENDPOINT, cls.DEFAULT_ENDPOINT)
    
    @classmethod
    def get_timeout(cls) -> int:
        """Get request timeout"""
        try:
            return int(os.getenv(cls.ENV_TIMEOUT, str(cls.DEFAULT_TIMEOUT)))
        except ValueError:
            return cls.DEFAULT_TIMEOUT
    
    @classmethod
    def get_defaults(cls) -> dict:
        """Get default generation parameters"""
        return {
            'max_tokens': cls.DEFAULT_MAX_TOKENS,
            'temperature': cls.DEFAULT_TEMPERATURE
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        mode = cls.get_mode()
        print("\n" + "=" * 60)
        print("🔧 AI Istanbul LLM Configuration")
        print("=" * 60)
        print(f"Mode:           {mode.value}")
        print(f"Endpoint:       {cls.get_endpoint()}")
        print(f"Timeout:        {cls.get_timeout()}s")
        print(f"Max Tokens:     {cls.DEFAULT_MAX_TOKENS}")
        print(f"Temperature:    {cls.DEFAULT_TEMPERATURE}")
        print("=" * 60 + "\n")


def get_configured_llm():
    """
    Get configured LLM instance based on current mode
    
    This is the MAIN entry point for all LLM requests in the system.
    All handlers, prompts, and ML components should use this function.
    
    Returns:
        Configured LLM client instance
    
    Examples:
        # In any module:
        from llm_config import get_configured_llm
        
        llm = get_configured_llm()
        response = llm.generate("Tell me about Istanbul", max_tokens=150)
    """
    mode = LLMConfig.get_mode()
    
    if mode == LLMMode.GOOGLE_CLOUD:
        # Production mode: Use Google Cloud Llama 3.1 8B
        logger.info("🚀 Using Google Cloud LLM (Llama 3.1 8B)")
        
        try:
            from google_cloud_llm_client import get_llm_client
            return get_llm_client(endpoint=LLMConfig.get_endpoint())
        except ImportError as e:
            logger.error(f"❌ Failed to import Google Cloud LLM client: {e}")
            logger.error("⚠️ Falling back to local mode")
            # Fallback to local
            mode = LLMMode.LOCAL
    
    if mode == LLMMode.LOCAL:
        # Development mode: Use local LLM wrapper
        logger.info("🔧 Using Local LLM (TinyLlama or Llama 3.1 8B)")
        
        try:
            from ml_systems.llm_service_wrapper import LLMServiceWrapper
            return LLMServiceWrapper()
        except ImportError as e:
            logger.error(f"❌ Failed to import LLM service wrapper: {e}")
            logger.error("⚠️ Falling back to mock mode")
            # Fallback to mock
            mode = LLMMode.MOCK
    
    if mode == LLMMode.MOCK:
        # Testing mode: Use mock LLM
        logger.warning("⚠️ Using Mock LLM (for testing only)")
        return MockLLM()
    
    # Should never reach here
    logger.error("❌ Failed to initialize any LLM mode")
    return MockLLM()


class MockLLM:
    """
    Mock LLM for testing
    
    Returns placeholder responses when no real LLM is available.
    """
    
    def __init__(self):
        logger.warning("⚠️ Mock LLM initialized - responses will be placeholders")
    
    def generate(self, prompt: str, max_tokens: int = 150, **kwargs) -> str:
        """Generate mock response"""
        return f"[Mock response to: {prompt[:50]}...]"
    
    def chat(self, prompt: str, max_tokens: int = 200, **kwargs) -> str:
        """Generate mock chat response"""
        return f"[Mock chat response to: {prompt[:50]}...]"
    
    def health_check(self) -> dict:
        """Mock health check"""
        return {
            "status": "mock",
            "model_loaded": False,
            "device": "none"
        }


# Convenience functions for common use cases

def generate_text(prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
    """
    Quick text generation using configured LLM
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    llm = get_configured_llm()
    return llm.generate(prompt, max_tokens=max_tokens, temperature=temperature)


def generate_chat_response(prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
    """
    Quick chat response using configured LLM
    
    Args:
        prompt: Chat prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Chat response
    """
    llm = get_configured_llm()
    return llm.chat(prompt, max_tokens=max_tokens, temperature=temperature)


def check_llm_health() -> dict:
    """
    Check health of configured LLM
    
    Returns:
        Health status dictionary
    """
    llm = get_configured_llm()
    if hasattr(llm, 'health_check'):
        return llm.health_check()
    else:
        return {
            "status": "unknown",
            "message": "Health check not available for this LLM type"
        }


# Module-level initialization
def _initialize():
    """Initialize LLM configuration on module import"""
    logger.info("🔧 Initializing AI Istanbul LLM configuration")
    
    mode = LLMConfig.get_mode()
    logger.info(f"📍 LLM Mode: {mode.value}")
    
    if mode == LLMMode.GOOGLE_CLOUD:
        endpoint = LLMConfig.get_endpoint()
        logger.info(f"🌐 Google Cloud Endpoint: {endpoint}")
        logger.info(f"⏱️  Timeout: {LLMConfig.get_timeout()}s")
    
    logger.info("✅ LLM configuration initialized")


# Run initialization
_initialize()


if __name__ == "__main__":
    # Print configuration and test
    LLMConfig.print_config()
    
    print("🧪 Testing LLM Configuration")
    print("=" * 60)
    
    # Get configured LLM
    llm = get_configured_llm()
    print(f"✅ LLM instance: {type(llm).__name__}")
    
    # Test health check
    health = check_llm_health()
    print(f"📊 Health: {health}")
    
    # Test generation
    print("\n🔤 Testing text generation:")
    response = generate_text("Istanbul is", max_tokens=50)
    print(f"Response: {response[:100]}...")
    
    # Test chat
    print("\n💬 Testing chat:")
    response = generate_chat_response("What's the best time to visit Istanbul?", max_tokens=100)
    print(f"Response: {response[:150]}...")
    
    print("\n" + "=" * 60)
    print("✅ Configuration test complete!")
