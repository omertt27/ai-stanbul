"""
Infrastructure Configuration
Central configuration for all infrastructure components
"""

import os
from typing import Optional
import redis
import logging

# Import all infrastructure components
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from utils.ttl_cache import TTLCache
from config.feature_manager import FeatureManager
from utils.rate_limiter import RateLimiter
from utils.response_cache import SmartResponseCache
from monitoring.system_monitor import SystemMonitor
from utils.graceful_degradation import GracefulDegradation
from utils.conversation_summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class InfrastructureConfig:
    """Central configuration and initialization for all infrastructure components"""
    
    def __init__(self):
        """Initialize infrastructure configuration"""
        self.redis_client = None
        self.feature_manager = None
        self.rate_limiter = None
        self.response_cache = None
        self.system_monitor = None
        self.graceful_degradation = None
        self.conversation_summarizer = None
        self.session_cache = None
        
        logger.info("🚀 Initializing Infrastructure Configuration...")
    
    def init_redis(self) -> Optional[redis.Redis]:
        """
        Initialize Redis client with fallback
        
        Returns:
            Redis client or None if unavailable
        """
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()  # Test connection
            
            logger.info(f"✅ Redis connected: {redis_url}")
            return client
        except Exception as e:
            logger.warning(f"⚠️ Redis not available: {e}")
            return None
    
    def init_feature_manager(self) -> FeatureManager:
        """
        Initialize feature manager
        
        Returns:
            FeatureManager instance
        """
        manager = FeatureManager()
        
        # FeatureManager is mainly for module imports, which we'll configure separately
        # Here we just initialize it for future use
        
        logger.info("✅ FeatureManager initialized")
        return manager
    
    def init_rate_limiter(self) -> RateLimiter:
        """
        Initialize rate limiter with service limits
        
        Returns:
            RateLimiter instance
        """
        limiter = RateLimiter()
        
        # Configure service-specific limits
        limiter.set_service_limit("openai", requests_per_minute=60)
        limiter.set_service_limit("transport_api", requests_per_minute=100)
        limiter.set_service_limit("weather_api", requests_per_minute=30)
        limiter.set_service_limit("restaurant_api", requests_per_minute=50)
        
        # Configure user limits
        limiter.set_user_limit(requests_per_minute=20)
        
        logger.info("✅ RateLimiter initialized with limits")
        return limiter
    
    def init_response_cache(self, redis_client: Optional[redis.Redis]) -> SmartResponseCache:
        """
        Initialize response cache
        
        Args:
            redis_client: Redis client (optional)
            
        Returns:
            SmartResponseCache instance
        """
        cache = SmartResponseCache(
            redis_client=redis_client,
            max_local_size=1000
        )
        
        logger.info("✅ SmartResponseCache initialized")
        return cache
    
    def init_system_monitor(self) -> SystemMonitor:
        """
        Initialize system monitor
        
        Returns:
            SystemMonitor instance
        """
        monitor = SystemMonitor(window_size=100)
        
        logger.info("✅ SystemMonitor initialized")
        return monitor
    
    def init_graceful_degradation(self) -> GracefulDegradation:
        """
        Initialize graceful degradation
        
        Returns:
            GracefulDegradation instance
        """
        degradation = GracefulDegradation()
        
        # Configure circuit breaker
        degradation.configure_circuit_breaker(
            failure_threshold=5,
            recovery_timeout=300,  # 5 minutes
            half_open_attempts=3
        )
        
        logger.info("✅ GracefulDegradation initialized")
        return degradation
    
    def init_conversation_summarizer(self) -> ConversationSummarizer:
        """
        Initialize conversation summarizer
        
        Returns:
            ConversationSummarizer instance
        """
        summarizer = ConversationSummarizer(
            max_tokens=4000,
            summary_ratio=0.3,
            min_messages=2
        )
        
        logger.info("✅ ConversationSummarizer initialized")
        return summarizer
    
    def init_session_cache(self) -> TTLCache:
        """
        Initialize session cache for user contexts
        
        Returns:
            TTLCache instance
        """
        cache = TTLCache(max_size=1000, default_ttl=1800)  # 30 minutes
        
        logger.info("✅ Session TTLCache initialized")
        return cache
    
    def initialize_all(self) -> dict:
        """
        Initialize all infrastructure components
        
        Returns:
            Dictionary with all initialized components
        """
        logger.info("🏗️ Initializing all infrastructure components...")
        
        # Initialize in order
        self.redis_client = self.init_redis()
        self.feature_manager = self.init_feature_manager()
        self.rate_limiter = self.init_rate_limiter()
        self.response_cache = self.init_response_cache(self.redis_client)
        self.system_monitor = self.init_system_monitor()
        self.graceful_degradation = self.init_graceful_degradation()
        self.conversation_summarizer = self.init_conversation_summarizer()
        self.session_cache = self.init_session_cache()
        
        components = {
            "redis": self.redis_client,
            "feature_manager": self.feature_manager,
            "rate_limiter": self.rate_limiter,
            "response_cache": self.response_cache,
            "system_monitor": self.system_monitor,
            "graceful_degradation": self.graceful_degradation,
            "conversation_summarizer": self.conversation_summarizer,
            "session_cache": self.session_cache
        }
        
        logger.info("✅ All infrastructure components initialized successfully!")
        return components
    
    # Feature initialization helpers
    def _init_openai_client(self):
        """Initialize OpenAI client"""
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        logger.info("✅ OpenAI client initialized")
        return openai
    
    def _init_transport_api(self):
        """Initialize transport API"""
        # Placeholder for transport API initialization
        logger.info("✅ Transport API initialized")
        return {"type": "transport_api"}
    
    def _init_restaurant_system(self):
        """Initialize restaurant system"""
        # Placeholder for restaurant system initialization
        logger.info("✅ Restaurant system initialized")
        return {"type": "restaurant_system"}
    
    def _init_weather_api(self):
        """Initialize weather API"""
        # Placeholder for weather API initialization
        logger.info("✅ Weather API initialized")
        return {"type": "weather_api"}


# Global infrastructure instance
_infrastructure = None


def get_infrastructure() -> InfrastructureConfig:
    """
    Get or create global infrastructure instance
    
    Returns:
        InfrastructureConfig instance
    """
    global _infrastructure
    if _infrastructure is None:
        _infrastructure = InfrastructureConfig()
    return _infrastructure


def init_infrastructure() -> dict:
    """
    Initialize global infrastructure
    
    Returns:
        Dictionary with all components
    """
    infrastructure = get_infrastructure()
    return infrastructure.initialize_all()


# Configuration constants
CONFIG = {
    # Cache TTLs by intent type (seconds)
    "CACHE_TTL": {
        "restaurant": 3600,      # 1 hour
        "museum": 86400,         # 24 hours
        "event": 1800,           # 30 minutes
        "transportation": 900,   # 15 minutes
        "weather": 600,          # 10 minutes
        "general": 7200,         # 2 hours
        "location": 3600,        # 1 hour
        "attraction": 43200,     # 12 hours
    },
    
    # Rate limits (requests per minute)
    "RATE_LIMITS": {
        "openai": 60,
        "transport_api": 100,
        "weather_api": 30,
        "restaurant_api": 50,
        "user_default": 20
    },
    
    # Circuit breaker settings
    "CIRCUIT_BREAKER": {
        "failure_threshold": 5,
        "recovery_timeout": 300,
        "half_open_attempts": 3
    },
    
    # Conversation settings
    "CONVERSATION": {
        "max_tokens": 4000,
        "summary_ratio": 0.3,
        "min_messages": 2
    },
    
    # Session settings
    "SESSION": {
        "cache_size": 1000,
        "ttl": 1800  # 30 minutes
    }
}


# Self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Testing InfrastructureConfig...")
    
    config = InfrastructureConfig()
    
    # Test individual initializations
    feature_manager = config.init_feature_manager()
    assert feature_manager is not None, "Should initialize FeatureManager"
    print("✅ FeatureManager initialization test passed")
    
    rate_limiter = config.init_rate_limiter()
    assert rate_limiter is not None, "Should initialize RateLimiter"
    print("✅ RateLimiter initialization test passed")
    
    response_cache = config.init_response_cache(None)
    assert response_cache is not None, "Should initialize SmartResponseCache"
    print("✅ SmartResponseCache initialization test passed")
    
    system_monitor = config.init_system_monitor()
    assert system_monitor is not None, "Should initialize SystemMonitor"
    print("✅ SystemMonitor initialization test passed")
    
    graceful_degradation = config.init_graceful_degradation()
    assert graceful_degradation is not None, "Should initialize GracefulDegradation"
    print("✅ GracefulDegradation initialization test passed")
    
    conversation_summarizer = config.init_conversation_summarizer()
    assert conversation_summarizer is not None, "Should initialize ConversationSummarizer"
    print("✅ ConversationSummarizer initialization test passed")
    
    session_cache = config.init_session_cache()
    assert session_cache is not None, "Should initialize Session TTLCache"
    print("✅ Session cache initialization test passed")
    
    # Test full initialization
    components = config.initialize_all()
    assert len(components) == 8, "Should have 8 components"
    print("✅ Full initialization test passed")
    print(f"📊 Initialized components: {list(components.keys())}")
    
    # Test global instance
    infra = get_infrastructure()
    assert infra is not None, "Should get global instance"
    print("✅ Global instance test passed")
    
    print("\n✅ All InfrastructureConfig tests passed!")
