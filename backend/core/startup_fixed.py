"""
Fixed Startup Manager - Fast and Resilient

This version focuses on:
1. Fast startup for Cloud Run (< 30 seconds)
2. Lazy initialization of heavy components
3. Graceful degradation if components fail
4. No blocking operations during startup

Author: AI Istanbul Team
Date: December 2025
"""

import logging
import asyncio
from typing import Optional
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class FastStartupManager:
    """Fast startup manager for Cloud Run with lazy initialization"""
    
    def __init__(self):
        self.db: Optional[Session] = None
        self.redis_cache = None
        self.service_manager = None
        self.pure_llm_core = None
        self.recommendation_engine = None
        self._llm_initialized = False
        self._services_initialized = False
    
    async def initialize(self):
        """Fast initialization - only critical components"""
        logger.info("🚀 Starting FAST initialization...")
        
        try:
            # 1. Database (critical)
            await self._initialize_database()
            
            # 2. Redis Cache (critical)
            await self._initialize_redis()
            
            # 3. Service Manager (critical)
            await self._initialize_service_manager()
            
            logger.info("✅ Critical components initialized - ready to serve traffic!")
            
            # 4. Schedule lazy initialization of heavy components
            asyncio.create_task(self._lazy_initialize_llm())
            
        except Exception as e:
            logger.error(f"❌ Critical initialization failed: {e}")
            raise
    
    async def _initialize_database(self):
        """Initialize database connection"""
        try:
            from database import get_db
            self.db = next(get_db())
            logger.info("✅ Database connection established")
        except Exception as e:
            logger.warning(f"⚠️ Database initialization failed: {e}")
            self.db = None
    
    async def _initialize_redis(self):
        """Initialize Redis cache"""
        try:
            from services.redis_cache import init_cache
            await init_cache()
            self.redis_cache = True
            logger.info("✅ Redis cache initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis cache initialization failed: {e}")
            self.redis_cache = None
    
    async def _initialize_service_manager(self):
        """Initialize local service manager"""
        try:
            from services.service_manager import service_manager
            await service_manager.initialize(db=self.db)
            self.service_manager = service_manager
            self._services_initialized = True
            logger.info("✅ Service Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Service Manager initialization failed: {e}")
            self.service_manager = None
    
    async def _lazy_initialize_llm(self):
        """Lazy initialization of Pure LLM Core (runs in background)"""
        try:
            logger.info("🔄 Starting background LLM initialization...")
            
            from services.llm import PureLLMCore
            
            # Get LLM client
            try:
                from services.runpod_llm_client import get_llm_client
                llm_client = get_llm_client()
            except Exception as e:
                logger.warning(f"⚠️ Using default OpenAI client: {e}")
                from openai import AsyncOpenAI
                from config.settings import settings
                llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Create Pure LLM Core config
            config = {
                'use_circuit_breaker': True,
                'circuit_breaker': {
                    'failure_threshold': 5,
                    'recovery_timeout': 60,
                    'expected_exception': Exception
                },
                'retry_config': {
                    'max_retries': 3,
                    'backoff_factor': 2.0,
                    'max_delay': 30.0
                },
                'timeout_config': {
                    'intent_classification': 5.0,
                    'signal_detection': 1.0,
                    'context_building': 5.0,
                    'llm_generation': 15.0,
                    'cache_storage': 1.0
                }
            }
            
            # Create Pure LLM Core
            self.pure_llm_core = PureLLMCore(
                llm_client=llm_client,
                db_connection=self.db,
                config=config,
                services=self.service_manager
            )
            
            self._llm_initialized = True
            logger.info("✅ Pure LLM Core initialized (background)")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pure LLM: {e}")
            self.pure_llm_core = None
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            if self.redis_cache:
                from services.redis_cache import shutdown_cache
                await shutdown_cache()
                logger.info("✅ Redis cache shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_pure_llm_core(self):
        """Get Pure LLM Core instance (may be None if still initializing)"""
        if not self._llm_initialized:
            logger.warning("⚠️ Pure LLM Core still initializing...")
        return self.pure_llm_core
    
    def get_recommendation_engine(self):
        """Get recommendation engine instance"""
        return self.recommendation_engine
    
    def is_llm_ready(self) -> bool:
        """Check if LLM is ready"""
        return self._llm_initialized and self.pure_llm_core is not None
    
    def is_services_ready(self) -> bool:
        """Check if services are ready"""
        return self._services_initialized and self.service_manager is not None


# Create singleton instance
fast_startup_manager = FastStartupManager()
