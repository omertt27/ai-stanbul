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
        self._initialization_errors = []
    
    async def initialize(self):
        """Fast initialization - minimal blocking"""
        logger.info("🚀 Starting ULTRA-FAST initialization...")
        
        try:
            # 1. Database (critical) - but don't wait too long
            asyncio.create_task(self._initialize_database())
            
            # 2. Redis Cache (non-critical) - initialize in background
            asyncio.create_task(self._initialize_redis())
            
            # 3. Service Manager (non-critical) - initialize in background
            asyncio.create_task(self._initialize_service_manager())
            
            logger.info("✅ App ready to serve traffic - components initializing in background!")
            
            # 4. Schedule lazy initialization of heavy components
            asyncio.create_task(self._lazy_initialize_llm())
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            # Don't raise - allow app to start even if background tasks fail
    
    async def _initialize_database(self):
        """Initialize database connection and create tables"""
        try:
            from database import get_db, engine, Base
            
            # Import all models to register them with Base metadata
            # This must be done BEFORE create_all() is called
            try:
                import models  # This registers all model classes with Base
                logger.info("✅ Models imported and registered")
            except Exception as e:
                logger.warning(f"⚠️ Could not import models: {e}")
            
            # Create all tables if they don't exist
            logger.info("🔄 Creating database tables if needed...")
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables ready")
            
            # Get database session using SessionLocal directly instead of generator
            from database import SessionLocal
            self.db = SessionLocal()
            logger.info("✅ Database connection established")
        except Exception as e:
            error_msg = f"Database initialization failed: {str(e)}"
            logger.warning(f"⚠️ {error_msg}")
            self._initialization_errors.append(error_msg)
            self.db = None
    
    async def _initialize_redis(self):
        """Initialize Redis cache with reasonable timeout for AWS MemoryDB via EC2 proxy"""
        try:
            from services.redis_cache import init_cache
            # 5 second timeout for AWS MemoryDB through EC2 proxy (local dev)
            # 1.5s timeout was too short for cross-region AWS connections
            await asyncio.wait_for(init_cache(), timeout=5.0)
            self.redis_cache = True
            logger.info("✅ Redis cache initialized successfully")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Redis cache timeout (5.0s) - likely firewall/network blocked")
            logger.warning("⚠️ Continuing WITHOUT Redis - sessions will not persist across restarts")
            self.redis_cache = None
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}")
            logger.warning("⚠️ Continuing WITHOUT Redis - sessions will not persist across restarts")
            self.redis_cache = None
    
    async def _initialize_service_manager(self):
        """Initialize local service manager"""
        try:
            from services.service_manager import service_manager
            # ServiceManager uses initialize_all(), not initialize()
            service_manager.initialize_all()
            self.service_manager = service_manager
            self._services_initialized = True
            logger.info("✅ Service Manager initialized")
        except Exception as e:
            error_msg = f"Service Manager initialization failed: {str(e)}"
            logger.warning(f"⚠️ {error_msg}")
            self._initialization_errors.append(error_msg)
            self.service_manager = None
    
    async def _lazy_initialize_llm(self):
        """Lazy initialization of Pure LLM Core (runs in background)"""
        try:
            logger.info("🔄 Starting background LLM initialization...")
            
            # 1. Preload ML models FIRST (before LLM, to share memory efficiently)
            try:
                from services.model_cache import preload_models
                preload_models(['default', 'multilingual'])
                logger.info("✅ ML models preloaded")
            except Exception as e:
                logger.warning(f"⚠️ Model preload skipped: {e}")
            
            # 2. Wait for database and service manager to be ready (up to 30 seconds)
            max_wait = 30
            waited = 0
            while waited < max_wait:
                if self.db is not None and self._services_initialized:
                    logger.info(f"✅ Dependencies ready after {waited}s")
                    break
                await asyncio.sleep(2)
                waited += 2
                logger.info(f"⏳ Waiting for dependencies... ({waited}s)")
            
            if self.db is None:
                logger.error("❌ Database not initialized - cannot start LLM")
                return
            
            if not self._services_initialized:
                logger.warning("⚠️ Service Manager not ready - LLM may have limited functionality")
            
            logger.info(f"🔍 DB initialized: {self.db is not None}, Services initialized: {self._services_initialized}")
            
            from services.llm import PureLLMCore
            
            # Get LLM client (RunPod/Llama)
            from services.runpod_llm_client import get_llm_client
            llm_client = get_llm_client()
            
            if not llm_client or not llm_client.enabled:
                raise Exception("LLM client not enabled - check LLM_API_URL environment variable")
            
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
            error_msg = f"Failed to initialize Pure LLM: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self._initialization_errors.append(error_msg)
            self.pure_llm_core = None
    
    async def shutdown(self):
        """Graceful shutdown"""
        try:
            # Close database session
            if self.db:
                self.db.close()
                logger.info("✅ Database session closed")
            
            # Shutdown Redis cache
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
