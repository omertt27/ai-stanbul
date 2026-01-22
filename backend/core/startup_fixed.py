"""
Fixed Startup Manager - Fast and Resilient

This version focuses on:
1. Fast startup for Cloud Run (< 30 seconds)
2. Lazy initialization of heavy components
3. Graceful degradation if components fail
4. No blocking operations during startup
5. SINGLE initialization via startup_guard (no duplicates)

Author: AI Istanbul Team
Date: December 2025
"""

import logging
import asyncio
from typing import Optional
from sqlalchemy.orm import Session

# Import startup guard to prevent duplicate initialization
from core.startup_guard import ensure_single_init, is_initialized, is_redis_ready

logger = logging.getLogger(__name__)


class FastStartupManager:
    """Fast startup manager for Cloud Run with lazy initialization"""
    
    def __init__(self):
        self.db: Optional[Session] = None
        self.redis_cache = None
        self.service_manager = None
        self.pure_llm_core = None
        self.recommendation_engine = None
        # Multi-route system components
        self.route_optimizer = None
        self.transportation_route_integration = None
        self.enhanced_map_visualization = None
        self._llm_initialized = False
        self._services_initialized = False
        self._multi_route_initialized = False
        self._initialization_errors = []
    
    async def initialize(self):
        """Fast initialization - minimal blocking"""
        # Guard against duplicate initialization
        if not ensure_single_init("startup_manager"):
            logger.info("� StartupManager already initialized, skipping")
            return
        
        logger.info("�🚀 Starting ULTRA-FAST initialization...")
        
        try:
            # 1. Database (critical) - but don't wait too long
            asyncio.create_task(self._initialize_database())
            
            # 2. Redis Cache - use centralized check from startup_guard
            if is_redis_ready():
                asyncio.create_task(self._initialize_redis())
            else:
                logger.info("⏭️ Skipping Redis init - not available (checked at startup)")
            
            # 3. Service Manager (non-critical) - initialize in background
            asyncio.create_task(self._initialize_service_manager())
            
            # 4. Multi-route system (enhanced map visualization)
            asyncio.create_task(self._initialize_multi_route_system())
            
            logger.info("✅ App ready to serve traffic - components initializing in background!")
            
            # 5. Schedule lazy initialization of heavy components
            asyncio.create_task(self._lazy_initialize_llm())
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            # Don't raise - allow app to start even if background tasks fail
    
    async def _initialize_database(self):
        """Initialize database connection and create tables"""
        # Guard against duplicate DB init
        if not ensure_single_init("database"):
            logger.info("🔁 Database already initialized, skipping")
            return
        
        try:
            from database import get_db, engine, Base, register_models
            
            # Register all models using lazy registration to avoid circular imports
            logger.info("🔄 Registering database models...")
            register_models()
            
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
        # Guard against duplicate Redis init
        if not ensure_single_init("redis"):
            logger.info("🔁 Redis already initialized, skipping")
            return
        
        try:
            from services.redis_cache import init_cache
            # 30 second timeout for cross-cloud GCP→AWS Redis connection
            # Allows time for VPC connector → Cloud NAT → AWS EC2 HAProxy → MemoryDB
            await asyncio.wait_for(init_cache(), timeout=30.0)
            self.redis_cache = True
            logger.info("✅ Redis cache initialized successfully")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Redis cache timeout (30.0s) - likely firewall/network blocked")
            logger.warning("⚠️ Continuing WITHOUT Redis - sessions will not persist across restarts")
            self.redis_cache = None
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable: {e}")
            logger.warning("⚠️ Continuing WITHOUT Redis - sessions will not persist across restarts")
            self.redis_cache = None
    
    async def _initialize_service_manager(self):
        """Initialize local service manager"""
        # Guard against duplicate service manager init
        if not ensure_single_init("services"):
            logger.info("🔁 Service Manager already initialized, skipping")
            return
        
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
    
    async def _initialize_multi_route_system(self):
        """Initialize multi-route transportation system with comfort scoring"""
        # Guard against duplicate multi-route init
        if not ensure_single_init("multi_route"):
            logger.info("🔁 Multi-route system already initialized, skipping")
            return
        
        try:
            logger.info("🔄 Initializing Moovit-style multi-route system...")
            
            # Import route optimizer
            from services.route_optimizer import RouteOptimizer
            self.route_optimizer = RouteOptimizer()
            logger.info("✅ Route Optimizer initialized")
            
            # Import transportation route integration
            from services.transportation_route_integration import get_route_integration
            self.transportation_route_integration = get_route_integration()
            logger.info("✅ Transportation Route Integration initialized")
            
            # Import enhanced map visualization service
            from services.enhanced_map_visualization_service import EnhancedMapVisualizationService
            self.enhanced_map_visualization = EnhancedMapVisualizationService()
            logger.info("✅ Enhanced Map Visualization Service initialized")
            
            self._multi_route_initialized = True
            logger.info("🗺️ Multi-route system ready - Moovit-level features enabled!")
            
        except Exception as e:
            error_msg = f"Multi-route system initialization failed: {str(e)}"
            logger.warning(f"⚠️ {error_msg}")
            logger.warning("⚠️ Continuing WITHOUT multi-route - fallback to single route")
            self._initialization_errors.append(error_msg)
            self.route_optimizer = None
            self.transportation_route_integration = None
            self.enhanced_map_visualization = None
    
    async def _lazy_initialize_llm(self):
        """Lazy initialization of Pure LLM Core (runs in background)"""
        # Guard against duplicate LLM init
        if not ensure_single_init("models"):
            logger.info("🔁 LLM already initialized, skipping")
            return
        
        try:
            logger.info("🔄 Starting background LLM initialization...")
            
            # 1. Preload ML models FIRST (before LLM, to share memory efficiently)
            # DISABLED FOR CLOUD RUN: We use RunPod LLM, no need to download models
            # This was causing 40-second startup delays trying to download from HuggingFace
            # try:
            #     from services.model_cache import preload_models
            #     preload_models(['default', 'multilingual'])
            #     logger.info("✅ ML models preloaded")
            # except Exception as e:
            #     logger.warning(f"⚠️ Model preload skipped: {e}")
            logger.info("⏭️ Model preloading disabled (using RunPod LLM)")
            
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
                logger.warning("⚠️ Database not initialized - LLM will work with limited functionality")
                # Proceed without database - LLM can still work
            
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
    
    def is_multi_route_ready(self) -> bool:
        """Check if multi-route system is ready"""
        return self._multi_route_initialized and self.route_optimizer is not None
    
    def get_route_optimizer(self):
        """Get route optimizer instance"""
        if not self._multi_route_initialized:
            logger.warning("⚠️ Multi-route system still initializing...")
        return self.route_optimizer
    
    def get_transportation_route_integration(self):
        """Get transportation route integration instance"""
        return self.transportation_route_integration
    
    def get_enhanced_map_visualization(self):
        """Get enhanced map visualization service"""
        return self.enhanced_map_visualization


# Create singleton instance
fast_startup_manager = FastStartupManager()
