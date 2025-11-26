"""
Application Startup Module

Initialization logic for all services and components
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class StartupManager:
    """Manages application startup and initialization"""
    
    def __init__(self):
        self.pure_llm_core = None
        self.recommendation_engine = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all services"""
        logger.info("🚀 Starting AI Istanbul Backend")
        logger.info("=" * 60)
        
        try:
            await self._initialize_pure_llm()
            await self._initialize_recommendation_engine()
            await self._check_ml_service()
            
            self.initialized = True
            logger.info("=" * 60)
            logger.info("✅ Backend startup complete")
            
        except Exception as e:
            logger.error(f"❌ Startup failed: {e}", exc_info=True)
            raise
    
    async def _initialize_pure_llm(self):
        """Initialize Pure LLM Handler with resilience features"""
        try:
            from config.settings import settings
            
            if not settings.PURE_LLM_MODE:
                logger.info("⚠️ Pure LLM mode disabled")
                return
            
            logger.info("⚡ Initializing Pure LLM Handler with resilience features...")
            
            # Import dependencies
            from database import get_db
            from services.llm import PureLLMCore
            
            try:
                from services.runpod_llm_client import get_llm_client
                import os
                llm_url = os.getenv("LLM_API_URL")
                logger.info(f"   🔍 LLM_API_URL from env: {llm_url}")
                llm_client = get_llm_client()
                if llm_client and llm_client.enabled:
                    logger.info(f"   ✅ RunPod LLM Client initialized: {llm_client.api_url}")
                else:
                    logger.warning(f"   ⚠️ RunPod LLM Client created but disabled (no URL)")
                    llm_client = None
            except Exception as e:
                logger.error(f"   ❌ RunPod LLM initialization failed: {e}")
                import traceback
                traceback.print_exc()
                llm_client = None
            
            # Get database
            db = next(get_db())
            
            # Configuration for Pure LLM Core
            config = {
                'rag_service': None,
                'redis_client': None,
                'weather_service': None,
                'events_service': None,
                'enable_cache': True,
                'enable_analytics': True,
                'enable_experimentation': False,
                'enable_conversation': True,
                'enable_query_enhancement': True,
                # Resilience configuration
                'llm_failure_threshold': 5,
                'db_failure_threshold': 3,
                'max_retries': 3,
                'timeouts': {
                    'query_enhancement': 2.0,
                    'cache_lookup': 0.5,
                    'signal_detection': 1.0,
                    'context_building': 5.0,
                    'llm_generation': 15.0,
                    'cache_storage': 1.0
                }
            }
            
            # Create Pure LLM Core with circuit breakers, retry, and timeout management
            self.pure_llm_core = PureLLMCore(
                llm_client=llm_client,
                db_connection=db,
                config=config
            )
            
            logger.info("✅ Pure LLM Core initialized with circuit breakers and resilience patterns")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Pure LLM: {e}")
            self.pure_llm_core = None
    
    async def _initialize_recommendation_engine(self):
        """Initialize recommendation engine"""
        try:
            from backend.services.integrated_recommendation_engine import IntegratedRecommendationEngine
            from config.settings import settings
            
            self.recommendation_engine = IntegratedRecommendationEngine(
                redis_url=settings.REDIS_URL,
                enable_contextual_bandits=True,
                enable_basic_bandits=True,
                n_candidates=100
            )
            
            logger.info("✅ Contextual Bandit Recommendation Engine initialized")
            
            # Start periodic state saving
            asyncio.create_task(self._periodic_state_save())
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Recommendation Engine: {e}")
            self.recommendation_engine = None
    
    async def _check_ml_service(self):
        """Check ML service availability"""
        try:
            from backend.ml_service_client import get_ml_status
            
            ml_status = await get_ml_status()
            if ml_status.get('ml_service', {}).get('healthy'):
                logger.info("✅ ML Answering Service: Connected and Healthy")
            else:
                logger.warning("⚠️ ML Answering Service: Available but Not Healthy")
                
        except Exception as e:
            logger.warning(f"⚠️ ML Answering Service: Error - {e}")
    
    async def _periodic_state_save(self):
        """Periodically save state to Redis"""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                if self.recommendation_engine:
                    await self.recommendation_engine.save_state()
                    logger.debug("💾 Contextual bandit state saved")
            except Exception as e:
                logger.error(f"Error saving state: {e}")
    
    def get_pure_llm_core(self):
        """Get Pure LLM Core instance"""
        return self.pure_llm_core
    
    def get_recommendation_engine(self):
        """Get recommendation engine instance"""
        return self.recommendation_engine


# Global startup manager
startup_manager = StartupManager()
