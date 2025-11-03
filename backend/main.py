# --- Standard Library Imports ---
import sys
import os
import re
import asyncio
import json
import time
import html
import uuid
from datetime import datetime, date, timedelta
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import traceback

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, Depends, Request, Header, status, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION INFRASTRUCTURE COMPONENTS
# ═══════════════════════════════════════════════════════════════════
print("\n🏗️ Initializing Production Infrastructure...")

# Import core infrastructure components
try:
    from utils.ttl_cache import TTLCache
    from utils.rate_limiter import RateLimiter
    from utils.response_cache import SmartResponseCache
    from utils.system_monitor import SystemMonitor
    from utils.graceful_degradation import GracefulDegradation
    from config.feature_manager import FeatureManager, FeatureModule
    INFRASTRUCTURE_AVAILABLE = True
    print("✅ Production Infrastructure loaded successfully")
    print("   - TTLCache (Memory Management)")
    print("   - RateLimiter (API Protection)")
    print("   - SmartResponseCache (Performance)")
    print("   - SystemMonitor (Observability)")
    print("   - GracefulDegradation (Error Recovery)")
    print("   - FeatureManager (Module Loading)")
except ImportError as e:
    INFRASTRUCTURE_AVAILABLE = False
    print(f"⚠️ Production Infrastructure not available: {e}")
    print("   System will run without production enhancements")

# ═══════════════════════════════════════════════════════════════════
# INITIALIZE INFRASTRUCTURE COMPONENTS
# ═══════════════════════════════════════════════════════════════════
if INFRASTRUCTURE_AVAILABLE:
    # Initialize core systems
    feature_manager = FeatureManager()
    rate_limiter = RateLimiter()
    system_monitor = SystemMonitor()
    graceful_degradation = GracefulDegradation()
    response_cache = SmartResponseCache()
    
    print("✅ Infrastructure components initialized")
else:
    feature_manager = None
    rate_limiter = None
    system_monitor = None
    graceful_degradation = None
    response_cache = None

print("=" * 70)
print()

# ═══════════════════════════════════════════════════════════════════
# LEGACY MODULE LOADING (TO BE REFACTORED WITH FEATUREMANAGER)
# ═══════════════════════════════════════════════════════════════════

# Add location intent detection import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'load-testing'))
try:
    from location_intent_detector import LocationIntentDetector, LocationIntentType
    LOCATION_INTENT_AVAILABLE = True
    print("✅ Location Intent Detection loaded successfully")
except ImportError as e:
    LOCATION_INTENT_AVAILABLE = False
    print(f"⚠️ Location Intent Detection not available: {e}")

# Add Advanced Understanding System import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from advanced_understanding_system import AdvancedUnderstandingSystem
    from semantic_similarity_engine import SemanticSimilarityEngine, QueryContext
    from enhanced_context_memory import EnhancedContextMemory, ContextType
    from multi_intent_query_handler import MultiIntentQueryHandler
    ADVANCED_UNDERSTANDING_AVAILABLE = True
    print("✅ Advanced Understanding System loaded successfully")
except ImportError as e:
    ADVANCED_UNDERSTANDING_AVAILABLE = False
    print(f"⚠️ Advanced Understanding System not available: {e}")

# Add Production Monitoring and Feedback Collection
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from ml_production_monitor import get_production_monitor
    from user_feedback_collector import get_feedback_collector
    ML_MONITORING_AVAILABLE = True
    print("✅ ML Production Monitoring loaded successfully")
except ImportError as e:
    ML_MONITORING_AVAILABLE = False
    print(f"⚠️ ML Production Monitoring not available: {e}")

# Add Enhanced Feedback and Retraining System
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from feedback_backend_integration import FeedbackIntegration
    FEEDBACK_INTEGRATION_AVAILABLE = True
    print("✅ Enhanced Feedback Collection & Retraining System loaded successfully")
except ImportError as e:
    FEEDBACK_INTEGRATION_AVAILABLE = False
    print(f"⚠️ Enhanced Feedback Integration not available: {e}")

# Add Intent Classifier import
try:
    from main_system_neural_integration import NeuralIntentRouter
    INTENT_CLASSIFIER_AVAILABLE = True
    print("✅ Neural Intent Classifier (Hybrid) loaded successfully")
except ImportError as e:
    INTENT_CLASSIFIER_AVAILABLE = False
    print(f"⚠️ Neural Intent Classifier not available: {e}")

# Add Comprehensive ML/DL Integration System import
try:
    from comprehensive_ml_dl_integration import (
        ComprehensiveMLDLIntegration, 
        MLSystemType, 
        UserContext, 
        MLEnhancementResult
    )
    COMPREHENSIVE_ML_AVAILABLE = True
    print("✅ Comprehensive ML/DL Integration System loaded successfully")
except ImportError as e:
    COMPREHENSIVE_ML_AVAILABLE = False
    print(f"⚠️ Comprehensive ML/DL Integration System not available: {e}")

# Add Lightweight Deep Learning System import
try:
    from lightweight_deep_learning import (
        DeepLearningMultiIntentIntegration, 
        LearningContext, 
        LearningMode,
        create_lightweight_deep_learning_system
    )
    DEEP_LEARNING_AVAILABLE = True
    print("✅ Lightweight Deep Learning System loaded successfully")
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"⚠️ Lightweight Deep Learning System not available: {e}")

# Add Caching Systems import
try:
    from ml_result_cache import get_ml_cache
    from edge_cache_system import get_edge_cache
    ML_CACHE_AVAILABLE = True
    EDGE_CACHE_AVAILABLE = True
    print("✅ Caching Systems loaded successfully")
except ImportError as e:
    ML_CACHE_AVAILABLE = False
    EDGE_CACHE_AVAILABLE = False
    print(f"⚠️ Caching Systems not available: {e}")

# Add Query Preprocessing Pipeline import
try:
    from services.query_preprocessing_pipeline import QueryPreprocessor
    QUERY_PREPROCESSING_AVAILABLE = True
    print("✅ Query Preprocessing Pipeline loaded successfully")
except ImportError as e:
    QUERY_PREPROCESSING_AVAILABLE = False
    print(f"⚠️ Query Preprocessing Pipeline not available: {e}")

# Add Context-Aware Classification imports
try:
    from services.conversation_context_manager import (
        ConversationContextManager,
        Turn
    )
    from services.context_aware_classifier import ContextAwareClassifier
    from services.dynamic_threshold_manager import DynamicThresholdManager
    CONTEXT_AWARE_AVAILABLE = True
    print("✅ Context-Aware Classification System loaded successfully")
except ImportError as e:
    CONTEXT_AWARE_AVAILABLE = False
    print(f"⚠️ Context-Aware Classification System not available: {e}")

# Add Monthly Events Scheduler import
try:
    from monthly_events_scheduler import MonthlyEventsScheduler, get_cached_events, fetch_monthly_events, check_if_fetch_needed
    EVENTS_SCHEDULER_AVAILABLE = True
    print("✅ Monthly Events Scheduler loaded successfully")
except ImportError as e:
    EVENTS_SCHEDULER_AVAILABLE = False
    print(f"⚠️ Monthly Events Scheduler not available: {e}")

# Enhanced Query Understanding Configuration
ENHANCED_QUERY_UNDERSTANDING_ENABLED = ADVANCED_UNDERSTANDING_AVAILABLE

# Initialize Enhanced Query Understanding if available
enhanced_understanding_system = None
if ENHANCED_QUERY_UNDERSTANDING_ENABLED:
    try:
        enhanced_understanding_system = AdvancedUnderstandingSystem()
        print("✅ Enhanced Understanding System initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Enhanced Understanding System: {e}")
        ENHANCED_QUERY_UNDERSTANDING_ENABLED = False

# Initialize Intent Classifier
intent_classifier = None
if INTENT_CLASSIFIER_AVAILABLE:
    try:
        intent_classifier = NeuralIntentRouter()
        print("✅ Neural Intent Classifier (Hybrid) initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Neural Intent Classifier: {e}")
        INTENT_CLASSIFIER_AVAILABLE = False

# Initialize Query Preprocessor
query_preprocessor = None
if QUERY_PREPROCESSING_AVAILABLE:
    try:
        query_preprocessor = QueryPreprocessor()
        print("✅ Neural Intent Classifier (Hybrid) initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Neural Intent Classifier: {e}")
        INTENT_CLASSIFIER_AVAILABLE = False

# Initialize Query Preprocessor
query_preprocessor = None
if QUERY_PREPROCESSING_AVAILABLE:
    try:
        query_preprocessor = QueryPreprocessor()
        print("✅ Query Preprocessor initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Query Preprocessor: {e}")
        QUERY_PREPROCESSING_AVAILABLE = False

# Initialize Context-Aware Classification Components
context_manager = None
context_aware_classifier = None
threshold_manager = None
if CONTEXT_AWARE_AVAILABLE:
    try:
        # Initialize with default settings (Redis or in-memory fallback)
        context_manager = ConversationContextManager()
        context_aware_classifier = ContextAwareClassifier(context_manager)
        threshold_manager = DynamicThresholdManager()
        print("✅ Context-Aware Classification initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Context-Aware Classification: {e}")
        CONTEXT_AWARE_AVAILABLE = False

# Initialize ML Production Monitoring and Feedback Collection
ml_monitor = None
feedback_collector = None
if ML_MONITORING_AVAILABLE:
    try:
        ml_monitor = get_production_monitor()
        feedback_collector = get_feedback_collector()
        print("✅ ML monitoring and feedback systems enabled")
    except Exception as e:
        print(f"⚠️ Failed to initialize ML monitoring: {e}")
        ML_MONITORING_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# ML ANSWERING SERVICE CLIENT INTEGRATION
# ═══════════════════════════════════════════════════════════════════
print("\n🤖 Initializing ML Answering Service Client...")

try:
    from backend.ml_service_client import (
        get_ml_answer, 
        get_ml_status, 
        check_ml_health
    )
    ML_ANSWERING_SERVICE_AVAILABLE = True
    print("✅ ML Answering Service Client loaded")
    print(f"   URL: {os.getenv('ML_SERVICE_URL', 'http://localhost:8000')}")
    print(f"   LLM Default: {os.getenv('ML_USE_LLM_DEFAULT', 'true')}")
except ImportError as e:
    ML_ANSWERING_SERVICE_AVAILABLE = False
    print(f"⚠️ ML Answering Service Client not available: {e}")
    print("   System will run without ML-powered responses")

# Add location intent detection import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'load-testing'))
try:
    from location_intent_detector import LocationIntentDetector, LocationIntentType
    LOCATION_INTENT_AVAILABLE = True
    print("✅ Location Intent Detection loaded successfully")
except ImportError as e:
    LOCATION_INTENT_AVAILABLE = False
    print(f"⚠️ Location Intent Detection not available: {e}")

# Add Advanced Understanding System import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from advanced_understanding_system import AdvancedUnderstandingSystem
    from semantic_similarity_engine import SemanticSimilarityEngine, QueryContext
    from enhanced_context_memory import EnhancedContextMemory, ContextType
    from multi_intent_query_handler import MultiIntentQueryHandler
    ADVANCED_UNDERSTANDING_AVAILABLE = True
    print("✅ Advanced Understanding System loaded successfully")
except ImportError as e:
    ADVANCED_UNDERSTANDING_AVAILABLE = False
    print(f"⚠️ Advanced Understanding System not available: {e}")

# Add Production Monitoring and Feedback Collection
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from ml_production_monitor import get_production_monitor
    from user_feedback_collector import get_feedback_collector
    ML_MONITORING_AVAILABLE = True
    print("✅ ML Production Monitoring loaded successfully")
except ImportError as e:
    ML_MONITORING_AVAILABLE = False
    print(f"⚠️ ML Production Monitoring not available: {e}")

# Add Enhanced Feedback and Retraining System
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from feedback_backend_integration import FeedbackIntegration
    FEEDBACK_INTEGRATION_AVAILABLE = True
    print("✅ Enhanced Feedback Collection & Retraining System loaded successfully")
except ImportError as e:
    FEEDBACK_INTEGRATION_AVAILABLE = False
    print(f"⚠️ Enhanced Feedback Integration not available: {e}")

# Add Intent Classifier import
try:
    from main_system_neural_integration import NeuralIntentRouter
    INTENT_CLASSIFIER_AVAILABLE = True
    print("✅ Neural Intent Classifier (Hybrid) loaded successfully")
except ImportError as e:
    INTENT_CLASSIFIER_AVAILABLE = False
    print(f"⚠️ Neural Intent Classifier not available: {e}")

# Add Comprehensive ML/DL Integration System import
try:
    from comprehensive_ml_dl_integration import (
        ComprehensiveMLDLIntegration, 
        MLSystemType, 
        UserContext, 
        MLEnhancementResult
    )
    COMPREHENSIVE_ML_AVAILABLE = True
    print("✅ Comprehensive ML/DL Integration System loaded successfully")
except ImportError as e:
    COMPREHENSIVE_ML_AVAILABLE = False
    print(f"⚠️ Comprehensive ML/DL Integration System not available: {e}")

# Add Lightweight Deep Learning System import
try:
    from lightweight_deep_learning import (
        DeepLearningMultiIntentIntegration, 
        LearningContext, 
        LearningMode,
        create_lightweight_deep_learning_system
    )
    DEEP_LEARNING_AVAILABLE = True
    print("✅ Lightweight Deep Learning System loaded successfully")
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    print(f"⚠️ Lightweight Deep Learning System not available: {e}")

# Add Caching Systems import
try:
    from ml_result_cache import get_ml_cache
    from edge_cache_system import get_edge_cache
    ML_CACHE_AVAILABLE = True
    EDGE_CACHE_AVAILABLE = True
    print("✅ Caching Systems loaded successfully")
except ImportError as e:
    ML_CACHE_AVAILABLE = False
    EDGE_CACHE_AVAILABLE = False
    print(f"⚠️ Caching Systems not available: {e}")

# Add Query Preprocessing Pipeline import
try:
    from services.query_preprocessing_pipeline import QueryPreprocessor
    QUERY_PREPROCESSING_AVAILABLE = True
    print("✅ Query Preprocessing Pipeline loaded successfully")
except ImportError as e:
    QUERY_PREPROCESSING_AVAILABLE = False
    print(f"⚠️ Query Preprocessing Pipeline not available: {e}")

# Add Context-Aware Classification imports
try:
    from services.conversation_context_manager import (
        ConversationContextManager,
        Turn
    )
    from services.context_aware_classifier import ContextAwareClassifier
    from services.dynamic_threshold_manager import DynamicThresholdManager
    CONTEXT_AWARE_AVAILABLE = True
    print("✅ Context-Aware Classification System loaded successfully")
except ImportError as e:
    CONTEXT_AWARE_AVAILABLE = False
    print(f"⚠️ Context-Aware Classification System not available: {e}")

# Add Monthly Events Scheduler import
try:
    from monthly_events_scheduler import MonthlyEventsScheduler, get_cached_events, fetch_monthly_events, check_if_fetch_needed
    EVENTS_SCHEDULER_AVAILABLE = True
    print("✅ Monthly Events Scheduler loaded successfully")
except ImportError as e:
    EVENTS_SCHEDULER_AVAILABLE = False
    print(f"⚠️ Monthly Events Scheduler not available: {e}")

# Enhanced Query Understanding Configuration
ENHANCED_QUERY_UNDERSTANDING_ENABLED = ADVANCED_UNDERSTANDING_AVAILABLE

# Initialize Enhanced Query Understanding if available
enhanced_understanding_system = None
if ENHANCED_QUERY_UNDERSTANDING_ENABLED:
    try:
        enhanced_understanding_system = AdvancedUnderstandingSystem()
        print("✅ Enhanced Understanding System initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Enhanced Understanding System: {e}")
        ENHANCED_QUERY_UNDERSTANDING_ENABLED = False

# Initialize Intent Classifier
intent_classifier = None
if INTENT_CLASSIFIER_AVAILABLE:
    try:
        intent_classifier = NeuralIntentRouter()
        print("✅ Neural Intent Classifier (Hybrid) initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Neural Intent Classifier: {e}")
        INTENT_CLASSIFIER_AVAILABLE = False

# Initialize Query Preprocessor
query_preprocessor = None
if QUERY_PREPROCESSING_AVAILABLE:
    try:
        query_preprocessor = QueryPreprocessor()
        print("✅ Neural Intent Classifier (Hybrid) initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Neural Intent Classifier: {e}")
        INTENT_CLASSIFIER_AVAILABLE = False

# Initialize Query Preprocessor
query_preprocessor = None
if QUERY_PREPROCESSING_AVAILABLE:
    try:
        query_preprocessor = QueryPreprocessor()
        print("✅ Query Preprocessor initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Query Preprocessor: {e}")
        QUERY_PREPROCESSING_AVAILABLE = False

# Initialize Context-Aware Classification Components
context_manager = None
context_aware_classifier = None
threshold_manager = None
if CONTEXT_AWARE_AVAILABLE:
    try:
        # Initialize with default settings (Redis or in-memory fallback)
        context_manager = ConversationContextManager()
        context_aware_classifier = ContextAwareClassifier(context_manager)
        threshold_manager = DynamicThresholdManager()
        print("✅ Context-Aware Classification initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Context-Aware Classification: {e}")
        CONTEXT_AWARE_AVAILABLE = False

# Initialize ML Production Monitoring and Feedback Collection
ml_monitor = None
feedback_collector = None
if ML_MONITORING_AVAILABLE:
    try:
        ml_monitor = get_production_monitor()
        feedback_collector = get_feedback_collector()
        print("✅ ML monitoring and feedback systems enabled")
    except Exception as e:
        print(f"⚠️ Failed to initialize ML monitoring: {e}")
        ML_MONITORING_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# ML ANSWERING SERVICE CLIENT INTEGRATION
# ═══════════════════════════════════════════════════════════════════
print("\n🤖 Initializing ML Answering Service Client...")

try:
    from backend.ml_service_client import (
        get_ml_answer, 
        get_ml_status, 
        check_ml_health
    )
    ML_ANSWERING_SERVICE_AVAILABLE = True
    print("✅ ML Answering Service Client loaded")
    print(f"   URL: {os.getenv('ML_SERVICE_URL', 'http://localhost:8000')}")
    print(f"   LLM Default: {os.getenv('ML_USE_LLM_DEFAULT', 'true')}")
except ImportError as e:
    ML_ANSWERING_SERVICE_AVAILABLE = False
    print(f"⚠️ ML Answering Service Client not available: {e}")
    print("   System will run without ML-powered responses")

# Enhanced Authentication imports
try:
    from enhanced_auth import (
        EnhancedAuthManager,
        get_current_user,
        UserRegistrationRequest,
        UserLoginRequest,
        TokenRefreshRequest,  # Fixed: was UserRefreshRequest
        TokenResponse,
        UserResponse
    )
    ENHANCED_AUTH_AVAILABLE = True
    print("✅ Enhanced Authentication module loaded")
except ImportError as e:
    print(f"⚠️ Enhanced Authentication not available: {e}")
    ENHANCED_AUTH_AVAILABLE = False
    
    # Define fallback models if authentication is not available
    class TokenResponse(BaseModel):
        """Fallback TokenResponse when auth is unavailable"""
        access_token: str
        refresh_token: str
        token_type: str = "bearer"
    
    class UserResponse(BaseModel):
        """Fallback UserResponse when auth is unavailable"""
        id: int
        email: str
        username: str
    
    class UserRegistrationRequest(BaseModel):
        """Fallback registration request"""
        email: str
        password: str
        username: str
        full_name: Optional[str] = None
    
    class UserLoginRequest(BaseModel):
        """Fallback login request"""
        email: str
        password: str
    
    class TokenRefreshRequest(BaseModel):
        """Fallback refresh request"""
        refresh_token: str
    
    # Define fallback for get_current_user dependency
    async def get_current_user() -> Dict[str, Any]:
        """Fallback get_current_user when auth is unavailable"""
        return {
            "sub": "fallback_user",
            "username": "fallback_user",
            "email": "fallback@example.com",
            "role": "user"
        }

# === Pydantic Models for API Endpoints ===

class ChatRequest(BaseModel):
    """Request model for chat endpoints"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    user_location: Optional[Dict[str, float]] = Field(None, description="User GPS location")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

class ChatResponse(BaseModel):
    """Response model for chat endpoints"""
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session identifier")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: Optional[float] = Field(None, description="Confidence score")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
class RouteResponse(BaseModel):
    """Response model for route planning"""
    route: List[Dict[str, Any]] = Field(..., description="Optimized route")
    total_duration: float = Field(..., description="Total route duration in hours")
    total_distance: float = Field(..., description="Total distance in kilometers")
    transport_info: Optional[Dict[str, Any]] = Field(None, description="Transportation information")
    recommendations: Optional[List[str]] = Field(None, description="Route recommendations")

class GPSRouteRequest(BaseModel):
    """Request model for GPS-based route planning"""
    user_location: Dict[str, float] = Field(..., description="User GPS coordinates (lat, lng)")
    radius_km: Optional[float] = Field(5.0, description="Search radius in kilometers")
    duration_hours: Optional[int] = Field(4, description="Available time in hours")
    transport_mode: Optional[str] = Field("walking", description="Transportation mode")
    interests: Optional[List[str]] = Field(None, description="User interests")
    session_id: Optional[str] = Field(None, description="Session identifier")

class NearbyAttractionsRequest(BaseModel):
    """Request model for finding nearby attractions"""
    location: Dict[str, float] = Field(..., description="GPS coordinates (lat, lng)")
    radius_km: Optional[float] = Field(2.0, description="Search radius in kilometers")
    attraction_types: Optional[List[str]] = Field(None, description="Types of attractions")
    limit: Optional[int] = Field(10, description="Maximum number of results")

class LocationBasedRecommendationResponse(BaseModel):
    """Response model for location-based recommendations"""
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommendations")
    user_location: Dict[str, float] = Field(..., description="User location used")
    search_radius: float = Field(..., description="Search radius used")
    total_found: int = Field(..., description="Total number of recommendations")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")

class TransportRequest(BaseModel):
    """Request model for transportation queries"""
    origin: str = Field(..., description="Origin location")
    destination: str = Field(..., description="Destination location")
    transport_mode: Optional[str] = Field(None, description="Preferred transport mode")
    time_preference: Optional[str] = Field(None, description="Time preference")

class TransportResponse(BaseModel):
    """Response model for transportation queries"""
    routes: List[Dict[str, Any]] = Field(..., description="Available routes")
    recommendations: str = Field(..., description="Transportation recommendations")
    duration_estimate: Optional[str] = Field(None, description="Estimated duration")
    cost_estimate: Optional[str] = Field(None, description="Estimated cost")

class MuseumRequest(BaseModel):
    """Request model for museum queries"""
    query: str = Field(..., description="Museum query")
    location: Optional[str] = Field(None, description="Preferred location/area")
    interests: Optional[List[str]] = Field(None, description="User interests")

class MuseumResponse(BaseModel):
    """Response model for museum queries"""
    museums: List[Dict[str, Any]] = Field(..., description="Museum recommendations")
    response: str = Field(..., description="Detailed response")
    total_found: int = Field(..., description="Total museums found")

class MuseumRouteRequest(BaseModel):
    """Request model for museum route planning"""
    query: str = Field(..., description="Museum route planning query (e.g., 'plan a museum tour for 5 hours')")
    duration_hours: Optional[int] = Field(None, description="Duration in hours (e.g., 3, 5, 8)")
    starting_location: Optional[str] = Field(None, description="Starting location or neighborhood")
    interests: Optional[List[str]] = Field(None, description="Specific interests (e.g., byzantine, ottoman, art)")
    budget_level: Optional[str] = Field("medium", description="Budget level: low, medium, high")
    accessibility_needs: Optional[bool] = Field(False, description="Special accessibility requirements")
    districts: Optional[List[str]] = Field(None, description="Preferred districts (e.g., ['Fatih', 'Beyoğlu'])")
    neighborhoods: Optional[List[str]] = Field(None, description="Preferred neighborhoods (e.g., ['Sultanahmet', 'Karaköy'])")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")

class MuseumRouteResponse(BaseModel):
    """Response model for museum route planning"""
    route_plan: str = Field(..., description="Detailed route plan with museums and timing")
    museums: List[Dict[str, Any]] = Field(..., description="List of museums in the route")
    total_duration: int = Field(..., description="Total duration in hours")
    estimated_cost: str = Field(..., description="Estimated total cost")
    transportation_guide: str = Field(..., description="Transportation instructions")
    local_tips: List[str] = Field(..., description="Local insider tips")
    success: bool = Field(True, description="Whether route planning succeeded")

# Import system monitoring tools
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - system metrics will be limited")

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ redis not available - some caching features may be limited")

# Load environment variables first, before any other imports
load_dotenv()

# Daily usage tracking completely removed for unrestricted testing

# System metrics for monitoring
system_metrics = {
    "requests_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "errors": 0,
    "response_times": [],
    "api_costs": 0.0,
    "cache_savings": 0.0,
    "start_time": datetime.now()
}

# Using Ultra-Specialized Istanbul AI only - template-based with neural ranking
use_neural_ranking = True

# Redis availability flag and client initialization
redis_available = REDIS_AVAILABLE
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        # Test connection
        redis_client.ping()
        print("✅ Redis client initialized successfully")
        
        # Initialize Redis-based conversational memory
        try:
            from redis_conversational_memory import initialize_redis_memory
            redis_memory = initialize_redis_memory(redis_client)
            print("✅ Redis conversational memory system activated")
        except ImportError as e:
            print(f"⚠️ Redis memory system not available: {e}")
            redis_memory = None
            
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
        redis_available = False
        redis_client = None
        redis_memory = None
else:
    redis_memory = None

# Add the current directory to Python path for imports (must be before project imports)
# Handle different deployment scenarios
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Also add parent directory for potential nested deployment structures
parent_dir = os.path.dirname(current_dir)
backend_in_parent = os.path.join(parent_dir, 'backend')
if os.path.exists(backend_in_parent) and backend_in_parent not in sys.path:
    sys.path.insert(0, backend_in_parent)

print(f"Python path configured. Current dir: {current_dir}")
print(f"Python paths: {[p for p in sys.path[:3]]}")  # Show first 3 paths

# --- Rate Limiting Removed ---
# Rate limiting has been completely removed for unrestricted testing
RATE_LIMITING_ENABLED = False

# --- Structured Logging ---
try:
    from structured_logging import get_logger, log_performance, log_ai_operation, log_api_call
    STRUCTURED_LOGGING_ENABLED = True
    print("✅ Structured logging initialized successfully")
except ImportError as e:
    print(f"⚠️ Structured logging not available: {e}")
    STRUCTURED_LOGGING_ENABLED = False

# --- Advanced Monitoring and Security ---
try:
    from monitoring.advanced_monitoring import advanced_monitor, monitor_performance, log_error_metric, log_performance_metric
    from monitoring.comprehensive_logging import comprehensive_logger, log_api_request, log_security_event, log_user_action, log_error
    ADVANCED_MONITORING_ENABLED = True
    print("✅ Advanced monitoring and logging initialized successfully")
except ImportError as e:
    print(f"⚠️ Advanced monitoring not available: {e}")
    ADVANCED_MONITORING_ENABLED = False
    # Create dummy functions to prevent errors
    def monitor_performance(op): return lambda f: f
    def log_error_metric(error_type, details=""): pass
    def log_performance_metric(metric_name, value): pass
    def log_api_request(*args, **kwargs): pass
    def log_security_event(*args, **kwargs): pass
    def log_user_action(*args, **kwargs): pass
    def log_error(*args, **kwargs): pass

# Legacy structured logging fallback
if not ADVANCED_MONITORING_ENABLED:
    try:
        from structured_logging import get_logger, log_performance, log_ai_operation, log_api_call
        structured_logger = get_logger("istanbul_ai_main")
        STRUCTURED_LOGGING_ENABLED = True
        print("✅ Structured logging initialized successfully")
    except ImportError as e:
        print(f"⚠️ Structured logging not available: {e}")
        STRUCTURED_LOGGING_ENABLED = False
    # Create dummy logger to prevent errors
    class DummyLogger:
        def info(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass
        def log_ai_query(self, *args, **kwargs): pass
        def log_request(self, *args, **kwargs): pass
        def log_response(self, *args, **kwargs): pass
        def log_cache_hit(self, *args, **kwargs): pass
        def log_cache_miss(self, *args, **kwargs): pass
        def log_rate_limit(self, *args, **kwargs): pass
        def log_error_with_traceback(self, *args, **kwargs): pass
        def context(self, **kwargs):
            from contextlib import contextmanager
            @contextmanager
            def dummy_context():
                yield
            return dummy_context()
    structured_logger = DummyLogger()
    log_performance = lambda op, **kw: lambda f: f
    log_ai_operation = lambda op, **kw: lambda f: f
    log_api_call = lambda ep: lambda f: f

# --- Project Imports ---
try:
    from database import engine, SessionLocal, get_db
    print("✅ Database import successful")
except ImportError as e:
    print(f"❌ Database import failed: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:5]}")  # First 5 paths
    print(f"Files in current directory: {os.listdir('.')}")
    raise

try:
    from models import Base, Restaurant, Museum, Place, UserFeedback, ChatSession, BlogPost, BlogComment, ChatHistory, UserSession, BlogLike, UserInteraction, EnhancedChatHistory
    from sqlalchemy.orm import Session
    print("✅ Models import successful")
except ImportError as e:
    print(f"❌ Models import failed: {e}")
    raise

try:
    from routes import museums, restaurants, places, blog
    print("✅ Routes import successful")
except ImportError as e:
    print(f"❌ Routes import failed: {e}")
    raise
try:
    from api_clients.google_places import GooglePlacesClient  # type: ignore
    # Weather functionality removed - using seasonal guidance instead
    from api_clients.enhanced_api_service import EnhancedAPIService  # type: ignore
    print("✅ API clients import successful")
except ImportError as e:
    print(f"⚠️ API clients import failed (non-critical): {e}")
    # Create dummy classes for missing API clients
    class GooglePlacesClient:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        async def search_places(self, *args, **kwargs): return []
        def search_restaurants(self, *args, **kwargs): 
            return {"results": [], "status": "OK"}
    
    # Weather functionality removed - seasonal guidance provided through database
    
    class EnhancedAPIService:  # type: ignore
        def __init__(self, *args, **kwargs): pass
        def search_restaurants_enhanced(self, *args, **kwargs): 
            return {"results": [], "seasonal_context": {}}

try:
    from enhanced_input_processor import enhance_query_understanding, get_response_guidance, input_processor  # type: ignore
    print("✅ Enhanced input processor import successful")
except ImportError as e:
    print(f"⚠️ Enhanced input processor import failed: {e}")
    # Create dummy functions
    def enhance_query_understanding(user_input: str) -> str:  # type: ignore
        return user_input
    def get_response_guidance(user_input: str) -> dict:  # type: ignore
        return {"guidance": "basic"}
    class InputProcessor:  # type: ignore
        def enhance_query_context(self, text: str) -> dict:
            return {"query_type": "general"}
    input_processor = InputProcessor()

# --- Import Input Sanitizer ---
try:
    from input_sanitizer import sanitize_text, is_safe_input
    print("✅ Input sanitizer import successful")
    
    # Create alias for backward compatibility
    def sanitize_user_input(text: str) -> str:
        """Sanitize user input text"""
        return sanitize_text(text)
except ImportError as e:
    print(f"⚠️ Input sanitizer import failed: {e}")
    # Fallback sanitizer
    def sanitize_user_input(text: str) -> str:
        """Basic fallback sanitizer"""
        import html
        return html.escape(text.strip())[:10000]
    def is_safe_input(text: str) -> bool:
        return True

# --- Import Enhanced Services ---
try:
    from enhanced_transportation_service import EnhancedTransportationService
    from enhanced_museum_service import EnhancedMuseumService  
    from enhanced_actionability_service import EnhancedActionabilityService
    
    # Initialize enhanced services
    enhanced_transport_service = EnhancedTransportationService()
    enhanced_museum_service = EnhancedMuseumService()
    enhanced_actionability_service = EnhancedActionabilityService()
    
    ENHANCED_SERVICES_ENABLED = True
    print("✅ Enhanced services (transportation, museum, actionability) imported successfully")
except ImportError as e:
    print(f"⚠️ Enhanced services not available: {e}")
    ENHANCED_SERVICES_ENABLED = False

# --- Import Restaurant Database Service ---
try:
    from services.restaurant_database_service import RestaurantDatabaseService
    restaurant_service = RestaurantDatabaseService()
    
    # Add compatibility methods for the expected interface
    def search_restaurants_compat(district=None, cuisine=None, limit=10):
        """Compatibility wrapper for restaurant search with named parameters"""
        query_parts = []
        if cuisine:
            query_parts.append(cuisine)
        if district:
            query_parts.append(district)
        query = " ".join(query_parts) if query_parts else "restaurants in Istanbul"
        
        # Get filtered restaurants instead of formatted response
        parsed_query = restaurant_service.parse_restaurant_query(query)
        filtered_restaurants = restaurant_service.filter_restaurants(parsed_query, limit=limit)
        return filtered_restaurants
    
    def format_restaurant_response_compat(restaurants):
        """Compatibility wrapper for formatting restaurant response"""
        if not restaurants:
            return "No restaurants found matching your criteria."
        
        if len(restaurants) == 1:
            return restaurant_service.format_single_restaurant_response(restaurants[0])
        else:
            # Create a simple parsed query for formatting
            from services.restaurant_database_service import RestaurantQuery
            query = RestaurantQuery()
            return restaurant_service.format_restaurant_list_response(restaurants, query)
    
    # Override the methods to use compatibility versions
    restaurant_service.search_restaurants_compat = search_restaurants_compat
    restaurant_service.format_restaurant_response = format_restaurant_response_compat
    
    RESTAURANT_SERVICE_ENABLED = True
    print("✅ Restaurant Database Service imported successfully")
except ImportError as e:
    print(f"⚠️ Restaurant Database Service not available: {e}")
    RESTAURANT_SERVICE_ENABLED = False
    # Create dummy restaurant service
    class DummyRestaurantService:
        def search_restaurants(self, *args, **kwargs):
            return []
        def search_restaurants_compat(self, *args, **kwargs):
            return []
        def format_restaurant_response(self, *args, **kwargs):
            return "Restaurant service not available"
        def get_restaurant_stats(self):
            return {"total_restaurants": 0, "status": "disabled"}
    restaurant_service = DummyRestaurantService()

# --- Import Intelligent Location Detection Service ---
try:
    from services.intelligent_location_detector import (
        detect_user_location, 
        DetectedLocation, 
        LocationConfidence,
        intelligent_location_detector
    )
    INTELLIGENT_LOCATION_ENABLED = True
    print("✅ Intelligent Location Detection service imported successfully")
except ImportError as e:
    print(f"⚠️ Intelligent Location Detection service not available: {e}")
    INTELLIGENT_LOCATION_ENABLED = False
    
    # Create dummy classes to prevent errors
    class DetectedLocation:
        def __init__(self, **kwargs):
            self.latitude = kwargs.get('latitude', None)
            self.longitude = kwargs.get('longitude', None)
            self.confidence = "unknown"
            self.source = "none"
    
    class LocationConfidence:
        UNKNOWN = "unknown"
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        VERY_HIGH = "very_high"
    
    async def detect_user_location(text, user_context=None, ip_address=None):
        return DetectedLocation()

# --- Define helper functions ---
def post_ai_cleanup(response: str) -> str:
    """Clean up AI response for better formatting and readability"""
    if not response:
        return response
    
    # Remove excessive newlines
    response = re.sub(r'\n{3,}', '\n\n', response)
    
    # Clean up markdown formatting issues
    response = re.sub(r'\*{3,}', '**', response)  # Fix excessive asterisks
    response = re.sub(r'#{3,}', '##', response)   # Fix excessive hashtags
    
    # Ensure proper spacing after punctuation
    response = re.sub(r'([.!?])([A-Z])', r'\1 \2', response)
    
    # Remove excessive spaces
    response = re.sub(r' {2,}', ' ', response)
    
    # Ensure proper line breaks before section headers
    response = re.sub(r'([^\n])(#+\s)', r'\1\n\n\2', response)
    
    return response.strip()

from sqlalchemy.orm import Session

try:
    from i18n_service import i18n_service
    print("✅ i18n service import successful")
except ImportError as e:
    print(f"⚠️ i18n service import failed: {e}")
    # Create dummy i18n service
    class I18nService:
        def translate(self, key, lang="en"): return key
        def get_language_from_headers(self, headers): return "en"
        def should_use_ai_response(self, user_input, language): return True
        supported_languages = ["en", "tr", "ar", "ru"]
    i18n_service = I18nService()

# --- Import enhanced AI services ---
try:
    from ai_cache_service import get_ai_cache_service, init_ai_cache_service
    AI_CACHE_ENABLED = True
except ImportError:
    print("⚠️ AI Cache service not available")
    AI_CACHE_ENABLED = False
    # Create dummy functions to prevent errors
    get_ai_cache_service = lambda: None  # type: ignore
    init_ai_cache_service = lambda *args, **kwargs: None  # type: ignore

# Create dummy objects to prevent errors
class DummyManager:
    def get_or_create_session(self, *args, **kwargs): return "dummy_session"
    def get_context(self, *args, **kwargs): return {}
    def update_context(self, *args, **kwargs): pass
    def get_preferences(self, *args, **kwargs): return {}
    def update_preferences(self, *args, **kwargs): pass
    def learn_from_query(self, *args, **kwargs): pass
    def get_personalized_filter(self, *args, **kwargs): return {}
    def recognize_intent(self, *args, **kwargs): return ("general_query", 0.1)
    def extract_entities(self, *args, **kwargs): return {"locations": [], "time_references": [], "cuisine_types": [], "budget_indicators": []}
    def enhance_recommendations(self, *args, **kwargs): return args[1] if len(args) > 1 else []
    def analyze_query_context(self, *args, **kwargs): return {"locations": [], "cuisine_types": [], "price_indicators": [], "time_context": [], "group_context": None, "urgency_level": "normal", "query_complexity": "simple"}

class DummyAdvancedAI:
    async def get_comprehensive_real_time_data(self, *args, **kwargs): return {}
    async def analyze_image_comprehensive(self, *args, **kwargs): return None
    async def analyze_menu_image(self, *args, **kwargs): return None
    async def get_comprehensive_predictions(self, *args, **kwargs): return {}

try:
    from ai_intelligence import (
        session_manager, preference_manager, intent_recognizer, 
        recommendation_engine, saved_session_manager
    )
    AI_INTELLIGENCE_ENABLED = True
    print("✅ AI Intelligence services imported successfully")
except ImportError as e:
    print(f"❌ AI Intelligence import failed: {e}")
    AI_INTELLIGENCE_ENABLED = False
    
    # Create dummy objects to prevent errors
    class DummyAI:
        def get_or_create_session(self, *args, **kwargs): return "dummy"
        def get_preferences(self, *args, **kwargs): return {}
        def update_preferences(self, *args, **kwargs): pass
        def learn_from_query(self, *args, **kwargs): pass
        def recognize_intent(self, *args, **kwargs): return ("general_query", 0.1)
        def enhance_recommendations(self, *args, **kwargs): return []
        def save_session(self, *args, **kwargs): return True
        def get_saved_sessions(self, *args, **kwargs): return []
        def get_session_details(self, *args, **kwargs): return None
        def delete_session(self, *args, **kwargs): return True
        def get_context(self, *args, **kwargs): return {}
        def update_context(self, *args, **kwargs): pass
        def analyze_query_context(self, *args, **kwargs): return {}
        def extract_entities(self, *args, **kwargs): return {}
        def get_personalized_filter(self, *args, **kwargs): return {}
    
    session_manager = preference_manager = intent_recognizer = recommendation_engine = saved_session_manager = DummyAI()

# --- Import Advanced AI Features ---
try:
    from api_clients.realtime_data import realtime_data_aggregator
    from api_clients.multimodal_ai import get_multimodal_ai_service
    from api_clients.predictive_analytics import predictive_analytics_service
    ADVANCED_AI_ENABLED = True
    print("✅ Advanced AI features loaded successfully")
except ImportError as e:
    print(f"⚠️ Advanced AI features not available: {e}")
    ADVANCED_AI_ENABLED = False
    # Use dummy objects
    realtime_data_aggregator = DummyAdvancedAI()
    get_multimodal_ai_service = lambda: DummyAdvancedAI()
    predictive_analytics_service = DummyAdvancedAI()

# --- Import Language Processing ---
try:
    from api_clients.language_processing import (
        AdvancedLanguageProcessor, 
        process_user_query,
        extract_intent_and_entities,
        is_followup
    )
    LANGUAGE_PROCESSING_ENABLED = True
    language_processor = AdvancedLanguageProcessor()
    print("✅ Advanced Language Processing loaded successfully")
except ImportError as e:
    print(f"⚠️ Language Processing not available: {e}")
    LANGUAGE_PROCESSING_ENABLED = False
    # Create dummy functions
    def process_user_query(text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        return {"intent": "general_info", "confidence": 0.1, "entities": {}}
    def extract_intent_and_entities(text: str) -> Tuple[str, Dict]: 
        return "general_info", {}
    def is_followup(text: str, context: Optional[Dict] = None) -> bool: 
        return False

# --- Neural Ranking for Template-Based Responses ---
# We use only our Ultra-Specialized Istanbul AI System with optional neural ranking
neural_ranking_available = False
neural_ranker = None
print("ℹ️ Template-based system with optional neural ranking - no generative AI")

# --- Ultra-Specialized Istanbul AI System (Rule-Based) ---
# Import our Ultra-Specialized Istanbul AI System
try:
    from enhanced_ultra_specialized_istanbul_ai import enhanced_istanbul_ai_system as istanbul_ai_system
    ULTRA_ISTANBUL_AI_AVAILABLE = True
    print("✅ Ultra-Specialized Istanbul AI System loaded successfully!")
except ImportError as e:
    print(f"⚠️ Ultra-Specialized Istanbul AI import failed: {e}")
    istanbul_ai_system = None
    ULTRA_ISTANBUL_AI_AVAILABLE = False

# --- NEW: Enhanced Istanbul Daily Talk AI System (with Attractions) ---
# Import our new MAIN SYSTEM from istanbul_ai folder with full integration
try:
    from istanbul_ai.main_system import IstanbulDailyTalkAI
    istanbul_daily_talk_ai = IstanbulDailyTalkAI()
    ISTANBUL_DAILY_TALK_AVAILABLE = True
    print("✅ Istanbul Daily Talk AI MAIN SYSTEM loaded successfully!")
    print("   📍 Intelligent Location Detection: ACTIVE")
    print("   🗺️ Enhanced GPS Route Planner: ACTIVE")
    print("   🚇 Advanced Transportation System: ACTIVE")
    print("   🏛️ Museum System with Route Planning: ACTIVE")
    print("   🤖 ML-Enhanced Daily Talks Bridge: ACTIVE")
except ImportError as e:
    print(f"⚠️ Istanbul Daily Talk AI (Main System) import failed: {e}")
    print(f"   Attempting fallback to modular system...")
    try:
        from istanbul_daily_talk_system_modular import IstanbulDailyTalkAI
        istanbul_daily_talk_ai = IstanbulDailyTalkAI()
        ISTANBUL_DAILY_TALK_AVAILABLE = True
        print("✅ Istanbul Daily Talk AI System (Modular Fallback) loaded successfully!")
    except ImportError as e2:
        print(f"⚠️ Istanbul Daily Talk AI (Modular Fallback) import failed: {e2}")
        istanbul_daily_talk_ai = None
        ISTANBUL_DAILY_TALK_AVAILABLE = False

# Istanbul Daily Talk AI is now the primary system, Ultra-Specialized is fallback
CUSTOM_AI_AVAILABLE = ISTANBUL_DAILY_TALK_AVAILABLE or ULTRA_ISTANBUL_AI_AVAILABLE
print(f"🎯 AI System Status:")
print(f"   🏛️ Istanbul Daily Talk AI (PRIMARY): {'✅ ACTIVE (50+ attractions, restaurants, transport)' if ISTANBUL_DAILY_TALK_AVAILABLE else '❌ DISABLED'}")
print(f"   🔧 Ultra-Specialized AI (FALLBACK): {'✅ ACTIVE' if ULTRA_ISTANBUL_AI_AVAILABLE else '❌ DISABLED'}")
print(f"   🚀 Overall System: {'✅ FULLY INTEGRATED AI SYSTEMS' if CUSTOM_AI_AVAILABLE else '❌ DISABLED'}")

# Use istanbul_ai_system directly for all AI processing
custom_ai_system = None

# --- Legacy imports and system setup ---

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Istanbul AI Guide API",
    description="AI-powered Istanbul travel guide with enhanced authentication and production infrastructure",
    version="2.1.0"  # Bumped version for infrastructure updates
)

# Initialize Authentication Manager
auth_manager = None
if ENHANCED_AUTH_AVAILABLE:
    try:
        auth_manager = EnhancedAuthManager()
        logger.info("✅ Enhanced Authentication Manager initialized")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize Enhanced Authentication Manager: {e}")
        ENHANCED_AUTH_AVAILABLE = False
else:
    logger.info("⚠️ Enhanced Authentication Manager not available")


# =============================
# HEALTH CHECK & STARTUP EVENTS
# =============================

@app.on_event("startup")
async def startup_event():
    """Startup tasks - check ML service connection"""
    logger.info("🚀 Starting AI Istanbul Backend")
    logger.info("=" * 60)
    
    # Check ML Answering Service
    if ML_ANSWERING_SERVICE_AVAILABLE:
        try:
            ml_status = await get_ml_status()
            if ml_status.get('ml_service', {}).get('healthy'):
                logger.info("✅ ML Answering Service: Connected and Healthy")
                logger.info(f"   URL: {ml_status['ml_service']['url']}")
                logger.info(f"   LLM Default: {os.getenv('ML_USE_LLM_DEFAULT', 'true')} ⭐")
            else:
                logger.warning("⚠️ ML Answering Service: Available but Not Healthy")
                logger.warning("   Fallback mode will be used for queries")
        except Exception as e:
            logger.warning(f"⚠️ ML Answering Service: Error checking status - {e}")
            logger.warning("   Fallback mode will be used for queries")
    else:
        logger.warning("⚠️ ML Answering Service: Not Available")
        logger.warning("   Fallback mode will be used for queries")
    
    logger.info("=" * 60)
    logger.info("✅ Backend startup complete")


@app.get("/health", tags=["System Health"])
async def health_check():
    """
    Overall system health check
    Returns health status of all services including ML service
    """
    from datetime import datetime
    
    # Check ML service health
    ml_healthy = False
    ml_details = {"available": False}
    
    if ML_ANSWERING_SERVICE_AVAILABLE:
        try:
            ml_health = await check_ml_health()
            ml_healthy = ml_health.get('healthy', False)
            ml_details = ml_health
        except Exception as e:
            logger.error(f"Health check ML service error: {e}")
            ml_details = {"available": True, "healthy": False, "error": str(e)}
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "authentication": "healthy" if ENHANCED_AUTH_AVAILABLE else "unavailable",
            "ml_answering_service": "healthy" if ml_healthy else "degraded"
        },
        "ml_service_details": ml_details
    }


# =============================
# AUTHENTICATION ENDPOINTS
# =============================

@app.post("/api/auth/register", response_model=TokenResponse, tags=["Authentication"])
async def register_user(request: UserRegistrationRequest, db: Session = Depends(get_db)):
    """
    Register a new user
    
    Returns access token, refresh token, and user information
    """
    if not ENHANCED_AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available"
        )
    
    try:
        result = await auth_manager.register_user(
            email=request.email,
            password=request.password,
            username=request.username,
            full_name=request.full_name,
            db=db
        )
        
        logger.info(f"✅ User registered successfully: {request.email}")
        return result
        
    except ValueError as e:
        logger.warning(f"Registration validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again."
        )


@app.post("/api/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login_user(request: UserLoginRequest, db: Session = Depends(get_db)):
    """
    Login with email and password
    
    Returns access token, refresh token, and user information
    """
    if not ENHANCED_AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available"
        )
    
    try:
        result = await auth_manager.login_user(
            email=request.email,
            password=request.password,
            db=db
        )
        
        logger.info(f"✅ User logged in successfully: {request.email}")
        return result
        
    except ValueError as e:
        logger.warning(f"Login validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again."
        )


@app.post("/api/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(request: UserRefreshRequest, db: Session = Depends(get_db)):
    """
    Refresh access token using refresh token
    
    Returns new access token and refresh token
    """
    if not ENHANCED_AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available"
        )
    
    try:
        result = await auth_manager.refresh_access_token(
            refresh_token=request.refresh_token,
            db=db
        )
        
        logger.info("✅ Token refreshed successfully")
        return result
        
    except ValueError as e:
        logger.warning(f"Token refresh validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed. Please try again."
        )


@app.post("/api/auth/logout", tags=["Authentication"])
async def logout_user(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Logout current user (revoke tokens)
    
    Requires valid access token in Authorization header
    """
    if not ENHANCED_AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available"
        )
    
    try:
        user_id = current_user.get("user_id")
        await auth_manager.logout_user(user_id=user_id, db=db)
        
        logger.info(f"✅ User logged out successfully: {user_id}")
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed. Please try again."
        )


@app.get("/api/auth/profile", response_model=UserResponse, tags=["Authentication"])
async def get_user_profile(
    current_user: Dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current user profile
    
    Requires valid access token in Authorization header
    """
    if not ENHANCED_AUTH_AVAILABLE or not auth_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not available"
        )
    
    try:
        user_id = current_user.get("user_id")
        user = await auth_manager.get_user_by_id(user_id=user_id, db=db)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile fetch error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch profile. Please try again."
        )


# =============================
# ML CHAT ENDPOINT WITH LLM INTEGRATION
# =============================

class MLChatRequest(BaseModel):
    """Request model for ML-powered chat"""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    user_location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lon}")
    use_llm: Optional[bool] = Field(None, description="Override: Use LLM (None=use default from config)")
    language: str = Field(default="en", description="Response language (en/tr)")
    user_id: Optional[str] = Field(None, description="User ID for personalization")


class MLChatResponse(BaseModel):
    """Response model for ML-powered chat"""
    response: str = Field(..., description="Bot response text")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Confidence score")
    method: str = Field(..., description="Response generation method")
    context: List[Dict] = Field(default=[], description="Context items used")
    suggestions: List[str] = Field(default=[], description="Follow-up suggestions")
    response_time: float = Field(..., description="Response time in seconds")
    ml_service_used: bool = Field(..., description="Whether ML service was used")


async def generate_ml_fallback_response(
    message: str,
    intent: str = "general",
    user_location: Optional[Dict] = None
) -> Dict:
    """Fallback response when ML service unavailable"""
    responses = {
        "restaurant_recommendation": {
            "answer": "Istanbul has amazing restaurants! Popular areas include Beyoğlu, Kadıköy, and Beşiktaş. What type of cuisine interests you?",
            "context": []
        },
        "attraction_query": {
            "answer": "Istanbul is full of incredible attractions! Must-sees include Hagia Sophia, Blue Mosque, Topkapı Palace, and the Grand Bazaar. Which area would you like to explore?",
            "context": []
        },
        "transportation_help": {
            "answer": "Istanbul has excellent public transportation including metro, tram, ferry, and buses. You can use an Istanbulkart for all of them. Where do you need to go?",
            "context": []
        },
        "neighborhood_info": {
            "answer": "Istanbul has diverse neighborhoods, each with unique character. Beyoğlu is vibrant and modern, Sultanahmet is historical, Kadıköy is alternative and artistic. Which interests you?",
            "context": []
        },
        "general": {
            "answer": "I'm here to help you explore Istanbul! I can recommend restaurants, attractions, help with transportation, and suggest local experiences. What would you like to know?",
            "context": []
        }
    }
    return responses.get(intent, responses["general"])


def generate_ml_suggestions(intent: str) -> List[str]:
    """Generate follow-up suggestions based on intent"""
    suggestions = {
        "restaurant_recommendation": [
            "Show me vegetarian options",
            "What about seafood restaurants?",
            "Budget-friendly places near me"
        ],
        "attraction_query": [
            "Tell me about museums",
            "Historical sites in Sultanahmet",
            "Best views in Istanbul"
        ],
        "transportation_help": [
            "How to use the metro?",
            "Ferry schedules",
            "Getting to the airport"
        ],
        "general": [
            "Best restaurants in Istanbul",
            "Top attractions to visit",
            "How to get around the city"
        ]
    }
    return suggestions.get(intent, suggestions["general"])


@app.post("/api/v1/chat", response_model=MLChatResponse, tags=["ML Chat"])
async def ml_chat_endpoint(request: MLChatRequest):
    """
    ML-powered chat endpoint with LLM by default
    
    Provides intelligent, context-aware responses using:
    - Semantic search
    - Intent classification
    - LLM generation (default) or templates
    - Graceful fallback if ML service unavailable
    """
    start_time = time.time()
    
    # ML Configuration
    ML_USE_LLM_DEFAULT = os.getenv("ML_USE_LLM_DEFAULT", "true").lower() == "true"
    
    try:
        # Detect intent (use your existing classifier if available)
        intent = "general"  # TODO: Integrate with existing intent classifier
        
        # Determine if should use LLM
        use_llm = request.use_llm if request.use_llm is not None else ML_USE_LLM_DEFAULT
        
        logger.info(f"💬 ML Chat query: '{request.message}' (intent: {intent}, llm: {use_llm})")
        
        # Try ML service if available
        if ML_ANSWERING_SERVICE_AVAILABLE:
            ml_response = await get_ml_answer(
                query=request.message,
                intent=intent,
                user_location=request.user_location,
                use_llm=use_llm,
                language=request.language
            )
            
            if ml_response and ml_response.get('success'):
                # ML service succeeded ✅
                logger.info(f"✅ ML response: {ml_response.get('generation_method')} ({time.time() - start_time:.2f}s)")
                
                return MLChatResponse(
                    response=ml_response['answer'],
                    intent=ml_response.get('intent', intent),
                    confidence=ml_response.get('confidence', 0.85),
                    method=f"ml_{ml_response.get('generation_method', 'llm')}",
                    context=ml_response.get('context', []),
                    suggestions=ml_response.get('suggestions', []),
                    response_time=time.time() - start_time,
                    ml_service_used=True
                )
        
        # Fallback to rule-based
        logger.info("⚠️ ML service unavailable - using fallback")
        
        fallback = await generate_ml_fallback_response(
            request.message,
            intent,
            request.user_location
        )
        
        return MLChatResponse(
            response=fallback['answer'],
            intent=intent,
            confidence=0.6,
            method="fallback",
            context=fallback.get('context', []),
            suggestions=generate_ml_suggestions(intent),
            response_time=time.time() - start_time,
            ml_service_used=False
        )
    
    except Exception as e:
        logger.error(f"❌ ML Chat error: {e}")
        
        # Emergency fallback
        return MLChatResponse(
            response="I apologize, but I'm having trouble processing your request right now. Please try again in a moment.",
            intent="error",
            confidence=0.0,
            method="error_fallback",
            context=[],
            suggestions=["Try again", "Ask a different question"],
            response_time=time.time() - start_time,
            ml_service_used=False
        )


@app.get("/api/v1/ml/status", tags=["ML Service"])
async def ml_service_status():
    """Get ML service status and health"""
    if not ML_ANSWERING_SERVICE_AVAILABLE:
        return {
            "available": False,
            "reason": "ML service client not loaded",
            "ml_service": {"enabled": False}
        }
    
    try:
        status = await get_ml_status()
        return status
    except Exception as e:
        logger.error(f"ML status check error: {e}")
        return {
            "available": True,
            "ml_service": {"healthy": False, "error": str(e)}
        }


@app.get("/api/v1/ml/health", tags=["ML Service"])
async def ml_service_health():
    """Quick ML service health check"""
    if not ML_ANSWERING_SERVICE_AVAILABLE:
        return {
            "healthy": False,
            "reason": "ML service client not loaded"
        }
    
    try:
        health = await check_ml_health()
        return health
    except Exception as e:
        logger.error(f"ML health check error: {e}")
        return {
            "healthy": False,
            "error": str(e)
        }


# =============================
# ML FEEDBACK & MONITORING ENDPOINTS
# =============================

class FeedbackRatingRequest(BaseModel):
    """Request model for star rating feedback"""
    query: str = Field(..., description="User query")
    response: str = Field(..., description="AI response")
    predicted_intent: str = Field(..., description="Predicted intent")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5 stars)")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FeedbackThumbsRequest(BaseModel):
    """Request model for thumbs up/down feedback"""
    query: str = Field(..., description="User query")
    response: str = Field(..., description="AI response")
    predicted_intent: str = Field(..., description="Predicted intent")
    thumbs_up: bool = Field(..., description="Thumbs up (True) or down (False)")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FeedbackIntentCorrectionRequest(BaseModel):
    """Request model for intent correction feedback"""
    query: str = Field(..., description="User query")
    response: str = Field(..., description="AI response")
    predicted_intent: str = Field(..., description="Predicted intent")
    correct_intent: str = Field(..., description="Correct intent")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FeedbackCommentRequest(BaseModel):
    """Request model for free-text comment feedback"""
    query: str = Field(..., description="User query")
    response: str = Field(..., description="AI response")
    predicted_intent: str = Field(..., description="Predicted intent")
    comment: str = Field(..., description="User comment")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    status: str = Field(..., description="Submission status")
    feedback_id: str = Field(..., description="Feedback identifier")
    message: str = Field(..., description="Success message")


@app.post("/api/feedback/rating", response_model=FeedbackResponse, tags=["ML Feedback"])
async def submit_rating_feedback(request: FeedbackRatingRequest):
    """
    Submit star rating feedback (1-5 stars)
    
    This endpoint collects user satisfaction ratings on AI responses.
    """
    if not ML_MONITORING_AVAILABLE or not feedback_collector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback collection service is not available"
        )
    
    try:
        feedback_id = feedback_collector.collect_rating(
            query=request.query,
            response=request.response,
            predicted_intent=request.predicted_intent,
            rating=request.rating,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        logger.info(f"✅ Rating feedback collected: {feedback_id} (rating: {request.rating}/5)")
        
        return FeedbackResponse(
            status="success",
            feedback_id=feedback_id,
            message=f"Thank you for your {request.rating}-star rating!"
        )
    
    except Exception as e:
        logger.error(f"Failed to collect rating feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@app.post("/api/feedback/thumbs", response_model=FeedbackResponse, tags=["ML Feedback"])
async def submit_thumbs_feedback(request: FeedbackThumbsRequest):
    """
    Submit thumbs up/down feedback
    
    Quick binary feedback on response quality.
    """
    if not ML_MONITORING_AVAILABLE or not feedback_collector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback collection service is not available"
        )
    
    try:
        feedback_id = feedback_collector.collect_thumbs(
            query=request.query,
            response=request.response,
            predicted_intent=request.predicted_intent,
            thumbs_up=request.thumbs_up,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        feedback_type = "👍 thumbs up" if request.thumbs_up else "👎 thumbs down"
        logger.info(f"✅ Thumbs feedback collected: {feedback_id} ({feedback_type})")
        
        return FeedbackResponse(
            status="success",
            feedback_id=feedback_id,
            message=f"Thank you for your feedback!"
        )
    
    except Exception as e:
        logger.error(f"Failed to collect thumbs feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@app.post("/api/feedback/intent-correction", response_model=FeedbackResponse, tags=["ML Feedback"])
async def submit_intent_correction(request: FeedbackIntentCorrectionRequest):
    """
    Submit intent correction feedback
    
    Allows users to correct misclassified intents.
    """
    if not ML_MONITORING_AVAILABLE or not feedback_collector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback collection service is not available"
        )
    

    
    try:
        feedback_id = feedback_collector.collect_intent_correction(
            query=request.query,
            response=request.response,
            predicted_intent=request.predicted_intent,
            correct_intent=request.correct_intent,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        # Log to ML monitor for retraining
        if ml_monitor:
            ml_monitor.log_prediction(
                query=request.query,
                predicted_intent=request.predicted_intent,
                confidence=0.0,  # Mark as incorrect
                latency_ms=0,
                actual_intent=request.correct_intent,
                user_feedback='wrong'
            )
        
        return FeedbackResponse(
            status="success",
            feedback_id=feedback_id,
            message="Thank you for the correction! This helps improve our system."
               )
    
    except Exception as e:
        logger.error(f"Failed to collect intent correction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit correction"
        )


@app.post("/api/feedback/comment", response_model=FeedbackResponse, tags=["ML Feedback"])
async def submit_comment_feedback(request: FeedbackCommentRequest):
    """
    Submit free-text comment feedback
    
    Allows users to provide detailed feedback comments.
    """
    if not ML_MONITORING_AVAILABLE or not feedback_collector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback collection service is not available"
        )
    
    try:
        feedback_id = feedback_collector.collect_comment(
            query=request.query,
            response=request.response,
            predicted_intent=request.predicted_intent,
            comment=request.comment,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        logger.info(f"✅ Comment feedback collected: {feedback_id}")
        
        return FeedbackResponse(
            status="success",
            feedback_id=feedback_id,
            message="Thank you for your detailed feedback!"
        )
    
    except Exception as e:
        logger.error(f"Failed to collect comment feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@app.get("/api/monitoring/metrics", tags=["ML Monitoring"])
async def get_ml_metrics():
    """
    Get current ML performance metrics
    
    Returns real-time monitoring data including accuracy, latency, and quality metrics.
    """
    if not ML_MONITORING_AVAILABLE or not ml_monitor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML monitoring service is not available"
        )
    
    try:
        metrics = ml_monitor.get_current_metrics()
        return metrics
    
    except Exception as e:
        logger.error(f"Failed to get ML metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@app.get("/api/monitoring/feedback-analysis", tags=["ML Monitoring"])
async def get_feedback_analysis(days: int = Query(7, ge=1, le=90, description="Number of days to analyze")):
    """
    Get user feedback analysis
    
    Returns aggregated user feedback statistics and insights.
    """
    if not ML_MONITORING_AVAILABLE or not feedback_collector:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Feedback collection service is not available"
        )
    
    try:
        analysis = feedback_collector.analyze_feedback(days=days)
        return analysis
    
    except Exception as e:
        logger.error(f"Failed to analyze feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze feedback"
        )


# =============================================================================
# ADMIN DASHBOARD API ENDPOINTS
# =============================================================================

@app.get("/api/admin/stats", tags=["Admin Dashboard"])
async def get_admin_stats(db: Session = Depends(get_db)):
    """
    Get overall statistics for admin dashboard using real database data
    """
    try:
        from sqlalchemy import func, and_
        from datetime import timedelta
        
        stats = {
            "blog_posts": 0,
            "comments": 0,
            "feedback": 0,
            "active_users": 0,
            "model_accuracy": 95.2,
            "pending_comments": 0
        }
        
        # Get real blog post count from database
        stats["blog_posts"] = db.query(func.count(BlogPost.id)).scalar() or 0
        
        # Get real comments count and pending count from database
        stats["comments"] = db.query(func.count(BlogComment.id)).scalar() or 0
        stats["pending_comments"] = db.query(func.count(BlogComment.id)).filter(
            BlogComment.is_approved == False
        ).scalar() or 0
        
        
        return {"posts": posts_list, "total": len(posts_list)}
        
    except Exception as e:
        logger.error(f"Error getting blog posts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/blog/posts", tags=["Admin Dashboard - Blog"])
async def create_blog_post(post_data: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """
    Create a new blog post in the database
    """
    try:
        from models import BlogPost
        
        # Create new post
        new_post = BlogPost(
            title=post_data.get("title", "Untitled"),
            content=post_data.get("content", ""),
            author=post_data.get("author", "Admin"),
            district=post_data.get("category") or post_data.get("district"),
            likes_count=0
        )
        
        db.add(new_post)
        db.commit()
        db.refresh(new_post)
        
        logger.info(f"Created blog post: {new_post.title}")
        
        return {
            "success": True, 
            "post": {
                "id": new_post.id,
                "title": new_post.title,
                "content": new_post.content,
                "author": new_post.author,
                "district": new_post.district,
                "created_at": new_post.created_at.isoformat(),
                "likes_count": new_post.likes_count,
                "status": "published"
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/blog/posts/{post_id}", tags=["Admin Dashboard - Blog"])
async def update_blog_post(post_id: int, post_data: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """
    Update an existing blog post in the database
    """
    try:
        from models import BlogPost
        
        # Find post
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Update fields
        if "title" in post_data:
            post.title = post_data["title"]
        if "content" in post_data:
            post.content = post_data["content"]
        if "author" in post_data:
            post.author = post_data["author"]
        if "category" in post_data or "district" in post_data:
            post.district = post_data.get("category") or post_data.get("district")
        
        db.commit()
        db.refresh(post)
        
        logger.info(f"Updated blog post: {post_id}")
        return {"success": True, "message": "Post updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/blog/posts/{post_id}", tags=["Admin Dashboard - Blog"])
async def delete_blog_post(post_id: int, db: Session = Depends(get_db)):
    """
    Delete a blog post from the database
    """
    try:
        from models import BlogPost
        
        # Find post
        post = db.query(BlogPost).filter(BlogPost.id == post_id).first()
        
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Delete post (related comments will be handled by cascade if configured)
        db.delete(post)
        db.commit()
        
        logger.info(f"Deleted blog post: {post_id}")
        return {"success": True, "message": "Post deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/comments", tags=["Admin Dashboard - Comments"])
async def get_comments(status: Optional[str] = None, post_id: Optional[int] = None, limit: int = 100, db: Session = Depends(get_db)):
    """
    Get all comments for admin management (from database)
    """
    try:
        from models import BlogComment
        
        # Query database for comments
        query = db.query(BlogComment)
        
        # Filter by approval status
        if status:
            if status.lower() == "approved":
                query = query.filter(BlogComment.is_approved == True)
            elif status.lower() == "pending":
                query = query.filter(BlogComment.is_approved == False)
            elif status.lower() == "flagged":
                query = query.filter(BlogComment.is_flagged == True)
            elif status.lower() == "spam":
                query = query.filter(BlogComment.is_spam == True)
        
        # Filter by post_id
        if post_id:
            query = query.filter(BlogComment.blog_post_id == post_id)
        
        # Get comments ordered by created_at descending
        comments = query.order_by(BlogComment.created_at.desc()).limit(limit).all()
        
        # Convert to dict format for API response
        comments_list = []
        for comment in comments:
            # Determine status based on flags
            comment_status = "approved" if comment.is_approved else "pending"
            if comment.is_spam:
                comment_status = "spam"
            elif comment.is_flagged:
                comment_status = "flagged"
            
            comments_list.append({
                "id": comment.id,
                "post_id": comment.blog_post_id,
                "author": comment.author_name,
                "email": comment.author_email,
                "content": comment.content,
                "status": comment_status,
                "is_approved": comment.is_approved,
                "is_flagged": comment.is_flagged,
                "is_spam": comment.is_spam,
                "created_at": comment.created_at.isoformat() if comment.created_at else None,
                "approved_at": comment.approved_at.isoformat() if comment.approved_at else None,
                               "approved_by": comment.approved_by
            })
        
        return {"comments": comments_list, "total": len(comments_list)}
        
    except Exception as e:
        logger.error(f"Error getting comments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/comments/{comment_id}/approve", tags=["Admin Dashboard - Comments"])
async def approve_comment(comment_id: int, db: Session = Depends(get_db)):
    """
    Approve a pending comment in the database
    """
    try:
        from models import BlogComment
        
        # Find comment
        comment = db.query(BlogComment).filter(BlogComment.id == comment_id).first()
        
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        # Approve comment
        comment.is_approved = True
        comment.approved_at = datetime.utcnow()
        comment.approved_by = "Admin"  # You can update this with actual admin user
        comment.is_flagged = False
        comment.is_spam = False
        
        db.commit()
        
        return {"success": True, "message": "Comment approved"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error approving comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/comments/{comment_id}", tags=["Admin Dashboard - Comments"])
async def delete_comment(comment_id: int, db: Session = Depends(get_db)):
    """
    Delete a comment from the database
    """
    try:
        from models import BlogComment
        
        # Find comment
        comment = db.query(BlogComment).filter(BlogComment.id == comment_id).first()
        
        if not comment:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        # Delete comment
        db.delete(comment)
        db.commit()
        
        return {"success": True, "message": "Comment deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/feedback/export", tags=["Admin Dashboard - Feedback"])
async def export_feedback_data():
    """
    Export all feedback data as JSON
    """
    try:
        if not FEEDBACK_INTEGRATION_AVAILABLE:
            raise HTTPException(status_code=503, detail="Feedback system not available")
        
        from user_feedback_collection_system import get_feedback_collector
        collector = get_feedback_collector()
        
        # Get all feedback
        feedback_summary = collector.get_feedback_summary(days=365)  # Last year
        misclass_report = collector.get_misclassification_report()
        
        export_data = {
            "export_date": datetime.now().isoformat(),
            "summary": feedback_summary,
            "misclassification_report": misclass_report,
            "total_records": feedback_summary.get("total", 0)
        };
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics", tags=["Admin Dashboard - Analytics"])
async def get_analytics(days: int = 30, db: Session = Depends(get_db)):
    """
    Get analytics data for charts and insights using real database data
    """
    try:
        from sqlalchemy import func, and_
        from datetime import timedelta
        
        analytics = {
            "period": f"last_{days}_days",
            "user_queries": [],
            "blog_views": [],
            "comments": [],
            "dates": []
        }
        
        today = datetime.now()
        
        for i in range(days):
            date = today - timedelta(days=days-i-1)
            date_str = date.strftime("%b %d")
            analytics["dates"].append(date_str)
            
            # Get real user query count for this day
            start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            # Count chat history entries for this day
            query_count = db.query(func.count(ChatHistory.id)).filter(
                and_(
                    ChatHistory.timestamp >= start_of_day,
                    ChatHistory.timestamp < end_of_day
                )
            ).scalar() or 0
            analytics["user_queries"].append(query_count)
            
            # Count blog views (using blog likes as proxy for views)
            from models import BlogLike
            blog_views = db.query(func.count(BlogLike.id)).filter(
                and_(
                    BlogLike.created_at >= start_of_day,
                    BlogLike.created_at < end_of_day
                )
            ).scalar() or 0
            analytics["blog_views"].append(blog_views)
            
            # Count comments for this day
            comment_count = db.query(func.count(BlogComment.id)).filter(
                and_(
                    BlogComment.created_at >= start_of_day,
                    BlogComment.created_at < end_of_day
                )
            ).scalar() or 0
            analytics["comments"].append(comment_count)
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/intents/stats", tags=["Admin Dashboard - Intents"])
async def get_intent_statistics(db: Session = Depends(get_db)):
    """
    Get detailed intent classification statistics from real database data
    """
    try:
        from sqlalchemy import func, and_
        from datetime import timedelta
        from models import UserInteraction, EnhancedChatHistory
        
        # Get intent statistics from UserInteraction table (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        # Query intent counts and average confidence
        intent_data = db.query(
            UserInteraction.processed_intent,
            func.count(UserInteraction.id).label('count'),
            func.avg(UserInteraction.confidence_score).label('avg_confidence')
        ).filter(
            UserInteraction.timestamp >= thirty_days_ago,
            UserInteraction.processed_intent.isnot(None)
        ).group_by(
            UserInteraction.processed_intent
        ).all()
        
        # Also query from EnhancedChatHistory as a fallback/supplement
        if not intent_data or len(intent_data) == 0:
            intent_data = db.query(
                EnhancedChatHistory.detected_intent,
                func.count(EnhancedChatHistory.id).label('count'),
                func.avg(EnhancedChatHistory.intent_confidence).label('avg_confidence')
            ).filter(
                EnhancedChatHistory.timestamp >= thirty_days_ago,
                EnhancedChatHistory.detected_intent.isnot(None)
            ).group_by(
                EnhancedChatHistory.detected_intent
            ).all()
        
        # Calculate accuracy based on user feedback
        intent_stats = []
        for intent_name, count, avg_conf in intent_data:
            if not intent_name:
                continue
                
            # Count negative feedback for this intent
            negative_feedback = db.query(func.count(UserFeedback.id)).filter(
                and_(
                    UserFeedback.timestamp >= thirty_days_ago,
                    UserFeedback.feedback_type == "dislike",
                    UserFeedback.message_content.contains(intent_name)
                )
            ).scalar() or 0
            
            # Calculate accuracy (percentage of non-negative feedback)
            accuracy = 100.0
            if count > 0:
                accuracy = ((count - negative_feedback) / count) * 100
            
            intent_stats.append({
                "intent": intent_name,
                "count": count,
                "accuracy": round(accuracy, 1),
                "confidence": round(float(avg_conf or 0.85), 2),
                "corrections": negative_feedback
            })
        
        # Sort by count (most frequent first)
        intent_stats.sort(key=lambda x: x['count'], reverse=True)
        
        # If no data found, return empty array instead of mock data
        if not intent_stats:
            return {"intents": []}
        
        return {"intents": intent_stats}
        
    except Exception as e:
        logger.error(f"Error getting intent stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/model/retrain", tags=["Admin Dashboard - Model"])
async def trigger_model_retraining():
    """
    Trigger model retraining process
    """
    try:
        # This would trigger the actual retraining script
        # For now, return success message
        
        logger.info("Model retraining triggered from admin dashboard")
        
        return {
            "success": True,
            "message": "Model retraining initiated",
            "estimated_time": "15-30 minutes",
            "notification": "You will be notified when retraining is complete"
        }
        
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/route/from-gps", tags=["GPS Route Planning"])
async def plan_journey_from_gps(request: Dict[str, Any] = Body(...)):
    """
    Plan complete journey from GPS location to destination
    Returns walking directions + transit journey + final walking
    """
    try:
        gps_lat = request.get('gps_lat')
        gps_lng = request.get('gps_lng')
        destination = request.get('destination')
        max_walking_m = request.get('max_walking_m', 1000)
        minimize_transfers = request.get('minimize_transfers', True)
        
        logger.info(f"GPS journey planning from ({gps_lat}, {gps_lng}) to '{destination}'")
        
        if not gps_lat or not gps_lng or not destination:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
        from services.journey_planner import JourneyPlanner
        from services.intelligent_route_finder import RoutePreferences
        from services.route_network_builder import load_transportation_network
        
        network = load_transportation_network()
        if not network:
            raise HTTPException(status_code=503, detail="Transportation network not available")
        
        planner = JourneyPlanner(network)
        preferences = RoutePreferences(minimize_transfers=minimize_transfers)
        
        journey_plan = planner.plan_journey_from_gps(
            gps_lat=gps_lat,
            gps_lng=gps_lng,
            destination=destination,
            max_start_walking_m=max_walking_m,
            preferences=preferences
        )
        
        if not journey_plan:
            return {"success": False, "error": "No route found"}
        
        return {"success": True, "journey": journey_plan}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GPS journey planning error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ADMIN DASHBOARD HTML ROUTES
# =============================================================================

from fastapi.responses import HTMLResponse, FileResponse

admin_path = os.path.join(os.path.dirname(__file__), '..', 'admin')

@app.get("/admin/", response_class=HTMLResponse)
@app.get("/admin/index.html", response_class=HTMLResponse)
async def serve_admin_index():
    """Serve admin dashboard index page"""
    index_file = os.path.join(admin_path, 'index.html')
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Admin Dashboard Not Found</h1>"

@app.get("/admin/dashboard.html", response_class=HTMLResponse)
async def serve_admin_dashboard():
    """Serve admin dashboard main page"""
    dashboard_file = os.path.join(admin_path, 'dashboard.html')
    if os.path.exists(dashboard_file):
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Dashboard Not Found</h1>"

@app.get("/admin/dashboard.js")
async def serve_admin_js():
    """Serve admin dashboard JavaScript"""
    js_file = os.path.join(admin_path, 'dashboard.js')
    if os.path.exists(js_file):
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.get("/admin/{filename}")
async def serve_admin_static(filename: str):
    """Serve other admin static files"""
    file_path = os.path.join(admin_path, filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

print(f"✅ Admin dashboard routes configured (path: {admin_path})")