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
        print("✅ Query Preprocessor initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Query Preprocessor: {e}")
        QUERY_PREPROCESSING_AVAILABLE = False

def generate_enhanced_suggestions(intent: str, secondary_intents: List, detected_location=None) -> List[str]:
    """Generate enhanced suggestions based on multi-intent analysis"""
    suggestions = []
    
    # Primary intent suggestions
    if intent == "recommendation":
        suggestions.extend([
            "Show me more restaurant recommendations",
            "What about attractions in this area?",
            "Any events happening nearby?"
        ])
    elif intent == "route_planning":
        suggestions.extend([
            "How long does it take to get there?",
            "What's the best time to travel?",
            "Are there alternative routes?"
        ])
    elif intent == "information_request":
        suggestions.extend([
            "Tell me more about this area",
            "What's the history of this place?",
            "Any local customs I should know?"
        ])
    else:
        suggestions.extend([
            "Tell me about Istanbul attractions",
            "Recommend some restaurants",
            "What events are happening?"
        ])
    
    # Add location-specific suggestions if available
    if detected_location:
        suggestions.append(f"More recommendations for {detected_location}")
    
    # Add secondary intent suggestions
    for secondary_intent in secondary_intents[:2]:  # Limit to 2 secondary intents
        if secondary_intent == "transportation":
            suggestions.append("How do I get there?")
        elif secondary_intent == "price_query":
            suggestions.append("What about budget options?")
        elif secondary_intent == "time_query":
            suggestions.append("What are the opening hours?")
    
    return suggestions[:5]  # Limit to 5 suggestions

def generate_traditional_suggestions(intent: str) -> List[str]:
    """Generate traditional suggestions based on single intent"""
    intent_suggestions = {
        "restaurant_query": [
            "Show me more restaurants",
            "What about vegetarian options?",
            "Any fine dining recommendations?"
        ],
        "events_query": [
            "What events are happening today?",
            "Any cultural performances?",
            "Where can I find live music?"
        ],
        "transportation_query": [
            "How do I use public transport?",
            "What about taxi options?",
            "Are there bike rentals?"
        ],
        "attraction_query": [
            "What other attractions are nearby?",
            "Any hidden gems?",
            "What are the must-see places?"
        ]
    }
    
    return intent_suggestions.get(intent, [
        "Tell me about Istanbul attractions",
        "Recommend some restaurants", 
        "What events are happening?"
    ])

def process_enhanced_query(user_input: str, session_id: str) -> Dict[str, Any]:
    """Process query using Enhanced Understanding System with Neural Intent Classifier and Query Preprocessing"""
    
    # Step 1: Preprocess the query (typo correction, dialect normalization, entity extraction)
    preprocessing_result = None
    preprocessed_query = user_input
    if QUERY_PREPROCESSING_AVAILABLE and query_preprocessor:
        try:
            preprocessing_result = query_preprocessor.preprocess(user_input)
            preprocessed_query = preprocessing_result['processed_text']
            logger.info(f"🔧 Query preprocessed: '{user_input}' -> '{preprocessed_query}'")
            if preprocessing_result.get('corrections'):
                logger.info(f"✏️ Corrections applied: {len(preprocessing_result['corrections'])}")
            if preprocessing_result.get('entities'):
                logger.info(f"🏷️ Entities extracted: {list(preprocessing_result['entities'].keys())}")
        except Exception as e:
            logger.warning(f"Preprocessing error: {e}, using original query")
            preprocessed_query = user_input
    
    # Step 2: Try the neural intent classifier with hybrid fallback
    intent_result = None
    if INTENT_CLASSIFIER_AVAILABLE and intent_classifier:
        try:
            import time
            start_time = time.time()
            
            # Use the neural router's route_query method with preprocessed query
            routing_result = intent_classifier.route_query(preprocessed_query)
            latency_ms = (time.time() - start_time) * 1000
            
            intent = routing_result['intent']
            confidence = routing_result['confidence']
            method = routing_result.get('method', 'unknown')
            
            intent_result = {
                'intent': intent,
                'confidence': confidence,
                'latency_ms': latency_ms,
                'method': method,  # 'neural' or 'fallback'
                'fallback_used': routing_result.get('fallback_used', False)
            }
            
            logger.info(f"🎯 Intent Classifier ({method}): {intent} (confidence: {confidence:.2f}, latency: {latency_ms:.2f}ms)")
        except Exception as e:
            logger.warning(f"Intent classifier error: {e}")
    
    # Step 3: Use the enhanced understanding system for deeper analysis
    if not ENHANCED_QUERY_UNDERSTANDING_ENABLED or not enhanced_understanding_system:
        # Fall back to intent classifier only
        entities = preprocessing_result.get('entities', {}) if preprocessing_result else {}
        corrections = preprocessing_result.get('corrections', []) if preprocessing_result else []
        
        if intent_result and intent_result['confidence'] >= 0.6:
            return {
                'success': True,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'entities': entities,
                'corrections': corrections,
                'normalized_query': preprocessed_query.lower().strip(),
                'original_query': user_input,
                'classifier_used': f"neural_{intent_result['method']}",
                'preprocessing_stats': preprocessing_result.get('statistics') if preprocessing_result else None
            }
        else:
            return {
                'success': False,
                'intent': 'general_info',
                'confidence': 0.3,
                'entities': entities,
                'corrections': corrections,
                'normalized_query': preprocessed_query.lower().strip(),
                'original_query': user_input,
                'preprocessing_stats': preprocessing_result.get('statistics') if preprocessing_result else None
            }
                'corrections': [],
                'normalized_query': user_input.lower().strip(),
                'original_query': user_input
            }
    
    try:
        # Use the enhanced understanding system with preprocessed query
        result = enhanced_understanding_system.understand_query(preprocessed_query, session_id=session_id)
        
        # Extract intent from multi_intent_result
        multi_intent = result.multi_intent_result
        primary_intent = multi_intent.primary_intent.type.value if multi_intent.primary_intent else 'general_info'
        
        # Use intent classifier result if it has higher confidence
        if intent_result and intent_result['confidence'] > result.understanding_confidence:
            primary_intent = intent_result['intent']
            confidence = intent_result['confidence']
            logger.info(f"✨ Using neural intent classifier result (higher confidence: {confidence:.2f} vs {result.understanding_confidence:.2f})")
        else:
            confidence = result.understanding_confidence
        
        # Merge entities from preprocessing and understanding system
        merged_entities = {}
        if preprocessing_result and preprocessing_result.get('entities'):
            merged_entities.update(preprocessing_result['entities'])
        if multi_intent and multi_intent.extracted_entities:
            merged_entities.update(multi_intent.extracted_entities)
        
        return {
            'success': True,
            'intent': primary_intent,
            'confidence': confidence,
            'entities': merged_entities,
            'corrections': preprocessing_result.get('corrections', []) if preprocessing_result else [],
            'normalized_query': preprocessed_query.lower().strip(),
            'original_query': user_input,
            'detailed_result': result,
            'intent_classifier_result': intent_result,
            'preprocessing_stats': preprocessing_result.get('statistics') if preprocessing_result else None
        }
    except Exception as e:
        logger.error(f"Error in process_enhanced_query: {e}")
        # Fall back to intent classifier only
        entities = preprocessing_result.get('entities', {}) if preprocessing_result else {}
        corrections = preprocessing_result.get('corrections', []) if preprocessing_result else []
        preprocessing_stats = preprocessing_result.get('statistics') if preprocessing_result else None
        
        if intent_result and intent_result['confidence'] >= 0.6:
            return {
                'success': True,
                'intent': intent_result['intent'],
                'confidence': intent_result['confidence'],
                'entities': entities,
                'corrections': corrections,
                'normalized_query': preprocessed_query.lower().strip(),
                'original_query': user_input,
                'classifier_used': f"neural_{intent_result['method']}_fallback",
                'preprocessing_stats': preprocessing_stats
            }
        else:
            return {
                'success': False,
                'intent': 'general_info',
                'confidence': 0.3,
                'entities': entities,
                'corrections': corrections,
                'normalized_query': preprocessed_query.lower().strip(),
                'original_query': user_input,
                'preprocessing_stats': preprocessing_stats
            }

def generate_sample_hidden_gems(area: str, language: str = 'en') -> List[Dict[str, Any]]:
    """Generate sample hidden gems for a given area"""
    # Sample hidden gems data based on area
    gems_data = {
        'sultanahmet': [
            {
                'name': 'Historic Cistern Coffee',
                'description': 'Hidden coffee shop built into ancient cistern walls',
                'location': 'Near Basilica Cistern',
                'type': 'cafe',
                'authenticity_score': 9.2,
                'local_rating': 4.8,
                'price_range': '₺₺'
            },
            {
                'name': 'Artisan Carpet Workshop',
                'description': 'Traditional carpet weaving workshop open to visitors',
                'location': 'Behind Blue Mosque',
                'type': 'cultural',
                'authenticity_score': 9.5,
                'local_rating': 4.9,
                'price_range': 'Free to visit'
            }
        ],
        'beyoglu': [
            {
                'name': 'Rooftop Garden Cafe',
                'description': 'Secret garden cafe with Bosphorus views',
                'location': 'Hidden in Galata backstreets',
                'type': 'cafe',
                'authenticity_score': 8.8,
                'local_rating': 4.7,
                'price_range': '₺₺₺'
            },
            {
                'name': 'Underground Jazz Club',
                'description': 'Intimate jazz venue in historic building basement',
                'location': 'Near Istiklal Avenue',
                'type': 'entertainment',
                'authenticity_score': 9.0,
                'local_rating': 4.8,
                'price_range': '₺₺'
            }
        ]
    }
    
    return gems_data.get(area.lower(), [
        {
            'name': 'Local Discovery',
            'description': f'Authentic local experience in {area}',
            'location': area,
            'type': 'cultural',
            'authenticity_score': 8.5,
            'local_rating': 4.5,
            'price_range': '₺₺'
        }
    ])

def generate_sample_localized_tips(location: str, language: str = 'en') -> List[Dict[str, Any]]:
    """Generate sample localized tips for a location"""
    tips_data = {
        'sultanahmet': [
            {
                'tip': 'Visit early morning to avoid crowds at major attractions',
                'category': 'timing',
                'usefulness_score': 9.1,
                'local_insight': True
            },
            {
                'tip': 'Small restaurants behind the mosque serve authentic food',
                'category': 'dining',
                'usefulness_score': 8.8,
                'local_insight': True
            }
        ],
        'beyoglu': [
            {
                'tip': 'Take the historic tunnel from Karakoy to avoid the steep walk',
                'category': 'transportation',
                'usefulness_score': 9.0,
                'local_insight': True
            },
            {
                'tip': 'Best nightlife starts after 22:00 on weekends',
                'category': 'entertainment',
                'usefulness_score': 8.5,
                'local_insight': True
            }
        ]
    }
    
    return tips_data.get(location.lower(), [
        {
            'tip': f'Explore local neighborhoods in {location} for authentic experiences',
            'category': 'general',
            'usefulness_score': 8.0,
            'local_insight': True
        }
    ])

# --- Third-Party Imports ---
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException, status, Body, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from thefuzz import fuzz, process

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
    location_context: Optional[Dict[str, Any]] = Field(None, description="Location context")

class RouteRequest(BaseModel):
    """Request model for route planning"""
    attractions: List[str] = Field(..., description="List of attractions to visit")
    start_location: Optional[Dict[str, float]] = Field(None, description="Starting location")
    transport_mode: Optional[str] = Field("walking", description="Transportation mode")
    duration_hours: Optional[int] = Field(8, description="Available hours for the route")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

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
    from models import Base, Restaurant, Museum, Place, UserFeedback, ChatSession, BlogPost, BlogComment, ChatHistory
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
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="AI Istanbul Backend",
    description="Intelligent Istanbul travel assistant with live data integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

print("✅ FastAPI app initialized successfully")

# Initialize Location Intent Detector
location_detector = None
if LOCATION_INTENT_AVAILABLE:
    try:
        location_detector = LocationIntentDetector()
        print("✅ Location Intent Detector initialized successfully")
    except Exception as e:
        print(f"⚠️ Failed to initialize Location Intent Detector: {e}")
        LOCATION_INTENT_AVAILABLE = False

# Initialize Advanced Understanding System
advanced_understanding = None
if ADVANCED_UNDERSTANDING_AVAILABLE:
    try:
        # Initialize Redis client if available
        redis_client = None
        if REDIS_AVAILABLE:
            try:
                redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=int(os.getenv('REDIS_DB', 2)),  # Use separate DB for context memory
                    decode_responses=True
                )
                # Test connection
                redis_client.ping()
                print("✅ Redis connection established for context memory")
            except Exception as e:
                print(f"⚠️ Redis not available for context memory: {e}")
                redis_client = None
        
        # Initialize the Advanced Understanding System
        advanced_understanding = AdvancedUnderstandingSystem(redis_client=redis_client)
        print("✅ Advanced Understanding System initialized successfully")
        print("  🧠 Semantic Similarity Engine: Ready")
        print("  🧠 Enhanced Context Memory: Ready")
        print("  🎯 Multi-Intent Query Handler: Ready")
        
        # Integrate with Istanbul Daily Talk AI System
        if ISTANBUL_DAILY_TALK_AVAILABLE and istanbul_daily_talk_ai:
            istanbul_daily_talk_ai.multi_intent_handler = advanced_understanding.multi_intent_handler
            print("✅ Multi-Intent Query Handler integrated with Istanbul Daily Talk AI")
    except Exception as e:
        print(f"⚠️ Failed to initialize Advanced Understanding System: {e}")
        ADVANCED_UNDERSTANDING_AVAILABLE = False
        advanced_understanding = None

# Initialize Comprehensive ML/DL Integration System
comprehensive_ml_system = None
if COMPREHENSIVE_ML_AVAILABLE:
    try:
        comprehensive_ml_system = ComprehensiveMLDLIntegration()
        print("✅ Comprehensive ML/DL Integration System initialized successfully")
        print("  🚀 ML Enhancement Systems: Ready")
        print("  🧠 Typo Correction: Ready")
        print("  ☀️ Weather Advisor: Ready")
        print("  🗺️ Route Optimizer: Ready")
        print("  🎭 Event Predictor: Ready")
        print("  🏘️ Neighborhood Matcher: Ready")
        
        # Integrate with Advanced Understanding System if available
        if advanced_understanding and hasattr(advanced_understanding, 'multi_intent_handler'):
            # The multi_intent_handler already has comprehensive_ml_system initialized
            print("✅ Comprehensive ML/DL Integration connected to Multi-Intent Handler")
    except Exception as e:
        print(f"⚠️ Failed to initialize Comprehensive ML/DL Integration System: {e}")
        COMPREHENSIVE_ML_AVAILABLE = False
        comprehensive_ml_system = None

# Initialize Lightweight Deep Learning System
deep_learning_system = None
if DEEP_LEARNING_AVAILABLE:
    try:
        deep_learning_system = create_lightweight_deep_learning_system()
        print("✅ Lightweight Deep Learning System initialized successfully")
        print("  🧠 Intent Classification: Ready")
        print("  📈 Learning Enhancement: Ready")
        
        # Integrate with Advanced Understanding System if available
        if advanced_understanding and hasattr(advanced_understanding, 'multi_intent_handler'):
            # The multi_intent_handler already has deep_learning_system initialized
            print("✅ Deep Learning System connected to Multi-Intent Handler")
    except Exception as e:
        print(f"⚠️ Failed to initialize Lightweight Deep Learning System: {e}")
        DEEP_LEARNING_AVAILABLE = False
        deep_learning_system = None

# Initialize Caching Systems
ml_cache = None
edge_cache = None

if ML_CACHE_AVAILABLE:
    try:
        ml_cache = get_ml_cache()
        print("✅ ML Result Cache initialized successfully")
        print(f"  📊 Cache entries: {len(ml_cache.memory_cache)}")
        print(f"  💾 Cache size: {ml_cache._get_cache_size_mb():.2f} MB")
    except Exception as e:
        print(f"⚠️ Failed to initialize ML Result Cache: {e}")
        ML_CACHE_AVAILABLE = False

if EDGE_CACHE_AVAILABLE:
    try:
        edge_cache = get_edge_cache()
        print("✅ Edge Cache Manager initialized successfully")
        print(f"  🌐 Cache entries: {len(edge_cache.cache_entries)}")
        
        # Refresh static data caches
        refresh_results = edge_cache.refresh_all_static_data()
        successful_refreshes = sum(1 for result in refresh_results.values() if result)
        print(f"  🔄 Refreshed {successful_refreshes}/{len(refresh_results)} static data caches")
        
    except Exception as e:
        print(f"⚠️ Failed to initialize Edge Cache Manager: {e}")
        EDGE_CACHE_AVAILABLE = False

# Integration with Enhanced AI System
try:
    from istanbul_ai_system_enhancement import EnhancedIstanbulAISystem
    enhanced_ai_system = EnhancedIstanbulAISystem()
    ENHANCED_FEATURES_AVAILABLE = True
    logger.info("🚀 Enhanced Istanbul AI System integrated successfully!")
except ImportError as e:
    logger.warning(f"Enhanced features not available: {e}")
    ENHANCED_FEATURES_AVAILABLE = False
    enhanced_ai_system = None

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3001",
        "http://localhost:3002", 
        "http://127.0.0.1:3002",
        "http://localhost:3003", 
        "http://127.0.0.1:3003",
        "http://localhost:5173",  # Vite default port
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

print("✅ CORS middleware configured")

# Add security headers middleware for production
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Essential security headers for production
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response

print("✅ Security headers middleware configured")

# --- Rate Limiting Removed ---
# Rate limiting has been completely removed for unrestricted testing
limiter = None
print("✅ Rate limiting completely removed for unrestricted testing")

# --- Optional Enhancement Systems Initialization ---
# Initialize Optional Enhancement Systems
hybrid_search = None
personalization_engine = None
mini_nlp = None
OPTIONAL_ENHANCEMENTS_ENABLED = False

try:
    from hybrid_search_system import HybridSearchSystem
    from lightweight_personalization_engine import LightweightPersonalizationEngine
    from mini_nlp_modules import MiniNLPProcessor
    OPTIONAL_ENHANCEMENTS_ENABLED = True
    print("✅ Optional enhancement systems loaded successfully")
    
    # Initialize Hybrid Search System
    hybrid_search = HybridSearchSystem()
    print("✅ Hybrid Search System initialized")
    
    # Initialize Personalization Engine  
    personalization_engine = LightweightPersonalizationEngine()
    print("✅ Personalization Engine initialized")
    
    # Initialize Mini NLP Modules
    mini_nlp = MiniNLPProcessor()
    print("✅ Mini NLP Modules initialized")
    
    print("🚀 All optional enhancement systems ready!")
    
except ImportError as e:
    print(f"⚠️ Optional enhancement systems not available: {e}")
    OPTIONAL_ENHANCEMENTS_ENABLED = False
    hybrid_search = None
    personalization_engine = None  
    mini_nlp = None
except Exception as e:
    print(f"⚠️ Optional enhancement systems initialization failed: {e}")
    OPTIONAL_ENHANCEMENTS_ENABLED = False
    hybrid_search = None
    personalization_engine = None
    mini_nlp = None

print(f"Optional Enhancement Systems Status: {'✅ ENABLED' if OPTIONAL_ENHANCEMENTS_ENABLED else '❌ DISABLED'}")

# (Removed duplicate project imports and load_dotenv)

def clean_text_formatting(text):
    """Enhanced text formatting for better readability while preserving structure"""
    if not text:
        return text
    
    # Remove excessive emojis but keep a few for context
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"  # supplemental symbols
        u"\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
        u"\U00002600-\U000026FF"  # miscellaneous symbols
        u"\U00002700-\U000027BF"  # dingbats
        u"\U0001F780-\U0001F7FF"  # geometric shapes extended
        u"\U0001F800-\U0001F8FF"  # supplemental arrows-c
        "]+", flags=re.UNICODE)
    
    # Count emojis and only remove if excessive (more than 3)
    emoji_count = len(emoji_pattern.findall(text))
    if emoji_count > 3:
        text = emoji_pattern.sub(r'', text)
    
    # PHASE 1: Remove explicit pricing amounts (all formats) - ENHANCED
    text = re.sub(r'\$\d+[\d.,]*', '', text)      # $20, $15.50
    text = re.sub(r'€\d+[\d.,]*', '', text)       # €20, €15.50
    text = re.sub(r'₺\d+[\d.,]*', '', text)       # ₺20, ₺15.50
    text = re.sub(r'\d+₺', '', text)              # 50₺
    text = re.sub(r'\d+\s*(?:\$|€|₺)', '', text)  # 20$, 50 €
    text = re.sub(r'(?:\$|€|₺)\s*\d+[\d.,]*', '', text)  # $ 20, € 15.50
    
    # Additional pricing patterns
    text = re.sub(r'£\d+[\d.,]*', '', text)       # £20, £15.50
    text = re.sub(r'\d+£', '', text)              # 50£
    text = re.sub(r'(?:USD|EUR|GBP|TRY|TL)\s*\d+[\d.,]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(?:USD|EUR|GBP|TRY|TL)', '', text, flags=re.IGNORECASE)
    
    # PHASE 2: Remove pricing words and phrases - ENHANCED
    text = re.sub(r'\d+\s*(?:lira|euro|euros|dollar|dollars|pound|pounds|tl|usd|eur|gbp|try)s?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:turkish\s+)?lira\s*\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:around|about|approximately|roughly)\s+\d+\s*(?:lira|euro|euros|dollar|dollars)', '', text, flags=re.IGNORECASE)
    
    # PHASE 3: Remove cost-related phrases with amounts - ENHANCED
    cost_patterns = [
        r'(?:costs?|prices?|fees?)\s*:?\s*(?:around\s+|about\s+|approximately\s+|roughly\s+)?\$?\€?₺?£?\d+[\d.,]*',
        r'(?:entrance|admission|ticket|entry)\s*(?:cost|price|fee)s?\s*:?\s*\$?\€?₺?£?\d+',
        r'(?:starting|starts)\s+(?:from|at)\s+\$?\€?₺?£?\d+',
        r'(?:only|just)\s+\$?\€?₺?£?\d+[\d.,]*',
        r'(?:per\s+person|each|pp)\s*:?\s*\$?\€?₺?£?\d+',
        r'\$?\€?₺?£?\d+[\d.,]*\s*(?:per\s+person|each|pp)',
        r'(?:budget|spend|pay)\s*:?\s*(?:around\s+|about\s+)?\$?\€?₺?£?\d+[\d.,]*',
        r'(?:between|from)\s+\$?\€?₺?£?\d+\s*(?:and|to|-|–)\s*\$?\€?₺?£?\d+',
        r'(?:range|varies)\s+(?:from|between)\s+\$?\€?₺?£?\d+',
    ]
    
    for pattern in cost_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # PHASE 4: Remove money emojis and pricing symbols - ENHANCED
    text = re.sub(r'💰|💵|💴|💶|💷|💸', '', text)
    text = re.sub(r'[\$€₺£¥₹₽₴₦₱₩₪₨₡₵₼₢₨₹₿]', '', text)
    
    # Remove pricing codes
    text = re.sub(r'\b(?:USD|EUR|GBP|TRY|TL|JPY|CHF|CAD|AUD)\b', '', text, flags=re.IGNORECASE)
    
    # PHASE 5: Remove ALL markdown formatting for clean responses
    # Remove **bold** and *italic* formatting completely
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold** but keep content
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic* but keep content
    
    # Remove any remaining asterisks that might be left over
    text = re.sub(r'\*+', '', text)  # Remove any standalone asterisks
    
    # Clean up any double spacing that might result from asterisk removal
    text = re.sub(r'  +', ' ', text)  # Multiple spaces to single space
    
    # Improve bullet points and structure
    text = re.sub(r'^[\s]*-\s*', '• ', text, flags=re.MULTILINE) 
    text = re.sub(r'([.!?])\s*\n\s*([A-Z])', r'\1\n\n\2', text)  # Add space between paragraphs
    
    # Clean up spacing while preserving structure
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:  # Only process non-empty lines
            line = re.sub(r'\s+', ' ', line)  # Multiple spaces to single space
            line = re.sub(r'\s*[:;,]\s*[:;,]+', ',', line)  # Clean up punctuation
            cleaned_lines.append(line)
        else:
            cleaned_lines.append('')  # Preserve empty lines for structure
    
    text = '\n'.join(cleaned_lines)
    
    # Final cleanup
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # No more than double line breaks
    text = re.sub(r'^\s+|\s+$', '', text)  # Remove leading/trailing whitespace
    
    return text

def sanitize_user_input(user_input: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    if not user_input:
        return ""
    
    # Limit input length
    max_length = 1000
    if len(user_input) > max_length:
        user_input = user_input[:max_length]
    
    # Remove potentially dangerous characters
    user_input = html.escape(user_input)
    
    # Remove SQL injection patterns
    dangerous_patterns = [
        r'union\s+select', r'drop\s+table', r'delete\s+from', r'insert\s+into',
        r'update\s+set', r'create\s+table', r'alter\s+table', r'exec\s*\(',
        r'script\s*>', r'javascript:', r'vbscript:', r'on\w+\s*=', r'<\s*script'
    ]
    
    for pattern in dangerous_patterns:
        user_input = re.sub(pattern, '', user_input, flags=re.IGNORECASE)
    
    return user_input.strip()

async def get_istanbul_ai_response_with_quality(user_input: str, session_id: str, user_ip: Optional[str] = None, location_context: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """SIMPLIFIED: Generate response using Istanbul Daily Talk AI (Primary System)"""
    try:
        # Sanitize input
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        print(f"🏛️ Using Istanbul Daily Talk AI (SIMPLIFIED) for session: {session_id}")
        
        # Use Istanbul Daily Talk AI as the primary system - it handles everything internally
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            try:
                # Let Istanbul Daily Talk AI handle all the complex processing internally
                ai_response = istanbul_daily_talk_ai.process_message(user_input, session_id)
                
                if ai_response and len(ai_response) > 20:
                    print(f"✅ Istanbul Daily Talk AI response: {len(ai_response)} characters")
                    
                    # Apply post-processing cleanup
                    ai_response = post_ai_cleanup(ai_response)
                    
                    return {
                        'success': True,
                        'response': ai_response,
                        'session_id': session_id,
                        'has_context': True,
                        'uses_neural_ranking': False,
                        'system_type': 'istanbul_daily_talk_ai',
                        'quality_assessment': {
                            'overall_score': 90,
                            'confidence': 0.85,
                            'used_fallback': False,
                            'processing_time': 0.2,
                            'system_version': 'istanbul_daily_talk_ai_integrated'
                        }
                    }
                else:
                    print("⚠️ Istanbul Daily Talk AI response insufficient")
                    return None
                    
            except Exception as e:
                print(f"⚠️ Istanbul Daily Talk AI error: {e}")
                return None
        else:
            print("❌ Istanbul Daily Talk AI not available")
            return None
            
    except Exception as e:
        print(f"❌ Error in simplified AI system: {str(e)}")
        return None

async def get_istanbul_ai_response(user_input: str, session_id: str, user_ip: Optional[str] = None) -> Optional[str]:
    """Generate response using Ultra-Specialized Istanbul AI (Rule-Based) - Simple version"""
    try:
        # Sanitize input first
        user_input = sanitize_user_input(user_input)
        if not user_input:
            return None
        
        # 🧠 ENHANCED QUERY UNDERSTANDING INTEGRATION (Simple version)
        query_analysis = {}
        if ENHANCED_QUERY_UNDERSTANDING_ENABLED:
            try:
                query_analysis = process_enhanced_query(user_input, session_id)
                if query_analysis.get('success'):
                    print(f"🧠 Simple Query Analysis - Intent: {query_analysis['intent']} "
                          f"(confidence: {query_analysis['confidence']:.2f})")
            except Exception as e:
                print(f"⚠️ Enhanced query understanding error (simple): {e}")
        
        # Prepare user context with query analysis
        user_context = {
            'session_id': session_id,
            'user_ip': user_ip,
            'timestamp': datetime.now().isoformat(),
            'query_analysis': query_analysis,
            'detected_intent': query_analysis.get('intent', 'general_info'),
            'query_confidence': query_analysis.get('confidence', 0.3),
            'extracted_entities': query_analysis.get('entities', {}),
            'normalized_query': query_analysis.get('normalized_query', user_input.lower().strip())
        }
        
        # �🏛️ PRIMARY: Use Istanbul Daily Talk AI with its enhanced query detection (Simple version)
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            print(f"🏛️ Using Istanbul Daily Talk AI (PRIMARY) with enhanced query detection for session: {session_id}")
            try:
                # Process with Istanbul Daily Talk AI - it has its own query detection
                ai_response = istanbul_daily_talk_ai.process_message(user_input, session_id)
                
                # Use Istanbul Daily Talk AI's internal query detection for restaurant enhancement
                if RESTAURANT_SERVICE_ENABLED:
                    try:
                        # Use Istanbul Daily Talk AI's enhanced intent classification (simpler approach)
                        entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(user_input)
                        
                        # Use the enhanced intent classification method directly
                        enhanced_intent = istanbul_daily_talk_ai._enhance_intent_classification(user_input)
                        
                        # Check restaurant-specific patterns
                        is_restaurant_pattern = istanbul_daily_talk_ai._is_restaurant_query(user_input)
                        
                        # Determine final intent
                        if is_restaurant_pattern or enhanced_intent in ['restaurant_query']:
                            intent = 'restaurant_query'
                        elif enhanced_intent == 'attraction_query':
                            intent = 'attraction_query'
                        elif enhanced_intent == 'transportation_query':
                            intent = 'transportation_query'
                        else:
                            intent = enhanced_intent if enhanced_intent != 'general_conversation' else 'general_info'
                        
                        # 🍽️ ENHANCE RESTAURANT RESPONSES WITH DATABASE DATA
                        restaurant_intents = ['restaurant_query', 'restaurant_recommendation', 'dining', 'food']
                        if intent in restaurant_intents and ai_response:
                            print(f"🍽️ Restaurant query detected by Istanbul Daily Talk AI - enhancing with database")
                            
                            # Extract search parameters from Istanbul Daily Talk AI's entities
                            district = entities.get('neighborhoods', [None])[0] if entities.get('neighborhoods') else None
                            cuisine = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
                            
                            # Search restaurant database
                            restaurants = restaurant_service.search_restaurants_compat(
                                district=district,
                                cuisine=cuisine,
                                limit=3
                            )
                            
                            if restaurants:
                                # Create enhanced response
                                database_response = restaurant_service.format_restaurant_response(restaurants)
                                # Combine responses for comprehensive answer
                                enhanced_response = f"{ai_response}\n\n{database_response}"
                                print(f"✅ Enhanced Istanbul Daily Talk AI response with {len(restaurants)} restaurants")
                                return enhanced_response
                            else:
                                print("⚠️ No restaurants found in database, using AI response only")
                                
                    except Exception as e:
                        print(f"⚠️ Restaurant enhancement error (simple): {e}")
                        # Continue with original AI response
                
                # Return Istanbul Daily Talk AI response
                if ai_response and len(ai_response) > 30:  # Simple threshold
                    print(f"✅ Istanbul Daily Talk AI (simple) response: {len(ai_response)} characters")
                    return ai_response
                else:
                    print("⚠️ Istanbul Daily Talk AI response too short, using fallback")
                    # Fall through to fallback system
            except Exception as e:
                print(f"⚠️ Istanbul Daily Talk AI error (simple): {e}")
                # Fall through to fallback system
        
        # FALLBACK: Use Ultra-Specialized Istanbul AI if Daily Talk AI fails or unavailable
        if ULTRA_ISTANBUL_AI_AVAILABLE and istanbul_ai_system:
            print(f"🏛️ Using Ultra-Specialized Istanbul AI (FALLBACK) for session: {session_id}")
            result = istanbul_ai_system.process_istanbul_query(user_input, user_context)
        else:
            print("❌ No AI systems available")
            return None
        
        if result.get('success'):
            ai_response = result['response']
            
            # 🍽️ RESTAURANT DATA INTEGRATION (Simple version) - Using Istanbul Daily Talk AI's detection
            # Note: This is fallback integration for the Ultra-Specialized system
            restaurant_intents = ['restaurant_query', 'restaurant_recommendation', 'dining', 'food']
            detected_intent = 'general_info'
            
            # Try to use Istanbul Daily Talk AI's query detection even in fallback mode
            if ISTANBUL_DAILY_TALK_AVAILABLE:
                try:
                    entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(user_input)
                    
                    # Use enhanced intent classification (simpler, no context needed)
                    enhanced_intent = istanbul_daily_talk_ai._enhance_intent_classification(user_input)
                    is_restaurant_pattern = istanbul_daily_talk_ai._is_restaurant_query(user_input)
                    
                    # Determine final intent
                    if is_restaurant_pattern or enhanced_intent in ['restaurant_query']:
                        detected_intent = 'restaurant_query'
                    else:
                        detected_intent = enhanced_intent if enhanced_intent != 'general_conversation' else 'general_info'
                except Exception as e:
                    print(f"⚠️ Could not use Istanbul Daily Talk AI detection in fallback: {e}")
                    detected_intent = query_analysis.get('intent', 'general_info')
            else:
                detected_intent = query_analysis.get('intent', 'general_info')
            
            if RESTAURANT_SERVICE_ENABLED and detected_intent in restaurant_intents:
                try:
                    # Simple restaurant search based on detected entities
                    entities = query_analysis.get('entities', {})
                    district = entities.get('districts', [None])[0] if entities.get('districts') else None
                    cuisine = entities.get('cuisines', [None])[0] if entities.get('cuisines') else None
                    
                    restaurants = restaurant_service.search_restaurants_compat(
                        district=district,
                        cuisine=cuisine,
                        limit=2
                    )
                    
                    if restaurants:
                        ai_response = restaurant_service.format_restaurant_response(restaurants)
                        print(f"✅ Simple restaurant integration: {len(restaurants)} restaurants")
                        
                except Exception as e:
                    print(f"⚠️ Simple restaurant integration error: {e}")
            
            print(f"✅ Ultra-Specialized AI response (simple) - Session: {session_id}, "
                  f"Confidence: {result.get('confidence', 0.7):.2f}, "
                  f"System: {result.get('system_version', 'ultra_specialized')}")
            
            # Apply post-processing cleanup
            ai_response = post_ai_cleanup(ai_response)
            
            # Apply restaurant response formatting
            try:
                from restaurant_response_formatter import format_restaurant_response
                ai_response = format_restaurant_response(ai_response, user_input)
            except ImportError:
                pass  # Formatter not available, continue without it
                
            return ai_response
        else:
            print(f"❌ Ultra-Specialized Istanbul AI failed: {result.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ Error in Ultra-Specialized Istanbul AI system: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_fallback_response(user_input: str, context: Optional[Dict] = None) -> str:
    """Create a fallback response when AI systems fail"""
    try:
        # Basic fallback responses based on keywords
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['restaurant', 'food', 'eat', 'dining', 'meal']):
            return """I'd be happy to help you find restaurants in Istanbul! Here are some popular areas for dining:

• **Sultanahmet** - Traditional Ottoman cuisine near historic sites
• **Beyoğlu** - International restaurants and trendy cafes
• **Karaköy** - Waterfront dining with Bosphorus views
• **Kadıköy** - Local favorites and street food

For specific restaurant recommendations, please let me know:
- Which district interests you?
- What type of cuisine do you prefer?
- What's your budget range?"""

        elif any(word in user_lower for word in ['museum', 'history', 'culture', 'art']):
            return """Istanbul has amazing museums and cultural sites! Here are the must-visit ones:

• **Hagia Sophia** - Iconic Byzantine and Ottoman architecture
• **Topkapi Palace** - Former Ottoman imperial palace
• **Istanbul Archaeological Museum** - Ancient artifacts and treasures
• **Istanbul Modern** - Contemporary Turkish and international art

Would you like more details about any specific museum or cultural site?"""

        elif any(word in user_lower for word in ['transport', 'metro', 'bus', 'travel', 'getting around']):
            return """Getting around Istanbul is easy with multiple transport options:

• **Metro** - Fast and efficient for major districts
• **Tram** - Great for tourist areas like Sultanahmet
• **Ferry** - Scenic Bosphorus crossings
• **Bus** - Extensive network covering the whole city
• **Istanbulkart** - Single card for all public transport

Need help planning a specific route?"""

        else:
            return """Welcome to Istanbul! I'm here to help you discover this amazing city.

I can assist you with:
• Restaurant recommendations and dining
• Museums and cultural attractions  
• Transportation and getting around
• Historical sites and hidden gems
• Local tips and insider knowledge

What would you like to know about Istanbul?"""
            
    except Exception as e:
        print(f"Error in create_fallback_response: {e}")
        return "I'm here to help you explore Istanbul! Please let me know what you'd like to discover about this amazing city."

# =============================
# ISTANBUL DAILY TALK SYSTEM API ENDPOINTS  
# =============================

# Request/Response Models for Istanbul Daily Talk System
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lng}")

class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    intent: Optional[str] = Field(None, description="Detected intent")
    confidence: Optional[float] = Field(None, description="Response confidence")
    suggestions: Optional[List[str]] = Field(None, description="Follow-up suggestions")
    detected_location: Optional[Dict[str, Any]] = Field(None, description="Detected user location information")
    nearby_events: Optional[List[Dict[str, Any]]] = Field(None, description="Events near detected location")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional response metadata (navigation, route data, etc.)")

class RouteRequest(BaseModel):
    message: str = Field(..., description="Route planning request")
    start_location: Dict[str, float] = Field(..., description="Starting point {lat, lng}")
    end_location: Optional[Dict[str, float]] = Field(None, description="End point {lat, lng}")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Route preferences")
    session_id: Optional[str] = Field(None, description="Session ID")

class TransportRequest(BaseModel):
    from_location: Dict[str, float] = Field(..., description="Starting location {lat, lng}")
    to_location: Dict[str, float] = Field(..., description="Destination {lat, lng}")
    transport_mode: Optional[str] = Field("public", description="Transport mode preference")
    time_preference: Optional[str] = Field("now", description="Time preference")

class MuseumRequest(BaseModel):
    query: str = Field(..., description="Museum query")
    location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lng}")
    interests: Optional[List[str]] = Field(None, description="User interests")

class MuseumResponse(BaseModel):
    museums: List[Dict[str, Any]] = Field(..., description="Museum recommendations")
    personalized_tips: List[str] = Field(..., description="Personalized museum tips")
    opening_hours: Dict[str, str] = Field(..., description="Current opening hours")
    ticket_info: Dict[str, Any] = Field(..., description="Ticket information")

# =============================
# GPS-BASED ROUTE PLANNING ENDPOINTS
# =============================

@app.post("/api/route/gps-plan", response_model=RouteResponse, tags=["GPS Route Planning"])
async def plan_route_from_gps_location(request: GPSRouteRequest):
    """
    Generate intelligent route plan based on user's GPS location
    Finds nearby attractions and creates optimized route
    """
    try:
        print(f"📍 GPS-based route planning request from location: {request.user_location}")
        
        if not ISTANBUL_DAILY_TALK_AVAILABLE:
            raise HTTPException(status_code=503, detail="GPS route planning service unavailable")
        
        # Generate session ID for personalization
        session_id = request.session_id or f"gps_route_{uuid.uuid4().hex[:8]}"
        
        # Get user profile for personalization
        user_profile = istanbul_daily_talk_ai.get_or_create_user_profile(session_id)
        
        # Update user profile with interests if provided
        if request.interests:
            user_profile.interests.extend([interest for interest in request.interests if interest not in user_profile.interests])
        
        # Create location-aware query
        lat, lng = request.user_location["lat"], request.user_location["lng"]
        
        # Generate contextual route planning query based on location and interests
        location_query_parts = [
            f"I'm currently at GPS location {lat:.4f}, {lng:.4f} in Istanbul."
        ]
        
        if request.interests:
            interests_str = ", ".join(request.interests)
            location_query_parts.append(f"I'm interested in {interests_str}.")
        
        if request.duration_hours:
            location_query_parts.append(f"I have {request.duration_hours} hours available.")
        
        if request.radius_km:
            location_query_parts.append(f"I prefer to stay within {request.radius_km}km of my current location.")
        
        location_query_parts.append("Please create an optimized route for me.")
        
        route_query = " ".join(location_query_parts)
        
        # Create context with GPS location
        from istanbul_daily_talk_system_modular import ConversationContext
        context = ConversationContext(
            session_id=session_id,
            user_profile=user_profile
        )
        context.context_memory = {
            "user_location": request.user_location,
            "radius_km": request.radius_km,
            "duration_hours": request.duration_hours,
            "transport_mode": request.transport_mode,
            "route_style": request.route_style
        }
        
        # === USE ENHANCED GPS ROUTE PLANNER FROM MAIN SYSTEM ===
        route_result = None
        if hasattr(istanbul_daily_talk_ai, 'gps_route_planner') and istanbul_daily_talk_ai.gps_route_planner:
            print("🗺️ Using Enhanced GPS Route Planner from Main System...")
            try:
                # Use the enhanced GPS route planner with fallback location detection
                route_result = istanbul_daily_talk_ai.gps_route_planner.plan_route(
                    user_location=request.user_location,
                    user_preferences={
                        'interests': request.interests or [],
                        'duration_hours': request.duration_hours,
                        'radius_km': request.radius_km,
                        'transport_mode': request.transport_mode,
                        'route_style': request.route_style
                    },
                    user_context={
                        'session_id': session_id,
                        'user_profile': user_profile.__dict__ if hasattr(user_profile, '__dict__') else {}
                    }
                )
                
                if route_result and route_result.get('success', True):
                    print(f"✅ Enhanced GPS route plan generated with {len(route_result.get('waypoints', []))} waypoints")
                    route_response = route_result.get('formatted_plan', route_result.get('description', ''))
                else:
                    print(f"⚠️ Enhanced GPS planner returned error, falling back...")
                    route_response = None
                    
            except Exception as e:
                print(f"⚠️ Enhanced GPS planner error: {e}, falling back...")
                route_result = None
                route_response = None
        
        # Fallback to traditional route planning if enhanced planner not available or failed
        if not route_result:
            if hasattr(istanbul_daily_talk_ai, 'handle_route_planning_query'):
                route_response = istanbul_daily_talk_ai.handle_route_planning_query(
                    route_query, user_profile, context, datetime.now()
                )
            else:
                # Final fallback to regular message processing with location context
                route_response = istanbul_daily_talk_ai.process_message(route_query, session_id)
        
        # Extract nearby attractions (if route maker service is available)
        nearby_attractions = []
        if hasattr(istanbul_daily_talk_ai, 'route_maker') and istanbul_daily_talk_ai.route_maker:
            try:
                # Get nearby attractions using route maker service
                from services.route_maker_service import get_route_maker
                route_maker = get_route_maker()
                
                # This would need to be implemented in the route maker service
                # nearby_attractions = route_maker.get_attractions_near_point(lat, lng, request.radius_km or 5.0)
                
            except Exception as e:
                print(f"⚠️ Could not get nearby attractions: {e}")
        
        # Create GPS-aware waypoints from enhanced route result or fallback parsing
        waypoints = []
        estimated_distance = 0
        estimated_duration = 0
        
        if route_result and route_result.get('waypoints'):
            # Use waypoints from enhanced GPS route planner
            for waypoint in route_result['waypoints']:
                waypoints.append({
                    "order": waypoint.get('order', len(waypoints) + 1),
                    "name": waypoint.get('name', 'Unknown Location'),
                    "description": waypoint.get('description', ''),
                    "estimated_time": waypoint.get('estimated_time', '45-90 minutes'),
                    "distance_from_start": waypoint.get('distance_from_start', f"{len(waypoints) * 0.8:.1f} km"),
                    "lat": waypoint.get('lat'),
                    "lng": waypoint.get('lng')
                })
            
            estimated_distance = route_result.get('total_distance_km', len(waypoints) * 1.2)
            estimated_duration = route_result.get('total_duration_hours', request.duration_hours or min(len(waypoints) * 1.5, 8))
            
        elif route_response and "→" in route_response:
            # Fallback: Parse waypoints from text response
            places = [place.strip() for place in route_response.split("→")]
            for i, place in enumerate(places):
                waypoints.append({
                    "order": i + 1,
                    "name": place,
                    "description": f"Stop {i + 1}: {place}",
                    "estimated_time": "45-90 minutes",
                    "distance_from_start": f"{i * 0.8:.1f} km"  # Estimated
                })
            
            estimated_distance = len(waypoints) * 1.2  # Rough estimate
            estimated_duration = request.duration_hours or min(len(waypoints) * 1.5, 8)
        else:
            # No waypoints could be extracted
            estimated_distance = 0
            estimated_duration = request.duration_hours or 4
        
        # Build comprehensive route data from enhanced result or fallback
        route_data = {
            "description": route_response or route_result.get('description', 'Custom GPS-based route created'),
            "optimized": True,
            "algorithm": route_result.get('algorithm', "GPS-aware TSP with local optimization"),
            "start_point": request.user_location,
            "gps_based": True,
            "radius_km": request.radius_km or 5.0,
            "interests_considered": request.interests or [],
            "transport_mode": request.transport_mode,
            "nearby_museums": route_result.get('nearby_museums', []) if route_result else [],
            "local_tips": route_result.get('local_tips', []) if route_result else [],
            "district_transport_tips": route_result.get('district_transport_tips', []) if route_result else []
        }
        
        # Generate suggestions from enhanced result or create defaults
        suggestions = []
        if route_result and route_result.get('local_tips'):
            suggestions = route_result['local_tips'][:4]
        else:
            suggestions = [
                f"Route optimized for {request.radius_km or 5.0}km radius from your location",
                f"Estimated total time: {estimated_duration:.1f} hours",
                "Consider traffic and opening hours",
                "Download offline maps for better navigation"
            ]
        
        return RouteResponse(
            route=route_data,
            total_duration=f"{estimated_duration:.1f} hours",
            total_distance=f"{estimated_distance:.1f} km",
            waypoints=waypoints,
            suggestions=suggestions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ GPS route planning error: {e}")
        raise HTTPException(status_code=500, detail="GPS route planning failed")

@app.post("/api/nearby/attractions", response_model=LocationBasedRecommendationResponse, tags=["GPS Route Planning"]) 
async def get_nearby_attractions(request: NearbyAttractionsRequest):
    """
    Get attractions near user's GPS location
    Returns personalized recommendations based on location
    """
    try:
        print(f"📍 Nearby attractions request for location: {request.location}")
        
        lat, lng = request.location["lat"], request.location["lng"]
        radius = request.radius_km or 2.0
        
        # Use Istanbul Daily Talk AI to get location-aware recommendations
        location_query = f"What attractions and interesting places are near GPS coordinates {lat:.4f}, {lng:.4f} within {radius}km?"
        
        if request.categories:
            categories_str = ", ".join(request.categories)
            location_query += f" I'm particularly interested in {categories_str}."
        
        session_id = f"nearby_{uuid.uuid4().hex[:8]}"
        
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            # Get AI recommendations
            ai_response = istanbul_daily_talk_ai.process_message(location_query, session_id)
            
            # Extract location information using entity recognizer
            entities = istanbul_daily_talk_ai.entity_recognizer.extract_entities(location_query)
            
            # Parse response for structured data
            recommendations = []
            
            # Extract attractions from AI response (basic parsing)
            lines = ai_response.split('\n')
            for line in lines:
                line = line.strip()
                if line and ('•' in line or '-' in line or line.startswith(('1.', '2.', '3.'))):
                    # Extract attraction name and description
                    clean_line = re.sub(r'^[•\-\d\.\s]+', '', line)
                    if clean_line:
                        recommendations.append({
                            "name": clean_line.split('-')[0].strip() if '-' in clean_line else clean_line[:50],
                            "description": clean_line,
                            "estimated_distance": f"{radius/2:.1f} km",  # Rough estimate
                            "category": "attraction",
                            "confidence": 0.8
                        })
            
            # Limit results
            recommendations = recommendations[:request.limit or 10]
            
        else:
            # Fallback static recommendations
            recommendations = [
                {
                    "name": "Hagia Sophia",
                    "description": "Historic Byzantine and Ottoman monument",
                    "estimated_distance": "1.2 km",
                    "category": "historical",
                    "confidence": 0.9
                },
                {
                    "name": "Blue Mosque",
                    "description": "Famous Ottoman mosque with blue tiles",
                    "estimated_distance": "0.8 km", 
                    "category": "religious",
                    "confidence": 0.9
                }
            ]
        
        # Generate route suggestions
        suggested_routes = [
            {
                "name": "Quick Tour",
                "duration": "2-3 hours",
                "attractions": min(len(recommendations), 3),
                "description": "Visit top nearby attractions"
            },
            {
                "name": "Full Day Tour", 
                "duration": "6-8 hours",
                "attractions": min(len(recommendations), 6),
                "description": "Comprehensive exploration of the area"
            }
        ]
        
        return LocationBasedRecommendationResponse(
            recommendations=recommendations,
            user_location=request.location,
            distance_info={
                "search_radius": f"{radius} km",
                "total_found": str(len(recommendations)),
                "closest": recommendations[0]["estimated_distance"] if recommendations else "unknown"
            },
            suggested_routes=suggested_routes
        )
        
    except Exception as e:
        print(f"❌ Nearby attractions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get nearby attractions")

@app.post("/api/route/gps-optimize", response_model=RouteResponse, tags=["GPS Route Planning"])
async def optimize_route_from_gps(
    user_location: Dict[str, float] = Body(..., description="User's GPS location"),
    destinations: List[Dict[str, Any]] = Body(..., description="List of destinations to visit"),
    preferences: Optional[Dict[str, Any]] = Body(None, description="Route optimization preferences")
):
    """
    Optimize route order based on user's GPS location and destinations
    Uses TSP algorithm for optimal routing
    """
    try:
        print(f"🗺️ GPS route optimization from {user_location} to {len(destinations)} destinations")
        
        if not destinations:
            raise HTTPException(status_code=400, detail="No destinations provided")
        
        # Create route optimization query
        destination_names = [dest.get("name", "Unknown") for dest in destinations]
        destinations_str = ", ".join(destination_names)
        
        optimization_query = (
            f"I'm at GPS location {user_location['lat']:.4f}, {user_location['lon']:.4f} "
            f"and want to visit these places: {destinations_str}. "
            f"What's the most efficient route order?"
        )
        
        session_id = f"optimize_{uuid.uuid4().hex[:8]}"
        
        if ISTANBUL_DAILY_TALK_AVAILABLE:
            # Use Istanbul Daily Talk AI for route optimization
            optimization_response = istanbul_daily_talk_ai.process_message(optimization_query, session_id)
            
            # Try to extract optimized order from response
            optimized_waypoints = []
            if "→" in optimization_response:
                ordered_places = [place.strip() for place in optimization_response.split("→")]
                for i, place in enumerate(ordered_places):
                    # Find matching destination
                    matching_dest = None
                    for dest in destinations:
                        if dest.get("name", "").lower() in place.lower():
                            matching_dest = dest
                            break
                    
                    waypoint = {
                        "order": i + 1,
                        "name": place,
                        "description": matching_dest.get("description", f"Visit {place}") if matching_dest else f"Visit {place}",
                        "estimated_time": "60-90 minutes",
                        "distance_from_start": f"{matching_dest.get('distance_from_start', 0):.1f} km" if matching_dest else "Unknown",
                        "lat": matching_dest.get("location", {}).get("lat") if matching_dest else None,
                        "lng": matching_dest.get("location", {}).get("lng") if matching_dest else None
                    }
                    optimized_waypoints.append(waypoint)
            else:
                # Fallback: use original order
                for i, dest in enumerate(destinations):
                    optimized_waypoints.append({
                        "order": i + 1,
                        "name": dest.get("name", f"Destination {i+1}"),
                        "description": dest.get("description", ""),
                        "estimated_time": "60-90 minutes",
                        "distance_from_start": f"{dest.get('distance_from_start', 0):.1f} km" if dest else "Unknown",
                        "lat": dest.get("location", {}).get("lat") if dest else None,
                        "lng": dest.get("location", {}).get("lng") if dest else None
                    })
            
            route_data = {
                "description": optimization_response,
                "optimized": True,
                "algorithm": "GPS-aware TSP optimization",
                "start_point": user_location,
                "end_point": user_location,
                "gps_based": True
            }
            
            return RouteResponse(
                route=route_data,
                total_duration=f"{len(optimized_waypoints) * 1.5:.1f} hours",
                total_distance=f"{len(optimized_waypoints) * 1.8:.1f} km",
                waypoints=optimized_waypoints,
                suggestions=[
                    "Route optimized for minimum travel time",
                    "Consider traffic conditions during peak hours",
                    "Allow extra time for popular attractions",
                    "Check opening hours before visiting"
                ]
            )
        else:
            raise HTTPException(status_code=503, detail="Route optimization service unavailable")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GPS route optimization error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Route optimization error: {str(e)}")


# =============================
# AI CHAT ENDPOINTS (MAIN CHAT INTERFACE)
# =============================

@app.post("/ai/chat", response_model=ChatResponse, tags=["AI Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint for AI Istanbul assistant with RICH METADATA
    Returns comprehensive POI data, district info, cultural tips, and route suggestions
    """
    try:
        # Sanitize user input
        user_input = sanitize_user_input(request.message)
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
        user_id = request.user_id or session_id
        
        logger.info(f"💬 Chat request - Session: {session_id}, Query: '{user_input[:50]}...'")
        
        # Run query preprocessing pipeline
        query_analysis = {}
        if QUERY_PREPROCESSING_AVAILABLE and query_preprocessor:
            try:
                query_analysis = process_enhanced_query(user_input, session_id)
                if query_analysis.get('success'):
                    logger.info(f"🔧 Query preprocessed - Intent: {query_analysis['intent']} "
                              f"(confidence: {query_analysis['confidence']:.2f})")
                    if query_analysis.get('corrections'):
                        logger.info(f"✏️ Applied {len(query_analysis['corrections'])} corrections")
                    if query_analysis.get('entities'):
                        logger.info(f"🏷️ Extracted entities: {list(query_analysis['entities'].keys())}")
            except Exception as e:
                logger.warning(f"Query preprocessing error: {e}")
        
        # Initialize comprehensive metadata
        metadata = {}
        cultural_tips = []
        
        # Add preprocessing results to metadata
        if query_analysis:
            metadata['query_preprocessing'] = {
                'original_query': query_analysis.get('original_query', user_input),
                'processed_query': query_analysis.get('normalized_query', user_input),
                'corrections_applied': len(query_analysis.get('corrections', [])),
                'entities_extracted': list(query_analysis.get('entities', {}).keys()),
                'detected_intent': query_analysis.get('intent'),
                'confidence': query_analysis.get('confidence'),
                'statistics': query_analysis.get('preprocessing_stats')
            }
        
        # Use Istanbul Daily Talk AI if available
        if ISTANBUL_DAILY_TALK_AVAILABLE and istanbul_daily_talk_ai:
            try:
                # Process message with the AI system
                ai_response = istanbul_daily_talk_ai.process_message(user_input, user_id)
                
                # ===== 1. RICH POI DATA (Museums & Attractions) - Including Contemporary Art Spaces =====
                museum_attraction_keywords = [
                    'museum', 'attraction', 'visit', 'see', 'tour', 'hagia', 'topkapi', 'palace', 'mosque',
                    'art', 'contemporary', 'modern', 'gallery', 'exhibition', 'arter', 'salt', 'pera',
                    'istanbul modern', 'dirimart', 'pi artworks', 'mixer', 'elgiz', 'akbank sanat',
                    'borusan', 'art museum', 'sanat', 'galeri', 'sergi'
                ]
                if any(word in user_input.lower() for word in museum_attraction_keywords):
                    pois = []
                    
                    # Try to get museum data from the main system
                    if hasattr(istanbul_daily_talk_ai, 'search_museums') and istanbul_daily_talk_ai.museum_available:
                        try:
                            museums = istanbul_daily_talk_ai.search_museums(user_input)
                            if museums:
                                for m in museums[:5]:  # Top 5 museums
                                    poi = {
                                        'name': m.get('name', ''),
                                        'type': m.get('type', 'museum'),
                                        'category': m.get('category', 'Museum'),
                                        'coordinates': m.get('coordinates', [41.0082, 28.9784]),
                                        'description': m.get('description', '')[:200],
                                        'highlights': m.get('highlights', ['Beautiful architecture', 'Rich history']),
                                        'local_tips': m.get('local_tips', ['Visit early to avoid crowds', 'Photography allowed']),
                                        'opening_hours': m.get('opening_hours', '9:00 AM - 5:00 PM'),
                                        'entrance_fee': m.get('entrance_fee', 'Varies'),
                                        'best_time_to_visit': m.get('best_time_to_visit', 'Early morning or late afternoon'),
                                        'visit_duration': m.get('visit_duration', '1-2 hours'),
                                        'accessibility': m.get('accessibility', 'Wheelchair accessible'),
                                        'nearby_transport': m.get('nearby_transport', 'Tram T1 nearby'),
                                        'nearby_attractions': m.get('nearby_attractions', []),
                                        'insider_tips': m.get('insider_tips', []),
                                        'website': m.get('website', ''),
                                        'phone': m.get('phone', '')
                                    }
                                    pois.append(poi)
                                
                                metadata['pois'] = pois
                                logger.info(f"✅ Added {len(pois)} POIs with rich metadata from Museum System")
                        except Exception as e:
                            logger.warning(f"Museum data error: {e}")
                            import traceback
                            logger.warning(traceback.format_exc())
                    
                    # Add famous attractions with detailed data if museums not found
                    if not pois and 'sultanahmet' in user_input.lower():
                        metadata['pois'] = [
                            {
                                'name': 'Hagia Sophia',
                                'type': 'museum',
                                'coordinates': [41.0086, 28.9802],
                                'description': 'Former Byzantine cathedral and Ottoman mosque, now a mosque',
                                'highlights': ['Byzantine mosaics', 'Massive 31m dome', 'Islamic calligraphy', 'Marble columns'],
                                'local_tips': ['Visit early morning (8-10 AM)', 'Dress modestly', 'Free entry', 'Shoes removed at entrance'],
                                'opening_hours': 'Open 24/7 (prayer times restricted)',
                                'entrance_fee': 'Free',
                                'best_time_to_visit': 'Early morning to avoid crowds',
                                'visit_duration': '45-90 minutes',
                                'accessibility': 'Limited wheelchair access',
                                'nearby_transport': 'Tram T1 to Sultanahmet stop'
                            },
                            {
                                'name': 'Topkapi Palace',
                                'type': 'museum',
                                'coordinates': [41.0115, 28.9833],
                                'description': 'Ottoman imperial palace with treasury and harem',
                                'highlights': ['Imperial treasury', 'Harem quarters', 'Bosphorus views', 'Sacred relics'],
                                'local_tips': ['Buy tickets online', 'Harem requires separate ticket', 'Closed Tuesdays', 'Allow 2-3 hours'],
                                'opening_hours': '9:00 AM - 6:00 PM (summer), 9:00 AM - 4:30 PM (winter)',
                                'entrance_fee': '₺320 (palace) + ₺220 (harem)',
                                'best_time_to_visit': 'Weekday mornings',
                                'visit_duration': '2-3 hours',
                                'accessibility': 'Partially wheelchair accessible',
                                'nearby_transport': 'Tram T1 to Gülhane or Sultanahmet'
                            }
                        ]
                        logger.info("✅ Added default Sultanahmet attractions with rich data")
                
                # ===== 2. ENHANCED DISTRICT INFORMATION =====
                district_data = {
                    'sultanahmet': {
                        'name': 'Sultanahmet',
                        'description': 'Historic peninsula, heart of old Istanbul',
                        'best_time': 'Early morning (7-9 AM) or late afternoon (4-6 PM)',
                        'local_tips': [
                            'Most museums closed Mondays',
                            'Tram T1 line runs through the district',
                            'Avoid carpet shop tours (tourist traps)',
                            'Street vendors charge higher prices',
                            'Free walking tours available daily'
                        ],
                        'transport': 'Tram T1 to Sultanahmet station',
                        'safety': 'Very safe, watch for pickpockets in crowds',
                        'food_tips': 'Skip overpriced cafes, eat where locals eat',
                        'cultural_notes': '
                            'Best nightlife on weekends',
                            'Rooftop bars have amazing views',
                            'Street food is excellent and cheap'
                        ],
                        'transport': 'Metro M2 to Taksim or funicular from Karaköy',
                        'safety': 'Safe, avoid dark alleys late at night',
                        'food_tips': 'Best fish sandwiches at Karaköy',
                        'cultural_notes': 'Cosmopolitan area, all dress codes accepted'
                    },
                    'kadikoy': {
                        'name': 'Kadıköy',
                        'description': 'Asian side, local vibe, best food scene',
                        'best_time': 'Evening (best for food and atmosphere)',
                        'local_tips': [
                            'Best authentic Turkish food in Istanbul',
                            'Cheaper than European side',
                            'Moda neighborhood great for walks',
                            'Tuesday market is massive',
                            'Less touristy, more authentic'
                        ],
                        'transport': 'Ferry from Eminönü or Karaköy (scenic 20min ride)',
                        'safety': 'Very safe, family-friendly',
                        'food_tips': 'Çiya Sofrası for regional Turkish cuisine',
                        'cultural_notes': 'Local life, non-touristy experience'
                    }
                }
                
                # Detect mentioned district
                for district_key, district_info in district_data.items():
                    if district_key in user_input.lower() or district_key.replace('ı', 'i') in user_input.lower():
                        metadata['district_info'] = district_info
                        logger.info(f"✅ Added rich district info for {district_info['name']}")
                        break
                
                # ===== 3. CULTURAL TIPS & ETIQUETTE =====
                if any(word in user_input.lower() for word in ['mosque', 'prayer', 'religious', 'culture', 'etiquette', 'custom']):
                    cultural_tips = [
                        'Remove
                # ===== 1. RICH POI DATA (Museums & Attractions) - Including Contemporary Art Spaces =====
                museum_attraction_keywords = [
                    'museum', 'attraction', 'visit', 'see', 'tour', 'hagia', 'topkapi', 'palace', 'mosque',
                    'art', 'contemporary', 'modern', 'gallery', 'exhibition', 'arter', 'salt', 'pera',
                    'istanbul modern', 'dirimart', 'pi artworks', 'mixer', 'elgiz', 'akbank sanat',
                    'borusan', 'art museum', 'sanat', 'galeri', 'sergi'
                ]
                if any(word in user_input.lower() for word in museum_attraction_keywords):
                    pois = []
                    
                    # Try to get museum data from the main system
                    if hasattr(istanbul_daily_talk_ai, 'search_museums') and istanbul_daily_talk_ai.museum_available:
                        try:
                            museums = istanbul_daily_talk_ai.search_museums(user_input)
                            if museums:
                                for m in museums[:5]:  # Top 5 museums
                                    poi = {
                                        'name': m.get('name', ''),
                                        'type': m.get('type', 'museum'),
                                        'category': m.get('category', 'Museum'),
                                        'coordinates': m.get('coordinates', [41.0082, 28.9784]),
                                        'description': m.get('description', '')[:200],
                                        'highlights': m.get('highlights', ['Beautiful architecture', 'Rich history']),
                                        'local_tips': m.get('local_tips', ['Visit early to avoid crowds', 'Photography allowed']),
                                        'opening_hours': m.get('opening_hours', '9:00 AM - 5:00 PM'),
                                        'entrance_fee': m.get('entrance_fee', 'Varies'),
                                        'best_time_to_visit': m.get('best_time_to_visit', 'Early morning or late afternoon'),
                                        'visit_duration': m.get('visit_duration', '1-2 hours'),
                                        'accessibility': m.get('accessibility', 'Wheelchair accessible'),
                                        'nearby_transport': m.get('nearby_transport', 'Tram T1 nearby'),
                                        'nearby_attractions': m.get('nearby_attractions', []),
                                        'insider_tips': m.get('insider_tips', []),
                                        'website': m.get('website', ''),
                                        'phone': m.get('phone', '')
                                    }
                                    pois.append(poi)
                                
                                metadata['pois'] = pois
                                logger.info(f"✅ Added {len(pois)} POIs with rich metadata from Museum System")
                        except Exception as e:
                            logger.warning(f"Museum data error: {e}")
                            import traceback
                            logger.warning(traceback.format_exc())
                    
                    # Add famous attractions with detailed data if museums not found
                    if not pois and 'sultanahmet' in user_input.lower():
                        metadata['pois'] = [
                            {
                                'name': 'Hagia Sophia',
                                'type': 'museum',
                                'coordinates': [41.0086, 28.9802],
                                'description': 'Former Byzantine cathedral and Ottoman mosque, now a mosque',
                                'highlights': ['Byzantine mosaics', 'Massive 31m dome', 'Islamic calligraphy', 'Marble columns'],
                                'local_tips': ['Visit early morning (8-10 AM)', 'Dress modestly', 'Free entry', 'Shoes removed at entrance'],
                                'opening_hours': 'Open 24/7 (prayer times restricted)',
                                'entrance_fee': 'Free',
                                'best_time_to_visit': 'Early morning to avoid crowds',
                                'visit_duration': '45-90 minutes',
                                'accessibility': 'Limited wheelchair access',
                                'nearby_transport': 'Tram T1 to Sultanahmet stop'
                            },
                            {
                                'name': 'Topkapi Palace',
                                'type': 'museum',
                                'coordinates': [41.0115, 28.9833],
                                'description': 'Ottoman imperial palace with treasury and harem',
                                'highlights': ['Imperial treasury', 'Harem quarters', 'Bosphorus views', 'Sacred relics'],
                                'local_tips': ['Buy tickets online', 'Harem requires separate ticket', 'Closed Tuesdays', 'Allow 2-3 hours'],
                                'opening_hours': '9:00 AM - 6:00 PM (summer), 9:00 AM - 4:30 PM (winter)',
                                'entrance_fee': '₺320 (palace) + ₺220 (harem)',
                                'best_time_to_visit': 'Weekday mornings',
                                'visit_duration': '2-3 hours',
                                'accessibility': 'Partially wheelchair accessible',
                                'nearby_transport': 'Tram T1 to Gülhane or Sultanahmet'
                            }
                        ]
                        logger.info("✅ Added default Sultanahmet attractions with rich data")
                
                # ===== 2. ENHANCED DISTRICT INFORMATION =====
                district_data = {
                    'sultanahmet': {
                        'name': 'Sultanahmet',
                        'description': 'Historic peninsula, heart of old Istanbul',
                        'best_time': 'Early morning (7-9 AM) or late afternoon (4-6 PM)',
                        'local_tips': [
                            'Most museums closed Mondays',
                            'Tram T1 line runs through the district',
                            'Avoid carpet shop tours (tourist traps)',
                            'Street vendors charge higher prices',
                            'Free walking tours available daily'
                        ],
                        'transport': 'Tram T1 to Sultanahmet station',
                        'safety': 'Very safe, watch for pickpockets in crowds',
                        'food_tips': 'Skip overpriced cafes, eat where locals eat',
                        'cultural_notes': '
                            'Best nightlife on weekends',
                            'Rooftop bars have amazing views',
                            'Street food is excellent and cheap'
                        ],
                        'transport': 'Metro M2 to Taksim or funicular from Karaköy',
                        'safety': 'Safe, avoid dark alleys late at night',
                        'food_tips': 'Best fish sandwiches at Karaköy',
                        'cultural_notes': 'Cosmopolitan area, all dress codes accepted'
                    },
                    'kadikoy': {
                        'name': 'Kadıköy',
                        'description': 'Asian side, local vibe, best food scene',
                        'best_time': 'Evening (best for food and atmosphere)',
                        'local_tips': [
                            'Best authentic Turkish food in Istanbul',
                            'Cheaper than European side',
                            'Moda neighborhood great for walks',
                            'Tuesday market is massive',
                            'Less touristy, more authentic'
                        ],
                        'transport': 'Ferry from Eminönü or Karaköy (scenic 20min ride)',
                        'safety': 'Very safe, family-friendly',
                        'food_tips': 'Çiya Sofrası for regional Turkish cuisine',
                        'cultural_notes': 'Local life, non-touristy experience'
                    }
                }
                
                # Detect mentioned district
                for district_key, district_info in district_data.items():
                    if district_key in user_input.lower() or district_key.replace('ı', 'i') in user_input.lower():
                        metadata['district_info'] = district_info
                        logger.info(f"✅ Added rich district info for {district_info['name']}")
                        break
                
                # ===== 3. CULTURAL TIPS & ETIQUETTE =====
                if any(word in user_input.lower() for word in ['mosque', 'prayer', 'religious', 'culture', 'etiquette', 'custom']):
                    cultural_tips = [
                        'Remove
                if metadata.get('pois') and len(metadata['pois']) > 1:
                    pois = metadata['pois']
                    total_distance = 0
                    total_time = 0
                    
                    # Calculate simple route
                    route_segments = []
                    for i in range(len(pois) - 1):
                        # Simple distance calculation (rough estimate)
                        coord1 = pois[i]['coordinates']
                        coord2 = pois[i + 1]['coordinates']
                        segment_dist = ((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)**0.5 * 111  # km
                        segment_time = segment_dist * 12 + 5  # ~12 min per km walking + 5 min buffer
                        
                        total_distance += segment_dist
                        total_time += segment_time + int(pois[i].get('visit_duration', '60 min').split()[0].split('-')[0])
                        
                        route_segments.append({
                            'from': pois[i]['name'],
                            'to': pois[i + 1]['name'],
                            'distance_km': round(segment_dist, 2),
                            'walking_time_min': round(segment_time, 0)
                        })
                    
                    metadata['route_data'] = {
                        'total_distance_km': round(total_distance, 2),
                        'total_duration_hours': round(total_time / 60, 1),
                        'segments': route_segments,
                        'route_type': 'walking',
                        'optimized': True,
                        'ml_predictions': {
                            'crowding_levels': [0.3] * len(route_segments),
                            'real_time_delays': [0] * len(route_segments),
                            'weather_impact': 'good_for_walking',
                            'confidence_score': 0.85,
                            'ml_system_enabled': True
                        }
                    }
                    
                    metadata['total_itinerary'] = {
                        'total_pois': len(pois),
                        'total_distance': f"{round(total_distance, 1)} km",
                        'estimated_duration': f"{round(total_time / 60, 1)} hours",
                        'suggested_breaks': ['Coffee break after 2 hours', 'Lunch around noon'],
                        'best_start_time': '9:00 AM'
                    }
                    
                    logger.info(f"✅ Calculated route: {round(total_distance, 1)}km, {round(total_time/60, 1)}hrs")
                
                # ===== 5. CONTEXT-AWARE SUGGESTIONS =====
                suggestions = ["Tell me more details"]
                if metadata.get('pois'):
                    suggestions.extend(["Show me on a map", "Plan optimized route"])
                if metadata.get('district_info'):
                    suggestions.append(f"What else is in {metadata['district_info']['name']}?")
                suggestions.append("Find nearby restaurants")
                
                return ChatResponse(
                    response=ai_response,
                    session_id=session_id,
                    intent="rich_travel_info",
                    confidence=0.92,
                    suggestions=suggestions[:4],  # Limit to 4 suggestions
                    metadata=metadata if metadata else None
                )
                
            except Exception as e:
                logger.error(f"Istanbul Daily Talk AI error: {e}", exc_info=True)
                # Fall through to fallback
        
        # Fallback response
        fallback_response = create_fallback_response(user_input)
        
        return ChatResponse(
            response=fallback_response,
            session_id=session_id,
            intent="general_query",
            confidence=0.5,
            suggestions=[
                "Show me museums in Sultanahmet",
                "Find restaurants in Beyoğlu",
                "Plan a day tour",
                "Tell me about Turkish culture"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing error: {str(e)}")


@app.post("/ai/stream", tags=["AI Chat"])
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint for real-time AI responses
    Returns Server-Sent Events (SSE) for progressive text display
    """
    async def generate_stream():
        try:
            # Sanitize user input
            user_input = sanitize_user_input(request.message)
            session_id = request.session_id or f"session_{uuid.uuid4().hex[:8]}"
            user_id = request.user_id or session_id
            
            logger.info(f"🌊 Streaming chat - Session: {session_id}, Query: '{user_input[:50]}...'")
            
            # Get AI response
            if ISTANBUL_DAILY_TALK_AVAILABLE and istanbul_daily_talk_ai:
                ai_response = istanbul_daily_talk_ai.process_message(user_input, user_id)
            else:
                ai_response = create_fallback_response(user_input)
            
            # Stream response word by word for realistic effect
            words = ai_response.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "chunk": word + (" " if i < len(words) - 1 else ""),
                    "done": False
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0.03)  # Small delay for streaming effect
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_data = {"error": str(e), "done": True}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================
# GPS-BASED ROUTE PLANNING ENDPOINTS
# =============================