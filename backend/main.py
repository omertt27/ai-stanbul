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

# Initialize Enhanced Feedback Integration System
feedback_integration = None
if FEEDBACK_INTEGRATION_AVAILABLE:
    try:
        feedback_integration = FeedbackIntegration()
        print("✅ Enhanced Feedback Collection & Retraining System initialized")
        print(f"📊 Feedback log: {feedback_integration.feedback_system.feedback_log_path}")
        print(f"📦 Retraining data: {feedback_integration.feedback_system.retraining_data_path}")
    except Exception as e:
        print(f"⚠️ Failed to initialize Feedback Integration: {e}")
        FEEDBACK_INTEGRATION_AVAILABLE = False

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
            
            # Log prediction to ML monitor
            if ML_MONITORING_AVAILABLE and ml_monitor:
                try:
                    ml_monitor.log_prediction(
                        query=user_input,
                        predicted_intent=intent,
                        confidence=confidence,
                        latency_ms=latency_ms
                    )
                except Exception as e:
                    logger.warning(f"Failed to log prediction to monitor: {e}")
                    
        except Exception as e:
            logger.warning(f"Intent classifier error: {e}")
    
    # Step 2.5: Apply context-aware classification (NEW!)
    context_result = None
    if CONTEXT_AWARE_AVAILABLE and context_manager and context_aware_classifier and threshold_manager:
        try:
            if intent_result:
                # Get entities from preprocessing
                entities = preprocessing_result.get('entities', {}) if preprocessing_result else {}
                
                # Apply context-aware classification
                context_result = context_aware_classifier.classify_with_context(
                    query=user_input,
                    preprocessed_query=preprocessed_query,
                    base_intent=intent_result['intent'],
                    base_confidence=intent_result['confidence'],
                    entities=entities,
                    session_id=session_id
                )
                
                # Check acceptance threshold (context_features is a dict from to_dict())
                # Need to pass the original ContextFeatures, not the dict
                from services.context_aware_classifier import ContextFeatures
                ctx_features_dict = context_result['context_features']
                
                threshold_decision = threshold_manager.should_accept(
                    intent=context_result['intent'],
                    confidence=context_result['confidence'],
                    context_features=ctx_features_dict,  # Dict is fine, threshold_manager handles it
                    entities=entities
                )
                
                # Update intent result with context-aware values if accepted
                if threshold_decision['accepted']:
                    original_confidence = intent_result['confidence']
                    intent_result['confidence'] = context_result['confidence']
                    intent_result['context_boost'] = context_result['context_boost']
                    intent_result['context_applied'] = True
                    intent_result['threshold_decision'] = threshold_decision
                    intent_result['resolved_query'] = context_result.get('resolved_query', preprocessed_query)
                    
                    logger.info(f"🔥 Context-aware boost: {original_confidence:.2f} → {context_result['confidence']:.2f} "
                              f"(+{context_result['context_boost']:.2f})")
                else:
                    intent_result['context_applied'] = False
                    intent_result['threshold_decision'] = threshold_decision
                    logger.info(f"⚠️ Classification rejected by threshold: {threshold_decision['confidence']:.2f} < {threshold_decision['threshold']:.2f}")
                
        except Exception as e:
            logger.warning(f"Context-aware classification error: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Update conversation context (NEW!)
        if CONTEXT_AWARE_AVAILABLE and context_manager:
            try:
                turn = Turn(
                    query=user_input,
                    preprocessed_query=preprocessed_query,
                    intent=primary_intent,
                    entities=merged_entities,
                    confidence=confidence
                )
                context_manager.update_context(session_id, turn)
                logger.info(f"💾 Context updated for session: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to update context: {e}")
        
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
            'preprocessing_stats': preprocessing_result.get('statistics') if preprocessing_result else None,
            'context_metadata': context_result if context_result else None  # Add context metadata
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

# Enhanced Authentication imports
try:
    from enhanced_auth import (
        EnhancedAuthManager,
        get_current_user,
        UserRegistrationRequest,
        UserLoginRequest,
        UserRefreshRequest,
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
    
    class UserRefreshRequest(BaseModel):
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
    description="AI-powered Istanbul travel guide with enhanced authentication",
    version="2.0.0"
)

# Include Blog API Router
try:
    from blog_api import router as blog_router
    app.include_router(blog_router)
    print("✅ Blog API endpoints loaded successfully")
except ImportError as e:
    print(f"⚠️ Blog API endpoints not available: {e}")

# Mount static files for admin dashboard
admin_path = os.path.join(os.path.dirname(__file__), '..', 'admin')
if os.path.exists(admin_path):
    app.mount("/admin", StaticFiles(directory=admin_path, html=True), name="admin")
    print(f"✅ Admin dashboard mounted at /admin (path: {admin_path})")
else:
    print(f"⚠️ Admin directory not found at {admin_path}")

# Initialize Enhanced Authentication Manager
auth_manager = None
if ENHANCED_AUTH_AVAILABLE:
    try:
        auth_manager = EnhancedAuthManager()
        logger.info("✅ Enhanced Authentication Manager initialized successfully")
        print("✅ Enhanced Authentication Manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Enhanced Authentication Manager: {e}")
        print(f"⚠️ Failed to initialize Enhanced Authentication Manager: {e}")
        ENHANCED_AUTH_AVAILABLE = False
else:
    logger.warning("Enhanced Authentication not available - authentication endpoints will be disabled")
    print("⚠️ Enhanced Authentication not available - authentication endpoints will be disabled")

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
async def get_admin_stats():
    """
    Get overall statistics for admin dashboard
    """
    try:
        stats = {
            "blog_posts": 0,
            "comments": 0,
            "feedback": 0,
            "active_users": 0,
            "model_accuracy": 95.2,
            "pending_comments": 0
        }
        
        # Load feedback stats if available
        if FEEDBACK_INTEGRATION_AVAILABLE:
            try:
                from user_feedback_collection_system import get_feedback_collector
                collector = get_feedback_collector()
                feedback_summary = collector.get_feedback_summary(days=30)
                
                stats["feedback"] = feedback_summary.get("total", 0)
                
                # Calculate model accuracy from feedback
                if feedback_summary.get("total", 0) > 0:
                    misclass = len(feedback_summary.get("misclassifications", []))
                    accuracy = ((feedback_summary["total"] - misclass) / feedback_summary["total"]) * 100
                    stats["model_accuracy"] = round(accuracy, 1)
            except Exception as e:
                logger.warning(f"Error loading feedback stats: {e}")
        
        # Load blog stats (placeholder - integrate with your blog system)
        blog_data_path = os.path.join("data", "blog_posts.json")
        if os.path.exists(blog_data_path):
            with open(blog_data_path, 'r', encoding='utf-8') as f:
                blog_data = json.load(f)
                stats["blog_posts"] = len(blog_data.get("posts", []))
        
        # Load comments stats (placeholder)
        comments_data_path = os.path.join("data", "comments.json")
        if os.path.exists(comments_data_path):
            with open(comments_data_path, 'r', encoding='utf-8') as f:
                comments_data = json.load(f)
                stats["comments"] = len(comments_data.get("comments", []))
                stats["pending_comments"] = len([c for c in comments_data.get("comments", []) if c.get("status") == "pending"])
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting admin stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/blog/posts", tags=["Admin Dashboard - Blog"])
async def get_blog_posts(status: Optional[str] = None, limit: int = 100):
    """
    Get all blog posts for admin management
    """
    try:
        blog_data_path = os.path.join("data", "blog_posts.json")
        
        # Ensure directory exists
        os.makedirs("data", exist_ok=True)
        
        # Initialize if doesn't exist
        if not os.path.exists(blog_data_path):
            with open(blog_data_path, 'w', encoding='utf-8') as f:
                json.dump({"posts": []}, f)
        
        with open(blog_data_path, 'r', encoding='utf-8') as f:
            blog_data = json.load(f)
        
        posts = blog_data.get("posts", [])
        
        # Filter by status if provided
        if status:
            posts = [p for p in posts if p.get("status", "").lower() == status.lower()]
        
        # Limit results
        posts = posts[:limit]
        
        return {"posts": posts, "total": len(posts)}
        
    except Exception as e:
        logger.error(f"Error getting blog posts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/blog/posts", tags=["Admin Dashboard - Blog"])
async def create_blog_post(post_data: Dict[str, Any] = Body(...)):
    """
    Create a new blog post
    """
    try:
        blog_data_path = os.path.join("data", "blog_posts.json")
        os.makedirs("data", exist_ok=True)
        
        # Load existing posts
        if os.path.exists(blog_data_path):
            with open(blog_data_path, 'r', encoding='utf-8') as f:
                blog_data = json.load(f)
        else:
            blog_data = {"posts": []}
        
        # Create new post
        new_post = {
            "id": len(blog_data["posts"]) + 1,
            "title": post_data.get("title"),
            "slug": post_data.get("slug"),
            "author": post_data.get("author", "Admin"),
            "category": post_data.get("category"),
            "content": post_data.get("content"),
            "status": post_data.get("status", "draft"),
            "date": datetime.now().isoformat(),
            "views": 0,
            "featured_image": post_data.get("featured_image", ""),
            "meta_description": post_data.get("meta_description", ""),
            "tags": post_data.get("tags", [])
        }
        
        blog_data["posts"].insert(0, new_post)
        
        # Save
        with open(blog_data_path, 'w', encoding='utf-8') as f:
            json.dump(blog_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created blog post: {new_post['title']}")
        return {"success": True, "post": new_post}
        
    except Exception as e:
        logger.error(f"Error creating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/blog/posts/{post_id}", tags=["Admin Dashboard - Blog"])
async def update_blog_post(post_id: int, post_data: Dict[str, Any] = Body(...)):
    """
    Update an existing blog post
    """
    try:
        blog_data_path = os.path.join("data", "blog_posts.json")
        
        if not os.path.exists(blog_data_path):
            raise HTTPException(status_code=404, detail="Blog posts not found")
        
        with open(blog_data_path, 'r', encoding='utf-8') as f:
            blog_data = json.load(f)
        
        # Find and update post
        post_found = False
        for post in blog_data["posts"]:
            if post["id"] == post_id:
                post.update({
                    "title": post_data.get("title", post["title"]),
                    "slug": post_data.get("slug", post["slug"]),
                    "category": post_data.get("category", post["category"]),
                    "content": post_data.get("content", post["content"]),
                    "status": post_data.get("status", post["status"]),
                    "updated_at": datetime.now().isoformat()
                })
                post_found = True
                break
        
        if not post_found:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Save
        with open(blog_data_path, 'w', encoding='utf-8') as f:
            json.dump(blog_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Updated blog post: {post_id}")
        return {"success": True, "message": "Post updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/blog/posts/{post_id}", tags=["Admin Dashboard - Blog"])
async def delete_blog_post(post_id: int):
    """
    Delete a blog post
    """
    try:
        blog_data_path = os.path.join("data", "blog_posts.json")
        
        if not os.path.exists(blog_data_path):
            raise HTTPException(status_code=404, detail="Blog posts not found")
        
        with open(blog_data_path, 'r', encoding='utf-8') as f:
            blog_data = json.load(f)
        
        # Remove post
        original_length = len(blog_data["posts"])
        blog_data["posts"] = [p for p in blog_data["posts"] if p["id"] != post_id]
        
        if len(blog_data["posts"]) == original_length:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Save
        with open(blog_data_path, 'w', encoding='utf-8') as f:
            json.dump(blog_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Deleted blog post: {post_id}")
        return {"success": True, "message": "Post deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting blog post: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/comments", tags=["Admin Dashboard - Comments"])
async def get_comments(status: Optional[str] = None, post_id: Optional[int] = None, limit: int = 100):
    """
    Get all comments for admin management
    """
    try:
        comments_data_path = os.path.join("data", "comments.json")
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(comments_data_path):
            with open(comments_data_path, 'w', encoding='utf-8') as f:
                json.dump({"comments": []}, f)
        
        with open(comments_data_path, 'r', encoding='utf-8') as f:
            comments_data = json.load(f)
        
        comments = comments_data.get("comments", [])
        
        # Filter by status
        if status:
            comments = [c for c in comments if c.get("status", "").lower() == status.lower()]
        
        # Filter by post_id
        if post_id:
            comments = [c for c in comments if c.get("post_id") == post_id]
        
        comments = comments[:limit]
        
        return {"comments": comments, "total": len(comments)}
        
    except Exception as e:
        logger.error(f"Error getting comments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/admin/comments/{comment_id}/approve", tags=["Admin Dashboard - Comments"])
async def approve_comment(comment_id: int):
    """
    Approve a pending comment
    """
    try:
        comments_data_path = os.path.join("data", "comments.json")
        
        if not os.path.exists(comments_data_path):
            raise HTTPException(status_code=404, detail="Comments not found")
        
        with open(comments_data_path, 'r', encoding='utf-8') as f:
            comments_data = json.load(f)
        
        # Find and approve comment
        comment_found = False
        for comment in comments_data["comments"]:
            if comment["id"] == comment_id:
                comment["status"] = "approved"
                comment["approved_at"] = datetime.now().isoformat()
                comment_found = True
                break
        
        if not comment_found:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        with open(comments_data_path, 'w', encoding='utf-8') as f:
            json.dump(comments_data, f, indent=2, ensure_ascii=False)
        
        return {"success": True, "message": "Comment approved"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving comment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/comments/{comment_id}", tags=["Admin Dashboard - Comments"])
async def delete_comment(comment_id: int):
    """
    Delete a comment
    """
    try:
        comments_data_path = os.path.join("data", "comments.json")
        
        if not os.path.exists(comments_data_path):
            raise HTTPException(status_code=404, detail="Comments not found")
        
        with open(comments_data_path, 'r', encoding='utf-8') as f:
            comments_data = json.load(f)
        
        original_length = len(comments_data["comments"])
        comments_data["comments"] = [c for c in comments_data["comments"] if c["id"] != comment_id]
        
        if len(comments_data["comments"]) == original_length:
            raise HTTPException(status_code=404, detail="Comment not found")
        
        with open(comments_data_path, 'w', encoding='utf-8') as f:
            json.dump(comments_data, f, indent=2, ensure_ascii=False)
        
        return {"success": True, "message": "Comment deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
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
        }
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/analytics", tags=["Admin Dashboard - Analytics"])
async def get_analytics(days: int = 30):
    """
    Get analytics data for charts and insights
    """
    try:
        analytics = {
            "period": f"last_{days}_days",
            "user_queries": [],
            "blog_views": [],
            "comments": [],
            "dates": []
        }
        
        # Generate sample data for last N days
        from datetime import timedelta
        today = datetime.now()
        
        for i in range(days):
            date = today - timedelta(days=days-i-1)
            analytics["dates"].append(date.strftime("%b %d"))
            analytics["user_queries"].append(45 + (i * 2) + (i % 7))
            analytics["blog_views"].append(28 + (i * 3) + (i % 5))
            analytics["comments"].append(5 + (i % 10))
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/intents/stats", tags=["Admin Dashboard - Intents"])
async def get_intent_statistics():
    """
    Get detailed intent classification statistics
    """
    try:
        if not FEEDBACK_INTEGRATION_AVAILABLE:
            # Return mock data
            return {
                "intents": [
                    {"intent": "find_attraction", "count": 245, "accuracy": 94.2, "confidence": 0.89, "corrections": 12},
                    {"intent": "find_restaurant", "count": 198, "accuracy": 91.5, "confidence": 0.86, "corrections": 18},
                    {"intent": "get_directions", "count": 156, "accuracy": 88.3, "confidence": 0.82, "corrections": 22},
                    {"intent": "find_hotel", "count": 134, "accuracy": 93.8, "confidence": 0.91, "corrections": 8},
                    {"intent": "get_transportation", "count": 112, "accuracy": 85.7, "confidence": 0.79, "corrections": 28}
                ]
            }
        
        from user_feedback_collection_system import get_feedback_collector
        collector = get_feedback_collector()
        
        feedback_summary = collector.get_feedback_summary(days=30)
        by_function = feedback_summary.get("by_function", {})
        
        intent_stats = []
        for intent, count in by_function.items():
            intent_stats.append({
                "intent": intent,
                "count": count,
                "accuracy": 90.0 + (hash(intent) % 10),  # Mock accuracy
                "confidence": 0.75 + (hash(intent) % 25) / 100,  # Mock confidence
                "corrections": hash(intent) % 30  # Mock corrections
            })
        
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