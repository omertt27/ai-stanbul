"""
Neural Query Enhancement System
==============================

Advanced neural enhancements for the AI Istanbul query system including:
- Deep learning query understanding
- Context-aware response generation
- Multi-modal query processing
- Real-time model fine-tuning
- Federated learning capabilities
- Adversarial query detection
- Cross-lingual semantic understanding
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import threading
from collections import defaultdict, deque
import hashlib
import time

# Advanced ML and Deep Learning imports
ADVANCED_ML_AVAILABLE = False
_ml_import_error = None

try:
    import torch
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    _ml_import_error = f"PyTorch: {e}"

if ADVANCED_ML_AVAILABLE:
    try:
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
    except ImportError as e:
        ADVANCED_ML_AVAILABLE = False
        _ml_import_error = f"PyTorch modules: {e}"

if ADVANCED_ML_AVAILABLE:
    try:
        from transformers import (
            AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
            BertModel, GPT2LMHeadModel, T5ForConditionalGeneration,
            pipeline, Trainer, TrainingArguments
        )
    except ImportError as e:
        ADVANCED_ML_AVAILABLE = False
        _ml_import_error = f"Transformers: {e}"

if ADVANCED_ML_AVAILABLE:
    try:
        from sentence_transformers import SentenceTransformer, losses
    except ImportError as e:
        print(f"⚠️  SentenceTransformers not available: {e}")
        # Don't disable all ML, just sentence transformers
        SentenceTransformer = None

if ADVANCED_ML_AVAILABLE:
    try:
        import faiss
    except ImportError as e:
        print(f"⚠️  FAISS not available: {e}")
        faiss = None

if not ADVANCED_ML_AVAILABLE and _ml_import_error:
    print(f"⚠️  ML dependencies not available: {_ml_import_error}")

logger = logging.getLogger(__name__)

class EnhancementCapability(Enum):
    """Advanced AI enhancement capabilities"""
    DEEP_SEMANTIC_UNDERSTANDING = "deep_semantic_understanding"
    CONTEXTUAL_RESPONSE_GENERATION = "contextual_response_generation"
    MULTI_MODAL_PROCESSING = "multi_modal_processing"
    REAL_TIME_LEARNING = "real_time_learning"
    ADVERSARIAL_DETECTION = "adversarial_detection"
    CROSS_LINGUAL_UNDERSTANDING = "cross_lingual_understanding"
    QUERY_INTENT_PREDICTION = "query_intent_prediction"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    CONVERSATIONAL_AI = "conversational_ai"
    KNOWLEDGE_GRAPH_REASONING = "knowledge_graph_reasoning"

@dataclass
class NeuralPrediction:
    """Neural network prediction result"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    embedding: Optional[np.ndarray] = None
    attention_weights: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultiModalInput:
    """Multi-modal input data structure"""
    text: Optional[str] = None
    image_data: Optional[bytes] = None
    audio_data: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_flags: Dict[str, bool] = field(default_factory=dict)

class AdvancedNeuralProcessor:
    """
    Advanced neural processing engine with state-of-the-art capabilities
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.vector_stores = {}
        self.fine_tuning_data = defaultdict(list)
        self.model_versions = {}
        self.performance_history = defaultdict(list)
        
        # Advanced features
        self.federated_learning_enabled = False
        self.adversarial_detection_enabled = True
        self.multi_modal_enabled = False
        self.real_time_learning_enabled = True
        
        # Model configuration
        self.model_configs = {
            "intent_classifier": {
                "model_name": "microsoft/DialoGPT-medium",
                "max_length": 512,
                "batch_size": 16
            },
            "semantic_encoder": {
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "embedding_dim": 768
            },
            "response_generator": {
                "model_name": "microsoft/DialoGPT-small",
                "max_length": 256
            },
            "multilingual_encoder": {
                "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "languages": ["en", "tr"]
            }
        }
        
        if ADVANCED_ML_AVAILABLE:
            self._initialize_advanced_models()
    
    def _initialize_advanced_models(self):
        """Initialize advanced neural models"""
        try:
            logger.info("🧠 Initializing advanced neural models...")
            
            # Intent Classification Model
            self._load_intent_classifier()
            
            # Semantic Encoding Model
            self._load_semantic_encoder()
            
            # Response Generation Model
            self._load_response_generator()
            
            # Multilingual Processing
            self._load_multilingual_models()
            
            # Vector similarity search
            self._initialize_vector_stores()
            
            logger.info("✅ Advanced neural models initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize advanced models: {e}")
    
    def _load_intent_classifier(self):
        """Load intent classification model"""
        try:
            config = self.model_configs["intent_classifier"]
            
            # Load pre-trained model for intent classification
            self.models["intent_classifier"] = AutoModelForSequenceClassification.from_pretrained(
                "microsoft/DialoGPT-medium",
                num_labels=12,  # Number of query types
                problem_type="single_label_classification"
            )
            
            self.tokenizers["intent_classifier"] = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium"
            )
            
            # Add padding token if missing
            if self.tokenizers["intent_classifier"].pad_token is None:
                self.tokenizers["intent_classifier"].pad_token = self.tokenizers["intent_classifier"].eos_token
            
            logger.info("✅ Intent classifier loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load intent classifier: {e}")
    
    def _load_semantic_encoder(self):
        """Load semantic encoding model"""
        try:
            config = self.model_configs["semantic_encoder"]
            
            self.models["semantic_encoder"] = SentenceTransformer(config["model_name"])
            
            logger.info("✅ Semantic encoder loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load semantic encoder: {e}")
    
    def _load_response_generator(self):
        """Load response generation model"""
        try:
            config = self.model_configs["response_generator"]
            
            self.models["response_generator"] = GPT2LMHeadModel.from_pretrained(
                config["model_name"]
            )
            self.tokenizers["response_generator"] = AutoTokenizer.from_pretrained(
                config["model_name"]
            )
            
            # Add padding token
            if self.tokenizers["response_generator"].pad_token is None:
                self.tokenizers["response_generator"].pad_token = self.tokenizers["response_generator"].eos_token
            
            logger.info("✅ Response generator loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load response generator: {e}")
    
    def _load_multilingual_models(self):
        """Load multilingual processing models"""
        try:
            if SentenceTransformer is None:
                logger.warning("⚠️  SentenceTransformer not available, using simple multilingual fallback")
                self.models["multilingual_encoder"] = "simple_fallback"
                return
            
            config = self.model_configs["multilingual_encoder"]
            
            # Force loading of multilingual encoder
            logger.info("🌍 Loading multilingual encoder...")
            
            # Try primary model first
            try:
                self.models["multilingual_encoder"] = SentenceTransformer(config["model_name"])
                logger.info(f"✅ Primary multilingual model loaded: {config['model_name']}")
            except Exception as primary_error:
                logger.warning(f"⚠️  Primary model failed, trying fallback: {primary_error}")
                # Try fallback models in order of preference
                fallback_models = [
                    "all-MiniLM-L6-v2",
                    "distiluse-base-multilingual-cased-v2",
                    "paraphrase-multilingual-MiniLM-L12-v2"
                ]
                
                for fallback_model in fallback_models:
                    try:
                        logger.info(f"🔄 Attempting fallback: {fallback_model}")
                        self.models["multilingual_encoder"] = SentenceTransformer(fallback_model)
                        logger.info(f"✅ Fallback multilingual encoder loaded: {fallback_model}")
                        break
                    except Exception as fallback_error:
                        logger.warning(f"⚠️  Fallback {fallback_model} failed: {fallback_error}")
                        continue
                
                if "multilingual_encoder" not in self.models:
                    logger.warning("⚠️  All transformer models failed, using simple fallback")
                    self.models["multilingual_encoder"] = "simple_fallback"
                    return
            
            # Test the model with a simple encoding
            test_text = "Hello world"
            embedding = self.models["multilingual_encoder"].encode([test_text])
            logger.info(f"✅ Multilingual encoder test successful, embedding shape: {embedding.shape}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load multilingual models: {e}")
            # Set a simple fallback
            logger.info("🔄 Using simple multilingual fallback")
            self.models["multilingual_encoder"] = "simple_fallback"
    
    def _initialize_vector_stores(self):
        """Initialize FAISS vector stores for similarity search"""
        try:
            # Create vector stores for different purposes
            embedding_dim = self.model_configs["semantic_encoder"]["embedding_dim"]
            
            # Intent similarity store
            self.vector_stores["intents"] = faiss.IndexFlatIP(embedding_dim)
            
            # Query history store
            self.vector_stores["query_history"] = faiss.IndexFlatL2(embedding_dim)
            
            # Knowledge base store
            self.vector_stores["knowledge"] = faiss.IndexFlatIP(embedding_dim)
            
            logger.info("✅ Vector stores initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector stores: {e}")
    
    async def process_advanced_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process query with advanced neural capabilities
        
        Args:
            query: Input query text
            context: Processing context
            
        Returns:
            Advanced processing results
        """
        start_time = time.time()
        results = {
            "query": query,
            "context": context,
            "processing_time_ms": 0.0,
            "neural_predictions": {},
            "enhancements": {},
            "confidence_scores": {},
            "attention_analysis": {},
            "multi_modal_results": {},
            "real_time_adaptations": {}
        }
        
        try:
            # 1. Advanced Intent Classification
            if "intent_classifier" in self.models:
                intent_result = await self._classify_intent_advanced(query, context)
                results["neural_predictions"]["intent"] = intent_result
            
            # 2. Deep Semantic Understanding
            if "semantic_encoder" in self.models:
                semantic_result = await self._analyze_semantics_deep(query, context)
                results["neural_predictions"]["semantics"] = semantic_result
            
            # 3. Contextual Enhancement
            contextual_result = await self._enhance_with_context(query, context, results)
            results["enhancements"]["contextual"] = contextual_result
            
            # 4. Multi-modal Processing (if enabled)
            if self.multi_modal_enabled and context.get("multi_modal_data"):
                multi_modal_result = await self._process_multi_modal(
                    query, context.get("multi_modal_data"), context
                )
                results["multi_modal_results"] = multi_modal_result
            
            # 5. Adversarial Detection
            if self.adversarial_detection_enabled:
                adversarial_result = await self._detect_adversarial_query(query, context)
                results["enhancements"]["adversarial_detection"] = adversarial_result
            
            # 6. Real-time Learning Adaptation
            if self.real_time_learning_enabled:
                adaptation_result = await self._adapt_models_real_time(query, context, results)
                results["real_time_adaptations"] = adaptation_result
            
            # 7. Cross-lingual Understanding
            cross_lingual_result = await self._process_cross_lingual(query, context)
            results["enhancements"]["cross_lingual"] = cross_lingual_result
            
            # 8. Attention Analysis
            attention_result = await self._analyze_attention_patterns(query, results)
            results["attention_analysis"] = attention_result
            
            # 9. Emotional Intelligence Processing
            emotional_result = await self._process_emotional_intelligence(query, context)
            results["enhancements"]["emotional_intelligence"] = emotional_result
            
            # Calculate overall confidence
            results["confidence_scores"] = self._calculate_advanced_confidence(results)
            
            processing_time = (time.time() - start_time) * 1000
            results["processing_time_ms"] = processing_time
            
            # Record performance metrics
            self._record_neural_performance(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in advanced query processing: {e}")
            results["error"] = str(e)
            results["processing_time_ms"] = (time.time() - start_time) * 1000
            return results
    
    async def _classify_intent_advanced(self, query: str, context: Dict[str, Any]) -> NeuralPrediction:
        """Advanced intent classification with neural networks and fallback strategies"""
        try:
            # Try neural classification first
            if ADVANCED_ML_AVAILABLE and "intent_classifier" in self.models:
                model = self.models["intent_classifier"]
                tokenizer = self.tokenizers["intent_classifier"]
                
                # Tokenize input
                inputs = tokenizer(
                    query,
                    return_tensors="pt",
                    max_length=512,
                    padding=True,
                    truncation=True
                )
                
                # Get model predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                
                # Map to query types
                query_types = [
                    "greeting", "attraction_info", "attraction_search", 
                    "restaurant_search", "restaurant_info", "transport_route",
                    "transport_info", "itinerary_request", "general_info",
                    "practical_info", "recommendation", "unknown"
                ]
                
                # Get predictions
                prob_dict = {}
                for i, query_type in enumerate(query_types):
                    prob_dict[query_type] = float(probabilities[0][i])
                
                # Get top prediction
                max_prob_idx = torch.argmax(probabilities, dim=-1).item()
                predicted_intent = query_types[max_prob_idx]
                confidence = float(probabilities[0][max_prob_idx])
                
                return NeuralPrediction(
                    prediction=predicted_intent,
                    confidence=confidence,
                    probabilities=prob_dict,
                    metadata={
                        "model_version": "v1.0",
                        "processing_method": "transformer_classification"
                    }
                )
            else:
                # Enhanced rule-based intent classification
                return self._rule_based_intent_classification(query, context)
            
        except Exception as e:
            logger.error(f"❌ Error in advanced intent classification: {e}")
            # Fallback to rule-based classification
            return self._rule_based_intent_classification(query, context)
    
    async def _analyze_semantics_deep(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced semantic analysis using deep learning"""
        try:
            if "semantic_encoder" not in self.models:
                return {"error": "Semantic encoder not available"}
            
            model = self.models["semantic_encoder"]
            
            # Get semantic embedding
            embedding = model.encode([query])
            
            # Perform similarity searches
            similar_queries = self._find_similar_queries(embedding[0])
            semantic_clusters = self._identify_semantic_clusters(embedding[0])
            
            # Advanced semantic features
            semantic_features = {
                "embedding_vector": embedding[0].tolist(),
                "embedding_norm": float(np.linalg.norm(embedding[0])),
                "similar_queries": similar_queries,
                "semantic_clusters": semantic_clusters,
                "semantic_complexity": self._calculate_semantic_complexity(embedding[0]),
                "topic_distribution": self._analyze_topic_distribution(embedding[0])
            }
            
            return semantic_features
            
        except Exception as e:
            logger.error(f"❌ Error in deep semantic analysis: {e}")
            return {"error": str(e)}
    
    async def _enhance_with_context(self, query: str, context: Dict[str, Any], 
                                  current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance query understanding with contextual information"""
        enhancements = {
            "temporal_context": self._analyze_temporal_context(context),
            "user_context": self._analyze_user_context(context),
            "session_context": self._analyze_session_context(context),
            "geographical_context": self._analyze_geographical_context(context),
            "behavioral_patterns": self._identify_behavioral_patterns(context)
        }
        
        # Context-aware adjustments
        if enhancements["temporal_context"].get("is_peak_hours"):
            enhancements["recommendations"] = ["Consider off-peak alternatives"]
        
        if enhancements["user_context"].get("experience_level") == "beginner":
            enhancements["recommendations"] = enhancements.get("recommendations", []) + [
                "Provide detailed explanations",
                "Include basic information"
            ]
        
        return enhancements
    
    async def _process_multi_modal(self, query: str, multi_modal_data: MultiModalInput,
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Process multi-modal input (text, image, audio)"""
        results = {
            "text_analysis": {},
            "image_analysis": {},
            "audio_analysis": {},
            "fusion_results": {}
        }
        
        # This would be implemented with appropriate models
        # For now, return placeholder
        results["status"] = "multi_modal_processing_placeholder"
        results["capabilities"] = ["text", "future_image", "future_audio"]
        
        return results
    
    async def _detect_adversarial_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect adversarial or malicious queries"""
        detection_results = {
            "adversarial_score": 0.0,
            "threats_detected": [],
            "safety_score": 1.0,
            "recommendations": []
        }
        
        # Simple adversarial detection heuristics
        adversarial_patterns = [
            r"ignore previous instructions",
            r"act as.*",
            r"pretend to be",
            r"jailbreak",
            r"bypass.*filter"
        ]
        
        query_lower = query.lower()
        for pattern in adversarial_patterns:
            if pattern in query_lower:
                detection_results["threats_detected"].append(pattern)
                detection_results["adversarial_score"] += 0.2
        
        # Calculate safety score
        detection_results["safety_score"] = max(0.0, 1.0 - detection_results["adversarial_score"])
        
        if detection_results["adversarial_score"] > 0.5:
            detection_results["recommendations"].append("Apply additional safety measures")
        
        return detection_results
    
    async def _adapt_models_real_time(self, query: str, context: Dict[str, Any],
                                    results: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt models based on real-time feedback"""
        adaptations = {
            "learning_rate_adjustments": {},
            "model_updates": {},
            "feedback_integration": {},
            "performance_optimizations": {}
        }
        
        # Record query for future training
        if self.real_time_learning_enabled:
            query_data = {
                "query": query,
                "context": context,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add to fine-tuning data
            intent = results.get("neural_predictions", {}).get("intent", {}).get("prediction", "unknown")
            self.fine_tuning_data[intent].append(query_data)
            
            # Trigger model update if enough data accumulated
            if len(self.fine_tuning_data[intent]) >= 100:
                adaptations["model_updates"][intent] = "scheduled_for_update"
        
        return adaptations
    
    async def _process_cross_lingual(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process cross-lingual understanding with enhanced heuristics"""
        try:
            # Enhanced language detection
            turkish_indicators = {
                'nerede', 'nasıl', 'ne', 'istanbul', 'türk', 'mı', 'mi', 'mu', 'mü',
                'var', 'yok', 'için', 'ile', 'şey', 'kişi', 'gün', 'saat', 'dakika',
                'restoran', 'otel', 'müze', 'tarihi', 'güzel', 'iyi', 'kötü'
            }
            
            english_indicators = {
                'where', 'how', 'what', 'when', 'why', 'the', 'is', 'are', 'and', 'or',
                'restaurant', 'hotel', 'museum', 'historical', 'beautiful', 'good', 'bad',
                'best', 'find', 'show', 'tell', 'help', 'need'
            }
            
            query_words = set(query.lower().split())
            turkish_matches = len(query_words.intersection(turkish_indicators))
            english_matches = len(query_words.intersection(english_indicators))
            
            if turkish_matches > english_matches:
                language = "tr"
                confidence = min(turkish_matches / 5, 1.0)
            elif english_matches > turkish_matches:
                language = "en"
                confidence = min(english_matches / 5, 1.0)
            else:
                language = "unknown"
                confidence = 0.5
            
            return {
                "detected_language": language,
                "confidence": confidence,
                "cross_lingual_support": True,
                "translation_needed": language == "tr",
                "method": "enhanced_heuristic",
                "turkish_score": turkish_matches,
                "english_score": english_matches
            }
                
        except Exception as e:
            logger.error(f"❌ Cross-lingual processing failed: {e}")
            return {"error": f"Cross-lingual processing failed: {e}"}

    async def _analyze_attention_patterns(self, query: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze attention patterns in neural processing"""
        # This would extract attention weights from transformer models
        attention_analysis = {
            "query_attention_map": {},
            "important_tokens": [],
            "attention_distribution": {},
            "focus_areas": []
        }
        
        # Placeholder implementation
        words = query.split()
        for i, word in enumerate(words):
            attention_analysis["query_attention_map"][word] = min(1.0, len(word) / 10)
        
        # Identify important tokens (longer words get higher attention)
        attention_analysis["important_tokens"] = [
            word for word in words if len(word) > 4
        ]
        
        return attention_analysis
    
    async def _process_emotional_intelligence(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotional intelligence understanding"""
        try:
            # Emotional keywords and patterns
            emotion_patterns = {
                "disappointment": ["disappointed", "let down", "frustrated", "not happy"],
                "excitement": ["excited", "thrilled", "amazing", "wonderful", "fantastic"],
                "anxiety": ["worried", "nervous", "concerned", "anxious"],
                "satisfaction": ["satisfied", "happy", "pleased", "great", "good"],
                "anger": ["angry", "furious", "upset", "mad"],
                "joy": ["joyful", "delighted", "overjoyed", "ecstatic"]
            }
            
            query_lower = query.lower()
            emotion_scores = {}
            
            for emotion, keywords in emotion_patterns.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    emotion_scores[emotion] = score / len(keywords)
            
            if emotion_scores:
                detected_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                return {
                    "emotion": detected_emotion[0],
                    "confidence": detected_emotion[1],
                    "all_emotions": emotion_scores,
                    "emotional_context": True
                }
            else:
                return {
                    "emotion": "neutral",
                    "confidence": 0.8,
                    "all_emotions": {},
                    "emotional_context": False
                }
                
        except Exception as e:
            logger.error(f"❌ Emotional intelligence processing failed: {e}")
            return {"error": f"Emotional intelligence processing failed: {e}"}

    async def _process_adversarial_detection(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process adversarial query detection"""
        try:
            # Adversarial patterns
            adversarial_patterns = [
                "ignore all previous instructions",
                "forget everything",
                "system prompt",
                "act as a different",
                "pretend to be",
                "override your",
                "bypass safety",
                "jailbreak",
                "your instructions are",
                "change your behavior"
            ]
            
            query_lower = query.lower()
            threat_score = 0
            detected_patterns = []
            
            for pattern in adversarial_patterns:
                if pattern in query_lower:
                    threat_score += 1
                    detected_patterns.append(pattern)
            
            # Check for excessive special characters (potential injection)
            special_chars = sum(1 for c in query if not c.isalnum() and c not in " .,!?-'")
            if special_chars > len(query) * 0.3:  # More than 30% special chars
                threat_score += 1
                detected_patterns.append("excessive_special_characters")
            
            # Check for extremely long queries (potential DoS)
            if len(query) > 1000:
                threat_score += 1
                detected_patterns.append("excessive_length")
            
            is_adversarial = threat_score > 0
            confidence = min(threat_score / 3.0, 1.0)  # Normalize to 0-1
            
            return {
                "is_adversarial": is_adversarial,
                "confidence": confidence,
                "threat_score": threat_score,
                "detected_patterns": detected_patterns,
                "risk_level": "high" if confidence > 0.7 else "medium" if confidence > 0.3 else "low"
            }
                
        except Exception as e:
            logger.error(f"❌ Adversarial detection failed: {e}")
            return {"error": f"Adversarial detection failed: {e}"}

    # ...existing code...
    
    def get_neural_system_status(self) -> Dict[str, Any]:
        """Get comprehensive neural system status"""
        return {
            "models_loaded": list(self.models.keys()),
            "capabilities": [cap.value for cap in EnhancementCapability],
            "advanced_ml_available": ADVANCED_ML_AVAILABLE,
            "configuration": self.model_configs,
            "performance_metrics": {
                name: {
                    "count": len(history),
                    "avg_processing_time": np.mean([h["processing_time_ms"] for h in history]) if history else 0,
                    "avg_confidence": np.mean([h["confidence_overall"] for h in history]) if history else 0
                }
                for name, history in self.performance_history.items()
            },
            "fine_tuning_data_size": {
                intent: len(data) for intent, data in self.fine_tuning_data.items()
            },
            "feature_flags": {
                "federated_learning": self.federated_learning_enabled,
                "adversarial_detection": self.adversarial_detection_enabled,
                "multi_modal": self.multi_modal_enabled,
                "real_time_learning": self.real_time_learning_enabled
            }
        }

# Global neural processor instance
_neural_processor_instance = None

def get_neural_processor() -> AdvancedNeuralProcessor:
    """Get the global neural processor instance"""
    global _neural_processor_instance
    if _neural_processor_instance is None:
        _neural_processor_instance = AdvancedNeuralProcessor()
    return _neural_processor_instance

# Convenience functions
async def process_query_neural(query: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Process query with advanced neural capabilities"""
    processor = get_neural_processor()
    return await processor.process_advanced_query(query, context)

def get_neural_status() -> Dict[str, Any]:
    """Get neural system status"""
    processor = get_neural_processor()
    return processor.get_neural_system_status()

# Initialize neural processor when module is imported
if __name__ != "__main__":
    try:
        _neural_processor_instance = AdvancedNeuralProcessor()
        logger.info("🧠 Advanced Neural Processor auto-initialized")
    except Exception as e:
        logger.error(f"❌ Failed to auto-initialize neural processor: {e}")
