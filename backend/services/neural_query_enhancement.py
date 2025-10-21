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
- Advanced natural language understanding
- Sophisticated intent recognition
- Semantic parsing and entity extraction
- Contextual reasoning and inference
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
import re
import spacy
from textblob import TextBlob

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
        import torch.nn.functional as F
    except ImportError as e:
        ADVANCED_ML_AVAILABLE = False
        _ml_import_error = f"PyTorch modules: {e}"

if ADVANCED_ML_AVAILABLE:
    try:
        from transformers import (
            AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
            BertModel, BertTokenizer,
            pipeline, Trainer, TrainingArguments,
            AutoModelForTokenClassification, AutoModelForQuestionAnswering
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

# NLP and Language Processing
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load spacy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            SPACY_AVAILABLE = False
            nlp = None
            print("⚠️  No spaCy model available. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("⚠️  spaCy not available")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    TextBlob = None
    print("⚠️  TextBlob not available")

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
    ADVANCED_NLP_PARSING = "advanced_nlp_parsing"
    ENTITY_EXTRACTION = "entity_extraction"
    SEMANTIC_ROLE_LABELING = "semantic_role_labeling"
    CONTEXTUAL_INFERENCE = "contextual_inference"
    DISCOURSE_ANALYSIS = "discourse_analysis"
    PRAGMATIC_UNDERSTANDING = "pragmatic_understanding"

class IntentComplexityLevel(Enum):
    """Intent complexity classification"""
    SIMPLE = "simple"          # Single intent, clear keywords
    MODERATE = "moderate"      # Multiple aspects, some ambiguity
    COMPLEX = "complex"        # Multiple intents, contextual dependencies
    VERY_COMPLEX = "very_complex"  # Implicit meaning, high context dependence

@dataclass
class EntityExtraction:
    """Extracted entity information"""
    entity: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    semantic_role: Optional[str] = None
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SemanticParsing:
    """Semantic parsing result"""
    semantic_roles: Dict[str, str]
    dependency_tree: Dict[str, Any]
    key_phrases: List[str]
    semantic_relations: List[Tuple[str, str, str]]
    discourse_markers: List[str]
    pragmatic_markers: Dict[str, Any]
    complexity_level: IntentComplexityLevel
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AdvancedNeuralPrediction:
    """Enhanced neural network prediction result"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    embedding: Optional[np.ndarray] = None
    attention_weights: Optional[Dict[str, float]] = None
    entities: List[EntityExtraction] = field(default_factory=list)
    semantic_parsing: Optional[SemanticParsing] = None
    contextual_features: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)
    alternative_interpretations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    Including sophisticated natural language understanding
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
        
        # NLP processing components
        self.nlp_processor = nlp if SPACY_AVAILABLE else None
        self.sentiment_analyzer = None
        self.entity_patterns = self._initialize_entity_patterns()
        self.intent_keywords = self._initialize_intent_keywords()
        self.semantic_relations = self._initialize_semantic_relations()
        
        # Model configuration
        self.model_configs = {
            "intent_classifier": {
                "model_name": "distilbert-base-uncased",  # Classification only
                "max_length": 512,
                "batch_size": 16
            },
            "semantic_encoder": {
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "embedding_dim": 768
            },
            "response_ranker": {  # Changed from response_generator to response_ranker
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Ranking model, not generative
                "max_length": 256
            },
            "multilingual_encoder": {
                "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                "languages": ["en", "tr"]
            },
            "entity_extractor": {
                "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english",
                "confidence_threshold": 0.8
            },
            "question_answering": {
                "model_name": "distilbert-base-cased-distilled-squad",
                "max_answer_length": 100
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
            
            # Response Ranking Model (not generative)
            self._load_response_ranker()
            
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
            
            # Load pre-trained BERT-based classifier (no generative models)
            self.models["intent_classifier"] = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=12,  # Number of query types
                problem_type="single_label_classification"
            )
            
            self.tokenizers["intent_classifier"] = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased"
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
    
    def _load_response_ranker(self):
        """Load response ranking model (not generative)"""
        try:
            config = self.model_configs["response_ranker"]
            
            # Use cross-encoder for ranking pre-defined responses, not generating text
            from sentence_transformers import CrossEncoder
            
            self.models["response_ranker"] = CrossEncoder(config["model_name"])
            
            logger.info("✅ Response ranker loaded (classification-based, not generative)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load response ranker: {e}")
            # Fallback to simple similarity scoring
            self.models["response_ranker"] = None
    
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
    
    async def process_advanced_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process query with advanced neural capabilities including sophisticated NLP
        """
        start_time = time.time()
        
        if context is None:
            context = {}
        
        try:
            logger.info(f"🧠 Processing advanced query: {query[:100]}...")
            
            # Initialize result structure
            result = {
                "query": query,
                "processing_timestamp": datetime.now().isoformat(),
                "capabilities_used": [],
                "advanced_nlp": {},
                "neural_analysis": {},
                "contextual_understanding": {},
                "final_recommendations": {}
            }
            
            # 1. Advanced NLP Processing
            nlp_result = await self._perform_advanced_nlp_analysis(query, context)
            result["advanced_nlp"] = nlp_result
            result["capabilities_used"].append("advanced_nlp_parsing")
            
            # 2. Neural Language Detection and Analysis
            neural_analysis = await self._perform_neural_language_analysis(query, context)
            result["neural_analysis"] = neural_analysis
            result["capabilities_used"].append("neural_language_analysis")
            
            # 3. Advanced Intent Recognition
            intent_analysis = await self._perform_advanced_intent_recognition(query, context, nlp_result)
            result["intent_recognition"] = intent_analysis
            result["capabilities_used"].append("advanced_intent_recognition")
            
            # 4. Contextual Understanding and Reasoning
            contextual_analysis = await self._perform_contextual_reasoning(query, context, nlp_result, intent_analysis)
            result["contextual_understanding"] = contextual_analysis
            result["capabilities_used"].append("contextual_reasoning")
            
            # 5. Semantic Understanding and Embeddings
            if "semantic_encoder" in self.models:
                semantic_analysis = await self._perform_semantic_analysis(query, context, nlp_result)
                result["semantic_analysis"] = semantic_analysis
                result["capabilities_used"].append("semantic_understanding")
            
            # 6. Generate Final Recommendations
            recommendations = await self._generate_contextual_recommendations(
                query, context, nlp_result, intent_analysis, contextual_analysis
            )
            result["final_recommendations"] = recommendations
            
            # Performance tracking
            processing_time = (time.time() - start_time) * 1000
            result["performance_metrics"] = {
                "processing_time_ms": processing_time,
                "complexity_level": nlp_result.get("complexity_level", "moderate"),
                "confidence_overall": intent_analysis.get("confidence", 0.5)
            }
            
            # Store performance data
            self.performance_history["advanced_query_processing"].append({
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": processing_time,
                "query_length": len(query),
                "capabilities_used": len(result["capabilities_used"]),
                "confidence_overall": result["performance_metrics"]["confidence_overall"]
            })
            
            logger.info(f"✅ Advanced query processing completed in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"❌ Advanced query processing failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "query": query,
                "processing_timestamp": datetime.now().isoformat(),
                "fallback_processing": await self._fallback_simple_processing(query, context)
            }
    
    async def _perform_advanced_nlp_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive NLP analysis"""
        try:
            nlp_analysis = {
                "text_preprocessing": {},
                "entity_extraction": [],
                "semantic_parsing": {},
                "linguistic_features": {},
                "complexity_assessment": {}
            }
            
            # Text preprocessing
            preprocessed = self._preprocess_text(query)
            nlp_analysis["text_preprocessing"] = {
                "original": query,
                "cleaned": preprocessed["cleaned"],
                "tokens": preprocessed["tokens"],
                "normalized": preprocessed["normalized"]
            }
            
            # Entity extraction
            entities = await self._extract_entities_advanced(query, context)
            nlp_analysis["entity_extraction"] = [
                {
                    "entity": entity.entity,
                    "label": entity.label,
                    "confidence": entity.confidence,
                    "position": [entity.start_pos, entity.end_pos],
                    "semantic_role": entity.semantic_role,
                    "normalized_value": entity.normalized_value
                }
                for entity in entities
            ]
            
            # Semantic parsing
            semantic_parsing = await self._perform_semantic_parsing(query, entities)
            nlp_analysis["semantic_parsing"] = {
                "semantic_roles": semantic_parsing.semantic_roles,
                "key_phrases": semantic_parsing.key_phrases,
                "semantic_relations": semantic_parsing.semantic_relations,
                "discourse_markers": semantic_parsing.discourse_markers,
                "complexity_level": semantic_parsing.complexity_level.value
            }
            
            # Linguistic features
            linguistic_features = self._extract_linguistic_features(query)
            nlp_analysis["linguistic_features"] = linguistic_features
            
            # Complexity assessment
            complexity = self._assess_query_complexity(query, entities, semantic_parsing)
            nlp_analysis["complexity_assessment"] = complexity
            
            return nlp_analysis
            
        except Exception as e:
            logger.error(f"❌ Advanced NLP analysis failed: {e}")
            return {"error": str(e)}
    
    def _preprocess_text(self, text: str) -> Dict[str, Any]:
        """Advanced text preprocessing"""
        # Basic cleaning
        cleaned = re.sub(r'\s+', ' ', text.strip())
        cleaned = re.sub(r'[^\w\s\-\.,!?]', '', cleaned)
        
        # Tokenization
        tokens = cleaned.lower().split()
        
        # Normalization
        normalized = cleaned.lower()
        
        # Remove stop words (basic implementation)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'}
        filtered_tokens = [token for token in tokens if token not in stop_words]
        
        return {
            "cleaned": cleaned,
            "tokens": tokens,
            "normalized": normalized,
            "filtered_tokens": filtered_tokens
        }
    
    async def _extract_entities_advanced(self, query: str, context: Dict[str, Any]) -> List[EntityExtraction]:
        """Advanced entity extraction using multiple methods"""
        entities = []
        
        try:
            # Method 1: spaCy NER (if available)
            if self.nlp_processor:
                spacy_entities = self._extract_entities_spacy(query)
                entities.extend(spacy_entities)
            
            # Method 2: Pattern-based extraction
            pattern_entities = self._extract_entities_patterns(query)
            entities.extend(pattern_entities)
            
            # Method 3: Transformer-based NER (if available)
            if "entity_extractor" in self.models:
                transformer_entities = await self._extract_entities_transformer(query)
                entities.extend(transformer_entities)
            
            # Method 4: Context-aware extraction
            context_entities = self._extract_entities_contextual(query, context)
            entities.extend(context_entities)
            
            # Deduplicate and merge entities
            entities = self._merge_duplicate_entities(entities)
            
            # Add semantic roles
            entities = self._assign_semantic_roles(entities, query)
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Entity extraction failed: {e}")
            return []
    
    def _extract_entities_spacy(self, query: str) -> List[EntityExtraction]:
        """Extract entities using spaCy"""
        entities = []
        
        if not self.nlp_processor:
            return entities
        
        try:
            doc = self.nlp_processor(query)
            
            for ent in doc.ents:
                entity = EntityExtraction(
                    entity=ent.text,
                    label=ent.label_,
                    confidence=0.8,  # spaCy doesn't provide confidence scores directly
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    context=query[max(0, ent.start_char-20):min(len(query), ent.end_char+20)],
                    metadata={"method": "spacy", "label_description": spacy.explain(ent.label_)}
                )
                entities.append(entity)
                
        except Exception as e:
            logger.error(f"❌ spaCy entity extraction failed: {e}")
        
        return entities
    
    def _extract_entities_patterns(self, query: str) -> List[EntityExtraction]:
        """Extract entities using pattern matching"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entity = EntityExtraction(
                        entity=match.group().strip(),
                        label=entity_type,
                        confidence=0.7,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=query[max(0, match.start()-20):min(len(query), match.end()+20)],
                        metadata={"method": "pattern", "pattern": pattern}
                    )
                    entities.append(entity)
        
        return entities
    
    async def _extract_entities_transformer(self, query: str) -> List[EntityExtraction]:
        """Extract entities using transformer models"""
        entities = []
        
        if "entity_extractor" not in self.models:
            return entities
        
        try:
            # Use Hugging Face NER pipeline
            if hasattr(self, '_ner_pipeline'):
                ner_pipeline = self._ner_pipeline
            else:
                config = self.model_configs["entity_extractor"]
                ner_pipeline = pipeline("ner", 
                                       model=config["model_name"], 
                                       aggregation_strategy="simple")
                self._ner_pipeline = ner_pipeline
            
            results = ner_pipeline(query)
            
            for result in results:
                if result['score'] >= self.model_configs["entity_extractor"]["confidence_threshold"]:
                    entity = EntityExtraction(
                        entity=result['word'],
                        label=result['entity_group'],
                        confidence=result['score'],
                        start_pos=result['start'],
                        end_pos=result['end'],
                        context=query[max(0, result['start']-20):min(len(query), result['end']+20)],
                        metadata={"method": "transformer", "model": "bert-ner"}
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logger.error(f"❌ Transformer entity extraction failed: {e}")
        
        return entities
    
    def _extract_entities_contextual(self, query: str, context: Dict[str, Any]) -> List[EntityExtraction]:
        """Extract entities based on context"""
        entities = []
        
        # Extract location from context
        if "user_location" in context:
            location_data = context["user_location"]
            if "address" in location_data:
                entity = EntityExtraction(
                    entity=location_data["address"],
                    label="USER_LOCATION",
                    confidence=0.9,
                    start_pos=0,
                    end_pos=0,
                    context="User's current location",
                    metadata={"method": "contextual", "source": "user_location"}
                )
                entities.append(entity)
        
        # Extract from session history
        if "session_context" in context and "previous_queries" in context["session_context"]:
            # Look for recurring entities in session
            prev_queries = context["session_context"]["previous_queries"]
            for prev_query in prev_queries[-3:]:  # Last 3 queries
                # Simple entity carryover logic
                if "restaurant" in prev_query.lower() and "restaurant" in query.lower():
                    entity = EntityExtraction(
                        entity="restaurant",
                        label="CONTEXT_CARRYOVER",
                        confidence=0.6,
                        start_pos=query.lower().find("restaurant"),
                        end_pos=query.lower().find("restaurant") + len("restaurant"),
                        context="Carried over from previous query",
                        metadata={"method": "contextual", "source": "session_history"}
                    )
                    entities.append(entity)
        
        return entities
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity recognition patterns"""
        return {
            "LOCATION": [
                r'\b(near|around|in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(district|area|neighborhood|quarter)\b',
                r'\b(istanbul|turkey|türkiye)\b',
                r'\b(europe|asia|european|asian)\s+(side)\b'
            ],
            "RESTAURANT": [
                r'\b(restaurant|cafe|bistro|eatery|diner|pub|bar)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(restaurant|cafe|bistro)\b',
                r'\b(turkish|italian|chinese|japanese|french|mediterranean|seafood|kebab)\s+(food|cuisine|restaurant)\b'
            ],
            "ATTRACTION": [
                r'\b(museum|palace|mosque|church|gallery|monument|landmark)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(museum|palace|mosque|church)\b',
                r'\b(hagia\s+sophia|blue\s+mosque|topkapi|galata\s+tower|bosphorus)\b'
            ],
            "TRANSPORTATION": [
                r'\b(metro|bus|tram|ferry|taxi|dolmus|minibus)\b',
                r'\b(public\s+transport|transportation|getting\s+around)\b',
                r'\b(marmaray|metrobus|iett)\b'
            ],
            "TIME": [
                r'\b(now|today|tomorrow|tonight|morning|afternoon|evening|night)\b',
                r'\b(\d{1,2}:\d{2}|\d{1,2}\s*(am|pm))\b',
                r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
            ],
            "PRICE": [
                r'\b(cheap|expensive|budget|affordable|luxury|free)\b',
                r'\b(\d+)\s*(lira|tl|dollar|euro|₺|\$|€)\b',
                r'\b(under|below|above|over)\s*(\d+)\b'
            ]
        }
    
    def _initialize_intent_keywords(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize intent classification keywords with weights"""
        return {
            "restaurant_search": {
                "high_confidence": ["restaurant", "eat", "food", "dining", "cuisine", "meal"],
                "medium_confidence": ["hungry", "lunch", "dinner", "breakfast", "cafe", "bar"],
                "contextual": ["good", "best", "recommend", "where", "find"]
            },
            "attraction_search": {
                "high_confidence": ["visit", "see", "tourist", "attraction", "sightseeing", "museum"],
                "medium_confidence": ["historical", "cultural", "famous", "landmark", "monument"],
                "contextual": ["what", "where", "show", "explore", "discover"]
            },
            "transportation_search": {
                "high_confidence": ["get", "go", "transport", "metro", "bus", "taxi", "travel"],
                "medium_confidence": ["how", "way", "route", "direction", "public"],
                "contextual": ["from", "to", "between", "connect"]
            },
            "hotel_search": {
                "high_confidence": ["hotel", "stay", "accommodation", "room", "booking"],
                "medium_confidence": ["sleep", "night", "lodge", "hostel", "inn"],
                "contextual": ["where", "book", "reserve", "check"]
            },
            "general_info": {
                "high_confidence": ["information", "about", "tell", "explain", "what"],
                "medium_confidence": ["help", "know", "understand", "learn"],
                "contextual": ["istanbul", "turkey", "city", "culture"]
            },
            "weather_query": {
                "high_confidence": ["weather", "temperature", "rain", "sunny", "cloudy"],
                "medium_confidence": ["hot", "cold", "warm", "forecast", "climate"],
                "contextual": ["today", "tomorrow", "now", "current"]
            }
        }
    
    def _initialize_semantic_relations(self) -> Dict[str, List[str]]:
        """Initialize semantic relation patterns"""
        return {
            "location_relations": ["near", "in", "at", "around", "close to", "next to"],
            "time_relations": ["during", "at", "in", "on", "before", "after"],
            "preference_relations": ["like", "prefer", "want", "need", "looking for"],
            "comparison_relations": ["better", "best", "worse", "similar", "different"],
            "causality_relations": ["because", "since", "due to", "as a result", "therefore"]
        }
    
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
            
            # 10. Advanced NLP Parsing
            if "advanced_nlp_parsing" in self.models:
                nlp_parsing_result = await self._parse_query_advanced(query, context)
                results["neural_predictions"]["nlp_parsing"] = nlp_parsing_result
            
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
        """Process cross-lingual understanding with enhanced neural capabilities"""
        try:
            # Enhanced language detection with neural features
            turkish_indicators = {
                'nerede', 'nasıl', 'ne', 'istanbul', 'türk', 'mı', 'mi', 'mu', 'mü',
                'var', 'yok', 'için', 'ile', 'şey', 'kişi', 'gün', 'saat', 'dakika',
                'restoran', 'otel', 'müze', 'tarihi', 'güzel', 'iyi', 'kötü', 'çok',
                'bir', 'bu', 'şu', 'o', 'bana', 'bize', 'orda', 'burada', 'şurada'
            }
            
            english_indicators = {
                'where', 'how', 'what', 'when', 'why', 'the', 'is', 'are', 'and', 'or',
                'restaurant', 'hotel', 'museum', 'historical', 'beautiful', 'good', 'bad',
                'best', 'find', 'show', 'tell', 'help', 'need', 'can', 'could', 'would',
                'please', 'thank', 'thanks', 'there', 'here', 'this', 'that'
            }
            
            query_words = set(query.lower().split())
            
            # Neural language detection enhancement
            if ADVANCED_ML_AVAILABLE and "multilingual_encoder" in self.models:
                neural_lang_detection = await self._neural_language_detection(query)
            else:
                neural_lang_detection = {"confidence": 0.0, "detected_language": "unknown"}
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

    async def _neural_language_detection(self, query: str) -> Dict[str, Any]:
        """Advanced neural language detection using multilingual models"""
        try:
            if self.models["multilingual_encoder"] == "simple_fallback":
                return {"confidence": 0.5, "detected_language": "mixed", "method": "fallback"}
            
            model = self.models["multilingual_encoder"]
            
            # Encode query in multilingual space
            embedding = model.encode([query])
            
            # Language-specific similarity comparison
            turkish_samples = [
                "Istanbul'da nerede yemek yiyebilirim?",
                "En iyi müzeler hangileri?",
                "Nasıl gidebilirim oraya?"
            ]
            
            english_samples = [
                "Where can I eat in Istanbul?",
                "What are the best museums?",
                "How can I get there?"
            ]
            
            # Calculate similarities
            turkish_embeddings = model.encode(turkish_samples)
            english_embeddings = model.encode(english_samples)
            
            # Compute average similarities
            turkish_sim = np.mean([np.dot(embedding[0], te) / (np.linalg.norm(embedding[0]) * np.linalg.norm(te)) 
                                  for te in turkish_embeddings])
            english_sim = np.mean([np.dot(embedding[0], ee) / (np.linalg.norm(embedding[0]) * np.linalg.norm(ee)) 
                                  for ee in english_embeddings])
            
            # Determine language
            if turkish_sim > english_sim and turkish_sim > 0.7:
                detected_lang = "turkish"
                confidence = float(turkish_sim)
            elif english_sim > turkish_sim and english_sim > 0.7:
                detected_lang = "english"
                confidence = float(english_sim)
            else:
                detected_lang = "mixed"
                confidence = float(max(turkish_sim, english_sim))
            
            return {
                "detected_language": detected_lang,
                "confidence": confidence,
                "turkish_similarity": float(turkish_sim),
                "english_similarity": float(english_sim),
                "method": "neural_embedding"
            }
            
        except Exception as e:
            logger.error(f"❌ Neural language detection failed: {e}")
            return {"confidence": 0.0, "detected_language": "unknown", "error": str(e)}
    
    async def _parse_query_advanced(self, query: str, context: Dict[str, Any]) -> SemanticParsing:
        """Advanced semantic parsing of the query"""
        try:
            if not SPACY_AVAILABLE:
                return {"error": "spaCy not available for parsing"}
            
            # Use spaCy for initial parsing
            doc = nlp(query)
            
            # Extract semantic roles and dependencies
            semantic_roles = {}
            dependencies = {}
            key_phrases = []
            relations = []
            discourse_markers = []
            pragmatic_markers = {}
            
            for token in doc:
                # Semantic role labeling (simple example)
                if token.dep_ in ["nsubj", "dobj", "pobj"]:
                    semantic_roles[token.text] = token.dep_
                
                # Dependency parsing
                dependencies[token.text] = {
                    "head": token.head.text,
                    "dep": token.dep_,
                    "children": [child.text for child in token.children]
                }
                
                # Key phrase extraction (nouns and verbs as key phrases)
                if token.pos_ in ["NOUN", "VERB"] and len(token.text) > 2:
                    key_phrases.append(token.text)
            
            # Relation extraction (simple co-occurrence based)
            for sent in doc.sents:
                for token1 in sent:
                    for token2 in sent:
                        if token1 != token2 and token1.dep_ != "punct":
                            relations.append((token1.text, "related_to", token2.text))
            
            # Discourse and pragmatic marker detection (simple heuristics)
            for sent in doc.sents:
                if any(marker in sent.text for marker in ["however", "furthermore", "meanwhile"]):
                    discourse_markers.append(sent.text)
                
                # Pragmatic markers based on sentiment analysis
                if TEXTBLOB_AVAILABLE:
                    blob = TextBlob(sent.text)
                    if blob.sentiment.polarity < -0.1:
                        pragmatic_markers[sent.text] = "negative_sentiment"
                    elif blob.sentiment.polarity > 0.1:
                        pragmatic_markers[sent.text] = "positive_sentiment"
            
            complexity_level = IntentComplexityLevel.SIMPLE
            if len(key_phrases) > 5 or len(relations) > 3:
                complexity_level = IntentComplexityLevel.COMPLEX
            elif len(key_phrases) > 3:
                complexity_level = IntentComplexityLevel.MODERATE
            
            confidence = min(1.0, 0.5 + len(key_phrases) * 0.1)
            
            return SemanticParsing(
                semantic_roles=semantic_roles,
                dependency_tree=dependencies,
                key_phrases=key_phrases,
                semantic_relations=relations,
                discourse_markers=discourse_markers,
                pragmatic_markers=pragmatic_markers,
                complexity_level=complexity_level,
                confidence=confidence,
                metadata={
                    "method": "spaCy_enhanced",
                    "parsed_entities": len([ent for ent in doc.ents]),
                    "sentences_analyzed": len(list(doc.sents))
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error in advanced semantic parsing: {e}")
            return {"error": str(e)}
    
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
