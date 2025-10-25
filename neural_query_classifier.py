#!/usr/bin/env python3
"""
Neural Query Classifier for Istanbul AI
Uses trained DistilBERT model to classify Turkish tourism queries into 25 intent categories
Model: phase2_extended_model.pth (81.1% accuracy, 11ms latency)
"""

import torch
import torch.nn as nn
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntentClassifierHead(nn.Module):
    """Custom classifier head matching training architecture"""
    
    def __init__(self, num_intents: int = 25):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_intents)
        )
    
    def forward(self, x):
        return self.classifier(x)


class NeuralQueryClassifier:
    """
    Neural query classifier using DistilBERT for Turkish intent classification
    
    Features:
    - Fast inference (<15ms)
    - 25 intent classes
    - Confidence scores
    - Prediction logging
    - Error handling
    - Model hot-swapping
    """
    
    # 25 Intent classes (matching training data)
    INTENT_CLASSES = [
        "accommodation", "attraction", "booking", "budget", "cultural_info",
        "emergency", "events", "family_activities", "food", "general_info",
        "gps_navigation", "hidden_gems", "history", "local_tips", "luxury",
        "museum", "nightlife", "price_info", "recommendation", "restaurant",
        "romantic", "route_planning", "shopping", "transportation", "weather"
    ]
    
    def __init__(
        self,
        model_path: str = "models/istanbul_intent_classifier_finetuned",  # Updated to use fine-tuned model
        confidence_threshold: float = 0.70,
        device: str = "auto",
        enable_logging: bool = True,
        log_file: str = "neural_predictions.jsonl"
    ):
        """
        Initialize neural query classifier
        
        Args:
            model_path: Path to trained model weights
            confidence_threshold: Minimum confidence for predictions
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
            enable_logging: Enable prediction logging
            log_file: Path to log file (JSON Lines format)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.enable_logging = enable_logging
        self.log_file = log_file
        
        # Setup device
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Intent mappings
        self.intent_to_id = {intent: idx for idx, intent in enumerate(self.INTENT_CLASSES)}
        self.id_to_intent = {idx: intent for intent, idx in self.intent_to_id.items()}
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None  # For fine-tuned model
        self.base_model = None  # For old format
        self.classifier = None  # For old format
        self.use_finetuned = False  # Flag to indicate which model type
        self._load_model()
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'errors': 0,
            'avg_latency_ms': 0.0
        }
        
        logger.info(f"Neural classifier initialized successfully")
        logger.info(f"Model: {model_path}")
        logger.info(f"Intents: {len(self.INTENT_CLASSES)}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            model_path = Path(self.model_path)
            
            # Check if this is a fine-tuned Hugging Face model (has config.json)
            if (model_path / "config.json").exists():
                logger.info(f"Loading fine-tuned Hugging Face model from {self.model_path}...")
                
                # Load intent mapping FIRST to get the correct number of labels
                intent_mapping_path = model_path / "intent_mapping.json"
                num_labels = len(self.INTENT_CLASSES)  # Default
                
                if intent_mapping_path.exists():
                    with open(intent_mapping_path, 'r', encoding='utf-8') as f:
                        mapping = json.load(f)
                        # Update intent mappings from the saved model
                        if 'intents' in mapping:
                            self.INTENT_CLASSES = mapping['intents']
                            self.intent_to_id = {intent: idx for idx, intent in enumerate(self.INTENT_CLASSES)}
                            self.id_to_intent = {idx: intent for intent, idx in self.intent_to_id.items()}
                            num_labels = len(self.INTENT_CLASSES)
                            logger.info(f"✅ Loaded intent mapping: {num_labels} intents")
                
                # Load tokenizer and model with correct num_labels
                from transformers import AutoModelForSequenceClassification
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=num_labels
                )
                
                # Load training metadata if available
                metadata_path = model_path / "training_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        logger.info(f"✅ Fine-tuned model loaded:")
                        logger.info(f"   Training accuracy: {metadata.get('final_train_accuracy', 0):.2%}")
                        logger.info(f"   Validation accuracy: {metadata.get('final_val_accuracy', 0):.2%}")
                        logger.info(f"   Dataset size: {metadata.get('dataset_size', 'unknown')} samples")
                        logger.info(f"   Trained on: {metadata.get('training_date', 'unknown')}")
                
                self.model.to(self.device)
                self.model.eval()
                self.use_finetuned = True
                
                logger.info("✅ Fine-tuned model loaded and ready")
                
            else:
                # Fallback to old .pth format
                logger.info("Loading base model with custom classifier head...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "distilbert-base-multilingual-cased"
                )
                
                self.base_model = AutoModel.from_pretrained(
                    "distilbert-base-multilingual-cased"
                )
                
                self.classifier = IntentClassifierHead(num_intents=len(self.INTENT_CLASSES))
                
                # Load trained weights if exists
                if model_path.exists() and model_path.is_file():
                    logger.info(f"Loading trained weights from {self.model_path}...")
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict) and 'classifier_state_dict' in checkpoint:
                        classifier_state = checkpoint['classifier_state_dict']
                        self.classifier.load_state_dict(classifier_state)
                        logger.info(f"✅ Loaded classifier head weights")
                        logger.info(f"   Accuracy: {checkpoint.get('accuracy', 'unknown')}")
                        logger.info(f"   Training samples: {checkpoint.get('training_samples', 'unknown')}")
                        logger.info(f"   Latency: {checkpoint.get('latency', 'unknown')}ms")
                    elif isinstance(checkpoint, dict):
                        self.classifier.load_state_dict(checkpoint)
                        logger.info("✅ Loaded full classifier weights")
                    else:
                        logger.error(f"Unknown checkpoint format")
                        raise ValueError("Cannot load checkpoint")
                    
                    logger.info("Trained weights loaded successfully")
                else:
                    logger.warning(f"Model file not found: {self.model_path}")
                    logger.warning("Using untrained classifier!")
                
                self.base_model.to(self.device)
                self.classifier.to(self.device)
                self.base_model.eval()
                self.classifier.eval()
                self.use_finetuned = False
                
                logger.info("Model loaded and ready")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predict intent for a single query
        
        Args:
            query: User query in Turkish
            
        Returns:
            Tuple of (intent, confidence)
            
        Example:
            intent, confidence = classifier.predict("En yakın restoran nerede?")
            # Returns: ("restaurant", 0.95)
        """
        start_time = time.time()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                query,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Predict based on model type
            with torch.no_grad():
                if self.use_finetuned:
                    # Use fine-tuned model directly
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                else:
                    # Use base model + classifier head
                    outputs = self.base_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                    logits = self.classifier(embeddings)
                
                probabilities = torch.softmax(logits, dim=1)
                confidence, predicted_id = probabilities.max(dim=1)
                
                predicted_id = predicted_id.item()
                confidence = confidence.item()
            
            # Get intent
            intent = self.id_to_intent[predicted_id]
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_stats(confidence, latency_ms)
            
            # Log prediction
            if self.enable_logging:
                self._log_prediction(query, intent, confidence, latency_ms)
            
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            self.stats['errors'] += 1
            return "general_info", 0.0  # Safe fallback
    
    def batch_predict(self, queries: List[str]) -> List[Tuple[str, float]]:
        """
        Predict intents for multiple queries (batched for efficiency)
        
        Args:
            queries: List of user queries
            
        Returns:
            List of (intent, confidence) tuples
        """
        results = []
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                queries,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            # Predict batch based on model type
            with torch.no_grad():
                if self.use_finetuned:
                    # Use fine-tuned model directly
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                else:
                    # Use base model + classifier head
                    outputs = self.base_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS tokens
                    logits = self.classifier(embeddings)
                
                probabilities = torch.softmax(logits, dim=1)
                confidences, predicted_ids = probabilities.max(dim=1)
            
            # Process results
            for i, query in enumerate(queries):
                predicted_id = predicted_ids[i].item()
                confidence = confidences[i].item()
                intent = self.id_to_intent[predicted_id]
                
                results.append((intent, confidence))
                
                if self.enable_logging:
                    self._log_prediction(query, intent, confidence, 0.0)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Return safe fallback for all
            return [("general_info", 0.0) for _ in queries]
    
    def get_top_k_intents(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top K most likely intents for a query
        
        Args:
            query: User query
            k: Number of top intents to return
            
        Returns:
            List of (intent, confidence) tuples, sorted by confidence
        """
        try:
            inputs = self.tokenizer(
                query,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                # Get embeddings
                outputs = self.base_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]
                
                # Classify
                logits = self.classifier(embeddings)
                probabilities = torch.softmax(logits, dim=1)
                top_k = probabilities.topk(k, dim=1)
            
            results = []
            for i in range(k):
                intent_id = top_k.indices[0][i].item()
                confidence = top_k.values[0][i].item()
                intent = self.id_to_intent[intent_id]
                results.append((intent, confidence))
            
            return results
            
        except Exception as e:
            logger.error(f"Top-K prediction error: {e}")
            return [("general_info", 0.0)]
    
    def is_high_confidence(self, confidence: float) -> bool:
        """Check if confidence meets threshold"""
        return confidence >= self.confidence_threshold
    
    def _update_stats(self, confidence: float, latency_ms: float):
        """Update prediction statistics"""
        self.stats['total_predictions'] += 1
        
        if confidence >= self.confidence_threshold:
            self.stats['high_confidence'] += 1
        else:
            self.stats['low_confidence'] += 1
        
        # Update average latency (moving average)
        n = self.stats['total_predictions']
        old_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (old_avg * (n - 1) + latency_ms) / n
    
    def _log_prediction(self, query: str, intent: str, confidence: float, latency_ms: float):
        """Log prediction to file for retraining"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'intent': intent,
                'confidence': round(confidence, 4),
                'latency_ms': round(latency_ms, 2),
                'high_confidence': confidence >= self.confidence_threshold
            }
            
            # Append to JSON Lines file
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Logging error: {e}")
    
    def get_stats(self) -> Dict:
        """Get classifier statistics"""
        stats = self.stats.copy()
        
        if stats['total_predictions'] > 0:
            stats['high_confidence_rate'] = stats['high_confidence'] / stats['total_predictions']
            stats['low_confidence_rate'] = stats['low_confidence'] / stats['total_predictions']
            stats['error_rate'] = stats['errors'] / stats['total_predictions']
        
        return stats
    
    def reload_model(self, new_model_path: Optional[str] = None):
        """
        Reload model (for hot-swapping updated models)
        
        Args:
            new_model_path: Path to new model file (optional)
        """
        if new_model_path:
            self.model_path = new_model_path
        
        logger.info(f"Reloading model from {self.model_path}...")
        self._load_model()
        logger.info("Model reloaded successfully")
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_predictions': 0,
            'high_confidence': 0,
            'low_confidence': 0,
            'errors': 0,
            'avg_latency_ms': 0.0
        }
        logger.info("Statistics reset")


# Singleton instance (lazy loading)
_classifier_instance = None


def get_classifier() -> NeuralQueryClassifier:
    """
    Get singleton classifier instance
    
    Returns:
        NeuralQueryClassifier instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = NeuralQueryClassifier()
    
    return _classifier_instance


# Quick test function
def test_classifier():
    """Test the classifier with sample queries"""
    print("=" * 80)
    print("Testing Neural Query Classifier")
    print("=" * 80)
    print()
    
    classifier = get_classifier()
    
    test_queries = [
        "En yakın restoran nerede?",
        "Ayasofya'yı görmek istiyorum",
        "Havaalanına nasıl giderim?",
        "Hava durumu nasıl?",
        "Müze önerileri",
        "Ucuz otel",
        "Acil durum!",
        "Boğaz turu",
        "Kebap nerede yiyebilirim?",
        "Taksim'e nasıl giderim?",
    ]
    
    print("Testing individual predictions:")
    print("-" * 80)
    
    for query in test_queries:
        intent, confidence = classifier.predict(query)
        status = "✅" if confidence >= 0.70 else "⚠️"
        print(f"{status} Query: '{query}'")
        print(f"   Intent: {intent}")
        print(f"   Confidence: {confidence:.1%}")
        print()
    
    print("=" * 80)
    print("Statistics:")
    print("=" * 80)
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    print("✅ Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Run test when executed directly
    test_classifier()
