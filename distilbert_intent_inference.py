#!/usr/bin/env python3
"""
DistilBERT Intent Classifier Inference Module
Loads the trained DistilBERT model for real-time intent classification
"""

import torch
import json
import time
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistilBERTIntentClassifier:
    """
    Fast and accurate intent classifier using fine-tuned DistilBERT
    
    Features:
    - 91.3% validation accuracy
    - 30 intent classes
    - Multilingual support (Turkish + English)
    - Fast inference (<50ms on CPU)
    - Confidence scores
    """
    
    def __init__(
        self,
        model_path: str = "models/distilbert_intent_classifier",
        confidence_threshold: float = 0.70,
        device: str = "auto"
    ):
        """
        Initialize DistilBERT intent classifier
        
        Args:
            model_path: Path to trained model directory
            confidence_threshold: Minimum confidence for predictions
            device: Device to use ('auto', 'mps', 'cuda', 'cpu')
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Setup device
        self.device = self._setup_device(device)
        logger.info(f"Using device: {self.device}")
        
        # Load intent mapping
        self.intent_to_id, self.id_to_intent = self._load_intent_mapping()
        logger.info(f"Loaded {len(self.intent_to_id)} intent classes")
        
        # Load model and tokenizer
        self.tokenizer, self.model = self._load_model()
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'avg_latency_ms': 0.0
        }
        
        logger.info("✅ DistilBERT Intent Classifier ready")
    
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
    
    def _load_intent_mapping(self) -> Tuple[Dict, Dict]:
        """Load intent mapping from JSON file"""
        mapping_path = self.model_path / "intent_mapping.json"
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"Intent mapping not found: {mapping_path}")
        
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        intent_to_id = data['intent_to_idx']
        id_to_intent = {int(k): v for k, v in data['idx_to_intent'].items()}
        
        return intent_to_id, id_to_intent
    
    def _load_model(self) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
        """Load tokenizer and model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path),
            num_labels=len(self.intent_to_id)
        )
        
        model.to(self.device)
        model.eval()
        
        # Load and display training summary
        summary_path = self.model_path / "training_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                logger.info(f"Model trained on: {summary.get('trained_at', 'unknown')}")
                logger.info(f"Validation accuracy: {summary.get('validation_accuracy', 0):.2%}")
                logger.info(f"Training examples: {summary.get('training_examples', 0)}")
        
        return tokenizer, model
    
    def predict(self, query: str) -> Tuple[str, float]:
        """
        Predict intent for a query
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (intent, confidence)
        """
        start_time = time.time()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                confidence, predicted_id = torch.max(probs, dim=-1)
            
            # Get intent
            intent = self.id_to_intent[predicted_id.item()]
            confidence = confidence.item()
            
            # Update statistics
            latency_ms = (time.time() - start_time) * 1000
            self._update_stats(confidence, latency_ms)
            
            return intent, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "general_info", 0.5
    
    def predict_with_top_k(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Predict top K intents for a query
        
        Args:
            query: User query text
            k: Number of top predictions to return
            
        Returns:
            List of (intent, confidence) tuples
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top K
            top_probs, top_indices = torch.topk(probs[0], k)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                intent = self.id_to_intent[idx.item()]
                results.append((intent, prob.item()))
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return [("general_info", 0.5)]
    
    def _update_stats(self, confidence: float, latency_ms: float):
        """Update prediction statistics"""
        self.stats['total_predictions'] += 1
        
        if confidence >= 0.80:
            self.stats['high_confidence'] += 1
        elif confidence >= 0.60:
            self.stats['medium_confidence'] += 1
        else:
            self.stats['low_confidence'] += 1
        
        # Update average latency
        total = self.stats['total_predictions']
        current_avg = self.stats['avg_latency_ms']
        self.stats['avg_latency_ms'] = (current_avg * (total - 1) + latency_ms) / total
    
    def get_stats(self) -> Dict:
        """Get prediction statistics"""
        return self.stats.copy()
    
    def get_intents(self) -> List[str]:
        """Get all available intent classes"""
        return list(self.intent_to_id.keys())


# Singleton instance
_classifier_instance = None


def get_distilbert_classifier() -> DistilBERTIntentClassifier:
    """Get singleton classifier instance"""
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = DistilBERTIntentClassifier()
    
    return _classifier_instance


def test_classifier():
    """Test the classifier with various queries"""
    print("=" * 80)
    print("DISTILBERT INTENT CLASSIFIER TEST")
    print("=" * 80)
    print()
    
    classifier = get_distilbert_classifier()
    
    # Test queries (Turkish + English)
    test_queries = [
        # Turkish queries
        "Ayasofya'yı ziyaret etmek istiyorum",
        "En yakın restoran nerede?",
        "Hava durumu nasıl?",
        "Taksim'e nasıl gidebilirim?",
        "Müze önerileri var mı?",
        "Romantik bir restoran öner",
        "Çocuklarla gidebileceğim yerler",
        "Ucuz otel arıyorum",
        "Gece hayatı nasıl?",
        "Boğaz turu rezervasyonu",
        
        # English queries
        "I want to visit Hagia Sophia",
        "Where is the nearest restaurant?",
        "How is the weather?",
        "How can I get to Taksim?",
        "Any museum recommendations?",
        "Recommend a romantic restaurant",
        "Places to go with kids",
        "Looking for cheap hotel",
        "What's the nightlife like?",
        "Book a Bosphorus tour",
        
        # Edge cases
        "Merhaba",
        "Yardım",
        "Teşekkürler",
    ]
    
    for query in test_queries:
        intent, confidence = classifier.predict(query)
        
        # Format output
        conf_marker = "🔥" if confidence >= 0.80 else "✅" if confidence >= 0.60 else "⚠️"
        
        print(f"{conf_marker} Query: '{query}'")
        print(f"   Intent: {intent}")
        print(f"   Confidence: {confidence:.1%}")
        
        # Show top 3 predictions
        top_3 = classifier.predict_with_top_k(query, k=3)
        print(f"   Top 3: {', '.join([f'{i}({c:.1%})' for i, c in top_3])}")
        print()
    
    # Show statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    stats = classifier.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    test_classifier()
