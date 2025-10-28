#!/usr/bin/env python3
"""
DistilBERT Intent Classifier for AI Istanbul
============================================

This module provides a robust multilingual intent classifier using DistilBERT
for both Turkish and English queries across all required categories.
"""

import torch
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntentPrediction:
    """Intent prediction result"""
    intent: str
    confidence: float
    probability_distribution: Dict[str, float]
    all_intents: List[Tuple[str, float]]  # All intents with probabilities
    metadata: Dict[str, any] = field(default_factory=dict)


class DistilBERTIntentClassifier:
    """
    DistilBERT-based intent classifier for multilingual intent detection
    """
    
    def __init__(self, model_path="models/distilbert_intent_classifier"):
        """
        Initialize the DistilBERT intent classifier
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.intent_mapping = None
        self.intents = []
        self.is_loaded = False
        
        # Try to load the model
        try:
            self._load_model()
        except Exception as e:
            logger.warning(f"⚠️  Could not load model from {model_path}: {e}")
            logger.warning("   Intent classifier will use fallback methods until model is trained")
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        logger.info(f"📥 Loading DistilBERT intent classifier from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Load intent mapping
        mapping_file = self.model_path / 'intent_mapping.json'
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.intent_mapping = json.load(f)
                self.intents = self.intent_mapping['intents']
                logger.info(f"✅ Loaded {len(self.intents)} intent classes: {', '.join(self.intents)}")
        else:
            raise FileNotFoundError(f"Intent mapping not found: {mapping_file}")
        
        self.is_loaded = True
        logger.info(f"✅ DistilBERT intent classifier loaded successfully on {self.device}")
    
    def predict(self, text: str, return_all_scores: bool = False) -> IntentPrediction:
        """
        Predict intent for given text
        
        Args:
            text: Input text to classify
            return_all_scores: If True, return probabilities for all intents
        
        Returns:
            IntentPrediction object with predicted intent and confidence
        """
        if not self.is_loaded:
            # Return unknown intent with low confidence if model not loaded
            return IntentPrediction(
                intent="unknown",
                confidence=0.0,
                probability_distribution={"unknown": 0.0},
                all_intents=[("unknown", 0.0)],
                metadata={"error": "Model not loaded"}
            )
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]
        
        # Get top prediction
        top_prob, top_idx = torch.max(probabilities, dim=0)
        predicted_intent = self.intents[top_idx.item()]
        confidence = top_prob.item()
        
        # Get all probabilities
        all_probs = probabilities.cpu().numpy()
        probability_distribution = {
            intent: float(prob) 
            for intent, prob in zip(self.intents, all_probs)
        }
        
        # Get top k intents
        top_k = min(5, len(self.intents))
        top_k_indices = torch.topk(probabilities, top_k).indices.cpu().numpy()
        all_intents = [
            (self.intents[idx], float(probabilities[idx].item()))
            for idx in top_k_indices
        ]
        
        return IntentPrediction(
            intent=predicted_intent,
            confidence=confidence,
            probability_distribution=probability_distribution,
            all_intents=all_intents,
            metadata={
                "text_length": len(text),
                "device": str(self.device)
            }
        )
    
    def batch_predict(self, texts: List[str]) -> List[IntentPrediction]:
        """
        Predict intents for multiple texts
        
        Args:
            texts: List of input texts
        
        Returns:
            List of IntentPrediction objects
        """
        if not self.is_loaded:
            return [
                IntentPrediction(
                    intent="unknown",
                    confidence=0.0,
                    probability_distribution={"unknown": 0.0},
                    all_intents=[("unknown", 0.0)],
                    metadata={"error": "Model not loaded"}
                )
                for _ in texts
            ]
        
        # Tokenize all inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Convert to IntentPrediction objects
        results = []
        for i, text in enumerate(texts):
            probs = probabilities[i]
            top_prob, top_idx = torch.max(probs, dim=0)
            predicted_intent = self.intents[top_idx.item()]
            confidence = top_prob.item()
            
            # Get all probabilities
            all_probs = probs.cpu().numpy()
            probability_distribution = {
                intent: float(prob) 
                for intent, prob in zip(self.intents, all_probs)
            }
            
            # Get top k intents
            top_k = min(5, len(self.intents))
            top_k_indices = torch.topk(probs, top_k).indices.cpu().numpy()
            all_intents = [
                (self.intents[idx], float(probs[idx].item()))
                for idx in top_k_indices
            ]
            
            results.append(IntentPrediction(
                intent=predicted_intent,
                confidence=confidence,
                probability_distribution=probability_distribution,
                all_intents=all_intents,
                metadata={"text_length": len(text)}
            ))
        
        return results
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.is_loaded
    
    def get_supported_intents(self) -> List[str]:
        """Get list of supported intents"""
        return self.intents if self.is_loaded else []


# Global instance
_classifier_instance = None


def get_intent_classifier(model_path="models/distilbert_intent_classifier") -> DistilBERTIntentClassifier:
    """
    Get or create global intent classifier instance
    
    Args:
        model_path: Path to the trained model
    
    Returns:
        DistilBERTIntentClassifier instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = DistilBERTIntentClassifier(model_path)
    
    return _classifier_instance


def classify_intent(text: str, model_path="models/distilbert_intent_classifier") -> IntentPrediction:
    """
    Classify intent for given text
    
    Args:
        text: Input text
        model_path: Path to the trained model
    
    Returns:
        IntentPrediction with predicted intent and confidence
    """
    classifier = get_intent_classifier(model_path)
    return classifier.predict(text)


def batch_classify_intent(texts: List[str], model_path="models/distilbert_intent_classifier") -> List[IntentPrediction]:
    """
    Classify intents for multiple texts
    
    Args:
        texts: List of input texts
        model_path: Path to the trained model
    
    Returns:
        List of IntentPrediction objects
    """
    classifier = get_intent_classifier(model_path)
    return classifier.batch_predict(texts)


if __name__ == "__main__":
    # Test the classifier
    print("\n" + "="*80)
    print("🧪 TESTING DISTILBERT INTENT CLASSIFIER")
    print("="*80 + "\n")
    
    # Test queries in Turkish and English
    test_queries = [
        "Merhaba",
        "Hello",
        "Nasılsın?",
        "How are you?",
        "Teşekkür ederim",
        "Thank you",
        "Güle güle",
        "Goodbye",
        "Galata Kulesi'ne nasıl gidebilirim?",
        "How do I get to Galata Tower?",
        "Taksim'de iyi bir restoran önerir misin?",
        "Can you recommend a good restaurant in Taksim?",
        "Yarın hava nasıl olacak?",
        "What will the weather be like tomorrow?",
        "Sultanahmet'te gezilecek yerler",
        "Places to visit in Sultanahmet",
        "İstanbul'da bu hafta sonu etkinlikler",
        "Events in Istanbul this weekend"
    ]
    
    classifier = get_intent_classifier()
    
    if not classifier.is_model_loaded():
        print("⚠️  Model not loaded. Please train the model first using:")
        print("   python train_distilbert_intent_classifier.py")
        exit(1)
    
    print(f"✅ Model loaded with {len(classifier.get_supported_intents())} intents\n")
    
    for query in test_queries:
        result = classify_intent(query)
        print(f"Query: {query}")
        print(f"  Intent: {result.intent}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Top 3 intents: {result.all_intents[:3]}")
        print()
    
    print("="*80)
    print("✅ Testing complete!")
    print("="*80 + "\n")
