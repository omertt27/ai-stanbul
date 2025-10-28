#!/usr/bin/env python3
"""
Comprehensive Intent Classifier Training Script
================================================

This script trains a multilingual (Turkish-English) intent classifier
for the AI Istanbul system, covering all required intent categories:
- Daily talk (greetings, thanks, farewell, help, activity suggestions)
- Places/Attractions
- Neighborhoods
- Transportation
- Weather
- Local tips (restaurants, hidden gems, shopping)
- Events
- Route planning

The trained model uses TF-IDF features with ensemble ML models
for robust intent detection with high confidence scores.
"""

import json
import logging
import os
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveIntentClassifier:
    """
    Multilingual intent classifier for AI Istanbul system
    """
    
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.classifier = None
        self.intent_categories = {}
        self.model_metadata = {
            'training_date': None,
            'num_samples': 0,
            'num_intents': 0,
            'languages': [],
            'accuracy': 0.0
        }
    
    def load_training_data(self, json_path: str) -> Tuple[List[str], List[str]]:
        """Load training data from JSON file"""
        
        logger.info(f"Loading training data from {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        training_data = data['training_data']
        self.intent_categories = data['intent_categories']
        
        texts = [item['text'] for item in training_data]
        intents = [item['intent'] for item in training_data]
        languages = list(set([item['language'] for item in training_data]))
        
        logger.info(f"✅ Loaded {len(texts)} training samples")
        logger.info(f"📊 Intents: {set(intents)}")
        logger.info(f"🌍 Languages: {languages}")
        
        self.model_metadata['num_samples'] = len(texts)
        self.model_metadata['num_intents'] = len(set(intents))
        self.model_metadata['languages'] = languages
        
        return texts, intents
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for feature extraction"""
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def train(self, texts: List[str], intents: List[str]):
        """Train the intent classifier"""
        
        logger.info("🚀 Starting model training...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Initialize TF-IDF vectorizer with multilingual support
        # Use character n-grams to capture both Turkish and English patterns
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=5000,
            min_df=1,
            max_df=0.9,
            sublinear_tf=True
        )
        
        # Fit and transform texts
        X = self.vectorizer.fit_transform(processed_texts)
        logger.info(f"📊 Feature matrix shape: {X.shape}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(intents)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        
        # Create ensemble classifier
        # Combine Random Forest (good for patterns) and MLP (good for complex relationships)
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Voting classifier for ensemble
        self.classifier = VotingClassifier(
            estimators=[
                ('rf', rf_classifier),
                ('mlp', mlp_classifier)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        # Train the ensemble
        logger.info("🎓 Training ensemble classifier (Random Forest + MLP)...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        train_accuracy = self.classifier.score(X_train, y_train)
        test_accuracy = self.classifier.score(X_test, y_test)
        
        logger.info(f"✅ Training accuracy: {train_accuracy:.4f}")
        logger.info(f"✅ Test accuracy: {test_accuracy:.4f}")
        
        self.model_metadata['accuracy'] = test_accuracy
        self.model_metadata['training_date'] = datetime.now().isoformat()
        
        # Cross-validation score
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)
        logger.info(f"✅ Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Detailed evaluation
        y_pred = self.classifier.predict(X_test)
        intent_names = self.label_encoder.classes_
        
        logger.info("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=intent_names, zero_division=0))
        
        logger.info("\n🎯 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, text: str, return_probabilities: bool = True) -> Dict:
        """Predict intent for given text"""
        
        if not self.classifier or not self.vectorizer:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed_text])
        
        # Predict
        predicted_label = self.classifier.predict(X)[0]
        predicted_intent = self.label_encoder.inverse_transform([predicted_label])[0]
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = float(probabilities[predicted_label])
        
        result = {
            'primary': predicted_intent,
            'confidence': confidence,
            'category': self.intent_categories.get(predicted_intent, 'general')
        }
        
        if return_probabilities:
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_intents = self.label_encoder.inverse_transform(top_indices)
            top_confidences = probabilities[top_indices]
            
            result['secondary'] = [
                {'intent': intent, 'confidence': float(conf)}
                for intent, conf in zip(top_intents[1:], top_confidences[1:])
            ]
        
        return result
    
    def save_model(self, model_dir: str = 'models'):
        """Save trained model and metadata"""
        
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'intent_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'intent_vectorizer.joblib')
        encoder_path = os.path.join(model_dir, 'intent_label_encoder.joblib')
        metadata_path = os.path.join(model_dir, 'intent_model_metadata.json')
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'model_metadata': self.model_metadata,
                'intent_categories': self.intent_categories
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Model saved to {model_dir}/")
        logger.info(f"   - Classifier: {model_path}")
        logger.info(f"   - Vectorizer: {vectorizer_path}")
        logger.info(f"   - Label Encoder: {encoder_path}")
        logger.info(f"   - Metadata: {metadata_path}")
    
    def load_model(self, model_dir: str = 'models'):
        """Load trained model and metadata"""
        
        model_path = os.path.join(model_dir, 'intent_classifier.joblib')
        vectorizer_path = os.path.join(model_dir, 'intent_vectorizer.joblib')
        encoder_path = os.path.join(model_dir, 'intent_label_encoder.joblib')
        metadata_path = os.path.join(model_dir, 'intent_model_metadata.json')
        
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.model_metadata = data['model_metadata']
            self.intent_categories = data['intent_categories']
        
        logger.info(f"✅ Model loaded from {model_dir}/")
        logger.info(f"   - Training date: {self.model_metadata['training_date']}")
        logger.info(f"   - Accuracy: {self.model_metadata['accuracy']:.4f}")
        logger.info(f"   - Intents: {self.model_metadata['num_intents']}")


def train_comprehensive_intent_classifier():
    """Main training function"""
    
    logger.info("=" * 80)
    logger.info("🚀 AI ISTANBUL - COMPREHENSIVE INTENT CLASSIFIER TRAINING")
    logger.info("=" * 80)
    
    # Initialize classifier
    classifier = ComprehensiveIntentClassifier()
    
    # Load training data
    training_data_path = 'comprehensive_intent_training_data.json'
    texts, intents = classifier.load_training_data(training_data_path)
    
    # Train model
    results = classifier.train(texts, intents)
    
    # Save model
    classifier.save_model()
    
    logger.info("=" * 80)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info(f"   - Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"   - CV Accuracy: {results['cv_accuracy']:.4f} (+/- {results['cv_std']:.4f})")
    logger.info("=" * 80)
    
    # Test with sample queries
    logger.info("\n🧪 Testing with sample queries:")
    
    test_queries = [
        "Merhaba",
        "Hello",
        "Teşekkürler",
        "Ayasofya'yı görmek istiyorum",
        "How to get to Sultanahmet",
        "Hava durumu nasıl",
        "Restaurant recommendation",
        "Bugün hangi etkinlik var",
        "Plan a route",
        "Gizli mekanlar"
    ]
    
    for query in test_queries:
        result = classifier.predict(query)
        logger.info(f"\n'{query}'")
        logger.info(f"  → Intent: {result['primary']}")
        logger.info(f"  → Confidence: {result['confidence']:.4f}")
        logger.info(f"  → Category: {result['category']}")
    
    return classifier


if __name__ == '__main__':
    train_comprehensive_intent_classifier()
