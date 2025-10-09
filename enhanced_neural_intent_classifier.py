#!/usr/bin/env python3
"""
Enhanced Neural Intent Classifier
=================================

Advanced deep learning-based intent classification system with:
- Transformer-based architecture
- Fine-tuning capabilities
- Multi-lingual support
- Context-aware predictions
- Real-time learning adaptation

This system significantly improves intent classification accuracy through
advanced neural networks and contextual understanding.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import re

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, EarlyStoppingCallback
    )
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class IntentPrediction:
    """Enhanced intent prediction result"""
    intent: str
    confidence: float
    probability_distribution: Dict[str, float]
    context_influence: Dict[str, float]
    sub_intents: List[str] = field(default_factory=list)
    uncertainty_score: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class TurkishTourismIntentDataset(Dataset):
    """Dataset for Turkish tourism intent classification"""
    
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class EnhancedNeuralIntentClassifier:
    """Advanced neural intent classifier with transformer architecture"""
    
    def __init__(self, model_name: str = "distilbert-base-multilingual-cased"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.intent_categories = []
        self.context_weights = {}
        self.fine_tuning_data = []
        self.performance_history = []
        
        # Istanbul-specific intent categories
        self.default_intents = [
            'restaurant_search',
            'restaurant_info', 
            'attraction_search',
            'attraction_info',
            'transport_route',
            'transport_info',
            'hotel_search',
            'hotel_info',
            'general_info',
            'practical_info',
            'recommendation_request',
            'comparison_request',
            'reservation_request',
            'navigation_help',
            'cultural_inquiry',
            'historical_question',
            'weather_inquiry',
            'event_search',
            'shopping_inquiry',
            'nightlife_search',
            'emergency_help',
            'language_help',
            'photo_request',
            'price_inquiry',
            'time_schedule',
            'greeting',
            'goodbye',
            'emotional_expression',
            'unknown'
        ]
        
        if ML_AVAILABLE:
            self._initialize_model()
            # Auto-train on Istanbul data for better performance
            if self.model:
                self._quick_train_on_istanbul_data()
    
    def _initialize_model(self):
        """Initialize the transformer model and tokenizer"""
        try:
            logger.info(f"🧠 Initializing neural intent classifier with {self.model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize model for classification
            num_labels = len(self.default_intents)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels,
                problem_type="single_label_classification"
            )
            
            # Set label encoder
            self.label_encoder.fit(self.default_intents)
            self.intent_categories = self.default_intents
            
            logger.info(f"✅ Neural intent classifier initialized with {num_labels} intent categories")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize neural intent classifier: {e}")
            self.model = None
            self.tokenizer = None
    
    def _create_training_data(self) -> Tuple[List[str], List[str]]:
        """Create training data for Istanbul tourism intents"""
        training_examples = [
            # Restaurant queries
            ("Best Turkish restaurants in Sultanahmet", "restaurant_search"),
            ("Where can I find authentic kebab?", "restaurant_search"),
            ("Recommend a good seafood restaurant", "restaurant_search"),
            ("What's the menu at Pandeli restaurant?", "restaurant_info"),
            ("Is Hamdi restaurant halal?", "restaurant_info"),
            ("When does Çiya Sofrası close?", "restaurant_info"),
            
            # Attraction queries
            ("What time does Hagia Sophia open?", "attraction_info"),
            ("How much does Blue Mosque tour cost?", "attraction_info"),
            ("Best historical sites in Istanbul", "attraction_search"),
            ("Show me Byzantine monuments", "attraction_search"),
            ("Photos of Galata Tower", "photo_request"),
            
            # Transport queries
            ("How to get to Taksim from airport?", "transport_route"),
            ("Best way to cross Bosphorus", "transport_route"),
            ("Metro schedule to Sultanahmet", "transport_info"),
            ("Taxi fare to Grand Bazaar", "price_inquiry"),
            
            # Hotel queries
            ("Luxury hotels in Beyoğlu", "hotel_search"),
            ("Budget accommodation near Sultanahmet", "hotel_search"),
            ("Hotel room availability tonight", "hotel_info"),
            ("Cancel my hotel booking", "reservation_request"),
            
            # General info
            ("What's the weather today?", "weather_inquiry"),
            ("Currency exchange rate", "practical_info"),
            ("Turkish phrases for tourists", "language_help"),
            ("Emergency numbers in Turkey", "emergency_help"),
            
            # Cultural and historical
            ("Ottoman Empire history", "historical_question"),
            ("Turkish cultural traditions", "cultural_inquiry"),
            ("Local festivals this month", "event_search"),
            
            # Shopping and nightlife
            ("Best shopping malls", "shopping_inquiry"),
            ("Nightlife in Beyoğlu", "nightlife_search"),
            ("Traditional Turkish souvenirs", "shopping_inquiry"),
            
            # Emotional expressions
            ("I'm excited about Istanbul!", "emotional_expression"),
            ("Feeling lost and confused", "emotional_expression"),
            ("This place is amazing!", "emotional_expression"),
            
            # Multi-lingual examples
            ("Sultanahmet'te en iyi restoran hangisi?", "restaurant_search"),
            ("Ayasofya'ya nasıl giderim?", "transport_route"),
            ("Bu fiyat çok pahalı", "price_inquiry"),
            ("Teşekkür ederim", "greeting"),
            
            # Complex queries
            ("Compare prices of Bosphorus cruise tours", "comparison_request"),
            ("Book a table for 4 people at sunset", "reservation_request"),
            ("Plan my 3-day Istanbul itinerary", "recommendation_request"),
            ("Navigate me to the nearest mosque", "navigation_help")
        ]
        
        # Add synthetic data augmentation
        augmented_examples = []
        for text, intent in training_examples:
            # Add variations
            variations = [
                text.lower(),
                text.upper(),
                f"Can you help me with: {text}",
                f"I need to know about {text}",
                f"Please tell me {text}"
            ]
            for variation in variations:
                augmented_examples.append((variation, intent))
        
        all_examples = training_examples + augmented_examples
        texts, labels = zip(*all_examples)
        
        return list(texts), list(labels)
    
    def train_model(self, custom_training_data: Optional[List[Tuple[str, str]]] = None, 
                   epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5):
        """Train the neural intent classifier"""
        if not ML_AVAILABLE or not self.model:
            logger.error("❌ ML libraries not available or model not initialized")
            return False
        
        try:
            logger.info("🎯 Training neural intent classifier...")
            
            # Prepare training data
            if custom_training_data:
                texts, labels = zip(*custom_training_data)
                texts, labels = list(texts), list(labels)
            else:
                texts, labels = self._create_training_data()
            
            # Encode labels
            encoded_labels = self.label_encoder.transform(labels)
            
            # Create dataset
            dataset = TurkishTourismIntentDataset(
                texts, encoded_labels, self.tokenizer
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./intent_classifier_model',
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                learning_rate=learning_rate,
                logging_dir='./logs',
                logging_steps=50,
                save_strategy="epoch",
                evaluation_strategy="no",
                load_best_model_at_end=False,
                dataloader_num_workers=0  # Set to 0 to avoid multiprocessing issues
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer
            )
            
            # Train the model
            trainer.train()
            
            # Save the model
            trainer.save_model('./intent_classifier_model')
            self.tokenizer.save_pretrained('./intent_classifier_model')
            
            logger.info("✅ Neural intent classifier training completed")
            
            # Record training metrics
            training_info = {
                "timestamp": datetime.now().isoformat(),
                "training_samples": len(texts),
                "unique_intents": len(set(labels)),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            }
            self.performance_history.append(training_info)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training neural intent classifier: {e}")
            return False
    
    def predict_intent(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentPrediction:
        """Predict intent with enhanced neural processing and pattern fallback"""
        
        # Try neural prediction first if available
        if ML_AVAILABLE and self.model and self.tokenizer:
            try:
                return self._predict_with_neural_model(query, context)
            except Exception as e:
                logger.warning(f"⚠️ Neural prediction failed, using pattern fallback: {e}")
        
        # Use pattern-based fallback for robust prediction
        return self._predict_with_patterns(query, context)
    
    def _predict_with_neural_model(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentPrediction:
        """Neural model prediction method"""
        
        # Tokenize input
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            max_length=512,
            padding=True,
            truncation=True
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predictions
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        base_confidence = float(probabilities[0][predicted_idx])
        
        # Boost confidence with pattern matching for Istanbul-specific queries
        pattern_boost = self._get_pattern_boost(query)
        enhanced_confidence = min(1.0, base_confidence + pattern_boost)
        
        # Decode prediction
        predicted_intent = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        # Create probability distribution
        probability_dist = {}
        for i, intent in enumerate(self.intent_categories):
            probability_dist[intent] = float(probabilities[0][i])
        
        # Sort by probability
        sorted_probs = sorted(probability_dist.items(), key=lambda x: x[1], reverse=True)
        
        # Context influence analysis
        context_influence = self._analyze_context_influence(query, context, sorted_probs)
        
        # Uncertainty estimation
        entropy = -sum(p * np.log(p + 1e-10) for p in probabilities[0].numpy())
        uncertainty_score = float(entropy / np.log(len(self.intent_categories)))
        
        # Generate explanation
        explanation = self._generate_explanation(
            query, predicted_intent, enhanced_confidence, sorted_probs[:3]
        )
        
        # Identify sub-intents
        sub_intents = [intent for intent, prob in sorted_probs[1:4] if prob > 0.1]
        
        return IntentPrediction(
            intent=predicted_intent,
            confidence=enhanced_confidence,
            probability_distribution=probability_dist,
            context_influence=context_influence,
            sub_intents=sub_intents,
            uncertainty_score=uncertainty_score,
            explanation=explanation,
            metadata={
                "model_name": self.model_name,
                "top_3_predictions": sorted_probs[:3],
                "query_length": len(query),
                "timestamp": datetime.now().isoformat(),
                "pattern_boost": pattern_boost
            }
        )
    
    def _predict_with_patterns(self, query: str, context: Optional[Dict[str, Any]] = None) -> IntentPrediction:
        """Robust pattern-based intent prediction as fallback"""
        
        # Enhanced pattern matching for Istanbul tourism
        intent_patterns = {
            'restaurant_search': {
                'patterns': [
                    r'(restaurant|dining|food|eat|cuisine|kebab|lokanta)',
                    r'(where.*eat|find.*restaurant|good.*food|best.*restaurant)',
                    r'(hungry|dinner|lunch|breakfast|meal|yemek)',
                    r'(turkish food|ottoman cuisine|seafood|meze)'
                ],
                'weight': 1.0
            },
            'restaurant_info': {
                'patterns': [
                    r'(menu|price|cost|opening|closing|hours|reservation)',
                    r'(halal|vegetarian|vegan|dietary)',
                    r'(address|location|phone|contact)',
                    r'(review|rating|quality)'
                ],
                'weight': 0.9
            },
            'attraction_search': {
                'patterns': [
                    r'(attraction|tourist|visit|see|sightseeing|monument)',
                    r'(hagia sophia|blue mosque|topkapi|galata tower|bosphorus)',
                    r'(museum|palace|church|mosque|historical|byzantine)',
                    r'(photo|picture|instagram|scenic)'
                ],
                'weight': 1.0
            },
            'attraction_info': {
                'patterns': [
                    r'(opening.*hours|ticket.*price|entrance.*fee)',
                    r'(how.*long|duration|time.*spend)',
                    r'(guide|tour|audio.*guide)',
                    r'(history|story|about.*this)'
                ],
                'weight': 0.9
            },
            'transport_route': {
                'patterns': [
                    r'(how.*get|transport|metro|bus|taxi|tram)',
                    r'(route|direction|travel|go.*to|from.*to)',
                    r'(airport|ferry|subway|train|dolmus)',
                    r'(quickest|fastest|cheapest.*way)'
                ],
                'weight': 1.0
            },
            'transport_info': {
                'patterns': [
                    r'(schedule|timetable|frequency|when.*next)',
                    r'(fare|price|cost|ticket)',
                    r'(stop|station|terminal)',
                    r'(card|payment|istanbulkart)'
                ],
                'weight': 0.9
            },
            'hotel_search': {
                'patterns': [
                    r'(hotel|accommodation|stay|room|booking|otel)',
                    r'(where.*stay|place.*sleep|hostel|guest.*house)',
                    r'(luxury|budget|cheap|expensive)',
                    r'(near|close.*to|walking.*distance)'
                ],
                'weight': 1.0
            },
            'price_inquiry': {
                'patterns': [
                    r'(price|cost|expensive|cheap|how.*much|budget)',
                    r'(fee|ticket|entrance|fare|ücret)',
                    r'(money|lira|dollar|euro|currency)',
                    r'(worth.*it|value.*for.*money)'
                ],
                'weight': 0.8
            },
            'weather_inquiry': {
                'patterns': [
                    r'(weather|temperature|rain|sunny|cloudy|forecast)',
                    r'(what.*weather|how.*weather|climate)',
                    r'(umbrella|jacket|clothes.*wear)',
                    r'(hot|cold|warm|cool)'
                ],
                'weight': 0.7
            },
            'greeting': {
                'patterns': [
                    r'(hello|hi|hey|good.*morning|good.*afternoon)',
                    r'(merhaba|selam|iyi.*günler)',
                    r'(help.*me|can.*you|please)',
                    r'(thank.*you|teşekkür)'
                ],
                'weight': 0.6
            }
        }
        
        query_lower = query.lower()
        intent_scores = {}
        
        # Calculate scores for each intent
        for intent, data in intent_patterns.items():
            total_score = 0
            matches = 0
            
            for pattern in data['patterns']:
                if re.search(pattern, query_lower):
                    total_score += data['weight']
                    matches += 1
            
            if matches > 0:
                # Normalize score and apply boosts
                normalized_score = total_score / len(data['patterns'])
                
                # Boost for multiple pattern matches
                if matches > 1:
                    normalized_score *= 1.2
                
                # Istanbul-specific boost
                if 'istanbul' in query_lower or 'turkey' in query_lower:
                    normalized_score *= 1.1
                
                intent_scores[intent] = min(1.0, normalized_score)
        
        # Determine best intent
        if not intent_scores:
            predicted_intent = "general_info"
            confidence = 0.4
        else:
            predicted_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
            confidence = intent_scores[predicted_intent]
            
            # Apply final confidence boost for high-confidence predictions
            if confidence > 0.7:
                confidence = min(1.0, confidence * 1.1)
        
        # Create probability distribution
        probability_dist = {intent: score for intent, score in intent_scores.items()}
        
        # Add zero scores for missing intents
        for intent in self.intent_categories:
            if intent not in probability_dist:
                probability_dist[intent] = 0.0
        
        # Context influence
        context_influence = self._analyze_context_influence(query, context, [(predicted_intent, confidence)])
        
        return IntentPrediction(
            intent=predicted_intent,
            confidence=confidence,
            probability_distribution=probability_dist,
            context_influence=context_influence,
            explanation=f"Pattern-based prediction: '{predicted_intent}' with {confidence:.1%} confidence",
            metadata={
                "method": "pattern_matching",
                "matches_found": len(intent_scores),
                "top_patterns": list(intent_scores.keys())[:3]
            }
        )
    
    def _get_pattern_boost(self, query: str) -> float:
        """Calculate confidence boost based on pattern matching"""
        query_lower = query.lower()
        boost = 0.0
        
        # Istanbul-specific terms
        istanbul_terms = ['istanbul', 'sultanahmet', 'galata', 'bosphorus', 'taksim', 'beyoglu']
        if any(term in query_lower for term in istanbul_terms):
            boost += 0.2
        
        # Turkish language indicators  
        turkish_chars = ['ç', 'ğ', 'ı', 'ö', 'ş', 'ü']
        if any(char in query_lower for char in turkish_chars):
            boost += 0.15
        
        # Tourism keywords
        tourism_terms = ['restaurant', 'hotel', 'museum', 'tour', 'attraction']
        if any(term in query_lower for term in tourism_terms):
            boost += 0.1
        
        return min(0.4, boost)  # Cap boost at 0.4

    def _analyze_context_influence(self, query: str, context: Optional[Dict[str, Any]], 
                                 sorted_probs: List[Tuple[str, float]]) -> Dict[str, float]:
        """Analyze how context influences intent prediction"""
        context_influence = {
            "temporal": 0.0,
            "user_type": 0.0,
            "location": 0.0,
            "query_history": 0.0,
            "language": 0.0
        }
        
        if not context:
            return context_influence
        
        # Temporal influence
        hour = context.get("current_hour", 12)
        if "restaurant" in sorted_probs[0][0] and (11 <= hour <= 14 or 18 <= hour <= 22):
            context_influence["temporal"] = 0.2
        elif "nightlife" in sorted_probs[0][0] and (20 <= hour or hour <= 2):
            context_influence["temporal"] = 0.3
        
        # User type influence
        user_type = context.get("user_type", "tourist")
        if user_type == "tourist" and any(intent in sorted_probs[0][0] 
                                         for intent in ["attraction", "photo", "transport"]):
            context_influence["user_type"] = 0.15
        elif user_type == "local" and "recommendation" in sorted_probs[0][0]:
            context_influence["user_type"] = 0.1
        
        # Language influence
        if any(char in query for char in "çğıöşüÇĞIÖŞÜ"):
            context_influence["language"] = 0.1
        
        return context_influence
    
    def _generate_explanation(self, query: str, intent: str, confidence: float, 
                            top_predictions: List[Tuple[str, float]]) -> str:
        """Generate human-readable explanation for the prediction"""
        
        explanation_parts = [
            f"Query '{query[:50]}...' classified as '{intent}' with {confidence:.1%} confidence."
        ]
        
        if confidence > 0.8:
            explanation_parts.append("High confidence prediction.")
        elif confidence > 0.6:
            explanation_parts.append("Moderate confidence prediction.")
        else:
            explanation_parts.append("Low confidence prediction - consider multiple intents.")
        
        if len(top_predictions) > 1:
            alternatives = ", ".join([f"{intent} ({prob:.1%})" for intent, prob in top_predictions[1:]])
            explanation_parts.append(f"Alternative interpretations: {alternatives}")
        
        return " ".join(explanation_parts)
    
    def add_training_example(self, query: str, intent: str, feedback_score: float = 1.0):
        """Add new training example from user feedback"""
        if intent in self.intent_categories:
            self.fine_tuning_data.append({
                "query": query,
                "intent": intent,
                "feedback_score": feedback_score,
                "timestamp": datetime.now().isoformat()
            })
            
            # Trigger re-training if enough new data
            if len(self.fine_tuning_data) >= 50:
                self._incremental_learning()
    
    def _incremental_learning(self):
        """Perform incremental learning with new data"""
        try:
            logger.info("🔄 Performing incremental learning...")
            
            # Prepare new training data
            new_texts = [item["query"] for item in self.fine_tuning_data]
            new_labels = [item["intent"] for item in self.fine_tuning_data]
            
            # Create smaller dataset for fine-tuning
            encoded_labels = self.label_encoder.transform(new_labels)
            dataset = TurkishTourismIntentDataset(
                new_texts, encoded_labels, self.tokenizer
            )
            
            # Fine-tune with lower learning rate
            training_args = TrainingArguments(
                output_dir='./intent_classifier_incremental',
                num_train_epochs=1,
                per_device_train_batch_size=8,
                learning_rate=1e-5,
                logging_steps=10,
                save_strategy="no",
                dataloader_num_workers=0
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=self.tokenizer
            )
            
            trainer.train()
            
            # Clear fine-tuning data
            self.fine_tuning_data = []
            
            logger.info("✅ Incremental learning completed")
            
        except Exception as e:
            logger.error(f"❌ Error in incremental learning: {e}")
    
    def evaluate_model(self, test_data: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        if not test_data:
            return {"error": "No test data provided"}
        
        predictions = []
        true_labels = []
        
        for query, true_intent in test_data:
            prediction = self.predict_intent(query)
            predictions.append(prediction.intent)
            true_labels.append(true_intent)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Create classification report
        try:
            report = classification_report(
                true_labels, predictions, 
                target_names=list(set(true_labels)),
                output_dict=True,
                zero_division=0
            )
        except Exception:
            report = {}
        
        return {
            "accuracy": accuracy,
            "total_samples": len(test_data),
            "classification_report": report,
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "model_available": self.model is not None,
            "intent_categories": self.intent_categories,
            "num_categories": len(self.intent_categories),
            "training_history": self.performance_history,
            "fine_tuning_samples": len(self.fine_tuning_data),
            "ml_libraries_available": ML_AVAILABLE
        }
    
    def save_model(self, save_path: str = "./enhanced_intent_classifier"):
        """Save the trained model"""
        if not self.model or not self.tokenizer:
            return False
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save additional metadata
            metadata = {
                "intent_categories": self.intent_categories,
                "performance_history": self.performance_history,
                "model_name": self.model_name
            }
            
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Model saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, load_path: str = "./enhanced_intent_classifier"):
        """Load a pre-trained model"""
        try:
            if not os.path.exists(load_path):
                logger.error(f"❌ Model path not found: {load_path}")
                return False
            
            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
            self.tokenizer = AutoTokenizer.from_pretrained(load_path)
            
            # Load metadata
            metadata_path = os.path.join(load_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                
                self.intent_categories = metadata.get("intent_categories", self.default_intents)
                self.performance_history = metadata.get("performance_history", [])
                self.model_name = metadata.get("model_name", self.model_name)
            
            # Re-fit label encoder
            self.label_encoder.fit(self.intent_categories)
            
            logger.info(f"✅ Model loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def _quick_train_on_istanbul_data(self):
        """Quick training on Istanbul-specific data for better predictions"""
        try:
            logger.info("🏋️ Quick training on Istanbul tourism data...")
            
            # Create focused Istanbul training data
            istanbul_training_data = [
                # High-confidence restaurant queries
                ("Best Turkish restaurants in Sultanahmet", "restaurant_search"),
                ("Where can I find authentic kebab in Istanbul?", "restaurant_search"),
                ("Recommend good seafood restaurant near Galata", "restaurant_search"),
                ("Ottoman cuisine restaurants in old city", "restaurant_search"),
                ("What's the menu at Pandeli restaurant?", "restaurant_info"),
                ("Is Hamdi restaurant halal?", "restaurant_info"),
                ("When does Çiya Sofrası close?", "restaurant_info"),
                ("Reservation at Mikla restaurant", "restaurant_info"),
                
                # High-confidence attraction queries
                ("What time does Hagia Sophia open?", "attraction_info"),
                ("Blue Mosque entrance fee", "attraction_info"),
                ("Best historical sites in Istanbul", "attraction_search"),
                ("Byzantine monuments in Istanbul", "attraction_search"),
                ("Topkapi Palace tour guide", "attraction_info"),
                ("Photos of Galata Tower", "photo_request"),
                ("Bosphorus cruise tours", "attraction_search"),
                
                # High-confidence transport queries
                ("How to get to Taksim from airport?", "transport_route"),
                ("Metro route to Sultanahmet", "transport_route"),
                ("Best way to cross Bosphorus", "transport_route"),
                ("Taxi fare to Grand Bazaar", "price_inquiry"),
                ("Metro schedule to Sultanahmet", "transport_info"),
                ("Ferry to Princes Islands", "transport_route"),
                
                # Turkish language examples
                ("Sultanahmet'te en iyi restoran hangisi?", "restaurant_search"),
                ("Ayasofya'ya nasıl giderim?", "transport_route"),
                ("Bu fiyat çok pahalı", "price_inquiry"),
                ("Teşekkür ederim", "greeting"),
                ("Galata Kulesi'ne metro ile nasıl gidilir?", "transport_route"),
                ("İyi bir otel önerir misiniz?", "hotel_search")
            ]
            
            # Quick fine-tuning (1 epoch, small batch)
            success = self.train_model(
                custom_training_data=istanbul_training_data, 
                epochs=1, 
                batch_size=8, 
                learning_rate=1e-5
            )
            
            if success:
                logger.info("✅ Quick Istanbul training completed")
            else:
                logger.warning("⚠️ Quick training failed, using untrained model")
                
        except Exception as e:
            logger.warning(f"⚠️ Quick training failed: {e}, continuing with pattern fallback")

# Global instance
_enhanced_intent_classifier = None

def get_enhanced_intent_classifier() -> EnhancedNeuralIntentClassifier:
    """Get the global enhanced intent classifier instance"""
    global _enhanced_intent_classifier
    if _enhanced_intent_classifier is None:
        _enhanced_intent_classifier = EnhancedNeuralIntentClassifier()
    return _enhanced_intent_classifier

# Convenience functions
def predict_intent_enhanced(query: str, context: Optional[Dict[str, Any]] = None) -> IntentPrediction:
    """Predict intent using enhanced neural classifier"""
    classifier = get_enhanced_intent_classifier()
    return classifier.predict_intent(query, context)

def train_intent_classifier(training_data: Optional[List[Tuple[str, str]]] = None, 
                          epochs: int = 3) -> bool:
    """Train the enhanced intent classifier"""
    classifier = get_enhanced_intent_classifier()
    return classifier.train_model(training_data, epochs=epochs)

# Example usage and testing
if __name__ == "__main__":
    print("🎯 Enhanced Neural Intent Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = EnhancedNeuralIntentClassifier()
    
    if classifier.model:
        print("✅ Neural classifier initialized successfully")
        
        # Train the model
        print("\n🏋️ Training neural intent classifier...")
        training_success = classifier.train_model(epochs=2)
        
        if training_success:
            print("✅ Training completed successfully")
            
            # Test predictions
            test_queries = [
                "Best Turkish restaurant in Sultanahmet",
                "How to get to Galata Tower?",
                "What time does Hagia Sophia open?",
                "I'm looking for a romantic dinner place",
                "Sultanahmet'te iyi bir restoran var mı?"
            ]
            
            print("\n🧪 Testing predictions:")
            for query in test_queries:
                prediction = classifier.predict_intent(query)
                print(f"  '{query}' -> {prediction.intent} ({prediction.confidence:.2%})")
        else:
            print("❌ Training failed")
    else:
        print("❌ Neural classifier initialization failed")
        print("   Make sure required libraries are installed:")
        print("   pip install torch transformers scikit-learn pandas")
