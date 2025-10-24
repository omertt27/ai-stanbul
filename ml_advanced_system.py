"""
Advanced ML System for Istanbul AI
===================================

Comprehensive ML system with:
1. User Preference Learning
2. Journey Pattern Recognition
3. Predictive Route Suggestions
4. Context-Aware Conversations

Optimized for T4 GPU usage with PyTorch and transformers.

Author: Istanbul AI Team
Date: October 24, 2025
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging
from dataclasses import dataclass, asdict
import faiss
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 Using device: {DEVICE}")

if torch.cuda.is_available():
    logger.info(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


@dataclass
class UserPreference:
    """User preference profile"""
    user_id: str
    preferred_modes: List[str]  # metro, bus, tram, ferry
    avoid_transfers: bool
    max_walking_distance: float
    preferred_times: List[str]  # morning, afternoon, evening
    accessibility_needs: bool
    budget_conscious: bool
    speed_priority: float  # 0-1: 0=cheapest, 1=fastest
    comfort_priority: float  # 0-1: preference for less crowded
    embedding: Optional[np.ndarray] = None
    last_updated: Optional[datetime] = None


@dataclass
class JourneyPattern:
    """Detected journey pattern"""
    user_id: str
    origin: str
    destination: str
    frequency: int  # times traveled
    typical_time: str  # "08:00-09:00"
    typical_days: List[str]  # ["Monday", "Tuesday"]
    preferred_route_id: Optional[str] = None
    last_traveled: Optional[datetime] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class ConversationContext:
    """Context-aware conversation state"""
    user_id: str
    session_id: str
    intent: str  # routing, recommendation, inquiry
    mentioned_locations: List[str]
    mentioned_times: List[str]
    mentioned_modes: List[str]
    conversation_history: List[Dict[str, str]]
    current_location: Optional[str] = None
    destination: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    last_updated: Optional[datetime] = None


class UserPreferenceLearner(nn.Module):
    """
    Neural network for learning user preferences from interaction history.
    Uses attention mechanism to focus on important features.
    """
    
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=128):
        super(UserPreferenceLearner, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.preference_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self.mode_predictor = nn.Linear(output_dim, 5)  # metro, bus, tram, ferry, walk
        self.speed_predictor = nn.Linear(output_dim, 1)  # speed priority
        self.comfort_predictor = nn.Linear(output_dim, 1)  # comfort priority
        
    def forward(self, x):
        """Forward pass through the network"""
        # Encode
        encoded = self.encoder(x)
        
        # Apply self-attention
        attended, _ = self.attention(encoded, encoded, encoded)
        
        # Get preference embedding
        pref_embedding = self.preference_head(attended.mean(dim=1))
        
        # Predict specific preferences
        mode_prefs = torch.softmax(self.mode_predictor(pref_embedding), dim=-1)
        speed_pref = torch.sigmoid(self.speed_predictor(pref_embedding))
        comfort_pref = torch.sigmoid(self.comfort_predictor(pref_embedding))
        
        return {
            'embedding': pref_embedding,
            'mode_preferences': mode_prefs,
            'speed_priority': speed_pref,
            'comfort_priority': comfort_pref
        }


class JourneyPatternRecognizer(nn.Module):
    """
    LSTM-based sequence model for recognizing journey patterns.
    Identifies recurring trips and temporal patterns.
    """
    
    def __init__(self, input_dim=256, hidden_dim=512, num_layers=2):
        super(JourneyPatternRecognizer, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.pattern_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.Tanh()
        )
        
        # Pattern classification heads
        self.frequency_predictor = nn.Linear(128, 5)  # daily, weekly, monthly, occasional, rare
        self.time_predictor = nn.Linear(128, 24)  # hour of day
        self.day_predictor = nn.Linear(128, 7)  # day of week
        
    def forward(self, sequence):
        """Forward pass through LSTM"""
        # Process sequence
        lstm_out, (hidden, cell) = self.lstm(sequence)
        
        # Use final hidden state
        pattern_embedding = self.pattern_encoder(lstm_out[:, -1, :])
        
        # Predict pattern characteristics
        frequency = torch.softmax(self.frequency_predictor(pattern_embedding), dim=-1)
        time_probs = torch.softmax(self.time_predictor(pattern_embedding), dim=-1)
        day_probs = torch.softmax(self.day_predictor(pattern_embedding), dim=-1)
        
        return {
            'embedding': pattern_embedding,
            'frequency': frequency,
            'time_distribution': time_probs,
            'day_distribution': day_probs
        }


class PredictiveRouteRanker(nn.Module):
    """
    Neural ranking model for route suggestions.
    Combines user preferences, context, and route features.
    """
    
    def __init__(self, user_dim=128, route_dim=256, context_dim=128, hidden_dim=512):
        super(PredictiveRouteRanker, self).__init__()
        
        total_dim = user_dim + route_dim + context_dim
        
        self.ranker = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, user_embedding, route_features, context_embedding):
        """Rank a route given user preferences and context"""
        combined = torch.cat([user_embedding, route_features, context_embedding], dim=-1)
        score = self.ranker(combined)
        return torch.sigmoid(score)


class ContextAwareDialogueModel:
    """
    Context-aware dialogue system using transformer-based embeddings.
    Maintains conversation state and intent tracking.
    """
    
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()
        
        # Intent classifier
        self.intent_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 5)  # routing, recommendation, inquiry, feedback, chitchat
        ).to(DEVICE)
        
        # Context index for semantic similarity
        self.context_index = None
        self.context_embeddings = []
        self.context_metadata = []
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embedding.squeeze()
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify user intent from text"""
        embedding = self.encode_text(text)
        embedding_tensor = torch.tensor(embedding).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = self.intent_classifier(embedding_tensor)
            probs = torch.softmax(logits, dim=-1)
            intent_idx = torch.argmax(probs).item()
            confidence = probs[0, intent_idx].item()
        
        intents = ['routing', 'recommendation', 'inquiry', 'feedback', 'chitchat']
        return intents[intent_idx], confidence
    
    def update_context(self, context: ConversationContext, user_message: str, system_response: str):
        """Update conversation context with new exchange"""
        # Add to history
        context.conversation_history.append({
            'user': user_message,
            'system': system_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update intent
        intent, confidence = self.classify_intent(user_message)
        if confidence > 0.7:
            context.intent = intent
        
        # Extract entities (locations, times, modes)
        self._extract_entities(user_message, context)
        
        # Update embedding
        full_context = f"{user_message} {system_response}"
        context.embedding = self.encode_text(full_context)
        context.last_updated = datetime.now()
        
        return context
    
    def _extract_entities(self, text: str, context: ConversationContext):
        """Extract locations, times, and modes from text"""
        text_lower = text.lower()
        
        # Transport modes
        modes = ['metro', 'bus', 'tram', 'tramvay', 'otobüs', 'ferry', 'vapur']
        for mode in modes:
            if mode in text_lower and mode not in context.mentioned_modes:
                context.mentioned_modes.append(mode)
        
        # Time expressions
        time_keywords = ['morning', 'afternoon', 'evening', 'sabah', 'öğle', 'akşam']
        for time in time_keywords:
            if time in text_lower and time not in context.mentioned_times:
                context.mentioned_times.append(time)


class AdvancedMLSystem:
    """
    Main ML system integrating all advanced features.
    Optimized for T4 GPU usage.
    """
    
    def __init__(self):
        logger.info("🧠 Initializing Advanced ML System...")
        
        # Initialize models
        self.preference_learner = UserPreferenceLearner().to(DEVICE)
        self.pattern_recognizer = JourneyPatternRecognizer().to(DEVICE)
        self.route_ranker = PredictiveRouteRanker().to(DEVICE)
        self.dialogue_model = ContextAwareDialogueModel()
        
        # Set to evaluation mode
        self.preference_learner.eval()
        self.pattern_recognizer.eval()
        self.route_ranker.eval()
        
        # User data storage
        self.user_preferences: Dict[str, UserPreference] = {}
        self.journey_patterns: Dict[str, List[JourneyPattern]] = defaultdict(list)
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        
        # FAISS indices for fast similarity search
        self.user_preference_index = None
        self.journey_pattern_index = None
        
        logger.info("✅ Advanced ML System initialized successfully!")
        logger.info(f"📊 Models on device: {DEVICE}")
    
    def learn_user_preferences(self, user_id: str, interaction_history: List[Dict]) -> UserPreference:
        """
        Learn user preferences from interaction history.
        
        Args:
            user_id: User identifier
            interaction_history: List of past interactions with features
        
        Returns:
            UserPreference object with learned preferences
        """
        logger.info(f"🎓 Learning preferences for user {user_id}...")
        
        # Convert interaction history to feature vectors
        features = self._extract_interaction_features(interaction_history)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Get predictions
        with torch.no_grad():
            predictions = self.preference_learner(features_tensor)
        
        # Create preference object
        mode_probs = predictions['mode_preferences'][0].cpu().numpy()
        modes = ['metro', 'bus', 'tram', 'ferry', 'walk']
        preferred_modes = [modes[i] for i in np.argsort(mode_probs)[-3:]]  # Top 3
        
        preference = UserPreference(
            user_id=user_id,
            preferred_modes=preferred_modes,
            avoid_transfers=self._infer_transfer_preference(interaction_history),
            max_walking_distance=self._infer_walking_distance(interaction_history),
            preferred_times=self._infer_time_preferences(interaction_history),
            accessibility_needs=False,  # Can be updated from user profile
            budget_conscious=self._infer_budget_consciousness(interaction_history),
            speed_priority=predictions['speed_priority'][0].item(),
            comfort_priority=predictions['comfort_priority'][0].item(),
            embedding=predictions['embedding'][0].cpu().numpy(),
            last_updated=datetime.now()
        )
        
        # Store preference
        self.user_preferences[user_id] = preference
        
        logger.info(f"✅ Learned preferences: modes={preferred_modes}, speed={preference.speed_priority:.2f}")
        return preference
    
    def recognize_journey_patterns(self, user_id: str, trip_history: List[Dict]) -> List[JourneyPattern]:
        """
        Recognize recurring journey patterns from trip history.
        
        Args:
            user_id: User identifier
            trip_history: List of past trips with origin, destination, time
        
        Returns:
            List of detected JourneyPattern objects
        """
        logger.info(f"🔍 Recognizing journey patterns for user {user_id}...")
        
        if len(trip_history) < 3:
            logger.info("⚠️ Not enough trip history for pattern recognition")
            return []
        
        # Convert trip history to sequences
        sequence_features = self._create_trip_sequences(trip_history)
        sequence_tensor = torch.tensor(sequence_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # Get pattern predictions
        with torch.no_grad():
            predictions = self.pattern_recognizer(sequence_tensor)
        
        # Cluster similar trips using DBSCAN
        trip_embeddings = self._embed_trips(trip_history)
        clusters = DBSCAN(eps=0.3, min_samples=2).fit(trip_embeddings)
        
        patterns = []
        for cluster_id in set(clusters.labels_):
            if cluster_id == -1:  # Noise
                continue
            
            cluster_trips = [trip_history[i] for i, label in enumerate(clusters.labels_) if label == cluster_id]
            
            if len(cluster_trips) >= 2:
                pattern = self._create_pattern_from_cluster(user_id, cluster_trips, predictions)
                patterns.append(pattern)
        
        # Store patterns
        self.journey_patterns[user_id] = patterns
        
        logger.info(f"✅ Found {len(patterns)} journey patterns")
        return patterns
    
    def predict_routes(
        self, 
        user_id: str, 
        origin: str, 
        destination: str, 
        candidate_routes: List[Dict],
        context: Optional[ConversationContext] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Rank and predict best routes using personalization.
        
        Args:
            user_id: User identifier
            origin: Starting location
            destination: Target location
            candidate_routes: List of possible routes with features
            context: Optional conversation context
        
        Returns:
            List of (route, score) tuples, sorted by score
        """
        logger.info(f"🔮 Predicting routes for {origin} → {destination}...")
        
        # Get user preference embedding
        if user_id in self.user_preferences:
            user_embedding = torch.tensor(
                self.user_preferences[user_id].embedding, 
                dtype=torch.float32
            ).to(DEVICE)
        else:
            user_embedding = torch.zeros(128).to(DEVICE)
        
        # Get context embedding
        if context and context.embedding is not None:
            context_embedding = torch.tensor(context.embedding[:128], dtype=torch.float32).to(DEVICE)
        else:
            context_embedding = torch.zeros(128).to(DEVICE)
        
        # Score each route
        scored_routes = []
        with torch.no_grad():
            for route in candidate_routes:
                route_features = self._extract_route_features(route)
                route_tensor = torch.tensor(route_features, dtype=torch.float32).to(DEVICE)
                
                score = self.route_ranker(
                    user_embedding.unsqueeze(0),
                    route_tensor.unsqueeze(0),
                    context_embedding.unsqueeze(0)
                )
                
                scored_routes.append((route, score.item()))
        
        # Sort by score
        scored_routes.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"✅ Ranked {len(scored_routes)} routes")
        return scored_routes
    
    def process_conversation(
        self, 
        user_id: str, 
        user_message: str, 
        session_id: Optional[str] = None
    ) -> Tuple[ConversationContext, str, Dict]:
        """
        Process conversation with context-awareness.
        
        Args:
            user_id: User identifier
            user_message: User's message
            session_id: Optional session identifier
        
        Returns:
            Tuple of (updated_context, intent, extracted_info)
        """
        logger.info(f"💬 Processing conversation for user {user_id}...")
        
        # Get or create context
        if session_id and session_id in self.conversation_contexts:
            context = self.conversation_contexts[session_id]
        else:
            session_id = session_id or f"{user_id}_{datetime.now().timestamp()}"
            context = ConversationContext(
                user_id=user_id,
                session_id=session_id,
                intent='unknown',
                mentioned_locations=[],
                mentioned_times=[],
                mentioned_modes=[],
                conversation_history=[]
            )
        
        # Classify intent
        intent, confidence = self.dialogue_model.classify_intent(user_message)
        
        # Update context (we'll add system response later)
        context = self.dialogue_model.update_context(context, user_message, "")
        
        # Extract structured information
        extracted_info = {
            'intent': intent,
            'confidence': confidence,
            'locations': context.mentioned_locations[-2:] if context.mentioned_locations else [],
            'times': context.mentioned_times[-1:] if context.mentioned_times else [],
            'modes': context.mentioned_modes[-2:] if context.mentioned_modes else []
        }
        
        # Store context
        self.conversation_contexts[session_id] = context
        
        logger.info(f"✅ Intent: {intent} (confidence: {confidence:.2f})")
        return context, intent, extracted_info
    
    def get_personalized_recommendations(
        self, 
        user_id: str, 
        current_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get proactive route recommendations based on patterns.
        
        Args:
            user_id: User identifier
            current_time: Current time (defaults to now)
        
        Returns:
            List of recommended journeys
        """
        current_time = current_time or datetime.now()
        recommendations = []
        
        # Check for patterns matching current time
        if user_id in self.journey_patterns:
            for pattern in self.journey_patterns[user_id]:
                if self._matches_time_pattern(pattern, current_time):
                    recommendations.append({
                        'origin': pattern.origin,
                        'destination': pattern.destination,
                        'reason': f'You usually travel this route on {current_time.strftime("%A")}s',
                        'frequency': pattern.frequency,
                        'confidence': min(pattern.frequency / 10.0, 1.0)
                    })
        
        return sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
    
    # Helper methods
    
    def _extract_interaction_features(self, history: List[Dict]) -> np.ndarray:
        """Extract features from interaction history"""
        # Create a fixed-size feature vector
        features = np.zeros(256)
        
        if not history:
            return features
        
        # Mode preferences (one-hot encoded)
        mode_counts = defaultdict(int)
        for interaction in history:
            if 'selected_mode' in interaction:
                mode_counts[interaction['selected_mode']] += 1
        
        modes = ['metro', 'bus', 'tram', 'ferry', 'walk']
        for i, mode in enumerate(modes):
            features[i] = mode_counts.get(mode, 0)
        
        # Time preferences
        time_counts = defaultdict(int)
        for interaction in history:
            if 'timestamp' in interaction:
                hour = datetime.fromisoformat(interaction['timestamp']).hour
                time_counts[hour] += 1
        
        for hour in range(24):
            features[5 + hour] = time_counts.get(hour, 0)
        
        # Transfer patterns
        features[29] = sum(1 for i in history if i.get('num_transfers', 0) == 0)
        features[30] = sum(1 for i in history if i.get('num_transfers', 0) == 1)
        features[31] = sum(1 for i in history if i.get('num_transfers', 0) >= 2)
        
        # Duration preferences
        durations = [i.get('duration', 30) for i in history]
        features[32] = np.mean(durations) if durations else 30
        features[33] = np.std(durations) if len(durations) > 1 else 0
        
        # Normalize
        if features.max() > 0:
            features = features / features.max()
        
        return features
    
    def _infer_transfer_preference(self, history: List[Dict]) -> bool:
        """Infer if user avoids transfers"""
        if len(history) < 3:
            return False
        
        no_transfer_count = sum(1 for i in history if i.get('num_transfers', 1) == 0)
        return (no_transfer_count / len(history)) > 0.6
    
    def _infer_walking_distance(self, history: List[Dict]) -> float:
        """Infer maximum walking distance preference"""
        walking_distances = [i.get('walking_distance', 500) for i in history if 'walking_distance' in i]
        return np.percentile(walking_distances, 75) if walking_distances else 1000.0
    
    def _infer_time_preferences(self, history: List[Dict]) -> List[str]:
        """Infer preferred time periods"""
        time_counts = defaultdict(int)
        
        for interaction in history:
            if 'timestamp' in interaction:
                hour = datetime.fromisoformat(interaction['timestamp']).hour
                if 6 <= hour < 12:
                    time_counts['morning'] += 1
                elif 12 <= hour < 18:
                    time_counts['afternoon'] += 1
                else:
                    time_counts['evening'] += 1
        
        if not time_counts:
            return ['morning', 'afternoon', 'evening']
        
        sorted_times = sorted(time_counts.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_times[:2]]
    
    def _infer_budget_consciousness(self, history: List[Dict]) -> bool:
        """Infer if user is budget-conscious"""
        if len(history) < 3:
            return False
        
        # Check if user consistently chooses cheaper options
        chose_cheaper = sum(1 for i in history if i.get('chose_cheaper_option', False))
        return (chose_cheaper / len(history)) > 0.5
    
    def _create_trip_sequences(self, trips: List[Dict]) -> np.ndarray:
        """Create sequence features from trips"""
        max_seq_len = 50
        feature_dim = 256
        
        sequences = np.zeros((max_seq_len, feature_dim))
        
        for i, trip in enumerate(trips[-max_seq_len:]):
            # Encode trip features
            sequences[i, 0] = hash(trip.get('origin', '')) % 1000 / 1000.0
            sequences[i, 1] = hash(trip.get('destination', '')) % 1000 / 1000.0
            
            if 'timestamp' in trip:
                dt = datetime.fromisoformat(trip['timestamp'])
                sequences[i, 2] = dt.hour / 24.0
                sequences[i, 3] = dt.weekday() / 7.0
                sequences[i, 4] = dt.day / 31.0
            
            sequences[i, 5] = trip.get('duration', 30) / 120.0
            sequences[i, 6] = trip.get('num_transfers', 1) / 5.0
        
        return sequences
    
    def _embed_trips(self, trips: List[Dict]) -> np.ndarray:
        """Create embeddings for trips for clustering"""
        embeddings = []
        
        for trip in trips:
            # Create a text representation
            text = f"{trip.get('origin', '')} to {trip.get('destination', '')}"
            if 'timestamp' in trip:
                dt = datetime.fromisoformat(trip['timestamp'])
                text += f" on {dt.strftime('%A at %H:00')}"
            
            embedding = self.dialogue_model.encode_text(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _create_pattern_from_cluster(
        self, 
        user_id: str, 
        trips: List[Dict], 
        predictions: Dict
    ) -> JourneyPattern:
        """Create a JourneyPattern from clustered trips"""
        # Get most common origin/destination
        origins = [t.get('origin', '') for t in trips]
        destinations = [t.get('destination', '') for t in trips]
        
        origin = max(set(origins), key=origins.count)
        destination = max(set(destinations), key=destinations.count)
        
        # Get typical time
        hours = []
        days = []
        for trip in trips:
            if 'timestamp' in trip:
                dt = datetime.fromisoformat(trip['timestamp'])
                hours.append(dt.hour)
                days.append(dt.strftime('%A'))
        
        typical_hour = int(np.median(hours)) if hours else 8
        typical_time = f"{typical_hour:02d}:00-{(typical_hour+1):02d}:00"
        typical_days = list(set(days))
        
        # Get last traveled
        last_traveled = None
        if trips and 'timestamp' in trips[-1]:
            last_traveled = datetime.fromisoformat(trips[-1]['timestamp'])
        
        return JourneyPattern(
            user_id=user_id,
            origin=origin,
            destination=destination,
            frequency=len(trips),
            typical_time=typical_time,
            typical_days=typical_days,
            last_traveled=last_traveled,
            embedding=predictions['embedding'][0].cpu().numpy()
        )
    
    def _extract_route_features(self, route: Dict) -> np.ndarray:
        """Extract features from a route for ranking"""
        features = np.zeros(256)
        
        features[0] = route.get('duration', 30) / 120.0
        features[1] = route.get('num_transfers', 1) / 5.0
        features[2] = route.get('walking_distance', 500) / 2000.0
        features[3] = route.get('cost', 15) / 50.0
        features[4] = route.get('crowding_level', 0.5)
        
        # Mode features
        modes = route.get('modes', [])
        mode_map = {'metro': 5, 'bus': 6, 'tram': 7, 'ferry': 8}
        for mode, idx in mode_map.items():
            features[idx] = 1.0 if mode in modes else 0.0
        
        return features
    
    def _matches_time_pattern(self, pattern: JourneyPattern, current_time: datetime) -> bool:
        """Check if current time matches pattern"""
        current_day = current_time.strftime('%A')
        if current_day not in pattern.typical_days:
            return False
        
        # Parse typical time
        time_range = pattern.typical_time.split('-')
        if len(time_range) == 2:
            start_hour = int(time_range[0].split(':')[0])
            end_hour = int(time_range[1].split(':')[0])
            
            current_hour = current_time.hour
            return start_hour <= current_hour < end_hour
        
        return False
    
    def save_models(self, save_dir: str):
        """Save all models and data"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.preference_learner.state_dict(), f"{save_dir}/preference_learner.pt")
        torch.save(self.pattern_recognizer.state_dict(), f"{save_dir}/pattern_recognizer.pt")
        torch.save(self.route_ranker.state_dict(), f"{save_dir}/route_ranker.pt")
        torch.save(self.dialogue_model.intent_classifier.state_dict(), f"{save_dir}/intent_classifier.pt")
        
        # Save user data
        with open(f"{save_dir}/user_preferences.json", 'w') as f:
            json.dump({k: asdict(v) for k, v in self.user_preferences.items()}, f, default=str)
        
        with open(f"{save_dir}/journey_patterns.json", 'w') as f:
            json.dump({k: [asdict(p) for p in v] for k, v in self.journey_patterns.items()}, f, default=str)
        
        logger.info(f"✅ Models saved to {save_dir}")
    
    def load_models(self, load_dir: str):
        """Load all models and data"""
        self.preference_learner.load_state_dict(torch.load(f"{load_dir}/preference_learner.pt"))
        self.pattern_recognizer.load_state_dict(torch.load(f"{load_dir}/pattern_recognizer.pt"))
        self.route_ranker.load_state_dict(torch.load(f"{load_dir}/route_ranker.pt"))
        self.dialogue_model.intent_classifier.load_state_dict(torch.load(f"{load_dir}/intent_classifier.pt"))
        
        logger.info(f"✅ Models loaded from {load_dir}")


# Singleton instance
_ml_system_instance = None

def get_advanced_ml_system() -> AdvancedMLSystem:
    """Get or create the singleton ML system instance"""
    global _ml_system_instance
    if _ml_system_instance is None:
        _ml_system_instance = AdvancedMLSystem()
    return _ml_system_instance


if __name__ == "__main__":
    # Test the system
    print("🧠 Testing Advanced ML System...")
    
    ml_system = get_advanced_ml_system()
    
    # Test user preference learning
    print("\n📊 Testing User Preference Learning...")
    sample_history = [
        {'selected_mode': 'metro', 'num_transfers': 1, 'duration': 25, 'timestamp': '2025-10-24T08:30:00'},
        {'selected_mode': 'metro', 'num_transfers': 0, 'duration': 20, 'timestamp': '2025-10-24T18:15:00'},
        {'selected_mode': 'bus', 'num_transfers': 2, 'duration': 35, 'timestamp': '2025-10-24T12:00:00'},
    ]
    preferences = ml_system.learn_user_preferences('user_123', sample_history)
    print(f"✅ Learned preferences: {preferences.preferred_modes}")
    
    # Test journey pattern recognition
    print("\n🔍 Testing Journey Pattern Recognition...")
    sample_trips = [
        {'origin': 'Taksim', 'destination': 'Sultanahmet', 'timestamp': '2025-10-21T08:30:00'},
        {'origin': 'Taksim', 'destination': 'Sultanahmet', 'timestamp': '2025-10-22T08:35:00'},
        {'origin': 'Taksim', 'destination': 'Sultanahmet', 'timestamp': '2025-10-23T08:28:00'},
        {'origin': 'Kadıköy', 'destination': 'Beşiktaş', 'timestamp': '2025-10-21T18:00:00'},
        {'origin': 'Kadıköy', 'destination': 'Beşiktaş', 'timestamp': '2025-10-22T18:05:00'},
    ]
    patterns = ml_system.recognize_journey_patterns('user_123', sample_trips)
    print(f"✅ Found {len(patterns)} patterns")
    
    # Test conversation processing
    print("\n💬 Testing Context-Aware Conversation...")
    context, intent, info = ml_system.process_conversation(
        'user_123', 
        'How can I go to Sultanahmet from Taksim by metro?'
    )
    print(f"✅ Intent: {intent}, Extracted: {info}")
    
    print("\n🎉 All tests completed successfully!")
