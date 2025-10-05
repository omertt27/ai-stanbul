# recommendation_enhancement_system.py - Advanced Recommendation Enhancement Engine

import numpy as np
import pandas as pd
import sqlite3
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
import warnings

warnings.filterwarnings('ignore')

class UserEmbeddingSystem:
    """Advanced user embedding system for deep personalization"""
    
    def __init__(self, embedding_dim: int = 64, db_path: str = 'ai_istanbul_users.db'):
        self.embedding_dim = embedding_dim
        self.db_path = db_path
        self.user_embeddings = {}
        self.attraction_embeddings = {}
        self.interaction_model = None
        self.scaler = StandardScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize embedding storage database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # User embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_embeddings (
                    user_id TEXT PRIMARY KEY,
                    embedding_vector TEXT,
                    last_updated TIMESTAMP,
                    embedding_version INTEGER DEFAULT 1
                )
            ''')
            
            # Attraction embeddings table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attraction_embeddings (
                    attraction_id TEXT PRIMARY KEY,
                    attraction_name TEXT,
                    category TEXT,
                    embedding_vector TEXT,
                    features_vector TEXT,
                    last_updated TIMESTAMP
                )
            ''')
            
            # User-attraction interactions for training
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interaction_matrix (
                    user_id TEXT,
                    attraction_id TEXT,
                    interaction_score REAL,
                    interaction_type TEXT,
                    timestamp TIMESTAMP,
                    PRIMARY KEY (user_id, attraction_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Embedding database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            
    def build_interaction_model(self):
        """Build neural collaborative filtering model"""
        try:
            # Get unique users and attractions
            conn = sqlite3.connect(self.db_path)
            
            users_df = pd.read_sql_query("SELECT DISTINCT user_id FROM user_profiles", conn)
            attractions_df = pd.read_sql_query("""
                SELECT DISTINCT attraction_id, attraction_name, category 
                FROM attraction_embeddings
            """, conn)
            
            num_users = len(users_df)
            num_attractions = len(attractions_df)
            
            # User and attraction inputs
            user_input = Input(shape=[], name='user_id')
            attraction_input = Input(shape=[], name='attraction_id')
            
            # Embedding layers
            user_embedding = Embedding(num_users, self.embedding_dim, name='user_embedding')(user_input)
            attraction_embedding = Embedding(num_attractions, self.embedding_dim, name='attraction_embedding')(attraction_input)
            
            # Flatten embeddings
            user_vec = tf.keras.layers.Flatten()(user_embedding)
            attraction_vec = tf.keras.layers.Flatten()(attraction_embedding)
            
            # Neural CF layers
            concat = Concatenate()([user_vec, attraction_vec])
            dense1 = Dense(128, activation='relu')(concat)
            dropout1 = Dropout(0.2)(dense1)
            dense2 = Dense(64, activation='relu')(dropout1)
            dropout2 = Dropout(0.2)(dense2)
            output = Dense(1, activation='sigmoid', name='interaction_score')(dropout2)
            
            # Build model
            self.interaction_model = Model(inputs=[user_input, attraction_input], outputs=output)
            self.interaction_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['mae', 'mse']
            )
            
            conn.close()
            self.logger.info(f"Neural CF model built: {num_users} users, {num_attractions} attractions")
            return True
            
        except Exception as e:
            self.logger.error(f"Model building error: {str(e)}")
            return False
            
    def train_embeddings(self, epochs: int = 50, batch_size: int = 32):
        """Train user and attraction embeddings"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load interaction data
            interactions_df = pd.read_sql_query("""
                SELECT user_id, attraction_id, interaction_score
                FROM interaction_matrix
                WHERE interaction_score > 0
            """, conn)
            
            if interactions_df.empty:
                self.logger.warning("No interaction data found for training")
                return False
                
            # Create user and attraction mappings
            users = interactions_df['user_id'].unique()
            attractions = interactions_df['attraction_id'].unique()
            
            user_to_idx = {user: idx for idx, user in enumerate(users)}
            attraction_to_idx = {attr: idx for idx, attr in enumerate(attractions)}
            
            # Prepare training data
            user_ids = interactions_df['user_id'].map(user_to_idx).values
            attraction_ids = interactions_df['attraction_id'].map(attraction_to_idx).values
            scores = interactions_df['interaction_score'].values
            
            # Normalize scores to 0-1 range
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            # Train model
            if self.interaction_model is None:
                self.build_interaction_model()
                
            history = self.interaction_model.fit(
                [user_ids, attraction_ids], scores,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # Extract learned embeddings
            user_embedding_layer = self.interaction_model.get_layer('user_embedding')
            attraction_embedding_layer = self.interaction_model.get_layer('attraction_embedding')
            
            user_embeddings = user_embedding_layer.get_weights()[0]
            attraction_embeddings = attraction_embedding_layer.get_weights()[0]
            
            # Store embeddings
            for idx, user_id in enumerate(users):
                self.user_embeddings[user_id] = user_embeddings[idx]
                
            for idx, attraction_id in enumerate(attractions):
                self.attraction_embeddings[attraction_id] = attraction_embeddings[idx]
                
            # Save to database
            self._save_embeddings_to_db()
            
            conn.close()
            self.logger.info(f"Embeddings trained successfully. Final loss: {history.history['loss'][-1]:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            return False
            
    def _save_embeddings_to_db(self):
        """Save learned embeddings to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save user embeddings
            for user_id, embedding in self.user_embeddings.items():
                embedding_json = json.dumps(embedding.tolist())
                cursor.execute('''
                    INSERT OR REPLACE INTO user_embeddings 
                    (user_id, embedding_vector, last_updated, embedding_version)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, embedding_json, datetime.now().isoformat(), 1))
                
            # Save attraction embeddings
            for attraction_id, embedding in self.attraction_embeddings.items():
                embedding_json = json.dumps(embedding.tolist())
                cursor.execute('''
                    INSERT OR REPLACE INTO attraction_embeddings 
                    (attraction_id, embedding_vector, last_updated)
                    VALUES (?, ?, ?)
                ''', (attraction_id, embedding_json, datetime.now().isoformat()))
                
            conn.commit()
            conn.close()
            self.logger.info("Embeddings saved to database")
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            
    def get_similar_users(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar users based on embeddings"""
        try:
            if user_id not in self.user_embeddings:
                return []
                
            user_embedding = self.user_embeddings[user_id]
            similarities = []
            
            for other_user_id, other_embedding in self.user_embeddings.items():
                if other_user_id != user_id:
                    similarity = cosine_similarity(
                        user_embedding.reshape(1, -1),
                        other_embedding.reshape(1, -1)
                    )[0][0]
                    similarities.append((other_user_id, similarity))
                    
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar users: {str(e)}")
            return []
            
    def get_attraction_recommendations(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get attraction recommendations using embeddings"""
        try:
            if user_id not in self.user_embeddings:
                return []
                
            user_embedding = self.user_embeddings[user_id]
            recommendations = []
            
            for attraction_id, attraction_embedding in self.attraction_embeddings.items():
                # Calculate similarity score
                similarity = cosine_similarity(
                    user_embedding.reshape(1, -1),
                    attraction_embedding.reshape(1, -1)
                )[0][0]
                
                recommendations.append((attraction_id, similarity))
                
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {str(e)}")
            return []

class RecommendationEnhancementEngine:
    """Main recommendation enhancement system"""
    
    def __init__(self, db_path: str = 'ai_istanbul_users.db'):
        self.db_path = db_path
        self.embedding_system = UserEmbeddingSystem(db_path=db_path)
        self.logger = logging.getLogger(__name__)
        
        # Istanbul attractions database
        self.attractions_db = {
            "hagia_sophia": {
                "name": "Hagia Sophia",
                "category": "historical",
                "features": ["byzantine", "architecture", "museum", "religious"],
                "rating": 4.8,
                "visit_duration": 2
            },
            "blue_mosque": {
                "name": "Blue Mosque",
                "category": "religious",
                "features": ["ottoman", "architecture", "mosque", "prayer"],
                "rating": 4.7,
                "visit_duration": 1.5
            },
            "grand_bazaar": {
                "name": "Grand Bazaar",
                "category": "shopping",
                "features": ["shopping", "traditional", "crafts", "souvenirs"],
                "rating": 4.5,
                "visit_duration": 3
            },
            "topkapi_palace": {
                "name": "Topkapi Palace",
                "category": "historical",
                "features": ["ottoman", "palace", "museum", "gardens"],
                "rating": 4.6,
                "visit_duration": 3
            },
            "galata_tower": {
                "name": "Galata Tower",
                "category": "landmark",
                "features": ["tower", "view", "medieval", "panoramic"],
                "rating": 4.4,
                "visit_duration": 1
            }
        }
        
    def initialize_attraction_embeddings(self):
        """Initialize attraction embeddings with features"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for attraction_id, attraction_data in self.attractions_db.items():
                # Create feature vector
                features = attraction_data["features"]
                features_text = " ".join(features + [attraction_data["category"]])
                
                # Store in database
                cursor.execute('''
                    INSERT OR REPLACE INTO attraction_embeddings 
                    (attraction_id, attraction_name, category, features_vector, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    attraction_id,
                    attraction_data["name"],
                    attraction_data["category"],
                    features_text,
                    datetime.now().isoformat()
                ))
                
            conn.commit()
            conn.close()
            self.logger.info("Attraction embeddings initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing attraction embeddings: {str(e)}")
            
    def generate_enhanced_recommendations(self, user_id: str, context: Dict = None) -> Dict:
        """Generate enhanced recommendations using multiple signals"""
        try:
            recommendations = {
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "embedding_based": [],
                "collaborative_filtering": [],
                "content_based": [],
                "hybrid_score": [],
                "context_aware": []
            }
            
            # 1. Embedding-based recommendations
            embedding_recs = self.embedding_system.get_attraction_recommendations(user_id, top_k=10)
            recommendations["embedding_based"] = [
                {"attraction_id": attr_id, "score": float(score)}
                for attr_id, score in embedding_recs
            ]
            
            # 2. Collaborative filtering (similar users)
            similar_users = self.embedding_system.get_similar_users(user_id, top_k=5)
            collab_recs = self._get_collaborative_recommendations(user_id, similar_users)
            recommendations["collaborative_filtering"] = collab_recs
            
            # 3. Content-based recommendations
            content_recs = self._get_content_based_recommendations(user_id)
            recommendations["content_based"] = content_recs
            
            # 4. Hybrid scoring
            hybrid_recs = self._calculate_hybrid_scores(
                embedding_recs, collab_recs, content_recs
            )
            recommendations["hybrid_score"] = hybrid_recs
            
            # 5. Context-aware adjustments
            if context:
                context_recs = self._apply_context_awareness(hybrid_recs, context)
                recommendations["context_aware"] = context_recs
                
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced recommendations: {str(e)}")
            return {"error": str(e)}
            
    def _get_collaborative_recommendations(self, user_id: str, similar_users: List[Tuple[str, float]]) -> List[Dict]:
        """Get recommendations based on similar users"""
        try:
            conn = sqlite3.connect(self.db_path)
            recommendations = {}
            
            for similar_user_id, similarity_score in similar_users:
                # Get attractions liked by similar user
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT attraction_id, interaction_score 
                    FROM interaction_matrix 
                    WHERE user_id = ? AND interaction_score > 0.7
                ''', (similar_user_id,))
                
                liked_attractions = cursor.fetchall()
                
                for attraction_id, interaction_score in liked_attractions:
                    if attraction_id not in recommendations:
                        recommendations[attraction_id] = 0
                    recommendations[attraction_id] += similarity_score * interaction_score
                    
            conn.close()
            
            # Sort and return top recommendations
            sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
            return [
                {"attraction_id": attr_id, "score": float(score)}
                for attr_id, score in sorted_recs[:10]
            ]
            
        except Exception as e:
            self.logger.error(f"Error in collaborative recommendations: {str(e)}")
            return []
            
    def _get_content_based_recommendations(self, user_id: str) -> List[Dict]:
        """Get content-based recommendations"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user's preferred categories/features
            cursor.execute('''
                SELECT p.preferences 
                FROM user_profiles p 
                WHERE p.user_id = ?
            ''', (user_id,))
            
            user_prefs = cursor.fetchone()
            if not user_prefs:
                return []
                
            preferences = json.loads(user_prefs[0])
            preferred_categories = preferences.get("categories", [])
            
            recommendations = []
            
            # Score attractions based on category match
            for attraction_id, attraction_data in self.attractions_db.items():
                score = 0
                if attraction_data["category"] in preferred_categories:
                    score += 0.8
                    
                # Feature matching
                user_interests = preferences.get("interests", [])
                feature_matches = len(set(attraction_data["features"]) & set(user_interests))
                score += feature_matches * 0.2
                
                if score > 0:
                    recommendations.append({
                        "attraction_id": attraction_id,
                        "score": float(score)
                    })
                    
            conn.close()
            return sorted(recommendations, key=lambda x: x["score"], reverse=True)[:10]
            
        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {str(e)}")
            return []
            
    def _calculate_hybrid_scores(self, embedding_recs: List, collab_recs: List, content_recs: List) -> List[Dict]:
        """Calculate hybrid recommendation scores"""
        try:
            all_attractions = set()
            scores = {}
            
            # Collect all recommended attractions
            for rec_list in [embedding_recs, collab_recs, content_recs]:
                for rec in rec_list:
                    attraction_id = rec["attraction_id"]
                    all_attractions.add(attraction_id)
                    if attraction_id not in scores:
                        scores[attraction_id] = {"embedding": 0, "collab": 0, "content": 0}
                        
            # Assign scores from each method
            for rec in embedding_recs:
                scores[rec["attraction_id"]]["embedding"] = rec["score"]
                
            for rec in collab_recs:
                scores[rec["attraction_id"]]["collab"] = rec["score"]
                
            for rec in content_recs:
                scores[rec["attraction_id"]]["content"] = rec["score"]
                
            # Calculate hybrid scores (weighted combination)
            hybrid_recommendations = []
            weights = {"embedding": 0.4, "collab": 0.35, "content": 0.25}
            
            for attraction_id, method_scores in scores.items():
                hybrid_score = (
                    weights["embedding"] * method_scores["embedding"] +
                    weights["collab"] * method_scores["collab"] +
                    weights["content"] * method_scores["content"]
                )
                
                hybrid_recommendations.append({
                    "attraction_id": attraction_id,
                    "hybrid_score": float(hybrid_score),
                    "component_scores": method_scores
                })
                
            return sorted(hybrid_recommendations, key=lambda x: x["hybrid_score"], reverse=True)[:10]
            
        except Exception as e:
            self.logger.error(f"Error calculating hybrid scores: {str(e)}")
            return []
            
    def _apply_context_awareness(self, recommendations: List[Dict], context: Dict) -> List[Dict]:
        """Apply context-aware adjustments to recommendations"""
        try:
            context_adjusted = []
            
            current_time = context.get("time_of_day", "daytime")
            weather = context.get("weather", "clear")
            budget = context.get("budget", "medium")
            group_size = context.get("group_size", 1)
            
            for rec in recommendations:
                attraction_id = rec["attraction_id"]
                base_score = rec["hybrid_score"]
                
                # Time-based adjustments
                if current_time == "evening" and attraction_id in ["galata_tower"]:
                    base_score *= 1.2  # Better for evening views
                elif current_time == "morning" and attraction_id in ["blue_mosque"]:
                    base_score *= 1.1  # Less crowded in morning
                    
                # Weather adjustments
                if weather == "rainy":
                    if attraction_id in ["hagia_sophia", "topkapi_palace"]:
                        base_score *= 1.3  # Indoor attractions
                    elif attraction_id in ["galata_tower"]:
                        base_score *= 0.7  # Outdoor view less appealing
                        
                # Group size adjustments
                if group_size > 4 and attraction_id == "grand_bazaar":
                    base_score *= 1.1  # Good for larger groups
                    
                context_adjusted.append({
                    "attraction_id": attraction_id,
                    "context_adjusted_score": float(base_score),
                    "original_score": rec["hybrid_score"],
                    "adjustments_applied": {
                        "time_of_day": current_time,
                        "weather": weather,
                        "group_size": group_size
                    }
                })
                
            return sorted(context_adjusted, key=lambda x: x["context_adjusted_score"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error applying context awareness: {str(e)}")
            return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Initialize enhancement engine
    enhancement_engine = RecommendationEnhancementEngine()
    
    # Initialize attraction embeddings
    enhancement_engine.initialize_attraction_embeddings()
    
    # Build and train embedding model (would need interaction data in real scenario)
    enhancement_engine.embedding_system.build_interaction_model()
    
    # Generate enhanced recommendations
    context = {
        "time_of_day": "morning",
        "weather": "clear",
        "budget": "medium",
        "group_size": 2
    }
    
    recommendations = enhancement_engine.generate_enhanced_recommendations(
        user_id="user_001",
        context=context
    )
    
    print("Enhanced Recommendations Generated:")
    print(json.dumps(recommendations, indent=2))
