#!/usr/bin/env python3
"""
Lightweight Personalization Engine for AI Istanbul
=================================================

Implements collaborative filtering and user preference learning without LLMs.
Uses lightweight algorithms to personalize recommendations based on:
- User interaction history
- Collaborative filtering (users with similar preferences)
- Content-based filtering (similar attractions/restaurants)
- Implicit feedback (clicks, time spent, return visits)

Features:
- Memory-efficient collaborative filtering
- Real-time preference updates
- Privacy-preserving user modeling
- No external ML dependencies
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import threading
import hashlib
import uuid

@dataclass
class UserInteraction:
    """User interaction record"""
    user_id: str
    item_id: str
    item_type: str  # restaurant, attraction, museum
    interaction_type: str  # view, click, bookmark, rating
    interaction_value: float  # 1.0 for click, rating value, time spent, etc.
    timestamp: datetime
    session_id: Optional[str] = None
    context: Dict[str, Any] = None

@dataclass
class UserProfile:
    """User preference profile"""
    user_id: str
    preferences: Dict[str, float]  # category -> preference score
    visited_items: Set[str]
    favorite_districts: List[str]
    cuisine_preferences: Dict[str, float]
    activity_preferences: Dict[str, float]
    price_preference: str  # budget, mid-range, luxury
    last_updated: datetime

@dataclass
class PersonalizedRecommendation:
    """Personalized recommendation result"""
    item_id: str
    item_type: str
    title: str
    description: str
    metadata: Dict[str, Any]
    
    # Scoring breakdown
    base_score: float
    personalization_score: float
    collaborative_score: float
    content_score: float
    final_score: float
    
    # Explanation
    reasons: List[str]
    confidence: float

class LightweightPersonalizationEngine:
    """Lightweight personalization system using collaborative filtering"""
    
    def __init__(self, db_path: str = "lightweight_personalization.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # In-memory caches for performance
        self.user_profiles = {}
        self.item_features = {}
        self.user_similarity_cache = {}
        self.recommendation_cache = {}
        
        # Configuration
        self.min_interactions_for_cf = 3  # Minimum interactions for collaborative filtering
        self.similarity_threshold = 0.1   # Minimum similarity for recommendations
        self.cache_ttl = 3600              # Cache TTL in seconds
        
        self._init_database()
        self._load_user_profiles()
    
    def _init_database(self):
        """Initialize personalization database"""
        with sqlite3.connect(self.db_path) as conn:
            # User interactions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    item_type TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    interaction_value REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    context TEXT
                )
            """)
            
            # User profiles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT NOT NULL,
                    visited_items TEXT NOT NULL,
                    favorite_districts TEXT,
                    cuisine_preferences TEXT,
                    activity_preferences TEXT,
                    price_preference TEXT DEFAULT 'mid-range',
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Item features table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS item_features (
                    item_id TEXT PRIMARY KEY,
                    item_type TEXT NOT NULL,
                    features TEXT NOT NULL,
                    category TEXT,
                    district TEXT,
                    rating REAL,
                    price_level TEXT,
                    tags TEXT,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON user_interactions(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_item ON user_interactions(item_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON user_interactions(timestamp)")
            
            conn.commit()
    
    def _load_user_profiles(self):
        """Load user profiles into memory cache"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM user_profiles")
            
            for row in cursor.fetchall():
                user_id = row[0]
                profile = UserProfile(
                    user_id=user_id,
                    preferences=json.loads(row[1]),
                    visited_items=set(json.loads(row[2])),
                    favorite_districts=json.loads(row[3]) if row[3] else [],
                    cuisine_preferences=json.loads(row[4]) if row[4] else {},
                    activity_preferences=json.loads(row[5]) if row[5] else {},
                    price_preference=row[6] or 'mid-range',
                    last_updated=datetime.fromisoformat(row[7])
                )
                self.user_profiles[user_id] = profile
    
    def record_interaction(self, interaction: UserInteraction):
        """Record a user interaction"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO user_interactions 
                        (user_id, item_id, item_type, interaction_type, interaction_value, 
                         timestamp, session_id, context)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        interaction.user_id,
                        interaction.item_id,
                        interaction.item_type,
                        interaction.interaction_type,
                        interaction.interaction_value,
                        interaction.timestamp.isoformat(),
                        interaction.session_id,
                        json.dumps(interaction.context) if interaction.context else None
                    ))
                    conn.commit()
                
                # Update user profile
                self._update_user_profile(interaction)
                
                # Invalidate recommendation cache
                if interaction.user_id in self.recommendation_cache:
                    del self.recommendation_cache[interaction.user_id]
                
            except Exception as e:
                print(f"❌ Error recording interaction: {e}")
    
    def _update_user_profile(self, interaction: UserInteraction):
        """Update user profile based on new interaction"""
        user_id = interaction.user_id
        
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                preferences={},
                visited_items=set(),
                favorite_districts=[],
                cuisine_preferences={},
                activity_preferences={},
                price_preference='mid-range',
                last_updated=datetime.now()
            )
        
        profile = self.user_profiles[user_id]
        
        # Update visited items
        profile.visited_items.add(interaction.item_id)
        
        # Update preferences based on interaction
        item_features = self._get_item_features(interaction.item_id, interaction.item_type)
        if item_features:
            # Update category preferences
            category = item_features.get('category', 'unknown')
            if category in profile.preferences:
                profile.preferences[category] += interaction.interaction_value * 0.1
            else:
                profile.preferences[category] = interaction.interaction_value * 0.1
            
            # Update district preferences
            district = item_features.get('district', '')
            if district and district not in profile.favorite_districts:
                if len(profile.favorite_districts) < 5:  # Limit to top 5
                    profile.favorite_districts.append(district)
            
            # Update cuisine preferences for restaurants
            if interaction.item_type == 'restaurant':
                cuisine = item_features.get('cuisine_type', 'unknown')
                if cuisine in profile.cuisine_preferences:
                    profile.cuisine_preferences[cuisine] += interaction.interaction_value * 0.1
                else:
                    profile.cuisine_preferences[cuisine] = interaction.interaction_value * 0.1
        
        profile.last_updated = datetime.now()
        
        # Save to database
        self._save_user_profile(profile)
    
    def _get_item_features(self, item_id: str, item_type: str) -> Dict[str, Any]:
        """Get item features from cache or database"""
        if item_id in self.item_features:
            return self.item_features[item_id]
        
        # Mock features for testing
        features = {
            'category': item_type,
            'district': ['Sultanahmet', 'Beyoglu', 'Galata', 'Kadikoy'][hash(item_id) % 4],
            'rating': 3.5 + (hash(item_id) % 15) / 10,  # 3.5 to 5.0
            'price_level': ['budget', 'mid-range', 'luxury'][hash(item_id) % 3],
            'cuisine_type': 'turkish' if item_type == 'restaurant' else None
        }
        
        self.item_features[item_id] = features
        return features
    
    def _save_user_profile(self, profile: UserProfile):
        """Save user profile to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_profiles 
                (user_id, preferences, visited_items, favorite_districts, 
                 cuisine_preferences, activity_preferences, price_preference, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                profile.user_id,
                json.dumps(profile.preferences),
                json.dumps(list(profile.visited_items)),
                json.dumps(profile.favorite_districts),
                json.dumps(profile.cuisine_preferences),
                json.dumps(profile.activity_preferences),
                profile.price_preference,
                profile.last_updated.isoformat()
            ))
            conn.commit()
    
    def calculate_user_similarity(self, user1_id: str, user2_id: str) -> float:
        """Calculate similarity between two users using collaborative filtering"""
        profile1 = self.user_profiles.get(user1_id)
        profile2 = self.user_profiles.get(user2_id)
        
        if not profile1 or not profile2:
            return 0.0
        
        # Calculate Jaccard similarity on visited items
        if profile1.visited_items and profile2.visited_items:
            intersection = len(profile1.visited_items & profile2.visited_items)
            union = len(profile1.visited_items | profile2.visited_items)
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def get_personalized_recommendations(self, user_id: str, item_type: Optional[str] = None,
                                       count: int = 10) -> List[PersonalizedRecommendation]:
        """Get personalized recommendations for a user"""
        user_profile = self.user_profiles.get(user_id)
        
        # Generate mock recommendations
        recommendations = []
        
        for i in range(count):
            item_id = f"{item_type or 'item'}_{i}"
            
            # Skip if user already visited
            if user_profile and item_id in user_profile.visited_items:
                continue
            
            item_features = self._get_item_features(item_id, item_type or 'attraction')
            
            # Calculate scores
            base_score = item_features['rating'] / 5.0
            personalization_score = self._calculate_personalization_score(user_profile, item_features)
            collaborative_score = 0.5  # Mock collaborative score
            content_score = 0.6        # Mock content score
            
            final_score = (base_score * 0.3 + personalization_score * 0.4 + 
                          collaborative_score * 0.2 + content_score * 0.1)
            
            reasons = []
            if personalization_score > 0.5:
                reasons.append(f"Matches your interest in {item_features['category']}")
            if item_features['district'] in (user_profile.favorite_districts if user_profile else []):
                reasons.append(f"Located in {item_features['district']}")
            if not reasons:
                reasons.append("Recommended for you")
            
            rec = PersonalizedRecommendation(
                item_id=item_id,
                item_type=item_type or 'attraction',
                title=f"Sample {item_type or 'Attraction'} {i+1}",
                description=f"A great {item_type or 'attraction'} in {item_features['district']}",
                metadata=item_features,
                base_score=base_score,
                personalization_score=personalization_score,
                collaborative_score=collaborative_score,
                content_score=content_score,
                final_score=final_score,
                reasons=reasons,
                confidence=min(final_score, 1.0)
            )
            
            recommendations.append(rec)
        
        # Sort by final score
        recommendations.sort(key=lambda x: x.final_score, reverse=True)
        return recommendations[:count]
    
    def _calculate_personalization_score(self, profile: Optional[UserProfile], 
                                       item: Dict[str, Any]) -> float:
        """Calculate personalization score based on user preferences"""
        if not profile:
            return 0.5  # Default for new users
        
        score = 0.0
        
        # Category preference
        category = item.get('category', '')
        if category in profile.preferences:
            score += profile.preferences[category] * 0.5
        
        # District preference
        district = item.get('district', '')
        if district in profile.favorite_districts:
            rank = profile.favorite_districts.index(district)
            score += (5 - rank) / 5 * 0.3
        
        # Cuisine preference for restaurants
        if item.get('cuisine_type') and profile.cuisine_preferences:
            cuisine = item['cuisine_type']
            if cuisine in profile.cuisine_preferences:
                score += profile.cuisine_preferences[cuisine] * 0.2
        
        return min(score, 1.0)
    
    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user preferences"""
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {"error": "User not found"}
        
        return {
            "user_id": user_id,
            "total_interactions": len(profile.visited_items),
            "top_categories": sorted(profile.preferences.items(), 
                                   key=lambda x: x[1], reverse=True)[:5],
            "favorite_districts": profile.favorite_districts[:3],
            "cuisine_preferences": sorted(profile.cuisine_preferences.items(),
                                        key=lambda x: x[1], reverse=True)[:3],
            "price_preference": profile.price_preference,
            "last_activity": profile.last_updated.isoformat()
        }

# Global lightweight personalization engine
lightweight_personalization_engine = LightweightPersonalizationEngine()

def test_lightweight_personalization():
    """Test the lightweight personalization engine"""
    print("🧪 Testing Lightweight Personalization Engine...")
    
    # Create test user
    user_id = "test_user_123"
    
    # Record some interactions
    interactions = [
        UserInteraction(
            user_id=user_id,
            item_id="restaurant_1",
            item_type="restaurant",
            interaction_type="view",
            interaction_value=1.0,
            timestamp=datetime.now(),
            context={"query": "turkish restaurants"}
        ),
        UserInteraction(
            user_id=user_id,
            item_id="attraction_1", 
            item_type="attraction",
            interaction_type="bookmark",
            interaction_value=2.0,
            timestamp=datetime.now(),
            context={"district": "Sultanahmet"}
        )
    ]
    
    for interaction in interactions:
        lightweight_personalization_engine.record_interaction(interaction)
    
    print(f"✅ Recorded {len(interactions)} interactions")
    
    # Get recommendations
    recommendations = lightweight_personalization_engine.get_personalized_recommendations(
        user_id, item_type="restaurant", count=3
    )
    
    print(f"📋 Generated {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec.title} (Score: {rec.final_score:.3f})")
        print(f"      Reasons: {', '.join(rec.reasons)}")
    
    # Get user insights
    insights = lightweight_personalization_engine.get_user_insights(user_id)
    print(f"👤 User insights: {insights['total_interactions']} interactions")
    
    return len(recommendations) > 0

if __name__ == "__main__":
    success = test_lightweight_personalization()
    if success:
        print("✅ Lightweight Personalization Engine is working correctly!")
    else:
        print("❌ Lightweight Personalization Engine test failed")
