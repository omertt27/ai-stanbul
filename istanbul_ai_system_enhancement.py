#!/usr/bin/env python3
"""
🚀 Istanbul AI System Enhancement Suite
Phase 2: Deep Learning, Analytics, Content Management & Advanced Features

This system provides:
1. 🧠 Deep Learning Integration with PyTorch & Transformers  
2. 📊 Analytics & User Feedback Loop
3. 🔄 Content Update Management System
4. 🌟 Seasonal & Event-Based Recommendations  
5. 📈 Performance Monitoring & Optimization
"""

import json
import logging
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import asyncio
from enum import Enum
import uuid

# Deep Learning & NLP imports
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModel
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    DEEP_LEARNING_AVAILABLE = True
    logging.info("🧠 Deep Learning libraries loaded successfully!")
except ImportError as e:
    DEEP_LEARNING_AVAILABLE = False
    logging.warning(f"⚠️ Deep learning not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    RATING = "rating"           # 1-5 star rating
    COMMENT = "comment"         # Text feedback
    CORRECTION = "correction"   # Information corrections
    SUGGESTION = "suggestion"   # New attraction suggestions
    BUG_REPORT = "bug_report"   # Technical issues
    QUERY_RATING = "query_rating"  # Rate specific responses

class EventType(Enum):
    """Types of Istanbul events"""
    FESTIVAL = "festival"
    EXHIBITION = "exhibition"
    CONCERT = "concert"
    SEASONAL = "seasonal"
    CULTURAL = "cultural"
    RELIGIOUS = "religious"
    WEATHER_DEPENDENT = "weather_dependent"

class Season(Enum):
    """Istanbul seasons for recommendations"""
    SPRING = "spring"    # March-May
    SUMMER = "summer"   # June-August  
    AUTUMN = "autumn"   # September-November
    WINTER = "winter"   # December-February

@dataclass
class UserFeedback:
    """User feedback data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    feedback_type: FeedbackType = FeedbackType.RATING
    content: str = ""
    rating: Optional[int] = None
    query: str = ""
    response_id: str = ""
    attraction_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentUpdate:
    """Content update tracking"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    attraction_id: str = ""
    update_type: str = ""  # hours, price, description, etc.
    old_value: str = ""
    new_value: str = ""
    source: str = ""  # manual, api, feedback
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    verified: bool = False

@dataclass
class SeasonalEvent:
    """Seasonal or special events"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    event_type: EventType = EventType.CULTURAL
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=datetime.now)
    location: str = ""
    description: str = ""
    related_attractions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    active: bool = True

class DeepLearningProcessor:
    """Enhanced NLP and ML processing using transformers"""
    
    def __init__(self):
        self.available = DEEP_LEARNING_AVAILABLE
        if self.available:
            try:
                # Initialize sentence transformer for semantic similarity
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Use a lightweight multilingual model for Turkish-English support
                model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                
                # Sentiment analysis pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
                
                # Question answering pipeline
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="deepset/roberta-base-squad2"
                )
                
                logger.info("🚀 Deep learning models loaded successfully!")
                
            except Exception as e:
                logger.error(f"Failed to initialize deep learning: {e}")
                self.available = False
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using transformers"""
        if not self.available:
            return np.array([])
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy().flatten()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.array([])
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of user feedback"""
        if not self.available:
            return {"label": "NEUTRAL", "score": 0.5}
        
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                "label": result["label"],
                "score": result["score"],
                "confidence": "high" if result["score"] > 0.8 else "medium" if result["score"] > 0.5 else "low"
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "NEUTRAL", "score": 0.5}
    
    def extract_intent_confidence(self, query: str, context: str = "") -> Dict[str, float]:
        """Enhanced intent classification with confidence scores"""
        if not self.available:
            return {"general": 0.5}
        
        try:
            # Define intent patterns with embeddings
            intent_examples = {
                "attraction_search": "I want to visit museums and historical sites in Istanbul",
                "restaurant_search": "Where can I eat good Turkish food in Istanbul",
                "transportation": "How do I get to Sultanahmet from airport",
                "accommodation": "Where should I stay in Istanbul hotels",
                "weather": "What's the weather like in Istanbul today",
                "events": "What events are happening in Istanbul this weekend",
                "shopping": "Where can I shop for souvenirs in Istanbul",
                "nightlife": "What are the best bars and clubs in Istanbul"
            }
            
            query_embedding = self.get_text_embedding(query.lower())
            confidences = {}
            
            for intent, example in intent_examples.items():
                example_embedding = self.get_text_embedding(example)
                if len(query_embedding) > 0 and len(example_embedding) > 0:
                    similarity = cosine_similarity(
                        query_embedding.reshape(1, -1),
                        example_embedding.reshape(1, -1)
                    )[0][0]
                    confidences[intent] = float(similarity)
            
            return confidences
            
        except Exception as e:
            logger.error(f"Intent extraction failed: {e}")
            return {"general": 0.5}

class AnalyticsSystem:
    """User analytics and feedback processing system"""
    
    def __init__(self, db_path: str = "istanbul_analytics.db"):
        self.db_path = db_path
        self.deep_learning = DeepLearningProcessor()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for analytics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    feedback_type TEXT,
                    content TEXT,
                    rating INTEGER,
                    query TEXT,
                    response_id TEXT,
                    attraction_id TEXT,
                    timestamp TEXT,
                    processed BOOLEAN,
                    sentiment_label TEXT,
                    sentiment_score REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_analytics (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    query TEXT,
                    intent TEXT,
                    confidence REAL,
                    response_length INTEGER,
                    response_time REAL,
                    satisfaction_rating INTEGER,
                    timestamp TEXT,
                    attraction_ids TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attraction_popularity (
                    attraction_id TEXT PRIMARY KEY,
                    view_count INTEGER DEFAULT 0,
                    positive_feedback INTEGER DEFAULT 0,
                    negative_feedback INTEGER DEFAULT 0,
                    avg_rating REAL DEFAULT 0.0,
                    last_updated TEXT,
                    trending_score REAL DEFAULT 0.0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_updates (
                    id TEXT PRIMARY KEY,
                    attraction_id TEXT,
                    update_type TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    source TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    verified BOOLEAN,
                    applied BOOLEAN DEFAULT FALSE
                )
            """)
    
    def record_feedback(self, feedback: UserFeedback) -> bool:
        """Record user feedback with sentiment analysis"""
        try:
            # Analyze sentiment if content provided
            sentiment_data = {"label": "NEUTRAL", "score": 0.5}
            if feedback.content:
                sentiment_data = self.deep_learning.analyze_sentiment(feedback.content)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_feedback 
                    (id, user_id, feedback_type, content, rating, query, response_id, 
                     attraction_id, timestamp, processed, sentiment_label, sentiment_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.id, feedback.user_id, feedback.feedback_type.value,
                    feedback.content, feedback.rating, feedback.query, feedback.response_id,
                    feedback.attraction_id, feedback.timestamp.isoformat(), feedback.processed,
                    sentiment_data["label"], sentiment_data["score"], json.dumps(feedback.metadata)
                ))
            
            logger.info(f"📊 Recorded feedback: {feedback.feedback_type.value} from {feedback.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    def record_query_analytics(self, user_id: str, query: str, response: str, 
                              response_time: float, attraction_ids: List[str] = None) -> str:
        """Record query analytics with intent analysis"""
        try:
            query_id = str(uuid.uuid4())
            
            # Analyze intent with confidence
            intent_confidences = self.deep_learning.extract_intent_confidence(query)
            primary_intent = max(intent_confidences.items(), key=lambda x: x[1])
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO query_analytics 
                    (id, user_id, query, intent, confidence, response_length, response_time,
                     timestamp, attraction_ids, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query_id, user_id, query, primary_intent[0], primary_intent[1],
                    len(response), response_time, datetime.now().isoformat(),
                    json.dumps(attraction_ids or []), json.dumps(intent_confidences)
                ))
            
            return query_id
            
        except Exception as e:
            logger.error(f"Failed to record query analytics: {e}")
            return ""
    
    def update_attraction_popularity(self, attraction_id: str, rating: Optional[int] = None):
        """Update attraction popularity metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if attraction exists in popularity table
                cursor = conn.execute(
                    "SELECT view_count, positive_feedback, negative_feedback, avg_rating FROM attraction_popularity WHERE attraction_id = ?",
                    (attraction_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    view_count, pos_feedback, neg_feedback, avg_rating = result
                    view_count += 1
                    
                    if rating:
                        if rating >= 4:
                            pos_feedback += 1
                        elif rating <= 2:
                            neg_feedback += 1
                        
                        # Recalculate average rating
                        total_ratings = pos_feedback + neg_feedback
                        if total_ratings > 0:
                            weighted_rating = (pos_feedback * 4.5 + neg_feedback * 1.5) / total_ratings
                            avg_rating = (avg_rating + weighted_rating) / 2
                    
                    # Calculate trending score (recent popularity boost)
                    trending_score = view_count * 0.1 + pos_feedback * 0.3 - neg_feedback * 0.2
                    
                    conn.execute("""
                        UPDATE attraction_popularity 
                        SET view_count = ?, positive_feedback = ?, negative_feedback = ?, 
                            avg_rating = ?, last_updated = ?, trending_score = ?
                        WHERE attraction_id = ?
                    """, (view_count, pos_feedback, neg_feedback, avg_rating, 
                          datetime.now().isoformat(), trending_score, attraction_id))
                else:
                    # Insert new record
                    initial_rating = rating if rating else 3.0
                    pos_feedback = 1 if rating and rating >= 4 else 0
                    neg_feedback = 1 if rating and rating <= 2 else 0
                    
                    conn.execute("""
                        INSERT INTO attraction_popularity 
                        (attraction_id, view_count, positive_feedback, negative_feedback, 
                         avg_rating, last_updated, trending_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (attraction_id, 1, pos_feedback, neg_feedback, initial_rating,
                          datetime.now().isoformat(), 1.0))
                
        except Exception as e:
            logger.error(f"Failed to update attraction popularity: {e}")
    
    def get_analytics_insights(self) -> Dict[str, Any]:
        """Generate comprehensive analytics insights"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                insights = {}
                
                # Query statistics
                cursor = conn.execute("""
                    SELECT intent, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM query_analytics 
                    WHERE timestamp > datetime('now', '-30 days')
                    GROUP BY intent
                    ORDER BY count DESC
                """)
                insights["popular_intents"] = [
                    {"intent": row[0], "count": row[1], "confidence": row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Top attractions by popularity
                cursor = conn.execute("""
                    SELECT attraction_id, view_count, avg_rating, trending_score
                    FROM attraction_popularity
                    ORDER BY trending_score DESC
                    LIMIT 10
                """)
                insights["trending_attractions"] = [
                    {"id": row[0], "views": row[1], "rating": row[2], "score": row[3]}
                    for row in cursor.fetchall()
                ]
                
                # Feedback summary
                cursor = conn.execute("""
                    SELECT feedback_type, sentiment_label, COUNT(*) as count
                    FROM user_feedback
                    WHERE timestamp > datetime('now', '-30 days')
                    GROUP BY feedback_type, sentiment_label
                """)
                insights["feedback_summary"] = [
                    {"type": row[0], "sentiment": row[1], "count": row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Average response time and satisfaction
                cursor = conn.execute("""
                    SELECT AVG(response_time) as avg_time, AVG(satisfaction_rating) as avg_satisfaction
                    FROM query_analytics
                    WHERE timestamp > datetime('now', '-7 days')
                """)
                result = cursor.fetchone()
                insights["performance"] = {
                    "avg_response_time": result[0] if result[0] else 0,
                    "avg_satisfaction": result[1] if result[1] else 0
                }
                
                return insights
                
        except Exception as e:
            logger.error(f"Failed to generate analytics insights: {e}")
            return {}

class ContentManagementSystem:
    """Automated content update and verification system"""
    
    def __init__(self, db_path: str = "istanbul_analytics.db"):
        self.db_path = db_path
        self.update_sources = {
            "google_places": "https://maps.googleapis.com/maps/api/place",
            "foursquare": "https://api.foursquare.com/v3/places",
            "official_websites": {},  # To be populated with attraction websites
        }
    
    def check_opening_hours_updates(self, attraction_id: str) -> Optional[ContentUpdate]:
        """Check for opening hours updates from external sources"""
        # Placeholder for API integration
        # In production, this would query Google Places API, official websites, etc.
        
        logger.info(f"🔍 Checking opening hours for {attraction_id}")
        
        # Simulate potential update detection
        import random
        if random.random() < 0.1:  # 10% chance of finding an update
            return ContentUpdate(
                attraction_id=attraction_id,
                update_type="opening_hours",
                old_value="09:00-17:00",
                new_value="09:00-18:00",
                source="google_places",
                confidence=0.85,
                verified=False
            )
        return None
    
    def process_feedback_corrections(self, feedback_list: List[UserFeedback]) -> List[ContentUpdate]:
        """Process user feedback for content corrections"""
        updates = []
        
        for feedback in feedback_list:
            if feedback.feedback_type == FeedbackType.CORRECTION and not feedback.processed:
                # Parse correction feedback using NLP
                update = self._parse_correction_feedback(feedback)
                if update:
                    updates.append(update)
                    # Mark feedback as processed
                    self._mark_feedback_processed(feedback.id)
        
        return updates
    
    def _parse_correction_feedback(self, feedback: UserFeedback) -> Optional[ContentUpdate]:
        """Parse user correction feedback to extract updates"""
        content = feedback.content.lower()
        
        # Simple pattern matching for common corrections
        if "price" in content or "cost" in content:
            return ContentUpdate(
                attraction_id=feedback.attraction_id or "",
                update_type="price",
                old_value="unknown",
                new_value=content,
                source="user_feedback",
                confidence=0.6,
                verified=False
            )
        elif "hours" in content or "open" in content or "closed" in content:
            return ContentUpdate(
                attraction_id=feedback.attraction_id or "",
                update_type="opening_hours",
                old_value="unknown",
                new_value=content,
                source="user_feedback",
                confidence=0.6,
                verified=False
            )
        
        return None
    
    def _mark_feedback_processed(self, feedback_id: str):
        """Mark feedback as processed in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE user_feedback SET processed = TRUE WHERE id = ?",
                    (feedback_id,)
                )
        except Exception as e:
            logger.error(f"Failed to mark feedback as processed: {e}")

class SeasonalEventSystem:
    """Manage seasonal recommendations and events"""
    
    def __init__(self):
        self.events_db = []
        self._load_seasonal_events()
    
    def _load_seasonal_events(self):
        """Load predefined seasonal events and recommendations"""
        self.events_db = [
            SeasonalEvent(
                name="Istanbul Tulip Festival",
                event_type=EventType.SEASONAL,
                start_date=datetime(2024, 4, 1),
                end_date=datetime(2024, 5, 15),
                location="Emirgan Park",
                description="Beautiful tulip displays across Istanbul parks",
                related_attractions=["emirgan_park", "gulhane_park", "yildiz_park"],
                tags=["spring", "flowers", "photography", "family"]
            ),
            SeasonalEvent(
                name="Summer Rooftop Season",
                event_type=EventType.SEASONAL,
                start_date=datetime(2024, 6, 1),
                end_date=datetime(2024, 9, 15),
                location="Various",
                description="Best rooftop bars and restaurants with Bosphorus views",
                related_attractions=["galata_tower", "swissotel_bosphorus_view"],
                tags=["summer", "rooftop", "sunset", "romantic", "drinks"]
            ),
            SeasonalEvent(
                name="Ramadan and Eid Celebrations",
                event_type=EventType.RELIGIOUS,
                start_date=datetime(2024, 3, 10),
                end_date=datetime(2024, 4, 10),
                location="Sultanahmet and Eyüp",
                description="Special atmosphere during Ramadan with iftar and festivities",
                related_attractions=["blue_mosque", "hagia_sophia", "eyup_sultan_mosque"],
                tags=["ramadan", "eid", "culture", "spiritual", "evening"]
            )
        ]
    
    def get_current_season(self) -> Season:
        """Determine current season in Istanbul"""
        month = datetime.now().month
        if 3 <= month <= 5:
            return Season.SPRING
        elif 6 <= month <= 8:
            return Season.SUMMER
        elif 9 <= month <= 11:
            return Season.AUTUMN
        else:
            return Season.WINTER
    
    def get_seasonal_recommendations(self, season: Season = None) -> List[Dict[str, Any]]:
        """Get recommendations based on current or specified season"""
        if not season:
            season = self.get_current_season()
        
        current_date = datetime.now()
        recommendations = []
        
        # Find active events for the season
        for event in self.events_db:
            if (event.start_date <= current_date <= event.end_date and
                season.value in [tag.lower() for tag in event.tags]):
                recommendations.append({
                    "type": "event",
                    "name": event.name,
                    "description": event.description,
                    "location": event.location,
                    "related_attractions": event.related_attractions,
                    "tags": event.tags
                })
        
        # Add seasonal attraction recommendations
        seasonal_attractions = self._get_seasonal_attraction_recommendations(season)
        recommendations.extend(seasonal_attractions)
        
        return recommendations
    
    def _get_seasonal_attraction_recommendations(self, season: Season) -> List[Dict[str, Any]]:
        """Get attraction recommendations specific to seasons"""
        recommendations = []
        
        if season == Season.SPRING:
            recommendations.append({
                "type": "seasonal_attraction",
                "title": "Spring Parks and Gardens",
                "description": "Perfect weather for outdoor exploration and tulip viewing",
                "attractions": ["emirgan_park", "gulhane_park", "yildiz_park", "macka_park"],
                "reason": "Mild weather and blooming flowers make parks especially beautiful"
            })
        
        elif season == Season.SUMMER:
            recommendations.append({
                "type": "seasonal_attraction",
                "title": "Bosphorus and Waterfront",
                "description": "Enjoy waterfront dining and Bosphorus cruises",
                "attractions": ["bosphorus_cruise", "karakoy_waterfront", "maiden_tower"],
                "reason": "Warm weather perfect for water activities and outdoor dining"
            })
            
            recommendations.append({
                "type": "seasonal_attraction",
                "title": "Islands Escape",
                "description": "Cool off on the Princes' Islands",
                "attractions": ["buyukada", "heybeliada", "burgazada", "kinaliada"],
                "reason": "Sea breeze and swimming opportunities provide relief from city heat"
            })
        
        elif season == Season.AUTUMN:
            recommendations.append({
                "type": "seasonal_attraction",
                "title": "Cultural Indoor Experiences",
                "description": "Perfect weather for walking and sightseeing",
                "attractions": ["topkapi_palace", "archaeological_museum", "turkish_islamic_arts_museum"],
                "reason": "Comfortable temperatures ideal for extensive sightseeing"
            })
        
        elif season == Season.WINTER:
            recommendations.append({
                "type": "seasonal_attraction",
                "title": "Cozy Indoor Attractions",
                "description": "Warm up in museums and covered bazaars",
                "attractions": ["grand_bazaar", "spice_bazaar", "galata_mevlevi_lodge", "salt_galata"],
                "reason": "Indoor attractions provide warmth and shelter from winter weather"
            })
        
        return recommendations

class EnhancedIstanbulAISystem:
    """Enhanced Istanbul AI system with all new features integrated"""
    
    def __init__(self):
        self.analytics = AnalyticsSystem()
        self.content_manager = ContentManagementSystem()
        self.seasonal_system = SeasonalEventSystem()
        self.deep_learning = DeepLearningProcessor()
        
        logger.info("🚀 Enhanced Istanbul AI System initialized!")
        logger.info(f"🧠 Deep Learning: {'✅ Available' if self.deep_learning.available else '❌ Fallback mode'}")
    
    def process_enhanced_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process query with enhanced analytics and recommendations"""
        start_time = datetime.now()
        
        # Analyze intent with confidence
        intent_analysis = self.deep_learning.extract_intent_confidence(query)
        primary_intent = max(intent_analysis.items(), key=lambda x: x[1])
        
        # Get seasonal recommendations if relevant
        seasonal_recs = []
        if any(keyword in query.lower() for keyword in ["recommend", "suggest", "what to", "where to"]):
            seasonal_recs = self.seasonal_system.get_seasonal_recommendations()
        
        # Generate response (placeholder - would integrate with existing system)
        response = self._generate_enhanced_response(query, primary_intent, seasonal_recs)
        
        # Record analytics
        response_time = (datetime.now() - start_time).total_seconds()
        query_id = self.analytics.record_query_analytics(
            user_id, query, response["content"], response_time, response.get("attraction_ids", [])
        )
        
        return {
            "response": response["content"],
            "intent": primary_intent[0],
            "confidence": primary_intent[1],
            "seasonal_recommendations": seasonal_recs,
            "query_id": query_id,
            "processing_time": response_time,
            "deep_learning_enabled": self.deep_learning.available
        }
    
    def _generate_enhanced_response(self, query: str, intent: Tuple[str, float], 
                                  seasonal_recs: List[Dict]) -> Dict[str, Any]:
        """Generate enhanced response with seasonal context"""
        base_response = f"Based on your query about {intent[0]}, here are my recommendations..."
        
        # Add seasonal context
        if seasonal_recs:
            season = self.seasonal_system.get_current_season()
            seasonal_text = f"\n\n🌟 Since it's {season.value}, I especially recommend:\n"
            for rec in seasonal_recs[:2]:  # Top 2 seasonal recommendations
                seasonal_text += f"• {rec.get('title', rec.get('name', ''))}: {rec.get('description', '')}\n"
            base_response += seasonal_text
        
        return {
            "content": base_response,
            "attraction_ids": [],  # Would be populated by actual system
            "enhanced_features": True
        }
    
    def submit_feedback(self, user_id: str, feedback_type: str, content: str, 
                       rating: Optional[int] = None, query_id: str = "", 
                       attraction_id: str = "") -> bool:
        """Submit user feedback with enhanced processing"""
        feedback = UserFeedback(
            user_id=user_id,
            feedback_type=FeedbackType(feedback_type),
            content=content,
            rating=rating,
            response_id=query_id,
            attraction_id=attraction_id
        )
        
        success = self.analytics.record_feedback(feedback)
        
        if success and attraction_id:
            self.analytics.update_attraction_popularity(attraction_id, rating)
        
        return success
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Get comprehensive system insights and analytics"""
        analytics = self.analytics.get_analytics_insights()
        
        return {
            "analytics": analytics,
            "system_status": {
                "deep_learning": self.deep_learning.available,
                "total_feedback": len(analytics.get("feedback_summary", [])),
                "trending_attractions": len(analytics.get("trending_attractions", [])),
                "current_season": self.seasonal_system.get_current_season().value,
                "active_events": len([e for e in self.seasonal_system.events_db 
                                   if e.start_date <= datetime.now() <= e.end_date])
            },
            "seasonal_context": self.seasonal_system.get_seasonal_recommendations()[:3]
        }

def main():
    """Demo of the enhanced system"""
    print("🚀 Istanbul AI System Enhancement Suite")
    print("=" * 50)
    
    # Initialize enhanced system
    system = EnhancedIstanbulAISystem()
    
    # Demo queries
    test_queries = [
        "What are the best attractions to visit in Istanbul?",
        "Where can I eat traditional Turkish food?",
        "What should I do in Istanbul during spring?",
        "Show me some rooftop bars with good views"
    ]
    
    user_id = "demo_user"
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = system.process_enhanced_query(user_id, query)
        
        print(f"🎯 Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"⏱️ Processing time: {result['processing_time']:.3f}s")
        print(f"🧠 Deep learning: {'✅' if result['deep_learning_enabled'] else '❌'}")
        
        if result['seasonal_recommendations']:
            print(f"🌟 Seasonal suggestions: {len(result['seasonal_recommendations'])} available")
    
    # Submit demo feedback
    print(f"\n📊 Submitting demo feedback...")
    system.submit_feedback(user_id, "rating", "Great recommendations!", 5)
    system.submit_feedback(user_id, "suggestion", "Please add more information about opening hours", attraction_id="topkapi_palace")
    
    # Get system insights
    insights = system.get_system_insights()
    print(f"\n📈 System Status:")
    print(f"  Current season: {insights['system_status']['current_season']}")
    print(f"  Deep learning: {'✅ Enabled' if insights['system_status']['deep_learning'] else '❌ Fallback'}")
    print(f"  Active events: {insights['system_status']['active_events']}")
    
    print("\n✅ Enhanced Istanbul AI System demonstration complete!")

if __name__ == "__main__":
    main()
