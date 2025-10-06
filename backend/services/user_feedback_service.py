"""
User Feedback and Rating System Service
Implements crowdsourced accuracy boost system for AI Istanbul recommendations
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import statistics
import uuid

class UserType(Enum):
    TOURIST = "tourist"
    LOCAL = "local"
    GUIDE = "guide"
    EXPERT = "expert"

class FeedbackCategory(Enum):
    OVERALL = "overall"
    AUTHENTICITY = "authenticity"
    ACCESSIBILITY = "accessibility"
    VALUE = "value"
    SERVICE = "service"
    ACCURACY = "accuracy"

@dataclass
class UserRating:
    id: str
    user_id: str
    attraction_id: int
    ratings: Dict[FeedbackCategory, float]
    comment: str
    visit_date: str
    user_type: UserType
    timestamp: datetime
    helpful_votes: int = 0
    verified: bool = False

class UserFeedbackService:
    """Service for managing user feedback and crowdsourced accuracy improvements"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or ":memory:"
        self.init_database()
        
    def init_database(self):
        """Initialize the feedback database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # User ratings table with multi-dimensional ratings
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_ratings (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            attraction_id INTEGER NOT NULL,
            overall_rating REAL NOT NULL CHECK (overall_rating >= 1.0 AND overall_rating <= 10.0),
            authenticity_rating REAL CHECK (authenticity_rating >= 1.0 AND authenticity_rating <= 10.0),
            accessibility_rating REAL CHECK (accessibility_rating >= 1.0 AND accessibility_rating <= 10.0),
            value_rating REAL CHECK (value_rating >= 1.0 AND value_rating <= 10.0),
            service_rating REAL CHECK (service_rating >= 1.0 AND service_rating <= 10.0),
            accuracy_rating REAL CHECK (accuracy_rating >= 1.0 AND accuracy_rating <= 10.0),
            comment TEXT,
            visit_date TEXT,
            user_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            helpful_votes INTEGER DEFAULT 0,
            verified BOOLEAN DEFAULT FALSE
        )
        ''')
        
        # Recommendation accuracy tracking
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS recommendation_accuracy (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            item_name TEXT NOT NULL,
            predicted_rating REAL,
            actual_rating REAL,
            accuracy_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Feedback trends and analytics
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_trends (
            id TEXT PRIMARY KEY,
            attraction_id INTEGER NOT NULL,
            period TEXT NOT NULL,  -- daily, weekly, monthly
            avg_overall_rating REAL,
            avg_authenticity_rating REAL,
            rating_count INTEGER,
            boost_factor REAL,
            calculated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # User credibility scores
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_credibility (
            user_id TEXT PRIMARY KEY,
            credibility_score REAL DEFAULT 1.0,
            total_ratings INTEGER DEFAULT 0,
            verified_local BOOLEAN DEFAULT FALSE,
            expertise_areas TEXT,  -- JSON array of areas
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
        
    def add_rating(self, user_id: str, attraction_id: int, ratings: Dict[str, float],
                   comment: str = "", visit_date: str = "", user_type: str = "tourist") -> str:
        """Add a new user rating"""
        rating_id = str(uuid.uuid4())
        
        self.cursor.execute('''
        INSERT INTO user_ratings (
            id, user_id, attraction_id, overall_rating, authenticity_rating,
            accessibility_rating, value_rating, service_rating, accuracy_rating,
            comment, visit_date, user_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            rating_id, user_id, attraction_id,
            ratings.get('overall', 5.0),
            ratings.get('authenticity', 5.0),
            ratings.get('accessibility', 5.0),
            ratings.get('value', 5.0),
            ratings.get('service', 5.0),
            ratings.get('accuracy', 5.0),
            comment, visit_date, user_type
        ))
        
        self.conn.commit()
        
        # Update user credibility
        self._update_user_credibility(user_id)
        
        return rating_id
    
    def get_attraction_ratings(self, attraction_id: int) -> Dict[str, Any]:
        """Get aggregated ratings for an attraction"""
        self.cursor.execute('''
        SELECT 
            AVG(overall_rating) as avg_overall,
            AVG(authenticity_rating) as avg_authenticity,
            AVG(accessibility_rating) as avg_accessibility,
            AVG(value_rating) as avg_value,
            AVG(service_rating) as avg_service,
            AVG(accuracy_rating) as avg_accuracy,
            COUNT(*) as total_ratings,
            AVG(CASE WHEN user_type = 'local' THEN overall_rating END) as local_avg,
            AVG(CASE WHEN user_type = 'tourist' THEN overall_rating END) as tourist_avg
        FROM user_ratings 
        WHERE attraction_id = ?
        ''', (attraction_id,))
        
        result = self.cursor.fetchone()
        if not result or result[6] == 0:  # total_ratings == 0
            return {
                'avg_ratings': {},
                'total_ratings': 0,
                'boost_factor': 0.0,
                'local_avg': 0.0,
                'tourist_avg': 0.0
            }
        
        avg_ratings = {
            'overall': result[0] or 0.0,
            'authenticity': result[1] or 0.0,
            'accessibility': result[2] or 0.0,
            'value': result[3] or 0.0,
            'service': result[4] or 0.0,
            'accuracy': result[5] or 0.0
        }
        
        # Calculate crowdsourced boost factor
        boost_factor = self._calculate_boost_factor(avg_ratings, result[6])
        
        return {
            'avg_ratings': avg_ratings,
            'total_ratings': result[6],
            'boost_factor': boost_factor,
            'local_avg': result[7] or 0.0,
            'tourist_avg': result[8] or 0.0
        }
    
    def _calculate_boost_factor(self, avg_ratings: Dict[str, float], rating_count: int) -> float:
        """Calculate the crowdsourced accuracy boost factor"""
        if rating_count == 0:
            return 0.0
        
        # Base boost from overall rating (above baseline of 7.0)
        overall_boost = max(0, (avg_ratings['overall'] - 7.0) * 0.1)
        
        # Authenticity weight (higher importance)
        auth_boost = max(0, (avg_ratings['authenticity'] - 7.0) * 0.15)
        
        # Accuracy weight (user feedback on recommendation accuracy)
        accuracy_boost = max(0, (avg_ratings['accuracy'] - 7.0) * 0.12)
        
        # Sample size confidence factor
        confidence_factor = min(1.0, rating_count / 10.0)  # Full confidence at 10+ ratings
        
        # Maximum boost of 1.0 point
        boost = min(1.0, (overall_boost + auth_boost + accuracy_boost) * confidence_factor)
        
        return round(boost, 2)
    
    def get_top_rated_attractions(self, limit: int = 10, user_type: str = None) -> List[Dict]:
        """Get top-rated attractions with boost factors applied"""
        where_clause = ""
        params = []
        
        if user_type:
            where_clause = "WHERE user_type = ?"
            params.append(user_type)
        
        self.cursor.execute(f'''
        SELECT 
            attraction_id,
            AVG(overall_rating) as avg_rating,
            AVG(authenticity_rating) as avg_auth,
            COUNT(*) as rating_count
        FROM user_ratings 
        {where_clause}
        GROUP BY attraction_id
        HAVING rating_count >= 2
        ORDER BY avg_rating DESC
        LIMIT ?
        ''', params + [limit])
        
        results = []
        for attraction_id, avg_rating, avg_auth, count in self.cursor.fetchall():
            boost = self._calculate_boost_factor(
                {'overall': avg_rating, 'authenticity': avg_auth, 'accuracy': avg_rating}, 
                count
            )
            results.append({
                'attraction_id': attraction_id,
                'avg_rating': round(avg_rating, 1),
                'boosted_rating': round(avg_rating + boost, 1),
                'rating_count': count,
                'boost_factor': boost
            })
        
        return results
    
    def get_recent_feedback(self, days: int = 7) -> List[Dict]:
        """Get recent user feedback for analysis"""
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        self.cursor.execute('''
        SELECT id, user_id, attraction_id, overall_rating, authenticity_rating,
               comment, user_type, timestamp
        FROM user_ratings 
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        ''', (since_date,))
        
        return [
            {
                'id': row[0],
                'user_id': row[1],
                'attraction_id': row[2],
                'overall_rating': row[3],
                'authenticity_rating': row[4],
                'comment': row[5],
                'user_type': row[6],
                'timestamp': row[7]
            }
            for row in self.cursor.fetchall()
        ]
    
    def _update_user_credibility(self, user_id: str):
        """Update user credibility score based on rating history"""
        # Get user's rating statistics
        self.cursor.execute('''
        SELECT COUNT(*) as total_ratings,
               AVG(overall_rating) as avg_rating,
               COUNT(CASE WHEN verified = TRUE THEN 1 END) as verified_count
        FROM user_ratings 
        WHERE user_id = ?
        ''', (user_id,))
        
        result = self.cursor.fetchone()
        if not result:
            return
        
        total_ratings, avg_rating, verified_count = result
        
        # Calculate credibility score
        base_score = 1.0
        
        # Experience factor (more ratings = higher credibility)
        experience_boost = min(0.5, total_ratings * 0.05)
        
        # Quality factor (consistent ratings)
        quality_boost = 0.3 if avg_rating and 6.0 <= avg_rating <= 9.0 else 0.0
        
        # Verification factor
        verification_boost = 0.2 if verified_count > 0 else 0.0
        
        credibility_score = base_score + experience_boost + quality_boost + verification_boost
        
        # Update credibility table
        self.cursor.execute('''
        INSERT OR REPLACE INTO user_credibility (
            user_id, credibility_score, total_ratings, last_updated
        ) VALUES (?, ?, ?, ?)
        ''', (user_id, round(credibility_score, 2), total_ratings, datetime.now().isoformat()))
        
        self.conn.commit()
    
    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Get comprehensive feedback analytics"""
        analytics = {}
        
        # Total feedback statistics
        self.cursor.execute('SELECT COUNT(*) FROM user_ratings')
        analytics['total_ratings'] = self.cursor.fetchone()[0]
        
        # Average ratings by category
        self.cursor.execute('''
        SELECT 
            AVG(overall_rating) as avg_overall,
            AVG(authenticity_rating) as avg_auth,
            AVG(accessibility_rating) as avg_access,
            AVG(value_rating) as avg_value,
            AVG(service_rating) as avg_service
        FROM user_ratings
        ''')
        
        result = self.cursor.fetchone()
        analytics['avg_ratings'] = {
            'overall': round(result[0] or 0, 2),
            'authenticity': round(result[1] or 0, 2),
            'accessibility': round(result[2] or 0, 2),
            'value': round(result[3] or 0, 2),
            'service': round(result[4] or 0, 2)
        }
        
        # User type distribution
        self.cursor.execute('''
        SELECT user_type, COUNT(*) 
        FROM user_ratings 
        GROUP BY user_type
        ''')
        
        analytics['user_type_distribution'] = dict(self.cursor.fetchall())
        
        # Recent activity (last 30 days)
        since_date = (datetime.now() - timedelta(days=30)).isoformat()
        self.cursor.execute('''
        SELECT COUNT(*) 
        FROM user_ratings 
        WHERE timestamp >= ?
        ''', (since_date,))
        
        analytics['recent_activity'] = self.cursor.fetchone()[0]
        
        return analytics
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
