#!/usr/bin/env python3
"""
Analytics Database for Real-time Blog Analytics
Tracks user engagement, page views, and metrics
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import logging

logger = logging.getLogger(__name__)

class AnalyticsDB:
    """Simple SQLite database for tracking blog analytics"""
    
    def __init__(self, db_path: str = "blog_analytics.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS page_views (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    page_path TEXT NOT NULL,
                    user_agent TEXT,
                    ip_address TEXT,
                    session_id TEXT,
                    duration_seconds INTEGER DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS blog_engagement (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    post_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_session TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS active_sessions (
                    session_id TEXT PRIMARY KEY,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    current_page TEXT,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_page_views_timestamp ON page_views(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_engagement_timestamp ON blog_engagement(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_active_sessions_activity ON active_sessions(last_activity)")
            
            conn.commit()
    
    def track_page_view(self, page_path: str, user_agent: Optional[str] = None, ip_address: Optional[str] = None, session_id: Optional[str] = None):
        """Track a page view"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO page_views (page_path, user_agent, ip_address, session_id)
                        VALUES (?, ?, ?, ?)
                    """, (page_path, user_agent, ip_address, session_id))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error tracking page view: {e}")
    
    def track_blog_engagement(self, post_id: str, event_type: str, user_session: Optional[str] = None, metadata: Optional[Dict] = None):
        """Track blog engagement event"""
        with self.lock:
            try:
                metadata_json = json.dumps(metadata) if metadata else None
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO blog_engagement (post_id, event_type, user_session, metadata)
                        VALUES (?, ?, ?, ?)
                    """, (post_id, event_type, user_session, metadata_json))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error tracking blog engagement: {e}")
    
    def update_active_session(self, session_id: str, current_page: str):
        """Update active session"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO active_sessions (session_id, last_activity, current_page, start_time)
                        VALUES (?, CURRENT_TIMESTAMP, ?, COALESCE((SELECT start_time FROM active_sessions WHERE session_id = ?), CURRENT_TIMESTAMP))
                    """, (session_id, current_page, session_id))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error updating active session: {e}")
    
    def get_active_readers_count(self, minutes: int = 5) -> int:
        """Get count of active readers in last N minutes"""
        try:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(DISTINCT session_id) 
                    FROM active_sessions 
                    WHERE last_activity > ?
                """, (cutoff_time.isoformat(),))
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting active readers: {e}")
            return 0
    
    def get_todays_stats(self) -> Dict[str, Any]:
        """Get today's statistics"""
        try:
            today = datetime.now().date()
            with sqlite3.connect(self.db_path) as conn:
                # Page views today
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM page_views 
                    WHERE DATE(timestamp) = ?
                """, (today.isoformat(),))
                page_views_today = cursor.fetchone()[0]
                
                # Blog posts read today (blog page views)
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM page_views 
                    WHERE DATE(timestamp) = ? AND page_path LIKE '/blog%'
                """, (today.isoformat(),))
                blog_reads_today = cursor.fetchone()[0]
                
                # Engagement events today
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) FROM blog_engagement 
                    WHERE DATE(timestamp) = ?
                    GROUP BY event_type
                """, (today.isoformat(),))
                engagement_today = dict(cursor.fetchall())
                
                return {
                    "page_views_today": page_views_today,
                    "blog_reads_today": blog_reads_today,
                    "engagement_today": engagement_today,
                    "likes_today": engagement_today.get("like", 0),
                    "shares_today": engagement_today.get("share", 0),
                    "comments_today": engagement_today.get("comment", 0)
                }
        except Exception as e:
            logger.error(f"Error getting today's stats: {e}")
            return {
                "page_views_today": 0,
                "blog_reads_today": 0,
                "engagement_today": {},
                "likes_today": 0,
                "shares_today": 0,
                "comments_today": 0
            }
    
    def get_top_posts(self, days: int = 7, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing posts"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        post_id,
                        COUNT(CASE WHEN event_type = 'view' THEN 1 END) as views,
                        COUNT(CASE WHEN event_type = 'like' THEN 1 END) as likes,
                        COUNT(CASE WHEN event_type = 'share' THEN 1 END) as shares,
                        COUNT(*) as total_engagement
                    FROM blog_engagement 
                    WHERE timestamp > ?
                    GROUP BY post_id
                    ORDER BY total_engagement DESC, views DESC
                    LIMIT ?
                """, (cutoff_date.isoformat(), limit))
                
                posts = []
                for row in cursor.fetchall():
                    post_id, views, likes, shares, total_engagement = row
                    engagement_rate = (likes + shares) / max(views, 1)
                    posts.append({
                        "post_id": post_id,
                        "views": views,
                        "likes": likes,
                        "shares": shares,
                        "engagement_rate": round(engagement_rate, 3),
                        "total_engagement": total_engagement
                    })
                
                return posts
        except Exception as e:
            logger.error(f"Error getting top posts: {e}")
            return []
    
    def get_hourly_engagement_rate(self) -> Dict[str, float]:
        """Get engagement rates for current hour"""
        try:
            hour_ago = datetime.now() - timedelta(hours=1)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT event_type, COUNT(*) FROM blog_engagement 
                    WHERE timestamp > ?
                    GROUP BY event_type
                """, (hour_ago.isoformat(),))
                
                engagement = dict(cursor.fetchall())
                return {
                    "likes_per_hour": engagement.get("like", 0),
                    "shares_per_hour": engagement.get("share", 0),
                    "comments_per_hour": engagement.get("comment", 0),
                    "views_per_hour": engagement.get("view", 0)
                }
        except Exception as e:
            logger.error(f"Error getting hourly engagement: {e}")
            return {
                "likes_per_hour": 0,
                "shares_per_hour": 0, 
                "comments_per_hour": 0,
                "views_per_hour": 0
            }
    
    def cleanup_old_sessions(self, hours: int = 24):
        """Clean up old inactive sessions"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM active_sessions 
                    WHERE last_activity < ?
                """, (cutoff_time.isoformat(),))
                conn.commit()
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")

# Global analytics instance
analytics_db = AnalyticsDB()
