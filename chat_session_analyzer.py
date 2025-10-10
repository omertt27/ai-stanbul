#!/usr/bin/env python3
"""
🎯 Istanbul AI Chat Session Analyzer
Enhanced analytics for like/dislike feedback integration with admin dashboard

This system:
- Analyzes chat sessions with like/dislike feedback
- Provides deep insights into user satisfaction patterns
- Identifies improvement opportunities based on feedback
- Integrates seamlessly with existing admin dashboard
- Generates actionable recommendations for content updates
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import Counter, defaultdict
import re
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Chat feedback types from like/dislike buttons"""
    LIKE = "like"
    DISLIKE = "dislike"
    MIXED = "mixed"  # Session with both likes and dislikes
    UNKNOWN = "unknown"  # No feedback given

@dataclass
class ChatMessage:
    """Individual chat message with feedback"""
    id: str
    user_message: str
    ai_response: str
    timestamp: datetime
    feedback: Optional[FeedbackType] = None
    intent_detected: Optional[str] = None
    confidence_score: Optional[float] = None
    response_time: Optional[float] = None

@dataclass
class ChatSession:
    """Complete chat session analysis"""
    session_id: str
    user_ip: str
    title: str
    messages: List[ChatMessage]
    overall_feedback: FeedbackType
    satisfaction_score: float  # 0-1 scale
    session_duration: float  # minutes
    total_messages: int
    like_count: int
    dislike_count: int
    created_at: datetime
    last_activity: datetime

@dataclass
class FeedbackAnalysis:
    """Deep analysis of user feedback patterns"""
    total_sessions: int
    total_messages: int
    satisfaction_rate: float
    like_percentage: float
    dislike_percentage: float
    mixed_feedback_percentage: float
    top_liked_topics: List[Tuple[str, int]]
    top_disliked_topics: List[Tuple[str, int]]
    improvement_suggestions: List[str]
    temporal_trends: Dict[str, Any]

class ChatSessionAnalyzer:
    """Enhanced chat session analyzer for admin dashboard integration"""
    
    def __init__(self, db_path: str = "chat_sessions.db"):
        self.db_path = db_path
        self.setup_database()
        logger.info("🎯 Chat Session Analyzer initialized")
    
    def setup_database(self):
        """Initialize database for chat session analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced chat sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_ip TEXT,
                title TEXT,
                total_messages INTEGER,
                like_count INTEGER,
                dislike_count INTEGER,
                overall_feedback TEXT,
                satisfaction_score REAL,
                session_duration REAL,
                created_at DATETIME,
                last_activity DATETIME,
                analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Enhanced messages analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS message_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id TEXT,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                feedback TEXT,
                intent_detected TEXT,
                confidence_score REAL,
                response_time REAL,
                timestamp DATETIME,
                topic_keywords TEXT,
                sentiment_score REAL,
                analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feedback insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT NOT NULL,
                topic TEXT,
                feedback_type TEXT,
                frequency INTEGER,
                confidence REAL,
                recommendation TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_sessions INTEGER,
                total_messages INTEGER,
                like_rate REAL,
                dislike_rate REAL,
                satisfaction_score REAL,
                top_issues TEXT,
                improvements_made TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("📊 Analytics database initialized")
    
    def import_chat_sessions_from_admin(self, admin_sessions_data: List[Dict]) -> int:
        """Import chat sessions from admin dashboard format"""
        imported_count = 0
        
        for session_data in admin_sessions_data:
            try:
                # Parse session data from admin dashboard format
                session = self._parse_admin_session_format(session_data)
                if session:
                    self._store_session_analysis(session)
                    imported_count += 1
            except Exception as e:
                logger.error(f"❌ Error importing session {session_data.get('id', 'unknown')}: {e}")
        
        logger.info(f"📥 Successfully imported {imported_count} chat sessions")
        return imported_count
    
    def _parse_admin_session_format(self, session_data: Dict) -> Optional[ChatSession]:
        """Parse session data from admin dashboard format"""
        try:
            session_id = session_data['id']
            user_ip = session_data.get('user_ip', 'unknown')
            title = session_data.get('title', 'Untitled Chat')
            conversation_history = session_data.get('conversation_history', [])
            
            # Parse messages
            messages = []
            like_count = 0
            dislike_count = 0
            
            for i, entry in enumerate(conversation_history):
                if isinstance(entry, dict):
                    user_msg = entry.get('user', entry.get('message', ''))
                    ai_response = entry.get('ai', entry.get('response', ''))
                    feedback_str = entry.get('feedback')
                    timestamp = datetime.fromisoformat(entry.get('timestamp', datetime.now().isoformat()))
                    
                    # Convert feedback
                    feedback = None
                    if feedback_str == 'like':
                        feedback = FeedbackType.LIKE
                        like_count += 1
                    elif feedback_str == 'dislike':
                        feedback = FeedbackType.DISLIKE
                        dislike_count += 1
                    
                    message = ChatMessage(
                        id=f"{session_id}_msg_{i}",
                        user_message=user_msg,
                        ai_response=ai_response,
                        timestamp=timestamp,
                        feedback=feedback,
                        intent_detected=self._extract_intent(user_msg),
                        confidence_score=entry.get('confidence', 0.8)
                    )
                    messages.append(message)
            
            # Determine overall feedback
            total_feedback = like_count + dislike_count
            if total_feedback == 0:
                overall_feedback = FeedbackType.UNKNOWN
                satisfaction_score = 0.5  # Neutral
            elif like_count > 0 and dislike_count > 0:
                overall_feedback = FeedbackType.MIXED
                satisfaction_score = like_count / total_feedback
            elif like_count > 0:
                overall_feedback = FeedbackType.LIKE
                satisfaction_score = 0.9
            else:
                overall_feedback = FeedbackType.DISLIKE
                satisfaction_score = 0.1
            
            # Calculate session duration
            if messages:
                start_time = min(msg.timestamp for msg in messages)
                end_time = max(msg.timestamp for msg in messages)
                duration = (end_time - start_time).total_seconds() / 60  # minutes
            else:
                duration = 0
            
            return ChatSession(
                session_id=session_id,
                user_ip=user_ip,
                title=title,
                messages=messages,
                overall_feedback=overall_feedback,
                satisfaction_score=satisfaction_score,
                session_duration=duration,
                total_messages=len(messages),
                like_count=like_count,
                dislike_count=dislike_count,
                created_at=datetime.fromisoformat(session_data.get('saved_at', datetime.now().isoformat())),
                last_activity=messages[-1].timestamp if messages else datetime.now()
            )
            
        except Exception as e:
            logger.error(f"❌ Error parsing session data: {e}")
            return None
    
    def _extract_intent(self, user_message: str) -> str:
        """Extract intent from user message (simple keyword-based)"""
        user_msg_lower = user_message.lower()
        
        if any(word in user_msg_lower for word in ['attraction', 'visit', 'see', 'show', 'recommend']):
            return 'attraction_search'
        elif any(word in user_msg_lower for word in ['restaurant', 'eat', 'food', 'dining']):
            return 'restaurant_search'
        elif any(word in user_msg_lower for word in ['hotel', 'stay', 'accommodation']):
            return 'accommodation'
        elif any(word in user_msg_lower for word in ['weather', 'temperature', 'rain', 'sunny']):
            return 'weather'
        elif any(word in user_msg_lower for word in ['transport', 'metro', 'bus', 'taxi']):
            return 'transportation'
        else:
            return 'general'
    
    def _store_session_analysis(self, session: ChatSession):
        """Store analyzed session data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store session analysis
        cursor.execute('''
            INSERT OR REPLACE INTO chat_sessions_analysis 
            (session_id, user_ip, title, total_messages, like_count, dislike_count,
             overall_feedback, satisfaction_score, session_duration, created_at, last_activity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session.session_id, session.user_ip, session.title, session.total_messages,
            session.like_count, session.dislike_count, session.overall_feedback.value,
            session.satisfaction_score, session.session_duration, session.created_at,
            session.last_activity
        ))
        
        # Store message analysis
        for message in session.messages:
            topic_keywords = self._extract_keywords(message.user_message + " " + message.ai_response)
            sentiment_score = self._calculate_sentiment_score(message.feedback)
            
            cursor.execute('''
                INSERT OR REPLACE INTO message_analysis 
                (session_id, message_id, user_message, ai_response, feedback,
                 intent_detected, confidence_score, response_time, timestamp,
                 topic_keywords, sentiment_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session.session_id, message.id, message.user_message, message.ai_response,
                message.feedback.value if message.feedback else None,
                message.intent_detected, message.confidence_score, message.response_time,
                message.timestamp, topic_keywords, sentiment_score
            ))
        
        conn.commit()
        conn.close()
    
    def _extract_keywords(self, text: str) -> str:
        """Extract keywords from text"""
        # Istanbul-specific keywords
        istanbul_keywords = [
            'hagia sophia', 'blue mosque', 'topkapi palace', 'grand bazaar', 'galata tower',
            'bosphorus', 'sultanahmet', 'taksim', 'beyoglu', 'kadikoy', 'uskudar',
            'museum', 'mosque', 'park', 'restaurant', 'hotel', 'shopping', 'ferry',
            'metro', 'tram', 'taxi', 'weather', 'food', 'turkish cuisine'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in istanbul_keywords if kw in text_lower]
        return ', '.join(found_keywords[:5])  # Top 5 keywords
    
    def _calculate_sentiment_score(self, feedback: Optional[FeedbackType]) -> float:
        """Calculate sentiment score based on feedback"""
        if feedback == FeedbackType.LIKE:
            return 0.9
        elif feedback == FeedbackType.DISLIKE:
            return 0.1
        elif feedback == FeedbackType.MIXED:
            return 0.5
        else:
            return 0.5  # Neutral for unknown
    
    def analyze_feedback_patterns(self, days_back: int = 30) -> FeedbackAnalysis:
        """Analyze feedback patterns over specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        since_date = datetime.now() - timedelta(days=days_back)
        
        # Overall metrics
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sessions,
                AVG(satisfaction_score) as avg_satisfaction,
                SUM(like_count) as total_likes,
                SUM(dislike_count) as total_dislikes,
                SUM(total_messages) as total_messages
            FROM chat_sessions_analysis 
            WHERE created_at >= ?
        ''', (since_date,))
        
        overall_stats = cursor.fetchone()
        total_sessions = overall_stats[0] or 0
        avg_satisfaction = overall_stats[1] or 0.5
        total_likes = overall_stats[2] or 0
        total_dislikes = overall_stats[3] or 0
        total_messages = overall_stats[4] or 0
        
        # Feedback distribution
        cursor.execute('''
            SELECT overall_feedback, COUNT(*) as count
            FROM chat_sessions_analysis 
            WHERE created_at >= ?
            GROUP BY overall_feedback
        ''', (since_date,))
        
        feedback_dist = dict(cursor.fetchall())
        
        # Calculate percentages
        if total_sessions > 0:
            like_percentage = (feedback_dist.get('like', 0) / total_sessions) * 100
            dislike_percentage = (feedback_dist.get('dislike', 0) / total_sessions) * 100
            mixed_percentage = (feedback_dist.get('mixed', 0) / total_sessions) * 100
        else:
            like_percentage = dislike_percentage = mixed_percentage = 0
        
        # Top liked topics
        cursor.execute('''
            SELECT topic_keywords, COUNT(*) as frequency
            FROM message_analysis 
            WHERE feedback = 'like' AND timestamp >= ? AND topic_keywords != ''
            GROUP BY topic_keywords
            ORDER BY frequency DESC
            LIMIT 10
        ''', (since_date,))
        
        top_liked_topics = [(row[0], row[1]) for row in cursor.fetchall()]
        
        # Top disliked topics
        cursor.execute('''
            SELECT topic_keywords, COUNT(*) as frequency
            FROM message_analysis 
            WHERE feedback = 'dislike' AND timestamp >= ? AND topic_keywords != ''
            GROUP BY topic_keywords
            ORDER BY frequency DESC
            LIMIT 10
        ''', (since_date,))
        
        top_disliked_topics = [(row[0], row[1]) for row in cursor.fetchall()]
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            avg_satisfaction, top_disliked_topics, like_percentage, dislike_percentage
        )
        
        # Temporal trends (last 7 days)
        temporal_trends = self._analyze_temporal_trends(cursor, 7)
        
        conn.close()
        
        analysis = FeedbackAnalysis(
            total_sessions=total_sessions,
            total_messages=total_messages,
            satisfaction_rate=avg_satisfaction,
            like_percentage=like_percentage,
            dislike_percentage=dislike_percentage,
            mixed_feedback_percentage=mixed_percentage,
            top_liked_topics=top_liked_topics,
            top_disliked_topics=top_disliked_topics,
            improvement_suggestions=improvement_suggestions,
            temporal_trends=temporal_trends
        )
        
        logger.info(f"📊 Analyzed feedback patterns: {total_sessions} sessions, {avg_satisfaction:.2f} satisfaction")
        return analysis
    
    def _generate_improvement_suggestions(self, satisfaction_rate: float, 
                                        disliked_topics: List[Tuple[str, int]],
                                        like_rate: float, dislike_rate: float) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Satisfaction-based suggestions
        if satisfaction_rate < 0.6:
            suggestions.append("🔴 CRITICAL: Overall satisfaction is low (<60%). Immediate content review needed.")
        elif satisfaction_rate < 0.7:
            suggestions.append("🟡 WARNING: Satisfaction rate is below 70%. Consider content improvements.")
        
        # Dislike rate suggestions
        if dislike_rate > 20:
            suggestions.append(f"👎 High dislike rate ({dislike_rate:.1f}%). Focus on response quality improvement.")
        
        # Topic-specific suggestions
        if disliked_topics:
            top_disliked = disliked_topics[0][0]  # Most disliked topic
            suggestions.append(f"📝 Most disliked topic: '{top_disliked}'. Review and improve content for this area.")
        
        # Engagement suggestions
        if like_rate < 30:
            suggestions.append("📈 Low engagement rate. Consider adding more interactive and helpful content.")
        
        # General improvements
        suggestions.extend([
            "🎯 Focus on Istanbul attraction details and practical information",
            "🗺️ Improve transportation and location guidance",
            "📱 Enhance real-time information (opening hours, prices, events)",
            "💡 Add more personalized recommendations based on user preferences"
        ])
        
        return suggestions[:10]  # Top 10 suggestions
    
    def _analyze_temporal_trends(self, cursor, days: int) -> Dict[str, Any]:
        """Analyze trends over time"""
        since_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as sessions,
                AVG(satisfaction_score) as avg_satisfaction,
                SUM(like_count) as likes,
                SUM(dislike_count) as dislikes
            FROM chat_sessions_analysis 
            WHERE created_at >= ?
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', (since_date,))
        
        daily_data = cursor.fetchall()
        
        if not daily_data:
            return {'trend': 'insufficient_data', 'daily_data': []}
        
        # Calculate trend
        satisfaction_scores = [row[2] for row in daily_data if row[2] is not None]
        if len(satisfaction_scores) >= 2:
            trend = 'improving' if satisfaction_scores[-1] > satisfaction_scores[0] else 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'daily_data': [
                {
                    'date': row[0],
                    'sessions': row[1],
                    'satisfaction': row[2],
                    'likes': row[3] or 0,
                    'dislikes': row[4] or 0
                }
                for row in daily_data
            ]
        }
    
    def get_admin_dashboard_integration_data(self) -> Dict[str, Any]:
        """Generate data for admin dashboard integration"""
        analysis = self.analyze_feedback_patterns()
        
        # Format for admin dashboard consumption
        dashboard_data = {
            'feedback_summary': {
                'total_sessions_with_feedback': analysis.total_sessions,
                'satisfaction_rate': round(analysis.satisfaction_rate * 100, 1),
                'like_percentage': round(analysis.like_percentage, 1),
                'dislike_percentage': round(analysis.dislike_percentage, 1),
                'mixed_percentage': round(analysis.mixed_feedback_percentage, 1)
            },
            'top_insights': {
                'most_liked_topics': analysis.top_liked_topics[:5],
                'most_disliked_topics': analysis.top_disliked_topics[:5],
                'improvement_priorities': analysis.improvement_suggestions[:5]
            },
            'performance_trend': {
                'trend_direction': analysis.temporal_trends.get('trend', 'stable'),
                'recent_data': analysis.temporal_trends.get('daily_data', [])[-7:]  # Last 7 days
            },
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info("🎯 Generated admin dashboard integration data")
        return dashboard_data
    
    def generate_weekly_report(self) -> Dict[str, Any]:
        """Generate comprehensive weekly feedback report"""
        analysis = self.analyze_feedback_patterns(days_back=7)
        
        report = {
            'report_period': '7 days',
            'summary': {
                'total_sessions': analysis.total_sessions,
                'total_messages': analysis.total_messages,
                'overall_satisfaction': f"{analysis.satisfaction_rate * 100:.1f}%",
                'feedback_distribution': {
                    'likes': f"{analysis.like_percentage:.1f}%",
                    'dislikes': f"{analysis.dislike_percentage:.1f}%",
                    'mixed': f"{analysis.mixed_feedback_percentage:.1f}%"
                }
            },
            'insights': {
                'top_performing_content': analysis.top_liked_topics,
                'improvement_areas': analysis.top_disliked_topics,
                'action_items': analysis.improvement_suggestions
            },
            'trend_analysis': analysis.temporal_trends,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info("📊 Generated weekly feedback report")
        return report

# Demo and testing functions
def demo_chat_session_analyzer():
    """Demonstrate the chat session analyzer capabilities"""
    print("🎯 ISTANBUL AI CHAT SESSION ANALYZER DEMO")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = ChatSessionAnalyzer()
    
    # Demo data (simulating admin dashboard format)
    demo_sessions = [
        {
            'id': 'session_001',
            'user_ip': '192.168.1.100',
            'title': 'Istanbul Attractions Help',
            'saved_at': datetime.now().isoformat(),
            'conversation_history': [
                {
                    'user': 'What are the best attractions in Istanbul?',
                    'ai': 'Istanbul has many amazing attractions! Here are the top recommendations: Hagia Sophia, Blue Mosque, Topkapi Palace, Grand Bazaar, and Galata Tower.',
                    'feedback': 'like',
                    'timestamp': datetime.now().isoformat()
                },
                {
                    'user': 'How do I get to Hagia Sophia?',
                    'ai': 'You can reach Hagia Sophia by taking the tram to Sultanahmet station. It\'s a short walk from there.',
                    'feedback': 'like',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        },
        {
            'id': 'session_002',
            'user_ip': '192.168.1.101',
            'title': 'Transportation Query',
            'saved_at': datetime.now().isoformat(),
            'conversation_history': [
                {
                    'user': 'How much does the metro cost?',
                    'ai': 'The Istanbul metro costs about 15 TL per ride.',
                    'feedback': 'dislike',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        }
    ]
    
    # Import demo sessions
    print("\n📥 IMPORTING DEMO SESSIONS:")
    imported_count = analyzer.import_chat_sessions_from_admin(demo_sessions)
    print(f"✅ Imported {imported_count} sessions")
    
    # Perform analysis
    print("\n📊 FEEDBACK PATTERN ANALYSIS:")
    analysis = analyzer.analyze_feedback_patterns()
    
    print(f"📈 Analysis Results:")
    print(f"  Total Sessions: {analysis.total_sessions}")
    print(f"  Total Messages: {analysis.total_messages}")
    print(f"  Satisfaction Rate: {analysis.satisfaction_rate * 100:.1f}%")
    print(f"  Like Rate: {analysis.like_percentage:.1f}%")
    print(f"  Dislike Rate: {analysis.dislike_percentage:.1f}%")
    
    # Show insights
    print(f"\n💡 TOP INSIGHTS:")
    if analysis.top_liked_topics:
        print("🟢 Most Liked Topics:")
        for topic, count in analysis.top_liked_topics[:3]:
            print(f"  • {topic} ({count} likes)")
    
    if analysis.top_disliked_topics:
        print("🔴 Most Disliked Topics:")
        for topic, count in analysis.top_disliked_topics[:3]:
            print(f"  • {topic} ({count} dislikes)")
    
    # Show improvement suggestions
    print(f"\n🎯 IMPROVEMENT SUGGESTIONS:")
    for i, suggestion in enumerate(analysis.improvement_suggestions[:5], 1):
        print(f"  {i}. {suggestion}")
    
    # Generate admin dashboard data
    print(f"\n🎛️ ADMIN DASHBOARD INTEGRATION:")
    dashboard_data = analyzer.get_admin_dashboard_integration_data()
    print(f"  Satisfaction Rate: {dashboard_data['feedback_summary']['satisfaction_rate']}%")
    print(f"  Trend Direction: {dashboard_data['performance_trend']['trend_direction']}")
    
    print(f"\n✅ Chat Session Analysis Demo Complete!")

if __name__ == "__main__":
    demo_chat_session_analyzer()
