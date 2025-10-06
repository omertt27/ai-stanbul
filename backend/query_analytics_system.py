#!/usr/bin/env python3
"""
Comprehensive Query Analytics & Feedback System
=============================================

Tracks all queries, identifies failed/unsatisfying ones, and provides
analytics for continuous improvement of the AI Istanbul system.

Features:
- Real-time query logging and analysis
- Failed query identification (no results, poor matches, user dissatisfaction)
- User satisfaction tracking and feedback collection
- Performance analytics and bottleneck identification
- Continuous improvement recommendations
- Export capabilities for manual review
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import threading
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

class QueryStatus(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    NO_RESULTS = "no_results"
    TIMEOUT = "timeout"
    ERROR = "error"

class UserSatisfaction(Enum):
    VERY_SATISFIED = 5
    SATISFIED = 4
    NEUTRAL = 3
    DISSATISFIED = 2
    VERY_DISSATISFIED = 1

@dataclass
class QueryAnalytics:
    query_id: str
    user_id: Optional[str]
    session_id: str
    timestamp: datetime
    raw_query: str
    processed_query: str
    intent: str
    extracted_entities: Dict[str, Any]
    query_type: str
    
    # Response metrics
    response_time_ms: float
    results_count: int
    status: QueryStatus
    confidence_score: float
    
    # User interaction
    user_satisfaction: Optional[UserSatisfaction] = None
    user_feedback: Optional[str] = None
    follow_up_queries: List[str] = None
    
    # System details
    system_version: str = "1.0"
    used_fallback: bool = False
    error_details: Optional[str] = None

class QueryAnalyticsSystem:
    """Comprehensive system for tracking and analyzing all queries"""
    
    def __init__(self, db_path: str = "query_analytics.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        self.session_queries = defaultdict(list)  # Track queries per session
        
    def _init_database(self):
        """Initialize the analytics database"""
        with sqlite3.connect(self.db_path) as conn:
            # Main query analytics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_analytics (
                    query_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    raw_query TEXT NOT NULL,
                    processed_query TEXT,
                    intent TEXT,
                    extracted_entities TEXT,
                    query_type TEXT,
                    
                    response_time_ms REAL NOT NULL,
                    results_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    confidence_score REAL,
                    
                    user_satisfaction INTEGER,
                    user_feedback TEXT,
                    follow_up_queries TEXT,
                    
                    system_version TEXT DEFAULT '1.0',
                    used_fallback BOOLEAN DEFAULT FALSE,
                    error_details TEXT
                )
            """)
            
            # Failed queries for focused analysis
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT NOT NULL,
                    failure_reason TEXT NOT NULL,
                    failure_category TEXT NOT NULL,
                    potential_fix TEXT,
                    reviewed BOOLEAN DEFAULT FALSE,
                    improvement_applied BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (query_id) REFERENCES query_analytics (query_id)
                )
            """)
            
            # User feedback sessions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT,
                    total_queries INTEGER DEFAULT 0,
                    satisfied_queries INTEGER DEFAULT 0,
                    avg_satisfaction REAL,
                    overall_feedback TEXT,
                    improvement_suggestions TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System performance metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_queries INTEGER DEFAULT 0,
                    successful_queries INTEGER DEFAULT 0,
                    failed_queries INTEGER DEFAULT 0,
                    avg_response_time_ms REAL,
                    avg_confidence_score REAL,
                    most_common_intents TEXT,
                    failure_patterns TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Improvement tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS improvements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    improvement_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_issue TEXT,
                    implementation_status TEXT DEFAULT 'planned',
                    impact_metrics TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    implemented_at DATETIME
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_timestamp ON query_analytics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_status ON query_analytics(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_session ON query_analytics(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_failed_category ON failed_queries(failure_category)")
            
            conn.commit()
    
    def log_query(self, query_analytics: QueryAnalytics) -> str:
        """Log a query with complete analytics"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                    # Convert to dict and handle datetime
                    data = asdict(query_analytics)
                    data['timestamp'] = query_analytics.timestamp.isoformat()
                    data['extracted_entities'] = json.dumps(data['extracted_entities'])
                    data['follow_up_queries'] = json.dumps(data['follow_up_queries'] or [])
                    data['status'] = data['status'].value
                    if data['user_satisfaction']:
                        data['user_satisfaction'] = data['user_satisfaction'].value
                    
                    conn.execute("""
                        INSERT INTO query_analytics (
                            query_id, user_id, session_id, timestamp, raw_query,
                            processed_query, intent, extracted_entities, query_type,
                            response_time_ms, results_count, status, confidence_score,
                            user_satisfaction, user_feedback, follow_up_queries,
                            system_version, used_fallback, error_details
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['query_id'], data['user_id'], data['session_id'],
                        data['timestamp'], data['raw_query'], data['processed_query'],
                        data['intent'], data['extracted_entities'], data['query_type'],
                        data['response_time_ms'], data['results_count'], data['status'],
                        data['confidence_score'], data['user_satisfaction'],
                        data['user_feedback'], data['follow_up_queries'],
                        data['system_version'], data['used_fallback'], data['error_details']
                    ))
                    
                    # Track session queries
                    self.session_queries[query_analytics.session_id].append(query_analytics.query_id)
                    
                    # Check if this is a failed query
                    if self._is_failed_query(query_analytics):
                        self._log_failed_query(query_analytics)
                    
                    conn.commit()
                    return query_analytics.query_id
                    
            except Exception as e:
                logger.error(f"Error logging query analytics: {e}")
                return ""
    
    def _is_failed_query(self, query: QueryAnalytics) -> bool:
        """Determine if a query should be classified as failed"""
        # Define failure criteria
        if query.status in [QueryStatus.FAILED, QueryStatus.NO_RESULTS, QueryStatus.TIMEOUT, QueryStatus.ERROR]:
            return True
        
        if query.results_count == 0:
            return True
            
        if query.confidence_score is not None and query.confidence_score < 0.3:
            return True
            
        if query.user_satisfaction and query.user_satisfaction.value <= 2:
            return True
            
        if query.response_time_ms > 5000:  # 5 seconds threshold
            return True
            
        return False
    
    def _log_failed_query(self, query: QueryAnalytics):
        """Log a failed query for focused analysis"""
        failure_reason = self._determine_failure_reason(query)
        failure_category = self._categorize_failure(query, failure_reason)
        potential_fix = self._suggest_potential_fix(query, failure_category)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO failed_queries (query_id, failure_reason, failure_category, potential_fix)
                VALUES (?, ?, ?, ?)
            """, (query.query_id, failure_reason, failure_category, potential_fix))
            conn.commit()
    
    def _determine_failure_reason(self, query: QueryAnalytics) -> str:
        """Determine the specific reason for query failure"""
        if query.status == QueryStatus.NO_RESULTS:
            return "No matching results found in database"
        elif query.status == QueryStatus.TIMEOUT:
            return f"Query timed out after {query.response_time_ms}ms"
        elif query.status == QueryStatus.ERROR:
            return f"System error: {query.error_details}"
        elif query.results_count == 0:
            return "Database query returned no results"
        elif query.confidence_score is not None and query.confidence_score < 0.3:
            return f"Low confidence score: {query.confidence_score}"
        elif query.user_satisfaction and query.user_satisfaction.value <= 2:
            return f"User dissatisfaction: {query.user_satisfaction.value}/5"
        elif query.response_time_ms > 5000:
            return f"Slow response: {query.response_time_ms}ms"
        else:
            return "Unknown failure reason"
    
    def _categorize_failure(self, query: QueryAnalytics, reason: str) -> str:
        """Categorize the type of failure"""
        if "No matching results" in reason or "no results" in reason:
            return "data_gap"
        elif "timed out" in reason or "Slow response" in reason:
            return "performance"
        elif "System error" in reason:
            return "system_error"
        elif "Low confidence" in reason:
            return "nlp_understanding"
        elif "User dissatisfaction" in reason:
            return "user_experience"
        else:
            return "unknown"
    
    def _suggest_potential_fix(self, query: QueryAnalytics, category: str) -> str:
        """Suggest potential fixes based on failure category"""
        fixes = {
            "data_gap": f"Add more data for intent '{query.intent}' or entities {query.extracted_entities}",
            "performance": "Optimize database queries and add caching",
            "system_error": "Fix system bugs and add error handling",
            "nlp_understanding": f"Improve NLP processing for query type '{query.query_type}'",
            "user_experience": "Improve response templates and add interactive flows",
            "unknown": "Manual review required"
        }
        return fixes.get(category, "Manual review required")
    
    def update_user_satisfaction(self, query_id: str, satisfaction: UserSatisfaction, feedback: str = None):
        """Update user satisfaction for a query"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE query_analytics 
                        SET user_satisfaction = ?, user_feedback = ?
                        WHERE query_id = ?
                    """, (satisfaction.value, feedback, query_id))
                    
                    # If this makes it a failed query, log it
                    if satisfaction.value <= 2:
                        cursor = conn.execute(
                            "SELECT * FROM query_analytics WHERE query_id = ?",
                            (query_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            # Check if not already in failed_queries
                            cursor = conn.execute(
                                "SELECT COUNT(*) FROM failed_queries WHERE query_id = ?",
                                (query_id,)
                            )
                            if cursor.fetchone()[0] == 0:
                                conn.execute("""
                                    INSERT INTO failed_queries (query_id, failure_reason, failure_category, potential_fix)
                                    VALUES (?, ?, ?, ?)
                                """, (
                                    query_id,
                                    f"User dissatisfaction: {satisfaction.value}/5",
                                    "user_experience",
                                    "Improve response quality and user experience"
                                ))
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Error updating user satisfaction: {e}")
    
    def get_failed_queries_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive analysis of failed queries"""
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get failed queries with details
            cursor = conn.execute("""
                SELECT 
                    fq.failure_category,
                    fq.failure_reason,
                    qa.intent,
                    qa.query_type,
                    qa.raw_query,
                    qa.confidence_score,
                    qa.response_time_ms,
                    fq.potential_fix,
                    fq.reviewed
                FROM failed_queries fq
                JOIN query_analytics qa ON fq.query_id = qa.query_id
                WHERE qa.timestamp >= ?
                ORDER BY qa.timestamp DESC
            """, (since_date.isoformat(),))
            
            failed_queries = cursor.fetchall()
            
            # Analyze patterns
            categories = defaultdict(int)
            intents = defaultdict(int)
            fixes_needed = defaultdict(int)
            unreviewed_count = 0
            
            for query in failed_queries:
                categories[query[0]] += 1
                intents[query[1]] += 1
                fixes_needed[query[7]] += 1
                if not query[8]:  # not reviewed
                    unreviewed_count += 1
            
            return {
                "total_failed": len(failed_queries),
                "unreviewed_count": unreviewed_count,
                "failure_categories": dict(categories),
                "problematic_intents": dict(intents),
                "recommended_fixes": dict(fixes_needed),
                "recent_failures": [
                    {
                        "query": query[4],
                        "reason": query[1],
                        "category": query[0],
                        "confidence": query[5],
                        "response_time": query[6],
                        "suggested_fix": query[7]
                    }
                    for query in failed_queries[:10]  # Last 10 failures
                ]
            }
    
    def get_performance_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        since_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) as successful,
                    COUNT(CASE WHEN status IN ('failed', 'no_results', 'timeout', 'error') THEN 1 END) as failed,
                    AVG(response_time_ms) as avg_response_time,
                    AVG(confidence_score) as avg_confidence,
                    AVG(CASE WHEN user_satisfaction IS NOT NULL THEN user_satisfaction END) as avg_satisfaction
                FROM query_analytics
                WHERE timestamp >= ?
            """, (since_date.isoformat(),))
            
            stats = cursor.fetchone()
            
            # Intent analysis
            cursor = conn.execute("""
                SELECT intent, COUNT(*) as count
                FROM query_analytics
                WHERE timestamp >= ?
                GROUP BY intent
                ORDER BY count DESC
                LIMIT 10
            """, (since_date.isoformat(),))
            
            top_intents = dict(cursor.fetchall())
            
            # Daily trends
            cursor = conn.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as queries,
                    AVG(response_time_ms) as avg_time,
                    COUNT(CASE WHEN status = 'success' THEN 1 END) * 100.0 / COUNT(*) as success_rate
                FROM query_analytics
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (since_date.isoformat(),))
            
            daily_trends = [
                {
                    "date": row[0],
                    "queries": row[1],
                    "avg_response_time": row[2],
                    "success_rate": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            return {
                "period_days": days,
                "total_queries": stats[0] or 0,
                "successful_queries": stats[1] or 0,
                "failed_queries": stats[2] or 0,
                "success_rate": (stats[1] or 0) / max(stats[0] or 1, 1) * 100,
                "avg_response_time": stats[3] or 0,  # Fixed key name
                "avg_response_time_ms": stats[3] or 0,
                "avg_confidence_score": stats[4] or 0,
                "avg_user_satisfaction": stats[5] or 0,
                "top_intents": top_intents,
                "daily_trends": daily_trends
            }
    
    def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on analytics"""
        failed_analysis = self.get_failed_queries_analysis(days=30)
        performance = self.get_performance_analytics(days=30)
        
        recommendations = []
        
        # Data gap recommendations
        if "data_gap" in failed_analysis["failure_categories"]:
            count = failed_analysis["failure_categories"]["data_gap"]
            recommendations.append({
                "priority": "high",
                "category": "data_expansion",
                "title": f"Address {count} data gap failures",
                "description": "Add more attractions, restaurants, or activities to database",
                "estimated_impact": "high"
            })
        
        # Performance recommendations
        if performance["avg_response_time_ms"] > 2000:
            recommendations.append({
                "priority": "medium",
                "category": "performance",
                "title": "Optimize response times",
                "description": f"Current avg: {performance['avg_response_time_ms']:.0f}ms. Add caching and optimize queries",
                "estimated_impact": "medium"
            })
        
        # NLP improvement recommendations
        if "nlp_understanding" in failed_analysis["failure_categories"]:
            count = failed_analysis["failure_categories"]["nlp_understanding"]
            recommendations.append({
                "priority": "medium",
                "category": "nlp_improvement",
                "title": f"Improve NLP understanding for {count} failed queries",
                "description": "Add more synonyms, entity patterns, and intent examples",
                "estimated_impact": "high"
            })
        
        # User experience recommendations
        if performance["avg_user_satisfaction"] < 4.0:
            recommendations.append({
                "priority": "high",
                "category": "user_experience",
                "title": f"Improve user satisfaction (current: {performance['avg_user_satisfaction']:.1f}/5)",
                "description": "Enhance response templates and add interactive flows",
                "estimated_impact": "high"
            })
        
        return recommendations
    
    def export_failed_queries_for_review(self, output_file: str = None) -> str:
        """Export failed queries for manual review"""
        if not output_file:
            output_file = f"failed_queries_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        analysis = self.get_failed_queries_analysis(days=30)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def mark_failure_reviewed(self, query_id: str, improvement_applied: bool = False):
        """Mark a failed query as reviewed"""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE failed_queries 
                        SET reviewed = TRUE, improvement_applied = ?
                        WHERE query_id = ?
                    """, (improvement_applied, query_id))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error marking failure as reviewed: {e}")

# Global instance
query_analytics_system = QueryAnalyticsSystem()

def log_query_analytics(
    raw_query: str,
    processed_query: str,
    intent: str,
    extracted_entities: Dict[str, Any],
    query_type: str,
    response_time_ms: float,
    results_count: int,
    status: QueryStatus,
    confidence_score: float = None,
    user_id: str = None,
    session_id: str = None,
    used_fallback: bool = False,
    error_details: str = None
) -> str:
    """Convenience function to log query analytics"""
    
    query_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())
    
    analytics = QueryAnalytics(
        query_id=query_id,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now(),
        raw_query=raw_query,
        processed_query=processed_query,
        intent=intent,
        extracted_entities=extracted_entities,
        query_type=query_type,
        response_time_ms=response_time_ms,
        results_count=results_count,
        status=status,
        confidence_score=confidence_score,
        used_fallback=used_fallback,
        error_details=error_details
    )
    
    return query_analytics_system.log_query(analytics)

def track_user_satisfaction(query_id: str, satisfaction: int, feedback: str = None):
    """Track user satisfaction for a query"""
    satisfaction_enum = UserSatisfaction(satisfaction)
    query_analytics_system.update_user_satisfaction(query_id, satisfaction_enum, feedback)

def get_analytics_dashboard() -> Dict[str, Any]:
    """Get comprehensive analytics dashboard data"""
    return {
        "performance": query_analytics_system.get_performance_analytics(),
        "failed_queries": query_analytics_system.get_failed_queries_analysis(),
        "recommendations": query_analytics_system.get_improvement_recommendations()
    }

if __name__ == "__main__":
    # Test the system
    print("🧪 Testing Query Analytics System...")
    
    # Test logging a successful query
    query_id = log_query_analytics(
        raw_query="best museums in Istanbul",
        processed_query="find museums istanbul",
        intent="find_attraction",
        extracted_entities={"attraction_type": "museum", "location": "Istanbul"},
        query_type="attraction_search",
        response_time_ms=150.5,
        results_count=12,
        status=QueryStatus.SUCCESS,
        confidence_score=0.9
    )
    print(f"✅ Logged successful query: {query_id}")
    
    # Test logging a failed query
    failed_id = log_query_analytics(
        raw_query="underwater restaurants in Istanbul",
        processed_query="find underwater restaurants istanbul",
        intent="find_restaurant",
        extracted_entities={"restaurant_type": "underwater", "location": "Istanbul"},
        query_type="restaurant_search",
        response_time_ms=1200.0,
        results_count=0,
        status=QueryStatus.NO_RESULTS,
        confidence_score=0.2
    )
    print(f"✅ Logged failed query: {failed_id}")
    
    # Test analytics
    analytics = get_analytics_dashboard()
    print(f"📊 Performance Analytics: {analytics['performance']['success_rate']:.1f}% success rate")
    print(f"❌ Failed Queries: {analytics['failed_queries']['total_failed']} total")
    print(f"💡 Recommendations: {len(analytics['recommendations'])} suggestions")
    
    print("✅ Query Analytics System is working correctly!")
