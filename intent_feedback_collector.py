#!/usr/bin/env python3
"""
Intent Feedback Collection System
Collects user corrections and feedback for intent classification retraining
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IntentFeedback:
    """Feedback entry for intent classification"""
    query: str
    predicted_intent: str
    predicted_confidence: float
    correct_intent: Optional[str]
    user_feedback: str  # 'correct', 'incorrect', 'ambiguous'
    language: str  # 'tr', 'en', 'unknown'
    timestamp: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None


class IntentFeedbackCollector:
    """
    Collects and stores user feedback for intent classification
    
    Features:
    - SQLite database for persistent storage
    - JSON export for retraining
    - Analytics and reporting
    - Automatic language detection
    """
    
    def __init__(self, db_path: str = "data/intent_feedback.db"):
        """
        Initialize feedback collector
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"✅ Intent Feedback Collector initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                predicted_intent TEXT NOT NULL,
                predicted_confidence REAL NOT NULL,
                correct_intent TEXT,
                user_feedback TEXT NOT NULL,
                language TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                user_id TEXT,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predicted_intent ON feedback(predicted_intent)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_feedback ON feedback(user_feedback)
        """)
        
        conn.commit()
        conn.close()
    
    def add_feedback(
        self,
        query: str,
        predicted_intent: str,
        predicted_confidence: float,
        user_feedback: str,
        correct_intent: Optional[str] = None,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add feedback entry
        
        Args:
            query: User query
            predicted_intent: Intent predicted by model
            predicted_confidence: Prediction confidence (0-1)
            user_feedback: 'correct', 'incorrect', 'ambiguous'
            correct_intent: Correct intent (if incorrect)
            language: Query language ('tr', 'en', 'unknown')
            session_id: Session identifier
            user_id: User identifier
            metadata: Additional metadata
            
        Returns:
            Feedback entry ID
        """
        # Auto-detect language if not provided
        if language is None:
            language = self._detect_language(query)
        
        feedback = IntentFeedback(
            query=query,
            predicted_intent=predicted_intent,
            predicted_confidence=predicted_confidence,
            correct_intent=correct_intent,
            user_feedback=user_feedback,
            language=language,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback (
                query, predicted_intent, predicted_confidence,
                correct_intent, user_feedback, language,
                timestamp, session_id, user_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.query,
            feedback.predicted_intent,
            feedback.predicted_confidence,
            feedback.correct_intent,
            feedback.user_feedback,
            feedback.language,
            feedback.timestamp,
            feedback.session_id,
            feedback.user_id,
            json.dumps(feedback.metadata) if feedback.metadata else None
        ))
        
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"📝 Feedback added (ID: {feedback_id}): {user_feedback} for '{query}' -> {predicted_intent}")
        
        return feedback_id
    
    def _detect_language(self, text: str) -> str:
        """Detect language from text"""
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        if any(char in turkish_chars for char in text):
            return 'tr'
        
        # Simple heuristic: check for common English/Turkish words
        turkish_words = {'bir', 'nerede', 'nasıl', 'için', 'var', 'mı', 'mi', 'mu', 'mü'}
        english_words = {'the', 'is', 'are', 'where', 'how', 'what', 'can', 'do'}
        
        words = set(text.lower().split())
        
        if words & turkish_words:
            return 'tr'
        elif words & english_words:
            return 'en'
        
        return 'unknown'
    
    def get_corrections(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all corrections (incorrect predictions with correct intent)
        
        Args:
            limit: Maximum number of corrections to return
            
        Returns:
            List of correction entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT id, query, predicted_intent, predicted_confidence,
                   correct_intent, language, timestamp
            FROM feedback
            WHERE user_feedback = 'incorrect' AND correct_intent IS NOT NULL
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        corrections = []
        for row in rows:
            corrections.append({
                'id': row[0],
                'query': row[1],
                'predicted_intent': row[2],
                'predicted_confidence': row[3],
                'correct_intent': row[4],
                'language': row[5],
                'timestamp': row[6]
            })
        
        return corrections
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total feedback
        cursor.execute("SELECT COUNT(*) FROM feedback")
        total = cursor.fetchone()[0]
        
        # Feedback by type
        cursor.execute("""
            SELECT user_feedback, COUNT(*) 
            FROM feedback 
            GROUP BY user_feedback
        """)
        by_type = dict(cursor.fetchall())
        
        # By language
        cursor.execute("""
            SELECT language, COUNT(*) 
            FROM feedback 
            GROUP BY language
        """)
        by_language = dict(cursor.fetchall())
        
        # Most corrected intents
        cursor.execute("""
            SELECT predicted_intent, COUNT(*) as count
            FROM feedback
            WHERE user_feedback = 'incorrect'
            GROUP BY predicted_intent
            ORDER BY count DESC
            LIMIT 10
        """)
        most_corrected = [
            {'intent': row[0], 'count': row[1]}
            for row in cursor.fetchall()
        ]
        
        # Accuracy estimate (correct / total)
        correct_count = by_type.get('correct', 0)
        accuracy = (correct_count / total * 100) if total > 0 else 0
        
        conn.close()
        
        return {
            'total_feedback': total,
            'by_type': by_type,
            'by_language': by_language,
            'most_corrected_intents': most_corrected,
            'estimated_accuracy': accuracy
        }
    
    def export_training_data(self, output_file: str = "data/feedback_training_data.json"):
        """
        Export corrections as training data
        
        Args:
            output_file: Output JSON file path
        """
        corrections = self.get_corrections()
        
        # Format as training data
        training_data = []
        for correction in corrections:
            training_data.append({
                'text': correction['query'],
                'intent': correction['correct_intent'],
                'language': correction['language'],
                'source': 'user_feedback',
                'timestamp': correction['timestamp']
            })
        
        # Load existing training data if available
        output_path = Path(output_file)
        existing_data = []
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'training_data' in data:
                    existing_data = data['training_data']
                elif isinstance(data, list):
                    existing_data = data
        
        # Merge and deduplicate
        existing_texts = {item['text'].lower() for item in existing_data}
        new_data = [item for item in training_data if item['text'].lower() not in existing_texts]
        
        merged_data = existing_data + new_data
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'training_data': merged_data,
                'metadata': {
                    'total_examples': len(merged_data),
                    'from_feedback': len(new_data),
                    'exported_at': datetime.now().isoformat()
                }
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Exported {len(new_data)} new feedback corrections to: {output_file}")
        logger.info(f"   Total training examples: {len(merged_data)}")
        
        return len(new_data)
    
    def generate_report(self) -> str:
        """Generate feedback report"""
        stats = self.get_statistics()
        corrections = self.get_corrections(limit=10)
        
        report = []
        report.append("=" * 80)
        report.append("INTENT FEEDBACK REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("📊 Summary:")
        report.append(f"  Total Feedback: {stats['total_feedback']}")
        report.append(f"  Estimated Accuracy: {stats['estimated_accuracy']:.1f}%")
        report.append("")
        
        # By type
        report.append("📈 Feedback by Type:")
        for feedback_type, count in stats['by_type'].items():
            percentage = (count / stats['total_feedback'] * 100) if stats['total_feedback'] > 0 else 0
            report.append(f"  {feedback_type}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # By language
        report.append("🌍 Feedback by Language:")
        for lang, count in stats['by_language'].items():
            percentage = (count / stats['total_feedback'] * 100) if stats['total_feedback'] > 0 else 0
            report.append(f"  {lang}: {count} ({percentage:.1f}%)")
        report.append("")
        
        # Most corrected
        report.append("⚠️ Most Corrected Intents (Top 10):")
        for item in stats['most_corrected_intents']:
            report.append(f"  {item['intent']}: {item['count']} corrections")
        report.append("")
        
        # Recent corrections
        report.append("🔍 Recent Corrections (Last 10):")
        for correction in corrections[:10]:
            report.append(f"  '{correction['query']}'")
            report.append(f"    Predicted: {correction['predicted_intent']} ({correction['predicted_confidence']:.1%})")
            report.append(f"    Correct: {correction['correct_intent']}")
            report.append(f"    Language: {correction['language']}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


# Singleton instance
_collector_instance = None


def get_feedback_collector() -> IntentFeedbackCollector:
    """Get singleton feedback collector instance"""
    global _collector_instance
    
    if _collector_instance is None:
        _collector_instance = IntentFeedbackCollector()
    
    return _collector_instance


def test_feedback_collector():
    """Test feedback collection system"""
    print("=" * 80)
    print("INTENT FEEDBACK COLLECTOR TEST")
    print("=" * 80)
    print()
    
    collector = get_feedback_collector()
    
    # Add some test feedback
    test_cases = [
        # Correct predictions
        {
            'query': "Ayasofya'yı görmek istiyorum",
            'predicted_intent': 'attraction',
            'predicted_confidence': 0.98,
            'user_feedback': 'correct',
            'language': 'tr'
        },
        {
            'query': "Where is the nearest restaurant?",
            'predicted_intent': 'restaurant',
            'predicted_confidence': 0.95,
            'user_feedback': 'correct',
            'language': 'en'
        },
        # Incorrect predictions with corrections
        {
            'query': "I want to book a hotel",
            'predicted_intent': 'restaurant',
            'predicted_confidence': 0.65,
            'user_feedback': 'incorrect',
            'correct_intent': 'accommodation',
            'language': 'en'
        },
        {
            'query': "How's the weather tomorrow?",
            'predicted_intent': 'general_info',
            'predicted_confidence': 0.55,
            'user_feedback': 'incorrect',
            'correct_intent': 'weather',
            'language': 'en'
        },
        {
            'query': "Müze tavsiyesi",
            'predicted_intent': 'shopping',
            'predicted_confidence': 0.60,
            'user_feedback': 'incorrect',
            'correct_intent': 'museum',
            'language': 'tr'
        }
    ]
    
    print("📝 Adding test feedback...")
    for case in test_cases:
        feedback_id = collector.add_feedback(**case)
        print(f"  Added feedback ID: {feedback_id}")
    print()
    
    # Get statistics
    print("📊 Statistics:")
    stats = collector.get_statistics()
    print(f"  Total feedback: {stats['total_feedback']}")
    print(f"  Estimated accuracy: {stats['estimated_accuracy']:.1f}%")
    print()
    
    # Get corrections
    print("🔍 Corrections:")
    corrections = collector.get_corrections()
    for correction in corrections:
        print(f"  '{correction['query']}'")
        print(f"    {correction['predicted_intent']} → {correction['correct_intent']}")
    print()
    
    # Export training data
    print("💾 Exporting training data...")
    count = collector.export_training_data()
    print(f"  Exported {count} new examples")
    print()
    
    # Generate report
    print(collector.generate_report())


if __name__ == "__main__":
    test_feedback_collector()
