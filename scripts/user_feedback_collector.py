#!/usr/bin/env python3
"""
User Feedback Collection System
Collects and analyzes user feedback on ML predictions
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UserFeedbackCollector:
    """
    Collect and analyze user feedback on AI responses
    
    Feedback types:
    - Thumbs up/down on responses
    - Intent correction
    - Response quality rating
    - Feature requests
    - Bug reports
    """
    
    def __init__(
        self,
        feedback_dir: str = "data/user_feedback",
        feedback_file: str = "data/user_feedback/feedback.jsonl"
    ):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ User Feedback Collector initialized")
    
    def collect_feedback(
        self,
        query: str,
        response: str,
        predicted_intent: str,
        feedback_type: str,
        feedback_value: any,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Collect user feedback
        
        Args:
            query: User's original query
            response: AI's response
            predicted_intent: Intent classifier's prediction
            feedback_type: Type of feedback ('rating', 'intent_correction', 'comment')
            feedback_value: Feedback value (rating number, corrected intent, comment text)
            user_id: Anonymous user identifier
            session_id: Session identifier
            metadata: Additional metadata (confidence, latency, etc.)
        
        Returns:
            Feedback ID
        """
        feedback_id = f"fb_{int(datetime.now().timestamp() * 1000)}"
        
        feedback_entry = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response[:500],  # Truncate long responses
            'predicted_intent': predicted_intent,
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'user_id': user_id,
            'session_id': session_id,
            'metadata': metadata or {}
        }
        
        # Write to file
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        
        logger.info(f"📝 Feedback collected: {feedback_id} ({feedback_type})")
        
        return feedback_id
    
    def collect_rating(
        self,
        query: str,
        response: str,
        predicted_intent: str,
        rating: int,  # 1-5 stars
        **kwargs
    ) -> str:
        """Collect star rating feedback"""
        return self.collect_feedback(
            query=query,
            response=response,
            predicted_intent=predicted_intent,
            feedback_type='rating',
            feedback_value=rating,
            **kwargs
        )
    
    def collect_thumbs(
        self,
        query: str,
        response: str,
        predicted_intent: str,
        thumbs_up: bool,
        **kwargs
    ) -> str:
        """Collect thumbs up/down feedback"""
        return self.collect_feedback(
            query=query,
            response=response,
            predicted_intent=predicted_intent,
            feedback_type='thumbs',
            feedback_value='up' if thumbs_up else 'down',
            **kwargs
        )
    
    def collect_intent_correction(
        self,
        query: str,
        response: str,
        predicted_intent: str,
        correct_intent: str,
        **kwargs
    ) -> str:
        """Collect intent correction from user"""
        return self.collect_feedback(
            query=query,
            response=response,
            predicted_intent=predicted_intent,
            feedback_type='intent_correction',
            feedback_value={
                'predicted': predicted_intent,
                'correct': correct_intent
            },
            **kwargs
        )
    
    def collect_comment(
        self,
        query: str,
        response: str,
        predicted_intent: str,
        comment: str,
        **kwargs
    ) -> str:
        """Collect free-text comment"""
        return self.collect_feedback(
            query=query,
            response=response,
            predicted_intent=predicted_intent,
            feedback_type='comment',
            feedback_value=comment,
            **kwargs
        )
    
    def analyze_feedback(self, days: int = 7) -> Dict:
        """
        Analyze feedback from last N days
        
        Returns comprehensive feedback analysis
        """
        if not self.feedback_file.exists():
            return {'status': 'no_feedback', 'message': 'No feedback collected yet'}
        
        # Load feedback
        feedback_entries = []
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    feedback_entries.append(entry)
                except:
                    continue
        
        if not feedback_entries:
            return {'status': 'no_feedback', 'message': 'No feedback collected yet'}
        
        # Filter by date
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_feedback = [
            f for f in feedback_entries
            if datetime.fromisoformat(f['timestamp']).timestamp() >= cutoff_date
        ]
        
        # Analyze ratings
        ratings = [
            f['feedback_value'] for f in recent_feedback
            if f['feedback_type'] == 'rating' and isinstance(f['feedback_value'], (int, float))
        ]
        
        # Analyze thumbs
        thumbs_up = sum(
            1 for f in recent_feedback
            if f['feedback_type'] == 'thumbs' and f['feedback_value'] == 'up'
        )
        thumbs_down = sum(
            1 for f in recent_feedback
            if f['feedback_type'] == 'thumbs' and f['feedback_value'] == 'down'
        )
        
        # Analyze intent corrections
        intent_corrections = [
            f for f in recent_feedback
            if f['feedback_type'] == 'intent_correction'
        ]
        
        # Intent accuracy by predicted intent
        intent_accuracy = {}
        for f in intent_corrections:
            predicted = f['predicted_intent']
            if predicted not in intent_accuracy:
                intent_accuracy[predicted] = {'total': 0, 'correct': 0}
            intent_accuracy[predicted]['total'] += 1
            if f['feedback_value']['predicted'] != f['feedback_value']['correct']:
                # Misclassification
                pass
            else:
                intent_accuracy[predicted]['correct'] += 1
        
        # Common issues
        negative_feedback = [
            f for f in recent_feedback
            if (f['feedback_type'] == 'rating' and f['feedback_value'] <= 2) or
               (f['feedback_type'] == 'thumbs' and f['feedback_value'] == 'down')
        ]
        
        negative_intents = Counter(f['predicted_intent'] for f in negative_feedback)
        
        return {
            'status': 'success',
            'period_days': days,
            'total_feedback': len(recent_feedback),
            'ratings': {
                'count': len(ratings),
                'average': sum(ratings) / len(ratings) if ratings else 0,
                'distribution': Counter(ratings)
            },
            'thumbs': {
                'up': thumbs_up,
                'down': thumbs_down,
                'satisfaction_rate': thumbs_up / (thumbs_up + thumbs_down) if (thumbs_up + thumbs_down) > 0 else 0
            },
            'intent_corrections': {
                'count': len(intent_corrections),
                'corrections_by_intent': intent_accuracy,
                'most_corrected': negative_intents.most_common(5)
            },
            'negative_feedback': {
                'count': len(negative_feedback),
                'top_problematic_intents': negative_intents.most_common(5)
            }
        }
    
    def generate_feedback_report(self, days: int = 7, output_file: Optional[str] = None) -> str:
        """Generate user feedback report"""
        analysis = self.analyze_feedback(days)
        
        if analysis['status'] == 'no_feedback':
            return "No user feedback collected yet."
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║     USER FEEDBACK ANALYSIS REPORT                            ║
╚══════════════════════════════════════════════════════════════╝

📅 PERIOD: Last {days} days
📊 Total Feedback: {analysis['total_feedback']}

⭐ RATINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Ratings:       {analysis['ratings']['count']}
Average Rating:      {analysis['ratings']['average']:.2f} / 5.0
"""
        
        if analysis['ratings']['distribution']:
            report += "\nRating Distribution:\n"
            for rating in sorted(analysis['ratings']['distribution'].keys(), reverse=True):
                count = analysis['ratings']['distribution'][rating]
                bar = '★' * rating + '☆' * (5 - rating)
                report += f"  {bar} ({rating}) - {count} responses\n"
        
        report += f"""
👍 THUMBS UP/DOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
👍 Thumbs Up:        {analysis['thumbs']['up']}
👎 Thumbs Down:      {analysis['thumbs']['down']}
Satisfaction Rate:   {analysis['thumbs']['satisfaction_rate']:.1%}

🎯 INTENT CORRECTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Corrections:   {analysis['intent_corrections']['count']}
"""
        
        if analysis['intent_corrections']['most_corrected']:
            report += "\nMost Frequently Corrected Intents:\n"
            for intent, count in analysis['intent_corrections']['most_corrected']:
                report += f"  • {intent:<30} {count} corrections\n"
        
        report += f"""
⚠️ PROBLEMATIC AREAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Negative Feedback:   {analysis['negative_feedback']['count']}
"""
        
        if analysis['negative_feedback']['top_problematic_intents']:
            report += "\nIntents with Most Negative Feedback:\n"
            for intent, count in analysis['negative_feedback']['top_problematic_intents']:
                report += f"  • {intent:<30} {count} negative responses\n"
        
        report += f"""
💡 RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Generate recommendations
        if analysis['thumbs']['satisfaction_rate'] < 0.7:
            report += "⚠️ Satisfaction rate is low (<70%). Review and improve responses.\n"
        
        if analysis['ratings']['average'] < 3.5:
            report += "⚠️ Average rating is low (<3.5/5). Focus on response quality.\n"
        
        if analysis['intent_corrections']['count'] > analysis['total_feedback'] * 0.1:
            report += "⚠️ High intent correction rate (>10%). Retrain intent classifier.\n"
        
        if len(analysis['negative_feedback']['top_problematic_intents']) > 0:
            problematic_intent = analysis['negative_feedback']['top_problematic_intents'][0][0]
            report += f"⚠️ Focus on improving: {problematic_intent}\n"
        
        if analysis['thumbs']['satisfaction_rate'] > 0.85 and analysis['ratings']['average'] > 4.0:
            report += "✅ Excellent performance! Users are satisfied.\n"
        
        report += "\n" + "═" * 60 + "\n"
        report += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📄 Feedback report saved to: {output_file}")
        
        return report
    
    def get_training_data_from_feedback(self, output_file: str = "data/feedback_training_data.json"):
        """
        Extract training data from user feedback
        
        Converts intent corrections and high-confidence positive feedback
        into training examples
        """
        if not self.feedback_file.exists():
            logger.warning("No feedback file found")
            return
        
        # Load feedback
        feedback_entries = []
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    feedback_entries.append(entry)
                except:
                    continue
        
        training_data = []
        
        # Extract from intent corrections
        for f in feedback_entries:
            if f['feedback_type'] == 'intent_correction':
                training_data.append({
                    'text': f['query'],
                    'intent': f['feedback_value']['correct'],
                    'source': 'user_correction',
                    'confidence': 1.0  # User-verified
                })
        
        # Extract from highly-rated responses
        for f in feedback_entries:
            if f['feedback_type'] == 'rating' and f['feedback_value'] >= 4:
                training_data.append({
                    'text': f['query'],
                    'intent': f['predicted_intent'],
                    'source': 'positive_feedback',
                    'confidence': 0.9  # High confidence from positive rating
                })
            elif f['feedback_type'] == 'thumbs' and f['feedback_value'] == 'up':
                training_data.append({
                    'text': f['query'],
                    'intent': f['predicted_intent'],
                    'source': 'thumbs_up',
                    'confidence': 0.85  # Good confidence from thumbs up
                })
        
        # Remove duplicates
        unique_data = []
        seen_queries = set()
        for item in training_data:
            if item['text'] not in seen_queries:
                unique_data.append(item)
                seen_queries.add(item['text'])
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Extracted {len(unique_data)} training examples from feedback: {output_file}")
        
        return unique_data


# Singleton instance
_feedback_collector_instance = None

def get_feedback_collector(**kwargs) -> UserFeedbackCollector:
    """Get or create singleton feedback collector"""
    global _feedback_collector_instance
    if _feedback_collector_instance is None:
        _feedback_collector_instance = UserFeedbackCollector(**kwargs)
    return _feedback_collector_instance


def main():
    """Demo usage"""
    collector = get_feedback_collector()
    
    print("📝 Simulating user feedback...\n")
    
    # Simulate various feedback types
    collector.collect_rating(
        query="Best kebab restaurants in Kadıköy",
        response="Here are the top kebab restaurants in Kadıköy...",
        predicted_intent="restaurant_search",
        rating=5
    )
    
    collector.collect_thumbs(
        query="How to get to Topkapı Palace",
        response="You can take the T1 tram from Sultanahmet...",
        predicted_intent="transport_route",
        thumbs_up=True
    )
    
    collector.collect_intent_correction(
        query="What's the weather like",
        response="The current weather in Istanbul is...",
        predicted_intent="daily_help",
        correct_intent="weather_query"
    )
    
    collector.collect_thumbs(
        query="Show me attractions",
        response="I don't understand...",
        predicted_intent="unknown",
        thumbs_up=False
    )
    
    # Generate report
    print(collector.generate_feedback_report(days=30))
    
    # Extract training data
    print("\n📤 Extracting training data from feedback...")
    training_data = collector.get_training_data_from_feedback()
    print(f"✅ Extracted {len(training_data)} examples\n")


if __name__ == "__main__":
    main()
