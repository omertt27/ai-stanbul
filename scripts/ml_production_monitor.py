#!/usr/bin/env python3
"""
ML Production Performance Monitor
Tracks model accuracy, latency, and usage patterns in real-time
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLProductionMonitor:
    """
    Real-time production ML monitoring system
    
    Tracks:
    - Prediction accuracy and confidence
    - Inference latency
    - Intent distribution
    - Error patterns
    - User feedback
    """
    
    def __init__(
        self,
        log_dir: str = "logs/ml_production",
        metrics_file: str = "logs/ml_production/metrics.jsonl"
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = Path(metrics_file)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics (reset daily)
        self.reset_metrics()
        
        logger.info("✅ ML Production Monitor initialized")
    
    def reset_metrics(self):
        """Reset daily metrics"""
        self.metrics = {
            'total_predictions': 0,
            'predictions_by_intent': defaultdict(int),
            'predictions_by_confidence': defaultdict(int),
            'latencies': [],
            'errors': [],
            'user_feedback': [],
            'low_confidence_queries': [],
            'misclassifications': [],
            'start_time': datetime.now().isoformat()
        }
    
    def log_prediction(
        self,
        query: str,
        predicted_intent: str,
        confidence: float,
        latency_ms: float,
        actual_intent: Optional[str] = None,
        user_feedback: Optional[str] = None
    ):
        """
        Log a prediction event
        
        Args:
            query: User query text
            predicted_intent: Model's predicted intent
            confidence: Prediction confidence (0-1)
            latency_ms: Inference time in milliseconds
            actual_intent: Ground truth intent (if known)
            user_feedback: User feedback ('correct', 'wrong', 'partial')
        """
        # Update in-memory metrics
        self.metrics['total_predictions'] += 1
        self.metrics['predictions_by_intent'][predicted_intent] += 1
        
        # Confidence buckets
        confidence_bucket = self._get_confidence_bucket(confidence)
        self.metrics['predictions_by_confidence'][confidence_bucket] += 1
        
        # Track latency
        self.metrics['latencies'].append(latency_ms)
        
        # Low confidence tracking
        if confidence < 0.7:
            self.metrics['low_confidence_queries'].append({
                'query': query,
                'predicted_intent': predicted_intent,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        
        # Misclassification tracking
        if actual_intent and actual_intent != predicted_intent:
            self.metrics['misclassifications'].append({
                'query': query,
                'predicted': predicted_intent,
                'actual': actual_intent,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
        
        # User feedback
        if user_feedback:
            self.metrics['user_feedback'].append({
                'query': query,
                'predicted_intent': predicted_intent,
                'confidence': confidence,
                'feedback': user_feedback,
                'timestamp': datetime.now().isoformat()
            })
        
        # Write to log file
        self._write_to_log({
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'actual_intent': actual_intent,
            'user_feedback': user_feedback
        })
    
    def log_error(
        self,
        error_type: str,
        query: str,
        error_message: str
    ):
        """Log ML system error"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'query': query,
            'error_message': error_message
        }
        
        self.metrics['errors'].append(error_entry)
        
        logger.error(f"ML Error: {error_type} - {error_message}")
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics summary"""
        if self.metrics['total_predictions'] == 0:
            return {
                'status': 'No predictions yet',
                'total_predictions': 0
            }
        
        # Calculate statistics
        avg_latency = sum(self.metrics['latencies']) / len(self.metrics['latencies']) if self.metrics['latencies'] else 0
        p95_latency = self._percentile(self.metrics['latencies'], 95) if self.metrics['latencies'] else 0
        p99_latency = self._percentile(self.metrics['latencies'], 99) if self.metrics['latencies'] else 0
        
        # Accuracy from feedback
        feedback_count = len(self.metrics['user_feedback'])
        correct_predictions = sum(
            1 for f in self.metrics['user_feedback']
            if f['feedback'] == 'correct'
        )
        accuracy = correct_predictions / feedback_count if feedback_count > 0 else None
        
        # Top intents
        top_intents = Counter(self.metrics['predictions_by_intent']).most_common(5)
        
        return {
            'status': 'active',
            'period_start': self.metrics['start_time'],
            'total_predictions': self.metrics['total_predictions'],
            'performance': {
                'avg_latency_ms': round(avg_latency, 2),
                'p95_latency_ms': round(p95_latency, 2),
                'p99_latency_ms': round(p99_latency, 2),
                'accuracy': round(accuracy * 100, 2) if accuracy else 'N/A'
            },
            'top_intents': [
                {'intent': intent, 'count': count, 'percentage': round(count / self.metrics['total_predictions'] * 100, 1)}
                for intent, count in top_intents
            ],
            'confidence_distribution': dict(self.metrics['predictions_by_confidence']),
            'quality_metrics': {
                'low_confidence_count': len(self.metrics['low_confidence_queries']),
                'low_confidence_rate': round(len(self.metrics['low_confidence_queries']) / self.metrics['total_predictions'] * 100, 1),
                'misclassification_count': len(self.metrics['misclassifications']),
                'error_count': len(self.metrics['errors'])
            },
            'user_feedback': {
                'total_feedback': feedback_count,
                'correct': sum(1 for f in self.metrics['user_feedback'] if f['feedback'] == 'correct'),
                'wrong': sum(1 for f in self.metrics['user_feedback'] if f['feedback'] == 'wrong'),
                'partial': sum(1 for f in self.metrics['user_feedback'] if f['feedback'] == 'partial')
            }
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate detailed monitoring report"""
        metrics = self.get_current_metrics()
        
        if metrics['status'] == 'No predictions yet':
            return "No predictions to report yet."
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║     ML PRODUCTION MONITORING REPORT                          ║
╚══════════════════════════════════════════════════════════════╝

📊 OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Period Start:        {metrics['period_start']}
Total Predictions:   {metrics['total_predictions']}
Status:              ✅ Active

⚡ PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average Latency:     {metrics['performance']['avg_latency_ms']} ms
P95 Latency:         {metrics['performance']['p95_latency_ms']} ms
P99 Latency:         {metrics['performance']['p99_latency_ms']} ms
Model Accuracy:      {metrics['performance']['accuracy']}{'%' if isinstance(metrics['performance']['accuracy'], (int, float)) else ''}

🎯 TOP PREDICTED INTENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, intent_data in enumerate(metrics['top_intents'], 1):
            report += f"{i}. {intent_data['intent']:<25} {intent_data['count']:>5} ({intent_data['percentage']}%)\n"
        
        report += f"""
📈 CONFIDENCE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for bucket, count in sorted(metrics['confidence_distribution'].items()):
            percentage = round(count / metrics['total_predictions'] * 100, 1)
            bar = '█' * int(percentage / 2)
            report += f"{bucket:<15} {count:>5} ({percentage:>5.1f}%) {bar}\n"
        
        report += f"""
⚠️ QUALITY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Low Confidence:      {metrics['quality_metrics']['low_confidence_count']} ({metrics['quality_metrics']['low_confidence_rate']}%)
Misclassifications:  {metrics['quality_metrics']['misclassification_count']}
System Errors:       {metrics['quality_metrics']['error_count']}

👤 USER FEEDBACK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Feedback:      {metrics['user_feedback']['total_feedback']}
✅ Correct:          {metrics['user_feedback']['correct']}
❌ Wrong:            {metrics['user_feedback']['wrong']}
⚠️ Partial:          {metrics['user_feedback']['partial']}
"""
        
        # Low confidence queries
        if self.metrics['low_confidence_queries']:
            report += "\n🔍 LOW CONFIDENCE QUERIES (Sample)\n"
            report += "━" * 60 + "\n"
            for query_data in self.metrics['low_confidence_queries'][:5]:
                report += f"• {query_data['query'][:50]}...\n"
                report += f"  Intent: {query_data['predicted_intent']} (conf: {query_data['confidence']:.2%})\n\n"
        
        # Misclassifications
        if self.metrics['misclassifications']:
            report += "\n❌ MISCLASSIFICATIONS (Sample)\n"
            report += "━" * 60 + "\n"
            for misc in self.metrics['misclassifications'][:5]:
                report += f"• {misc['query'][:50]}...\n"
                report += f"  Predicted: {misc['predicted']} | Actual: {misc['actual']}\n\n"
        
        report += "\n" + "═" * 60 + "\n"
        report += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📄 Report saved to: {output_file}")
        
        return report
    
    def get_training_candidates(self, limit: int = 100) -> List[Dict]:
        """
        Get queries that should be added to training data
        
        Returns queries that are:
        - Low confidence
        - Misclassified
        - Received negative feedback
        """
        candidates = []
        
        # Low confidence queries
        for query_data in self.metrics['low_confidence_queries']:
            candidates.append({
                'query': query_data['query'],
                'suggested_intent': query_data['predicted_intent'],
                'confidence': query_data['confidence'],
                'reason': 'low_confidence',
                'priority': 'medium'
            })
        
        # Misclassifications
        for misc in self.metrics['misclassifications']:
            candidates.append({
                'query': misc['query'],
                'suggested_intent': misc['actual'],
                'confidence': misc['confidence'],
                'reason': 'misclassified',
                'priority': 'high'
            })
        
        # Negative feedback
        for feedback in self.metrics['user_feedback']:
            if feedback['feedback'] == 'wrong':
                candidates.append({
                    'query': feedback['query'],
                    'suggested_intent': feedback['predicted_intent'],
                    'confidence': feedback['confidence'],
                    'reason': 'negative_feedback',
                    'priority': 'high'
                })
        
        # Sort by priority and limit
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        candidates.sort(key=lambda x: priority_order[x['priority']])
        
        return candidates[:limit]
    
    def export_for_retraining(self, output_file: str = "data/production_training_candidates.json"):
        """Export training candidates for model retraining"""
        candidates = self.get_training_candidates()
        
        training_data = {
            'export_date': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'candidates': candidates,
            'instructions': (
                "Review these queries and assign correct intents. "
                "Then add them to the training dataset and retrain the model."
            )
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Exported {len(candidates)} training candidates to: {output_file}")
        
        return output_path
    
    def _write_to_log(self, entry: Dict):
        """Write entry to log file"""
        try:
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log: {e}")
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket label"""
        if confidence >= 0.9:
            return 'Very High (>90%)'
        elif confidence >= 0.8:
            return 'High (80-90%)'
        elif confidence >= 0.7:
            return 'Medium (70-80%)'
        elif confidence >= 0.6:
            return 'Low (60-70%)'
        else:
            return 'Very Low (<60%)'
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# Singleton instance
_monitor_instance = None

def get_production_monitor(**kwargs) -> MLProductionMonitor:
    """Get or create singleton production monitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = MLProductionMonitor(**kwargs)
    return _monitor_instance


def main():
    """Demo usage"""
    monitor = get_production_monitor()
    
    # Simulate some predictions
    print("📊 Simulating ML predictions...\n")
    
    test_queries = [
        ("Best seafood restaurants in Beyoğlu", "restaurant_search", 0.95, 45),
        ("How to get to Blue Mosque", "transport_route", 0.88, 52),
        ("Weather tomorrow", "weather_query", 0.62, 48),
        ("Thanks for your help", "daily_gratitude", 0.91, 38),
        ("Show me museums", "attraction_search", 0.87, 51),
    ]
    
    for query, intent, conf, latency in test_queries:
        monitor.log_prediction(
            query=query,
            predicted_intent=intent,
            confidence=conf,
            latency_ms=latency
        )
        time.sleep(0.1)
    
    # Generate report
    print(monitor.generate_report())
    
    # Export training candidates
    print("\n📤 Exporting training candidates...")
    monitor.export_for_retraining()
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
