#!/usr/bin/env python3
"""
User Feedback Collection System for Intent Classifier Retraining
Collects user corrections, misclassifications, and new examples for continuous improvement

Core Functions Covered:
- Daily talks
- Places/attractions
- Neighborhood guides
- Transportation
- Events advising
- Route planner
- Weather system
- Local tips/hidden gems
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserFeedbackCollector:
    """
    Collects and manages user feedback for intent classifier improvement
    
    Features:
    - Misclassification reporting
    - User corrections
    - New example submissions
    - Confidence tracking
    - Feedback analytics
    - Automatic retraining data generation
    """
    
    # Map user functions to intents
    FUNCTION_TO_INTENT_MAPPING = {
        # Daily talks -> greeting, farewell, thanks, help, general_info
        "daily_talks": ["greeting", "farewell", "thanks", "help", "general_info"],
        
        # Places/attractions -> attraction, museum, hidden_gems
        "places_attractions": ["attraction", "museum", "hidden_gems"],
        
        # Neighborhood guides -> neighborhoods, local_tips
        "neighborhood_guides": ["neighborhoods", "local_tips"],
        
        # Transportation -> transportation, gps_navigation, route_planning
        "transportation": ["transportation", "gps_navigation", "route_planning"],
        
        # Events advising -> events, cultural_info
        "events_advising": ["events", "cultural_info"],
        
        # Route planner -> route_planning, gps_navigation
        "route_planner": ["route_planning", "gps_navigation"],
        
        # Weather system -> weather
        "weather_system": ["weather"],
        
        # Local tips/hidden gems -> hidden_gems, local_tips, recommendation
        "local_tips_hidden_gems": ["hidden_gems", "local_tips", "recommendation"],
    }
    
    def __init__(self, feedback_file: str = "data/user_feedback.jsonl"):
        """Initialize feedback collector"""
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add properties for backend compatibility
        self.feedback_log_path = str(self.feedback_file)
        self.retraining_data_path = "data/retraining_data.json"
        
        # Create file if it doesn't exist
        if not self.feedback_file.exists():
            self.feedback_file.touch()
        
        self.stats = {
            'total_feedback': 0,
            'misclassifications': 0,
            'corrections': 0,
            'new_examples': 0,
            'low_confidence': 0
        }
        
        logger.info(f"✅ Feedback collector initialized: {self.feedback_file}")
    
    def record_prediction(
        self,
        query: str,
        predicted_intent: str,
        confidence: float,
        actual_intent: Optional[str] = None,
        user_function: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Record a prediction and optional user correction
        
        Args:
            query: User's input query
            predicted_intent: Intent predicted by model
            confidence: Prediction confidence
            actual_intent: Correct intent (if user provided feedback)
            user_function: Which function was used (e.g., "transportation")
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())
        
        feedback_entry = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'predicted_intent': predicted_intent,
            'confidence': confidence,
            'actual_intent': actual_intent,
            'user_function': user_function,
            'user_id': user_id,
            'session_id': session_id,
            'metadata': metadata or {},
            'feedback_type': self._determine_feedback_type(
                predicted_intent, actual_intent, confidence
            )
        }
        
        # Write to JSONL file
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        
        # Update stats
        self.stats['total_feedback'] += 1
        if actual_intent and actual_intent != predicted_intent:
            self.stats['misclassifications'] += 1
        if confidence < 0.6:
            self.stats['low_confidence'] += 1
        
        logger.info(f"📝 Feedback recorded: {feedback_id}")
        return feedback_id
    
    def record_correction(
        self,
        query: str,
        wrong_intent: str,
        correct_intent: str,
        user_function: str,
        language: str = "auto",
        user_id: Optional[str] = None
    ) -> str:
        """
        Record a user correction for misclassification
        
        Args:
            query: The misclassified query
            wrong_intent: What the model predicted
            correct_intent: What it should have been
            user_function: Which function was being used
            language: Query language (tr/en/auto)
            user_id: User identifier
            
        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())
        
        correction_entry = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'wrong_intent': wrong_intent,
            'correct_intent': correct_intent,
            'user_function': user_function,
            'language': language if language != "auto" else self._detect_language(query),
            'user_id': user_id,
            'feedback_type': 'correction',
            'confidence': 0.0  # User correction, so original prediction was wrong
        }
        
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(correction_entry, ensure_ascii=False) + '\n')
        
        self.stats['corrections'] += 1
        logger.info(f"✏️ Correction recorded: {wrong_intent} -> {correct_intent}")
        return feedback_id
    
    def record_new_example(
        self,
        query: str,
        intent: str,
        user_function: str,
        language: str = "auto",
        confidence: float = 1.0,
        source: str = "user_submission"
    ) -> str:
        """
        Record a new training example submitted by user or admin
        
        Args:
            query: New example query
            intent: Correct intent
            user_function: Related function
            language: Query language
            confidence: How confident (1.0 for human-labeled)
            source: Where it came from
            
        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())
        
        new_example_entry = {
            'feedback_id': feedback_id,
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'intent': intent,
            'user_function': user_function,
            'language': language if language != "auto" else self._detect_language(query),
            'confidence': confidence,
            'source': source,
            'feedback_type': 'new_example'
        }
        
        with open(self.feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_example_entry, ensure_ascii=False) + '\n')
        
        self.stats['new_examples'] += 1
        logger.info(f"➕ New example added: {intent}")
        return feedback_id
    
    def get_feedback_summary(self, days: int = 7) -> Dict:
        """Get feedback summary for last N days"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        summary = {
            'total': 0,
            'misclassifications': [],
            'corrections': [],
            'new_examples': [],
            'low_confidence': [],
            'by_intent': defaultdict(int),
            'by_function': defaultdict(int),
            'by_language': defaultdict(int)
        }
        
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                entry_date = datetime.fromisoformat(entry['timestamp'])
                
                if entry_date >= cutoff_date:
                    summary['total'] += 1
                    
                    feedback_type = entry.get('feedback_type', 'unknown')
                    
                    if feedback_type == 'misclassification':
                        summary['misclassifications'].append(entry)
                    elif feedback_type == 'correction':
                        summary['corrections'].append(entry)
                    elif feedback_type == 'new_example':
                        summary['new_examples'].append(entry)
                    elif feedback_type == 'low_confidence':
                        summary['low_confidence'].append(entry)
                    
                    # Track by intent
                    intent = entry.get('correct_intent') or entry.get('intent') or entry.get('predicted_intent')
                    if intent:
                        summary['by_intent'][intent] += 1
                    
                    # Track by function
                    function = entry.get('user_function')
                    if function:
                        summary['by_function'][function] += 1
                    
                    # Track by language
                    language = entry.get('language', 'unknown')
                    summary['by_language'][language] += 1
        
        return summary
    
    def generate_retraining_data(
        self,
        min_corrections: int = 3,
        include_new_examples: bool = True,
        output_file: str = "data/retraining_data.json"
    ) -> Tuple[int, str]:
        """
        Generate retraining data from collected feedback
        
        Args:
            min_corrections: Minimum corrections needed to include an intent
            include_new_examples: Include user-submitted examples
            output_file: Where to save retraining data
            
        Returns:
            Tuple of (number of examples, output file path)
        """
        retraining_examples = []
        intent_corrections = defaultdict(list)
        
        # Read all feedback
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                
                feedback_type = entry.get('feedback_type', 'unknown')
                
                # Collect corrections
                if feedback_type == 'correction':
                    intent = entry['correct_intent']
                    intent_corrections[intent].append({
                        'text': entry['query'],
                        'intent': intent,
                        'language': entry.get('language', 'unknown'),
                        'source': 'user_correction',
                        'user_function': entry.get('user_function')
                    })
                
                # Collect new examples
                elif feedback_type == 'new_example' and include_new_examples:
                    retraining_examples.append({
                        'text': entry['query'],
                        'intent': entry['intent'],
                        'language': entry.get('language', 'unknown'),
                        'source': entry.get('source', 'user_submission'),
                        'user_function': entry.get('user_function')
                    })
        
        # Add corrections that meet minimum threshold
        for intent, examples in intent_corrections.items():
            if len(examples) >= min_corrections:
                retraining_examples.extend(examples)
                logger.info(f"✅ Added {len(examples)} corrections for '{intent}'")
            else:
                logger.info(f"⏭️ Skipped '{intent}' ({len(examples)} < {min_corrections})")
        
        # Save retraining data
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'training_data': retraining_examples,
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'total_examples': len(retraining_examples),
                    'min_corrections_threshold': min_corrections,
                    'includes_new_examples': include_new_examples
                }
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Retraining data saved: {output_path} ({len(retraining_examples)} examples)")
        
        return len(retraining_examples), str(output_path)
    
    def get_misclassification_report(self) -> Dict:
        """Generate misclassification report"""
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        intent_errors = defaultdict(list)
        
        with open(self.feedback_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                
                if entry.get('feedback_type') in ['misclassification', 'correction']:
                    wrong = entry.get('wrong_intent') or entry.get('predicted_intent')
                    correct = entry.get('correct_intent') or entry.get('actual_intent')
                    
                    if wrong and correct and wrong != correct:
                        confusion_matrix[correct][wrong] += 1
                        intent_errors[correct].append({
                            'query': entry['query'],
                            'predicted': wrong,
                            'confidence': entry.get('confidence', 0.0)
                        })
        
        return {
            'confusion_matrix': dict(confusion_matrix),
            'intent_errors': dict(intent_errors),
            'most_confused_pairs': self._get_most_confused_pairs(confusion_matrix)
        }
    
    def _determine_feedback_type(
        self,
        predicted: str,
        actual: Optional[str],
        confidence: float
    ) -> str:
        """Determine type of feedback"""
        if actual and actual != predicted:
            return 'misclassification'
        elif confidence < 0.6:
            return 'low_confidence'
        else:
            return 'prediction'
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        turkish_chars = set('çğıöşüÇĞİÖŞÜ')
        return 'tr' if any(char in turkish_chars for char in text) else 'en'
    
    def _get_most_confused_pairs(self, confusion_matrix: Dict, top_n: int = 10) -> List[Tuple]:
        """Get most commonly confused intent pairs"""
        pairs = []
        for correct, predictions in confusion_matrix.items():
            for predicted, count in predictions.items():
                pairs.append((correct, predicted, count))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]
    
    def print_statistics(self):
        """Print feedback statistics"""
        print("\n" + "=" * 80)
        print("📊 FEEDBACK COLLECTION STATISTICS")
        print("=" * 80)
        
        summary = self.get_feedback_summary(days=30)
        
        print(f"\n📈 Last 30 Days:")
        print(f"   Total Feedback: {summary['total']}")
        print(f"   Misclassifications: {len(summary['misclassifications'])}")
        print(f"   Corrections: {len(summary['corrections'])}")
        print(f"   New Examples: {len(summary['new_examples'])}")
        print(f"   Low Confidence: {len(summary['low_confidence'])}")
        
        print(f"\n🎯 By Intent:")
        for intent, count in sorted(summary['by_intent'].items(), key=lambda x: -x[1])[:10]:
            print(f"   {intent:20s}: {count:3d} feedback items")
        
        print(f"\n🔧 By Function:")
        for function, count in sorted(summary['by_function'].items(), key=lambda x: -x[1]):
            print(f"   {function:25s}: {count:3d} feedback items")
        
        print(f"\n🌍 By Language:")
        for language, count in summary['by_language'].items():
            print(f"   {language:10s}: {count:3d} feedback items")
        
        # Misclassification report
        report = self.get_misclassification_report()
        if report['most_confused_pairs']:
            print(f"\n❌ Most Confused Intent Pairs:")
            for correct, predicted, count in report['most_confused_pairs'][:5]:
                print(f"   {correct:20s} → {predicted:20s}: {count:3d} times")
        
        print("\n" + "=" * 80 + "\n")


# Singleton instance
_feedback_collector = None


def get_feedback_collector() -> UserFeedbackCollector:
    """Get singleton feedback collector"""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = UserFeedbackCollector()
    return _feedback_collector


def demo_feedback_collection():
    """Demo the feedback collection system"""
    print("=" * 80)
    print("USER FEEDBACK COLLECTION SYSTEM DEMO")
    print("=" * 80)
    print()
    
    collector = get_feedback_collector()
    
    # Simulate some feedback
    print("📝 Recording sample feedback...")
    
    # 1. Misclassification
    collector.record_prediction(
        query="Where is the nearest metro station?",
        predicted_intent="attraction",  # Wrong!
        confidence=0.45,
        actual_intent="transportation",  # Correct
        user_function="transportation"
    )
    
    # 2. User correction
    collector.record_correction(
        query="Best hotels near Sultanahmet",
        wrong_intent="attraction",
        correct_intent="accommodation",
        user_function="places_attractions",
        language="en"
    )
    
    # 3. New example submission
    collector.record_new_example(
        query="Çocuk dostu müzeler",
        intent="family_activities",
        user_function="places_attractions",
        language="tr"
    )
    
    # 4. Low confidence prediction
    collector.record_prediction(
        query="What's the weather like?",
        predicted_intent="weather",
        confidence=0.52,
        user_function="weather_system"
    )
    
    print("✅ Sample feedback recorded\n")
    
    # Show statistics
    collector.print_statistics()
    
    # Generate retraining data
    print("🔄 Generating retraining data...")
    count, path = collector.generate_retraining_data(min_corrections=1)
    print(f"✅ Generated {count} retraining examples in {path}\n")


if __name__ == "__main__":
    demo_feedback_collection()
